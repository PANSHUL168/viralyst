"""Stateful LangGraph orchestration for the complete Viralyst simulation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal, cast

import networkx as nx
import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agents.audience import heuristic_reaction, react_many
from agents.llm import LLMError
from agents.recommender import (
    heuristic_recommendations,
    recommend,
    select_reason_samples,
)
from personas.generator import generate_personas
from schemas import EngagementMetrics, Persona, Reaction, VideoFeatures, Wave
from simulation.metrics import aggregate
from simulation.propagation import next_wave, should_continue, summarize_wave
from simulation.social_graph import build_graph, graph_stats, select_seed_users
from utils import timer
from video.features import analyze_video
from workflow.state import SimulationState

RECURSION_LIMIT = 50
Route = Literal["continue", "stop"]


def analyze_video_node(state: SimulationState) -> dict[str, Any]:
    """Load cached or fresh multimodal features for the input video."""

    with timer("analyze_video") as measured:
        features = analyze_video(
            state["video_path"],
            use_cache=bool(state["config"].get("use_video_cache", True)),
        )
    return {
        "video_features": features.model_dump(mode="json"),
        "timings": _timing_update(state, "analyze_video", measured.elapsed_sec),
    }


def generate_personas_node(state: SimulationState) -> dict[str, Any]:
    """Create the configured deterministic synthetic population."""

    config = state["config"]
    with timer("generate_personas") as measured:
        personas = generate_personas(
            int(config["n_personas"]),
            int(config["seed"]),
        )
    return {
        "personas": [persona.model_dump(mode="json") for persona in personas],
        "timings": _timing_update(
            state,
            "generate_personas",
            measured.elapsed_sec,
        ),
    }


def build_social_graph_node(state: SimulationState) -> dict[str, Any]:
    """Construct and retain the frozen homophily graph and its statistics."""

    personas = _personas(state)
    with timer("build_social_graph") as measured:
        graph = build_graph(personas, seed=int(state["config"]["seed"]))
        stats = graph_stats(graph)
    return {
        "social_graph": graph,
        "graph_stats": stats,
        "timings": _timing_update(
            state,
            "build_social_graph",
            measured.elapsed_sec,
        ),
    }


def seed_audience_node(state: SimulationState) -> dict[str, Any]:
    """Choose the initial audience and initialize the simulation loop."""

    graph = _social_graph(state)
    personas = _personas(state)
    config = state["config"]
    with timer("seed_audience") as measured:
        seed_ids = select_seed_users(
            graph,
            personas,
            k=min(int(config["seed_size"]), len(personas)),
            seed=int(config["seed"]),
            strategy=str(config["seed_strategy"]),
        )
    return {
        "wave_index": 0,
        "active_users": seed_ids,
        "pending_users": [],
        "exposed_ids": list(seed_ids),
        "continue_simulation": bool(seed_ids),
        "stop_reason": "",
        "timings": _timing_update(
            state,
            "seed_audience",
            measured.elapsed_sec,
        ),
    }


def simulate_agents_node(state: SimulationState) -> dict[str, Any]:
    """Generate one reaction for every persona active in the current wave."""

    features = VideoFeatures.model_validate(state["video_features"])
    personas_by_id = {persona.id: persona for persona in _personas(state)}
    active_ids = list(state["active_users"])
    if len(active_ids) != len(set(active_ids)):
        raise ValueError("active_users must be unique")
    missing = sorted(set(active_ids).difference(personas_by_id))
    if missing:
        raise ValueError(f"active users have no matching personas: {missing}")
    active_personas = [personas_by_id[persona_id] for persona_id in active_ids]
    wave_index = int(state["wave_index"])
    timing_key = f"simulate_agents_wave_{wave_index}"

    with timer(timing_key) as measured:
        if bool(state["config"].get("heuristic_only", False)):
            reactions = [
                heuristic_reaction(features, persona, wave_index)
                for persona in active_personas
            ]
        else:
            reactions = react_many(features, active_personas, wave_index)
    return {
        "reactions": [reaction.model_dump(mode="json") for reaction in reactions],
        "timings": _timing_update(state, timing_key, measured.elapsed_sec),
    }


def aggregate_wave_node(state: SimulationState) -> dict[str, Any]:
    """Summarize only the reactions produced for the current wave."""

    wave_index = int(state["wave_index"])
    reactions = _wave_reactions(state, wave_index)
    timing_key = f"aggregate_wave_{wave_index}"
    with timer(timing_key) as measured:
        wave = summarize_wave(
            wave_index,
            state["active_users"],
            reactions,
            cumulative_reached=len(state["exposed_ids"]),
        )
    return {
        "waves": [*state.get("waves", []), wave.model_dump(mode="json")],
        "timings": _timing_update(state, timing_key, measured.elapsed_sec),
    }


def propagate_node(state: SimulationState) -> dict[str, Any]:
    """Calculate candidate users for the next wave without committing reach."""

    wave_index = int(state["wave_index"])
    reactions = _wave_reactions(state, wave_index)
    seed_sequence = np.random.SeedSequence(
        [int(state["config"]["seed"]), wave_index]
    )
    rng = np.random.default_rng(seed_sequence)
    timing_key = f"propagate_wave_{wave_index}"
    with timer(timing_key) as measured:
        pending_users = next_wave(
            _social_graph(state),
            reactions,
            state["exposed_ids"],
            rng,
            max_new=int(state["config"]["max_new_per_wave"]),
        )
    return {
        "pending_users": pending_users,
        "timings": _timing_update(state, timing_key, measured.elapsed_sec),
    }


def check_threshold_node(state: SimulationState) -> dict[str, Any]:
    """Decide whether to admit pending users into another simulation wave."""

    waves = _waves(state)
    if not waves:
        raise ValueError("cannot check thresholds before a wave is aggregated")
    current = waves[-1]
    timing_key = f"check_threshold_wave_{current.index}"
    with timer(timing_key) as measured:
        continued, reason = should_continue(
            current,
            current.index,
            reached=len(state["exposed_ids"]),
            population=len(state["personas"]),
        )
        pending = list(state.get("pending_users", []))
        if continued and not pending:
            continued = False
            reason = "stopped: no new users were exposed"

        updated_wave = current.model_copy(update={"continued": continued})
        updates: dict[str, Any] = {
            "waves": [
                *(wave.model_dump(mode="json") for wave in waves[:-1]),
                updated_wave.model_dump(mode="json"),
            ],
            "continue_simulation": continued,
            "stop_reason": reason,
        }
        if continued:
            updates.update(
                {
                    "active_users": pending,
                    "pending_users": [],
                    "exposed_ids": [*state["exposed_ids"], *pending],
                    "wave_index": current.index + 1,
                }
            )
        else:
            updates.update({"active_users": [], "pending_users": []})
    updates["timings"] = _timing_update(
        state,
        timing_key,
        measured.elapsed_sec,
    )
    return updates


def route_after_threshold(state: SimulationState) -> Route:
    """Route without mutating state; threshold calculations happen once."""

    return "continue" if state.get("continue_simulation", False) else "stop"


def finalize_metrics_node(state: SimulationState) -> dict[str, Any]:
    """Aggregate all cumulative reactions into the final metrics contract."""

    reactions = _reactions(state)
    personas = _personas(state)
    reaction_ids = {reaction.persona_id for reaction in reactions}
    exposed_ids = set(state["exposed_ids"])
    if reaction_ids != exposed_ids:
        raise ValueError(
            "final reactions must match all exposed personas exactly"
        )

    with timer("finalize_metrics") as measured:
        metrics = aggregate(
            reactions,
            personas,
            reached=len(exposed_ids),
            population=len(personas),
        )
    return {
        "engagement_metrics": metrics.model_dump(mode="json"),
        "virality_score": metrics.virality_score,
        "timings": _timing_update(
            state,
            "finalize_metrics",
            measured.elapsed_sec,
        ),
    }


def generate_recommendations_node(state: SimulationState) -> dict[str, Any]:
    """Generate evidence-grounded advice with a visible deterministic fallback."""

    with timer("generate_recommendations") as measured:
        features = VideoFeatures.model_validate(state["video_features"])
        metrics = EngagementMetrics.model_validate(state["engagement_metrics"])
        waves = _waves(state)
        reactions = _reactions(state)
        personas = _personas(state)
        reasons = select_reason_samples(reactions, personas)
        fallback = bool(state["config"].get("heuristic_only", False))
        errors = list(state.get("errors", []))
        if fallback:
            result = heuristic_recommendations(features, metrics, waves)
        else:
            try:
                result = recommend(features, metrics, waves, reasons)
            except (LLMError, ValidationError, TypeError, ValueError) as exc:
                fallback = True
                errors.append(f"Recommendation fallback used: {exc}")
                result = heuristic_recommendations(features, metrics, waves)
    return {
        "recommendations": result.model_dump(mode="json"),
        "recommendation_fallback": fallback,
        "errors": errors,
        "timings": _timing_update(
            state,
            "generate_recommendations",
            measured.elapsed_sec,
        ),
    }


def build_workflow() -> CompiledStateGraph:
    """Compile the Viralyst node graph and its conditional wave loop."""

    builder = StateGraph(SimulationState)
    builder.add_node("analyze_video", analyze_video_node)
    builder.add_node("generate_personas", generate_personas_node)
    builder.add_node("build_social_graph", build_social_graph_node)
    builder.add_node("seed_audience", seed_audience_node)
    builder.add_node("simulate_agents", simulate_agents_node)
    builder.add_node("aggregate_wave", aggregate_wave_node)
    builder.add_node("propagate", propagate_node)
    builder.add_node("check_threshold", check_threshold_node)
    builder.add_node("finalize_metrics", finalize_metrics_node)
    builder.add_node("generate_recommendations", generate_recommendations_node)

    builder.add_edge(START, "analyze_video")
    builder.add_edge("analyze_video", "generate_personas")
    builder.add_edge("generate_personas", "build_social_graph")
    builder.add_edge("build_social_graph", "seed_audience")
    builder.add_edge("seed_audience", "simulate_agents")
    builder.add_edge("simulate_agents", "aggregate_wave")
    builder.add_edge("aggregate_wave", "propagate")
    builder.add_edge("propagate", "check_threshold")
    builder.add_conditional_edges(
        "check_threshold",
        route_after_threshold,
        {
            "continue": "simulate_agents",
            "stop": "finalize_metrics",
        },
    )
    builder.add_edge("finalize_metrics", "generate_recommendations")
    builder.add_edge("generate_recommendations", END)
    return builder.compile()


def run_workflow(state: SimulationState) -> SimulationState:
    """Invoke the compiled graph with an explicit loop safety limit."""

    result = get_workflow().invoke(
        state,
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return cast(SimulationState, result)


def stream_workflow(
    state: SimulationState,
) -> Iterator[dict[str, dict[str, Any]]]:
    """Yield per-node state updates for a CLI or Streamlit progress view."""

    yield from get_workflow().stream(
        state,
        config={"recursion_limit": RECURSION_LIMIT},
        stream_mode="updates",
    )


def merge_update(
    state: SimulationState,
    update: Mapping[str, Any],
) -> SimulationState:
    """Apply a streamed node update using the state's reducer semantics."""

    merged = dict(state)
    for key, value in update.items():
        if key == "reactions":
            merged[key] = [*merged.get(key, []), *value]
        else:
            merged[key] = value
    return cast(SimulationState, merged)


_WORKFLOW: CompiledStateGraph | None = None


def get_workflow() -> CompiledStateGraph:
    """Return one compiled graph; compilation is deterministic and reusable."""

    global _WORKFLOW
    if _WORKFLOW is None:
        _WORKFLOW = build_workflow()
    return _WORKFLOW


def _personas(state: SimulationState) -> list[Persona]:
    return [Persona.model_validate(item) for item in state.get("personas", [])]


def _reactions(state: SimulationState) -> list[Reaction]:
    return [Reaction.model_validate(item) for item in state.get("reactions", [])]


def _wave_reactions(
    state: SimulationState,
    wave_index: int,
) -> list[Reaction]:
    return [
        reaction
        for reaction in _reactions(state)
        if reaction.wave == wave_index
    ]


def _waves(state: SimulationState) -> list[Wave]:
    return [Wave.model_validate(item) for item in state.get("waves", [])]


def _social_graph(state: SimulationState) -> nx.Graph:
    graph = state.get("social_graph")
    if not isinstance(graph, nx.Graph):
        raise ValueError("social_graph has not been constructed")
    return graph


def _timing_update(
    state: SimulationState,
    key: str,
    elapsed_sec: float,
) -> dict[str, float]:
    return {
        **state.get("timings", {}),
        key: round(float(elapsed_sec), 6),
    }


__all__ = [
    "RECURSION_LIMIT",
    "aggregate_wave_node",
    "analyze_video_node",
    "build_social_graph_node",
    "build_workflow",
    "check_threshold_node",
    "finalize_metrics_node",
    "generate_personas_node",
    "generate_recommendations_node",
    "get_workflow",
    "merge_update",
    "propagate_node",
    "route_after_threshold",
    "run_workflow",
    "seed_audience_node",
    "simulate_agents_node",
    "stream_workflow",
]
