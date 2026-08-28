"""Typed state and validated defaults shared by LangGraph nodes."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from config import (
    MAX_NEW_PER_WAVE,
    N_PERSONAS,
    RANDOM_SEED,
    SEED_SIZE,
    SEED_STRATEGY,
)


class SimulationConfig(TypedDict, total=False):
    """Per-run options that can safely differ from project defaults."""

    n_personas: int
    seed: int
    seed_size: int
    seed_strategy: str
    max_new_per_wave: int
    heuristic_only: bool
    use_video_cache: bool


class SimulationState(TypedDict, total=False):
    """Complete accumulating state for one Viralyst workflow execution."""

    video_path: str
    config: SimulationConfig

    video_features: dict[str, Any]
    personas: list[dict[str, Any]]
    social_graph: Any
    graph_stats: dict[str, Any]

    wave_index: int
    active_users: list[int]
    pending_users: list[int]
    exposed_ids: list[int]
    reactions: Annotated[list[dict[str, Any]], operator.add]
    waves: list[dict[str, Any]]
    continue_simulation: bool

    engagement_metrics: dict[str, Any]
    virality_score: float
    recommendations: dict[str, Any]
    recommendation_fallback: bool

    stop_reason: str
    timings: dict[str, float]
    errors: list[str]


def create_initial_state(
    video_path: str | Path,
    *,
    n_personas: int = N_PERSONAS,
    seed: int = RANDOM_SEED,
    seed_size: int = SEED_SIZE,
    seed_strategy: str = SEED_STRATEGY,
    max_new_per_wave: int = MAX_NEW_PER_WAVE,
    heuristic_only: bool = False,
    use_video_cache: bool = True,
) -> SimulationState:
    """Validate run options and return a complete initial workflow state."""

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: '{path}'.")
    if not isinstance(n_personas, int):
        raise TypeError("n_personas must be an integer")
    if n_personas <= 0:
        raise ValueError("n_personas must be greater than zero")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(seed_size, int):
        raise TypeError("seed_size must be an integer")
    if seed_size <= 0:
        raise ValueError("seed_size must be greater than zero")
    if seed_strategy not in {"random", "degree", "mixed"}:
        raise ValueError("seed_strategy must be 'random', 'degree', or 'mixed'")
    if not isinstance(max_new_per_wave, int):
        raise TypeError("max_new_per_wave must be an integer")
    if max_new_per_wave <= 0:
        raise ValueError("max_new_per_wave must be greater than zero")

    return SimulationState(
        video_path=str(path),
        config=SimulationConfig(
            n_personas=n_personas,
            seed=seed,
            seed_size=min(seed_size, n_personas),
            seed_strategy=seed_strategy,
            max_new_per_wave=max_new_per_wave,
            heuristic_only=bool(heuristic_only),
            use_video_cache=bool(use_video_cache),
        ),
        video_features={},
        personas=[],
        social_graph=None,
        graph_stats={},
        wave_index=0,
        active_users=[],
        pending_users=[],
        exposed_ids=[],
        reactions=[],
        waves=[],
        continue_simulation=False,
        engagement_metrics={},
        virality_score=0.0,
        recommendations={},
        recommendation_fallback=False,
        stop_reason="",
        timings={},
        errors=[],
    )


__all__ = ["SimulationConfig", "SimulationState", "create_initial_state"]
