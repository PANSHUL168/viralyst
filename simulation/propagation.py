"""Probabilistic, deterministic-by-seed multi-wave content propagation."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal

import networkx as nx
import numpy as np

from config import (
    CONTINUE_SHARE_RATE,
    EXPOSURE_PROBS,
    MAX_NEW_PER_WAVE,
    MAX_WAVES,
    MIN_NEW_EXPOSURES,
    SATURATION_PCT,
)
from schemas import Reaction, Wave

Action = Literal["share", "comment", "like", "watch", "skip"]


def dominant_action(reaction: Reaction) -> Action:
    """Return the strongest action that controls propagation probability."""

    if reaction.skipped:
        return "skip"
    if reaction.shared:
        return "share"
    if reaction.commented:
        return "comment"
    if reaction.liked:
        return "like"
    return "watch"


def next_wave(
    graph: nx.Graph,
    reactions: Sequence[Reaction],
    already_exposed: Collection[int],
    rng: np.random.Generator,
    max_new: int = MAX_NEW_PER_WAVE,
) -> list[int]:
    """Select newly exposed neighbours using engagement and edge strength."""

    if not isinstance(max_new, int):
        raise TypeError("max_new must be an integer")
    if max_new < 0:
        raise ValueError("max_new must be greater than or equal to zero")
    if max_new == 0 or not reactions:
        return []

    reaction_ids = [reaction.persona_id for reaction in reactions]
    if len(reaction_ids) != len(set(reaction_ids)):
        raise ValueError("reactions must contain each persona at most once")
    missing_ids = sorted(set(reaction_ids).difference(graph.nodes))
    if missing_ids:
        raise ValueError(f"reaction personas are missing from graph: {missing_ids}")

    excluded = set(already_exposed)
    candidate_probabilities: dict[int, float] = {}
    for reaction in sorted(reactions, key=lambda item: item.persona_id):
        base_probability = EXPOSURE_PROBS[dominant_action(reaction)]
        if base_probability <= 0.0:
            continue
        for neighbour_id in sorted(graph.neighbors(reaction.persona_id)):
            if neighbour_id in excluded:
                continue
            edge_weight = float(
                np.clip(
                    graph[reaction.persona_id][neighbour_id].get("weight", 0.0),
                    0.0,
                    1.0,
                )
            )
            probability = float(
                np.clip(
                    base_probability * (0.7 + 0.3 * edge_weight),
                    0.0,
                    1.0,
                )
            )
            previous = candidate_probabilities.get(neighbour_id, 0.0)
            candidate_probabilities[neighbour_id] = 1.0 - (
                (1.0 - previous) * (1.0 - probability)
            )

    selected = [
        persona_id
        for persona_id, probability in sorted(candidate_probabilities.items())
        if float(rng.random()) < probability
    ]
    if len(selected) <= max_new:
        return selected

    return sorted(
        selected,
        key=lambda persona_id: (
            -candidate_probabilities[persona_id],
            persona_id,
        ),
    )[:max_new]


def summarize_wave(
    index: int,
    exposed_ids: Sequence[int],
    reactions: Sequence[Reaction],
    cumulative_reached: int,
    continued: bool = False,
) -> Wave:
    """Convert one wave's validated reactions into display-ready statistics."""

    if index < 0:
        raise ValueError("index must be greater than or equal to zero")
    if cumulative_reached < 0:
        raise ValueError("cumulative_reached must be greater than or equal to zero")

    exposed = list(exposed_ids)
    if len(exposed) != len(set(exposed)):
        raise ValueError("exposed_ids must be unique within a wave")
    if cumulative_reached < len(exposed):
        raise ValueError("cumulative_reached cannot be smaller than this wave")

    reaction_ids = [reaction.persona_id for reaction in reactions]
    if len(reaction_ids) != len(set(reaction_ids)):
        raise ValueError("reactions must contain each persona at most once")
    if set(reaction_ids) != set(exposed):
        raise ValueError("reactions must match the personas exposed in this wave")
    if any(reaction.wave != index for reaction in reactions):
        raise ValueError("every reaction must match the wave index")

    count = len(reactions)
    avg_watch = (
        sum(reaction.watch_percentage for reaction in reactions) / count
        if count
        else 0.0
    )
    return Wave(
        index=index,
        exposed_ids=exposed,
        new_exposures=len(exposed),
        cumulative_reached=cumulative_reached,
        shares=sum(reaction.shared for reaction in reactions),
        comments=sum(reaction.commented for reaction in reactions),
        likes=sum(reaction.liked for reaction in reactions),
        skips=sum(reaction.skipped for reaction in reactions),
        avg_watch=float(avg_watch),
        continued=continued,
    )


def should_continue(
    wave: Wave,
    wave_index: int,
    reached: int,
    population: int,
) -> tuple[bool, str]:
    """Apply every cascade stopping rule and return an explainable decision."""

    if wave_index < 0:
        raise ValueError("wave_index must be greater than or equal to zero")
    if wave.index != wave_index:
        raise ValueError("wave_index must match wave.index")
    if population < 0:
        raise ValueError("population must be greater than or equal to zero")
    if reached < 0 or reached > population:
        raise ValueError("reached must be between zero and population")

    if wave_index + 1 >= MAX_WAVES:
        return False, f"stopped: maximum of {MAX_WAVES} waves reached"
    if wave.new_exposures < MIN_NEW_EXPOSURES:
        return (
            False,
            "stopped: only "
            f"{wave.new_exposures} new exposures; minimum is "
            f"{MIN_NEW_EXPOSURES}",
        )

    share_rate = wave.shares / wave.new_exposures
    if share_rate < CONTINUE_SHARE_RATE:
        return (
            False,
            f"stopped: share rate {share_rate:.3f} is below "
            f"{CONTINUE_SHARE_RATE:.3f}",
        )
    if population == 0:
        return False, "stopped: population is empty"

    reach_pct = reached / population
    if reach_pct >= SATURATION_PCT:
        return (
            False,
            f"stopped: reach {reach_pct:.1%} met the "
            f"{SATURATION_PCT:.0%} saturation threshold",
        )
    return True, "continue: propagation thresholds satisfied"


__all__ = [
    "Action",
    "dominant_action",
    "next_wave",
    "should_continue",
    "summarize_wave",
]
