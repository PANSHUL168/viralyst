"""Engagement aggregation, segmentation, and virality scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from config import (
    REACH_BONUS_MAX,
    W_COMMENT,
    W_COMPLETION,
    W_SHARE,
    W_SKIP,
)
from schemas import EngagementMetrics, Persona, Reaction, SegmentMetrics

CALIBRATION_INPUTS = (0.00, 0.15, 0.30, 0.40, 0.55, 1.00)
CALIBRATION_OUTPUTS = (0.0, 25.0, 50.0, 70.0, 90.0, 100.0)


def segment_table(
    reactions: Sequence[Reaction],
    personas: Sequence[Persona],
) -> list[SegmentMetrics]:
    """Group exposed users by archetype and calculate engagement rates."""

    persona_by_id = _persona_index(personas)
    _validate_reactions(reactions, persona_by_id)

    grouped: dict[str, list[Reaction]] = defaultdict(list)
    for reaction in reactions:
        grouped[persona_by_id[reaction.persona_id].archetype].append(reaction)

    segments: list[SegmentMetrics] = []
    for segment in sorted(grouped):
        members = grouped[segment]
        count = len(members)
        segments.append(
            SegmentMetrics(
                segment=segment,
                n=count,
                avg_watch=sum(item.watch_percentage for item in members) / count,
                like_rate=sum(item.liked for item in members) / count,
                share_rate=sum(item.shared for item in members) / count,
                comment_rate=sum(item.commented for item in members) / count,
                skip_rate=sum(item.skipped for item in members) / count,
            )
        )
    return segments


def virality_score(metrics: Mapping[str, float]) -> float:
    """Return the documented calibrated score without implying prediction."""

    completion = _bounded_rate(metrics, "completion_rate")
    share = _bounded_rate(metrics, "share_rate")
    comment = _bounded_rate(metrics, "comment_rate")
    skip = _bounded_rate(metrics, "skip_rate")
    reach_pct = _bounded_rate(metrics, "reach_pct")

    raw = (
        W_COMPLETION * completion
        + W_SHARE * share
        + W_COMMENT * comment
        + W_SKIP * skip
    )
    raw = float(np.clip(raw, 0.0, 1.0))
    calibrated = float(
        np.interp(raw, CALIBRATION_INPUTS, CALIBRATION_OUTPUTS)
    )
    return float(
        np.clip(calibrated + REACH_BONUS_MAX * reach_pct, 0.0, 100.0)
    )


def aggregate(
    reactions: Sequence[Reaction],
    personas: Sequence[Persona],
    reached: int,
    population: int,
) -> EngagementMetrics:
    """Calculate overall and segment metrics across all exposed personas."""

    if not isinstance(reached, int) or not isinstance(population, int):
        raise TypeError("reached and population must be integers")
    if population < 0:
        raise ValueError("population must be greater than or equal to zero")
    if reached < 0 or reached > population:
        raise ValueError("reached must be between zero and population")

    persona_by_id = _persona_index(personas)
    _validate_reactions(reactions, persona_by_id)
    count = len(reactions)
    completion_rate = (
        sum(reaction.watch_percentage for reaction in reactions) / (100.0 * count)
        if count
        else 0.0
    )
    rates = {
        "completion_rate": completion_rate,
        "skip_rate": _boolean_rate(reactions, "skipped"),
        "like_rate": _boolean_rate(reactions, "liked"),
        "share_rate": _boolean_rate(reactions, "shared"),
        "comment_rate": _boolean_rate(reactions, "commented"),
        "follow_rate": _boolean_rate(reactions, "followed_creator"),
        "reach_pct": reached / population if population else 0.0,
    }
    purchase_intent = (
        sum(reaction.purchase_intent for reaction in reactions) / count
        if count
        else 0.0
    )
    score = virality_score(rates)
    return EngagementMetrics(
        n_reactions=count,
        completion_rate=rates["completion_rate"],
        skip_rate=rates["skip_rate"],
        like_rate=rates["like_rate"],
        share_rate=rates["share_rate"],
        comment_rate=rates["comment_rate"],
        follow_rate=rates["follow_rate"],
        avg_purchase_intent=purchase_intent,
        reach=reached,
        reach_pct=rates["reach_pct"],
        virality_score=score,
        segments=segment_table(reactions, personas),
    )


def _persona_index(personas: Sequence[Persona]) -> dict[int, Persona]:
    persona_by_id = {persona.id: persona for persona in personas}
    if len(persona_by_id) != len(personas):
        raise ValueError("persona IDs must be unique")
    return persona_by_id


def _validate_reactions(
    reactions: Sequence[Reaction],
    persona_by_id: Mapping[int, Persona],
) -> None:
    reaction_ids = [reaction.persona_id for reaction in reactions]
    if len(reaction_ids) != len(set(reaction_ids)):
        raise ValueError("reactions must contain each persona at most once")
    missing_ids = sorted(set(reaction_ids).difference(persona_by_id))
    if missing_ids:
        raise ValueError(f"reactions reference unknown personas: {missing_ids}")


def _boolean_rate(reactions: Sequence[Reaction], field: str) -> float:
    return (
        sum(bool(getattr(reaction, field)) for reaction in reactions)
        / len(reactions)
        if reactions
        else 0.0
    )


def _bounded_rate(metrics: Mapping[str, float], key: str) -> float:
    if key not in metrics:
        raise ValueError(f"missing score input: {key}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"score input '{key}' must be finite")
    return float(np.clip(value, 0.0, 1.0))


__all__ = [
    "CALIBRATION_INPUTS",
    "CALIBRATION_OUTPUTS",
    "aggregate",
    "segment_table",
    "virality_score",
]
