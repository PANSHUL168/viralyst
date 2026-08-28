"""Convert workflow state into display-ready data frames."""

from __future__ import annotations

from typing import Any

import pandas as pd


def reaction_frame(state: dict[str, Any]) -> pd.DataFrame:
    """Join reactions to their persona labels for inspection and export."""

    personas = {
        int(persona["id"]): persona for persona in state.get("personas", [])
    }
    rows: list[dict[str, Any]] = []
    for reaction in state.get("reactions", []):
        persona = personas.get(int(reaction["persona_id"]), {})
        rows.append(
            {
                "persona_id": reaction["persona_id"],
                "name": persona.get("name", "Unknown"),
                "segment": persona.get("archetype", "Unknown"),
                "wave": reaction["wave"],
                "watch_percentage": reaction["watch_percentage"],
                "skipped": reaction["skipped"],
                "liked": reaction["liked"],
                "commented": reaction["commented"],
                "shared": reaction["shared"],
                "followed_creator": reaction["followed_creator"],
                "purchase_intent": reaction["purchase_intent"],
                "fallback": reaction.get("fallback", False),
                "reason": reaction["reason"],
            }
        )
    return pd.DataFrame(rows)


def segment_frame(state: dict[str, Any]) -> pd.DataFrame:
    """Return segment metrics with rates represented as percentages."""

    segments = state.get("engagement_metrics", {}).get("segments", [])
    frame = pd.DataFrame(segments)
    for column in ("like_rate", "share_rate", "comment_rate", "skip_rate"):
        if column in frame:
            frame[column] = frame[column] * 100.0
    return frame


def wave_frame(state: dict[str, Any]) -> pd.DataFrame:
    """Return one row per propagation wave."""

    return pd.DataFrame(state.get("waves", []))


def score_label(score: float) -> str:
    """Translate the directional score into a restrained qualitative label."""

    if score >= 75:
        return "Strong simulated signal"
    if score >= 50:
        return "Promising simulated signal"
    if score >= 25:
        return "Mixed simulated signal"
    return "Weak simulated signal"


def fallback_counts(state: dict[str, Any]) -> tuple[int, int]:
    """Return audience-reaction fallbacks and total reactions."""

    reactions = state.get("reactions", [])
    return (
        sum(bool(item.get("fallback", False)) for item in reactions),
        len(reactions),
    )


__all__ = [
    "fallback_counts",
    "reaction_frame",
    "score_label",
    "segment_frame",
    "wave_frame",
]
