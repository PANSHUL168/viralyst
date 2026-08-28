"""Evidence-grounded recommendation generation with deterministic fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agents.llm import chat_json
from config import (
    MAX_RECOMMENDATION_ITEMS,
    RECOMMENDATION_REASON_LIMIT,
    RECOMMENDER_TEMPERATURE,
    TRANSCRIPT_CHAR_LIMIT,
)
from schemas import (
    EngagementMetrics,
    Persona,
    Reaction,
    RecommendationItem,
    Recommendations,
    VideoFeatures,
    Wave,
)

SYSTEM_PROMPT = """You are a short-form video strategist reviewing output
from a synthetic-audience simulation. These numbers are directional signals,
not real measurements or a validated prediction.

Identify why engagement broke down for specific audience segments and propose
concrete edits grounded only in the supplied evidence. Reference the actual
opening wording, pacing, duration, visual objects, wave behavior, or audience
reasons. Do not invent timestamps beyond the documented 0-3 second hook. Never
give vague advice such as "make it engaging" or "improve the thumbnail".

Return ONLY one JSON object. No Markdown and no surrounding prose.
"""

RECOMMENDATION_SCHEMA_HINT = """{
  "summary": "<one evidence-based sentence>",
  "strengths": ["<specific strength>", "<specific strength>"],
  "items": [
    {
      "problem": "<measured or observed problem>",
      "likely_cause": "<cause grounded in supplied evidence>",
      "recommendation": "<specific edit the creator can make>",
      "priority": "high|medium|low",
      "target_segment": "<actual segment name or null>"
    }
  ]
}"""

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def select_reason_samples(
    reactions: Sequence[Reaction],
    personas: Sequence[Persona],
    limit: int = RECOMMENDATION_REASON_LIMIT,
) -> list[str]:
    """Select deterministic low-watch evidence while retaining segment spread."""

    if not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be greater than or equal to zero")
    if limit == 0 or not reactions:
        return []

    persona_by_id = {persona.id: persona for persona in personas}
    if len(persona_by_id) != len(personas):
        raise ValueError("persona IDs must be unique")
    missing = sorted(
        {reaction.persona_id for reaction in reactions}.difference(persona_by_id)
    )
    if missing:
        raise ValueError(f"reactions reference unknown personas: {missing}")

    ranked = sorted(
        reactions,
        key=lambda reaction: (
            not reaction.skipped,
            reaction.watch_percentage,
            reaction.wave,
            reaction.persona_id,
        ),
    )
    selected: list[Reaction] = []
    represented_segments: set[str] = set()
    for reaction in ranked:
        segment = persona_by_id[reaction.persona_id].archetype
        if segment in represented_segments:
            continue
        selected.append(reaction)
        represented_segments.add(segment)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_ids = {reaction.persona_id for reaction in selected}
        selected.extend(
            reaction
            for reaction in ranked
            if reaction.persona_id not in selected_ids
        )
    return [
        _format_reason(reaction, persona_by_id[reaction.persona_id])
        for reaction in selected[:limit]
    ]


def build_prompt(
    features: VideoFeatures,
    metrics: EngagementMetrics,
    waves: Sequence[Wave],
    sample_reasons: Sequence[str],
) -> tuple[str, str]:
    """Build a compact strategy prompt from validated simulation evidence."""

    objects = ", ".join(
        f"{name} ({count})"
        for name, count in sorted(
            features.objects.counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
    ) or "none detected"
    segment_lines = [
        (
            f"- {segment.segment}: n={segment.n}, "
            f"watch={segment.avg_watch:.1f}%, "
            f"skip={segment.skip_rate:.1%}, like={segment.like_rate:.1%}, "
            f"comment={segment.comment_rate:.1%}, "
            f"share={segment.share_rate:.1%}"
        )
        for segment in metrics.segments
    ] or ["- no segment data"]
    wave_lines = [
        (
            f"- wave {wave.index}: exposed={wave.new_exposures}, "
            f"cumulative={wave.cumulative_reached}, "
            f"watch={wave.avg_watch:.1f}%, shares={wave.shares}"
        )
        for wave in waves
    ] or ["- no wave data"]
    reasons = list(sample_reasons) or ["- no audience reasons available"]
    hook = features.hook_transcript.strip() or "(no spoken hook detected)"
    transcript = (
        features.transcript.strip()[:TRANSCRIPT_CHAR_LIMIT]
        or "(no transcript available)"
    )

    user_prompt = f"""VIDEO EVIDENCE
Duration: {features.duration_sec:.1f}s
Orientation: {"vertical" if features.is_vertical else "horizontal"}
Pacing: {features.pacing_label}
Speech rate: {features.speech_rate_wps:.2f} words/sec
Scene changes: {features.scene_changes}
Visual density: {features.density_label}
Detected objects: {objects}
Motion: {features.motion_score:.3f}
Opening 0-3 second hook: "{hook}"
Transcript (truncated): "{transcript}"

OVERALL SIMULATION
Virality score: {metrics.virality_score:.2f}/100
Completion: {metrics.completion_rate:.1%}
Skip: {metrics.skip_rate:.1%}
Like: {metrics.like_rate:.1%}
Comment: {metrics.comment_rate:.1%}
Share: {metrics.share_rate:.1%}
Follow: {metrics.follow_rate:.1%}
Reach: {metrics.reach} ({metrics.reach_pct:.1%})

SEGMENTS
{chr(10).join(segment_lines)}

PROPAGATION
{chr(10).join(wave_lines)}

REPRESENTATIVE AUDIENCE REASONS
{chr(10).join(reasons)}

Return at most {MAX_RECOMMENDATION_ITEMS} recommendations. Every item must name
a specific problem, evidence-grounded cause, and concrete edit. Use only actual
segment names listed above, or null.

Return exactly this JSON shape:
{RECOMMENDATION_SCHEMA_HINT}
"""
    return SYSTEM_PROMPT, user_prompt


def recommend(
    features: VideoFeatures,
    metrics: EngagementMetrics,
    waves: Sequence[Wave],
    sample_reasons: Sequence[str],
) -> Recommendations:
    """Request, validate, normalize, and prioritize one recommendation set."""

    system, user = build_prompt(features, metrics, waves, sample_reasons)
    payload = chat_json(
        system,
        user,
        RECOMMENDATION_SCHEMA_HINT,
        temperature=RECOMMENDER_TEMPERATURE,
    )
    payload = _normalize_payload(payload)
    recommendations = Recommendations.model_validate(payload)
    if not recommendations.summary.strip():
        raise ValueError("recommendation summary cannot be empty")
    if not recommendations.items:
        raise ValueError("at least one recommendation item is required")

    valid_segments = {segment.segment for segment in metrics.segments}
    normalized_items: list[RecommendationItem] = []
    seen: set[tuple[str, str]] = set()
    for item in recommendations.items:
        normalized = item.model_copy(
            update={
                "target_segment": (
                    item.target_segment
                    if item.target_segment in valid_segments
                    else None
                )
            }
        )
        signature = (
            normalized.problem.strip().casefold(),
            normalized.recommendation.strip().casefold(),
        )
        if signature in seen:
            continue
        seen.add(signature)
        normalized_items.append(normalized)
    normalized_items.sort(
        key=lambda item: _PRIORITY_ORDER[item.priority]
    )
    if not normalized_items:
        raise ValueError("recommendation items cannot all be duplicates")
    return recommendations.model_copy(
        update={
            "strengths": [
                strength.strip()
                for strength in recommendations.strengths
                if strength.strip()
            ][:3],
            "items": normalized_items[:MAX_RECOMMENDATION_ITEMS],
        }
    )


def heuristic_recommendations(
    features: VideoFeatures,
    metrics: EngagementMetrics,
    waves: Sequence[Wave],
) -> Recommendations:
    """Produce transparent metric-driven advice when recommendation LLM fails."""

    hook = features.hook_transcript.strip() or "the opening visual"
    hook_excerpt = " ".join(hook.split()[:12])
    sufficiently_sized = [segment for segment in metrics.segments if segment.n >= 3]
    weakest = min(
        sufficiently_sized,
        key=lambda segment: (segment.avg_watch, -segment.skip_rate),
        default=None,
    )
    target = weakest.segment if weakest is not None else None
    items: list[RecommendationItem] = []

    if metrics.skip_rate >= 0.5 or metrics.completion_rate < 0.4:
        items.append(
            RecommendationItem(
                problem=(
                    f"The simulation produced {metrics.skip_rate:.0%} skipping "
                    f"and {metrics.completion_rate:.0%} completion."
                ),
                likely_cause=(
                    f"The 0-3 second opening, '{hook_excerpt}', does not provide "
                    "enough immediate proof or payoff for skeptical viewers."
                ),
                recommendation=(
                    "Replace the opening setup with the strongest visible evidence "
                    "or outcome, then explain the claim after viewers see proof."
                ),
                priority="high",
                target_segment=target,
            )
        )
    if metrics.share_rate < 0.05:
        items.append(
            RecommendationItem(
                problem=f"Only {metrics.share_rate:.0%} of exposed personas shared.",
                likely_cause=(
                    "The video reports a claim but offers little concrete material "
                    "that viewers would confidently pass to someone else."
                ),
                recommendation=(
                    "Add one verifiable source, comparison, or visual takeaway that "
                    "a viewer can understand and share without extra explanation."
                ),
                priority="medium",
                target_segment=None,
            )
        )
    if features.duration_sec > 45 or features.pacing_label == "slow":
        items.append(
            RecommendationItem(
                problem=(
                    f"The video runs {features.duration_sec:.1f} seconds with "
                    f"{features.pacing_label} pacing."
                ),
                likely_cause="The payoff may arrive after low-attention users leave.",
                recommendation=(
                    "Remove repeated setup and place the main evidence before the "
                    "first explanatory section."
                ),
                priority="medium",
                target_segment=target,
            )
        )
    if not items:
        items.append(
            RecommendationItem(
                problem="The simulation is positive but still has avoidable drop-off.",
                likely_cause="Some viewers do not receive the main payoff immediately.",
                recommendation=(
                    "Preserve the current structure and test a version that shows "
                    "the final outcome inside the first three seconds."
                ),
                priority="low",
                target_segment=target,
            )
        )

    strengths: list[str] = []
    if metrics.completion_rate >= 0.6:
        strengths.append(
            f"Completion reached {metrics.completion_rate:.0%} among exposed personas."
        )
    if metrics.reach_pct >= 0.5:
        strengths.append(
            f"The cascade reached {metrics.reach_pct:.0%} of the synthetic population."
        )
    if features.hook_transcript.strip():
        strengths.append(f"The opening clearly introduces: '{hook_excerpt}'.")

    return Recommendations(
        summary=(
            f"The simulation scored {metrics.virality_score:.1f}/100; "
            f"retention and sharing are the clearest editing signals."
        ),
        strengths=strengths[:3],
        items=items[:MAX_RECOMMENDATION_ITEMS],
    )


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Repair a common one-item flattened response without another LLM call."""

    expected = {"problem", "likely_cause", "recommendation", "priority"}
    if expected.issubset(payload) and "items" not in payload:
        problem = str(payload.get("problem", "")).strip()
        return {
            "summary": (
                f"The simulation's clearest actionable issue is: {problem}"
            ),
            "strengths": [],
            "items": [payload],
        }
    return payload


def _format_reason(reaction: Reaction, persona: Persona) -> str:
    actions = []
    if reaction.skipped:
        actions.append("skipped")
    if reaction.liked:
        actions.append("liked")
    if reaction.commented:
        actions.append("commented")
    if reaction.shared:
        actions.append("shared")
    action_text = ", ".join(actions) or "watched only"
    return (
        f"- [{persona.archetype} | wave {reaction.wave} | "
        f"watched {reaction.watch_percentage}% | {action_text}] "
        f"{reaction.reason.strip()}"
    )


__all__ = [
    "RECOMMENDATION_SCHEMA_HINT",
    "SYSTEM_PROMPT",
    "build_prompt",
    "heuristic_recommendations",
    "recommend",
    "select_reason_samples",
]
