"""Prompting, validation, fallback, and batching for audience reactions."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from time import perf_counter

import numpy as np
from pydantic import ValidationError

from agents.llm import LLMError, ProgressCallback, chat_json, run_batch
from config import AUDIENCE_TEMPERATURE, MAX_WORKERS, TRANSCRIPT_CHAR_LIMIT
from schemas import Persona, Reaction, VideoFeatures

LOGGER = logging.getLogger("viralyst.agents.audience")

SYSTEM_PROMPT = """You are simulating a single social-media user's authentic
reaction to a short-form video. You are NOT an assistant and you are NOT
evaluating the video's quality objectively. You embody one specific person
with specific tastes, patience, and biases.

Rules:
- Be decisive. Real users skip fast and share rarely.
- Low attention_span means you stop watching early unless the hook grabs you.
- High skepticism means you distrust bold or unproven claims.
- Sharing is RARE. Only share if you would genuinely put your name on this.
- Your reason must reference something specific about THIS video.
- Return ONLY a JSON object. No prose and no Markdown fences.
"""

REACTION_SCHEMA_HINT = """{
  "watch_percentage": <integer 0-100>,
  "skipped": <boolean>,
  "liked": <boolean>,
  "commented": <boolean>,
  "shared": <boolean>,
  "followed_creator": <boolean>,
  "purchase_intent": <number 0-1>,
  "reason": "<one specific sentence, at most 30 words>"
}"""

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def build_prompt(features: VideoFeatures, persona: Persona) -> tuple[str, str]:
    """Build the behavior-focused system and user messages."""

    object_counts = sorted(
        features.objects.counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:5]
    objects = ", ".join(
        f"{name} ({count})" for name, count in object_counts
    ) or "no recognizable objects detected"
    hook = features.hook_transcript.strip() or "(no speech detected in opening)"
    transcript = features.transcript.strip()[:TRANSCRIPT_CHAR_LIMIT]
    if not transcript:
        transcript = "(silent or music-only video; no transcript available)"
    orientation = "vertical" if features.is_vertical else "horizontal"

    user_prompt = f"""VIDEO
Duration: {features.duration_sec:.1f}s ({orientation})
Pacing: {features.pacing_label} | Speech rate: {features.speech_rate_wps:.2f} words/sec
Visual density: {features.density_label} | Scene changes: {features.scene_changes}
Objects on screen: {objects}
Brightness: {features.avg_brightness:.3f} | Motion: {features.motion_score:.3f}

First 3 seconds (the hook):
"{hook}"

Full transcript ({features.word_count} words, truncated if needed):
"{transcript}"

YOU
Age: {persona.age} | Profession: {persona.profession}
Archetype: {persona.archetype}
Interests: {", ".join(persona.interests)}
Openness: {persona.openness:.3f} |
Conscientiousness: {persona.conscientiousness:.3f}
Extraversion: {persona.extraversion:.3f} | Agreeableness: {persona.agreeableness:.3f}
Neuroticism: {persona.neuroticism:.3f}
Attention span: {persona.attention_span:.3f} | Skepticism: {persona.skepticism:.3f}
Share propensity: {persona.share_propensity:.3f} | Comment propensity: {persona.comment_propensity:.3f}

Simulate how YOU would react to this video while scrolling.

Return exactly this JSON shape:
{REACTION_SCHEMA_HINT}
"""
    return SYSTEM_PROMPT, user_prompt


def react(features: VideoFeatures, persona: Persona, wave: int) -> Reaction:
    """Generate one validated reaction or a visible deterministic fallback."""

    if wave < 0:
        raise ValueError("wave must be greater than or equal to zero")
    system, user = build_prompt(features, persona)
    started_at = perf_counter()
    try:
        payload = chat_json(
            system,
            user,
            REACTION_SCHEMA_HINT,
            temperature=AUDIENCE_TEMPERATURE,
        )
        latency_ms = round((perf_counter() - started_at) * 1000)
        return Reaction.model_validate(
            {
                **payload,
                "persona_id": persona.id,
                "wave": wave,
                "latency_ms": latency_ms,
                "fallback": False,
            }
        )
    except (LLMError, ValidationError, TypeError, ValueError) as exc:
        latency_ms = round((perf_counter() - started_at) * 1000)
        LOGGER.warning(
            "Using fallback reaction for persona %s: %s",
            persona.id,
            exc,
        )
        fallback = heuristic_reaction(features, persona, wave)
        return fallback.model_copy(update={"latency_ms": latency_ms})


def heuristic_reaction(
    features: VideoFeatures,
    persona: Persona,
    wave: int,
) -> Reaction:
    """Create a deterministic non-LLM reaction for one failed agent call."""

    if wave < 0:
        raise ValueError("wave must be greater than or equal to zero")
    rng = np.random.default_rng(_fallback_seed(features, persona, wave))
    interest_match = _interest_match(features, persona)

    watch = 40.0 + 40.0 * interest_match + 20.0 * persona.attention_span
    if features.duration_sec > 45.0:
        watch -= 15.0
    if not features.hook_transcript.strip():
        watch -= 12.0
    if features.pacing_label == "fast":
        watch += 8.0 * (1.0 - persona.attention_span)
    elif features.pacing_label == "slow":
        watch -= 10.0 * (1.0 - persona.attention_span)
    watch -= 10.0 * persona.skepticism * (1.0 - interest_match)
    watch += 4.0 * features.motion_score
    watch_percentage = int(round(np.clip(watch, 0.0, 100.0)))

    marginal_skip_chance = float(
        np.clip(
            (42.0 - watch_percentage) / 40.0
            + 0.20 * (1.0 - persona.attention_span),
            0.0,
            0.80,
        )
    )
    skipped = watch_percentage < 22 or rng.random() < marginal_skip_chance
    if skipped:
        watch_percentage = min(watch_percentage, 30)

    like_chance = float(
        np.clip(0.20 + 0.50 * interest_match + 0.15 * persona.openness, 0.0, 0.90)
    )
    liked = (
        not skipped
        and watch_percentage >= 55
        and rng.random() < like_chance
    )
    comment_chance = float(
        np.clip(
            persona.comment_propensity * (0.25 + 0.20 * persona.skepticism),
            0.0,
            0.45,
        )
    )
    commented = (
        not skipped
        and watch_percentage >= 40
        and rng.random() < comment_chance
    )
    share_chance = float(
        np.clip(
            persona.share_propensity * 0.22 * (0.50 + 0.50 * interest_match),
            0.0,
            0.22,
        )
    )
    shared = (
        liked
        and watch_percentage >= 70
        and rng.random() < share_chance
    )
    followed_creator = (
        (liked or shared)
        and watch_percentage >= 78
        and rng.random() < 0.08 + 0.15 * interest_match
    )
    purchase_intent = round(
        float(
            np.clip(
                0.03 + 0.25 * interest_match + (0.08 if liked else 0.0),
                0.0,
                1.0,
            )
        ),
        3,
    )

    return Reaction(
        persona_id=persona.id,
        wave=wave,
        watch_percentage=watch_percentage,
        skipped=skipped,
        liked=liked,
        commented=commented,
        shared=shared,
        followed_creator=followed_creator,
        purchase_intent=purchase_intent,
        reason=_fallback_reason(
            features,
            persona,
            interest_match,
            skipped,
        ),
        fallback=True,
    )


def react_many(
    features: VideoFeatures,
    personas: Sequence[Persona],
    wave: int,
    progress_cb: ProgressCallback | None = None,
) -> list[Reaction]:
    """React for the active personas concurrently while preserving their order."""

    jobs: list[Callable[[], Reaction]] = [
        lambda persona=persona: react(features, persona, wave)
        for persona in personas
    ]
    return run_batch(jobs, max_workers=MAX_WORKERS, progress_cb=progress_cb)


def _fallback_seed(
    features: VideoFeatures,
    persona: Persona,
    wave: int,
) -> int:
    material = f"{features.video_hash}:{persona.id}:{wave}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _interest_match(features: VideoFeatures, persona: Persona) -> float:
    content = " ".join(
        [
            features.hook_transcript,
            features.transcript,
            *features.objects.counts.keys(),
        ]
    )
    content_tokens = {_stem(token) for token in _TOKEN_PATTERN.findall(content.lower())}
    matches = 0
    for interest in persona.interests:
        interest_tokens = {
            _stem(token) for token in _TOKEN_PATTERN.findall(interest.lower())
        }
        if content_tokens.intersection(interest_tokens):
            matches += 1
    return matches / len(persona.interests) if persona.interests else 0.0


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _fallback_reason(
    features: VideoFeatures,
    persona: Persona,
    interest_match: float,
    skipped: bool,
) -> str:
    source = features.hook_transcript.strip() or features.transcript.strip()
    subject = " ".join(source.split()[:8])
    if not subject:
        subject = features.objects.dominant_class or "the silent opening"
    if skipped:
        reason = (
            f"The opening about {subject} did not hold my attention long enough."
        )
    elif interest_match > 0:
        reason = (
            f"The discussion of {subject} matched my interests and kept me watching."
        )
    elif persona.skepticism > 0.65:
        reason = f"The claims about {subject} felt difficult for me to trust."
    else:
        reason = (
            f"The {features.pacing_label} presentation of {subject} was only "
            "moderately relevant to me."
        )
    return reason[:280]


__all__ = [
    "REACTION_SCHEMA_HINT",
    "SYSTEM_PROMPT",
    "build_prompt",
    "heuristic_reaction",
    "react",
    "react_many",
]
