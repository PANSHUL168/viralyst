"""Tests for persona prompting, validation, fallback, and batching."""

from __future__ import annotations

import pytest

from agents import audience
from agents.llm import LLMFormatError
from personas.generator import generate_personas
from schemas import ObjectStats, Reaction, VideoFeatures


@pytest.fixture
def features() -> VideoFeatures:
    return VideoFeatures(
        video_hash="abc123",
        filename="sample.mp4",
        duration_sec=30.0,
        fps=30.0,
        frame_count=900,
        sampled_frame_count=20,
        scene_changes=8,
        scene_change_rate=0.267,
        avg_brightness=0.55,
        brightness_variance=0.1,
        motion_score=0.4,
        aspect_ratio=0.5625,
        is_vertical=True,
        objects=ObjectStats(
            counts={"laptop": 12, "person": 20},
            unique_classes=2,
            avg_objects_per_frame=1.6,
            dominant_class="person",
        ),
        transcript="Learn Python programming with this quick laptop tutorial.",
        language="en",
        word_count=9,
        speech_duration_sec=6.0,
        speech_rate_wps=1.5,
        silence_ratio=0.8,
        has_speech=True,
        hook_transcript="Learn Python quickly",
        pacing_label="moderate",
        density_label="balanced",
    )


@pytest.fixture
def persona():
    return generate_personas(1, seed=42)[0]


def test_prompt_exposes_video_and_numeric_persona_traits(
    features: VideoFeatures,
    persona,
) -> None:
    system, user = audience.build_prompt(features, persona)

    assert "Sharing is RARE" in system
    assert features.hook_transcript in user
    assert persona.archetype in user
    assert f"Attention span: {persona.attention_span:.3f}" in user
    assert "laptop (12)" in user


def test_react_overrides_identity_and_repairs_contradictions(
    monkeypatch: pytest.MonkeyPatch,
    features: VideoFeatures,
    persona,
) -> None:
    payload = {
        "persona_id": 999,
        "wave": 99,
        "watch_percentage": 10,
        "skipped": True,
        "liked": True,
        "commented": True,
        "shared": True,
        "followed_creator": True,
        "purchase_intent": 0.2,
        "reason": "The Python tutorial did not keep my attention.",
    }
    monkeypatch.setattr(audience, "chat_json", lambda *args, **kwargs: payload)
    times = iter((10.0, 10.125))
    monkeypatch.setattr(audience, "perf_counter", lambda: next(times))

    reaction = audience.react(features, persona, wave=2)

    assert reaction.persona_id == persona.id
    assert reaction.wave == 2
    assert reaction.latency_ms == 125
    assert reaction.fallback is False
    assert not any(
        [
            reaction.liked,
            reaction.commented,
            reaction.shared,
            reaction.followed_creator,
        ]
    )


def test_react_uses_deterministic_fallback(
    monkeypatch: pytest.MonkeyPatch,
    features: VideoFeatures,
    persona,
) -> None:
    def fail(*args, **kwargs):
        raise LLMFormatError("bad JSON")

    monkeypatch.setattr(audience, "chat_json", fail)
    monkeypatch.setattr(audience, "perf_counter", lambda: 1.0)

    first = audience.react(features, persona, wave=0)
    second = audience.react(features, persona, wave=0)

    assert first == second
    assert first.fallback is True
    assert first.persona_id == persona.id


def test_heuristic_reaction_changes_with_wave(
    features: VideoFeatures,
    persona,
) -> None:
    wave_zero = audience.heuristic_reaction(features, persona, wave=0)
    wave_one = audience.heuristic_reaction(features, persona, wave=1)

    assert wave_zero.persona_id == wave_one.persona_id
    assert wave_zero.wave != wave_one.wave


def test_react_many_preserves_persona_order(
    monkeypatch: pytest.MonkeyPatch,
    features: VideoFeatures,
) -> None:
    personas = generate_personas(8, seed=8)

    def fake_react(features, persona, wave):
        return Reaction(
            persona_id=persona.id,
            wave=wave,
            watch_percentage=50,
            skipped=False,
            liked=False,
            commented=False,
            shared=False,
            followed_creator=False,
            purchase_intent=0.0,
            reason="The sample was moderately relevant.",
        )

    monkeypatch.setattr(audience, "react", fake_react)

    reactions = audience.react_many(features, personas, wave=3)

    assert [reaction.persona_id for reaction in reactions] == [
        persona.id for persona in personas
    ]
    assert {reaction.wave for reaction in reactions} == {3}


def test_negative_wave_is_rejected(features: VideoFeatures, persona) -> None:
    with pytest.raises(ValueError):
        audience.react(features, persona, wave=-1)
