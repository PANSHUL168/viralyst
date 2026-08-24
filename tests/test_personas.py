"""Tests for deterministic synthetic persona generation."""

from __future__ import annotations

import pytest

from personas.generator import (
    ARCHETYPES,
    NORMALIZED_TRAITS,
    generate_personas,
    population_summary,
)
from schemas import Persona


def _serialized(personas) -> list[dict]:
    return [persona.model_dump(mode="json") for persona in personas]


@pytest.fixture(scope="module")
def population_1000() -> list[Persona]:
    return generate_personas(1_000, seed=42)


def test_same_seed_produces_identical_personas() -> None:
    first = generate_personas(100, seed=42)
    second = generate_personas(100, seed=42)

    assert _serialized(first) == _serialized(second)


def test_different_seed_changes_population() -> None:
    assert _serialized(generate_personas(20, seed=1)) != _serialized(
        generate_personas(20, seed=2)
    )


def test_persona_fields_stay_within_contract(population_1000) -> None:
    personas = population_1000

    assert [persona.id for persona in personas] == list(range(1_000))
    assert len({persona.name for persona in personas}) == 1_000
    for persona in personas:
        assert persona.name == f"P{persona.id}"
        assert 16 <= persona.age <= 55
        assert 2 <= len(persona.interests) <= 4
        assert len(persona.interests) == len(set(persona.interests))
        assert 0.5 <= persona.daily_scroll_hours <= 6.0
        for trait in NORMALIZED_TRAITS:
            assert 0.05 <= getattr(persona, trait) <= 0.95


def test_archetype_distribution_matches_weights(population_1000) -> None:
    personas = population_1000
    counts = {
        spec.name: sum(persona.archetype == spec.name for persona in personas)
        for spec in ARCHETYPES
    }

    for spec in ARCHETYPES:
        actual = counts[spec.name] / len(personas)
        assert actual == pytest.approx(spec.weight, abs=0.08)


def test_archetype_skews_create_behavioral_differences(population_1000) -> None:
    summary = population_summary(population_1000).set_index("archetype")

    assert (
        summary.loc["developer", "avg_skepticism"]
        > summary.loc["casual_viewer", "avg_skepticism"] + 0.20
    )
    assert (
        summary.loc["professional", "avg_attention_span"]
        > summary.loc["casual_viewer", "avg_attention_span"] + 0.30
    )
    assert (
        summary.loc["creator", "avg_comment_propensity"]
        > summary.loc["developer", "avg_comment_propensity"] + 0.12
    )


def test_population_summary_has_stable_order_and_totals() -> None:
    personas = generate_personas(100, seed=42)
    summary = population_summary(personas)

    expected_order = [
        spec.name
        for spec in ARCHETYPES
        if any(persona.archetype == spec.name for persona in personas)
    ]
    assert summary["archetype"].tolist() == expected_order
    assert int(summary["count"].sum()) == 100
    assert float(summary["population_pct"].sum()) == pytest.approx(100.0)
    assert summary["avg_attention_span"].between(0.05, 0.95).all()


def test_empty_population_has_documented_summary_columns() -> None:
    summary = population_summary([])

    assert summary.empty
    assert "archetype" in summary.columns
    assert "avg_daily_scroll_hours" in summary.columns


@pytest.mark.parametrize("n", [-1, -100])
def test_negative_population_size_is_rejected(n) -> None:
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        generate_personas(n)
