"""Deterministic generation and summarization of synthetic audience personas."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import N_PERSONAS, RANDOM_SEED
from schemas import Persona

NORMALIZED_TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "attention_span",
    "skepticism",
    "share_propensity",
    "comment_propensity",
)

SUMMARY_COLUMNS = (
    "archetype",
    "count",
    "population_pct",
    "avg_age",
    *(f"avg_{trait}" for trait in NORMALIZED_TRAITS),
    "avg_daily_scroll_hours",
)


@dataclass(frozen=True, slots=True)
class ArchetypeSpec:
    """Controlled distribution parameters for one audience segment."""

    name: str
    weight: float
    age_range: tuple[int, int]
    professions: tuple[str, ...]
    interests: tuple[str, ...]
    trait_skews: Mapping[str, float]
    scroll_hours_range: tuple[float, float]


ARCHETYPES: tuple[ArchetypeSpec, ...] = (
    ArchetypeSpec(
        name="tech_enthusiast",
        weight=0.18,
        age_range=(18, 35),
        professions=(
            "software engineer",
            "product manager",
            "startup founder",
            "technology analyst",
        ),
        interests=(
            "AI",
            "technology",
            "gadgets",
            "startups",
            "programming",
            "productivity",
        ),
        trait_skews={
            "openness": 0.20,
            "conscientiousness": 0.05,
            "extraversion": 0.05,
            "attention_span": 0.05,
            "skepticism": 0.05,
            "share_propensity": 0.15,
            "comment_propensity": 0.05,
        },
        scroll_hours_range=(1.5, 4.5),
    ),
    ArchetypeSpec(
        name="student",
        weight=0.20,
        age_range=(16, 24),
        professions=(
            "high school student",
            "university student",
            "graduate student",
        ),
        interests=(
            "study",
            "gaming",
            "music",
            "memes",
            "technology",
            "sports",
        ),
        trait_skews={
            "openness": 0.10,
            "conscientiousness": -0.10,
            "extraversion": 0.10,
            "neuroticism": 0.05,
            "attention_span": -0.20,
            "skepticism": -0.05,
            "share_propensity": 0.15,
            "comment_propensity": 0.10,
        },
        scroll_hours_range=(2.5, 6.0),
    ),
    ArchetypeSpec(
        name="developer",
        weight=0.14,
        age_range=(22, 40),
        professions=(
            "backend developer",
            "frontend developer",
            "mobile developer",
            "data engineer",
        ),
        interests=(
            "programming",
            "AI",
            "open source",
            "technology",
            "gaming",
            "productivity",
        ),
        trait_skews={
            "openness": 0.15,
            "conscientiousness": 0.10,
            "extraversion": -0.10,
            "agreeableness": -0.05,
            "attention_span": 0.10,
            "skepticism": 0.25,
            "share_propensity": -0.15,
            "comment_propensity": 0.05,
        },
        scroll_hours_range=(1.0, 4.0),
    ),
    ArchetypeSpec(
        name="marketer",
        weight=0.12,
        age_range=(24, 42),
        professions=(
            "digital marketer",
            "growth marketer",
            "brand strategist",
            "marketing analyst",
        ),
        interests=(
            "marketing",
            "business",
            "design",
            "startups",
            "content",
            "technology",
        ),
        trait_skews={
            "openness": 0.10,
            "conscientiousness": 0.10,
            "extraversion": 0.15,
            "agreeableness": 0.05,
            "attention_span": 0.05,
            "skepticism": 0.05,
            "share_propensity": 0.05,
            "comment_propensity": 0.20,
        },
        scroll_hours_range=(1.5, 4.5),
    ),
    ArchetypeSpec(
        name="creator",
        weight=0.10,
        age_range=(18, 35),
        professions=(
            "video creator",
            "photographer",
            "streamer",
            "visual designer",
        ),
        interests=(
            "content",
            "video",
            "photography",
            "design",
            "music",
            "technology",
        ),
        trait_skews={
            "openness": 0.25,
            "extraversion": 0.15,
            "neuroticism": 0.05,
            "share_propensity": 0.15,
            "comment_propensity": 0.25,
        },
        scroll_hours_range=(2.0, 5.5),
    ),
    ArchetypeSpec(
        name="professional",
        weight=0.14,
        age_range=(28, 55),
        professions=(
            "consultant",
            "accountant",
            "project manager",
            "financial analyst",
        ),
        interests=(
            "business",
            "finance",
            "productivity",
            "technology",
            "travel",
            "health",
        ),
        trait_skews={
            "conscientiousness": 0.20,
            "neuroticism": -0.05,
            "attention_span": 0.20,
            "skepticism": 0.10,
            "share_propensity": -0.15,
            "comment_propensity": -0.05,
        },
        scroll_hours_range=(0.5, 3.0),
    ),
    ArchetypeSpec(
        name="casual_viewer",
        weight=0.12,
        age_range=(16, 50),
        professions=(
            "retail associate",
            "hospitality worker",
            "office assistant",
            "freelancer",
        ),
        interests=(
            "entertainment",
            "food",
            "travel",
            "sports",
            "music",
            "memes",
        ),
        trait_skews={
            "conscientiousness": -0.10,
            "extraversion": 0.05,
            "agreeableness": 0.10,
            "attention_span": -0.30,
            "skepticism": -0.10,
            "comment_propensity": -0.10,
        },
        scroll_hours_range=(2.0, 6.0),
    ),
)

ARCHETYPE_BY_NAME = {spec.name: spec for spec in ARCHETYPES}

if not math.isclose(sum(spec.weight for spec in ARCHETYPES), 1.0):
    raise RuntimeError("archetype weights must sum to 1.0")


def generate_personas(
    n: int = N_PERSONAS,
    seed: int = RANDOM_SEED,
) -> list[Persona]:
    """Generate a reproducible heterogeneous population using one seeded RNG."""

    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be greater than or equal to zero")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = np.random.default_rng(seed)
    archetype_indices = rng.choice(
        len(ARCHETYPES),
        size=n,
        p=[spec.weight for spec in ARCHETYPES],
    )
    personas: list[Persona] = []

    for persona_id, archetype_index in enumerate(archetype_indices.tolist()):
        spec = ARCHETYPES[archetype_index]
        age = int(rng.integers(spec.age_range[0], spec.age_range[1] + 1))
        profession = spec.professions[int(rng.integers(len(spec.professions)))]
        interest_count = int(rng.integers(2, min(4, len(spec.interests)) + 1))
        interest_indices = rng.choice(
            len(spec.interests),
            size=interest_count,
            replace=False,
        )
        interests = sorted(spec.interests[index] for index in interest_indices)
        traits = {
            trait: _normalized_trait(rng, spec.trait_skews.get(trait, 0.0))
            for trait in NORMALIZED_TRAITS
        }
        daily_scroll_hours = round(
            float(rng.uniform(*spec.scroll_hours_range)),
            3,
        )

        personas.append(
            Persona(
                id=persona_id,
                name=f"P{persona_id}",
                age=age,
                profession=profession,
                archetype=spec.name,
                interests=interests,
                daily_scroll_hours=daily_scroll_hours,
                **traits,
            )
        )

    return personas


def population_summary(personas: Sequence[Persona]) -> pd.DataFrame:
    """Summarize counts and mean traits for each represented archetype."""

    if not personas:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict[str, int | float | str]] = []
    population_size = len(personas)
    for spec in ARCHETYPES:
        members = [persona for persona in personas if persona.archetype == spec.name]
        if not members:
            continue
        row: dict[str, int | float | str] = {
            "archetype": spec.name,
            "count": len(members),
            "population_pct": round(100.0 * len(members) / population_size, 3),
            "avg_age": round(float(np.mean([member.age for member in members])), 3),
        }
        for trait in NORMALIZED_TRAITS:
            row[f"avg_{trait}"] = round(
                float(np.mean([getattr(member, trait) for member in members])),
                3,
            )
        row["avg_daily_scroll_hours"] = round(
            float(np.mean([member.daily_scroll_hours for member in members])),
            3,
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _normalized_trait(rng: np.random.Generator, skew: float) -> float:
    """Draw a centered trait, apply its archetype skew, and bound extremes."""

    value = np.clip(rng.beta(2.0, 2.0) + skew, 0.05, 0.95)
    return round(float(value), 6)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_PERSONAS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full persona JSON instead of the segment summary",
    )
    args = parser.parse_args(argv)

    personas = generate_personas(args.n, args.seed)
    if args.json:
        payload = [persona.model_dump(mode="json") for persona in personas]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(population_summary(personas).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHETYPES",
    "ARCHETYPE_BY_NAME",
    "ArchetypeSpec",
    "NORMALIZED_TRAITS",
    "generate_personas",
    "population_summary",
]
