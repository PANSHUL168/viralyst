"""Run Phase 4 audience reactions against one video's extracted features."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from agents.audience import heuristic_reaction, react_many
from agents.llm import health_check
from config import RANDOM_SEED
from personas.generator import generate_personas
from schemas import Persona, Reaction
from utils import setup_logging

LOGGER = logging.getLogger("viralyst.scripts.dry_run")


def reaction_table(
    personas: Sequence[Persona],
    reactions: Sequence[Reaction],
) -> pd.DataFrame:
    """Join reactions to their persona labels for human inspection."""

    persona_by_id = {persona.id: persona for persona in personas}
    rows: list[dict[str, object]] = []
    for reaction in reactions:
        persona = persona_by_id[reaction.persona_id]
        rows.append(
            {
                "persona": persona.name,
                "archetype": persona.archetype,
                "watch_pct": reaction.watch_percentage,
                "skipped": reaction.skipped,
                "liked": reaction.liked,
                "commented": reaction.commented,
                "shared": reaction.shared,
                "followed": reaction.followed_creator,
                "purchase_intent": reaction.purchase_intent,
                "fallback": reaction.fallback,
                "latency_ms": reaction.latency_ms,
                "reason": reaction.reason,
            }
        )
    return pd.DataFrame(rows)


def print_summary(table: pd.DataFrame) -> None:
    """Print compact overall and per-archetype diagnostics."""

    if table.empty:
        print("No reactions were generated.")
        return

    fallback_rate = 100.0 * float(table["fallback"].mean())
    print("\nSummary")
    print(f"Reactions: {len(table)}")
    print(f"Average watch: {table['watch_pct'].mean():.1f}%")
    print(f"Skip rate: {100.0 * table['skipped'].mean():.1f}%")
    print(f"Share rate: {100.0 * table['shared'].mean():.1f}%")
    print(f"Fallback rate: {fallback_rate:.1f}%")
    print(f"Average latency: {table['latency_ms'].mean():.0f} ms")

    segments = (
        table.groupby("archetype", sort=True)
        .agg(
            n=("persona", "count"),
            avg_watch=("watch_pct", "mean"),
            skip_rate=("skipped", "mean"),
            share_rate=("shared", "mean"),
        )
        .reset_index()
    )
    segments["avg_watch"] = segments["avg_watch"].round(1)
    segments["skip_rate"] = (100.0 * segments["skip_rate"]).round(1)
    segments["share_rate"] = (100.0 * segments["share_rate"]).round(1)
    print("\nBy archetype")
    print(segments.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--wave", type=int, default=0)
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Skip Ollama and exercise the deterministic fallback model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination for the complete reaction CSV",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.n <= 0:
        LOGGER.error("--n must be greater than zero")
        return 2
    if args.wave < 0:
        LOGGER.error("--wave must be greater than or equal to zero")
        return 2
    if not args.video.is_file():
        LOGGER.error("Video does not exist: '%s'", args.video)
        return 2

    if not args.heuristic_only:
        ready, message = health_check()
        if not ready:
            LOGGER.error(message)
            return 1
        LOGGER.info(message)

    try:
        from video.features import analyze_video

        LOGGER.info("Loading video features for %s", args.video)
        features = analyze_video(args.video)
        personas = generate_personas(args.n, args.seed)
        if args.heuristic_only:
            reactions = [
                heuristic_reaction(features, persona, args.wave)
                for persona in personas
            ]
        else:
            reactions = react_many(
                features,
                personas,
                args.wave,
                progress_cb=lambda done, total: LOGGER.info(
                    "Audience reactions: %d/%d", done, total
                ),
            )
    except Exception as exc:
        LOGGER.exception("Dry run failed: %s", exc)
        return 1

    table = reaction_table(personas, reactions)
    print(table.to_string(index=False))
    print_summary(table)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nSaved reactions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "print_summary", "reaction_table"]
