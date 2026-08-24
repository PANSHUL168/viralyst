"""Tests for the Phase 4 command-line checkpoint."""

import pandas as pd

from personas.generator import generate_personas
from schemas import Reaction
from scripts.dry_run import print_summary, reaction_table


def test_reaction_table_and_summary(capsys) -> None:
    personas = generate_personas(1, seed=4)
    reaction = Reaction(
        persona_id=personas[0].id,
        wave=0,
        watch_percentage=72,
        skipped=False,
        liked=True,
        commented=False,
        shared=False,
        followed_creator=False,
        purchase_intent=0.2,
        reason="The programming example matched my interests.",
    )

    table = reaction_table(personas, [reaction])
    print_summary(table)

    assert isinstance(table, pd.DataFrame)
    assert table.loc[0, "archetype"] == personas[0].archetype
    assert "Fallback rate: 0.0%" in capsys.readouterr().out
