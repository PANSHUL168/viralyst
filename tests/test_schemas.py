"""Smoke tests for the foundational Pydantic contracts."""

from schemas import Reaction


def test_skipped_reaction_is_repaired() -> None:
    reaction = Reaction(
        persona_id=1,
        wave=0,
        watch_percentage=80,
        skipped=True,
        liked=True,
        commented=True,
        shared=True,
        followed_creator=True,
        purchase_intent=0.2,
        reason="Schema smoke test.",
    )

    assert reaction.watch_percentage == 30
    assert not reaction.liked
    assert not reaction.commented
    assert not reaction.shared
    assert not reaction.followed_creator

