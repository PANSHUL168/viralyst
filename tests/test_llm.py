"""Tests for the shared Ollama boundary without real model calls."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from agents import llm


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses: Iterator[Any] = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {"message": {"content": response}}


def test_chat_json_accepts_embedded_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(['Result: {"liked": true}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)

    result = llm.chat_json("system", "user", '{"liked": boolean}')

    assert result == {"liked": True}
    assert client.calls[0]["format"] == "json"


def test_chat_json_repairs_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(["not JSON", '{"watch_percentage": 70}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)

    result = llm.chat_json("system", "user", "schema", retries=1)

    assert result == {"watch_percentage": 70}
    assert len(client.calls) == 2
    assert "invalid JSON" in client.calls[1]["messages"][-1]["content"]


def test_chat_json_raises_after_repair_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm,
        "get_client",
        lambda: FakeClient(["bad", "still bad"]),
    )

    with pytest.raises(llm.LLMFormatError):
        llm.chat_json("system", "user", "schema", retries=1)


@pytest.mark.parametrize(
    ("models", "ready"),
    [
        ([{"name": llm.OLLAMA_MODEL}], True),
        ([{"name": "another-model:latest"}], False),
    ],
)
def test_health_check_detects_model(
    monkeypatch: pytest.MonkeyPatch,
    models: list[dict[str, str]],
    ready: bool,
) -> None:
    class HealthClient:
        def list(self) -> dict[str, object]:
            return {"models": models}

    monkeypatch.setattr(llm, "_make_client", lambda timeout: HealthClient())

    assert llm.health_check()[0] is ready


def test_health_check_handles_unreachable_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(timeout: float) -> object:
        raise OSError("connection refused")

    monkeypatch.setattr(llm, "_make_client", fail)

    ready, message = llm.health_check()

    assert ready is False
    assert "not reachable" in message


def test_run_batch_preserves_order_and_reports_progress() -> None:
    progress: list[tuple[int, int]] = []
    jobs = [lambda value=value: value for value in (4, 2, 9)]

    results = llm.run_batch(
        jobs,
        progress_cb=lambda done, total: progress.append((done, total)),
    )

    assert results == [4, 2, 9]
    assert progress[-1] == (3, 3)
