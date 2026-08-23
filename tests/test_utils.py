"""Tests for shared utility functions."""

from __future__ import annotations

import hashlib
import io
import logging

import pytest

import utils
from utils import JSONCoercionError, coerce_json, setup_logging, sha256_file, timer


def test_sha256_file_streams_expected_digest(tmp_path) -> None:
    content = b"viralyst" * 10
    path = tmp_path / "sample.bin"
    path.write_bytes(content)

    assert sha256_file(path, chunk_size=3) == hashlib.sha256(content).hexdigest()


def test_sha256_file_rejects_invalid_chunk_size(tmp_path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"data")

    with pytest.raises(ValueError, match="greater than zero"):
        sha256_file(path, chunk_size=0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('{"shared": false}', {"shared": False}),
        (b'{"watch_percentage": 72}', {"watch_percentage": 72}),
        ("```json\n{\"liked\": true}\n```", {"liked": True}),
        (
            'Model output follows: {"reason": "specific {hook}"} done.',
            {"reason": "specific {hook}"},
        ),
        ({"wave": 2}, {"wave": 2}),
    ],
)
def test_coerce_json_accepts_supported_formats(value, expected) -> None:
    assert coerce_json(value) == expected


@pytest.mark.parametrize("value", ["", "[]", "not JSON", b"\xff"])
def test_coerce_json_rejects_invalid_objects(value) -> None:
    with pytest.raises(JSONCoercionError):
        coerce_json(value)


def test_setup_logging_is_idempotent() -> None:
    project_logger = logging.getLogger(utils.LOGGER_NAME)
    project_logger.handlers.clear()
    stream = io.StringIO()

    first = setup_logging("DEBUG", stream=stream)
    second = setup_logging("INFO", stream=stream)
    second.info("ready")

    assert first is second
    assert len(project_logger.handlers) == 1
    assert "ready" in stream.getvalue()


def test_timer_records_elapsed_time(monkeypatch) -> None:
    readings = iter([10.0, 10.125])
    monkeypatch.setattr(utils, "perf_counter", lambda: next(readings))

    with timer("stage") as result:
        pass

    assert result.label == "stage"
    assert result.elapsed_sec == pytest.approx(0.125)
    assert result.elapsed_ms == 125
