"""Cross-cutting helpers for logging, hashing, timing, and JSON coercion."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

LOGGER_NAME = "viralyst"
DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024
_HANDLER_MARKER = "_viralyst_handler"


class JSONCoercionError(ValueError):
    """Raised when text does not contain a valid JSON object."""


@dataclass(slots=True)
class TimerResult:
    """Timing information populated when a :func:`timer` block exits."""

    label: str
    elapsed_sec: float = 0.0

    @property
    def elapsed_ms(self) -> int:
        """Return elapsed time rounded to the nearest millisecond."""

        return round(self.elapsed_sec * 1000)


def setup_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure and return the project logger without duplicating handlers.

    Calling this function repeatedly is safe, which is important under
    Streamlit's rerun model. The logger does not propagate to the root logger.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    handler = next(
        (
            existing
            for existing in logger.handlers
            if getattr(existing, _HANDLER_MARKER, False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(stream or sys.stderr)
        setattr(handler, _HANDLER_MARKER, True)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    handler.setLevel(level)
    return logger


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Return the hexadecimal SHA-256 digest of a file using bounded memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def timer(
    label: str,
    *,
    logger: logging.Logger | None = None,
) -> Iterator[TimerResult]:
    """Measure a block and optionally log its duration, including on failure."""

    result = TimerResult(label=label)
    started_at = perf_counter()
    try:
        yield result
    finally:
        result.elapsed_sec = perf_counter() - started_at
        if logger is not None:
            logger.info("%s completed in %.3fs", label, result.elapsed_sec)


def coerce_json(value: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Return the first valid JSON object found in common LLM output formats.

    Clean JSON, UTF-8 bytes, Markdown code fences, and prose surrounding a JSON
    object are accepted. Arrays and scalar JSON values are rejected because all
    Viralyst agent contracts require an object at the top level.
    """

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JSONCoercionError("JSON bytes must be UTF-8 encoded") from exc
    if not isinstance(value, str):
        raise TypeError("value must be a string, UTF-8 bytes, or mapping")

    text = value.strip().lstrip("\ufeff")
    if not text:
        raise JSONCoercionError("JSON content is empty")

    direct = _decode_json_object(text)
    if direct is not None:
        return direct

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate

    raise JSONCoercionError("response does not contain a valid JSON object")


def _decode_json_object(text: str) -> dict[str, Any] | None:
    """Decode ``text`` only when its complete JSON value is an object."""

    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        return None
    return candidate if isinstance(candidate, dict) else None


__all__ = [
    "JSONCoercionError",
    "TimerResult",
    "coerce_json",
    "setup_logging",
    "sha256_file",
    "timer",
]
