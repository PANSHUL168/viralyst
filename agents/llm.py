"""Shared Ollama JSON client, health checks, retry, and batch execution."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar, cast

import ollama

from config import (
    AUDIENCE_TEMPERATURE,
    LLM_RETRIES,
    MAX_WORKERS,
    OLLAMA_HEALTH_TIMEOUT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_REQUEST_TIMEOUT,
)
from utils import JSONCoercionError, coerce_json

T = TypeVar("T")
ProgressCallback = Callable[[int, int], None]

_CLIENT: ollama.Client | None = None
_CLIENT_LOCK = threading.Lock()


class LLMError(RuntimeError):
    """Base exception for controlled local-LLM failures."""


class OllamaUnavailableError(LLMError):
    """Raised when the Ollama service cannot be reached."""


class ModelUnavailableError(LLMError):
    """Raised when Ollama does not have the configured model."""


class LLMRequestError(LLMError):
    """Raised when Ollama rejects or fails a generation request."""


class LLMFormatError(LLMError):
    """Raised when repair attempts cannot produce a JSON object."""


def get_client() -> ollama.Client:
    """Return the shared Ollama client used for generation requests."""

    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = _make_client(OLLAMA_REQUEST_TIMEOUT)
    return _CLIENT


def _make_client(timeout: float) -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST, timeout=timeout)


def health_check() -> tuple[bool, str]:
    """Check service reachability and confirm the configured model exists."""

    try:
        response = _make_client(OLLAMA_HEALTH_TIMEOUT).list()
    except Exception as exc:
        return (
            False,
            f"Ollama is not reachable at {OLLAMA_HOST}. Run 'ollama serve'. "
            f"Details: {exc}",
        )

    model_names = _model_names(response)
    configured = OLLAMA_MODEL.casefold()
    aliases = {configured}
    if ":" not in configured:
        aliases.add(f"{configured}:latest")
    if not aliases.intersection(model_names):
        return (
            False,
            f"Model '{OLLAMA_MODEL}' is not installed. Run: "
            f"ollama pull {OLLAMA_MODEL}",
        )
    return True, f"Ollama is ready with model '{OLLAMA_MODEL}'."


def _model_names(response: Any) -> set[str]:
    """Read model names from mapping or object responses across client versions."""

    models = (
        response.get("models", [])
        if isinstance(response, Mapping)
        else getattr(response, "models", [])
    )
    names: set[str] = set()
    for model in models:
        if isinstance(model, Mapping):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.add(str(name).casefold())
    return names


def chat_json(
    system: str,
    user: str,
    schema_hint: str,
    temperature: float = AUDIENCE_TEMPERATURE,
    retries: int = LLM_RETRIES,
) -> dict[str, Any]:
    """Request one JSON object, retrying malformed output with a repair prompt."""

    if retries < 0:
        raise ValueError("retries must be greater than or equal to zero")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        response = _chat(messages, temperature)
        content = _response_content(response)
        try:
            return coerce_json(content)
        except (JSONCoercionError, TypeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was invalid JSON. Return ONLY "
                            "one JSON object matching this shape:\n"
                            f"{schema_hint}"
                        ),
                    },
                ]
            )

    raise LLMFormatError(
        f"Ollama did not return a valid JSON object after {retries + 1} attempt(s)."
    ) from last_error


def _chat(messages: list[dict[str, str]], temperature: float) -> Any:
    try:
        return get_client().chat(
            model=OLLAMA_MODEL,
            messages=messages,
            format="json",
            options={
                "temperature": temperature,
                "num_predict": OLLAMA_NUM_PREDICT,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        )
    except ollama.RequestError as exc:
        raise OllamaUnavailableError(
            f"Ollama is not reachable at {OLLAMA_HOST}."
        ) from exc
    except ollama.ResponseError as exc:
        message = str(exc)
        if exc.status_code == 404 or "not found" in message.casefold():
            raise ModelUnavailableError(
                f"Model '{OLLAMA_MODEL}' is unavailable. Run: "
                f"ollama pull {OLLAMA_MODEL}"
            ) from exc
        raise LLMRequestError(f"Ollama request failed: {message}") from exc
    except LLMError:
        raise
    except Exception as exc:
        raise LLMRequestError(f"Ollama request failed: {exc}") from exc


def _response_content(response: Any) -> str:
    if isinstance(response, Mapping):
        message = response.get("message", {})
    else:
        message = getattr(response, "message", {})
    if isinstance(message, Mapping):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return str(content or "")


def run_batch(
    jobs: Sequence[Callable[[], T]],
    max_workers: int = MAX_WORKERS,
    progress_cb: ProgressCallback | None = None,
) -> list[T]:
    """Run jobs concurrently while preserving input order in the results."""

    if max_workers <= 0:
        raise ValueError("max_workers must be greater than zero")
    if not jobs:
        return []

    results: list[Any] = [None] * len(jobs)
    completed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
        future_indices = {
            executor.submit(job): index for index, job in enumerate(jobs)
        }
        for future in as_completed(future_indices):
            index = future_indices[future]
            results[index] = future.result()
            completed += 1
            if progress_cb is not None:
                progress_cb(completed, len(jobs))

    return cast(list[T], results)


__all__ = [
    "LLMError",
    "LLMFormatError",
    "LLMRequestError",
    "ModelUnavailableError",
    "OllamaUnavailableError",
    "ProgressCallback",
    "chat_json",
    "get_client",
    "health_check",
    "run_batch",
]
