"""Lazy faster-whisper transcription and hook extraction."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config import (
    HOOK_SECONDS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DOWNLOAD_ROOT,
    WHISPER_MODEL_SIZE,
    WHISPER_NO_SPEECH_THRESHOLD,
)
from schemas import TranscriptResult, TranscriptSegment

_MODEL: Any | None = None
_MODEL_LOCK = threading.Lock()


class TranscriptionError(RuntimeError):
    """Raised when Whisper cannot load or transcribe an audio file."""


def get_model() -> Any:
    """Return a process-wide lazy faster-whisper singleton."""

    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = _load_model()
    return _MODEL


def _load_model() -> Any:
    """Load the configured CPU model into the project cache."""

    WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        from faster_whisper import WhisperModel

        return WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=str(WHISPER_DOWNLOAD_ROOT),
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Could not load Whisper model '{WHISPER_MODEL_SIZE}'."
        ) from exc


def transcribe(wav_path: str | Path) -> TranscriptResult:
    """Transcribe a WAV and return normalized, validated speech data."""

    audio_path = Path(wav_path)
    if not audio_path.is_file():
        raise TranscriptionError(f"Audio file does not exist: '{audio_path}'.")

    try:
        raw_segments, info = get_model().transcribe(
            str(audio_path),
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments: list[TranscriptSegment] = []
        for raw_segment in raw_segments:
            text = " ".join(str(getattr(raw_segment, "text", "")).split())
            no_speech_prob = float(
                getattr(raw_segment, "no_speech_prob", 0.0) or 0.0
            )
            if not text or no_speech_prob > WHISPER_NO_SPEECH_THRESHOLD:
                continue
            start = max(0.0, float(getattr(raw_segment, "start", 0.0)))
            end = max(start, float(getattr(raw_segment, "end", start)))
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    no_speech_prob=min(max(no_speech_prob, 0.0), 1.0),
                )
            )
    except ValueError as exc:
        # faster-whisper 1.0.3 raises this during language detection when VAD
        # removes every frame (for example, music-only or silent audio).
        if "max() iterable argument is empty" in str(exc):
            return empty_transcript()
        raise TranscriptionError(
            f"Whisper failed to transcribe '{audio_path}'."
        ) from exc
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(
            f"Whisper failed to transcribe '{audio_path}'."
        ) from exc

    transcript = " ".join(segment.text for segment in segments)
    language = str(getattr(info, "language", "unknown") or "unknown")
    return TranscriptResult(
        transcript=transcript,
        language=language,
        segments=segments,
        speech_duration_sec=sum(segment.end - segment.start for segment in segments),
        word_count=len(transcript.split()),
    )


def hook_text(
    segments: Sequence[TranscriptSegment],
    seconds: float = HOOK_SECONDS,
) -> str:
    """Join speech segments that overlap the opening time window."""

    if seconds <= 0:
        raise ValueError("seconds must be greater than zero")
    return " ".join(
        segment.text
        for segment in segments
        if segment.start < seconds and segment.end > 0
    )


def empty_transcript(language: str = "unknown") -> TranscriptResult:
    """Return the canonical transcription result for a silent video."""

    return TranscriptResult(
        transcript="",
        language=language,
        segments=[],
        speech_duration_sec=0.0,
        word_count=0,
    )


__all__ = [
    "TranscriptionError",
    "empty_transcript",
    "get_model",
    "hook_text",
    "transcribe",
]
