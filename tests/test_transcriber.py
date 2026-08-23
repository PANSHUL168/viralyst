"""Tests for faster-whisper normalization without model downloads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from schemas import TranscriptSegment
from video import transcriber
from video.transcriber import (
    TranscriptionError,
    empty_transcript,
    hook_text,
    transcribe,
)


def test_transcribe_normalizes_and_filters_segments(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    raw_segments = [
        SimpleNamespace(
            start=0.0,
            end=1.5,
            text="  Strong   opening ",
            no_speech_prob=0.1,
        ),
        SimpleNamespace(
            start=1.5,
            end=2.0,
            text="hallucination",
            no_speech_prob=0.9,
        ),
        SimpleNamespace(
            start=2.0,
            end=4.0,
            text=" useful details ",
            no_speech_prob=0.2,
        ),
    ]

    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert path == str(audio)
            assert kwargs == {
                "beam_size": 1,
                "vad_filter": True,
                "condition_on_previous_text": False,
            }
            return iter(raw_segments), SimpleNamespace(language="en")

    monkeypatch.setattr(transcriber, "get_model", lambda: FakeModel())

    result = transcribe(audio)

    assert result.transcript == "Strong opening useful details"
    assert result.language == "en"
    assert result.word_count == 4
    assert result.speech_duration_sec == pytest.approx(3.5)
    assert len(result.segments) == 2


def test_transcribe_requires_existing_audio(tmp_path) -> None:
    with pytest.raises(TranscriptionError, match="does not exist"):
        transcribe(tmp_path / "missing.wav")


def test_transcribe_treats_empty_vad_language_detection_as_no_speech(
    tmp_path,
    monkeypatch,
) -> None:
    audio = tmp_path / "music.wav"
    audio.write_bytes(b"RIFF")

    class MusicOnlyModel:
        def transcribe(self, *_args, **_kwargs):
            raise ValueError("max() iterable argument is empty")

    monkeypatch.setattr(transcriber, "get_model", lambda: MusicOnlyModel())

    assert transcribe(audio) == empty_transcript()


def test_hook_text_uses_overlapping_opening_segments() -> None:
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="first"),
        TranscriptSegment(start=2.5, end=3.5, text="overlap"),
        TranscriptSegment(start=3.0, end=4.0, text="late"),
    ]

    assert hook_text(segments, seconds=3.0) == "first overlap"


def test_empty_transcript_is_canonical() -> None:
    result = empty_transcript()

    assert result.transcript == ""
    assert result.language == "unknown"
    assert result.segments == []
    assert result.word_count == 0
    assert result.speech_duration_sec == 0.0


def test_get_model_is_singleton(monkeypatch) -> None:
    loaded: list[object] = []

    def fake_load():
        model = object()
        loaded.append(model)
        return model

    monkeypatch.setattr(transcriber, "_MODEL", None)
    monkeypatch.setattr(transcriber, "_load_model", fake_load)

    assert transcriber.get_model() is transcriber.get_model()
    assert len(loaded) == 1
