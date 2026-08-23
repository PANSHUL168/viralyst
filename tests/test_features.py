"""Tests for multimodal feature fusion and cache behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from schemas import ObjectStats, TranscriptResult, TranscriptSegment
from video import features


def _configure_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(features, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(features, "TMP_DIR", tmp_path / "tmp")


def _mock_pipeline(monkeypatch, transcript: TranscriptResult) -> None:
    monkeypatch.setattr(features, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        features.extractor,
        "probe",
        lambda _path: {
            "duration_sec": 10.0,
            "fps": 30.0,
            "frame_count": 300,
            "width": 1080,
            "height": 1920,
            "aspect_ratio": 0.5625,
            "is_vertical": True,
        },
    )
    monkeypatch.setattr(
        features.extractor,
        "sample_frames",
        lambda _path: [np.zeros((4, 4, 3), dtype=np.uint8)] * 2,
    )
    monkeypatch.setattr(
        features.extractor,
        "temporal_stats",
        lambda _frames: {
            "avg_brightness": 0.4,
            "brightness_variance": 0.02,
            "motion_score": 0.3,
            "scene_changes": 1,
        },
    )
    monkeypatch.setattr(
        features.detector,
        "detect",
        lambda _frames: ObjectStats(
            counts={"person": 2, "phone": 2},
            unique_classes=2,
            avg_objects_per_frame=2.0,
            dominant_class="person",
        ),
    )

    def fake_extract(_video_path, audio_path):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"RIFF")
        return audio_path

    monkeypatch.setattr(features.extractor, "extract_audio", fake_extract)
    monkeypatch.setattr(features.transcriber, "transcribe", lambda _path: transcript)


def _spoken_transcript() -> TranscriptResult:
    return TranscriptResult(
        transcript="one two three four five six seven eight",
        language="en",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="one two three four five six seven eight",
                no_speech_prob=0.1,
            )
        ],
        speech_duration_sec=2.0,
        word_count=8,
    )


def test_analyze_video_builds_and_caches_features(tmp_path, monkeypatch) -> None:
    _configure_paths(monkeypatch, tmp_path)
    _mock_pipeline(monkeypatch, _spoken_transcript())
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    result = features.analyze_video(video)

    assert result.video_hash == "a" * 16
    assert result.filename == "clip.mp4"
    assert result.is_vertical is True
    assert result.scene_change_rate == pytest.approx(0.1)
    assert result.speech_rate_wps == pytest.approx(4.0)
    assert result.silence_ratio == pytest.approx(0.8)
    assert result.hook_transcript.startswith("one two")
    assert result.pacing_label == "fast"
    assert result.density_label == "balanced"
    assert (tmp_path / "cache" / f"{'a' * 16}.json").is_file()
    assert not (tmp_path / "tmp" / f"{'a' * 16}.wav").exists()


def test_cache_hit_skips_expensive_analysis_and_updates_filename(
    tmp_path,
    monkeypatch,
) -> None:
    _configure_paths(monkeypatch, tmp_path)
    _mock_pipeline(monkeypatch, _spoken_transcript())
    original = tmp_path / "original.mp4"
    original.write_bytes(b"same video")
    features.analyze_video(original)

    renamed = tmp_path / "renamed.mp4"
    renamed.write_bytes(b"same video")
    monkeypatch.setattr(
        features.extractor,
        "probe",
        lambda _path: pytest.fail("cache hit should skip extraction"),
    )

    cached = features.analyze_video(renamed)

    assert cached.filename == "renamed.mp4"
    assert cached.video_hash == "a" * 16


def test_silent_video_skips_whisper(tmp_path, monkeypatch) -> None:
    _configure_paths(monkeypatch, tmp_path)
    _mock_pipeline(monkeypatch, _spoken_transcript())
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"silent")
    monkeypatch.setattr(features.extractor, "extract_audio", lambda *_args: None)
    monkeypatch.setattr(
        features.transcriber,
        "transcribe",
        lambda _path: pytest.fail("silent video should skip Whisper"),
    )

    result = features.analyze_video(video)

    assert result.has_speech is False
    assert result.transcript == ""
    assert result.word_count == 0
    assert result.speech_rate_wps == 0.0
    assert result.silence_ratio == 1.0


def test_corrupt_cache_is_regenerated(tmp_path, monkeypatch) -> None:
    _configure_paths(monkeypatch, tmp_path)
    _mock_pipeline(monkeypatch, _spoken_transcript())
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    cache_path = tmp_path / "cache" / f"{'a' * 16}.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not valid JSON", encoding="utf-8")

    result = features.analyze_video(video)

    assert result.video_hash == "a" * 16
    assert "not valid JSON" not in cache_path.read_text("utf-8")
