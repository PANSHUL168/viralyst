"""Tests for OpenCV and FFmpeg video extraction helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from config import AUDIO_SAMPLE_RATE, DEFAULT_VIDEO_FPS, FRAME_RESIZE_LONG_SIDE
from video import extractor
from video.extractor import (
    AudioExtractionError,
    VideoReadError,
    extract_audio,
    probe,
    sample_frames,
    temporal_stats,
)


def _write_test_video(
    path: Path,
    *,
    size: tuple[int, int] = (80, 40),
    frame_count: int = 10,
    fps: float = 5.0,
) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG encoder is unavailable")
    try:
        width, height = size
        for index in range(frame_count):
            intensity = round(255 * index / max(frame_count - 1, 1))
            frame = np.full((height, width, 3), intensity, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def sample_video(tmp_path) -> Path:
    return _write_test_video(tmp_path / "sample.avi")


def test_probe_returns_expected_metadata(sample_video) -> None:
    metadata = probe(sample_video)

    assert metadata["fps"] == pytest.approx(5.0)
    assert metadata["frame_count"] == 10
    assert metadata["duration_sec"] == pytest.approx(2.0)
    assert metadata["width"] == 80
    assert metadata["height"] == 40
    assert metadata["aspect_ratio"] == pytest.approx(2.0)
    assert metadata["is_vertical"] is False


def test_probe_rejects_missing_video(tmp_path) -> None:
    with pytest.raises(VideoReadError, match="does not exist"):
        probe(tmp_path / "missing.mp4")


def test_probe_uses_default_for_invalid_fps(tmp_path, monkeypatch) -> None:
    path = tmp_path / "video.webm"
    path.write_bytes(b"placeholder")

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            values = {
                cv2.CAP_PROP_FPS: 0.0,
                cv2.CAP_PROP_FRAME_COUNT: 60.0,
                cv2.CAP_PROP_FRAME_WIDTH: 720.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 1280.0,
            }
            return values[property_id]

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())

    metadata = probe(path)
    assert metadata["fps"] == DEFAULT_VIDEO_FPS
    assert metadata["duration_sec"] == pytest.approx(2.0)
    assert metadata["is_vertical"] is True


def test_sample_frames_is_bounded_and_deterministic(sample_video) -> None:
    first = sample_frames(sample_video, max_frames=6, target_fps=10.0)
    second = sample_frames(sample_video, max_frames=6, target_fps=10.0)

    assert len(first) == 6
    assert len(second) == 6
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_sample_frames_preserves_aspect_ratio_when_resizing(tmp_path) -> None:
    path = _write_test_video(
        tmp_path / "large.avi",
        size=(800, 400),
        frame_count=5,
    )

    frames = sample_frames(path)

    assert len(frames) == 5
    assert frames[0].shape[:2] == (320, FRAME_RESIZE_LONG_SIDE)


def test_temporal_stats_handles_empty_frames() -> None:
    assert temporal_stats([]) == {
        "avg_brightness": 0.0,
        "brightness_variance": 0.0,
        "motion_score": 0.0,
        "scene_changes": 0,
    }


def test_temporal_stats_detects_brightness_motion_and_cut() -> None:
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full((8, 8, 3), 255, dtype=np.uint8)

    stats = temporal_stats([black, white])

    assert stats["avg_brightness"] == pytest.approx(0.5)
    assert stats["brightness_variance"] == pytest.approx(0.25)
    assert stats["motion_score"] == pytest.approx(1.0)
    assert stats["scene_changes"] == 1


def test_extract_audio_builds_expected_ffmpeg_command(
    tmp_path,
    sample_video,
    monkeypatch,
) -> None:
    output = tmp_path / "audio.wav"
    observed_command: list[str] = []

    def fake_run(command, **_kwargs):
        observed_command.extend(command)
        Path(command[-1]).write_bytes(b"RIFF-test-audio")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert extract_audio(sample_video, output) == output
    assert output.read_bytes() == b"RIFF-test-audio"
    assert observed_command[observed_command.index("-ar") + 1] == str(
        AUDIO_SAMPLE_RATE
    )
    assert observed_command[-1].endswith(".part.wav")


def test_extract_audio_returns_none_for_silent_video(
    tmp_path,
    sample_video,
    monkeypatch,
) -> None:
    output = tmp_path / "audio.wav"

    def fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            "Output file does not contain any stream",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert extract_audio(sample_video, output) is None
    assert not output.exists()
    assert not (tmp_path / "audio.part.wav").exists()


def test_extract_audio_reports_missing_ffmpeg(
    tmp_path,
    sample_video,
    monkeypatch,
) -> None:
    def missing_ffmpeg(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_ffmpeg)

    with pytest.raises(AudioExtractionError, match="was not found"):
        extract_audio(sample_video, tmp_path / "audio.wav")


def test_cli_prints_extraction_result(sample_video, capsys) -> None:
    assert extractor.main([str(sample_video)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["frame_count"] == 10
    assert output["sampled_frame_count"] == 5
    assert 0.0 <= output["motion_score"] <= 1.0
