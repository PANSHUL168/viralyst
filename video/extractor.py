"""Video metadata, frame sampling, temporal statistics, and audio extraction."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from config import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_VIDEO_FPS,
    FFMPEG_BINARY,
    FFMPEG_TIMEOUT_SECONDS,
    FRAME_RESIZE_LONG_SIDE,
    FRAME_SAMPLE_FPS,
    MAX_SAMPLED_FRAMES,
    MIN_SAMPLED_FRAMES,
    SCENE_CHANGE_THRESHOLD,
)
from utils import setup_logging

LOGGER = logging.getLogger("viralyst.video.extractor")
_NO_AUDIO_MESSAGES = (
    "does not contain any stream",
    "matches no streams",
    "no audio",
)


class VideoReadError(RuntimeError):
    """Raised when OpenCV cannot read usable video metadata or frames."""


class AudioExtractionError(RuntimeError):
    """Raised when FFmpeg cannot extract an available audio track."""


def probe(path: str | Path) -> dict[str, int | float | bool]:
    """Read core video metadata with OpenCV.

    Videos that report an invalid FPS use ``DEFAULT_VIDEO_FPS`` so common
    WebM metadata problems do not cause division-by-zero failures.
    """

    video_path = _require_file(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(
            f"Could not open '{video_path}'. Try re-encoding it as H.264 MP4."
        )

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()

    if frame_count <= 0:
        raise VideoReadError(f"Video '{video_path}' contains no readable frames.")
    if width <= 0 or height <= 0:
        raise VideoReadError(f"Video '{video_path}' has invalid dimensions.")
    if not math.isfinite(fps) or fps <= 0:
        LOGGER.warning(
            "Video reported invalid FPS %r; using %.1f FPS",
            fps,
            DEFAULT_VIDEO_FPS,
        )
        fps = DEFAULT_VIDEO_FPS

    duration_sec = frame_count / fps
    aspect_ratio = width / height
    return {
        "duration_sec": float(duration_sec),
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "aspect_ratio": float(aspect_ratio),
        "is_vertical": height > width,
    }


def sample_frames(
    path: str | Path,
    max_frames: int = MAX_SAMPLED_FRAMES,
    target_fps: float = FRAME_SAMPLE_FPS,
) -> list[np.ndarray]:
    """Return deterministic, evenly spaced BGR frames from a video."""

    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a finite value greater than zero")

    video_path = _require_file(path)
    metadata = probe(video_path)
    frame_count = int(metadata["frame_count"])
    duration_sec = float(metadata["duration_sec"])
    estimated_count = max(
        MIN_SAMPLED_FRAMES,
        math.ceil(duration_sec * target_fps),
    )
    sample_count = min(frame_count, max_frames, estimated_count)
    indices = np.linspace(
        0,
        frame_count - 1,
        num=sample_count,
        dtype=np.int64,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"Could not reopen '{video_path}' for frame sampling.")

    frames: list[np.ndarray] = []
    try:
        for frame_index in indices.tolist():
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                LOGGER.warning(
                    "Could not decode frame %d from %s",
                    frame_index,
                    video_path,
                )
                continue
            frames.append(_resize_frame(frame, FRAME_RESIZE_LONG_SIDE))
    finally:
        capture.release()

    if not frames:
        raise VideoReadError(f"No frames could be decoded from '{video_path}'.")
    return frames


def temporal_stats(frames: Sequence[np.ndarray]) -> dict[str, int | float]:
    """Calculate normalized brightness, motion, and approximate scene cuts."""

    if not frames:
        return {
            "avg_brightness": 0.0,
            "brightness_variance": 0.0,
            "motion_score": 0.0,
            "scene_changes": 0,
        }

    grayscale_frames: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    brightness_values: list[float] = []

    for frame in frames:
        _validate_frame(frame)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_frames.append(grayscale)
        brightness_values.append(float(np.mean(grayscale) / 255.0))

        histogram = cv2.calcHist(
            [frame],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256],
        )
        histograms.append(cv2.normalize(histogram, histogram).flatten())

    motion_values: list[float] = []
    scene_changes = 0
    for index in range(1, len(grayscale_frames)):
        previous = grayscale_frames[index - 1]
        current = grayscale_frames[index]
        if current.shape != previous.shape:
            current = cv2.resize(
                current,
                (previous.shape[1], previous.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        difference = cv2.absdiff(previous, current)
        motion_values.append(float(np.mean(difference) / 255.0))

        correlation = cv2.compareHist(
            histograms[index - 1],
            histograms[index],
            cv2.HISTCMP_CORREL,
        )
        if correlation < SCENE_CHANGE_THRESHOLD:
            scene_changes += 1

    return {
        "avg_brightness": float(np.clip(np.mean(brightness_values), 0.0, 1.0)),
        "brightness_variance": float(max(np.var(brightness_values), 0.0)),
        "motion_score": float(
            np.clip(np.mean(motion_values) if motion_values else 0.0, 0.0, 1.0)
        ),
        "scene_changes": scene_changes,
    }


def extract_audio(path: str | Path, out_wav: str | Path) -> Path | None:
    """Extract mono 16 kHz PCM audio, returning ``None`` for silent videos."""

    video_path = _require_file(path)
    output_path = Path(out_wav)
    if video_path.resolve() == output_path.resolve():
        raise ValueError("audio output path must differ from the video path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.part{output_path.suffix or '.wav'}"
    )
    temporary_path.unlink(missing_ok=True)

    command = [
        FFMPEG_BINARY,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(temporary_path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioExtractionError(
            f"FFmpeg executable '{FFMPEG_BINARY}' was not found."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        temporary_path.unlink(missing_ok=True)
        raise AudioExtractionError(
            f"FFmpeg exceeded the {FFMPEG_TIMEOUT_SECONDS}-second timeout."
        ) from exc

    if completed.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        diagnostic = (completed.stderr or "").strip()
        if any(message in diagnostic.lower() for message in _NO_AUDIO_MESSAGES):
            return None
        detail = diagnostic[-500:] or "unknown FFmpeg error"
        raise AudioExtractionError(f"FFmpeg audio extraction failed: {detail}")

    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        return None

    temporary_path.replace(output_path)
    return output_path


def _require_file(path: str | Path) -> Path:
    """Return a verified file path with a consistent public error."""

    video_path = Path(path)
    if not video_path.is_file():
        raise VideoReadError(f"Video file does not exist: '{video_path}'.")
    return video_path


def _resize_frame(frame: np.ndarray, longest_side: int) -> np.ndarray:
    """Downscale a frame while preserving its aspect ratio."""

    _validate_frame(frame)
    if longest_side <= 0:
        raise ValueError("longest_side must be greater than zero")

    height, width = frame.shape[:2]
    current_longest = max(width, height)
    if current_longest <= longest_side:
        return frame

    scale = longest_side / current_longest
    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _validate_frame(frame: np.ndarray) -> None:
    """Reject malformed arrays before passing them into OpenCV."""

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frames must be non-empty NumPy arrays")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frames must be BGR arrays with shape (height, width, 3)")


def main(argv: Sequence[str] | None = None) -> int:
    """Print metadata and temporal statistics for a video."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--audio-out",
        type=Path,
        help="Optionally extract audio to this WAV path",
    )
    args = parser.parse_args(argv)
    setup_logging()

    try:
        metadata = probe(args.video)
        frames = sample_frames(args.video)
        result: dict[str, Any] = {
            **metadata,
            "sampled_frame_count": len(frames),
            **temporal_stats(frames),
        }
        if args.audio_out is not None:
            audio_path = extract_audio(args.video, args.audio_out)
            result["audio_path"] = str(audio_path) if audio_path else None
    except (VideoReadError, AudioExtractionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AudioExtractionError",
    "VideoReadError",
    "extract_audio",
    "probe",
    "sample_frames",
    "temporal_stats",
]
