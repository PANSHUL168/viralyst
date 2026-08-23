"""Fuse OpenCV, YOLO, and Whisper output into cached video features."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pydantic import ValidationError

from config import (
    BUSY_OBJECTS_PER_FRAME,
    CACHE_DIR,
    FAST_SCENE_CHANGE_RATE,
    FAST_SPEECH_RATE_WPS,
    SLOW_SCENE_CHANGE_RATE,
    SLOW_SPEECH_RATE_WPS,
    SPARSE_OBJECTS_PER_FRAME,
    TMP_DIR,
)
from schemas import ObjectStats, TranscriptResult, VideoFeatures
from utils import setup_logging, sha256_file
from video import detector, extractor, transcriber
from video.detector import DetectionError
from video.extractor import AudioExtractionError, VideoReadError
from video.transcriber import TranscriptionError

LOGGER = logging.getLogger("viralyst.video.features")


def analyze_video(path: str | Path, use_cache: bool = True) -> VideoFeatures:
    """Analyze a video and return a validated multimodal feature object."""

    video_path = Path(path)
    if not video_path.is_file():
        raise VideoReadError(f"Video file does not exist: '{video_path}'.")

    video_hash = sha256_file(video_path)[:16]
    cache_path = CACHE_DIR / f"{video_hash}.json"
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached.model_copy(update={"filename": video_path.name})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    metadata = extractor.probe(video_path)
    frames = extractor.sample_frames(video_path)
    temporal = extractor.temporal_stats(frames)
    objects = detector.detect(frames)
    transcript = _analyze_audio(video_path, video_hash)

    features = build_features(
        video_path=video_path,
        video_hash=video_hash,
        metadata=metadata,
        sampled_frame_count=len(frames),
        temporal=temporal,
        objects=objects,
        transcript=transcript,
    )
    _write_cache(cache_path, features)
    return features


def _analyze_audio(video_path: Path, video_hash: str) -> TranscriptResult:
    """Extract, transcribe, and clean up temporary audio."""

    audio_path = TMP_DIR / f"{video_hash}.wav"
    try:
        extracted_path = extractor.extract_audio(video_path, audio_path)
        if extracted_path is None:
            return transcriber.empty_transcript()
        return transcriber.transcribe(extracted_path)
    finally:
        audio_path.unlink(missing_ok=True)
        audio_path.with_name(f"{audio_path.stem}.part{audio_path.suffix}").unlink(
            missing_ok=True
        )


def build_features(
    *,
    video_path: Path,
    video_hash: str,
    metadata: dict[str, int | float | bool],
    sampled_frame_count: int,
    temporal: dict[str, int | float],
    objects: ObjectStats,
    transcript: TranscriptResult,
) -> VideoFeatures:
    """Calculate derived values and validate the public feature contract."""

    duration_sec = max(float(metadata["duration_sec"]), 0.0)
    scene_changes = int(temporal["scene_changes"])
    scene_change_rate = scene_changes / duration_sec if duration_sec else 0.0
    speech_duration_sec = min(
        max(transcript.speech_duration_sec, 0.0),
        duration_sec,
    )
    has_speech = bool(
        transcript.transcript.strip()
        and transcript.word_count > 0
        and speech_duration_sec > 0
    )
    speech_rate_wps = (
        transcript.word_count / max(speech_duration_sec, 1e-6)
        if has_speech
        else 0.0
    )
    silence_ratio = (
        float(np.clip(1.0 - speech_duration_sec / duration_sec, 0.0, 1.0))
        if duration_sec
        else 1.0
    )

    return VideoFeatures(
        video_hash=video_hash,
        filename=video_path.name,
        duration_sec=duration_sec,
        fps=float(metadata["fps"]),
        frame_count=int(metadata["frame_count"]),
        sampled_frame_count=sampled_frame_count,
        scene_changes=scene_changes,
        scene_change_rate=scene_change_rate,
        avg_brightness=float(temporal["avg_brightness"]),
        brightness_variance=float(temporal["brightness_variance"]),
        motion_score=float(temporal["motion_score"]),
        aspect_ratio=float(metadata["aspect_ratio"]),
        is_vertical=bool(metadata["is_vertical"]),
        objects=objects,
        transcript=transcript.transcript,
        language=transcript.language,
        word_count=transcript.word_count,
        speech_duration_sec=speech_duration_sec,
        speech_rate_wps=speech_rate_wps,
        silence_ratio=silence_ratio,
        has_speech=has_speech,
        hook_transcript=transcriber.hook_text(transcript.segments),
        pacing_label=_pacing_label(speech_rate_wps, scene_change_rate),
        density_label=_density_label(objects.avg_objects_per_frame),
    )


def _pacing_label(speech_rate_wps: float, scene_change_rate: float) -> str:
    if (
        speech_rate_wps > FAST_SPEECH_RATE_WPS
        or scene_change_rate > FAST_SCENE_CHANGE_RATE
    ):
        return "fast"
    if (
        speech_rate_wps < SLOW_SPEECH_RATE_WPS
        and scene_change_rate < SLOW_SCENE_CHANGE_RATE
    ):
        return "slow"
    return "moderate"


def _density_label(avg_objects_per_frame: float) -> str:
    if avg_objects_per_frame > BUSY_OBJECTS_PER_FRAME:
        return "busy"
    if avg_objects_per_frame < SPARSE_OBJECTS_PER_FRAME:
        return "sparse"
    return "balanced"


def _load_cache(cache_path: Path) -> VideoFeatures | None:
    """Return a valid cache entry or allow analysis to regenerate it."""

    if not cache_path.is_file():
        return None
    try:
        return VideoFeatures.model_validate_json(cache_path.read_text("utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        LOGGER.warning("Ignoring invalid feature cache %s: %s", cache_path, exc)
        return None


def _write_cache(cache_path: Path, features: VideoFeatures) -> None:
    """Atomically persist a feature result."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
    )
    try:
        serialized = json.dumps(
            features.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Analyze a video and print its complete feature JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force analysis even when a cached result exists",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print JSON on one line",
    )
    args = parser.parse_args(argv)
    setup_logging()

    try:
        result = analyze_video(args.video, use_cache=not args.no_cache)
    except (
        AudioExtractionError,
        DetectionError,
        TranscriptionError,
        VideoReadError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1

    payload: dict[str, Any] = result.model_dump(mode="json")
    print(
        json.dumps(
            payload,
            indent=None if args.compact else 2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["analyze_video", "build_features"]
