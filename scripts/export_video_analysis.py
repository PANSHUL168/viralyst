"""Export each completed video-analysis stage as inspectable artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import cv2

from utils import setup_logging, sha256_file, timer
from video import detector, extractor, transcriber
from video.features import build_features

LOGGER = logging.getLogger("viralyst.scripts.export_video_analysis")


def export_analysis(video_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run each perception stage once and persist all intermediate outputs."""

    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: '{video_path}'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "02_sampled_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    artifacts: dict[str, Any] = {}
    full_hash = sha256_file(video_path)
    source = {
        "filename": video_path.name,
        "source_path": str(video_path.resolve()),
        "sha256": full_hash,
        "size_bytes": video_path.stat().st_size,
    }
    _write_json(output_dir / "00_source.json", source)
    artifacts["source"] = "00_source.json"

    with timer("metadata") as measured:
        metadata = extractor.probe(video_path)
    timings[measured.label] = measured.elapsed_sec
    _write_json(output_dir / "01_metadata.json", metadata)
    artifacts["metadata"] = "01_metadata.json"

    with timer("frame_sampling") as measured:
        frames = extractor.sample_frames(video_path)
        frame_files: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            filename = f"frame_{index:02d}.jpg"
            destination = frames_dir / filename
            if not cv2.imwrite(str(destination), frame):
                raise OSError(f"Could not write sampled frame '{destination}'.")
            frame_files.append(
                {
                    "index": index,
                    "file": f"02_sampled_frames/{filename}",
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                }
            )
    timings[measured.label] = measured.elapsed_sec
    frame_manifest = {
        "sampled_frame_count": len(frames),
        "frames": frame_files,
    }
    _write_json(output_dir / "02_sampled_frames.json", frame_manifest)
    artifacts["sampled_frames"] = "02_sampled_frames.json"

    with timer("temporal_statistics") as measured:
        temporal = extractor.temporal_stats(frames)
    timings[measured.label] = measured.elapsed_sec
    _write_json(output_dir / "03_temporal_stats.json", temporal)
    artifacts["temporal_statistics"] = "03_temporal_stats.json"

    with timer("object_detection") as measured:
        objects = detector.detect(frames)
    timings[measured.label] = measured.elapsed_sec
    _write_json(
        output_dir / "04_object_detections.json",
        objects.model_dump(mode="json"),
    )
    artifacts["object_detection"] = "04_object_detections.json"

    audio_path = output_dir / "05_audio.wav"
    with timer("audio_extraction") as measured:
        extracted_audio = extractor.extract_audio(video_path, audio_path)
    timings[measured.label] = measured.elapsed_sec
    artifacts["audio"] = audio_path.name if extracted_audio else None

    with timer("transcription") as measured:
        transcript = (
            transcriber.transcribe(extracted_audio)
            if extracted_audio
            else transcriber.empty_transcript()
        )
    timings[measured.label] = measured.elapsed_sec
    _write_json(
        output_dir / "06_transcript.json",
        transcript.model_dump(mode="json"),
    )
    artifacts["transcript"] = "06_transcript.json"

    with timer("feature_fusion") as measured:
        features = build_features(
            video_path=video_path,
            video_hash=full_hash[:16],
            metadata=metadata,
            sampled_frame_count=len(frames),
            temporal=temporal,
            objects=objects,
            transcript=transcript,
        )
    timings[measured.label] = measured.elapsed_sec
    _write_json(
        output_dir / "07_video_features.json",
        features.model_dump(mode="json"),
    )
    artifacts["video_features"] = "07_video_features.json"

    manifest = {
        "status": "complete",
        "source_sha256": full_hash,
        "artifacts": artifacts,
        "timings_seconds": {
            name: round(seconds, 4) for name, seconds in timings.items()
        },
        "total_seconds": round(sum(timings.values()), 4),
    }
    _write_json(output_dir / "08_run_manifest.json", manifest)
    return manifest


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    setup_logging()

    try:
        manifest = export_analysis(args.video, args.output)
    except Exception as exc:
        LOGGER.exception("Analysis export failed: %s", exc)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
