"""Lazy YOLO object detection over sampled video frames."""

from __future__ import annotations

import os
import threading
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from config import TMP_DIR, YOLO_CONF, YOLO_MODEL
from schemas import ObjectStats

_MODEL: Any | None = None
_MODEL_LOCK = threading.Lock()


class DetectionError(RuntimeError):
    """Raised when YOLO cannot load or complete inference."""


def get_model() -> Any:
    """Return a process-wide lazy YOLO singleton."""

    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = _load_model()
    return _MODEL


def _load_model() -> Any:
    """Load YOLO while keeping its writable settings inside the project."""

    ultralytics_dir = TMP_DIR / "ultralytics"
    matplotlib_dir = TMP_DIR / "matplotlib"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    try:
        from ultralytics import YOLO

        return YOLO(YOLO_MODEL)
    except Exception as exc:
        raise DetectionError(f"Could not load YOLO model '{YOLO_MODEL}'.") from exc


def detect(
    frames: Sequence[np.ndarray],
    conf: float = YOLO_CONF,
) -> ObjectStats:
    """Detect and aggregate COCO object classes across sampled frames."""

    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")
    if not frames:
        return _empty_stats()
    for frame in frames:
        if (
            not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError(
                "frames must be non-empty BGR arrays with shape (height, width, 3)"
            )

    model = get_model()
    try:
        results = model.predict(source=list(frames), conf=conf, verbose=False)
        counts: Counter[str] = Counter()
        for result in results:
            names = getattr(result, "names", None) or getattr(model, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None or getattr(boxes, "cls", None) is None:
                continue
            for class_id in _to_class_ids(boxes.cls):
                counts[_class_name(names, class_id)] += 1
    except DetectionError:
        raise
    except Exception as exc:
        raise DetectionError("YOLO inference failed.") from exc

    total_objects = sum(counts.values())
    dominant_class = (
        min(counts, key=lambda name: (-counts[name], name)) if counts else None
    )
    return ObjectStats(
        counts=dict(sorted(counts.items())),
        unique_classes=len(counts),
        avg_objects_per_frame=total_objects / len(frames),
        dominant_class=dominant_class,
    )


def _to_class_ids(values: Any) -> list[int]:
    """Convert Torch, NumPy, or plain class arrays into integer IDs."""

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        values = [values]
    return [int(value) for value in values]


def _class_name(names: Any, class_id: int) -> str:
    """Resolve a model class ID without assuming mapping or list storage."""

    try:
        return str(names[class_id])
    except (KeyError, IndexError, TypeError):
        return f"class_{class_id}"


def _empty_stats() -> ObjectStats:
    return ObjectStats(
        counts={},
        unique_classes=0,
        avg_objects_per_frame=0.0,
        dominant_class=None,
    )


__all__ = ["DetectionError", "detect", "get_model"]
