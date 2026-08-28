"""Safe upload persistence and state serialization helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config import MAX_UPLOAD_MB, TMP_DIR

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


class UploadValidationError(ValueError):
    """Raised when an uploaded file is not an accepted local video."""


def validate_upload(
    filename: str,
    size_bytes: int,
    max_upload_mb: int = MAX_UPLOAD_MB,
) -> str:
    """Validate upload metadata and return its normalized suffix."""

    suffix = Path(filename).suffix.casefold()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_SUFFIXES))
        raise UploadValidationError(f"Use one of these video types: {allowed}.")
    if size_bytes <= 0:
        raise UploadValidationError("The uploaded video is empty.")
    max_bytes = max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise UploadValidationError(
            f"The video exceeds the {max_upload_mb} MB upload limit."
        )
    return suffix


def save_uploaded_video(filename: str, data: bytes) -> tuple[Path, str]:
    """Store one content-addressed upload and return its path and hash."""

    suffix = validate_upload(filename, len(data))
    digest = hashlib.sha256(data).hexdigest()
    upload_dir = TMP_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{digest}{suffix}"
    if not destination.exists():
        destination.write_bytes(data)
    return destination, digest


def serializable_state(state: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only objects and verify JSON serialization."""

    clean = {key: value for key, value in state.items() if key != "social_graph"}
    return json.loads(json.dumps(clean, default=str))


__all__ = [
    "ALLOWED_VIDEO_SUFFIXES",
    "UploadValidationError",
    "save_uploaded_video",
    "serializable_state",
    "validate_upload",
]
