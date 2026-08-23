"""Tests for YOLO result aggregation without loading model weights."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from video import detector
from video.detector import DetectionError, detect


def test_detect_empty_frames_does_not_load_model(monkeypatch) -> None:
    monkeypatch.setattr(
        detector,
        "get_model",
        lambda: pytest.fail("model should not load for empty frames"),
    )

    stats = detect([])

    assert stats.counts == {}
    assert stats.unique_classes == 0
    assert stats.avg_objects_per_frame == 0.0
    assert stats.dominant_class is None


def test_detect_aggregates_classes_across_frames(monkeypatch) -> None:
    results = [
        SimpleNamespace(
            names={0: "person", 1: "phone"},
            boxes=SimpleNamespace(cls=np.array([0, 1])),
        ),
        SimpleNamespace(
            names={0: "person", 1: "phone"},
            boxes=SimpleNamespace(cls=np.array([0])),
        ),
        SimpleNamespace(names={0: "person", 1: "phone"}, boxes=None),
    ]
    observed: dict = {}

    class FakeModel:
        names = {0: "person", 1: "phone"}

        def predict(self, **kwargs):
            observed.update(kwargs)
            return results

    monkeypatch.setattr(detector, "get_model", lambda: FakeModel())
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

    stats = detect(frames, conf=0.42)

    assert stats.counts == {"person": 2, "phone": 1}
    assert stats.unique_classes == 2
    assert stats.avg_objects_per_frame == pytest.approx(1.0)
    assert stats.dominant_class == "person"
    assert observed["conf"] == 0.42
    assert observed["verbose"] is False
    assert len(observed["source"]) == 3


def test_detect_uses_alphabetical_tie_break(monkeypatch) -> None:
    result = SimpleNamespace(
        names={0: "zebra", 1: "apple"},
        boxes=SimpleNamespace(cls=np.array([0, 1])),
    )
    model = SimpleNamespace(
        names=result.names,
        predict=lambda **_kwargs: [result],
    )
    monkeypatch.setattr(detector, "get_model", lambda: model)

    stats = detect([np.zeros((2, 2, 3), dtype=np.uint8)])

    assert stats.dominant_class == "apple"


def test_detect_wraps_inference_failure(monkeypatch) -> None:
    class BrokenModel:
        def predict(self, **_kwargs):
            raise RuntimeError("backend failure")

    monkeypatch.setattr(detector, "get_model", lambda: BrokenModel())

    with pytest.raises(DetectionError, match="inference failed"):
        detect([np.zeros((2, 2, 3), dtype=np.uint8)])


def test_get_model_is_singleton(monkeypatch) -> None:
    loaded: list[object] = []

    def fake_load():
        model = object()
        loaded.append(model)
        return model

    monkeypatch.setattr(detector, "_MODEL", None)
    monkeypatch.setattr(detector, "_load_model", fake_load)

    assert detector.get_model() is detector.get_model()
    assert len(loaded) == 1

