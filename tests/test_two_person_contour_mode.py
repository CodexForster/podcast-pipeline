from __future__ import annotations

import logging

import numpy as np
import pytest

from src.speaker_detection.face_detection import FaceDetection
from src.speaker_detection.two_person_contour_mode import TwoPersonContourProcessor


class StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        return list(self._detections)


def _cfg(**overrides):
    base = {
        "speaker_detection": {
            "detector_backend": "mediapipe",
            "detection_confidence": 0.5,
            "min_face_size": 8,
        },
        "two_person_contour_mode": {
            "enabled": True,
            "fail_on_less_than_two": True,
            "contour_thickness": 4,
            "contour_color": {"r": 255, "g": 255, "b": 255},
            "expand_x": 1.4,
            "expand_top": 0.35,
            "expand_bottom": 3.0,
            "min_area": 10,
        },
    }
    base["two_person_contour_mode"].update(overrides)
    return base


def test_selects_two_strongest_and_orders_left_to_right():
    detections = [
        FaceDetection((140, 20, 18, 20), 0.7),
        FaceDetection((20, 20, 20, 20), 0.9),
        FaceDetection((80, 20, 15, 15), 0.6),
    ]
    processor = TwoPersonContourProcessor(_cfg(), logging.getLogger("t"), detector=StubDetector(detections))

    _, result = processor.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))

    assert len(result.ordered_faces) == 2
    assert result.ordered_faces[0].bbox[0] < result.ordered_faces[1].bbox[0]
    assert result.ordered_faces[0].bbox == (20, 20, 20, 20)
    assert result.ordered_faces[1].bbox == (140, 20, 18, 20)


def test_fails_fast_when_less_than_two_detected():
    detections = [FaceDetection((30, 30, 20, 20), 0.95)]
    processor = TwoPersonContourProcessor(_cfg(fail_on_less_than_two=True), logging.getLogger("t"), detector=StubDetector(detections))

    with pytest.raises(RuntimeError, match="Expected 2 people"):
        processor.process_frame(np.zeros((200, 300, 3), dtype=np.uint8))


def test_warns_when_less_than_two_if_not_strict(caplog):
    detections = [FaceDetection((30, 30, 20, 20), 0.95)]
    processor = TwoPersonContourProcessor(_cfg(fail_on_less_than_two=False), logging.getLogger("t"), detector=StubDetector(detections))

    with caplog.at_level("WARNING"):
        _, result = processor.process_frame(np.zeros((200, 300, 3), dtype=np.uint8))

    assert "Expected 2 people" in caplog.text
    assert len(result.ordered_faces) == 1


def test_draws_two_contours_for_two_faces():
    detections = [
        FaceDetection((40, 30, 28, 28), 0.9),
        FaceDetection((150, 32, 30, 30), 0.88),
    ]
    processor = TwoPersonContourProcessor(_cfg(), logging.getLogger("t"), detector=StubDetector(detections))

    rendered, result = processor.process_frame(np.zeros((260, 360, 3), dtype=np.uint8))

    assert len(result.contours) == 2
    assert rendered.sum() > 0
