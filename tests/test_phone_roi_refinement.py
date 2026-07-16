from __future__ import annotations

from unittest import TestCase

import numpy as np

from app.detector import PersonDetector, _normalise_phone_roi_config
from vision.types import Detection


class PhoneROIRefinementTests(TestCase):
    def test_maps_phone_from_person_crop_and_aggregates_profile(self) -> None:
        fake = _FakeRKNNDetector()
        detector = _person_detector(fake, detect_every=1)

        detections = detector.detect_scene(np.zeros((720, 1280, 3), dtype=np.uint8))

        phones = [item for item in detections if item.label == "cell phone"]
        self.assertEqual(len(phones), 1)
        self.assertEqual(phones[0].bbox, (580.0, 196.0, 620.0, 276.0))
        self.assertEqual(fake.last_thresholds, {"cell phone": 0.16})
        self.assertEqual(detector.last_profile["inference_ms"], 17.0)
        self.assertEqual(detector.phone_roi_runs, 1)
        self.assertEqual(detector.phone_roi_hits, 1)

    def test_reuses_roi_result_between_low_frequency_runs(self) -> None:
        fake = _FakeRKNNDetector()
        detector = _person_detector(fake, detect_every=2)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        first = detector.detect_scene(frame)
        second = detector.detect_scene(frame)

        self.assertEqual(len([item for item in first if item.label == "cell phone"]), 1)
        self.assertEqual(len([item for item in second if item.label == "cell phone"]), 1)
        self.assertEqual(fake.roi_calls, 1)
        self.assertEqual(detector.phone_roi_cache_reuses, 1)
        self.assertEqual(detector.last_profile["inference_ms"], 9.0)

    def test_does_not_run_roi_without_person(self) -> None:
        fake = _FakeRKNNDetector(include_person=False)
        detector = _person_detector(fake, detect_every=1)

        detections = detector.detect_scene(np.zeros((720, 1280, 3), dtype=np.uint8))

        self.assertEqual(detections, [])
        self.assertEqual(fake.roi_calls, 0)
        self.assertEqual(detector.phone_roi_runs, 0)


class _FakeRKNNDetector:
    def __init__(self, include_person: bool = True) -> None:
        self.include_person = include_person
        self.last_profile = {}
        self.roi_calls = 0
        self.last_thresholds = None

    def detect(self, frame, *, class_confidence_thresholds=None):
        if class_confidence_thresholds is None:
            self.last_profile = {"preprocess_ms": 1.0, "inference_ms": 9.0, "postprocess_ms": 0.5}
            return [Detection((400, 100, 800, 700), 0.9, 0, "person")] if self.include_person else []
        self.roi_calls += 1
        self.last_thresholds = class_confidence_thresholds
        self.last_profile = {"preprocess_ms": 0.7, "inference_ms": 8.0, "postprocess_ms": 0.3}
        return [Detection((260, 120, 300, 200), 0.18, 67, "cell phone")]


def _person_detector(fake: _FakeRKNNDetector, detect_every: int) -> PersonDetector:
    detector = object.__new__(PersonDetector)
    detector._detector = fake
    detector.phone_roi_config = _normalise_phone_roi_config({
        "enabled": True,
        "confidence_threshold": 0.16,
        "detect_every_n_primary_frames": detect_every,
        "max_people": 1,
        "min_person_height_px": 160,
        "horizontal_expansion_ratio": 0.20,
        "top_expansion_ratio": 0.04,
        "upper_body_ratio": 0.90,
        "cache_primary_frames": 1,
        "max_phone_area_ratio": 0.25,
        "nms_threshold": 0.35,
        "containment_threshold": 0.70,
    })
    detector.phone_roi_enabled = True
    detector.phone_roi_error = ""
    detector.phone_roi_runs = 0
    detector.phone_roi_hits = 0
    detector.phone_roi_cache_reuses = 0
    detector.phone_roi_last_count = 0
    detector.phone_roi_last_inference_ms = 0.0
    detector._phone_roi_primary_calls = 0
    detector._phone_roi_cache_age = 0
    detector._phone_roi_cache = []
    detector._phone_label = "cell phone"
    detector.last_profile = {}
    return detector
