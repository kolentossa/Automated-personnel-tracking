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
        self.assertEqual(
            len([item for item in second if item.label == "cell phone"]), 1
        )
        self.assertEqual(fake.roi_calls, 1)
        self.assertEqual(detector.phone_roi_cache_reuses, 1)
        self.assertEqual(detector.last_profile["inference_ms"], 9.0)

    def test_reprojects_short_phone_memory_with_moving_person(self) -> None:
        fake = _FakeRKNNDetector(
            person_bboxes=[(400, 100, 800, 700), (500, 100, 900, 700)]
        )
        detector = _person_detector(fake, detect_every=2)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        detector.detect_scene(frame)
        second = detector.detect_scene(frame)

        phone = next(item for item in second if item.label == "cell phone")
        self.assertEqual(phone.bbox, (680.0, 196.0, 720.0, 276.0))
        self.assertAlmostEqual(phone.confidence, 0.162)

    def test_alternates_head_and_torso_crops_without_extra_inferences(self) -> None:
        fake = _FakeRKNNDetector()
        detector = _person_detector(fake, detect_every=1)
        detector.phone_roi_config["crop_modes"] = [
            {
                "name": "head_shoulders",
                "top_ratio": 0.0,
                "bottom_ratio": 0.52,
                "horizontal_expansion_ratio": 0.28,
                "top_expansion_ratio": 0.06,
            },
            {
                "name": "hands_torso",
                "top_ratio": 0.18,
                "bottom_ratio": 0.88,
                "horizontal_expansion_ratio": 0.22,
                "top_expansion_ratio": 0.0,
            },
        ]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        detector.detect_scene(frame)
        detector.detect_scene(frame)

        self.assertEqual(fake.roi_calls, 2)
        self.assertEqual(fake.roi_shapes, [(348, 624), (420, 576)])
        self.assertEqual(detector.phone_roi_last_crop_mode, "hands_torso")

    def test_uses_context_model_only_for_configured_crop(self) -> None:
        primary = _FakeRKNNDetector()
        context = _FakeContextDetector()
        detector = _person_detector(primary, detect_every=1)
        detector._phone_context_detector = context
        detector.phone_roi_config["crop_modes"] = _normalise_phone_roi_config(
            {
                "enabled": True,
                "crop_modes": [
                    {
                        "name": "head_shoulders",
                        "detector": "primary",
                        "confidence_threshold": 0.12,
                        "top_ratio": 0.0,
                        "bottom_ratio": 0.52,
                    },
                    {
                        "name": "hands_torso",
                        "detector": "primary",
                        "confidence_threshold": 0.20,
                        "top_ratio": 0.18,
                        "bottom_ratio": 0.88,
                    },
                    {
                        "name": "upper_body_context",
                        "detector": "context",
                        "confidence_threshold": 0.35,
                        "top_ratio": 0.0,
                        "bottom_ratio": 0.90,
                    },
                ],
            }
        )["crop_modes"]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        detector.detect_scene(frame)
        detector.detect_scene(frame)
        detector.detect_scene(frame)

        self.assertEqual(primary.roi_calls, 2)
        self.assertEqual(
            primary.threshold_history,
            [{"cell phone": 0.12}, {"cell phone": 0.20}],
        )
        self.assertEqual(context.roi_calls, 1)
        self.assertEqual(context.last_thresholds, {"cell phone": 0.35})
        self.assertEqual(detector.phone_roi_last_detector, "rknn-yolo11n")

    def test_weak_full_frame_phone_gets_roi_review_but_strong_phone_skips_it(
        self,
    ) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        weak_fake = _FakeRKNNDetector(primary_phone_confidence=0.20)
        weak_detector = _person_detector(weak_fake, detect_every=1)
        weak_detector.detect_scene(frame)
        self.assertEqual(weak_fake.roi_calls, 1)

        strong_fake = _FakeRKNNDetector(primary_phone_confidence=0.40)
        strong_detector = _person_detector(strong_fake, detect_every=1)
        strong_detector.detect_scene(frame)
        self.assertEqual(strong_fake.roi_calls, 0)

    def test_does_not_run_roi_without_person(self) -> None:
        fake = _FakeRKNNDetector(include_person=False)
        detector = _person_detector(fake, detect_every=1)

        detections = detector.detect_scene(np.zeros((720, 1280, 3), dtype=np.uint8))

        self.assertEqual(detections, [])
        self.assertEqual(fake.roi_calls, 0)
        self.assertEqual(detector.phone_roi_runs, 0)

    def test_reset_temporal_state_clears_cross_scene_phone_memory(self) -> None:
        detector = _person_detector(_FakeRKNNDetector(), detect_every=1)
        detector.detect_scene(np.zeros((720, 1280, 3), dtype=np.uint8))

        self.assertTrue(detector._phone_memory)
        detector.reset_temporal_state()

        self.assertEqual(detector._phone_memory, [])
        self.assertEqual(detector._phone_roi_cache, [])
        self.assertEqual(detector._phone_roi_primary_calls, 0)


class _FakeRKNNDetector:
    def __init__(
        self,
        include_person: bool = True,
        person_bboxes=None,
        primary_phone_confidence=None,
    ) -> None:
        self.include_person = include_person
        self.person_bboxes = list(person_bboxes or [(400, 100, 800, 700)])
        self.primary_phone_confidence = primary_phone_confidence
        self.last_profile = {}
        self.primary_calls = 0
        self.roi_calls = 0
        self.roi_shapes = []
        self.last_thresholds = None
        self.threshold_history = []

    def detect(self, frame, *, class_confidence_thresholds=None):
        if class_confidence_thresholds is None:
            self.last_profile = {
                "preprocess_ms": 1.0,
                "inference_ms": 9.0,
                "postprocess_ms": 0.5,
            }
            if not self.include_person:
                return []
            person_bbox = self.person_bboxes[
                min(self.primary_calls, len(self.person_bboxes) - 1)
            ]
            self.primary_calls += 1
            result = [Detection(person_bbox, 0.9, 0, "person")]
            if self.primary_phone_confidence is not None:
                result.append(
                    Detection(
                        (580, 196, 620, 276),
                        self.primary_phone_confidence,
                        67,
                        "cell phone",
                    )
                )
            return result
        self.roi_calls += 1
        self.roi_shapes.append(tuple(frame.shape[:2]))
        self.last_thresholds = class_confidence_thresholds
        self.threshold_history.append(class_confidence_thresholds)
        self.last_profile = {
            "preprocess_ms": 0.7,
            "inference_ms": 8.0,
            "postprocess_ms": 0.3,
        }
        return [Detection((260, 120, 300, 200), 0.18, 67, "cell phone")]


class _FakeContextDetector:
    name = "rknn-yolo11n"

    def __init__(self) -> None:
        self.last_profile = {}
        self.roi_calls = 0
        self.last_thresholds = None

    def detect(self, frame, *, class_confidence_thresholds=None):
        self.roi_calls += 1
        self.last_thresholds = class_confidence_thresholds
        self.last_profile = {
            "preprocess_ms": 0.8,
            "inference_ms": 8.5,
            "postprocess_ms": 0.4,
        }
        return [Detection((260, 120, 300, 200), 0.40, 67, "cell phone")]


def _person_detector(fake: _FakeRKNNDetector, detect_every: int) -> PersonDetector:
    detector = object.__new__(PersonDetector)
    detector._detector = fake
    detector.phone_roi_config = _normalise_phone_roi_config(
        {
            "enabled": True,
            "confidence_threshold": 0.16,
            "refine_below_confidence": 0.30,
            "detect_every_n_primary_frames": detect_every,
            "max_people": 1,
            "min_person_height_px": 160,
            "horizontal_expansion_ratio": 0.20,
            "top_expansion_ratio": 0.04,
            "upper_body_ratio": 0.90,
            "cache_primary_frames": 1,
            "cache_confidence_decay": 0.90,
            "cache_min_confidence": 0.10,
            "max_phone_area_ratio": 0.25,
            "nms_threshold": 0.35,
            "containment_threshold": 0.70,
        }
    )
    detector.phone_roi_enabled = True
    detector.phone_roi_error = ""
    detector.phone_roi_runs = 0
    detector.phone_roi_hits = 0
    detector.phone_roi_cache_reuses = 0
    detector.phone_roi_last_count = 0
    detector.phone_roi_last_inference_ms = 0.0
    detector.phone_roi_last_crop_mode = ""
    detector.phone_roi_last_detector = ""
    detector.phone_context_model_error = ""
    detector.phone_context_model_name = ""
    detector._phone_context_detector = None
    detector._phone_roi_primary_calls = 0
    detector._phone_roi_cache_age = 0
    detector._phone_roi_cache = []
    detector._phone_memory = []
    detector._phone_label = "cell phone"
    detector.last_profile = {}
    return detector
