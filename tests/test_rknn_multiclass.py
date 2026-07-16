from __future__ import annotations

from unittest import TestCase

import numpy as np

from app.behavior.geometry import associate_targets
from app.detectors.rknn_yolo import COCO_CLASS_NAMES, RKNNYoloDetector, _preprocess_damoyolo
from vision.types import Detection, TrackedPerson


class RKNNMultiClassPostprocessTests(TestCase):
    def test_overlapping_person_and_phone_survive_class_aware_nms(self) -> None:
        detector = object.__new__(RKNNYoloDetector)
        detector.input_size = 640
        detector.confidence_threshold = 0.35
        detector.nms_threshold = 0.45
        detector.class_confidence_thresholds = {}
        detector.box_filter = {}
        detector.class_names = {index: label for index, label in enumerate(COCO_CLASS_NAMES)}
        detector.selected_class_ids = {0, 67}

        outputs = []
        for branch in range(3):
            boxes = np.zeros((1, 64, 1, 1), dtype=np.float32)
            classes = np.zeros((1, 80, 1, 1), dtype=np.float32)
            auxiliary = np.zeros((1, 1, 1, 1), dtype=np.float32)
            if branch == 0:
                classes[0, 0, 0, 0] = 0.9
                classes[0, 67, 0, 0] = 0.8
            outputs.extend([boxes, classes, auxiliary])

        detections = detector._postprocess_yolov8_heads(outputs, (640, 640), 1.0, (0.0, 0.0))
        self.assertIsNotNone(detections)
        self.assertEqual({item.label for item in detections}, {"person", "cell phone"})

    def test_class_specific_threshold_recovers_weak_phone_without_weak_person(self) -> None:
        detector = object.__new__(RKNNYoloDetector)
        detector.input_size = 640
        detector.confidence_threshold = 0.35
        detector.class_confidence_thresholds = {67: 0.20}
        detector.nms_threshold = 0.45
        detector.box_filter = {}
        detector.class_names = {index: label for index, label in enumerate(COCO_CLASS_NAMES)}
        detector.selected_class_ids = {0, 67}

        outputs = []
        for branch in range(3):
            boxes = np.zeros((1, 64, 1, 1), dtype=np.float32)
            classes = np.zeros((1, 80, 1, 1), dtype=np.float32)
            auxiliary = np.zeros((1, 1, 1, 1), dtype=np.float32)
            if branch == 0:
                classes[0, 0, 0, 0] = 0.34
                classes[0, 67, 0, 0] = 0.22
            outputs.extend([boxes, classes, auxiliary])

        detections = detector._postprocess_yolov8_heads(outputs, (640, 640), 1.0, (0.0, 0.0))
        self.assertEqual([item.label for item in detections], ["cell phone"])

    def test_damoyolo_decodes_strict_pair_and_direct_resize(self) -> None:
        detector = _damo_detector()
        scores = np.zeros((1, 8400, 2), dtype=np.float32)
        boxes = np.zeros((1, 8400, 4), dtype=np.float32)
        scores[0, 17, 0] = 0.9
        boxes[0, 17] = [64.0, 128.0, 320.0, 512.0]
        detections = detector._postprocess_damoyolo([scores, boxes], (320, 640))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "cigarette")
        self.assertTrue(np.allclose(detections[0].bbox, (64.0, 64.0, 320.0, 256.0)))

        frame = np.full((2, 4, 3), (1, 2, 3), dtype=np.uint8)
        prepared = _preprocess_damoyolo(frame, 4)
        self.assertEqual(prepared.shape, (1, 4, 4, 3))
        self.assertEqual(prepared[0, 0, 0].tolist(), [3, 2, 1])

    def test_damoyolo_output_shape_mismatch_is_explicit(self) -> None:
        detector = _damo_detector()
        scores = np.zeros((1, 8400, 1), dtype=np.float32)
        boxes = np.zeros((1, 8400, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "scores shape mismatch"):
            detector._postprocess_damoyolo([scores, boxes], (640, 640))
        with self.assertRaisesRegex(ValueError, "output count mismatch"):
            detector._postprocess_damoyolo([scores], (640, 640))

    def test_damoyolo_class_map_mismatch_is_explicit(self) -> None:
        detector = _damo_detector()
        detector.class_names = {0: "cigarette"}
        scores = np.zeros((1, 8400, 2), dtype=np.float32)
        boxes = np.zeros((1, 8400, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "scores shape mismatch"):
            detector._postprocess_damoyolo([scores, boxes], (640, 640))

    def test_damoyolo_filters_elongated_false_positive(self) -> None:
        detector = _damo_detector()
        detector.box_filter = {"max_aspect_ratio": 4.0}
        scores = np.zeros((1, 8400, 2), dtype=np.float32)
        boxes = np.zeros((1, 8400, 4), dtype=np.float32)
        scores[0, 12, 0] = 0.92
        boxes[0, 12] = [40.0, 100.0, 600.0, 130.0]

        self.assertEqual(detector._postprocess_damoyolo([scores, boxes], (640, 640)), [])

    def test_damoyolo_suppresses_nested_duplicate_boxes(self) -> None:
        detector = _damo_detector()
        detector.nms_threshold = 0.35
        detector.box_filter = {"duplicate_containment_threshold": 0.70}
        scores = np.zeros((1, 8400, 2), dtype=np.float32)
        boxes = np.zeros((1, 8400, 4), dtype=np.float32)
        scores[0, 12, 0] = 0.92
        scores[0, 13, 0] = 0.81
        boxes[0, 12] = [100.0, 100.0, 220.0, 220.0]
        boxes[0, 13] = [130.0, 130.0, 170.0, 170.0]

        detections = detector._postprocess_damoyolo([scores, boxes], (640, 640))
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.92, places=5)


class TargetAssociationTests(TestCase):
    def test_phone_is_assigned_to_nearest_person_only(self) -> None:
        tracks = [
            TrackedPerson(1, (0, 0, 200, 400), 0.9),
            TrackedPerson(2, (150, 0, 350, 400), 0.9),
        ]
        phone = Detection((250, 160, 280, 210), 0.8, 67, "cell phone")
        assigned = associate_targets(tracks, [phone], ["cell phone"], expansion_ratio=0.2)
        self.assertEqual(assigned[1], [])
        self.assertEqual(assigned[2], [phone])


def _damo_detector() -> RKNNYoloDetector:
    detector = object.__new__(RKNNYoloDetector)
    detector.input_size = 640
    detector.confidence_threshold = 0.35
    detector.nms_threshold = 0.70
    detector.class_confidence_thresholds = {}
    detector.box_filter = {}
    detector.class_names = {0: "cigarette", 1: "__unused__"}
    detector.selected_class_ids = {0}
    return detector
