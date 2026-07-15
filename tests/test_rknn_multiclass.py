from __future__ import annotations

from unittest import TestCase

import numpy as np

from app.behavior.geometry import associate_targets
from app.detectors.rknn_yolo import COCO_CLASS_NAMES, RKNNYoloDetector
from vision.types import Detection, TrackedPerson


class RKNNMultiClassPostprocessTests(TestCase):
    def test_overlapping_person_and_phone_survive_class_aware_nms(self) -> None:
        detector = object.__new__(RKNNYoloDetector)
        detector.input_size = 640
        detector.confidence_threshold = 0.35
        detector.nms_threshold = 0.45
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
