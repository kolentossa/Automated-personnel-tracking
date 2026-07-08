"""Local no-model detector used for the generated video demo."""

from __future__ import annotations

from typing import List

import cv2

from vision.detection.detector import Detector
from vision.types import Detection, Frame


class MotionPersonDetector(Detector):
    """Detect moving person-like blobs without cloud calls or model downloads.

    This detector is intentionally simple. It makes the first demo runnable on a
    board with no camera and no model file. Production deployments should place a
    YOLO ONNX or RKNN model in models/ and select the corresponding detector.
    """

    def __init__(self, min_area: int = 500, max_area_ratio: float = 0.35) -> None:
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self._subtracter = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=36,
            detectShadows=False,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: Frame) -> List[Detection]:
        height, width = frame.shape[:2]
        frame_area = float(width * height)
        mask = self._subtracter.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > frame_area * self.max_area_ratio:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0 or w <= 0:
                continue
            aspect = h / float(w)
            if aspect < 0.6 or aspect > 5.0:
                continue
            confidence = min(0.95, max(0.2, area / max(1.0, float(w * h))))
            detections.append(Detection((x, y, x + w, y + h), confidence, 0, "person"))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections
