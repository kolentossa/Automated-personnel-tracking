"""YOLO person detector using OpenCV DNN and local model files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision.detection.detector import Detector
from vision.types import Detection, Frame


class OpenCVDNNYoloDetector(Detector):
    """Run a local YOLO ONNX model and keep only the person class.

    The detector never downloads weights. Put exported models such as
    yolov8n.onnx under models/ and pass that path to the backend or scripts.
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(str(self.model_path))

    def detect(self, frame: Frame) -> List[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        rows = self._normalise_output(outputs)
        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        x_factor = width / float(self.input_size[0])
        y_factor = height / float(self.input_size[1])

        for row in rows:
            parsed = self._parse_person_row(row)
            if parsed is None:
                continue
            center_x, center_y, box_w, box_h, confidence = parsed
            if confidence < self.confidence_threshold:
                continue
            x = int((center_x - box_w / 2.0) * x_factor)
            y = int((center_y - box_h / 2.0) * y_factor)
            w = int(box_w * x_factor)
            h = int(box_h * y_factor)
            boxes.append((x, y, max(1, w), max(1, h)))
            scores.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
        detections: List[Detection] = []
        for index in self._flatten_indices(indices):
            x, y, w, h = boxes[index]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x + w)
            y2 = min(height - 1, y + h)
            detections.append(Detection((x1, y1, x2, y2), scores[index], 0, "person"))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    @staticmethod
    def _normalise_output(outputs: object) -> np.ndarray:
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs
        rows = np.squeeze(output)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim == 2 and rows.shape[0] < rows.shape[1] and rows.shape[0] in (84, 85):
            rows = rows.T
        return rows

    @staticmethod
    def _parse_person_row(row: Sequence[float]) -> Optional[Tuple[float, float, float, float, float]]:
        if len(row) < 6:
            return None
        center_x, center_y, box_w, box_h = [float(v) for v in row[:4]]
        if len(row) == 6:
            confidence = float(row[4]) if int(row[5]) == 0 else 0.0
        elif len(row) >= 85:
            objectness = float(row[4])
            confidence = objectness * float(row[5])
        else:
            confidence = float(row[4])
        return center_x, center_y, box_w, box_h, confidence

    @staticmethod
    def _flatten_indices(indices: object) -> List[int]:
        if indices is None:
            return []
        array = np.array(indices).reshape(-1)
        return [int(item) for item in array]
