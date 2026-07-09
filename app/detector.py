"""Person detector adapter for existing demo detectors and future NPU use."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import project_path
from vision.detection.motion_detector import MotionPersonDetector
from vision.detection.yolo_detector import OpenCVDNNYoloDetector
from vision.types import Detection, Frame


class PersonDetector:
    """Detect only people without training or downloading at runtime."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_path = str(config.get("model_path") or "").strip()
        self.model_config_path = str(config.get("model_config_path") or "").strip()
        self.name = "motion-person-detector"
        self.allow_motion_fallback = bool(config.get("allow_motion_fallback", False))
        self.warning = ""
        self._detector = self._create_detector()

    def detect(self, frame: Frame) -> List[Detection]:
        detections = self._detector.detect(frame)
        return [item for item in detections if int(item.class_id) == 0 or item.label == "person"]

    def _create_detector(self):
        min_area = int(self.config.get("motion_min_area") or 500)
        if not self.model_path:
            return self._fallback(min_area, "No semantic person model configured.")
        path = project_path(self.model_path)
        if not path.exists():
            return self._fallback(min_area, f"Configured model not found: {path}")
        try:
            if path.suffix.lower() == ".onnx":
                input_size = int(self.config.get("input_size") or 640)
                self.name = "opencv-dnn-yolo-onnx"
                return OpenCVDNNYoloDetector(
                    str(path),
                    input_size=(input_size, input_size),
                    confidence_threshold=float(self.config.get("confidence_threshold") or 0.35),
                    nms_threshold=float(self.config.get("nms_threshold") or 0.45),
                )
            if path.suffix.lower() == ".caffemodel":
                config_path = self._resolve_caffe_config(path)
                self.name = "opencv-dnn-mobilenetssd-caffe"
                return MobileNetSSDCaffePersonDetector(
                    model_path=path,
                    config_path=config_path,
                    confidence_threshold=float(self.config.get("confidence_threshold") or 0.45),
                    input_size=int(self.config.get("input_size") or 300),
                )
            if path.suffix.lower() == ".rknn":
                return self._fallback(min_area, "RKNN/NPU detector hook is reserved for the next step.")
            return self._fallback(min_area, f"Unsupported model format: {path}")
        except Exception as exc:
            return self._fallback(min_area, f"Could not load configured detector: {exc}")

    def _fallback(self, min_area: int, reason: str):
        if self.allow_motion_fallback:
            self.name = "motion-person-detector"
            self.warning = f"{reason} Using motion fallback."
            return MotionPersonDetector(min_area=min_area)
        self.name = "no-op-person-detector"
        self.warning = f"{reason} Motion fallback is disabled to avoid marking non-person moving objects."
        return NoOpPersonDetector()


    def _resolve_caffe_config(self, model_path: Path) -> Path:
        if self.model_config_path:
            config_path = project_path(self.model_config_path)
            if config_path.exists():
                return config_path
            raise FileNotFoundError(f"Caffe prototxt not found: {config_path}")
        candidates = [
            model_path.with_suffix(".prototxt"),
            model_path.parent / "MobileNetSSD_deploy.prototxt",
            model_path.parent / "deploy.prototxt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Caffe prototxt not found; set detection.model_config_path")


class MobileNetSSDCaffePersonDetector:
    """OpenCV DNN MobileNetSSD detector, keeping only VOC class 15: person."""

    PERSON_CLASS_ID = 15

    def __init__(self, model_path: Path, config_path: Path, confidence_threshold: float = 0.45, input_size: int = 300) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.net = cv2.dnn.readNetFromCaffe(str(config_path), str(model_path))

    def detect(self, frame: Frame) -> List[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,
            size=(self.input_size, self.input_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        detections: List[Detection] = []
        for row in output.reshape(-1, 7):
            confidence = float(row[2])
            class_id = int(row[1])
            if class_id != self.PERSON_CLASS_ID or confidence < self.confidence_threshold:
                continue
            x1 = int(max(0, min(width - 1, row[3] * width)))
            y1 = int(max(0, min(height - 1, row[4] * height)))
            x2 = int(max(0, min(width, row[5] * width)))
            y2 = int(max(0, min(height, row[6] * height)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection((x1, y1, x2, y2), confidence, 0, "person"))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

class NoOpPersonDetector:
    def detect(self, frame: Frame) -> List[Detection]:
        return []
