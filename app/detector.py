"""Person detector adapter for existing demo detectors and future NPU use."""

from __future__ import annotations

from typing import Any, Dict, List

from app.config import project_path
from vision.detection.motion_detector import MotionPersonDetector
from vision.detection.yolo_detector import OpenCVDNNYoloDetector
from vision.types import Detection, Frame


class PersonDetector:
    """Detect only the person class without training or downloading models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_path = str(config.get("model_path") or "").strip()
        self.name = "motion-person-detector"
        self.warning = ""
        self._detector = self._create_detector()

    def detect(self, frame: Frame) -> List[Detection]:
        detections = self._detector.detect(frame)
        return [item for item in detections if int(item.class_id) == 0 or item.label == "person"]

    def _create_detector(self):
        min_area = int(self.config.get("motion_min_area") or 500)
        if not self.model_path:
            return MotionPersonDetector(min_area=min_area)
        path = project_path(self.model_path)
        if not path.exists():
            self.warning = f"Configured model not found, using motion detector: {path}"
            return MotionPersonDetector(min_area=min_area)
        if path.suffix.lower() == ".onnx":
            input_size = int(self.config.get("input_size") or 640)
            self.name = "opencv-dnn-yolo-onnx"
            return OpenCVDNNYoloDetector(
                str(path),
                input_size=(input_size, input_size),
                confidence_threshold=float(self.config.get("confidence_threshold") or 0.35),
                nms_threshold=float(self.config.get("nms_threshold") or 0.45),
            )
        if path.suffix.lower() == ".rknn":
            self.warning = "RKNN/NPU detector hook is reserved; falling back to motion detector for this MVP."
            return MotionPersonDetector(min_area=min_area)
        self.warning = f"Unsupported model format, using motion detector: {path}"
        return MotionPersonDetector(min_area=min_area)