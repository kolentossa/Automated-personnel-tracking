"""Person detector adapter for CPU detectors and RK3588 NPU YOLO."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from app.config import project_path
from app.detectors.rknn_yolo import RKNNYoloDetector
from vision.detection.motion_detector import MotionPersonDetector
from vision.detection.yolo_detector import OpenCVDNNYoloDetector
from vision.types import Detection, Frame


class PersonDetector:
    """Detect only people without training or downloading at runtime."""

    def __init__(self, config: Dict[str, Any], fallback_config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.fallback_config = dict(fallback_config or {})
        self.detector_type = str(self.config.get("type") or "").strip().lower()
        self.model_path = str(self.config.get("model_path") or "").strip()
        self.model_config_path = str(self.config.get("model_config_path") or "").strip()
        self.name = "no-op-person-detector"
        self.npu_enabled = False
        self.allow_motion_fallback = bool(self.config.get("allow_motion_fallback", False))
        self.fallback_to_cpu = bool(self.config.get("fallback_to_cpu", False))
        self.warning = ""
        self.last_profile: Dict[str, float] = _empty_profile()
        self._detector = self._create_detector()

    def detect(self, frame: Frame) -> List[Detection]:
        started = time.monotonic()
        detections = self._detector.detect(frame)
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
        profile = getattr(self._detector, "last_profile", None)
        if profile:
            self.last_profile = _normalise_profile(profile, elapsed_ms)
        else:
            self.last_profile = {
                "preprocess_ms": 0.0,
                "inference_ms": elapsed_ms,
                "postprocess_ms": 0.0,
            }
        return [item for item in detections if int(item.class_id) == 0 or item.label == "person"]

    def _create_detector(self):
        min_area = int(self.config.get("motion_min_area") or 500)
        if not self.model_path:
            return self._motion_or_noop(min_area, "No semantic person model configured.")

        path = project_path(self.model_path)
        if self._wants_rknn(path):
            return self._create_rknn_or_fallback(path, min_area)
        if not path.exists():
            return self._motion_or_noop(min_area, f"Configured model not found: {path}")
        return self._create_cpu_detector(path, self.config, min_area)

    def _wants_rknn(self, path: Path) -> bool:
        return self.detector_type in {"rknn-yolo", "rknn-yolov8", "rknn-yolov8n"} or path.suffix.lower() == ".rknn"

    def _create_rknn_or_fallback(self, path: Path, min_area: int):
        if not path.exists():
            return self._cpu_or_noop(min_area, f"Configured RKNN model not found: {path}")
        try:
            detector = RKNNYoloDetector(
                path,
                model_family=str(self.config.get("model_family") or "yolov8"),
                input_size=int(self.config.get("input_size") or 640),
                confidence_threshold=float(self.config.get("confidence_threshold") or 0.35),
                nms_threshold=float(self.config.get("nms_threshold") or 0.45),
                class_filter=self.config.get("class_filter") or ["person"],
            )
            self.name = detector.name
            self.npu_enabled = True
            self.warning = ""
            return detector
        except Exception as exc:
            return self._cpu_or_noop(min_area, f"Could not load RKNN detector: {exc}")

    def _cpu_or_noop(self, min_area: int, reason: str):
        if self.fallback_to_cpu:
            try:
                fallback_path_value = str(self.fallback_config.get("model_path") or "").strip()
                if fallback_path_value:
                    fallback_path = project_path(fallback_path_value)
                    detector = self._create_cpu_detector(fallback_path, self.fallback_config, min_area)
                    self.warning = f"{reason} Using CPU fallback."
                    self.npu_enabled = False
                    return detector
            except Exception as exc:
                reason = f"{reason} CPU fallback failed: {exc}"
        return self._motion_or_noop(min_area, reason)

    def _create_cpu_detector(self, path: Path, config: Dict[str, Any], min_area: int):
        if not path.exists():
            return self._motion_or_noop(min_area, f"Configured CPU model not found: {path}")
        try:
            if path.suffix.lower() == ".onnx":
                input_size = int(config.get("input_size") or 640)
                self.name = "opencv-dnn-yolo-onnx"
                self.npu_enabled = False
                return OpenCVDNNYoloDetector(
                    str(path),
                    input_size=(input_size, input_size),
                    confidence_threshold=float(config.get("confidence_threshold") or 0.35),
                    nms_threshold=float(config.get("nms_threshold") or 0.45),
                )
            if path.suffix.lower() == ".caffemodel":
                config_path = self._resolve_caffe_config(path, config)
                self.name = "opencv-dnn-mobilenetssd-caffe"
                self.npu_enabled = False
                return MobileNetSSDCaffePersonDetector(
                    model_path=path,
                    config_path=config_path,
                    confidence_threshold=float(config.get("confidence_threshold") or 0.45),
                    input_size=int(config.get("input_size") or 300),
                )
            return self._motion_or_noop(min_area, f"Unsupported model format: {path}")
        except Exception as exc:
            return self._motion_or_noop(min_area, f"Could not load configured detector: {exc}")

    def _motion_or_noop(self, min_area: int, reason: str):
        if self.allow_motion_fallback:
            self.name = "motion-person-detector"
            self.npu_enabled = False
            self.warning = f"{reason} Using motion fallback."
            return MotionPersonDetector(min_area=min_area)
        self.name = "no-op-person-detector"
        self.npu_enabled = False
        self.warning = f"{reason} Motion fallback is disabled to avoid marking non-person moving objects."
        return NoOpPersonDetector()

    def _resolve_caffe_config(self, model_path: Path, config: Dict[str, Any]) -> Path:
        config_path_value = str(config.get("model_config_path") or self.model_config_path or "").strip()
        if config_path_value:
            config_path = project_path(config_path_value)
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
        self.last_profile: Dict[str, float] = _empty_profile()

    def detect(self, frame: Frame) -> List[Detection]:
        started = time.monotonic()
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,
            size=(self.input_size, self.input_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )
        preprocessed = time.monotonic()
        self.net.setInput(blob)
        output = self.net.forward()
        inferred = time.monotonic()
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
        finished = time.monotonic()
        self.last_profile = {
            "preprocess_ms": _ms(preprocessed - started),
            "inference_ms": _ms(inferred - preprocessed),
            "postprocess_ms": _ms(finished - inferred),
        }
        return detections


class NoOpPersonDetector:
    def __init__(self) -> None:
        self.last_profile: Dict[str, float] = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}

    def detect(self, frame: Frame) -> List[Detection]:
        return []


def _empty_profile() -> Dict[str, float]:
    return {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}


def _normalise_profile(profile: Dict[str, Any], elapsed_ms: float) -> Dict[str, float]:
    result = _empty_profile()
    for key in result:
        result[key] = round(float(profile.get(key, 0.0) or 0.0), 1)
    if not any(result.values()):
        result["inference_ms"] = elapsed_ms
    return result


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)
