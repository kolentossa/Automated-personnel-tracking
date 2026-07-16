"""Optional RKNN detector for cigarette, smoke, flame, lighter, and hand classes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from app.config import project_path
from app.detectors.rknn_yolo import RKNNYoloDetector
from vision.types import Detection, Frame

DIRECT_SMOKING_LABELS = {
    "cigarette",
    "cigar",
    "smoking",
    "person smoking",
    "hand with cigarette",
}


class BehaviorObjectDetector:
    """Load a custom behavior model only when explicitly enabled."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.required = bool(self.config.get("required", False))
        self.detect_every_n_frames = max(
            1, int(self.config.get("detect_every_n_frames") or 3)
        )
        self.model_path = str(self.config.get("model_path") or "")
        self.name = "behavior-model-disabled"
        self.available = False
        self.npu_enabled = False
        self.error = ""
        self.last_profile: Dict[str, float] = _empty_profile()
        self.last_result_is_fresh = False
        self._detector = None
        self._last_detections: List[Detection] = []
        self._has_run = False
        self._configured_labels = _configured_labels(self.config)
        if self.enabled:
            self._initialise()

    def detect(self, frame: Frame, frame_index: int) -> List[Detection]:
        if self._detector is None:
            self.last_result_is_fresh = False
            self.last_profile = _empty_profile()
            return []
        if frame_index % self.detect_every_n_frames != 0 and self._has_run:
            self.last_result_is_fresh = False
            self.last_profile = _empty_profile()
            return list(self._last_detections)
        try:
            detections = self._detector.detect(frame)
            self.last_result_is_fresh = True
            self.last_profile = dict(self._detector.last_profile)
            self._last_detections = detections
            self._has_run = True
            self.error = ""
            return list(detections)
        except Exception as exc:
            self.last_result_is_fresh = False
            self._has_run = True
            self.error = f"Behavior RKNN inference failed: {exc}"
            self.last_profile = _empty_profile()
            if self.required:
                raise RuntimeError(self.error) from exc
            return list(self._last_detections)

    def release(self) -> None:
        if self._detector is not None:
            self._detector.release()

    def snapshot(self) -> dict:
        return {
            "behavior_model_enabled": self.enabled,
            "behavior_model_required": self.required,
            "behavior_model_available": self.available,
            "behavior_model_loaded": self.available,
            "behavior_model": self.name,
            "behavior_model_path": self.model_path,
            "behavior_model_npu_enabled": self.npu_enabled,
            "smoking_detection_available": self.smoking_detection_available,
            "behavior_model_error": self.error,
            "behavior_model_detect_every_n_frames": self.detect_every_n_frames,
            "behavior_model_result_fresh": self.last_result_is_fresh,
        }

    @property
    def smoking_detection_available(self) -> bool:
        return bool(
            self.available
            and self.npu_enabled
            and self._configured_labels.intersection(DIRECT_SMOKING_LABELS)
        )

    def _initialise(self) -> None:
        path = project_path(self.model_path)
        if not path.exists():
            self.name = "behavior-model-unavailable"
            self.error = f"Behavior model not found: {path}"
            return
        expected_sha256 = str(self.config.get("expected_sha256") or "").strip().lower()
        if expected_sha256:
            actual_sha256 = _sha256(path)
            if actual_sha256 != expected_sha256:
                self.name = "behavior-model-invalid"
                self.error = f"Behavior model SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
                return
        try:
            self._detector = RKNNYoloDetector(
                path,
                model_family=str(self.config.get("model_family") or "yolov8"),
                input_size=int(self.config.get("input_size") or 640),
                confidence_threshold=float(
                    self.config.get("confidence_threshold") or 0.3
                ),
                nms_threshold=float(self.config.get("nms_threshold") or 0.45),
                class_filter=self.config.get("class_filter") or [],
                class_names=self.config.get("class_names"),
                class_confidence_thresholds=self.config.get(
                    "class_confidence_thresholds"
                ),
                box_filter=self.config.get("box_filter"),
                core_mask=self.config.get("core_mask") or "2",
            )
            self.name = self._detector.name
            self.available = True
            self.npu_enabled = True
            self.error = ""
        except Exception as exc:
            self.name = "behavior-model-unavailable"
            self.error = f"Could not load behavior RKNN model: {exc}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_profile() -> Dict[str, float]:
    return {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}


def _configured_labels(config: dict) -> set[str]:
    class_names = config.get("class_names") or []
    values = class_names.values() if isinstance(class_names, dict) else class_names
    available = {str(value).strip().lower() for value in values}
    selected = {
        str(value).strip().lower() for value in (config.get("class_filter") or [])
    }
    return available.intersection(selected) if selected else available
