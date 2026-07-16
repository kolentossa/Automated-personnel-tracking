"""Person detector adapter for CPU detectors and RK3588 NPU YOLO."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from app.config import project_path
from app.detectors.rknn_yolo import RKNNYoloDetector
from vision.detection.motion_detector import MotionPersonDetector
from vision.detection.yolo_detector import OpenCVDNNYoloDetector
from vision.types import BBox, Detection, Frame

PHONE_LABELS = {"cell phone", "phone", "mobile phone"}


@dataclass(frozen=True)
class _PhoneMemory:
    """A phone box stored relative to its associated person."""

    person_bbox: BBox
    relative_bbox: BBox
    confidence: float
    class_id: int
    label: str


class PersonDetector:
    """Detect only people without training or downloading at runtime."""

    def __init__(
        self, config: Dict[str, Any], fallback_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.config = dict(config or {})
        self.fallback_config = dict(fallback_config or {})
        self.detector_type = str(self.config.get("type") or "").strip().lower()
        self.model_path = str(self.config.get("model_path") or "").strip()
        self.model_config_path = str(self.config.get("model_config_path") or "").strip()
        self.name = "no-op-person-detector"
        self.npu_enabled = False
        self.allow_motion_fallback = bool(
            self.config.get("allow_motion_fallback", False)
        )
        self.fallback_to_cpu = bool(self.config.get("fallback_to_cpu", False))
        self.warning = ""
        self.last_profile: Dict[str, float] = _empty_profile()
        self.phone_roi_config = _normalise_phone_roi_config(
            self.config.get("phone_roi_refinement")
        )
        self.phone_roi_enabled = False
        self.phone_roi_error = ""
        self.phone_roi_runs = 0
        self.phone_roi_hits = 0
        self.phone_roi_cache_reuses = 0
        self.phone_roi_last_count = 0
        self.phone_roi_last_inference_ms = 0.0
        self.phone_roi_last_crop_mode = ""
        self.phone_roi_last_detector = ""
        self.phone_context_model_error = ""
        self.phone_context_model_name = ""
        self._phone_roi_primary_calls = 0
        self._phone_roi_cache_age = 0
        self._phone_roi_cache: List[Detection] = []
        self._phone_memory: List[_PhoneMemory] = []
        self._detector = self._create_detector()
        self.output_labels = self._discover_output_labels()
        self._phone_label = next(
            (label for label in PHONE_LABELS if label in self.output_labels),
            "cell phone",
        )
        self.phone_roi_enabled = bool(
            self.phone_roi_config["enabled"]
            and isinstance(self._detector, RKNNYoloDetector)
            and self._phone_label in self.output_labels
        )
        self._phone_context_detector = self._create_phone_context_detector()

    def detect(self, frame: Frame) -> List[Detection]:
        return [
            item
            for item in self.detect_scene(frame)
            if int(item.class_id) == 0 or item.label == "person"
        ]

    def detect_scene(self, frame: Frame) -> List[Detection]:
        """Return every configured semantic class from one inference pass."""

        started = time.monotonic()
        detections = self._detector.detect(frame)
        base_elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
        base_profile = _normalise_profile(
            getattr(self._detector, "last_profile", None) or {}, base_elapsed_ms
        )
        detections, roi_profile = self._refine_phone_detections(frame, detections)
        self.last_profile = {
            key: round(base_profile[key] + roi_profile[key], 1)
            for key in _empty_profile()
        }
        return detections

    def release(self) -> None:
        for detector in (self._detector, self._phone_context_detector):
            release = getattr(detector, "release", None)
            if callable(release):
                release()

    def supports_label(self, label: str) -> bool:
        return str(label).strip().lower() in self.output_labels

    def reset_temporal_state(self) -> None:
        """Clear phone memory when a caller switches to an unrelated scene."""

        self._clear_phone_roi_cache()
        self._phone_roi_primary_calls = 0

    def snapshot(self) -> dict:
        return {
            "phone_roi_refinement_enabled": self.phone_roi_enabled,
            "phone_roi_refinement_configured": bool(self.phone_roi_config["enabled"]),
            "phone_roi_refinement_runs": self.phone_roi_runs,
            "phone_roi_refinement_hits": self.phone_roi_hits,
            "phone_roi_refinement_cache_reuses": self.phone_roi_cache_reuses,
            "phone_roi_refinement_last_count": self.phone_roi_last_count,
            "phone_roi_refinement_inference_ms": self.phone_roi_last_inference_ms,
            "phone_roi_refinement_last_crop_mode": self.phone_roi_last_crop_mode,
            "phone_roi_refinement_last_detector": self.phone_roi_last_detector,
            "phone_context_model_configured": bool(
                self.phone_roi_config["context_model"]["enabled"]
            ),
            "phone_context_model_enabled": self._phone_context_detector is not None,
            "phone_context_model_name": self.phone_context_model_name,
            "phone_context_model_error": self.phone_context_model_error,
            "phone_temporal_memory_count": len(self._phone_memory),
            "phone_temporal_memory_age": self._phone_roi_cache_age,
            "phone_roi_refinement_error": self.phone_roi_error,
        }

    def _refine_phone_detections(
        self, frame: Frame, detections: List[Detection]
    ) -> tuple[List[Detection], Dict[str, float]]:
        empty_profile = _empty_profile()
        if not self.phone_roi_enabled:
            return detections, empty_profile

        people = [
            item
            for item in detections
            if _label(item) == "person" or int(item.class_id) == 0
        ]
        if not people:
            self._clear_phone_roi_cache()
            return detections, empty_profile
        if self._phone_memory:
            self._phone_roi_cache_age += 1
        if self._phone_roi_cache_age > int(
            self.phone_roi_config["cache_primary_frames"]
        ):
            self._clear_phone_roi_cache()
        cached = self._valid_cached_phones(people)
        phones = [item for item in detections if _label(item) in PHONE_LABELS]
        skip_threshold = float(self.phone_roi_config["refine_below_confidence"])
        targets = [
            person
            for person in people
            if not any(
                phone.confidence >= skip_threshold
                and _center_inside(phone.bbox, _expand_box(person.bbox, 0.08))
                for phone in phones
            )
        ]
        if not targets:
            self._remember_phones(phones, people)
            return detections, empty_profile

        targets.sort(key=lambda item: _box_area(item.bbox), reverse=True)
        self._phone_roi_primary_calls += 1
        interval = int(self.phone_roi_config["detect_every_n_primary_frames"])
        should_run = (self._phone_roi_primary_calls - 1) % interval == 0
        if not should_run:
            if cached:
                self.phone_roi_cache_reuses += 1
                return self._merge_phone_detections(detections, cached), empty_profile
            return detections, empty_profile

        self.phone_roi_runs += 1
        self.phone_roi_last_count = 0
        self.phone_roi_last_inference_ms = 0.0
        roi_profile = _empty_profile()
        refined: List[Detection] = []
        height, width = frame.shape[:2]
        crop_modes = self.phone_roi_config["crop_modes"]
        crop_mode = (
            crop_modes[(self._phone_roi_primary_calls - 1) % len(crop_modes)]
            if crop_modes
            else None
        )
        self.phone_roi_last_crop_mode = str(
            (crop_mode or {}).get("name") or "upper_body"
        )
        requested_detector = str((crop_mode or {}).get("detector") or "primary")
        roi_detector = self._detector
        if requested_detector == "context" and self._phone_context_detector is not None:
            roi_detector = self._phone_context_detector
        self.phone_roi_last_detector = str(
            getattr(roi_detector, "name", requested_detector)
        )
        mode_threshold = (crop_mode or {}).get("confidence_threshold")
        roi_threshold = float(
            self.phone_roi_config["confidence_threshold"]
            if mode_threshold is None
            else mode_threshold
        )
        try:
            for person in targets[: int(self.phone_roi_config["max_people"])]:
                if person.bbox[3] - person.bbox[1] < float(
                    self.phone_roi_config["min_person_height_px"]
                ):
                    continue
                crop_box = _phone_roi_crop(
                    person.bbox, width, height, self.phone_roi_config, crop_mode
                )
                x1, y1, x2, y2 = crop_box
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_detections = roi_detector.detect(
                    crop,
                    class_confidence_thresholds={self._phone_label: roi_threshold},
                )
                current_profile = _normalise_profile(
                    getattr(roi_detector, "last_profile", None) or {}, 0.0
                )
                for key in roi_profile:
                    roi_profile[key] += current_profile[key]
                for item in crop_detections:
                    if _label(item) not in PHONE_LABELS:
                        continue
                    mapped = Detection(
                        (
                            max(0.0, min(float(width), item.bbox[0] + x1)),
                            max(0.0, min(float(height), item.bbox[1] + y1)),
                            max(0.0, min(float(width), item.bbox[2] + x1)),
                            max(0.0, min(float(height), item.bbox[3] + y1)),
                        ),
                        item.confidence,
                        item.class_id,
                        item.label,
                    )
                    area_ratio = _box_area(mapped.bbox) / max(
                        1.0, _box_area(person.bbox)
                    )
                    if area_ratio <= float(
                        self.phone_roi_config["max_phone_area_ratio"]
                    ):
                        refined.append(mapped)
            self.phone_roi_error = ""
        except Exception as exc:
            self.phone_roi_error = f"Phone ROI inference failed: {exc}"
            self._clear_phone_roi_cache()
            return detections, roi_profile

        refined = _deduplicate_detections(
            refined,
            float(self.phone_roi_config["nms_threshold"]),
            float(self.phone_roi_config["containment_threshold"]),
        )
        self.phone_roi_last_count = len(refined)
        self.phone_roi_last_inference_ms = round(roi_profile["inference_ms"], 1)
        if refined:
            self.phone_roi_hits += 1
        observed = _deduplicate_detections(
            phones + refined,
            float(self.phone_roi_config["nms_threshold"]),
            float(self.phone_roi_config["containment_threshold"]),
        )
        if observed:
            self._remember_phones(observed, people)
        elif cached:
            self.phone_roi_cache_reuses += 1
        return self._merge_phone_detections(detections, refined + cached), roi_profile

    def _valid_cached_phones(self, people: List[Detection]) -> List[Detection]:
        if not self._phone_memory:
            return []
        decay = float(self.phone_roi_config["cache_confidence_decay"])
        minimum = float(self.phone_roi_config["cache_min_confidence"])
        projected: List[Detection] = []
        refreshed: List[_PhoneMemory] = []
        for memory in self._phone_memory:
            person = _match_person(memory.person_bbox, people)
            if person is None:
                continue
            bbox = _relative_to_absolute(memory.relative_bbox, person.bbox)
            confidence = memory.confidence * (decay**self._phone_roi_cache_age)
            if confidence < minimum:
                continue
            phone = Detection(bbox, confidence, memory.class_id, memory.label)
            projected.append(phone)
            refreshed.append(
                _PhoneMemory(
                    person.bbox,
                    memory.relative_bbox,
                    memory.confidence,
                    memory.class_id,
                    memory.label,
                )
            )
        self._phone_memory = refreshed
        self._phone_roi_cache = projected
        return projected

    def _remember_phones(
        self, phones: List[Detection], people: List[Detection]
    ) -> None:
        memories: List[_PhoneMemory] = []
        for phone in phones:
            person = _person_containing_phone(phone.bbox, people)
            if person is None:
                continue
            memories.append(
                _PhoneMemory(
                    person.bbox,
                    _absolute_to_relative(phone.bbox, person.bbox),
                    float(phone.confidence),
                    int(phone.class_id),
                    str(phone.label),
                )
            )
        self._phone_memory = memories
        self._phone_roi_cache = list(phones) if memories else []
        self._phone_roi_cache_age = 0

    def _merge_phone_detections(
        self, detections: List[Detection], additions: List[Detection]
    ) -> List[Detection]:
        non_phones = [item for item in detections if _label(item) not in PHONE_LABELS]
        phones = [item for item in detections if _label(item) in PHONE_LABELS]
        phones.extend(additions)
        phones = _deduplicate_detections(
            phones,
            float(self.phone_roi_config["nms_threshold"]),
            float(self.phone_roi_config["containment_threshold"]),
        )
        return non_phones + phones

    def _clear_phone_roi_cache(self) -> None:
        self._phone_roi_cache = []
        self._phone_memory = []
        self._phone_roi_cache_age = 0
        self.phone_roi_last_count = 0

    def _create_detector(self):
        min_area = int(self.config.get("motion_min_area") or 500)
        if not self.model_path:
            return self._motion_or_noop(
                min_area, "No semantic person model configured."
            )

        path = project_path(self.model_path)
        if self._wants_rknn(path):
            hash_error = self._validate_model_hash(path)
            if hash_error:
                return self._cpu_or_noop(min_area, hash_error)
            return self._create_rknn_or_fallback(path, min_area)
        if not path.exists():
            return self._motion_or_noop(min_area, f"Configured model not found: {path}")
        return self._create_cpu_detector(path, self.config, min_area)

    def _create_phone_context_detector(self) -> Optional[RKNNYoloDetector]:
        config = self.phone_roi_config["context_model"]
        if not self.phone_roi_enabled or not config["enabled"]:
            return None
        path = project_path(str(config["model_path"]))
        error = self._validate_model_hash(path, config)
        if error:
            self.phone_context_model_error = error
            return None
        if not path.exists():
            self.phone_context_model_error = (
                f"Phone context RKNN model not found: {path}"
            )
            return None
        try:
            detector = RKNNYoloDetector(
                path,
                model_family=str(config.get("model_family") or "yolo11"),
                input_size=int(config.get("input_size") or 640),
                confidence_threshold=float(config.get("confidence_threshold") or 0.35),
                nms_threshold=float(config.get("nms_threshold") or 0.45),
                class_filter=config.get("class_filter") or ["cell phone"],
                class_names=config.get("class_names"),
                class_confidence_thresholds=config.get("class_confidence_thresholds"),
                core_mask=config.get("core_mask") or "0_1_2",
            )
        except Exception as exc:
            self.phone_context_model_error = (
                f"Could not load phone context RKNN model: {exc}"
            )
            return None
        self.phone_context_model_name = detector.name
        self.phone_context_model_error = ""
        return detector

    def _validate_model_hash(
        self, path: Path, config: Optional[Dict[str, Any]] = None
    ) -> str:
        selected_config = config or self.config
        expected = str(selected_config.get("expected_sha256") or "").strip().lower()
        if not expected or not path.exists():
            return ""
        actual = _sha256(path)
        if actual == expected:
            return ""
        return f"Configured model SHA-256 mismatch: expected {expected}, got {actual}"

    def _discover_output_labels(self) -> set[str]:
        selected = getattr(self._detector, "selected_class_ids", None)
        class_names = getattr(self._detector, "class_names", None)
        if selected is not None and isinstance(class_names, dict):
            return {
                str(class_names[class_id]).strip().lower()
                for class_id in selected
                if class_id in class_names
            }
        if self.name == "no-op-person-detector":
            return set()
        return {"person"}

    def _wants_rknn(self, path: Path) -> bool:
        return (
            self.detector_type in {"rknn-yolo", "rknn-yolov8", "rknn-yolov8n"}
            or path.suffix.lower() == ".rknn"
        )

    def _create_rknn_or_fallback(self, path: Path, min_area: int):
        if not path.exists():
            return self._cpu_or_noop(
                min_area, f"Configured RKNN model not found: {path}"
            )
        try:
            detector = RKNNYoloDetector(
                path,
                model_family=str(self.config.get("model_family") or "yolov8"),
                input_size=int(self.config.get("input_size") or 640),
                confidence_threshold=float(
                    self.config.get("confidence_threshold") or 0.35
                ),
                nms_threshold=float(self.config.get("nms_threshold") or 0.45),
                class_filter=self.config.get("class_filter") or ["person"],
                class_names=self.config.get("class_names"),
                class_confidence_thresholds=self.config.get(
                    "class_confidence_thresholds"
                ),
                box_filter=self.config.get("box_filter"),
                core_mask=self.config.get("core_mask") or "0_1_2",
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
                fallback_path_value = str(
                    self.fallback_config.get("model_path") or ""
                ).strip()
                if fallback_path_value:
                    fallback_path = project_path(fallback_path_value)
                    detector = self._create_cpu_detector(
                        fallback_path, self.fallback_config, min_area
                    )
                    self.warning = f"{reason} Using CPU fallback."
                    self.npu_enabled = False
                    return detector
            except Exception as exc:
                reason = f"{reason} CPU fallback failed: {exc}"
        return self._motion_or_noop(min_area, reason)

    def _create_cpu_detector(self, path: Path, config: Dict[str, Any], min_area: int):
        if not path.exists():
            return self._motion_or_noop(
                min_area, f"Configured CPU model not found: {path}"
            )
        try:
            if path.suffix.lower() == ".onnx":
                input_size = int(config.get("input_size") or 640)
                self.name = "opencv-dnn-yolo-onnx"
                self.npu_enabled = False
                return OpenCVDNNYoloDetector(
                    str(path),
                    input_size=(input_size, input_size),
                    confidence_threshold=float(
                        config.get("confidence_threshold") or 0.35
                    ),
                    nms_threshold=float(config.get("nms_threshold") or 0.45),
                )
            if path.suffix.lower() == ".caffemodel":
                config_path = self._resolve_caffe_config(path, config)
                self.name = "opencv-dnn-mobilenetssd-caffe"
                self.npu_enabled = False
                return MobileNetSSDCaffePersonDetector(
                    model_path=path,
                    config_path=config_path,
                    confidence_threshold=float(
                        config.get("confidence_threshold") or 0.45
                    ),
                    input_size=int(config.get("input_size") or 300),
                )
            return self._motion_or_noop(min_area, f"Unsupported model format: {path}")
        except Exception as exc:
            return self._motion_or_noop(
                min_area, f"Could not load configured detector: {exc}"
            )

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
        config_path_value = str(
            config.get("model_config_path") or self.model_config_path or ""
        ).strip()
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
        raise FileNotFoundError(
            "Caffe prototxt not found; set detection.model_config_path"
        )


class MobileNetSSDCaffePersonDetector:
    """OpenCV DNN MobileNetSSD detector, keeping only VOC class 15: person."""

    PERSON_CLASS_ID = 15

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        confidence_threshold: float = 0.45,
        input_size: int = 300,
    ) -> None:
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
            if (
                class_id != self.PERSON_CLASS_ID
                or confidence < self.confidence_threshold
            ):
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
        self.last_profile: Dict[str, float] = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
        }

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


def _normalise_phone_roi_config(value: Any) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": False,
        "confidence_threshold": 0.16,
        "refine_below_confidence": 0.35,
        "detect_every_n_primary_frames": 2,
        "max_people": 1,
        "min_person_height_px": 160,
        "horizontal_expansion_ratio": 0.20,
        "top_expansion_ratio": 0.04,
        "upper_body_ratio": 0.90,
        "crop_modes": [],
        "cache_primary_frames": 1,
        "cache_confidence_decay": 0.90,
        "cache_min_confidence": 0.10,
        "max_phone_area_ratio": 0.25,
        "nms_threshold": 0.35,
        "containment_threshold": 0.70,
        "context_model": {
            "enabled": False,
            "model_path": "models/yolo11n.rknn",
            "model_family": "yolo11",
            "input_size": 640,
            "confidence_threshold": 0.35,
            "nms_threshold": 0.45,
            "class_filter": ["cell phone"],
            "core_mask": "0_1_2",
            "expected_sha256": "",
        },
    }
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise TypeError("detector.phone_roi_refinement must be a mapping")
    unknown = set(value).difference(defaults)
    if unknown:
        raise ValueError(f"Unknown phone_roi_refinement options: {sorted(unknown)}")
    result = dict(defaults)
    result.update(value)
    result["enabled"] = bool(result["enabled"])
    for key in ("detect_every_n_primary_frames", "max_people"):
        result[key] = int(result[key])
        if result[key] < 1:
            raise ValueError(f"phone_roi_refinement.{key} must be positive")
    result["cache_primary_frames"] = int(result["cache_primary_frames"])
    if result["cache_primary_frames"] < 0:
        raise ValueError(
            "phone_roi_refinement.cache_primary_frames must be non-negative"
        )
    result["min_person_height_px"] = float(result["min_person_height_px"])
    if result["min_person_height_px"] < 1:
        raise ValueError("phone_roi_refinement.min_person_height_px must be positive")
    result["crop_modes"] = _normalise_phone_crop_modes(result["crop_modes"])
    result["context_model"] = _normalise_phone_context_model(result["context_model"])
    for key in (
        "confidence_threshold",
        "refine_below_confidence",
        "horizontal_expansion_ratio",
        "top_expansion_ratio",
        "upper_body_ratio",
        "cache_confidence_decay",
        "cache_min_confidence",
        "max_phone_area_ratio",
        "nms_threshold",
        "containment_threshold",
    ):
        result[key] = float(result[key])
        if not 0.0 <= result[key] <= 1.0:
            raise ValueError(f"phone_roi_refinement.{key} must be between 0 and 1")
    if result["upper_body_ratio"] < 0.25:
        raise ValueError("phone_roi_refinement.upper_body_ratio must be at least 0.25")
    if result["cache_confidence_decay"] <= 0.0:
        raise ValueError("phone_roi_refinement.cache_confidence_decay must be positive")
    return result


def _normalise_phone_crop_modes(value: Any) -> List[Dict[str, Any]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise TypeError("phone_roi_refinement.crop_modes must be a list")
    allowed = {
        "name",
        "top_ratio",
        "bottom_ratio",
        "horizontal_expansion_ratio",
        "top_expansion_ratio",
        "detector",
        "confidence_threshold",
    }
    result: List[Dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise TypeError(
                f"phone_roi_refinement.crop_modes[{index}] must be a mapping"
            )
        unknown = set(raw).difference(allowed)
        if unknown:
            raise ValueError(
                f"Unknown phone_roi_refinement.crop_modes[{index}] options: {sorted(unknown)}"
            )
        mode = {
            "name": str(raw.get("name") or f"crop_{index + 1}"),
            "detector": str(raw.get("detector") or "primary").strip().lower(),
            "top_ratio": float(raw.get("top_ratio", 0.0)),
            "bottom_ratio": float(raw.get("bottom_ratio", 0.90)),
            "horizontal_expansion_ratio": float(
                raw.get("horizontal_expansion_ratio", 0.20)
            ),
            "top_expansion_ratio": float(raw.get("top_expansion_ratio", 0.0)),
            "confidence_threshold": (
                float(raw["confidence_threshold"])
                if raw.get("confidence_threshold") is not None
                else None
            ),
        }
        if mode["detector"] not in {"primary", "context"}:
            raise ValueError(
                f"phone_roi_refinement.crop_modes[{index}].detector must be primary or context"
            )
        for key in (
            "top_ratio",
            "bottom_ratio",
            "horizontal_expansion_ratio",
            "top_expansion_ratio",
        ):
            if not 0.0 <= mode[key] <= 1.0:
                raise ValueError(
                    f"phone_roi_refinement.crop_modes[{index}].{key} must be between 0 and 1"
                )
        if mode["bottom_ratio"] <= mode["top_ratio"]:
            raise ValueError(
                f"phone_roi_refinement.crop_modes[{index}] bottom_ratio must exceed top_ratio"
            )
        if (
            mode["confidence_threshold"] is not None
            and not 0.0 <= mode["confidence_threshold"] <= 1.0
        ):
            raise ValueError(
                f"phone_roi_refinement.crop_modes[{index}].confidence_threshold must be between 0 and 1"
            )
        result.append(mode)
    return result


def _normalise_phone_context_model(value: Any) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": False,
        "model_path": "models/yolo11n.rknn",
        "model_family": "yolo11",
        "input_size": 640,
        "confidence_threshold": 0.35,
        "nms_threshold": 0.45,
        "class_filter": ["cell phone"],
        "class_names": None,
        "class_confidence_thresholds": None,
        "core_mask": "0_1_2",
        "expected_sha256": "",
    }
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise TypeError("detector.phone_roi_refinement.context_model must be a mapping")
    unknown = set(value).difference(defaults)
    if unknown:
        raise ValueError(
            f"Unknown phone_roi_refinement.context_model options: {sorted(unknown)}"
        )
    result = dict(defaults)
    result.update(value)
    result["enabled"] = bool(result["enabled"])
    result["model_path"] = str(result["model_path"] or "").strip()
    result["model_family"] = str(result["model_family"] or "yolo11").strip().lower()
    result["input_size"] = int(result["input_size"])
    result["confidence_threshold"] = float(result["confidence_threshold"])
    result["nms_threshold"] = float(result["nms_threshold"])
    result["core_mask"] = str(result["core_mask"] or "0_1_2")
    result["expected_sha256"] = str(result["expected_sha256"] or "").strip().lower()
    if result["enabled"] and not result["model_path"]:
        raise ValueError("phone_roi_refinement.context_model.model_path is required")
    if result["input_size"] < 32 or result["input_size"] % 32:
        raise ValueError(
            "phone_roi_refinement.context_model.input_size must be divisible by 32"
        )
    for key in ("confidence_threshold", "nms_threshold"):
        if not 0.0 <= result[key] <= 1.0:
            raise ValueError(
                f"phone_roi_refinement.context_model.{key} must be between 0 and 1"
            )
    expected = result["expected_sha256"]
    if expected and (
        len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected)
    ):
        raise ValueError(
            "phone_roi_refinement.context_model.expected_sha256 must be a SHA-256 hex digest"
        )
    return result


def _phone_roi_crop(
    person_bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    config: Dict[str, Any],
    mode: Optional[Dict[str, Any]] = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = person_bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    selected = mode or {}
    horizontal = (
        float(
            selected.get(
                "horizontal_expansion_ratio", config["horizontal_expansion_ratio"]
            )
        )
        * width
    )
    top = (
        float(selected.get("top_expansion_ratio", config["top_expansion_ratio"]))
        * height
    )
    top_ratio = float(selected.get("top_ratio", 0.0))
    bottom_ratio = float(selected.get("bottom_ratio", config["upper_body_ratio"]))
    crop_x1 = max(0, int(math.floor(x1 - horizontal)))
    crop_y1 = max(0, int(math.floor(y1 + top_ratio * height - top)))
    crop_x2 = min(frame_width, int(math.ceil(x2 + horizontal)))
    crop_y2 = min(frame_height, int(math.ceil(y1 + bottom_ratio * height)))
    return crop_x1, crop_y1, max(crop_x1 + 1, crop_x2), max(crop_y1 + 1, crop_y2)


def _person_containing_phone(
    phone_bbox: BBox, people: List[Detection]
) -> Optional[Detection]:
    candidates = [
        person
        for person in people
        if _center_inside(phone_bbox, _expand_box(person.bbox, 0.15))
    ]
    if not candidates:
        return None
    phone_center = _box_center(phone_bbox)
    return min(
        candidates,
        key=lambda person: _normalised_center_distance(phone_center, person.bbox),
    )


def _match_person(previous_bbox: BBox, people: List[Detection]) -> Optional[Detection]:
    if not people:
        return None
    ranked = sorted(
        people,
        key=lambda person: (
            _box_iou(previous_bbox, person.bbox),
            -_normalised_center_distance(_box_center(previous_bbox), person.bbox),
        ),
        reverse=True,
    )
    best = ranked[0]
    if _box_iou(previous_bbox, best.bbox) > 0.05:
        return best
    if _normalised_center_distance(_box_center(previous_bbox), best.bbox) <= 0.75:
        return best
    return None


def _absolute_to_relative(box: BBox, person_bbox: BBox) -> BBox:
    x1, y1, x2, y2 = person_bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return (
        (box[0] - x1) / width,
        (box[1] - y1) / height,
        (box[2] - x1) / width,
        (box[3] - y1) / height,
    )


def _relative_to_absolute(relative: BBox, person_bbox: BBox) -> BBox:
    x1, y1, x2, y2 = person_bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return (
        x1 + relative[0] * width,
        y1 + relative[1] * height,
        x1 + relative[2] * width,
        y1 + relative[3] * height,
    )


def _box_center(box: BBox) -> tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def _normalised_center_distance(point: tuple[float, float], region: BBox) -> float:
    center = _box_center(region)
    diagonal = max(1.0, math.hypot(region[2] - region[0], region[3] - region[1]))
    return math.dist(point, center) / diagonal


def _deduplicate_detections(
    detections: List[Detection], iou_threshold: float, containment_threshold: float
) -> List[Detection]:
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: List[Detection] = []
    for candidate in ordered:
        if any(
            _box_iou(candidate.bbox, existing.bbox) > iou_threshold
            or _intersection_over_smaller(candidate.bbox, existing.bbox)
            >= containment_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _label(detection: Detection) -> str:
    return str(detection.label).strip().lower()


def _expand_box(
    box: tuple[float, float, float, float], ratio: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    dx = max(0.0, x2 - x1) * ratio
    dy = max(0.0, y2 - y1) * ratio
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def _center_inside(
    target: tuple[float, float, float, float], region: tuple[float, float, float, float]
) -> bool:
    center_x = (target[0] + target[2]) * 0.5
    center_y = (target[1] + target[3]) * 0.5
    return region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    return _box_area(
        (
            max(first[0], second[0]),
            max(first[1], second[1]),
            min(first[2], second[2]),
            min(first[3], second[3]),
        )
    )


def _box_iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    intersection = _intersection_area(first, second)
    return intersection / max(1.0, _box_area(first) + _box_area(second) - intersection)


def _intersection_over_smaller(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    intersection = _intersection_area(first, second)
    return intersection / max(1.0, min(_box_area(first), _box_area(second)))


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
