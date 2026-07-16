"""Configuration loading and validation for behavior detection."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from app.config import PROJECT_ROOT, load_accuracy_profile, load_yaml_file

BEHAVIOR_CONFIG_PATH = PROJECT_ROOT / "configs" / "rk3588_behavior_detection.yaml"

DEFAULT_BEHAVIOR_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "camera_id": "rk3588-camera-01",
    "source": {
        "use_for_runtime": False,
        "type": "camera",
        "camera_device": "/dev/video11",
        "rtsp_url": "",
        "video_file": "",
    },
    "models": {
        "primary": {
            "use_for_runtime": False,
            "model_path": "models/yolov8n.rknn",
            "model_family": "yolov8",
            "class_names": {"0": "person", "67": "cell phone"},
            "class_confidence_thresholds": {"person": 0.35, "cell phone": 0.35},
            "core_mask": "0_1_2",
            "expected_sha256": "ff3a64e6fe180203128c8d42456b458d208d3a1e2217d63683af00d6194e82ea",
        },
        "behavior": {
            "enabled": True,
            "required": True,
            "type": "rknn-yolo",
            "model_path": "models/behavior_damoyolo_cigarette_int8.rknn",
            "model_family": "damoyolo",
            "input_size": 640,
            "confidence_threshold": 0.35,
            "nms_threshold": 0.35,
            "class_names": {"0": "cigarette", "1": "__unused__"},
            "class_filter": ["cigarette"],
            "box_filter": {
                "min_side_px": 5,
                "max_aspect_ratio": 4.0,
                "max_frame_area_ratio": 0.025,
                "duplicate_containment_threshold": 0.70,
            },
            "core_mask": "0_1_2",
            "detect_every_n_frames": 5,
            "expected_sha256": "d04c43a3a695c9985fbd03db1e0a2956763374fd686d949b8cd96cabdc7c5941",
        },
    },
    "class_groups": {
        "phone": ["cell phone", "phone", "mobile phone"],
        "cigarette": ["cigarette", "cigar"],
        "smoke": ["smoke"],
        "flame": ["flame", "fire"],
        "lighter": ["lighter"],
        "hand": ["hand", "left hand", "right hand"],
    },
    "association": {"person_expansion_ratio": 0.12, "stale_track_ms": 5000},
    "phone": {
        "enabled": True,
        "phone_call": {
            "duration_ms": 1200,
            "min_consecutive_frames": 6,
            "confidence_threshold": 0.17,
            "cooldown_ms": 10000,
            "max_gap_frames": 10,
            "rearm_absence_ms": 2500,
            "max_head_distance_ratio": 1.15,
            "max_horizontal_offset_ratio": 1.0,
            "max_vertical_offset_ratio": 0.75,
        },
        "phone_playing": {
            "duration_ms": 1800,
            "min_consecutive_frames": 8,
            "confidence_threshold": 0.17,
            "cooldown_ms": 10000,
            "max_gap_frames": 10,
            "rearm_absence_ms": 2500,
            "max_head_exclusion_distance_ratio": 0.75,
        },
        "unauthorized_photography": {
            "duration_ms": 1200,
            "min_consecutive_frames": 6,
            "confidence_threshold": 0.35,
            "cooldown_ms": 15000,
            "max_gap_frames": 2,
            "rearm_absence_ms": 2500,
            "max_alignment_angle_deg": 35,
        },
        "prohibited_rois": [],
    },
    "smoking": {
        "enabled": True,
        "duration_ms": 2000,
        "min_consecutive_frames": 8,
        "confidence_threshold": 0.35,
        "cooldown_ms": 15000,
        "max_gap_frames": 5,
        "rearm_absence_ms": 3000,
        "max_mouth_distance_ratio": 1.1,
        "allow_persistent_cigarette": False,
        "smoke_only_environment_event": False,
    },
    "evidence": {
        "snapshot_dir": "data/behavior_events/snapshots",
        "video_clip_dir": "data/behavior_events/clips",
        "event_log_path": "logs/behavior_events.jsonl",
        "save_unannotated_snapshot": True,
        "save_annotated_snapshot": True,
        "save_video_clip": False,
        "save_debug_frames": False,
        "jpeg_quality": 90,
        "queue_size": 8,
    },
    "logging": {"level": "INFO"},
}


def load_behavior_config(path: Path = BEHAVIOR_CONFIG_PATH) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_BEHAVIOR_CONFIG)
    _deep_update(config, load_yaml_file(path))
    _apply_accuracy_profile(config, load_accuracy_profile().get("behavior", {}))
    _validate(config)
    return config


def _apply_accuracy_profile(config: Dict[str, Any], profile: Dict[str, Any]) -> None:
    phone = config.setdefault("phone", {})
    smoking = config.setdefault("smoking", {})
    phone["enabled"] = bool(profile.get("phone_detection_enabled", True))
    smoking["enabled"] = bool(profile.get("smoking_detection_enabled", True))
    head_region = dict(profile.get("head_region") or {})
    phone["head_region"] = head_region
    smoking["head_region"] = head_region

    phone_confidence = float(profile.get("phone_confidence", 0.16))
    phone.setdefault("phone_call", {})["confidence_threshold"] = float(
        profile.get("phone_call_confidence", phone_confidence)
    )
    phone.setdefault("phone_playing", {})["confidence_threshold"] = float(
        profile.get("phone_playing_confidence", phone_confidence)
    )
    if "unauthorized_photography_confidence" in profile:
        phone.setdefault("unauthorized_photography", {})["confidence_threshold"] = (
            float(profile["unauthorized_photography_confidence"])
        )

    passthrough = (
        "phone_confidence",
        "cigarette_raw_confidence",
        "cigarette_verified_confidence",
        "person_smoking_confidence",
        "smoke_confidence",
        "fire_confidence",
        "lighter_confidence",
        "phone_cigarette_ioa_threshold",
        "phone_confidence_margin",
        "temporal_window_frames",
        "minimum_verified_frames",
        "max_candidate_length_person_ratio",
        "max_candidate_area_person_ratio",
        "min_candidate_center_person_ratio",
        "upper_body_ratio",
        "candidate_motion_ratio",
        "hard_negative_confidence",
        "hard_negative_ioa_threshold",
        "hard_negative_labels",
    )
    for key in passthrough:
        if key in profile:
            smoking[key] = profile[key]
    smoking["duration_ms"] = int(profile.get("smoking_duration_ms", 1800))
    smoking["cooldown_ms"] = int(profile.get("event_cooldown_ms", 15000))

    model = config.setdefault("models", {}).setdefault("behavior", {})
    raw_confidence = float(profile.get("cigarette_raw_confidence", 0.30))
    model["confidence_threshold"] = raw_confidence
    model["class_confidence_thresholds"] = {"cigarette": raw_confidence}


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _validate(config: Dict[str, Any]) -> None:
    if not str(config.get("camera_id") or "").strip():
        raise ValueError("behavior camera_id must not be empty")
    for event_name in ("phone_call", "phone_playing", "unauthorized_photography"):
        _validate_rule(event_name, config.get("phone", {}).get(event_name, {}))
    _validate_rule("smoking", config.get("smoking", {}))
    rois = config.get("phone", {}).get("prohibited_rois", [])
    if not isinstance(rois, list):
        raise ValueError("phone.prohibited_rois must be an inline JSON list")
    _validate_behavior_model(config)


def _validate_behavior_model(config: Dict[str, Any]) -> None:
    model = config.get("models", {}).get("behavior", {})
    if not bool(model.get("enabled", False)):
        return
    if int(model.get("input_size") or 0) < 32:
        raise ValueError("models.behavior.input_size must be at least 32")
    if int(model.get("detect_every_n_frames") or 0) < 1:
        raise ValueError("models.behavior.detect_every_n_frames must be positive")
    class_names = model.get("class_names") or []
    values = class_names.values() if isinstance(class_names, dict) else class_names
    labels = {str(value).strip().lower() for value in values}
    if not labels or "" in labels:
        raise ValueError("models.behavior.class_names must contain non-empty labels")
    selected = {
        str(value).strip().lower() for value in (model.get("class_filter") or [])
    }
    unknown = selected.difference(labels)
    if unknown:
        raise ValueError(
            f"models.behavior.class_filter contains unknown labels: {sorted(unknown)}"
        )
    direct_evidence = {
        "cigarette",
        "cigar",
        "smoking",
        "person smoking",
        "hand with cigarette",
    }
    effective = selected or labels
    if bool(
        config.get("smoking", {}).get("enabled", True)
    ) and not effective.intersection(direct_evidence):
        raise ValueError(
            "Enabled smoking detection requires a direct cigarette or smoking class"
        )


def _validate_rule(name: str, value: Dict[str, Any]) -> None:
    for key in (
        "duration_ms",
        "min_consecutive_frames",
        "cooldown_ms",
        "max_gap_frames",
        "rearm_absence_ms",
    ):
        if int(value.get(key) or 0) < 0:
            raise ValueError(f"{name}.{key} must be non-negative")
    confidence = float(value.get("confidence_threshold") or 0.0)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name}.confidence_threshold must be between 0 and 1")
