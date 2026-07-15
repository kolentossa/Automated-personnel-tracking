"""Configuration loading and validation for behavior detection."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from app.config import PROJECT_ROOT, load_yaml_file

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
            "class_names": {"0": "person", "67": "cell phone"},
            "core_mask": "0_1_2",
            "expected_sha256": "ff3a64e6fe180203128c8d42456b458d208d3a1e2217d63683af00d6194e82ea",
        },
        "behavior": {
            "enabled": False,
            "required": False,
            "type": "rknn-yolo",
            "model_path": "models/behavior_yolov8n.rknn",
            "model_family": "yolov8",
            "input_size": 640,
            "confidence_threshold": 0.3,
            "nms_threshold": 0.45,
            "class_names": ["person", "cell phone", "cigarette", "smoke", "flame", "lighter", "hand"],
            "class_filter": ["cigarette", "smoke", "flame", "lighter", "hand"],
            "core_mask": "2",
            "detect_every_n_frames": 3,
            "expected_sha256": "",
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
            "duration_ms": 1500,
            "min_consecutive_frames": 8,
            "confidence_threshold": 0.35,
            "cooldown_ms": 10000,
            "max_gap_frames": 3,
            "max_face_distance_ratio": 0.9,
        },
        "phone_playing": {
            "duration_ms": 2500,
            "min_consecutive_frames": 12,
            "confidence_threshold": 0.35,
            "cooldown_ms": 10000,
            "max_gap_frames": 3,
        },
        "unauthorized_photography": {
            "duration_ms": 1200,
            "min_consecutive_frames": 6,
            "confidence_threshold": 0.35,
            "cooldown_ms": 15000,
            "max_gap_frames": 2,
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
        "max_mouth_distance_ratio": 1.1,
        "allow_persistent_cigarette": True,
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
    _validate(config)
    return config


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


def _validate_rule(name: str, value: Dict[str, Any]) -> None:
    for key in ("duration_ms", "min_consecutive_frames", "cooldown_ms", "max_gap_frames"):
        if int(value.get(key) or 0) < 0:
            raise ValueError(f"{name}.{key} must be non-negative")
    confidence = float(value.get("confidence_threshold") or 0.0)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name}.confidence_threshold must be between 0 and 1")
