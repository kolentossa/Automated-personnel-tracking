"""配置只有两步：用 YAML 库读取并补默认值；保存时只替换 counting 段。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "camera_device": "/dev/video11",
        "auto_detect": True,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "retry_interval_sec": 3.0,
    },
    "detector": {
        "model_path": "models/crowdhuman448_sigmoid_yolov5m_int8.rknn",
        "input_size": 448,
        "confidence_threshold": 0.35,
        "nms_threshold": 0.45,
    },
    "performance": {
        "target_fps": 30,
        "cpu_affinity": "4-7",
        "timing_window_frames": 30,
    },
    "stream": {"jpeg_quality": 74, "width": 800, "height": 450},
    "tracking": {
        "track_high_thresh": 0.35, "track_low_thresh": 0.15, "new_track_thresh": 0.35,
        "match_thresh": 0.8, "lost_track_buffer_seconds": 4.0,
        "prediction_display_seconds": 0.15, "reference_fps": 30,
    },
    "counting": {
        "line": {"x1": 640, "y1": 0, "x2": 640, "y2": 720},
        "direction": {"mode": "left_to_right"},
        "anchor": "auto",
        "cooldown_frames": 8,
        "hysteresis_px": 64,
        "rearm_distance_px": 96,
        "confirmation_frames": 5,
        "time_confirmation_enabled": True,
        "confirmation_max_gap_seconds": 0.2,
        "state_recovery_enabled": True,
        "state_recovery_max_gap_seconds": 0.6,
        "track_state_ttl_frames": 120,
        "count_once_per_direction_per_track": False,
    },
    "privacy": {
        "face_mosaic_enabled": True,
        "face_detector": "retinaface-onnx",
        "face_model_path": "models/RetinaFace_mobile320.onnx",
        "face_input_size": 320,
        "face_confidence_threshold": 0.6,
        "face_nms_threshold": 0.4,
        "face_detector_threads": 1,
        "face_detect_every_n_frames": 5,
        "face_result_max_age_ms": 1000,
        "face_mosaic_padding": 0.25,
        "face_tracking_enabled": True,
        "face_tracking_min_points": 4,
        "face_tracking_win_size": 31,
        "face_tracking_max_level": 3,
        "face_tracking_max_motion_px": 96,
        "head_fallback_enabled": True,
        "mosaic_strength": 14,
    },
}


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return _parse_config(text)


def _parse_config(text: str) -> Dict[str, Any]:
    """深复制默认配置，避免一次运行或测试修改了下次加载的初始值。"""
    config = deepcopy(DEFAULT_CONFIG)
    _deep_update(config, yaml.safe_load(text) or {})
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def save_counting_config(
    line: Dict[str, Any],
    direction: str,
    cooldown_frames: int = 8,
    path: Path = CONFIG_PATH,
    *,
    hysteresis_px: float = 64,
    rearm_distance_px: float = 96,
    confirmation_frames: int = 5,
    track_state_ttl_frames: int = 120,
    count_once_per_direction_per_track: bool = False,
    anchor: str = "auto",
) -> None:
    """只保存计数设置，摄像头、模型和其他配置的原文保持不动。"""

    block = "\n".join(
        [
            "counting:",
            "  line:",
            f"    x1: {_format_number(line['x1'])}",
            f"    y1: {_format_number(line['y1'])}",
            f"    x2: {_format_number(line['x2'])}",
            f"    y2: {_format_number(line['y2'])}",
            "  direction:",
            f"    mode: {direction}",
            f"  anchor: {anchor}",
            f"  cooldown_frames: {int(cooldown_frames)}",
            f"  hysteresis_px: {_format_number(hysteresis_px)}",
            f"  rearm_distance_px: {_format_number(rearm_distance_px)}",
            f"  confirmation_frames: {int(confirmation_frames)}",
            f"  track_state_ttl_frames: {int(track_state_ttl_frames)}",
            "  count_once_per_direction_per_track: "
            + ("true" if count_once_per_direction_per_track else "false"),
        ]
    )

    text = path.read_text(encoding="utf-8") if path.exists() else ""
    # 页面只编辑线和基础距离，已有的时间确认与轨迹恢复开关也要保留。
    settings = _parse_config(text)["counting"]
    for key in ("time_confirmation_enabled", "state_recovery_enabled"):
        block += f"\n  {key}: {str(settings[key]).lower()}"
    for key in ("confirmation_max_gap_seconds", "state_recovery_max_gap_seconds"):
        block += f"\n  {key}: {_format_number(settings[key])}"
    path.write_text(_replace_top_level_block(text, "counting", block), encoding="utf-8")


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _replace_top_level_block(text: str, key: str, block: str) -> str:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith(f"{key}:")), None)
    block_lines = block.splitlines()
    if start is None:
        prefix = lines[:]
        if prefix and prefix[-1].strip():
            prefix.append("")
        return "\n".join(prefix + block_lines) + "\n"

    end = start + 1
    while end < len(lines):
        raw = lines[end]
        if raw.strip() and not raw.startswith((" ", "\t")) and ":" in raw:
            break
        end += 1
    return "\n".join(lines[:start] + block_lines + lines[end:]) + "\n"


def _format_number(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")
