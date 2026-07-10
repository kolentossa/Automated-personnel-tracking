"""Small config loader for the RK3588 web tracking app.

The board environment intentionally avoids extra dependencies, so this module
parses the simple config.yaml shape used by the project instead of requiring
PyYAML.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {"host": "0.0.0.0", "port": 8000},
    "camera": {
        "source_type": "camera",
        "camera_device": "/dev/video11",
        "capture_backend": "gstreamer",
        "auto_detect": True,
        "width": 1280,
        "height": 720,
        "fps": 15,
        "gstreamer_pipeline": "v4l2src device={device} ! video/x-raw,format=NV12,width={width},height={height} ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false",
        "retry_interval_sec": 3.0,
        "video_file": "data/sample.mp4",
    },
    "detection": {
        "model_path": "models/MobileNetSSD_deploy.caffemodel",
        "model_config_path": "models/MobileNetSSD_deploy.prototxt",
        "confidence_threshold": 0.45,
        "input_size": 300,
        "nms_threshold": 0.45,
        "motion_min_area": 500,
        "allow_motion_fallback": False,
    },
    "detector": {
        "type": "rknn-yolo",
        "model_path": "models/yolov8n.rknn",
        "model_family": "yolov8",
        "input_size": 640,
        "confidence_threshold": 0.35,
        "nms_threshold": 0.45,
        "class_filter": ["person"],
        "fallback_to_cpu": False,
    },
    "performance": {
        "target_fps": 30,
        "target_latency_ms": 30,
        "latest_frame_only": True,
        "queue_size": 1,
        "detect_every_n_frames": 3,
        "cpu_affinity": "4-7",
        "timing_window_frames": 30,
    },
    "stream": {"jpeg_quality": 74, "width": 800, "height": 450},
    "tracking": {"iou_threshold": 0.3, "max_missing": 20, "min_confidence": 0.15},
    "counting": {
        "line": {"x1": 640, "y1": 0, "x2": 640, "y2": 720},
        "direction": {"mode": "left_to_right"},
        "cooldown_frames": 20,
    },
    "privacy": {
        "face_mosaic_enabled": True,
        "face_detector": "retinaface-onnx",
        "face_model_path": "models/RetinaFace_mobile320.onnx",
        "face_input_size": 320,
        "face_confidence_threshold": 0.6,
        "face_nms_threshold": 0.4,
        "face_detector_threads": 1,
        "face_detect_every_n_frames": 10,
        "face_result_max_age_ms": 1000,
        "face_mosaic_padding": 0.15,
        "head_fallback_enabled": True,
        "mosaic_strength": 14,
    },
}


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path.exists():
        parsed = _parse_simple_yaml(path.read_text(encoding="utf-8"))
        _deep_update(config, parsed)
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def save_counting_config(
    line: Dict[str, Any],
    direction: str,
    cooldown_frames: int = 20,
    path: Path = CONFIG_PATH,
) -> None:
    """Persist only the counting block while preserving the rest of config.yaml."""

    block = _render_counting_block(line, direction, cooldown_frames)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(_replace_top_level_block(text, "counting", block), encoding="utf-8")


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _render_counting_block(line: Dict[str, Any], direction: str, cooldown_frames: int) -> str:
    return "\n".join(
        [
            "counting:",
            "  line:",
            f"    x1: {_format_number(line['x1'])}",
            f"    y1: {_format_number(line['y1'])}",
            f"    x2: {_format_number(line['x2'])}",
            f"    y2: {_format_number(line['y2'])}",
            "  direction:",
            f"    mode: {direction}",
            f"  cooldown_frames: {int(cooldown_frames)}",
        ]
    )


def _replace_top_level_block(text: str, key: str, block: str) -> str:
    lines = text.splitlines()
    start = _find_top_level_key(lines, key)
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


def _find_top_level_key(lines: Iterable[str], key: str) -> int | None:
    prefix = f"{key}:"
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return index
    return None


def _format_number(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = _strip_comment(raw_line.rstrip())
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            continue
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)
    return root


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index].rstrip()
    return line


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
