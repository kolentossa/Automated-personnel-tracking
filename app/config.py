"""Small config loader for the RK3588 web tracking app.

The board environment intentionally avoids extra dependencies, so this module
parses the simple config.yaml shape used by the project instead of requiring
PyYAML.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

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
        "model_path": "",
        "confidence_threshold": 0.35,
        "input_size": 640,
        "nms_threshold": 0.45,
        "motion_min_area": 500,
    },
    "tracking": {"iou_threshold": 0.3, "max_missing": 20, "min_confidence": 0.15},
    "counting": {"line": "auto", "enter_direction": "positive_to_negative", "cooldown_frames": 20},
    "privacy": {"face_mosaic_enabled": True, "head_fallback_enabled": True, "mosaic_strength": 14},
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


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


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