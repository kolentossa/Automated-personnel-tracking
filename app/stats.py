"""Thread-safe line-crossing statistics for the Web API."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Sequence, Tuple

from vision.types import TrackedPerson, bbox_centroid

Point = Tuple[float, float]
Line = Tuple[Point, Point]

TIMING_KEYS = (
    "capture_ms",
    "queue_wait_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "behavior_preprocess_ms",
    "behavior_inference_ms",
    "behavior_postprocess_ms",
    "behavior_analysis_ms",
    "event_output_ms",
    "tracking_ms",
    "privacy_ms",
    "draw_ms",
    "encode_ms",
    "total_latency_ms",
)

DIRECTION_TO_ENTER = {
    "left_to_right": "positive_to_negative",
    "right_to_left": "negative_to_positive",
}

ENTER_TO_DIRECTION = {value: key for key, value in DIRECTION_TO_ENTER.items()}


class StatsManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.direction_mode = _direction_mode_from_config(config)
        self.enter_direction = DIRECTION_TO_ENTER[self.direction_mode]
        self.cooldown_frames = int(config.get("cooldown_frames") or 20)
        self._line_arg = config.get("line", "auto")
        self._line: Optional[Line] = None
        self.frame_width = int(config.get("_default_frame_width") or 960)
        self.frame_height = int(config.get("_default_frame_height") or 540)
        self._lock = threading.RLock()
        self._last_sides: Dict[int, int] = {}
        self._cooldowns: Dict[int, int] = {}
        self._events: Deque[dict] = deque(maxlen=20)
        self.current_occupancy = 0
        self.total_entered = 0
        self.total_exited = 0
        self.active_tracks = 0
        self.visible_persons = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.frame_index = 0
        self.camera_status = "offline"
        self.last_error = ""
        self.source = ""
        self.model = ""
        self.detector = ""
        self.npu_enabled = False
        self.running = False
        self.privacy_mode = True
        self.face_mosaic_enabled = True
        self.available_cameras = []
        self.timings: Dict[str, float] = {key: 0.0 for key in TIMING_KEYS}
        timing_window = max(1, int(config.get("_timing_window_frames") or 30))
        self._timing_samples: Dict[str, Deque[float]] = {
            key: deque(maxlen=timing_window) for key in TIMING_KEYS
        }

    def update_tracks(self, tracks: Iterable[TrackedPerson], frame_shape: Sequence[int]) -> None:
        track_list = list(tracks)
        height, width = int(frame_shape[0]), int(frame_shape[1])
        line = self.line_for_frame(width, height)
        with self._lock:
            self.frame_width = width
            self.frame_height = height
            self.active_tracks = len(track_list)
            self.visible_persons = len(track_list)
            for track in track_list:
                side = _side_of_line(bbox_centroid(track.bbox), line)
                if side == 0:
                    continue
                previous = self._last_sides.get(track.track_id)
                cooldown = self._cooldowns.get(track.track_id, 0)
                if previous is not None and previous != side and cooldown <= 0:
                    direction = _direction(previous, side)
                    event_type = "ENTER" if direction == self.enter_direction else "EXIT"
                    self._record_event(track.track_id, event_type)
                    self._cooldowns[track.track_id] = self.cooldown_frames
                self._last_sides[track.track_id] = side
                if track.track_id in self._cooldowns:
                    self._cooldowns[track.track_id] = max(0, self._cooldowns[track.track_id] - 1)

    def update_runtime(
        self,
        *,
        fps: float,
        latency_ms: float,
        frame_index: int,
        camera_status: str,
        source: str,
        model: str,
        detector: str,
        running: bool,
        privacy_mode: bool,
        face_mosaic_enabled: bool,
        last_error: str,
        available_cameras: list,
        npu_enabled: bool = False,
        timings: Optional[Dict[str, float]] = None,
    ) -> None:
        with self._lock:
            self.fps = round(float(fps), 2)
            self.latency_ms = round(float(latency_ms), 1)
            self.frame_index = int(frame_index)
            self.camera_status = camera_status
            self.source = source
            self.model = model
            self.detector = detector
            self.npu_enabled = bool(npu_enabled)
            self.running = running
            self.privacy_mode = privacy_mode
            self.face_mosaic_enabled = face_mosaic_enabled
            self.last_error = last_error
            self.available_cameras = available_cameras
            if timings is not None:
                for key in TIMING_KEYS:
                    self._timing_samples[key].append(float(timings.get(key, 0.0) or 0.0))
                    samples = self._timing_samples[key]
                    self.timings[key] = round(sum(samples) / len(samples), 1)
                self.latency_ms = self.timings["total_latency_ms"] or self.latency_ms

    def reset(self) -> None:
        with self._lock:
            self.current_occupancy = 0
            self.total_entered = 0
            self.total_exited = 0
            self._events.clear()
            self._last_sides.clear()
            self._cooldowns.clear()

    def configure_counting(self, line: Dict[str, Any], direction: str) -> dict:
        line_dict = _coerce_line_dict(line)
        direction_mode = _normalize_direction_mode(direction)
        parsed_line = _parse_line(line_dict, self.frame_width, self.frame_height)
        with self._lock:
            self._line_arg = line_dict
            self._line = parsed_line
            self.direction_mode = direction_mode
            self.enter_direction = DIRECTION_TO_ENTER[direction_mode]
            self.config["line"] = line_dict
            self.config["direction"] = {"mode": direction_mode}
            self.config.pop("enter_direction", None)
            self._last_sides.clear()
            self._cooldowns.clear()
            return self.counting_config()

    def counting_config(self) -> dict:
        with self._lock:
            line = self._line or _parse_line(self._line_arg, self.frame_width, self.frame_height)
            return {
                "line": _line_as_dict(line),
                "direction": self.direction_mode,
                "cooldown_frames": self.cooldown_frames,
                "frame_size": {"width": self.frame_width, "height": self.frame_height},
            }

    def snapshot(self) -> dict:
        with self._lock:
            line = self._line
            data = {
                "current_occupancy": self.current_occupancy,
                "total_entered": self.total_entered,
                "total_exited": self.total_exited,
                "active_tracks": self.active_tracks,
                "visible_persons": self.visible_persons,
                "fps": self.fps,
                "latency_ms": self.latency_ms,
                "camera_status": self.camera_status,
                "frame_index": self.frame_index,
                "source": self.source,
                "model": self.model,
                "detector": self.detector,
                "npu_enabled": self.npu_enabled,
                "privacy_mode": self.privacy_mode,
                "face_mosaic_enabled": self.face_mosaic_enabled,
                "running": self.running,
                "last_error": self.last_error,
                "line": _line_as_list(line),
                "recent_events": list(self._events),
                "available_cameras": self.available_cameras,
            }
            data.update(self.timings)
            if not data["total_latency_ms"]:
                data["total_latency_ms"] = self.latency_ms
            return data

    def line_for_frame(self, width: int, height: int) -> Line:
        if self._line is None:
            self._line = _parse_line(self._line_arg, width, height)
        return self._line

    def current_line(self) -> Optional[Line]:
        return self._line

    def _record_event(self, track_id: int, event_type: str) -> None:
        if event_type == "ENTER":
            self.total_entered += 1
            self.current_occupancy += 1
        else:
            self.total_exited += 1
            self.current_occupancy = max(0, self.current_occupancy - 1)
        self._events.appendleft(
            {
                "time": time.strftime("%H:%M:%S"),
                "track_id": int(track_id),
                "event_type": event_type,
                "current_occupancy": self.current_occupancy,
                "total_entered": self.total_entered,
                "total_exited": self.total_exited,
            }
        )


def _parse_line(value: Any, width: int, height: int) -> Line:
    if not value or str(value).lower() == "auto":
        x = width / 2.0
        return (x, 0.0), (x, float(height))
    if isinstance(value, dict):
        parts = [float(value[key]) for key in ("x1", "y1", "x2", "y2")]
        return (parts[0], parts[1]), (parts[2], parts[3])
    if isinstance(value, list) and len(value) == 4:
        parts = [float(item) for item in value]
    else:
        parts = [float(item.strip()) for item in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("counting.line must be auto or x1,y1,x2,y2")
    return (parts[0], parts[1]), (parts[2], parts[3])


def _side_of_line(point: Point, line: Line) -> int:
    (x1, y1), (x2, y2) = line
    px, py = point
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def _direction(previous: int, current: int) -> str:
    before = "positive" if previous > 0 else "negative"
    after = "positive" if current > 0 else "negative"
    return f"{before}_to_{after}"


def _line_as_list(line: Optional[Line]) -> list:
    if line is None:
        return []
    return [[line[0][0], line[0][1]], [line[1][0], line[1][1]]]


def _line_as_dict(line: Line) -> dict:
    return {
        "x1": _clean_number(line[0][0]),
        "y1": _clean_number(line[0][1]),
        "x2": _clean_number(line[1][0]),
        "y2": _clean_number(line[1][1]),
    }


def _coerce_line_dict(value: Dict[str, Any]) -> dict:
    if not isinstance(value, dict):
        raise ValueError("line must be an object with x1, y1, x2, y2")
    line = {key: float(value[key]) for key in ("x1", "y1", "x2", "y2")}
    if line["x1"] == line["x2"] and line["y1"] == line["y2"]:
        raise ValueError("counting line start and end points must be different")
    return {key: _clean_number(number) for key, number in line.items()}


def _direction_mode_from_config(config: Dict[str, Any]) -> str:
    direction = config.get("direction")
    if isinstance(direction, dict):
        mode = direction.get("mode")
    else:
        mode = direction
    if mode:
        return _normalize_direction_mode(str(mode))
    return ENTER_TO_DIRECTION.get(str(config.get("enter_direction") or ""), "left_to_right")


def _normalize_direction_mode(value: str) -> str:
    direction = str(value or "").strip().lower()
    if direction not in DIRECTION_TO_ENTER:
        raise ValueError("direction must be left_to_right or right_to_left")
    return direction


def _clean_number(value: float) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return round(number, 3)
