"""Thread-safe line-crossing statistics for the Web API."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from math import hypot
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


@dataclass
class _TrackCrossingState:
    confirmed_side: Optional[int] = None
    candidate_side: int = 0
    candidate_frames: int = 0
    crossed_corridor: bool = False
    last_seen_frame: int = 0
    last_event_frame: int = -1_000_000
    entered_counted: bool = False
    exited_counted: bool = False


class StatsManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.direction_mode = _direction_mode_from_config(config)
        self.enter_direction = DIRECTION_TO_ENTER[self.direction_mode]
        self.cooldown_frames = _int_setting(config, "cooldown_frames", 8, minimum=0)
        self.hysteresis_px = _float_setting(config, "hysteresis_px", 64.0, minimum=1.0)
        self.rearm_distance_px = max(
            self.hysteresis_px + 1.0,
            _float_setting(config, "rearm_distance_px", 96.0, minimum=1.0),
        )
        self.confirmation_frames = _int_setting(config, "confirmation_frames", 5, minimum=1)
        self.track_state_ttl_frames = _int_setting(config, "track_state_ttl_frames", 120, minimum=1)
        self.count_once_per_direction_per_track = _bool_setting(
            config, "count_once_per_direction_per_track", False
        )
        self._line_arg = config.get("line", "auto")
        self._line: Optional[Line] = None
        self.frame_width = int(config.get("_default_frame_width") or 960)
        self.frame_height = int(config.get("_default_frame_height") or 540)
        self._lock = threading.RLock()
        self._track_states: Dict[int, _TrackCrossingState] = {}
        self._latest_track_points: Dict[int, Point] = {}
        self._track_frame_index = 0
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
        with self._lock:
            line = self.line_for_frame(width, height)
            self._track_frame_index += 1
            tracking_frame = self._track_frame_index
            self.frame_width = width
            self.frame_height = height
            self.active_tracks = len(track_list)
            self.visible_persons = len(track_list)
            self._latest_track_points = {
                track.track_id: bbox_centroid(track.bbox) for track in track_list
            }
            for track in track_list:
                distance = _signed_distance_to_line(self._latest_track_points[track.track_id], line)
                side = _side_outside_margin(distance, self.hysteresis_px)
                stable_side = _side_outside_margin(distance, self.rearm_distance_px)
                state = self._track_states.setdefault(track.track_id, _TrackCrossingState())
                state.last_seen_frame = tracking_frame
                if state.confirmed_side is not None and (
                    side == 0 or side != state.confirmed_side
                ):
                    state.crossed_corridor = True

                confirmation_side = side if state.confirmed_side is None else stable_side
                if confirmation_side == 0:
                    state.candidate_side = 0
                    state.candidate_frames = 0
                    continue
                if state.candidate_side == confirmation_side:
                    state.candidate_frames += 1
                else:
                    state.candidate_side = confirmation_side
                    state.candidate_frames = 1
                if state.candidate_frames < self.confirmation_frames:
                    continue

                previous = state.confirmed_side
                state.candidate_frames = self.confirmation_frames
                if previous is None:
                    state.confirmed_side = confirmation_side
                    state.crossed_corridor = False
                    continue
                if previous == confirmation_side:
                    state.crossed_corridor = False
                    continue
                if not state.crossed_corridor:
                    continue
                if tracking_frame - state.last_event_frame < self.cooldown_frames:
                    continue

                state.confirmed_side = confirmation_side
                state.crossed_corridor = False
                direction = _direction(previous, confirmation_side)
                event_type = "ENTER" if direction == self.enter_direction else "EXIT"
                if self.count_once_per_direction_per_track:
                    if event_type == "ENTER" and state.entered_counted:
                        continue
                    if event_type == "EXIT" and state.exited_counted:
                        continue
                self._record_event(track.track_id, event_type)
                if event_type == "ENTER":
                    state.entered_counted = True
                else:
                    state.exited_counted = True
                state.last_event_frame = tracking_frame

            stale_before = tracking_frame - self.track_state_ttl_frames
            stale_track_ids = [
                track_id
                for track_id, state in self._track_states.items()
                if state.last_seen_frame < stale_before
            ]
            for track_id in stale_track_ids:
                del self._track_states[track_id]

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
            self._track_states.clear()
            self._latest_track_points.clear()
            self._track_frame_index = 0

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
            self._track_states = self._states_rebased_to_line(parsed_line)
            return self.counting_config()

    def _states_rebased_to_line(self, line: Line) -> Dict[int, _TrackCrossingState]:
        states: Dict[int, _TrackCrossingState] = {}
        for track_id, point in self._latest_track_points.items():
            side = _side_outside_margin(_signed_distance_to_line(point, line), 0.0)
            if side != 0:
                states[track_id] = _TrackCrossingState(
                    confirmed_side=side,
                    last_seen_frame=self._track_frame_index,
                )
        return states

    def counting_config(self) -> dict:
        with self._lock:
            line = self._line or _parse_line(self._line_arg, self.frame_width, self.frame_height)
            return {
                "line": _line_as_dict(line),
                "direction": self.direction_mode,
                "cooldown_frames": self.cooldown_frames,
                "hysteresis_px": _clean_number(self.hysteresis_px),
                "rearm_distance_px": _clean_number(self.rearm_distance_px),
                "confirmation_frames": self.confirmation_frames,
                "track_state_ttl_frames": self.track_state_ttl_frames,
                "count_once_per_direction_per_track": self.count_once_per_direction_per_track,
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
    return _side_outside_margin(_signed_distance_to_line(point, line), 0.0)


def _signed_distance_to_line(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    length = hypot(x2 - x1, y2 - y1)
    if length <= 0.0:
        return 0.0
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return cross / length


def _side_outside_margin(distance: float, margin: float) -> int:
    if distance > margin:
        return 1
    if distance < -margin:
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


def _int_setting(config: Dict[str, Any], key: str, default: int, minimum: int) -> int:
    value = config.get(key)
    return max(minimum, int(default if value is None else value))


def _float_setting(config: Dict[str, Any], key: str, default: float, minimum: float) -> float:
    value = config.get(key)
    return max(minimum, float(default if value is None else value))


def _bool_setting(config: Dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)
