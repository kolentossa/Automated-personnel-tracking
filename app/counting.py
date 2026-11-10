"""过线计数状态机：真实观测 → 候选侧 → 稳定确认 → 进入/离开事件。

create_counter 建立数据，update_counting 每帧推进一次计数。
每个人的临时 ID 对应一份过线进度；预测框不作为新的过线证据。
两个 dataclass 只存数据，所有计算都在普通函数中完成。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from math import hypot, isfinite
from statistics import median
from types import SimpleNamespace
from typing import Any, Deque, Dict, Iterable, Optional, Sequence, Tuple

from app.types import TrackedPerson, bbox_centroid

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
SCREEN_DIRECTIONS = {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"}
ANCHORS = {"auto", "center", "bottom_center"}

ENTER_TO_DIRECTION = {value: key for key, value in DIRECTION_TO_ENTER.items()}


@dataclass(frozen=True)
class CrossingObservation:
    """一帧真实检测留下的位置、框和置信度，用于短暂断轨后的计数接续。"""

    timestamp: float
    point: Point
    bbox: tuple
    confidence: float


@dataclass
class _TrackCrossingState:
    """一个 ID 的过线进度：先记住起始侧，再确认是否到达另一侧。"""
    # +1 和 -1 表示线的两侧，0 表示仍在缓冲带，None 表示起始侧尚未确定。
    confirmed_side: Optional[int] = None
    candidate_side: int = 0
    candidate_frames: int = 0
    last_seen_frame: int = 0
    last_event_frame: int = -1_000_000
    counted_events: set[str] = field(default_factory=set)  # 已计过的 ENTER / EXIT
    last_seen_at: Optional[float] = None
    last_event_at: Optional[float] = None
    candidate_since: Optional[float] = None
    outer_observations: int = 0
    observations: Deque[CrossingObservation] = field(default_factory=lambda: deque(maxlen=12))


def create_counter(config: Dict[str, Any]) -> SimpleNamespace:
    """按配置建立计数数据；counter 只存字段，不绑定算法方法。"""
    counter = SimpleNamespace()
    # 计数规则：线的位置、进入方向、确认距离和最短观测时间。
    counter.direction_mode = _direction_mode_from_config(config)
    counter.enter_direction = DIRECTION_TO_ENTER.get(counter.direction_mode, "negative_to_positive")
    counter.anchor = _normalize_anchor(config.get("anchor", "auto"))
    counter.cooldown_frames = _int_setting(config, "cooldown_frames", 8, minimum=0)
    counter.hysteresis_px = _float_setting(config, "hysteresis_px", 64.0, minimum=1.0)
    counter.rearm_distance_px = max(
        counter.hysteresis_px + 1.0,
        _float_setting(config, "rearm_distance_px", 96.0, minimum=1.0),
    )
    counter.confirmation_frames = _int_setting(config, "confirmation_frames", 5, minimum=1)
    counter.track_state_ttl_frames = _int_setting(config, "track_state_ttl_frames", 120, minimum=1)
    counter.reference_fps = float(config.get("_reference_fps", 30))
    counter.time_confirmation_enabled = _bool_setting(config, "time_confirmation_enabled", True)
    counter.confirmation_max_gap_seconds = _float_setting(
        config, "confirmation_max_gap_seconds", 0.2, minimum=0.01
    )
    counter.state_recovery_enabled = _bool_setting(config, "state_recovery_enabled", True)
    counter.state_recovery_max_gap_seconds = _float_setting(
        config, "state_recovery_max_gap_seconds", 0.6, minimum=0.01
    )
    counter.counting_state_recoveries = 0
    counter.track_state_ttl_seconds = max(
        counter.track_state_ttl_frames / counter.reference_fps,
        float(config.get("_minimum_retention_seconds", 0)),
    )
    counter._last_timestamp: Optional[float] = None
    counter.count_once_per_direction_per_track = _bool_setting(
        config, "count_once_per_direction_per_track", False
    )
    counter._line_arg = config.get("line", "auto")
    counter._line: Optional[Line] = None
    counter.frame_width = int(config.get("_default_frame_width") or 960)
    counter.frame_height = int(config.get("_default_frame_height") or 540)
    # 两个线程会读写计数：视频处理线程更新，网页请求线程读取或修改。
    counter._lock = threading.RLock()
    # 每个 ID 的过线进度，以及最近一次真实观测的位置。
    counter._track_states: Dict[int, _TrackCrossingState] = {}
    counter._latest_track_boxes: Dict[int, tuple] = {}
    counter._track_frame_index = 0
    counter._events: Deque[dict] = deque(maxlen=20)
    # 页面上的人数与事件记录。
    counter.current_occupancy = 0
    counter.total_entered = 0
    counter.total_exited = 0
    counter.active_tracks = 0
    counter.visible_persons = 0
    # API 返回的运行状态和各处理步骤的平均耗时。
    counter.runtime = {
        "fps": 0.0,
        "latency_ms": 0.0,
        "frame_index": 0,
        "camera_status": "offline",
        "last_error": "",
        "source": "",
        "model": "",
        "detector": "",
        "npu_enabled": False,
        "running": False,
        "privacy_mode": True,
        "face_mosaic_enabled": True,
        "available_cameras": [],
        **{key: 0.0 for key in TIMING_KEYS},
    }
    timing_window = max(1, int(config.get("_timing_window_frames") or 30))
    counter._timing_samples: Dict[str, Deque[float]] = {
        key: deque(maxlen=timing_window) for key in TIMING_KEYS
    }
    return counter


def update_counting(
    counter, tracks: Iterable[TrackedPerson], frame_shape: Sequence[int], *,
    timestamp: Optional[float] = None, observed_frame: bool = True,
) -> None:
    """接收本帧轨迹和图像大小，更新 counter 中的人数、事件和过线进度。"""
    track_list = list(tracks)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    with counter._lock:
        if timestamp is not None:
            timestamp = float(timestamp)
            if not isfinite(timestamp) or (
                counter._last_timestamp is not None and timestamp < counter._last_timestamp
            ):
                raise ValueError("Counting timestamps must be finite and monotonic")
            # 同一帧重复到达时不重复采信；长时间没出现的人可以从内存移除。
            if timestamp == counter._last_timestamp:
                observed_frame = False
            counter._last_timestamp = timestamp
            counter._track_states = {
                key: state for key, state in counter._track_states.items()
                if state.last_seen_at is None
                or timestamp - state.last_seen_at <= counter.track_state_ttl_seconds
            }
        line = line_for_frame(counter, width, height)
        counter._track_frame_index += 1
        tracking_frame = counter._track_frame_index
        counter.frame_width = width
        counter.frame_height = height
        counter.active_tracks = len(track_list)
        # 跟踪器预测出来的框不算新的检测，不能用来证明一个人越过了线。
        observed = [track for track in track_list if track.observed] if observed_frame else []
        counter.visible_persons = len(observed)
        if observed_frame:
            counter._latest_track_boxes = {
                track.track_id: tuple(track.observation_bbox or track.bbox) for track in observed
            }
            # 检测中断会打断连续确认，但保留最后已确认的所在侧。
            for track_id, state in counter._track_states.items():
                if track_id not in counter._latest_track_boxes:
                    _clear_candidate(state)
            if timestamp is not None:
                for track in observed:
                    state = counter._track_states.setdefault(track.track_id, _TrackCrossingState())
                    state.observations.append(
                        CrossingObservation(
                            timestamp, tracking_point(counter, counter._latest_track_boxes[track.track_id]),
                            counter._latest_track_boxes[track.track_id], track.confidence,
                        )
                    )
                _recover_crossing_states(counter, timestamp, line)
        for track in observed:
            _update_observed_track(counter, track, line, tracking_frame, timestamp)

        if timestamp is None:
            stale_before = tracking_frame - counter.track_state_ttl_frames
            counter._track_states = {
                track_id: state for track_id, state in counter._track_states.items()
                if state.last_seen_frame >= stale_before
            }


def _update_observed_track(
    counter, track: TrackedPerson, line: Line, tracking_frame: int,
    timestamp: Optional[float],
) -> None:
    """判断一个人是否稳定换侧；确认后才增加进入或离开人数。"""
    point = tracking_point(counter, counter._latest_track_boxes[track.track_id])
    distance = _signed_distance_to_line(point, line)
    # 内边界过滤线附近的抖动；外边界用来确认人确实已经走到另一侧。
    side = _side_outside_margin(distance, counter.hysteresis_px)
    stable_side = _side_outside_margin(distance, counter.rearm_distance_px)
    state = counter._track_states.setdefault(track.track_id, _TrackCrossingState())
    gap = None if timestamp is None or state.last_seen_at is None else timestamp - state.last_seen_at
    state.last_seen_frame = tracking_frame
    state.last_seen_at = timestamp
    if track.recovered or (
        counter.time_confirmation_enabled
        and gap is not None
        and gap > counter.confirmation_max_gap_seconds
    ):
        _clear_candidate(state)
    timed = timestamp is not None and counter.time_confirmation_enabled
    confirmation_side = side if timed or state.confirmed_side is None else stable_side
    if not _advance_candidate(counter, state, confirmation_side, stable_side, timestamp, timed):
        return

    previous = state.confirmed_side
    state.candidate_frames = counter.confirmation_frames
    if previous is None or previous == confirmation_side:
        # 第一次稳定观测只确定起点；留在原侧也不产生事件。
        state.confirmed_side = confirmation_side
        return
    cooling_down = tracking_frame - state.last_event_frame < counter.cooldown_frames
    if timestamp is not None:
        cooling_down = (
            state.last_event_at is not None
            and timestamp - state.last_event_at < counter.cooldown_frames / counter.reference_fps
        )
    if cooling_down:
        return

    state.confirmed_side = confirmation_side
    direction = _direction(previous, confirmation_side)
    event_type = "ENTER" if direction == counter.enter_direction else "EXIT"
    if counter.count_once_per_direction_per_track and event_type in state.counted_events:
        return
    _record_event(counter, track.track_id, event_type)
    state.counted_events.add(event_type)
    state.last_event_frame = tracking_frame
    state.last_event_at = timestamp


def _advance_candidate(
    counter, state: _TrackCrossingState, side: int, stable_side: int,
    timestamp: Optional[float], timed: bool,
) -> bool:
    """累计一个人在候选侧的观测；证据足够时返回 True。"""
    if side == 0:
        _clear_candidate(state)
        return False
    if state.candidate_side != side:
        _clear_candidate(state)
        state.candidate_side = side
        state.candidate_since = timestamp
    state.candidate_frames += 1
    state.outer_observations = state.outer_observations + 1 if stable_side == side else 0
    if timed:
        # 例如 30 fps 下原先要求 5 帧，对应从首帧到末帧经过 4/30 秒。
        # 降帧后仍按这个时间判断，但至少要有数次真实观测，并到达外边界。
        duration = (counter.confirmation_frames - 1) / counter.reference_fps
        confirmed = (
            state.candidate_frames >= min(3, counter.confirmation_frames)
            and state.candidate_since is not None
            and timestamp - state.candidate_since + 1e-9 >= duration
        )
        if state.confirmed_side is not None:
            confirmed = confirmed and state.outer_observations >= min(2, counter.confirmation_frames)
        return confirmed
    return state.candidate_frames >= counter.confirmation_frames


def _clear_candidate(state: _TrackCrossingState) -> None:
    """清空尚未确认的换侧证据，保留这个人原先已经确认的所在侧。"""
    state.candidate_side = 0
    state.candidate_frames = 0
    state.candidate_since = None
    state.outer_observations = 0


def _record_event(counter, track_id: int, event_type: str) -> None:
    """更新累计人数，并把最新事件放在队列开头。"""
    if event_type == "ENTER":
        counter.total_entered += 1
        counter.current_occupancy += 1
    else:
        counter.total_exited += 1
        counter.current_occupancy = max(0, counter.current_occupancy - 1)
    counter._events.appendleft(
        {
            "time": time.strftime("%H:%M:%S"),
            "track_id": int(track_id),
            "event_type": event_type,
            "current_occupancy": counter.current_occupancy,
            "total_entered": counter.total_entered,
            "total_exited": counter.total_exited,
        }
    )


def _recover_crossing_states(counter, timestamp: float, line: Line) -> None:
    """仅在旧轨迹与新片段具有唯一对应证据时，接续旧轨迹的计数进度。"""
    if not counter.state_recovery_enabled:
        return
    visible = set(counter._latest_track_boxes)
    missing_states = {
        track_id: state for track_id, state in counter._track_states.items()
        if track_id not in visible and state.confirmed_side is not None
        and state.last_seen_at is not None
        and 0 < timestamp - state.last_seen_at <= counter.state_recovery_max_gap_seconds + 0.4
    }
    new_states = {
        track_id: counter._track_states[track_id] for track_id in visible
        if counter._track_states[track_id].confirmed_side is None
    }
    costs = {}
    for previous_id, previous in missing_states.items():
        for current_id, current in new_states.items():
            cost = handoff_cost(
                previous.observations, current.observations, previous.confirmed_side,
                line, counter.rearm_distance_px, counter.state_recovery_max_gap_seconds,
            )
            if cost is not None:
                costs[previous_id, current_id] = cost
    for previous_id, current_id in unambiguous_pairs(costs):
        # 新 ID 接过旧 ID 的起始侧和计数记录；这次接续本身不增加人数。
        previous = counter._track_states.pop(previous_id)
        current = counter._track_states[current_id]
        current.confirmed_side = previous.confirmed_side
        current.last_event_at = previous.last_event_at
        current.last_event_frame = previous.last_event_frame
        current.counted_events = previous.counted_events.copy()
        _clear_candidate(current)
        counter.counting_state_recoveries += 1


def reset_counter(counter) -> None:
    """人数清零，并移除旧的过线进度；保留当前计数线和设置。"""
    with counter._lock:
        counter.current_occupancy = 0
        counter.total_entered = 0
        counter.total_exited = 0
        counter._events.clear()
        counter._track_states.clear()
        counter._latest_track_boxes.clear()
        counter._track_frame_index = 0
        counter._last_timestamp = None
        counter.counting_state_recoveries = 0


def configure_counting(
    counter, line: Dict[str, Any], direction: str, *, anchor: Optional[str] = None,
    hysteresis_px: Optional[float] = None, rearm_distance_px: Optional[float] = None,
) -> dict:
    """检查并应用网页提交的设置，返回页面需要显示的当前设置。"""
    line_dict = _coerce_line_dict(line)
    direction_mode = _normalize_direction_mode(direction)
    parsed_line = _parse_line(line_dict, counter.frame_width, counter.frame_height)
    with counter._lock:
        anchor_mode = counter.anchor if anchor is None else _normalize_anchor(anchor)
        inner = counter.hysteresis_px if hysteresis_px is None else float(hysteresis_px)
        outer = counter.rearm_distance_px if rearm_distance_px is None else float(rearm_distance_px)
        if not isfinite(inner) or not isfinite(outer) or inner < 1 or outer <= inner:
            raise ValueError(
                "Confirmation distance must exceed dead zone; both must be finite and at least 1 px"
            )
        direction_mode = _direction_mode_for_line(direction_mode, parsed_line)
        counter._line_arg = line_dict
        counter._line = parsed_line
        counter.direction_mode = direction_mode
        counter.enter_direction = _enter_direction_for_line(direction_mode, parsed_line)
        counter.anchor = anchor_mode
        counter.hysteresis_px = inner
        counter.rearm_distance_px = outer
        counter._track_states = _states_rebased_to_line(counter, parsed_line)
        return counting_config(counter)


def _states_rebased_to_line(counter, line: Line) -> Dict[int, _TrackCrossingState]:
    """换线后按当前人所在的位置重新确定起始侧，避免画线动作被当成过线。"""
    states: Dict[int, _TrackCrossingState] = {}
    for track_id, box in counter._latest_track_boxes.items():
        point = tracking_point(counter, box)
        side = _side_outside_margin(_signed_distance_to_line(point, line), 0.0)
        if side != 0:
            states[track_id] = _TrackCrossingState(
                confirmed_side=side,
                last_seen_frame=counter._track_frame_index,
                last_seen_at=counter._last_timestamp,
            )
    return states


def counting_config(counter) -> dict:
    """返回计数设置；检查图像两侧是否都留有足够的确认空间。"""
    with counter._lock:
        line = line_for_frame(counter, counter.frame_width, counter.frame_height)
        distances = [_signed_distance_to_line(point, line) for point in (
            (0, 0), (counter.frame_width, 0), (0, counter.frame_height),
            (counter.frame_width, counter.frame_height),
        )]
        warnings = []
        if min(distances) >= -counter.rearm_distance_px or max(distances) <= counter.rearm_distance_px:
            warnings.append("One confirmation side is outside the image")
        return {
            "line": _line_as_dict(line),
            "direction": counter.direction_mode,
            "anchor": counter.anchor,
            "effective_anchor": effective_anchor(counter),
            "warnings": warnings,
            "cooldown_frames": counter.cooldown_frames,
            "hysteresis_px": _clean_number(counter.hysteresis_px),
            "rearm_distance_px": _clean_number(counter.rearm_distance_px),
            "confirmation_frames": counter.confirmation_frames,
            "time_confirmation_enabled": counter.time_confirmation_enabled,
            "confirmation_max_gap_seconds": counter.confirmation_max_gap_seconds,
            "state_recovery_enabled": counter.state_recovery_enabled,
            "state_recovery_max_gap_seconds": counter.state_recovery_max_gap_seconds,
            "track_state_ttl_frames": counter.track_state_ttl_frames,
            "count_once_per_direction_per_track": counter.count_once_per_direction_per_track,
            "frame_size": {"width": counter.frame_width, "height": counter.frame_height},
        }


def counter_snapshot(counter) -> dict:
    """一次读出人数、事件和运行状态，供网页的状态 API 返回。"""
    with counter._lock:
        data = {
            **counter.runtime,
            "current_occupancy": counter.current_occupancy,
            "total_entered": counter.total_entered,
            "total_exited": counter.total_exited,
            "active_tracks": counter.active_tracks,
            "counting_state_recoveries": counter.counting_state_recoveries,
            "visible_persons": counter.visible_persons,
            "line": _line_as_list(counter._line),
            "recent_events": list(counter._events),
        }
        if not data["total_latency_ms"]:
            data["total_latency_ms"] = data["latency_ms"]
        return data


def update_runtime_metrics(
    counter,
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
    """接收主循环的运行指标；各步骤耗时取最近若干帧的平均值。"""
    with counter._lock:
        runtime = counter.runtime
        runtime["fps"] = round(float(fps), 2)
        runtime["latency_ms"] = round(float(latency_ms), 1)
        runtime["frame_index"] = int(frame_index)
        runtime["camera_status"] = camera_status
        runtime["source"] = source
        runtime["model"] = model
        runtime["detector"] = detector
        runtime["npu_enabled"] = bool(npu_enabled)
        runtime["running"] = running
        runtime["privacy_mode"] = privacy_mode
        runtime["face_mosaic_enabled"] = face_mosaic_enabled
        runtime["last_error"] = last_error
        runtime["available_cameras"] = available_cameras
        if timings is not None:
            for key in TIMING_KEYS:
                counter._timing_samples[key].append(float(timings.get(key, 0.0) or 0.0))
                samples = counter._timing_samples[key]
                runtime[key] = round(sum(samples) / len(samples), 1)
            runtime["latency_ms"] = runtime["total_latency_ms"] or runtime["latency_ms"]


def line_for_frame(counter, width: int, height: int) -> Line:
    """第一次使用时把配置转换为线坐标；auto 表示图像中央的竖线。"""
    if counter._line is None:
        counter._line = _parse_line(counter._line_arg, width, height)
        counter.direction_mode = _direction_mode_for_line(counter.direction_mode, counter._line)
        counter.enter_direction = _enter_direction_for_line(counter.direction_mode, counter._line)
    return counter._line


def current_line(counter) -> Optional[Line]:
    """返回已经确定的线坐标；尚未初始化时返回 None。"""
    return counter._line


def effective_anchor(counter) -> str:
    """决定用框的哪个点计数：auto 在横线上用脚点，在竖线上用中心。"""
    if counter.anchor != "auto":
        return counter.anchor
    line = counter._line or _parse_line(counter._line_arg, counter.frame_width, counter.frame_height)
    return "bottom_center" if _horizontal_line(line) else "center"


def tracking_point(counter, bbox: Sequence[float]) -> Point:
    """从一个检测框取出计数用的点，结果为 (x, y)。"""
    if effective_anchor(counter) == "bottom_center":
        return (bbox[0] + bbox[2]) / 2.0, bbox[3]
    return bbox_centroid(bbox)


def _velocity(samples: Sequence[CrossingObservation]) -> Optional[Point]:
    """取多个观测对的速度中位数，避免单次检测抖动决定接续方向。"""
    velocities = []
    for first, last in combinations(samples, 2):
        elapsed = last.timestamp - first.timestamp
        if 0.04 <= elapsed <= 0.5:
            velocities.append(((last.point[0] - first.point[0]) / elapsed,
                               (last.point[1] - first.point[1]) / elapsed))
    if len(velocities) < 2:
        return None
    return median(v[0] for v in velocities), median(v[1] for v in velocities)


def _iou(first: tuple, second: tuple) -> float:
    """两个框的交并比：重叠面积占合并面积的比例。"""
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def handoff_cost(
    old: Sequence[CrossingObservation], new: Sequence[CrossingObservation],
    side: int, line: Line, margin: float, max_gap: float,
) -> Optional[float]:
    """判断新旧片段能否接续；证据不足返回 None，合格时分数越低越可信。"""
    # 低分框仍可用于跟踪，但不足以证明两个不同 ID 应共享计数状态。
    old = [sample for sample in old if sample.confidence >= 0.35]
    new = [sample for sample in new if sample.confidence >= 0.35]
    if len(old) < 3 or len(new) < 3:
        return None
    last, first = old[-1], new[0]
    gap = first.timestamp - last.timestamp
    if not 0 < gap <= max_gap or new[-1].timestamp - first.timestamp > 0.4:
        return None
    old = [sample for sample in old if last.timestamp - sample.timestamp <= 0.4]
    old_velocity, new_velocity = _velocity(old), _velocity(new)
    if old_velocity is None or new_velocity is None:
        return None

    (x1, y1), (x2, y2) = line
    length = hypot(x2 - x1, y2 - y1)
    normal = (-(y2 - y1) / length, (x2 - x1) / length)

    def distance(point: Point) -> float:
        return normal[0] * (point[0] - x1) + normal[1] * (point[1] - y1)

    old_height, new_height = last.bbox[3] - last.bbox[1], first.bbox[3] - first.bbox[1]
    old_width, new_width = last.bbox[2] - last.bbox[0], first.bbox[2] - first.bbox[0]
    if (min(old_height, new_height, old_width, new_width) <= 0
            or not 0.6 <= new_height / old_height <= 1.67
            or not 0.4 <= new_width / old_width <= 2.5):
        return None
    scale = max(old_height, new_height)
    if max(abs(distance(last.point)), abs(distance(first.point))) > 2 * margin:
        return None

    # 两段都必须从旧的已确认侧朝线移动；反向或速度异常的片段不接续。
    for velocity in (old_velocity, new_velocity):
        normal_speed = velocity[0] * normal[0] + velocity[1] * normal[1]
        if side * normal_speed >= -0.08 * scale or hypot(*velocity) > 6 * scale:
            return None
    if side * (distance(first.point) - distance(last.point)) > 0.15 * scale:
        return None

    # 用旧速度预测新框的位置，同时检查中心误差与框重叠，防止串到邻近的人。
    shift = (old_velocity[0] * gap, old_velocity[1] * gap)
    predicted = tuple(last.bbox[index] + shift[index % 2] for index in range(4))
    error = hypot(first.point[0] - last.point[0] - shift[0],
                  first.point[1] - last.point[1] - shift[1]) / scale
    overlap = _iou(predicted, first.bbox)
    if error > 0.65 or overlap < 0.1:
        return None
    return error + 0.25 * (1 - overlap) + 0.1 * gap


def unambiguous_pairs(costs: dict, separation: float = 0.15) -> list:
    """只接续相互最匹配、且明显优于所有竞争者的一对，避免一人多领或多人抢领。"""
    accepted = []
    for (old, new), cost in costs.items():
        alternatives = [value for (other_old, other_new), value in costs.items()
                        if (other_old == old or other_new == new) and (other_old, other_new) != (old, new)]
        if all(other - cost >= separation for other in alternatives):
            accepted.append((old, new))
    return accepted


def _parse_line(value: Any, width: int, height: int) -> Line:
    if not value or str(value).lower() == "auto":
        x = width / 2.0
        return (x, 0.0), (x, float(height))
    if isinstance(value, dict):
        parts = [float(value[key]) for key in ("x1", "y1", "x2", "y2")]
    elif isinstance(value, list) and len(value) == 4:
        parts = [float(item) for item in value]
    else:
        parts = [float(item.strip()) for item in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("counting.line must be auto or x1,y1,x2,y2")
    return (parts[0], parts[1]), (parts[2], parts[3])


def _signed_distance_to_line(point: Point, line: Line) -> float:
    """点到直线的垂直距离；正负号区分两侧，绝对值的单位是像素。"""
    (x1, y1), (x2, y2) = line
    px, py = point
    length = hypot(x2 - x1, y2 - y1)
    if length <= 0.0:
        return 0.0
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return cross / length


def _side_outside_margin(distance: float, margin: float) -> int:
    """距离超过缓冲带才返回 +1 或 -1；仍在带内则返回 0。"""
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
    if not all(isfinite(number) for number in line.values()):
        raise ValueError("counting line coordinates must be finite")
    line = {key: _clean_number(number) for key, number in line.items()}
    if line["x1"] == line["x2"] and line["y1"] == line["y2"]:
        raise ValueError("counting line start and end points must be different")
    return line


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
    if direction not in SCREEN_DIRECTIONS:
        raise ValueError("direction must be left_to_right, right_to_left, top_to_bottom or bottom_to_top")
    return direction


def _normalize_anchor(value: str) -> str:
    anchor = str(value).strip().lower()
    if anchor not in ANCHORS:
        raise ValueError("anchor must be auto, center or bottom_center")
    return anchor


def _horizontal_line(line: Line) -> bool:
    (x1, y1), (x2, y2) = line
    return abs(x2 - x1) > abs(y2 - y1)


def _direction_mode_for_line(mode: str, line: Line) -> str:
    forward = mode in {"left_to_right", "top_to_bottom"}
    if _horizontal_line(line):
        return "top_to_bottom" if forward else "bottom_to_top"
    return "left_to_right" if forward else "right_to_left"


def _enter_direction_for_line(mode: str, line: Line) -> str:
    # 屏幕上的进入方向不随画线时点击端点的先后顺序改变。
    (x1, y1), (x2, y2) = line
    change = {
        "left_to_right": -(y2 - y1), "right_to_left": y2 - y1,
        "top_to_bottom": x2 - x1, "bottom_to_top": -(x2 - x1),
    }[mode]
    return "negative_to_positive" if change > 0 else "positive_to_negative"


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
