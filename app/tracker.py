"""用 ByteTrack 关联相邻帧的人体框，按真实时间处理丢失和短时预测。"""

from __future__ import annotations

from collections import deque
from math import isfinite
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from app.bytetrack.core import ByteTrack
from app.bytetrack.single_object_track import TrackState
from app.types import Detection, TrackedPerson


def create_tracker(config: dict) -> SimpleNamespace:
    """保存阈值和跟踪状态，创建第三方 ByteTrack 跟踪器。"""
    tracker = SimpleNamespace(
        name="bytetrack",
        high_threshold=float(config.get("track_high_thresh", 0.35)),
        low_threshold=float(config.get("track_low_thresh", 0.15)),
        new_threshold=float(config.get("new_track_thresh", config.get("track_high_thresh", 0.35))),
        match_threshold=float(config.get("match_thresh", 0.8)),
        buffer_seconds=float(config.get("lost_track_buffer_seconds", 4.0)),
        prediction_seconds=float(config.get("prediction_display_seconds", 0.15)),
        reference_fps=float(config.get("reference_fps", 30)),
        last_seen={},
        last_observation=None,
        timestamp=None,
        observation_times=deque(maxlen=60),
        recovered_tracks=0,
        low_score_matches=0,
        observed_tracks=0,
    )
    values = (
        tracker.high_threshold, tracker.low_threshold, tracker.new_threshold,
        tracker.match_threshold, tracker.buffer_seconds, tracker.prediction_seconds,
        tracker.reference_fps,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("ByteTrack settings must be finite")
    if not 0.1 <= tracker.low_threshold < tracker.high_threshold <= tracker.new_threshold <= 1:
        raise ValueError("ByteTrack requires 0.1 <= low < high <= new <= 1")
    if not 0 < tracker.match_threshold <= 1 or tracker.buffer_seconds <= 0 or tracker.reference_fps <= 0:
        raise ValueError("Invalid ByteTrack matching, buffer or frame rate")
    if not 0 <= tracker.prediction_seconds <= tracker.buffer_seconds:
        raise ValueError("Prediction display time must not exceed the lost-track buffer")
    tracker.core = ByteTrack(
        track_activation_threshold=tracker.high_threshold,
        minimum_matching_threshold=tracker.match_threshold,
        minimum_consecutive_frames=1,
    )
    tracker.core.det_thresh = np.float32(tracker.new_threshold)
    # 本模块按真实时间清理轨迹，关闭 ByteTrack 按处理帧数进行的清理。
    tracker.core.max_time_lost = 1_000_000_000
    return tracker


def track_people(
    tracker: SimpleNamespace,
    detections: Iterable[Detection],
    *,
    timestamp: float | None = None,
    observed_frame: bool = True,
) -> list[TrackedPerson]:
    """筛选检测框 → 更新身份 → 补充短时预测，最后按人员编号返回。"""
    if timestamp is not None:
        now = float(timestamp)
    elif tracker.timestamp is None:
        now = 0.0
    else:
        now = tracker.timestamp + 1 / tracker.reference_fps
    if not isfinite(now) or (tracker.timestamp is not None and now < tracker.timestamp):
        raise ValueError("Tracker timestamps must be finite and monotonic")
    if tracker.timestamp is not None and now == tracker.timestamp:
        observed_frame = False
    tracker.timestamp = now
    _expire_tracks(tracker, now)

    output = []
    if observed_frame:
        rows = []
        for detection in detections:
            box = np.asarray(detection.bbox, dtype=np.float64)
            if detection.label != "person" or detection.class_id != 0 or len(box) != 4:
                continue
            if not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            if not isfinite(detection.confidence) or not tracker.low_threshold <= detection.confidence <= 1:
                continue
            rows.append([*box, detection.confidence])
        tensors = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
        if tracker.last_observation is None:
            elapsed_frames = 1.0
        else:
            elapsed_frames = max(0.001, (now - tracker.last_observation) * tracker.reference_fps)
        _set_prediction_interval(tracker, elapsed_frames)
        lost_ids = {track.internal_track_id for track in tracker.core.lost_tracks}
        tracks = tracker.core.update_with_tensors(tensors)
        tracker.last_observation = now
        tracker.observation_times.append(now)

        for track in tracker.core.tracked_tracks:
            if track.frame_id == tracker.core.frame_id:
                tracker.last_seen[track.internal_track_id] = now
        for track in tracks:
            if track.frame_id != tracker.core.frame_id or track.external_track_id < 0:
                continue
            recovered = track.internal_track_id in lost_ids
            tracker.recovered_tracks += int(recovered)
            tracker.low_score_matches += int(track.score < tracker.high_threshold)
            output.append(TrackedPerson(
                track_id=track.external_track_id,
                bbox=tuple(float(value) for value in track.tlbr),
                confidence=float(track.score),
                observed=True,
                observation_bbox=tuple(float(value) for value in track.observed_tlbr),
                recovered=recovered,
            ))
        alive = {track.internal_track_id for track in tracker.core.tracked_tracks + tracker.core.lost_tracks}
        tracker.last_seen = {key: value for key, value in tracker.last_seen.items() if key in alive}

    tracker.observed_tracks = len(output)
    observed_ids = {track.track_id for track in output}
    output.extend(_predict_missing_people(tracker, now, observed_ids))
    return sorted(output, key=lambda track: track.track_id)


def tracker_snapshot(tracker: SimpleNamespace) -> dict:
    times = tracker.observation_times
    if len(times) > 1 and times[-1] > times[0]:
        rate = (len(times) - 1) / (times[-1] - times[0])
    else:
        rate = 0.0
    return {
        "tracker": tracker.name,
        "tracking_observed_tracks": tracker.observed_tracks,
        "tracking_lost_tracks": len(tracker.core.lost_tracks),
        "tracking_recovered_total": tracker.recovered_tracks,
        "tracking_low_score_matches": tracker.low_score_matches,
        "tracking_observation_fps": round(rate, 2),
    }


def detector_config_for_tracking(detector_config: dict, tracking_config: dict) -> dict:
    """保留低分人体框，让 ByteTrack 用它们继续匹配已有轨迹。"""
    result = dict(detector_config)
    result["confidence_threshold"] = min(
        float(result.get("confidence_threshold", 0.35)),
        float(tracking_config.get("track_low_thresh", 0.15)),
    )
    return result


def _expire_tracks(tracker: SimpleNamespace, now: float) -> None:
    for name in ("tracked_tracks", "lost_tracks"):
        retained = []
        for track in getattr(tracker.core, name):
            last_seen = tracker.last_seen.get(track.internal_track_id, now)
            if now - last_seen > tracker.buffer_seconds:
                track.state = TrackState.Removed
                tracker.last_seen.pop(track.internal_track_id, None)
            else:
                retained.append(track)
        setattr(tracker.core, name, retained)


def _set_prediction_interval(tracker: SimpleNamespace, elapsed_frames: float) -> None:
    # Kalman 预测使用实际间隔：间隔越长，按速度推算的位移越大。
    for kalman in (tracker.core.shared_kalman, tracker.core.kalman_filter):
        kalman._motion_mat[np.arange(4), np.arange(4) + 4] = elapsed_frames


def _predict_missing_people(tracker: SimpleNamespace, now: float, observed_ids: set[int]) -> list[TrackedPerson]:
    """暂时没检测到的人可以短时显示预测框；预测框不提供计数证据。"""
    if tracker.last_observation is None:
        return []
    result = []
    elapsed_frames = max(0.0, (now - tracker.last_observation) * tracker.reference_fps)
    _set_prediction_interval(tracker, elapsed_frames)
    for track in tracker.core.tracked_tracks + tracker.core.lost_tracks:
        if not track.is_activated or track.external_track_id < 0 or track.external_track_id in observed_ids:
            continue
        if now - tracker.last_seen.get(track.internal_track_id, -float("inf")) > tracker.prediction_seconds:
            continue
        mean = track.mean.copy()
        if elapsed_frames:
            if track.state != TrackState.Tracked:
                mean[7] = 0
            mean, _ = tracker.core.shared_kalman.predict(mean, track.covariance)
        width, height = mean[2] * mean[3], mean[3]
        box = (
            mean[0] - width / 2, mean[1] - height / 2,
            mean[0] + width / 2, mean[1] + height / 2,
        )
        if width > 0 and height > 0 and all(isfinite(value) for value in box):
            result.append(TrackedPerson(
                track_id=track.external_track_id,
                bbox=tuple(float(value) for value in box),
                confidence=float(track.score),
                observed=False,
            ))
    return result
