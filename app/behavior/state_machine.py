"""Reusable temporal debounce, deduplication, and cooldown state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from app.behavior.types import BehaviorSignal, BehaviorTrigger


@dataclass(frozen=True)
class BehaviorRule:
    duration_ms: int
    min_consecutive_frames: int
    confidence_threshold: float
    cooldown_ms: int
    max_gap_frames: int = 2


@dataclass
class _TemporalState:
    first_seen_ms: float
    last_seen_ms: float
    last_update_ms: float
    evidence_frames: int = 0
    missed_frames: int = 0
    confidence_sum: float = 0.0
    triggered: bool = False
    latest_signal: Optional[BehaviorSignal] = None


class TemporalEventStateMachine:
    """Emit once per continuous behavior and retain cooldown across resets."""

    def __init__(self) -> None:
        self._states: Dict[Tuple[int, str], _TemporalState] = {}
        self._last_alert_ms: Dict[Tuple[int, str], float] = {}

    def update(self, signal: BehaviorSignal, rule: BehaviorRule, now_ms: float) -> Optional[BehaviorTrigger]:
        key = (int(signal.track_id), signal.event_type)
        state = self._states.get(key)
        if not signal.observed:
            if state is not None:
                state.missed_frames += 1
                state.last_update_ms = now_ms
                if state.missed_frames > rule.max_gap_frames:
                    del self._states[key]
            return None

        if state is None:
            state = _TemporalState(first_seen_ms=now_ms, last_seen_ms=now_ms, last_update_ms=now_ms)
            self._states[key] = state
        state.last_seen_ms = now_ms
        state.last_update_ms = now_ms
        state.missed_frames = 0
        state.evidence_frames += 1
        state.confidence_sum += max(0.0, min(1.0, float(signal.confidence)))
        state.latest_signal = signal

        duration_ms = max(0, int(round(now_ms - state.first_seen_ms)))
        confidence = state.confidence_sum / max(1, state.evidence_frames)
        last_alert = self._last_alert_ms.get(key)
        cooldown_ready = last_alert is None or now_ms - last_alert >= rule.cooldown_ms
        ready = (
            not state.triggered
            and cooldown_ready
            and duration_ms >= rule.duration_ms
            and state.evidence_frames >= rule.min_consecutive_frames
            and confidence >= rule.confidence_threshold
        )
        if not ready:
            return None

        state.triggered = True
        self._last_alert_ms[key] = now_ms
        latest = state.latest_signal or signal
        return BehaviorTrigger(
            event_type=latest.event_type,
            track_id=latest.track_id,
            confidence=confidence,
            duration_ms=duration_ms,
            person_bbox=latest.person_bbox,
            object_bboxes=dict(latest.object_bboxes),
            evidence=latest.evidence,
        )

    def cleanup(self, active_track_ids: Iterable[int], now_ms: float, stale_track_ms: int) -> None:
        active = {int(track_id) for track_id in active_track_ids}
        stale_keys = [
            key
            for key, state in self._states.items()
            if key[0] not in active and now_ms - state.last_update_ms >= stale_track_ms
        ]
        for key in stale_keys:
            self._states.pop(key, None)
        old_alerts = [key for key, value in self._last_alert_ms.items() if now_ms - value >= stale_track_ms * 4]
        for key in old_alerts:
            self._last_alert_ms.pop(key, None)

    def reset(self) -> None:
        self._states.clear()
        self._last_alert_ms.clear()

    def snapshot(self) -> dict:
        active = []
        for (track_id, event_type), state in sorted(self._states.items()):
            active.append({
                "track_id": track_id,
                "event_type": event_type,
                "evidence_frames": state.evidence_frames,
                "missed_frames": state.missed_frames,
                "triggered": state.triggered,
                "duration_ms": max(0, int(state.last_seen_ms - state.first_seen_ms)),
            })
        return {"active_candidates": active, "active_candidate_count": len(active)}


def rule_from_config(config: dict) -> BehaviorRule:
    return BehaviorRule(
        duration_ms=max(0, int(config.get("duration_ms") or 0)),
        min_consecutive_frames=max(1, int(config.get("min_consecutive_frames") or 1)),
        confidence_threshold=max(0.0, min(1.0, float(config.get("confidence_threshold") or 0.0))),
        cooldown_ms=max(0, int(config.get("cooldown_ms") or 0)),
        max_gap_frames=max(0, int(config.get("max_gap_frames") or 0)),
    )
