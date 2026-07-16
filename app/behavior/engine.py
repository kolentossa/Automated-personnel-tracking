"""Compose behavior association and independent temporal state machines."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Sequence

from app.behavior.phone import PhoneBehaviorStateMachine
from app.behavior.smoking import SmokingBehaviorStateMachine
from app.behavior.types import BehaviorTrigger
from vision.types import Detection, TrackedPerson


@dataclass(frozen=True)
class BehaviorResult:
    triggers: List[BehaviorTrigger]
    analysis_ms: float


class BehaviorEngine:
    def __init__(self, config: dict) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        groups = self.config.get("class_groups", {})
        association = self.config.get("association", {})
        self.phone = PhoneBehaviorStateMachine(
            self.config.get("phone", {}), groups, association
        )
        self.smoking = SmokingBehaviorStateMachine(
            self.config.get("smoking", {}), groups, association
        )
        self._lock = threading.RLock()
        self._counts: Counter[str] = Counter()
        self._recent: Deque[dict] = deque(maxlen=20)
        self._analysis_ms = 0.0

    def update(
        self,
        tracks: Iterable[TrackedPerson],
        detections: Iterable[Detection],
        frame_shape: Sequence[int],
        now_ms: float | None = None,
        frame_index: int = 0,
        behavior_model_fresh: bool = True,
    ) -> BehaviorResult:
        started = time.monotonic()
        if not self.enabled:
            return BehaviorResult([], 0.0)
        timestamp_ms = float(
            now_ms if now_ms is not None else time.monotonic() * 1000.0
        )
        people = list(tracks)
        objects = list(detections)
        triggers = self.phone.update(people, objects, frame_shape, timestamp_ms)
        triggers.extend(
            self.smoking.update(
                people,
                objects,
                frame_shape,
                timestamp_ms,
                frame_index,
                behavior_model_fresh,
            )
        )
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
        with self._lock:
            self._analysis_ms = elapsed_ms
            for trigger in triggers:
                self._counts[trigger.event_type] += 1
                self._recent.appendleft(
                    {
                        "event_type": trigger.event_type,
                        "track_id": trigger.track_id,
                        "confidence": round(trigger.confidence, 3),
                        "duration_ms": trigger.duration_ms,
                    }
                )
        return BehaviorResult(triggers, elapsed_ms)

    def reset(self) -> None:
        self.phone.reset()
        self.smoking.reset()
        with self._lock:
            self._counts.clear()
            self._recent.clear()

    def snapshot(self) -> dict:
        with self._lock:
            phone = self.phone.snapshot()
            smoking = self.smoking.snapshot()
            result = {
                "behavior_detection_enabled": self.enabled,
                "behavior_analysis_ms": self._analysis_ms,
                "behavior_event_counts": dict(self._counts),
                "recent_behavior_triggers": list(self._recent),
                "phone_active_candidates": phone["active_candidates"],
                "phone_active_candidate_count": phone["active_candidate_count"],
                "phone_behavior_enabled": phone["phone_behavior_enabled"],
                "smoking_active_candidates": smoking["active_candidates"],
                "smoking_active_candidate_count": smoking["active_candidate_count"],
                "smoking_behavior_enabled": smoking["smoking_behavior_enabled"],
            }
            for key in (
                "raw_cigarette_candidates",
                "verified_cigarette_candidates",
                "cigarette_filter_reasons",
                "cigarette_candidate_states",
            ):
                result[key] = smoking.get(
                    key, {} if key.endswith(("reasons", "states")) else 0
                )
            return result
