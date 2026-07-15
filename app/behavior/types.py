"""Typed behavior observations, triggers, and unified event records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from uuid import uuid4

from vision.types import BBox


@dataclass(frozen=True)
class BehaviorSignal:
    event_type: str
    track_id: int
    observed: bool
    confidence: float
    person_bbox: BBox
    object_bboxes: Dict[str, BBox] = field(default_factory=dict)
    evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BehaviorTrigger:
    event_type: str
    track_id: int
    confidence: float
    duration_ms: int
    person_bbox: BBox
    object_bboxes: Dict[str, BBox]
    evidence: Tuple[str, ...]


@dataclass(frozen=True)
class BehaviorEvent:
    event_id: str
    event_type: str
    camera_id: str
    timestamp: str
    track_id: int
    confidence: float
    duration_ms: int
    bboxes: Dict[str, BBox]
    evidence: Tuple[str, ...]
    snapshot_path: Optional[str] = None
    annotated_snapshot_path: Optional[str] = None
    video_clip_path: Optional[str] = None

    @classmethod
    def from_trigger(cls, camera_id: str, trigger: BehaviorTrigger) -> "BehaviorEvent":
        bboxes = {"person": trigger.person_bbox, **trigger.object_bboxes}
        return cls(
            event_id=str(uuid4()),
            event_type=trigger.event_type,
            camera_id=camera_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            track_id=trigger.track_id,
            confidence=trigger.confidence,
            duration_ms=trigger.duration_ms,
            bboxes=bboxes,
            evidence=trigger.evidence,
        )

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "track_id": int(self.track_id),
            "confidence": round(float(self.confidence), 4),
            "duration_ms": int(self.duration_ms),
            "bboxes": {key: [float(value) for value in box] for key, box in self.bboxes.items()},
            "snapshot_path": self.snapshot_path,
            "annotated_snapshot_path": self.annotated_snapshot_path,
            "video_clip_path": self.video_clip_path,
            "evidence": list(self.evidence),
        }
