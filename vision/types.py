"""Shared typed objects used by the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

BBox = Tuple[float, float, float, float]
Frame = Any


@dataclass(frozen=True)
class Detection:
    """One anonymous person detection in image coordinates."""

    bbox: BBox
    confidence: float
    class_id: int = 0
    label: str = "person"

    def as_dict(self) -> Dict[str, object]:
        return {
            "bbox": [float(v) for v in self.bbox],
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "label": self.label,
        }


@dataclass(frozen=True)
class TrackedPerson:
    """A tracked anonymous person, identified only by a temporary track id."""

    track_id: int
    bbox: BBox
    confidence: float

    @property
    def id(self) -> int:
        return self.track_id

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": int(self.track_id),
            "bbox": [float(v) for v in self.bbox],
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class CountingEvent:
    """Anonymous entry/exit event safe to persist locally."""

    timestamp: str
    event_type: str
    tracking_id: int

    @classmethod
    def now(cls, event_type: str, tracking_id: int) -> "CountingEvent":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            tracking_id=tracking_id,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "tracking_id": int(self.tracking_id),
        }


def bbox_centroid(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
