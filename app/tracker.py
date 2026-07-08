"""Tracking adapter around the existing lightweight IoU tracker."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from vision.tracking.bytetrack import ByteTrackTracker
from vision.types import Detection, TrackedPerson


class PersonTracker:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._tracker = ByteTrackTracker(
            iou_threshold=float(config.get("iou_threshold") or 0.3),
            max_missing=int(config.get("max_missing") or 20),
            min_confidence=float(config.get("min_confidence") or 0.15),
        )

    def update(self, detections: Iterable[Detection]) -> List[TrackedPerson]:
        return self._tracker.update(detections)