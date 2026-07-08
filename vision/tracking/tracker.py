"""Tracker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from vision.types import Detection, TrackedPerson


class Tracker(ABC):
    """Assign stable temporary ids to detections across frames."""

    @abstractmethod
    def update(self, detections: Iterable[Detection]) -> List[TrackedPerson]:
        """Update tracks from detections and return visible tracked people."""
