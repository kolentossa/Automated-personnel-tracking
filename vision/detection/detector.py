"""Detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from vision.types import Detection, Frame


class Detector(ABC):
    """Return anonymous person detections for one frame."""

    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        """Detect people in a frame."""
