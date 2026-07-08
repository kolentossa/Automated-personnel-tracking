"""USB webcam source for future hardware use."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2

from vision.camera.base import Camera
from vision.types import Frame


class WebCamera(Camera):
    """Read frames from a local USB camera device index."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._capture = cv2.VideoCapture(device_index)
        self._status = "online" if self._capture.isOpened() else "offline"

    def read(self) -> Optional[Frame]:
        if not self._capture.isOpened():
            self._status = "offline"
            return None
        ok, frame = self._capture.read()
        self._status = "online" if ok else "offline"
        return frame if ok else None

    def release(self) -> None:
        self._capture.release()
        self._status = "offline"

    @property
    def status(self) -> str:
        return self._status

    @property
    def fps(self) -> float:
        return float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def frame_size(self) -> Tuple[int, int]:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (width, height)
