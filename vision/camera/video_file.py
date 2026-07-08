"""Video file source used by the first hardware-free demo."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2

from vision.camera.base import Camera
from vision.types import Frame


class VideoFileCamera(Camera):
    """Read frames from a local video file with optional looping for demos."""

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.loop = loop
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video file: {self.path}")
        self._status = "online"

    def read(self) -> Optional[Frame]:
        ok, frame = self._capture.read()
        if ok:
            self._status = "online"
            return frame
        if self.loop:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok:
                self._status = "online"
                return frame
        self._status = "ended"
        return None

    def release(self) -> None:
        self._capture.release()
        self._status = "offline"

    @property
    def status(self) -> str:
        return self._status

    @property
    def fps(self) -> float:
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0 else 25.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (width, height)
