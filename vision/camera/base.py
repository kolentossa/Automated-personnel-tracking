"""Abstract camera interface used by all downstream pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from vision.types import Frame


class Camera(ABC):
    """Minimal frame source contract for files, USB cameras, and RTSP streams."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame, or None when no frame is available."""

    @abstractmethod
    def release(self) -> None:
        """Release the underlying capture resource."""

    @property
    @abstractmethod
    def status(self) -> str:
        """Return a simple status string such as online, offline, or ended."""

    @property
    def fps(self) -> float:
        return 0.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (0, 0)
