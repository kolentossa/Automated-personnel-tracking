"""Camera source adapters."""

from .base import Camera
from .rtsp import RTSPCamera
from .video_file import VideoFileCamera
from .webcam import WebCamera

__all__ = ["Camera", "RTSPCamera", "VideoFileCamera", "WebCamera"]
