"""OpenCV/V4L2 camera access with automatic device probing."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from app.config import project_path


@dataclass(frozen=True)
class CameraFrame:
    frame: Any
    captured_at: float
    source: str


class CameraSource:
    """Read frames from a V4L2 camera, with a video-file interface reserved."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.source_type = str(config.get("source_type", "camera"))
        self.camera_device = str(config.get("camera_device") or "/dev/video0")
        self.auto_detect = bool(config.get("auto_detect", True))
        self.width = int(config.get("width") or 0)
        self.height = int(config.get("height") or 0)
        self.target_fps = float(config.get("fps") or 0)
        self.retry_interval_sec = float(config.get("retry_interval_sec") or 3.0)
        self.video_file = str(config.get("video_file") or "data/sample.mp4")
        self.status = "offline"
        self.error = ""
        self.selected_source = ""
        self.available_devices: List[Dict[str, Any]] = []
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_open_attempt = 0.0

    def read(self) -> Optional[CameraFrame]:
        if self._capture is None:
            now = time.monotonic()
            if now - self._last_open_attempt < self.retry_interval_sec:
                return None
            if not self.open():
                return None
        assert self._capture is not None
        captured_at = time.monotonic()
        ok, frame = self._capture.read()
        if ok and frame is not None:
            self.status = "online"
            self.error = ""
            return CameraFrame(frame=frame, captured_at=captured_at, source=self.selected_source)
        if self.source_type == "video" and self._capture is not None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok and frame is not None:
                self.status = "online"
                self.error = ""
                return CameraFrame(frame=frame, captured_at=captured_at, source=self.selected_source)
        self.status = "error"
        self.error = f"Could not read a frame from {self.selected_source or self.camera_device}"
        self.release()
        return None

    def open(self) -> bool:
        self.release()
        self._last_open_attempt = time.monotonic()
        if self.source_type == "video":
            return self._open_video_file()
        return self._open_camera()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        if self.status == "online":
            self.status = "offline"

    def _open_video_file(self) -> bool:
        path = project_path(self.video_file)
        self.selected_source = str(path)
        if not path.exists():
            self.status = "error"
            self.error = f"Video file does not exist: {path}"
            return False
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            self.status = "error"
            self.error = f"OpenCV could not open video file: {path}"
            return False
        self._capture = capture
        self.status = "online"
        self.error = ""
        return True

    def _open_camera(self) -> bool:
        candidates = self._candidate_devices()
        self.available_devices = []
        for device in candidates:
            probe = probe_video_device(device, self.width, self.height, self.target_fps)
            self.available_devices.append(probe)
            if not probe.get("readable"):
                continue
            capture = open_v4l2_capture(device, self.width, self.height, self.target_fps)
            if capture is None:
                continue
            self._capture = capture
            self.selected_source = device
            self.status = "online"
            self.error = ""
            return True
        self.selected_source = candidates[0] if candidates else self.camera_device
        self.status = "error"
        probed = ", ".join(candidates) if candidates else "no /dev/video* devices"
        self.error = (
            "No readable OpenCV/V4L2 camera device found. "
            f"Configured={self.camera_device}; probed={probed}. "
            "If v4l2-ctl can stream but OpenCV cannot, this RKISP device may need "
            "a board-specific GStreamer/RKISP adapter."
        )
        return False

    def _candidate_devices(self) -> List[str]:
        candidates: List[str] = []
        if self.camera_device:
            candidates.append(self.camera_device)
        if self.auto_detect:
            if Path("/dev/video-camera0").exists():
                candidates.append("/dev/video-camera0")
            candidates.extend(
                str(path)
                for path in sorted(Path("/dev").glob("video*"), key=_natural_video_key)
                if re.fullmatch(r"video\d+", path.name)
            )
        unique: List[str] = []
        for candidate in candidates:
            if candidate not in unique and Path(candidate).exists():
                unique.append(candidate)
        return unique


def probe_video_device(device: str, width: int = 0, height: int = 0, fps: float = 0.0) -> Dict[str, Any]:
    started = time.monotonic()
    capture = open_v4l2_capture(device, width, height, fps, require_read=False)
    result: Dict[str, Any] = {
        "device": device,
        "opened": False,
        "readable": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "error": "",
    }
    if capture is None:
        result["error"] = "OpenCV VideoCapture could not open this V4L2 device"
        return result
    try:
        result["opened"] = True
        ok, frame = capture.read()
        result["readable"] = bool(ok and frame is not None)
        if frame is not None:
            result["height"], result["width"] = frame.shape[:2]
        else:
            result["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            result["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        result["fps"] = round(float(capture.get(cv2.CAP_PROP_FPS) or 0.0), 2)
        if not result["readable"]:
            result["error"] = "Opened but did not return a frame"
    finally:
        capture.release()
    result["probe_ms"] = round((time.monotonic() - started) * 1000.0, 1)
    return result


def open_v4l2_capture(device: str, width: int = 0, height: int = 0, fps: float = 0.0, require_read: bool = True) -> Optional[cv2.VideoCapture]:
    attempts: List[Any] = [device]
    match = re.fullmatch(r"/dev/video(\d+)", device)
    if match:
        attempts.append(int(match.group(1)))
    for source in attempts:
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            continue
        _apply_capture_settings(capture, width, height, fps)
        if require_read:
            ok, _ = capture.read()
            if not ok:
                capture.release()
                continue
        return capture
    return None


def _apply_capture_settings(capture: cv2.VideoCapture, width: int, height: int, fps: float) -> None:
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        capture.set(cv2.CAP_PROP_FPS, float(fps))


def _natural_video_key(path: Path) -> tuple[int, str]:
    name = path.name
    match = re.fullmatch(r"video(\d+)", name)
    if match:
        return int(match.group(1)), name
    return 100000, name