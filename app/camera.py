"""OpenCV camera access for RK3588 V4L2 and GStreamer sources."""

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
    """Read frames from an RK3588 camera, with a video-file interface reserved."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.source_type = str(config.get("source_type", "camera"))
        self.camera_device = str(config.get("camera_device") or "/dev/video11")
        self.capture_backend = str(config.get("capture_backend") or "auto").lower()
        self.auto_detect = bool(config.get("auto_detect", True))
        self.width = int(config.get("width") or 1280)
        self.height = int(config.get("height") or 720)
        self.target_fps = float(config.get("fps") or 0)
        self.retry_interval_sec = float(config.get("retry_interval_sec") or 3.0)
        self.video_file = str(config.get("video_file") or "data/sample.mp4")
        self.gstreamer_pipeline_template = str(
            config.get("gstreamer_pipeline")
            or "v4l2src device={device} ! video/x-raw,format=NV12,width={width},height={height} "
            "! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
        )
        self.status = "offline"
        self.error = ""
        self.selected_source = ""
        self.selected_backend = ""
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
        ok, frame = self._capture.read()
        if ok and frame is not None:
            self.status = "online"
            self.error = ""
            return CameraFrame(frame=frame, captured_at=time.monotonic(), source=self.selected_source)
        if self.source_type == "video" and self._capture is not None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok and frame is not None:
                self.status = "online"
                self.error = ""
                return CameraFrame(frame=frame, captured_at=time.monotonic(), source=self.selected_source)
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
        self.selected_backend = "video-file"
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
            if self.capture_backend in {"auto", "gstreamer", "gst"}:
                capture = open_gstreamer_capture(device, self.width, self.height, self.gstreamer_pipeline_template)
                probe = probe_capture(capture, device, "gstreamer")
                self.available_devices.append(probe)
                if probe.get("readable") and capture is not None:
                    self._capture = capture
                    self.selected_source = device
                    self.selected_backend = "gstreamer"
                    self.status = "online"
                    self.error = ""
                    return True
                if capture is not None:
                    capture.release()
            if self.capture_backend in {"auto", "v4l2"}:
                capture = open_v4l2_capture(device, self.width, self.height, self.target_fps)
                probe = probe_capture(capture, device, "v4l2")
                self.available_devices.append(probe)
                if probe.get("readable") and capture is not None:
                    self._capture = capture
                    self.selected_source = device
                    self.selected_backend = "v4l2"
                    self.status = "online"
                    self.error = ""
                    return True
                if capture is not None:
                    capture.release()
        self.selected_source = candidates[0] if candidates else self.camera_device
        self.selected_backend = self.capture_backend
        self.status = "error"
        probed = ", ".join(candidates) if candidates else "no /dev/video* devices"
        self.error = f"No readable camera found. backend={self.capture_backend}; configured={self.camera_device}; probed={probed}."
        return False

    def _candidate_devices(self) -> List[str]:
        candidates: List[str] = []
        if self.camera_device:
            candidates.append(_normalise_device(self.camera_device))
        if self.auto_detect:
            preferred = ["/dev/video11", "/dev/video0"]
            candidates.extend(device for device in preferred if Path(device).exists())
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


def probe_capture(capture: Optional[cv2.VideoCapture], device: str, backend: str) -> Dict[str, Any]:
    started = time.monotonic()
    result: Dict[str, Any] = {
        "device": device,
        "backend": backend,
        "opened": False,
        "readable": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "error": "",
    }
    if capture is None or not capture.isOpened():
        result["error"] = f"OpenCV could not open this camera with {backend}"
        return result
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
    result["probe_ms"] = round((time.monotonic() - started) * 1000.0, 1)
    return result


def open_gstreamer_capture(device: str, width: int, height: int, template: str) -> Optional[cv2.VideoCapture]:
    pipeline = template.format(device=device, width=int(width), height=int(height))
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def open_v4l2_capture(device: str, width: int = 0, height: int = 0, fps: float = 0.0) -> Optional[cv2.VideoCapture]:
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
        return capture
    return None


def _apply_capture_settings(capture: cv2.VideoCapture, width: int, height: int, fps: float) -> None:
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        capture.set(cv2.CAP_PROP_FPS, float(fps))


def _normalise_device(value: str) -> str:
    path = Path(value)
    if path.is_symlink():
        resolved = path.resolve()
        if re.fullmatch(r"video\d+", resolved.name):
            return str(resolved)
    return value


def _natural_video_key(path: Path) -> tuple[int, str]:
    name = path.name
    match = re.fullmatch(r"video(\d+)", name)
    if match:
        return int(match.group(1)), name
    return 100000, name
