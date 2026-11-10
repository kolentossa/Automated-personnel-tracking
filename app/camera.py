"""通过 V4L2 读取摄像头；断流后按配置的间隔重新连接。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2


@dataclass(frozen=True)
class CameraFrame:
    frame: Any
    captured_at: float
    source: str


def create_camera(config: dict) -> SimpleNamespace:
    """保存设置和连接状态；第一次读取时才打开设备。"""
    return SimpleNamespace(
        camera_device=str(config.get("camera_device") or "/dev/video11"),
        auto_detect=bool(config.get("auto_detect", True)),
        width=int(config.get("width") or 1280),
        height=int(config.get("height") or 720),
        target_fps=float(config.get("fps") or 0),
        retry_interval_sec=float(config.get("retry_interval_sec") or 3.0),
        status="offline",
        error="",
        selected_source="",
        available_devices=[],
        capture=None,
        last_open_attempt=0.0,
    )


def read_frame(camera: SimpleNamespace) -> CameraFrame | None:
    """读取一帧；失败时关闭连接，等到重试间隔结束再打开。"""
    if camera.capture is None:
        now = time.monotonic()
        if now - camera.last_open_attempt < camera.retry_interval_sec:
            return None
        if not open_camera(camera):
            return None
    ok, frame = camera.capture.read()
    if ok and frame is not None:
        camera.status = "online"
        camera.error = ""
        return CameraFrame(frame, time.monotonic(), camera.selected_source)
    camera.status = "error"
    camera.error = f"Could not read a frame from {camera.selected_source or camera.camera_device}"
    close_camera(camera)
    return None


def open_camera(camera: SimpleNamespace) -> bool:
    """按顺序试读设备，使用第一个能返回图像的摄像头。"""
    close_camera(camera)
    camera.last_open_attempt = time.monotonic()
    candidates = _candidate_devices(camera)
    camera.available_devices = []
    for device in candidates:
        capture = open_v4l2_capture(device, camera.width, camera.height, camera.target_fps)
        probe = probe_capture(capture, device, "v4l2")
        camera.available_devices.append(probe)
        if probe.get("readable"):
            camera.capture = capture
            camera.selected_source = device
            camera.status = "online"
            camera.error = ""
            return True
        if capture is not None:
            capture.release()
    camera.selected_source = candidates[0] if candidates else camera.camera_device
    camera.status = "error"
    probed = ", ".join(candidates) if candidates else "no /dev/video* devices"
    camera.error = f"No readable camera found. backend=v4l2; configured={camera.camera_device}; probed={probed}."
    return False


def close_camera(camera: SimpleNamespace) -> None:
    """释放设备，保留错误文字供网页显示。"""
    if camera.capture is not None:
        camera.capture.release()
    camera.capture = None
    if camera.status == "online":
        camera.status = "offline"


def _candidate_devices(camera: SimpleNamespace) -> list[str]:
    # 配置设备排第一；自动发现的设备按编号排序，排除重复及不存在的路径。
    candidates = []
    if camera.camera_device:
        candidates.append(_normalise_device(camera.camera_device))
    if camera.auto_detect:
        preferred = ["/dev/video11", "/dev/video0"]
        candidates.extend(device for device in preferred if Path(device).exists())
        candidates.extend(
            str(path)
            for path in sorted(Path("/dev").glob("video*"), key=_natural_video_key)
            if re.fullmatch(r"video\d+", path.name)
        )
    unique = []
    for candidate in candidates:
        if candidate not in unique and Path(candidate).exists():
            unique.append(candidate)
    return unique


def open_v4l2_capture(device: str, width: int = 0, height: int = 0, fps: float = 0.0):
    # OpenCV 可接受路径或设备编号；路径打不开时再试编号。
    attempts = [device]
    match = re.fullmatch(r"/dev/video(\d+)", device)
    if match:
        attempts.append(int(match.group(1)))
    for source in attempts:
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            continue
        if width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps > 0:
            capture.set(cv2.CAP_PROP_FPS, float(fps))
        return capture
    return None


def probe_capture(capture, device: str, backend: str) -> dict:
    """试读一次，并记录设备打开情况和图像大小。"""
    started = time.monotonic()
    result = {
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


def _normalise_device(value: str) -> str:
    path = Path(value)
    if path.is_symlink():
        resolved = path.resolve()
        if re.fullmatch(r"video\d+", resolved.name):
            return str(resolved)
    return value


def _natural_video_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"video(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return 100000, path.name
