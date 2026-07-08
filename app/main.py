"""FastAPI app for the RK3588 local camera tracking MVP."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.camera import CameraSource
from app.config import load_config
from app.detector import PersonDetector
from app.privacy import apply_privacy_mosaic
from app.stats import Line, StatsManager
from app.tracker import PersonTracker
from vision.types import Detection, TrackedPerson

APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"


def _placeholder_jpeg(message: str) -> bytes:
    frame = np.full((480, 854, 3), (236, 240, 243), dtype=np.uint8)
    cv2.putText(frame, "RK3588 Personnel Tracking", (32, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 43, 49), 2)
    for index, line in enumerate(_wrap_text(message, 72)[:5]):
        cv2.putText(frame, line, (32, 142 + index * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 96, 102), 2)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return encoded.tobytes() if ok else b""


def _wrap_text(value: str, size: int) -> list[str]:
    words = str(value).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > size and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]

class TrackingRuntime:
    def __init__(self) -> None:
        self.config = load_config()
        self.camera = CameraSource(self.config["camera"])
        self.detector = PersonDetector(self.config["detection"])
        self.tracker = PersonTracker(self.config["tracking"])
        self.stats = StatsManager(self.config["counting"])
        self.privacy_config = self.config["privacy"]
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_jpeg = _placeholder_jpeg("Starting camera tracking service")
        self._frame_index = 0
        self._fps = 0.0
        self._fps_count = 0
        self._fps_started = time.monotonic()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="rk3588-camera-web-tracking", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        self.camera.release()
        self.stats.update_runtime(
            fps=self._fps,
            latency_ms=0.0,
            frame_index=self._frame_index,
            camera_status="offline",
            source=self.camera.selected_source,
            model=self.detector.model_path,
            detector=self.detector.name,
            running=False,
            privacy_mode=bool(self.privacy_config.get("face_mosaic_enabled", True)),
            face_mosaic_enabled=bool(self.privacy_config.get("face_mosaic_enabled", True)),
            last_error=self.camera.error,
            available_cameras=self.camera.available_devices,
        )

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            return bytes(self._latest_jpeg)

    def reset_stats(self) -> dict:
        self.stats.reset()
        return self.stats.snapshot()

    def health(self) -> dict:
        stats = self.stats.snapshot()
        ok = bool(stats.get("running")) and stats.get("camera_status") == "online" and not stats.get("last_error")
        return {
            "status": "ok" if ok else "error",
            "running": stats.get("running", False),
            "camera_status": stats.get("camera_status", "offline"),
            "source": stats.get("source", ""),
            "fps": stats.get("fps", 0.0),
            "latency_ms": stats.get("latency_ms", 0.0),
            "error": stats.get("last_error", ""),
            "available_cameras": stats.get("available_cameras", []),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame_packet = self.camera.read()
            if frame_packet is None:
                self._publish_error(self.camera.error or "Waiting for a readable camera frame")
                time.sleep(0.2)
                continue
            try:
                frame = frame_packet.frame
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections)
                self.stats.update_tracks(tracks, frame.shape[:2])
                display = self._prepare_display_frame(frame, tracks, detections)
                line = self.stats.current_line()
                self._draw_overlay(display, tracks, line)
                ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if not ok:
                    raise RuntimeError("Could not encode JPEG frame")
                latency_ms = (time.monotonic() - frame_packet.captured_at) * 1000.0
                self._frame_index += 1
                self._update_fps()
                with self._lock:
                    self._latest_jpeg = encoded.tobytes()
                self.stats.update_runtime(
                    fps=self._fps,
                    latency_ms=latency_ms,
                    frame_index=self._frame_index,
                    camera_status=self.camera.status,
                    source=frame_packet.source,
                    model=self.detector.model_path,
                    detector=self.detector.name,
                    running=True,
                    privacy_mode=bool(self.privacy_config.get("face_mosaic_enabled", True)),
                    face_mosaic_enabled=bool(self.privacy_config.get("face_mosaic_enabled", True)),
                    last_error=self.detector.warning,
                    available_cameras=self.camera.available_devices,
                )
            except Exception as exc:
                self._publish_error(str(exc))
                time.sleep(0.1)

    def _prepare_display_frame(
        self,
        frame: np.ndarray,
        tracks: Iterable[TrackedPerson],
        detections: Iterable[Detection],
    ) -> np.ndarray:
        track_boxes = [track.bbox for track in tracks]
        detection_boxes = [detection.bbox for detection in detections]
        person_boxes = track_boxes or detection_boxes
        return apply_privacy_mosaic(
            frame,
            person_boxes=person_boxes,
            face_mosaic_enabled=bool(self.privacy_config.get("face_mosaic_enabled", True)),
            head_fallback_enabled=bool(self.privacy_config.get("head_fallback_enabled", True)),
            mosaic_strength=int(self.privacy_config.get("mosaic_strength") or 14),
        )

    def _draw_overlay(self, frame: np.ndarray, tracks: Iterable[TrackedPerson], line: Optional[Line]) -> None:
        if line is not None:
            (x1, y1), (x2, y2) = line
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 215, 255), 2)
        for track in tracks:
            bx1, by1, bx2, by2 = [int(value) for value in track.bbox]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (42, 166, 85), 2)
            label = f"ID {track.track_id} {track.confidence:.2f}"
            cv2.putText(frame, label, (bx1, max(24, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (42, 166, 85), 2)

    def _publish_error(self, message: str) -> None:
        with self._lock:
            self._latest_jpeg = _placeholder_jpeg(message)
        self.stats.update_runtime(
            fps=0.0,
            latency_ms=0.0,
            frame_index=self._frame_index,
            camera_status=self.camera.status,
            source=self.camera.selected_source,
            model=self.detector.model_path,
            detector=self.detector.name,
            running=True,
            privacy_mode=bool(self.privacy_config.get("face_mosaic_enabled", True)),
            face_mosaic_enabled=bool(self.privacy_config.get("face_mosaic_enabled", True)),
            last_error=message,
            available_cameras=self.camera.available_devices,
        )

    def _update_fps(self) -> None:
        self._fps_count += 1
        if self._fps_count < 10:
            return
        now = time.monotonic()
        elapsed = max(0.001, now - self._fps_started)
        self._fps = self._fps_count / elapsed
        self._fps_count = 0
        self._fps_started = now


runtime = TrackingRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="RK3588 Personnel Tracking", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_ROOT / "index.html"))


@app.get("/video")
def video():
    return StreamingResponse(_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/stats")
def api_stats():
    return JSONResponse(runtime.stats.snapshot())


@app.get("/api/health")
def api_health():
    return JSONResponse(runtime.health())


@app.post("/api/reset-stats")
def api_reset_stats():
    return JSONResponse(runtime.reset_stats())


def _mjpeg_stream():
    while not runtime._stop_event.is_set():
        frame = runtime.get_latest_jpeg()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)
