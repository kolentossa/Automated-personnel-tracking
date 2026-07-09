"""FastAPI app for the RK3588 local camera tracking MVP."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.camera import CameraFrame, CameraSource
from app.config import load_config, save_counting_config
from app.detector import PersonDetector
from app.privacy import apply_privacy_mosaic
from app.stats import Line, StatsManager, TIMING_KEYS
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
        detector_config = self.config.get("detector") or self.config.get("detection", {})
        self.detector = PersonDetector(detector_config, fallback_config=self.config.get("detection", {}))
        self.tracker = PersonTracker(self.config["tracking"])
        counting_config = dict(self.config["counting"])
        counting_config["_default_frame_width"] = int(self.config["camera"].get("width") or 960)
        counting_config["_default_frame_height"] = int(self.config["camera"].get("height") or 540)
        self.stats = StatsManager(counting_config)
        self.privacy_config = self.config["privacy"]
        self.performance_config = self.config.get("performance", {})
        self.stream_config = self.config.get("stream", {})
        self.target_fps = float(self.performance_config.get("target_fps") or 30)
        self.detect_every_n_frames = max(1, int(self.performance_config.get("detect_every_n_frames") or 1))
        self.jpeg_quality = max(35, min(95, int(self.stream_config.get("jpeg_quality") or 80)))
        self.stream_width = max(0, int(self.stream_config.get("width") or 0))
        self.stream_height = max(0, int(self.stream_config.get("height") or 0))
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._camera_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._latest_jpeg = _placeholder_jpeg("Starting camera tracking service")
        self._latest_camera_frame: Optional[CameraFrame] = None
        self._latest_camera_frame_id = 0
        self._last_capture_ms = 0.0
        self._last_detections: list[Detection] = []
        self._frame_index = 0
        self._fps = 0.0
        self._fps_count = 0
        self._fps_started = time.monotonic()
        self._update_runtime(
            fps=0.0,
            timings=_blank_timings(),
            camera_status=self.camera.status,
            source=self.camera.selected_source,
            running=False,
            last_error=self.detector.warning,
        )

    def start(self) -> None:
        with self._lock:
            if self._camera_thread and self._camera_thread.is_alive():
                return
            self._stop_event.clear()
            self._camera_thread = threading.Thread(target=self._camera_loop, name="rk3588-camera-capture", daemon=True)
            self._processing_thread = threading.Thread(target=self._processing_loop, name="rk3588-camera-processing", daemon=True)
            self._camera_thread.start()
            self._processing_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in (self._camera_thread, self._processing_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        self.camera.release()
        self._update_runtime(
            fps=self._fps,
            timings=_blank_timings(),
            camera_status="offline",
            source=self.camera.selected_source,
            running=False,
            last_error=self.camera.error,
        )

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            return bytes(self._latest_jpeg)

    def reset_stats(self) -> dict:
        self.stats.reset()
        return self.stats.snapshot()

    def get_counting_config(self) -> dict:
        return self.stats.counting_config()

    def update_counting_config(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        result = self.stats.configure_counting(payload.get("line"), str(payload.get("direction") or ""))
        save_counting_config(
            result["line"],
            result["direction"],
            int(result.get("cooldown_frames") or 20),
        )
        self.config["counting"] = {
            "line": result["line"],
            "direction": {"mode": result["direction"]},
            "cooldown_frames": int(result.get("cooldown_frames") or 20),
        }
        return result

    def health(self) -> dict:
        stats = self.stats.snapshot()
        ok = bool(stats.get("running")) and stats.get("camera_status") == "online" and not stats.get("last_error")
        data = {
            "status": "ok" if ok else "error",
            "running": stats.get("running", False),
            "camera_status": stats.get("camera_status", "offline"),
            "source": stats.get("source", ""),
            "fps": stats.get("fps", 0.0),
            "latency_ms": stats.get("latency_ms", 0.0),
            "error": stats.get("last_error", ""),
            "available_cameras": stats.get("available_cameras", []),
            "detector": stats.get("detector", ""),
            "npu_enabled": stats.get("npu_enabled", False),
        }
        for key in TIMING_KEYS:
            data[key] = stats.get(key, 0.0)
        return data

    def stream_interval(self) -> float:
        return 1.0 / max(1.0, min(60.0, self.target_fps))

    def _camera_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            frame_packet = self.camera.read()
            capture_ms = (time.monotonic() - started) * 1000.0
            if frame_packet is None:
                self._publish_error(self.camera.error or "Waiting for a readable camera frame")
                time.sleep(0.2)
                continue
            with self._lock:
                self._latest_camera_frame = frame_packet
                self._latest_camera_frame_id += 1
                self._last_capture_ms = round(capture_ms, 1)
            time.sleep(0.001)

    def _processing_loop(self) -> None:
        last_processed_frame_id = 0
        while not self._stop_event.is_set():
            with self._lock:
                frame_packet = self._latest_camera_frame
                frame_id = self._latest_camera_frame_id
                capture_ms = self._last_capture_ms
            if frame_packet is None or frame_id == last_processed_frame_id:
                time.sleep(0.003)
                continue
            last_processed_frame_id = frame_id
            try:
                queue_wait_ms = max(0.0, _ms(time.monotonic() - frame_packet.captured_at) - capture_ms)
                self._process_frame(frame_packet, capture_ms, queue_wait_ms)
            except Exception as exc:
                self._publish_error(str(exc))
                time.sleep(0.05)

    def _process_frame(self, frame_packet: CameraFrame, capture_ms: float, queue_wait_ms: float) -> None:
        frame = frame_packet.frame
        detect_this_frame = (self._frame_index % self.detect_every_n_frames) == 0 or not self._last_detections

        if detect_this_frame:
            detections = self.detector.detect(frame)
            detector_profile = dict(self.detector.last_profile)
            self._last_detections = detections
        else:
            detections = list(self._last_detections)
            detector_profile = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}

        tracking_started = time.monotonic()
        tracks = self.tracker.update(detections)
        self.stats.update_tracks(tracks, frame.shape[:2])
        tracking_ms = _ms(time.monotonic() - tracking_started)

        privacy_started = time.monotonic()
        display = self._prepare_display_frame(frame, tracks, detections)
        privacy_ms = _ms(time.monotonic() - privacy_started)

        draw_started = time.monotonic()
        line = self.stats.current_line()
        self._draw_overlay(display, tracks, line)
        draw_ms = _ms(time.monotonic() - draw_started)

        encode_started = time.monotonic()
        stream_frame = self._prepare_stream_frame(display)
        ok, encoded = cv2.imencode(".jpg", stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        encode_ms = _ms(time.monotonic() - encode_started)
        if not ok:
            raise RuntimeError("Could not encode JPEG frame")

        self._frame_index += 1
        self._update_fps()
        with self._lock:
            self._latest_jpeg = encoded.tobytes()

        total_latency_ms = _ms(time.monotonic() - frame_packet.captured_at)
        timings = {
            "capture_ms": capture_ms,
            "queue_wait_ms": queue_wait_ms,
            "preprocess_ms": detector_profile.get("preprocess_ms", 0.0),
            "inference_ms": detector_profile.get("inference_ms", 0.0),
            "postprocess_ms": detector_profile.get("postprocess_ms", 0.0),
            "tracking_ms": tracking_ms,
            "privacy_ms": privacy_ms,
            "draw_ms": draw_ms,
            "encode_ms": encode_ms,
            "total_latency_ms": total_latency_ms,
        }
        self._update_runtime(
            fps=self._fps,
            timings=timings,
            camera_status=self.camera.status,
            source=frame_packet.source,
            running=True,
            last_error=self.detector.warning,
        )

    def _prepare_stream_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.stream_width <= 0 or self.stream_height <= 0:
            return frame
        height, width = frame.shape[:2]
        if width == self.stream_width and height == self.stream_height:
            return frame
        return cv2.resize(frame, (self.stream_width, self.stream_height), interpolation=cv2.INTER_NEAREST)

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
        self._update_runtime(
            fps=0.0,
            timings=_blank_timings(),
            camera_status=self.camera.status,
            source=self.camera.selected_source,
            running=True,
            last_error=message,
        )

    def _update_runtime(
        self,
        *,
        fps: float,
        timings: dict,
        camera_status: str,
        source: str,
        running: bool,
        last_error: str,
    ) -> None:
        self.stats.update_runtime(
            fps=fps,
            latency_ms=float(timings.get("total_latency_ms", 0.0) or 0.0),
            frame_index=self._frame_index,
            camera_status=camera_status,
            source=source,
            model=self.detector.model_path,
            detector=self.detector.name,
            npu_enabled=self.detector.npu_enabled,
            running=running,
            privacy_mode=bool(self.privacy_config.get("face_mosaic_enabled", True)),
            face_mosaic_enabled=bool(self.privacy_config.get("face_mosaic_enabled", True)),
            last_error=last_error,
            available_cameras=self.camera.available_devices,
            timings=timings,
        )

    def _update_fps(self) -> None:
        self._fps_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_started
        if elapsed < 1.0:
            return
        self._fps = self._fps_count / max(0.001, elapsed)
        self._fps_count = 0
        self._fps_started = now


def _blank_timings() -> dict:
    return {key: 0.0 for key in TIMING_KEYS}


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)


runtime = TrackingRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="RK3588 Personnel Tracking", version="0.3.1", lifespan=lifespan)
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


@app.get("/api/config/counting")
def api_get_counting_config():
    return JSONResponse(runtime.get_counting_config())


@app.post("/api/config/counting")
def api_update_counting_config(payload: dict):
    try:
        return JSONResponse(runtime.update_counting_config(payload))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _mjpeg_stream():
    while not runtime._stop_event.is_set():
        frame = runtime.get_latest_jpeg()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(runtime.stream_interval())
