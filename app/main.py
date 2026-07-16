"""FastAPI app for the RK3588 local camera tracking MVP."""

from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.camera import CameraFrame, CameraSource
from app.behavior import BehaviorEngine, BehaviorEventManager, load_behavior_config
from app.behavior.detector import BehaviorObjectDetector
from app.config import load_config, save_counting_config
from app.detector import PersonDetector
from app.privacy import FaceMosaicProcessor
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
        self.behavior_config = load_behavior_config()
        self.performance_config = self.config.get("performance", {})
        self.cpu_affinity, self.cpu_affinity_error = _apply_cpu_affinity(
            self.performance_config.get("cpu_affinity")
        )
        camera_config = dict(self.config["camera"])
        _apply_behavior_source(camera_config, self.behavior_config.get("source", {}))
        self.camera = CameraSource(camera_config)
        detector_config = dict(self.config.get("detector") or self.config.get("detection", {}))
        primary_override = self.behavior_config.get("models", {}).get("primary", {})
        if bool(primary_override.get("use_for_runtime", False)):
            detector_config.update({key: value for key, value in primary_override.items() if key != "use_for_runtime"})
        self.detector = PersonDetector(detector_config, fallback_config=self.config.get("detection", {}))
        self.tracker = PersonTracker(self.config["tracking"])
        counting_config = dict(self.config["counting"])
        counting_config["_default_frame_width"] = int(self.config["camera"].get("width") or 960)
        counting_config["_default_frame_height"] = int(self.config["camera"].get("height") or 540)
        counting_config["_timing_window_frames"] = int(self.performance_config.get("timing_window_frames") or 30)
        self.stats = StatsManager(counting_config)
        self.privacy_config = self.config["privacy"]
        self.privacy = FaceMosaicProcessor(self.privacy_config)
        self.behavior_detector = BehaviorObjectDetector(
            self.behavior_config.get("models", {}).get("behavior", {})
        )
        self.behavior = BehaviorEngine(self.behavior_config)
        evidence_config = dict(self.behavior_config.get("evidence", {}))
        evidence_config["logging_level"] = self.behavior_config.get("logging", {}).get("level", "INFO")
        self.behavior_events = BehaviorEventManager(
            evidence_config,
            str(self.behavior_config.get("camera_id") or "rk3588-camera-01"),
        )
        self.stream_config = self.config.get("stream", {})
        self.target_fps = float(self.performance_config.get("target_fps") or 30)
        self.detect_every_n_frames = max(1, int(self.performance_config.get("detect_every_n_frames") or 1))
        self.memory_trim_every_n_frames = max(
            0, int(self.performance_config.get("memory_trim_every_n_frames") or 0)
        )
        self._malloc_trim, self._heap_trim_error = _load_malloc_trim()
        self._heap_trim_count = 0
        self._heap_trim_released_count = 0
        self._heap_gc_run_count = 0
        self._heap_gc_collected_count = 0
        self.jpeg_quality = max(35, min(95, int(self.stream_config.get("jpeg_quality") or 80)))
        self.stream_width = max(0, int(self.stream_config.get("width") or 0))
        self.stream_height = max(0, int(self.stream_config.get("height") or 0))
        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._camera_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._latest_jpeg = _placeholder_jpeg("Starting camera tracking service")
        self._latest_camera_frame: Optional[CameraFrame] = None
        self._latest_camera_frame_id = 0
        self._last_processed_camera_frame_id = 0
        self._camera_frames_captured = 0
        self._camera_frames_processed = 0
        self._camera_frames_dropped = 0
        self._last_capture_ms = 0.0
        self._last_detections: list[Detection] = []
        self._last_scene_detections: list[Detection] = []
        self._detector_has_run = False
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
            self.privacy.start()
            self.behavior_events.start()
            self._camera_thread = threading.Thread(target=self._camera_loop, name="rk3588-camera-capture", daemon=True)
            self._processing_thread = threading.Thread(target=self._processing_loop, name="rk3588-camera-processing", daemon=True)
            self._camera_thread.start()
            self._processing_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        for thread in (self._camera_thread, self._processing_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        self.privacy.stop()
        self.camera.release()
        self.detector.release()
        self.behavior_detector.release()
        self.behavior_events.stop()
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
        self.behavior.reset()
        self.behavior_events.reset()
        return self.stats_snapshot()

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
        stats = self.stats_snapshot()
        privacy_required = bool(self.privacy_config.get("face_mosaic_enabled", True))
        privacy_ok = not privacy_required or bool(stats.get("face_detector_available"))
        behavior_required = bool(stats.get("behavior_model_enabled")) and bool(stats.get("behavior_model_required"))
        behavior_ok = not behavior_required or bool(stats.get("behavior_model_available"))
        behavior_error = str(stats.get("behavior_model_error") or "")
        ok = (
            bool(stats.get("running"))
            and stats.get("camera_status") == "online"
            and not stats.get("last_error")
            and privacy_ok
            and behavior_ok
            and not behavior_error
        )
        data = {
            "status": "ok" if ok else "error",
            "running": stats.get("running", False),
            "camera_status": stats.get("camera_status", "offline"),
            "source": stats.get("source", ""),
            "fps": stats.get("fps", 0.0),
            "latency_ms": stats.get("latency_ms", 0.0),
            "error": stats.get("last_error", "") or behavior_error,
            "available_cameras": stats.get("available_cameras", []),
            "detector": stats.get("detector", ""),
            "npu_enabled": stats.get("npu_enabled", False),
        }
        for key in TIMING_KEYS:
            data[key] = stats.get(key, 0.0)
        for key in (
            "capture_queue_capacity",
            "capture_queue_depth",
            "camera_frames_captured",
            "camera_frames_processed",
            "camera_frames_dropped",
            "camera_open_attempts",
            "camera_successful_opens",
            "camera_reconnect_count",
            "camera_read_failures",
            "heap_trim_every_n_frames",
            "heap_trim_count",
            "heap_trim_released_count",
            "heap_gc_run_count",
            "heap_gc_collected_count",
        ):
            data[key] = stats.get(key, 0)
        data["heap_trim_error"] = stats.get("heap_trim_error", "")
        data.update(self.privacy.snapshot())
        data.update(self.detector.snapshot())
        data.update(self.behavior.snapshot())
        data.update(self.behavior_detector.snapshot())
        data.update(self.behavior_events.snapshot())
        data["phone_detection_available"] = self._phone_detection_available()
        data["smoking_detection_available"] = self.behavior_detector.smoking_detection_available
        data["behavior_status"] = self._behavior_status()
        data["cpu_affinity"] = self.cpu_affinity
        data["cpu_affinity_error"] = self.cpu_affinity_error
        return data

    def stats_snapshot(self) -> dict:
        data = self.stats.snapshot()
        data.update(self.camera.snapshot())
        data.update(self._pipeline_snapshot())
        data.update(self.privacy.snapshot())
        data.update(self.detector.snapshot())
        data.update(self.behavior.snapshot())
        data.update(self.behavior_detector.snapshot())
        data.update(self.behavior_events.snapshot())
        data["phone_detection_available"] = self._phone_detection_available()
        data["smoking_detection_available"] = self.behavior_detector.smoking_detection_available
        data["behavior_status"] = self._behavior_status()
        data["cpu_affinity"] = self.cpu_affinity
        data["cpu_affinity_error"] = self.cpu_affinity_error
        return data

    def _pipeline_snapshot(self) -> dict:
        with self._lock:
            queue_depth = int(self._latest_camera_frame_id > self._last_processed_camera_frame_id)
            return {
                "capture_queue_capacity": 1,
                "capture_queue_depth": queue_depth,
                "camera_frames_captured": int(self._camera_frames_captured),
                "camera_frames_processed": int(self._camera_frames_processed),
                "camera_frames_dropped": int(self._camera_frames_dropped),
                "heap_trim_every_n_frames": int(self.memory_trim_every_n_frames),
                "heap_trim_count": int(self._heap_trim_count),
                "heap_trim_released_count": int(self._heap_trim_released_count),
                "heap_gc_run_count": int(self._heap_gc_run_count),
                "heap_gc_collected_count": int(self._heap_gc_collected_count),
                "heap_trim_error": self._heap_trim_error,
            }

    def recent_behavior_events(self, limit: int = 50) -> dict:
        return {"events": self.behavior_events.recent(limit)}

    def _phone_detection_available(self) -> bool:
        return any(self.detector.supports_label(label) for label in ("cell phone", "phone", "mobile phone"))

    def _behavior_status(self) -> str:
        if not self.behavior.enabled:
            return "disabled"
        model = self.behavior_detector
        if model.error:
            return "error" if model.required else "degraded"
        if self.behavior.smoking.enabled and not model.smoking_detection_available:
            return "error" if model.required else "degraded"
        return "ready"

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
            with self._frame_ready:
                if self._latest_camera_frame_id > self._last_processed_camera_frame_id:
                    self._camera_frames_dropped += 1
                self._latest_camera_frame = frame_packet
                self._latest_camera_frame_id += 1
                self._camera_frames_captured += 1
                self._last_capture_ms = round(capture_ms, 1)
                self._frame_ready.notify()

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._frame_ready:
                if self._latest_camera_frame_id == self._last_processed_camera_frame_id:
                    self._frame_ready.wait(timeout=0.1)
                frame_packet = self._latest_camera_frame
                frame_id = self._latest_camera_frame_id
                capture_ms = self._last_capture_ms
                has_new_frame = frame_packet is not None and frame_id != self._last_processed_camera_frame_id
                if has_new_frame:
                    self._last_processed_camera_frame_id = frame_id
                    self._camera_frames_processed += 1
            if not has_new_frame:
                continue
            try:
                queue_wait_ms = max(0.0, _ms(time.monotonic() - frame_packet.captured_at))
                self._process_frame(frame_packet, capture_ms, queue_wait_ms)
            except Exception as exc:
                self._publish_error(str(exc))
                time.sleep(0.05)

    def _process_frame(self, frame_packet: CameraFrame, capture_ms: float, queue_wait_ms: float) -> None:
        frame = frame_packet.frame
        detect_this_frame = (self._frame_index % self.detect_every_n_frames) == 0 or not self._detector_has_run

        if detect_this_frame:
            scene_detections = self.detector.detect_scene(frame)
            detections = [
                item for item in scene_detections if int(item.class_id) == 0 or item.label == "person"
            ]
            detector_profile = dict(self.detector.last_profile)
            self._last_detections = detections
            self._last_scene_detections = scene_detections
            self._detector_has_run = True
        else:
            detections = list(self._last_detections)
            scene_detections = list(self._last_scene_detections)
            detector_profile = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}

        behavior_detections = self.behavior_detector.detect(frame, self._frame_index)
        behavior_detector_profile = dict(self.behavior_detector.last_profile)
        all_scene_detections = scene_detections + behavior_detections

        tracking_started = time.monotonic()
        tracks = self.tracker.update(detections)
        self.stats.update_tracks(tracks, frame.shape[:2])
        tracking_ms = _ms(time.monotonic() - tracking_started)

        privacy_started = time.monotonic()
        display = self._prepare_display_frame(frame, tracks, detections)
        privacy_ms = _ms(time.monotonic() - privacy_started)

        behavior_result = self.behavior.update(
            tracks,
            all_scene_detections,
            self.privacy.face_boxes(),
            frame.shape[:2],
        )
        event_started = time.monotonic()
        for trigger in behavior_result.triggers:
            self.behavior_events.emit(trigger, display)
        event_output_ms = _ms(time.monotonic() - event_started)

        draw_started = time.monotonic()
        line = self.stats.current_line()
        self._draw_overlay(display, tracks, line, all_scene_detections)
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
            "behavior_preprocess_ms": behavior_detector_profile.get("preprocess_ms", 0.0),
            "behavior_inference_ms": behavior_detector_profile.get("inference_ms", 0.0),
            "behavior_postprocess_ms": behavior_detector_profile.get("postprocess_ms", 0.0),
            "behavior_analysis_ms": behavior_result.analysis_ms,
            "event_output_ms": event_output_ms,
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
        self._maybe_trim_heap()

    def _maybe_trim_heap(self) -> None:
        interval = self.memory_trim_every_n_frames
        if interval <= 0 or self._frame_index % interval != 0:
            return
        collected = gc.collect()
        self._heap_gc_run_count += 1
        self._heap_gc_collected_count += int(collected)
        if self._malloc_trim is None:
            return
        try:
            released = int(self._malloc_trim(0))
            self._heap_trim_count += 1
            self._heap_trim_released_count += int(released > 0)
            self._heap_trim_error = ""
        except Exception as exc:
            self._heap_trim_error = str(exc)
            self._malloc_trim = None

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
        return self.privacy.process(frame, person_boxes)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        tracks: Iterable[TrackedPerson],
        line: Optional[Line],
        scene_detections: Iterable[Detection] = (),
    ) -> None:
        if line is not None:
            (x1, y1), (x2, y2) = line
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 215, 255), 2)
        for track in tracks:
            bx1, by1, bx2, by2 = [int(value) for value in track.bbox]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (42, 166, 85), 2)
            label = f"ID {track.track_id} {track.confidence:.2f}"
            cv2.putText(frame, label, (bx1, max(24, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (42, 166, 85), 2)
        colors = {
            "cell phone": (0, 165, 255),
            "phone": (0, 165, 255),
            "cigarette": (46, 46, 220),
            "smoke": (170, 170, 170),
            "flame": (0, 90, 255),
            "lighter": (160, 80, 190),
            "hand": (190, 150, 30),
        }
        for detection in scene_detections:
            label = str(detection.label).strip().lower()
            if label == "person" or label not in colors:
                continue
            bx1, by1, bx2, by2 = [int(value) for value in detection.bbox]
            color = colors[label]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(
                frame,
                f"{label} {detection.confidence:.2f}",
                (bx1, max(18, by1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                2,
            )

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


def _apply_behavior_source(camera_config: dict, source_config: dict) -> None:
    """Apply source fields from the unified behavior config to CameraSource."""

    if not bool(source_config.get("use_for_runtime", False)):
        return
    source_type = str(source_config.get("type") or "").strip()
    if source_type:
        camera_config["source_type"] = source_type
    for key in ("camera_device", "rtsp_url", "video_file"):
        value = source_config.get(key)
        if value not in (None, ""):
            camera_config[key] = value


def _apply_cpu_affinity(value) -> tuple[list[int], str]:
    if not hasattr(os, "sched_setaffinity"):
        return [], "CPU affinity is not supported on this platform"
    try:
        requested: set[int] = set()
        parts = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
        for part in parts:
            text = str(part).strip()
            if not text:
                continue
            if "-" in text:
                start, end = (int(item.strip()) for item in text.split("-", 1))
                requested.update(range(min(start, end), max(start, end) + 1))
            else:
                requested.add(int(text))
        available = set(os.sched_getaffinity(0))
        selected = requested & available if requested else available
        if not selected:
            raise ValueError(f"No requested CPUs are available: {sorted(requested)}")
        os.sched_setaffinity(0, selected)
        return sorted(os.sched_getaffinity(0)), ""
    except Exception as exc:
        return [], str(exc)


def _load_malloc_trim():
    if os.name != "posix":
        return None, "malloc_trim is only available on POSIX glibc systems"
    try:
        import ctypes

        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return trim, ""
    except Exception as exc:
        return None, str(exc)


runtime = TrackingRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="RK3588 Personnel Tracking", version="0.4.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_ROOT / "index.html"))


@app.get("/video")
def video():
    return StreamingResponse(_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/stats")
def api_stats():
    return JSONResponse(runtime.stats_snapshot())


@app.get("/api/health")
def api_health():
    return JSONResponse(runtime.health())


@app.post("/api/reset-stats")
def api_reset_stats():
    return JSONResponse(runtime.reset_stats())


@app.get("/api/events")
def api_behavior_events(limit: int = Query(default=50, ge=1, le=200)):
    return JSONResponse(runtime.recent_behavior_events(limit))


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
