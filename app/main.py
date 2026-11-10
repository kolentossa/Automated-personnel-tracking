"""程序入口：先读 process_frame，再看启动、后台循环和末尾的 Web 接口。"""

from __future__ import annotations

import os
import threading
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import camera, counting, detector, privacy, tracker
from app.camera import CameraFrame
from app.config import load_config, save_counting_config
from app.counting import Line, TIMING_KEYS
from app.types import Detection, TrackedPerson


@dataclass(frozen=True)
class FrameJob:
    """把画面、顺序号和采集耗时放在一起，防止异步结果用错帧。"""
    packet: CameraFrame
    frame_id: int
    capture_ms: float
    queue_wait_ms: float


def process_frame(program, frame_packet: CameraFrame, capture_ms, queue_wait_ms, detection_result):
    """一帧的主线：跟踪计数 → 遮挡画框 → 编码 → 发布。

    检测已在 NPU 线程完成，detection_result 始终对应 frame_packet。
    timings 只记录耗时，不参与人员判断。
    """
    detections, detector_profile = detection_result
    timings = {
        "capture_ms": capture_ms,
        "queue_wait_ms": queue_wait_ms,
        "preprocess_ms": detector_profile.get("preprocess_ms", 0.0),
        "inference_ms": detector_profile.get("inference_ms", 0.0),
        "postprocess_ms": detector_profile.get("postprocess_ms", 0.0),
    }
    tracks = track_and_count(program, frame_packet, detections, timings)
    display = prepare_display_frame(program, frame_packet.frame, tracks, detections, timings)
    encoded = encode_frame(program, display, timings)

    program.frame_index += 1
    update_fps(program)
    with program.lock:
        program.latest_jpeg = encoded.tobytes()
    timings["total_latency_ms"] = _ms(time.monotonic() - frame_packet.captured_at)
    update_runtime(program, timings=timings, source=frame_packet.source)


def track_and_count(program, packet, detections, timings):
    """给检测框接上 ID，再用本帧真实位置更新人数，返回轨迹列表。"""
    started = time.monotonic()
    tracks = tracker.track_people(program.tracker, detections, timestamp=packet.captured_at)
    counting.update_counting(
        program.counter, tracks, packet.frame.shape[:2], timestamp=packet.captured_at
    )
    timings["tracking_ms"] = _ms(time.monotonic() - started)
    return tracks


def prepare_display_frame(
    program,
    frame: np.ndarray,
    tracks: Iterable[TrackedPerson],
    detections: Iterable[Detection],
    timings=None,
) -> np.ndarray:
    """在原图副本上遮挡人脸，再画框和计数线；原图留给检测和计数。"""
    if timings is None:
        timings = {}
    started = time.monotonic()
    track_boxes = [track.bbox for track in tracks]
    detection_boxes = [detection.bbox for detection in detections]
    display = privacy.mask_faces(program.privacy, frame, track_boxes + detection_boxes)
    timings["privacy_ms"] = _ms(time.monotonic() - started)
    started = time.monotonic()
    draw_overlay(program, display, tracks, counting.current_line(program.counter))
    timings["draw_ms"] = _ms(time.monotonic() - started)
    return display


def draw_overlay(program, frame: np.ndarray, tracks: Iterable[TrackedPerson], line: Optional[Line]) -> None:
    """实测轨迹画绿色框和计数锚点，预测轨迹画灰框；不把统计数画进视频。"""
    if line is not None:
        (x1, y1), (x2, y2) = line
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 215, 255), 2)
    for track in tracks:
        bx1, by1, bx2, by2 = [int(value) for value in track.bbox]
        if track.observed:
            point = tuple(int(value) for value in counting.tracking_point(program.counter, track.observation_bbox or track.bbox))
            cv2.circle(frame, point, 4, (255, 220, 0), -1)
        color = (42, 166, 85) if track.observed else (145, 145, 145)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        label = f"ID {track.track_id} {track.confidence:.2f}"
        cv2.putText(frame, label, (bx1, max(24, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)


def encode_frame(program, frame, timings):
    """按网页尺寸编码 JPEG；编码失败时不发布新画面。"""
    started = time.monotonic()
    width, height = program.stream_width, program.stream_height
    if width > 0 and height > 0 and frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), program.jpeg_quality])
    timings["encode_ms"] = _ms(time.monotonic() - started)
    if not ok:
        raise RuntimeError("Could not encode JPEG frame")
    return encoded


def create_program() -> SimpleNamespace:
    """读配置、建立组件和共享数据。这里不启动线程。

    SimpleNamespace 只保存数据，例如 program.camera；它没有业务方法。
    后续函数显式接收 program，读写同一份状态。
    """
    config = load_config()
    performance = config.get("performance", {})
    stream = config.get("stream", {})
    affinity, affinity_error = _apply_cpu_affinity(performance.get("cpu_affinity"))
    camera_state = camera.create_camera(config["camera"])
    tracker_state = tracker.create_tracker(config["tracking"])
    # ByteTrack 需要低分框做第二次匹配，所以检测阈值取两者中的较低值。
    detector_config = tracker.detector_config_for_tracking(config["detector"], config["tracking"])
    detector_state = detector.load_detector(detector_config)

    counting_config = dict(config["counting"])
    counting_config["_default_frame_width"] = int(config["camera"].get("width") or 960)
    counting_config["_default_frame_height"] = int(config["camera"].get("height") or 540)
    counting_config["_timing_window_frames"] = int(performance.get("timing_window_frames") or 30)
    counting_config["_reference_fps"] = float(config["tracking"].get("reference_fps", 30))
    # 计数记忆比丢失轨迹多留一秒，遮挡后仍能判断这个人从哪一侧走来。
    counting_config["_minimum_retention_seconds"] = float(config["tracking"].get("lost_track_buffer_seconds", 4)) + 1
    counter_state = counting.create_counter(counting_config)
    privacy_state = privacy.create_privacy(config["privacy"])

    lock = threading.RLock()
    program = SimpleNamespace(
        camera=camera_state,
        detector=detector_state,
        tracker=tracker_state,
        counter=counter_state,
        privacy=privacy_state,
        privacy_config=config["privacy"],
        cpu_affinity=affinity,
        cpu_affinity_error=affinity_error,
        target_fps=float(performance.get("target_fps") or 30),
        overlap_inference=True,
        jpeg_quality=max(35, min(95, int(stream.get("jpeg_quality") or 80))),
        stream_width=max(0, int(stream.get("width") or 0)),
        stream_height=max(0, int(stream.get("height") or 0)),
        lock=lock,
        frame_ready=threading.Condition(lock),
        stop_event=threading.Event(),
        camera_thread=None,
        processing_thread=None,
        latest_jpeg=_placeholder_jpeg("Starting camera tracking service"),
        latest_camera_frame=None,
        latest_camera_frame_id=0,
        last_capture_ms=0.0,
        frame_index=0,
        fps=0.0,
        fps_count=0,
        fps_started=time.monotonic(),
    )
    update_runtime(program, running=False)
    return program


def start_program(program) -> None:
    """启动隐私工作线程、采集线程和处理线程；重复调用不会重复启动。"""
    with program.lock:
        if program.camera_thread and program.camera_thread.is_alive():
            return
        program.stop_event.clear()
        privacy.start_privacy(program.privacy)
        program.camera_thread = threading.Thread(
            target=camera_loop, args=(program,), name="rk3588-camera-capture", daemon=True
        )
        program.processing_thread = threading.Thread(
            target=processing_loop, args=(program,), name="rk3588-camera-processing", daemon=True
        )
        program.camera_thread.start()
        program.processing_thread.start()


def stop_program(program) -> None:
    """先通知停止，再唤醒并等待工作线程，最后释放摄像头。"""
    program.stop_event.set()
    with program.frame_ready:
        program.frame_ready.notify_all()
    for thread in (program.camera_thread, program.processing_thread):
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
    privacy.stop_privacy(program.privacy)
    camera.close_camera(program.camera)
    update_runtime(program, running=False, camera_status="offline", last_error=program.camera.error)


def camera_loop(program) -> None:
    """持续采集：发布最新帧，并通知处理线程有更新。"""
    while not program.stop_event.is_set():
        started = time.monotonic()
        frame_packet = camera.read_frame(program.camera)
        capture_ms = (time.monotonic() - started) * 1000.0
        if frame_packet is None:
            publish_error(program, program.camera.error or "Waiting for a readable camera frame")
            time.sleep(0.2)
            continue
        with program.frame_ready:
            program.latest_camera_frame = frame_packet
            program.latest_camera_frame_id += 1
            program.last_capture_ms = round(capture_ms, 1)
            program.frame_ready.notify()


def processing_loop(program) -> None:
    """下一帧推理与当前帧后处理并行；最多一个推理任务在途。"""
    last_submitted = 0
    pending = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rk3588-npu") as executor:
        while not program.stop_event.is_set():
            try:
                if pending is None:
                    job = latest_frame_job(program, last_submitted, wait=True)
                    if job is None:
                        continue
                    last_submitted = job.frame_id
                    pending = (job, executor.submit(detect_profiled, program, job.packet.frame))
                job, future = pending
                pending = None
                result = future.result()
                if program.stop_event.is_set():
                    break
                # 先提交下一帧推理，再绘制/编码当前帧；检测器仍只有一个调用者。
                next_job = latest_frame_job(program, last_submitted, wait=False)
                if next_job is not None:
                    last_submitted = next_job.frame_id
                    pending = (next_job, executor.submit(detect_profiled, program, next_job.packet.frame))
                process_frame(program, job.packet, job.capture_ms, job.queue_wait_ms, detection_result=result)
            except Exception as exc:
                # 推理失败时不能借用上一帧结果；记录错误后继续处理最新帧。
                publish_error(program, str(exc))
                program.stop_event.wait(0.05)


def latest_frame_job(program, last_id: int, *, wait: bool) -> Optional[FrameJob]:
    """读取最新帧引用及其采集时间；没有新帧时按需等待一次。"""
    with program.frame_ready:
        if wait and program.latest_camera_frame_id == last_id:
            program.frame_ready.wait(timeout=0.1)
        packet = program.latest_camera_frame
        if packet is None or program.latest_camera_frame_id == last_id:
            return None
        return FrameJob(
            packet,
            program.latest_camera_frame_id,
            program.last_capture_ms,
            max(0.0, _ms(time.monotonic() - packet.captured_at)),
        )


def detect_profiled(program, frame):
    detections = detector.detect_people(program.detector, frame)
    return detections, dict(program.detector.last_profile)


def get_latest_jpeg(program) -> bytes:
    with program.lock:
        return program.latest_jpeg


def stream_interval(program) -> float:
    return 1.0 / max(1.0, min(60.0, program.target_fps))


def reset_stats(program) -> dict:
    """清空计数与过线记忆；不重置摄像头或跟踪 ID。"""
    counting.reset_counter(program.counter)
    return counting.counter_snapshot(program.counter)


def get_counting_config(program) -> dict:
    return counting.counting_config(program.counter)


def update_counting_config(program, payload: dict) -> dict:
    """先验证并应用计数设置，再写回配置文件，保留其他配置分区。"""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    result = counting.configure_counting(
        program.counter, payload.get("line"), str(payload.get("direction") or ""),
        anchor=payload.get("anchor"), hysteresis_px=payload.get("hysteresis_px"),
        rearm_distance_px=payload.get("rearm_distance_px"),
    )
    settings = {
        "anchor": result["anchor"],
        "cooldown_frames": result["cooldown_frames"],
        "hysteresis_px": result["hysteresis_px"],
        "rearm_distance_px": result["rearm_distance_px"],
        "confirmation_frames": result["confirmation_frames"],
        "track_state_ttl_frames": result["track_state_ttl_frames"],
        "count_once_per_direction_per_track": result["count_once_per_direction_per_track"],
    }
    save_counting_config(result["line"], result["direction"], **settings)
    return result


def stats_snapshot(program) -> dict:
    data = counting.counter_snapshot(program.counter)
    data.update(diagnostics_snapshot(program))
    return data


def diagnostics_snapshot(program) -> dict:
    """人数接口和健康接口共用的跟踪、隐私及处理设置。"""
    return {
        **tracker.tracker_snapshot(program.tracker),
        **privacy.privacy_snapshot(program.privacy),
        "overlap_inference": program.overlap_inference,
        "cpu_affinity": program.cpu_affinity,
        "cpu_affinity_error": program.cpu_affinity_error,
    }


def health(program) -> dict:
    """返回机器可读的完整健康信息；界面隐藏性能字段不会删除此接口。"""
    stats = counting.counter_snapshot(program.counter)
    diagnostics = diagnostics_snapshot(program)
    privacy_required = bool(program.privacy_config.get("face_mosaic_enabled", True))
    privacy_ok = not privacy_required or diagnostics["face_detector_available"]
    ok = (
        stats["running"]
        and stats["camera_status"] == "online"
        and not stats["last_error"]
        and privacy_ok
    )
    data = {
        "status": "ok" if ok else "error",
        "running": stats["running"],
        "camera_status": stats["camera_status"],
        "source": stats["source"],
        "fps": stats["fps"],
        "latency_ms": stats["latency_ms"],
        "error": stats["last_error"],
        "available_cameras": stats["available_cameras"],
        "detector": stats["detector"],
        "npu_enabled": stats["npu_enabled"],
    }
    for key in TIMING_KEYS:
        data[key] = stats[key]
    data.update(diagnostics)
    return data


def publish_error(program, message: str) -> None:
    """发布错误占位图；工作循环仍可继续重试摄像头或处理后续帧。"""
    with program.lock:
        program.latest_jpeg = _placeholder_jpeg(message)
    update_runtime(program, fps=0.0, last_error=message)


def update_runtime(program, *, timings=None, fps=None, camera_status=None, source=None,
                   running=True, last_error=None):
    """默认读取当前组件状态；启动、停止和错误处理只覆盖不同的字段。"""
    if timings is None:
        timings = _blank_timings()
    face_mosaic = bool(program.privacy_config.get("face_mosaic_enabled", True))
    counting.update_runtime_metrics(
        program.counter,
        fps=program.fps if fps is None else fps,
        latency_ms=float(timings.get("total_latency_ms", 0.0) or 0.0),
        frame_index=program.frame_index,
        camera_status=program.camera.status if camera_status is None else camera_status,
        source=program.camera.selected_source if source is None else source,
        model=program.detector.model_path,
        detector=program.detector.name,
        npu_enabled=program.detector.npu_enabled,
        running=running,
        privacy_mode=face_mosaic,
        face_mosaic_enabled=face_mosaic,
        last_error=program.detector.warning if last_error is None else last_error,
        available_cameras=program.camera.available_devices,
        timings=timings,
    )


def update_fps(program) -> None:
    program.fps_count += 1
    now = time.monotonic()
    elapsed = now - program.fps_started
    if elapsed < 1.0:
        return
    program.fps = program.fps_count / elapsed
    program.fps_count = 0
    program.fps_started = now


def _placeholder_jpeg(message: str) -> bytes:
    frame = np.full((480, 854, 3), (236, 240, 243), dtype=np.uint8)
    cv2.putText(frame, "RK3588 Personnel Tracking", (32, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 43, 49), 2)
    for index, line in enumerate(_wrap_text(message, 72)[:5]):
        cv2.putText(frame, line, (32, 142 + index * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 96, 102), 2)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return encoded.tobytes() if ok else b""


def _wrap_text(value: str, size: int) -> list[str]:
    return textwrap.wrap(
        " ".join(str(value).split()), width=max(1, size),
        break_long_words=False, break_on_hyphens=False,
    ) or [""]


def _blank_timings() -> dict:
    return {key: 0.0 for key in TIMING_KEYS}


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)


def _apply_cpu_affinity(value) -> tuple[list[int], str]:
    """把配置的 CPU 编号/范围与可用 CPU 取交集；失败时返回说明。"""
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


# Web 接口只读写共享状态；视频处理由上面的后台函数完成。
APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
program = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """服务启动时建立唯一一份程序状态，退出时停止后台线程。"""
    global program
    program = create_program()
    start_program(program)
    try:
        yield
    finally:
        stop_program(program)


app = FastAPI(title="RK3588 Personnel Tracking", version="0.3.1", lifespan=lifespan)


class RevalidatingStaticFiles(StaticFiles):
    """框架的小扩展：浏览器每次检查文件版本，避免更新后混用旧脚本。"""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatingStaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_ROOT / "index.html"), headers={"Cache-Control": "no-cache"})


@app.get("/video")
def video():
    return StreamingResponse(_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/stats")
def api_stats():
    return JSONResponse(stats_snapshot(program))


@app.get("/api/health")
def api_health():
    return JSONResponse(health(program))


@app.post("/api/reset-stats")
def api_reset_stats():
    return JSONResponse(reset_stats(program))


@app.get("/api/config/counting")
def api_get_counting_config():
    return JSONResponse(get_counting_config(program))


@app.post("/api/config/counting")
def api_update_counting_config(payload: dict):
    try:
        return JSONResponse(update_counting_config(program, payload))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _mjpeg_stream():
    """反复发送最新 JPEG，页面的 img 元素就能显示连续画面。"""
    while not program.stop_event.is_set():
        frame = get_latest_jpeg(program)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(stream_interval(program))
