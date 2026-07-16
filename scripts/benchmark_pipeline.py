#!/usr/bin/env python3
"""Benchmark the low-latency camera detection pipeline for 300 frames."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera import CameraSource  # noqa: E402
from app.config import load_config  # noqa: E402
from app.detector import PersonDetector  # noqa: E402
from app.stats import TIMING_KEYS  # noqa: E402
from app.tracker import PersonTracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark RK3588 person-tracking pipeline latency."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of processed frames to benchmark.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Warmup frames excluded from metrics.",
    )
    parser.add_argument(
        "--video", type=Path, help="Optional prerecorded input using the same pipeline."
    )
    args = parser.parse_args()

    config = load_config()
    camera_config = dict(config["camera"])
    if args.video:
        camera_config.update(
            {
                "source_type": "video",
                "video_file": str(args.video),
                "auto_detect": False,
            }
        )
    camera = CameraSource(camera_config)
    detector = PersonDetector(
        config.get("detector") or config.get("detection", {}),
        fallback_config=config.get("detection", {}),
    )
    tracker = PersonTracker(config["tracking"])
    stream_config = config.get("stream", {})
    jpeg_quality = max(35, min(95, int(stream_config.get("jpeg_quality") or 80)))
    stream_width = max(0, int(stream_config.get("width") or 0))
    stream_height = max(0, int(stream_config.get("height") or 0))
    detect_every_n_frames = max(
        1, int(config.get("performance", {}).get("detect_every_n_frames") or 1)
    )
    if detector.name in {"no-op-person-detector", "motion-person-detector"}:
        print(f"detector: {detector.name}")
        print(f"npu_enabled: {str(detector.npu_enabled).lower()}")
        if detector.warning:
            print(f"warning: {detector.warning}")
        print("error: refusing to benchmark placeholder detector", file=sys.stderr)
        return 2
    timings: Dict[str, List[float]] = {key: [] for key in TIMING_KEYS}
    processed = 0
    frame_index = 0
    detector_has_run = False
    last_detections = []
    started = 0.0

    try:
        while processed < args.frames:
            capture_started = time.monotonic()
            packet = camera.read()
            capture_ms = _ms(time.monotonic() - capture_started)
            if packet is None:
                print(f"waiting for camera: {camera.error}", file=sys.stderr)
                time.sleep(0.2)
                continue

            if frame_index == max(0, args.warmup_frames):
                started = time.monotonic()

            detect_this_frame = (
                frame_index % detect_every_n_frames == 0 or not detector_has_run
            )
            if detect_this_frame:
                detections = detector.detect(packet.frame)
                detector_profile = dict(detector.last_profile)
                last_detections = detections
                detector_has_run = True
            else:
                detections = list(last_detections)
                detector_profile = {
                    "preprocess_ms": 0.0,
                    "inference_ms": 0.0,
                    "postprocess_ms": 0.0,
                }

            tracking_started = time.monotonic()
            tracks = tracker.update(detections)
            tracking_ms = _ms(time.monotonic() - tracking_started)

            display = packet.frame.copy()
            privacy_ms = 0.0

            draw_started = time.monotonic()
            _draw_tracks(display, tracks)
            draw_ms = _ms(time.monotonic() - draw_started)

            encode_started = time.monotonic()
            stream_frame = _resize_for_stream(display, stream_width, stream_height)
            ok, _ = cv2.imencode(
                ".jpg", stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            encode_ms = _ms(time.monotonic() - encode_started)
            if not ok:
                raise RuntimeError("Could not encode JPEG frame")

            total_latency_ms = _ms(time.monotonic() - packet.captured_at)
            values = {
                "capture_ms": capture_ms,
                "queue_wait_ms": 0.0,
                "preprocess_ms": detector_profile.get("preprocess_ms", 0.0),
                "inference_ms": detector_profile.get("inference_ms", 0.0),
                "postprocess_ms": detector_profile.get("postprocess_ms", 0.0),
                "tracking_ms": tracking_ms,
                "privacy_ms": privacy_ms,
                "draw_ms": draw_ms,
                "encode_ms": encode_ms,
                "total_latency_ms": total_latency_ms,
            }
            if frame_index >= max(0, args.warmup_frames):
                for key, value in values.items():
                    timings[key].append(float(value))
                processed += 1
            frame_index += 1
    finally:
        camera.release()

    elapsed = time.monotonic() - started if started else 0.0
    fps = processed / max(0.001, elapsed)
    latencies = timings["total_latency_ms"]
    if not latencies:
        print("error: no frames processed", file=sys.stderr)
        return 3
    print(f"frames: {processed}")
    print(f"warmup_frames: {max(0, args.warmup_frames)}")
    print(f"detector: {detector.name}")
    print(f"npu_enabled: {str(detector.npu_enabled).lower()}")
    print("privacy_mode: no_mosaic")
    print("face_detection_enabled: false")
    print("mosaic_enabled: false")
    print(f"average_fps: {fps:.2f}")
    print(f"p50_latency_ms: {_percentile(latencies, 50):.1f}")
    print(f"p95_latency_ms: {_percentile(latencies, 95):.1f}")
    print(f"min_latency_ms: {min(latencies):.1f}")
    print(f"max_latency_ms: {max(latencies):.1f}")
    for key in (
        "capture_ms",
        "queue_wait_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "tracking_ms",
        "privacy_ms",
        "draw_ms",
        "encode_ms",
        "total_latency_ms",
    ):
        print(f"avg_{key}: {_mean(timings[key]):.1f}")
    if detector.warning:
        print(f"warning: {detector.warning}")
    return 0


def _percentile(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * (percentile / 100.0)))
    return float(ordered[max(0, min(len(ordered) - 1, index))])


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _draw_tracks(frame, tracks) -> None:
    for track in tracks:
        x1, y1, x2, y2 = [int(value) for value in track.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (42, 166, 85), 2)
        label = f"ID {track.track_id} {track.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (42, 166, 85),
            2,
        )


def _resize_for_stream(frame, width: int, height: int):
    if width <= 0 or height <= 0:
        return frame
    current_height, current_width = frame.shape[:2]
    if current_width == width and current_height == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)


if __name__ == "__main__":
    raise SystemExit(main())
