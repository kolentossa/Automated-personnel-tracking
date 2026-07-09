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
from app.privacy import apply_privacy_mosaic  # noqa: E402
from app.stats import TIMING_KEYS  # noqa: E402
from app.tracker import PersonTracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RK3588 person-tracking pipeline latency.")
    parser.add_argument("--frames", type=int, default=300, help="Number of processed frames to benchmark.")
    args = parser.parse_args()

    config = load_config()
    camera = CameraSource(config["camera"])
    detector = PersonDetector(config.get("detector") or config.get("detection", {}), fallback_config=config.get("detection", {}))
    tracker = PersonTracker(config["tracking"])
    privacy_config = config["privacy"]
    if detector.name in {"no-op-person-detector", "motion-person-detector"}:
        print(f"detector: {detector.name}")
        print(f"npu_enabled: {str(detector.npu_enabled).lower()}")
        if detector.warning:
            print(f"warning: {detector.warning}")
        print("error: refusing to benchmark placeholder detector", file=sys.stderr)
        return 2

    timings: Dict[str, List[float]] = {key: [] for key in TIMING_KEYS}
    processed = 0
    started = time.monotonic()

    try:
        while processed < args.frames:
            capture_started = time.monotonic()
            packet = camera.read()
            capture_ms = _ms(time.monotonic() - capture_started)
            if packet is None:
                print(f"waiting for camera: {camera.error}", file=sys.stderr)
                time.sleep(0.2)
                continue

            detections = detector.detect(packet.frame)
            detector_profile = dict(detector.last_profile)

            tracking_started = time.monotonic()
            tracks = tracker.update(detections)
            tracking_ms = _ms(time.monotonic() - tracking_started)

            privacy_started = time.monotonic()
            display = apply_privacy_mosaic(
                packet.frame,
                person_boxes=[track.bbox for track in tracks] or [detection.bbox for detection in detections],
                face_mosaic_enabled=bool(privacy_config.get("face_mosaic_enabled", True)),
                head_fallback_enabled=bool(privacy_config.get("head_fallback_enabled", True)),
                mosaic_strength=int(privacy_config.get("mosaic_strength") or 14),
            )
            privacy_ms = _ms(time.monotonic() - privacy_started)

            draw_started = time.monotonic()
            _draw_tracks(display, tracks)
            draw_ms = _ms(time.monotonic() - draw_started)

            encode_started = time.monotonic()
            ok, _ = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            encode_ms = _ms(time.monotonic() - encode_started)
            if not ok:
                raise RuntimeError("Could not encode JPEG frame")

            total_latency_ms = _ms(time.monotonic() - packet.captured_at)
            values = {
                "capture_ms": capture_ms,
                "preprocess_ms": detector_profile.get("preprocess_ms", 0.0),
                "inference_ms": detector_profile.get("inference_ms", 0.0),
                "postprocess_ms": detector_profile.get("postprocess_ms", 0.0),
                "tracking_ms": tracking_ms,
                "privacy_ms": privacy_ms,
                "draw_ms": draw_ms,
                "encode_ms": encode_ms,
                "total_latency_ms": total_latency_ms,
            }
            for key, value in values.items():
                timings[key].append(float(value))
            processed += 1
    finally:
        camera.release()

    elapsed = time.monotonic() - started
    fps = processed / max(0.001, elapsed)
    latencies = timings["total_latency_ms"]
    if not latencies:
        print("error: no frames processed", file=sys.stderr)
        return 3
    print(f"frames: {processed}")
    print(f"detector: {detector.name}")
    print(f"npu_enabled: {str(detector.npu_enabled).lower()}")
    print(f"average_fps: {fps:.2f}")
    print(f"p50_latency_ms: {_percentile(latencies, 50):.1f}")
    print(f"p95_latency_ms: {_percentile(latencies, 95):.1f}")
    print(f"min_latency_ms: {min(latencies):.1f}")
    print(f"max_latency_ms: {max(latencies):.1f}")
    for key in (
        "capture_ms",
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
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (42, 166, 85), 2)


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)


if __name__ == "__main__":
    raise SystemExit(main())
