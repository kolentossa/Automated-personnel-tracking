#!/usr/bin/env python3
"""Benchmark the behavior-enabled pipeline on an RK3588 or local video."""

from __future__ import annotations

import argparse
import json
import re
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.behavior import BehaviorEngine, load_behavior_config  # noqa: E402
from app.behavior.detector import BehaviorObjectDetector  # noqa: E402
from app.camera import CameraSource  # noqa: E402
from app.config import load_config  # noqa: E402
from app.detector import PersonDetector  # noqa: E402
from app.privacy import FaceMosaicProcessor  # noqa: E402
from app.tracker import PersonTracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--video", help="Use a local video instead of the configured camera")
    parser.add_argument("--json-output", help="Optional path for the machine-readable result")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    config = load_config()
    behavior_config = load_behavior_config()
    camera_config = dict(config["camera"])
    if args.video:
        camera_config.update({"source_type": "video", "video_file": args.video, "auto_detect": False})

    camera = CameraSource(camera_config)
    detector = PersonDetector(config.get("detector", {}), fallback_config=config.get("detection", {}))
    behavior_detector = BehaviorObjectDetector(behavior_config.get("models", {}).get("behavior", {}))
    tracker = PersonTracker(config["tracking"])
    privacy = FaceMosaicProcessor(config["privacy"])
    behavior = BehaviorEngine(behavior_config)
    detect_interval = max(1, int(config.get("performance", {}).get("detect_every_n_frames") or 1))
    jpeg_quality = int(config.get("stream", {}).get("jpeg_quality") or 74)
    stream_width = int(config.get("stream", {}).get("width") or 0)
    stream_height = int(config.get("stream", {}).get("height") or 0)
    samples: Dict[str, List[float]] = {key: [] for key in (
        "capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
        "behavior_inference_ms", "tracking_ms", "privacy_ms", "behavior_analysis_ms",
        "encode_ms", "total_latency_ms",
    )}
    npu_samples: List[float] = []
    last_scene = []
    last_people = []
    detector_has_run = False
    frame_size = None
    cpu_started = _cpu_seconds()
    wall_started = time.monotonic()
    privacy.start()
    try:
        for frame_index in range(args.frames):
            frame_started = time.monotonic()
            capture_started = time.monotonic()
            packet = camera.read()
            samples["capture_ms"].append(_ms(time.monotonic() - capture_started))
            if packet is None:
                raise RuntimeError(camera.error or "Camera returned no frame")
            frame = packet.frame
            frame_size = (int(frame.shape[1]), int(frame.shape[0]))

            if frame_index % detect_interval == 0 or not detector_has_run:
                last_scene = detector.detect_scene(frame)
                last_people = [item for item in last_scene if item.label == "person" or item.class_id == 0]
                profile = detector.last_profile
                detector_has_run = True
            else:
                profile = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}
            for key in ("preprocess_ms", "inference_ms", "postprocess_ms"):
                samples[key].append(float(profile.get(key, 0.0) or 0.0))

            auxiliary = behavior_detector.detect(frame, frame_index)
            samples["behavior_inference_ms"].append(
                float(behavior_detector.last_profile.get("inference_ms", 0.0) or 0.0)
            )
            tracking_started = time.monotonic()
            tracks = tracker.update(last_people)
            samples["tracking_ms"].append(_ms(time.monotonic() - tracking_started))

            privacy_started = time.monotonic()
            protected = privacy.process(frame, [track.bbox for track in tracks])
            samples["privacy_ms"].append(_ms(time.monotonic() - privacy_started))
            result = behavior.update(tracks, list(last_scene) + auxiliary, privacy.face_boxes(), frame.shape[:2])
            samples["behavior_analysis_ms"].append(result.analysis_ms)

            encode_started = time.monotonic()
            stream_frame = protected
            if stream_width > 0 and stream_height > 0 and protected.shape[1::-1] != (stream_width, stream_height):
                stream_frame = cv2.resize(
                    protected,
                    (stream_width, stream_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            ok, _ = cv2.imencode(".jpg", stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            samples["encode_ms"].append(_ms(time.monotonic() - encode_started))
            samples["total_latency_ms"].append(_ms(time.monotonic() - frame_started))
            if frame_index % 10 == 0:
                npu_samples.extend(_read_npu_load())
    finally:
        privacy.stop()
        camera.release()
        detector.release()
        behavior_detector.release()

    wall_seconds = max(0.001, time.monotonic() - wall_started)
    cpu_seconds = max(0.0, _cpu_seconds() - cpu_started)
    result = {
        "platform": _platform_name(),
        "input_resolution": {"width": frame_size[0], "height": frame_size[1]} if frame_size else None,
        "stream_resolution": {"width": stream_width, "height": stream_height},
        "frames": args.frames,
        "average_fps": round(args.frames / wall_seconds, 2),
        "average_latency_ms": round(statistics.fmean(samples["total_latency_ms"]), 2),
        "p95_latency_ms": round(_percentile(samples["total_latency_ms"], 95), 2),
        "average_stage_ms": {key: round(statistics.fmean(values), 2) for key, values in samples.items() if values},
        "cpu_percent_process": round(cpu_seconds / wall_seconds * 100.0, 2),
        "peak_memory_mb": round(_peak_memory_mb(), 2),
        "npu_load_percent_average": round(statistics.fmean(npu_samples), 2) if npu_samples else None,
        "npu_load_samples": len(npu_samples),
        "detector": detector.name,
        "npu_enabled": detector.npu_enabled,
        "behavior_model": behavior_detector.snapshot(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output:
        output = Path(args.json_output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _read_npu_load() -> List[float]:
    candidates = (
        Path("/sys/kernel/debug/rknpu/load"),
        Path("/sys/class/devfreq/fdab0000.npu/load"),
        Path("/sys/class/devfreq/fdab0000.npu/device/load"),
        Path("/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/load"),
    )
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
        if not values:
            values = [float(value) for value in re.findall(r"(?:^|\s)(\d+(?:\.\d+)?)@\d+Hz", text)]
        if values:
            return values
    return []


def _percentile(values: List[float], percentile: int) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _peak_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _platform_name() -> str:
    import platform
    return f"{platform.machine()} {platform.system()} {platform.release()}"


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 2)


if __name__ == "__main__":
    raise SystemExit(main())
