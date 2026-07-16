#!/usr/bin/env python3
"""Isolate native-memory growth in RK3588 pipeline components."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.behavior import load_behavior_config  # noqa: E402
from app.behavior.detector import BehaviorObjectDetector  # noqa: E402
from app.camera import CameraSource  # noqa: E402
from app.config import load_config  # noqa: E402
from app.detector import PersonDetector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("capture", "primary", "behavior", "all"))
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--warmup-frames", type=int, default=300)
    parser.add_argument("--sample-every-frames", type=int, default=300)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.frames, args.sample_every_frames) <= 0 or args.warmup_frames < 0:
        parser.error("frame counts must be positive and warmup must be non-negative")

    config = load_config()
    camera = CameraSource(config["camera"])
    primary = None
    behavior = None
    if args.component in {"primary", "all"}:
        primary = PersonDetector(
            config["detector"], fallback_config=config.get("detection", {})
        )
    if args.component in {"behavior", "all"}:
        behavior_config = load_behavior_config()["models"]["behavior"]
        behavior = BehaviorObjectDetector(behavior_config)
    trim = _malloc_trim()
    total_frames = args.warmup_frames + args.frames
    primary_interval = max(
        1, int(config.get("performance", {}).get("detect_every_n_frames") or 1)
    )
    samples: List[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for frame_index in range(total_frames + 1):
            packet = camera.read()
            if packet is None:
                raise RuntimeError(camera.error or "camera returned no frame")
            frame = packet.frame
            if primary is not None and frame_index % primary_interval == 0:
                primary.detect_scene(frame)
            if behavior is not None:
                behavior.detect(frame, frame_index)
            stream = cv2.resize(frame, (800, 450), interpolation=cv2.INTER_NEAREST)
            ok, _ = cv2.imencode(".jpg", stream, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
            if not ok:
                raise RuntimeError("JPEG encoding failed")

            measured_frame = frame_index - args.warmup_frames
            if measured_frame >= 0 and measured_frame % args.sample_every_frames == 0:
                collected = gc.collect()
                released = int(trim(0)) if trim is not None else 0
                samples.append(
                    {
                        "frame": measured_frame,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "rss_mb": _status_value_mb("VmRSS"),
                        "anonymous_mb": _rollup_value_mb("Pss_Anon"),
                        "gc_collected": collected,
                        "trim_released": bool(released),
                    }
                )
    finally:
        camera.release()
        if primary is not None:
            primary.release()
        if behavior is not None:
            behavior.release()

    elapsed = max(0.001, time.monotonic() - started)
    result = {
        "component": args.component,
        "warmup_frames": args.warmup_frames,
        "measured_frames": args.frames,
        "average_fps": round(total_frames / elapsed, 2),
        "rss_growth_mb": round(samples[-1]["rss_mb"] - samples[0]["rss_mb"], 3),
        "anonymous_growth_mb": round(
            samples[-1]["anonymous_mb"] - samples[0]["anonymous_mb"], 3
        ),
        "rss_slope_mb_per_hour": _slope(samples, "rss_mb"),
        "anonymous_slope_mb_per_hour": _slope(samples, "anonymous_mb"),
        "samples": samples,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


def _malloc_trim():
    if os.name != "posix":
        return None
    import ctypes

    trim = ctypes.CDLL(None).malloc_trim
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


def _status_value_mb(key: str) -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(key + ":"):
            return round(float(line.split()[1]) / 1024.0, 3)
    raise RuntimeError(f"missing {key} in /proc/self/status")


def _rollup_value_mb(key: str) -> float:
    for line in (
        Path("/proc/self/smaps_rollup").read_text(encoding="utf-8").splitlines()
    ):
        if line.startswith(key + ":"):
            return round(float(line.split()[1]) / 1024.0, 3)
    raise RuntimeError(f"missing {key} in /proc/self/smaps_rollup")


def _slope(samples: List[dict[str, Any]], field: str) -> float:
    x = [float(item["elapsed_seconds"]) for item in samples]
    y = [float(item[field]) for item in samples]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / max(
        1e-9, denominator
    )
    return round(slope * 3600.0, 3)


if __name__ == "__main__":
    raise SystemExit(main())
