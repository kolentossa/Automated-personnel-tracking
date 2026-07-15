#!/usr/bin/env python3
"""Monitor the supervised behavior Web service for a fixed stability window."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--url", default="http://127.0.0.1:8001/api/health")
    parser.add_argument("--pid-file", default="run/behavior-8001-supervisor.pid")
    parser.add_argument("--output", default="data/stability-behavior-rk3588.json")
    parser.add_argument("--max-rss-growth-mb", type=float, default=64.0)
    parser.add_argument("--max-rss-slope-mb-per-hour", type=float, default=64.0)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        parser.error("duration and interval must be positive")

    pid_file = _project_path(args.pid_file)
    started = time.monotonic()
    samples: List[Dict[str, Any]] = []
    previous_cpu: Optional[tuple[int, float, float]] = None
    while True:
        sample_started = time.monotonic()
        elapsed = sample_started - started
        sample: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
        }
        try:
            health = _read_json(args.url)
            sample.update(_health_fields(health))
            sample["api_error"] = ""
        except Exception as exc:
            sample["api_error"] = str(exc)

        supervisor_pid = _read_pid(pid_file)
        worker_pid = _first_child(supervisor_pid)
        sample["supervisor_pid"] = supervisor_pid
        sample["worker_pid"] = worker_pid
        sample["supervisor_alive"] = _pid_alive(supervisor_pid)
        sample["worker_alive"] = _pid_alive(worker_pid)
        sample["rss_mb"] = _rss_mb(worker_pid)
        cpu_ticks = _cpu_ticks(worker_pid)
        if worker_pid and cpu_ticks is not None:
            now = time.monotonic()
            if previous_cpu and previous_cpu[0] == worker_pid:
                _, old_ticks, old_time = previous_cpu
                elapsed_cpu = max(0.001, now - old_time)
                sample["cpu_percent"] = round(
                    (cpu_ticks - old_ticks) / float(os.sysconf("SC_CLK_TCK")) / elapsed_cpu * 100.0,
                    2,
                )
            previous_cpu = (worker_pid, cpu_ticks, now)
        sample["npu_load_percent"] = _read_npu_load()
        samples.append(sample)

        if elapsed >= args.duration_seconds:
            break
        sleep_seconds = min(
            args.interval_seconds,
            max(0.0, args.duration_seconds - (time.monotonic() - started)),
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    summary = _summarise(
        samples,
        time.monotonic() - started,
        args.max_rss_growth_mb,
        args.max_rss_slope_mb_per_hour,
    )
    result = {
        "configuration": {
            "url": args.url,
            "duration_seconds_requested": args.duration_seconds,
            "interval_seconds": args.interval_seconds,
            "max_rss_growth_mb": args.max_rss_growth_mb,
            "max_rss_slope_mb_per_hour": args.max_rss_slope_mb_per_hour,
        },
        "summary": summary,
        "samples": samples,
    }
    output = _project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    failed = (
        summary["api_error_count"] > 0
        or summary["status_error_count"] > 0
        or summary["camera_offline_count"] > 0
        or summary["supervisor_dead_count"] > 0
        or summary["worker_dead_count"] > 0
        or summary["worker_restart_count"] > 0
        or not summary["memory_stable"]
    )
    return 1 if failed else 0


def _health_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "status",
        "camera_status",
        "fps",
        "total_latency_ms",
        "inference_ms",
        "behavior_inference_ms",
        "capture_queue_depth",
        "camera_frames_captured",
        "camera_frames_processed",
        "camera_frames_dropped",
        "camera_reconnect_count",
        "camera_read_failures",
        "behavior_event_queue_depth",
        "behavior_event_dropped",
        "behavior_event_write_failures",
        "heap_trim_count",
        "heap_trim_released_count",
        "heap_gc_run_count",
        "heap_gc_collected_count",
        "heap_trim_error",
        "npu_enabled",
        "behavior_model_npu_enabled",
        "phone_detection_available",
        "smoking_detection_available",
    )
    return {key: data.get(key) for key in fields}


def _summarise(
    samples: List[Dict[str, Any]],
    duration_seconds: float,
    max_rss_growth_mb: float,
    max_rss_slope_mb_per_hour: float,
) -> Dict[str, Any]:
    worker_pids = [int(item["worker_pid"]) for item in samples if item.get("worker_pid")]
    supervisor_pids = [int(item["supervisor_pid"]) for item in samples if item.get("supervisor_pid")]
    summary: Dict[str, Any] = {
        "duration_seconds_observed": round(duration_seconds, 2),
        "sample_count": len(samples),
        "api_error_count": sum(bool(item.get("api_error")) for item in samples),
        "status_error_count": sum(item.get("status") != "ok" for item in samples),
        "camera_offline_count": sum(item.get("camera_status") != "online" for item in samples),
        "supervisor_dead_count": sum(not item.get("supervisor_alive") for item in samples),
        "worker_dead_count": sum(not item.get("worker_alive") for item in samples),
        "supervisor_restart_count": max(0, len(_dedupe(supervisor_pids)) - 1),
        "worker_restart_count": max(0, len(_dedupe(worker_pids)) - 1),
        "supervisor_pids": _dedupe(supervisor_pids),
        "worker_pids": _dedupe(worker_pids),
    }
    for field in (
        "fps",
        "total_latency_ms",
        "inference_ms",
        "behavior_inference_ms",
        "cpu_percent",
        "rss_mb",
        "npu_load_percent",
    ):
        summary[field] = _numeric_summary(samples, field)
    for field in (
        "capture_queue_depth",
        "behavior_event_queue_depth",
    ):
        values = _numbers(samples, field)
        summary[field + "_maximum"] = max(values) if values else None
    for field in (
        "camera_frames_captured",
        "camera_frames_processed",
        "camera_frames_dropped",
        "camera_reconnect_count",
        "camera_read_failures",
        "behavior_event_dropped",
        "behavior_event_write_failures",
        "heap_trim_count",
        "heap_trim_released_count",
        "heap_gc_run_count",
        "heap_gc_collected_count",
    ):
        values = _numbers(samples, field)
        summary[field + "_start"] = values[0] if values else None
        summary[field + "_end"] = values[-1] if values else None
        summary[field + "_delta"] = values[-1] - values[0] if values else None
    rss = _numbers(samples, "rss_mb")
    elapsed = _numbers(samples, "elapsed_seconds")
    summary["rss_growth_mb"] = round(rss[-1] - rss[0], 3) if rss else None
    summary["rss_slope_mb_per_hour"] = _linear_slope_per_hour(elapsed[-len(rss):], rss) if rss else None
    summary["memory_stable"] = bool(
        rss
        and summary["rss_growth_mb"] <= max_rss_growth_mb
        and summary["rss_slope_mb_per_hour"] is not None
        and summary["rss_slope_mb_per_hour"] <= max_rss_slope_mb_per_hour
    )
    return summary


def _numeric_summary(samples: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    values = _numbers(samples, field)
    if not values:
        return {"count": 0, "min": None, "average": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "average": round(statistics.fmean(values), 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "max": round(max(values), 3),
    }


def _numbers(samples: List[Dict[str, Any]], field: str) -> List[float]:
    result = []
    for sample in samples:
        value = sample.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _percentile(ordered: List[float], percentile: int) -> float:
    index = max(0, min(len(ordered) - 1, int(round(percentile / 100.0 * (len(ordered) - 1)))))
    return ordered[index]


def _linear_slope_per_hour(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        return None
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator
    return round(slope * 3600.0, 3)


def _read_json(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5.0) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read())


def _read_pid(path: Path) -> Optional[int]:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value if value > 0 else None


def _first_child(pid: Optional[int]) -> Optional[int]:
    if not pid:
        return None
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    children = [int(value) for value in text.split() if value.isdigit()]
    return children[0] if children else None


def _pid_alive(pid: Optional[int]) -> bool:
    return bool(pid and Path(f"/proc/{pid}").exists())


def _rss_mb(pid: Optional[int]) -> Optional[float]:
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(float(line.split()[1]) / 1024.0, 3)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cpu_ticks(pid: Optional[int]) -> Optional[float]:
    if not pid:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return float(fields[13]) + float(fields[14])
    except (OSError, ValueError, IndexError):
        return None


def _read_npu_load() -> Optional[float]:
    candidates = (
        Path("/sys/kernel/debug/rknpu/load"),
        Path("/sys/class/devfreq/fdab0000.npu/load"),
        Path("/sys/class/devfreq/fdab0000.npu/device/load"),
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
            return round(statistics.fmean(values), 2)
    return None


def _dedupe(values: List[int]) -> List[int]:
    result = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
