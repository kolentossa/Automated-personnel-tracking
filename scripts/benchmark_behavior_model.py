#!/usr/bin/env python3
"""Benchmark the configured RKNN behavior detector on local images."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.behavior import load_behavior_config  # noqa: E402
from app.behavior.detector import BehaviorObjectDetector  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--model", type=Path, help="Override the configured behavior RKNN path")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--annotated-dir", type=Path)
    parser.add_argument("--save-limit", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    paths = _image_paths(args.images)
    if not paths:
        parser.error(f"No images found: {args.images}")
    loaded = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError(f"Unreadable image: {path}")
        loaded.append((path, image))

    config = dict(load_behavior_config()["models"]["behavior"])
    config.update({"enabled": True, "required": True, "detect_every_n_frames": 1})
    if args.model:
        config["model_path"] = str(args.model.resolve())
        config["expected_sha256"] = ""
    detector = BehaviorObjectDetector(config)
    if not detector.available or not detector.npu_enabled:
        raise RuntimeError(detector.error or "Behavior RKNN detector did not enable the NPU")

    stage_samples = {key: [] for key in ("preprocess_ms", "inference_ms", "postprocess_ms")}
    total_samples = []
    per_image = {}
    try:
        for index in range(args.warmup):
            detector.detect(loaded[index % len(loaded)][1], index)
        wall_started = time.perf_counter()
        for frame_index in range(args.frames):
            path, image = loaded[frame_index % len(loaded)]
            started = time.perf_counter()
            detections = detector.detect(image, frame_index)
            total_samples.append((time.perf_counter() - started) * 1000.0)
            for key in stage_samples:
                stage_samples[key].append(float(detector.last_profile.get(key, 0.0) or 0.0))
            if path.name not in per_image:
                per_image[path.name] = [
                    {
                        "label": item.label,
                        "confidence": round(float(item.confidence), 6),
                        "bbox": [round(float(value), 2) for value in item.bbox],
                    }
                    for item in detections
                ]
                if args.annotated_dir and len(per_image) <= args.save_limit:
                    _write_annotated(args.annotated_dir, path.name, image, detections)
        wall_seconds = max(1e-9, time.perf_counter() - wall_started)
    finally:
        detector.release()

    report = {
        "schema_version": 1,
        "model": detector.name,
        "model_path": config["model_path"],
        "npu_enabled": detector.npu_enabled,
        "input_images": len(loaded),
        "frames": args.frames,
        "average_fps": round(args.frames / wall_seconds, 3),
        "total_latency_ms": _summary(total_samples),
        "stages_ms": {key: _summary(values) for key, values in stage_samples.items()},
        "images_with_detections": sum(bool(items) for items in per_image.values()),
        "detections": per_image,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _image_paths(value: Path) -> list[Path]:
    if value.is_file() and value.suffix.lower() in IMAGE_SUFFIXES:
        return [value]
    if not value.is_dir():
        return []
    return sorted(
        (path for path in value.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def _write_annotated(output_dir: Path, name: str, image, detections) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = image.copy()
    for item in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in item.bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(
            result,
            f"{item.label} {item.confidence:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 220),
            2,
        )
    target = output_dir / name
    if not cv2.imwrite(str(target), result):
        raise RuntimeError(f"Could not write annotated image: {target}")


def _summary(values: list[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p95": round(_percentile(values, 95), 3),
        "minimum": round(min(values), 3),
        "maximum": round(max(values), 3),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


if __name__ == "__main__":
    raise SystemExit(main())
