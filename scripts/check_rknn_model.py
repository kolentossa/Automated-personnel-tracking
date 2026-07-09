#!/usr/bin/env python3
"""Load the configured RKNN YOLO model and run one smoke-test inference."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera import CameraSource  # noqa: E402
from app.config import load_config, project_path  # noqa: E402
from app.detectors.rknn_yolo import RKNNYoloDetector  # noqa: E402


def main() -> int:
    config = load_config()
    detector_config = dict(config.get("detector") or {})
    model_path_value = str(detector_config.get("model_path") or "").strip()
    if not model_path_value:
        print("error: config.yaml detector.model_path is empty", file=sys.stderr)
        return 2

    model_path = project_path(model_path_value)
    if not model_path.exists():
        print(f"error: RKNN model does not exist: {model_path}", file=sys.stderr)
        return 2

    try:
        from rknnlite.api import RKNNLite  # noqa: F401
    except Exception as exc:
        print(f"error: RKNNLite import failed: {exc}", file=sys.stderr)
        return 3

    try:
        detector = RKNNYoloDetector(
            model_path,
            model_family=str(detector_config.get("model_family") or "yolov8"),
            input_size=int(detector_config.get("input_size") or 640),
            confidence_threshold=float(detector_config.get("confidence_threshold") or 0.35),
            nms_threshold=float(detector_config.get("nms_threshold") or 0.45),
            class_filter=detector_config.get("class_filter") or ["person"],
        )
    except Exception as exc:
        print(f"error: RKNN detector load failed: {exc}", file=sys.stderr)
        return 4

    frame, frame_source = _read_one_frame(config)
    try:
        started = time.monotonic()
        detections = detector.detect(frame)
        total_ms = round((time.monotonic() - started) * 1000.0, 1)
    except Exception as exc:
        print(f"error: RKNN inference failed: {exc}", file=sys.stderr)
        return 5
    finally:
        detector.release()

    profile = detector.last_profile
    print(f"detector name: {detector.name}")
    print(f"model_path: {model_path}")
    print(f"model_family: {detector.model_family}")
    print(f"npu_enabled: {str(detector.npu_enabled).lower()}")
    print(f"input_size: {detector.input_size}")
    print(f"frame_source: {frame_source}")
    print(f"preprocess_ms: {profile.get('preprocess_ms', 0.0):.1f}")
    print(f"inference_ms: {profile.get('inference_ms', 0.0):.1f}")
    print(f"postprocess_ms: {profile.get('postprocess_ms', 0.0):.1f}")
    print(f"total_smoke_ms: {total_ms:.1f}")
    print(f"boxes count: {len(detections)}")
    print("first boxes:")
    print(json.dumps([detection.as_dict() for detection in detections[:5]], ensure_ascii=False, indent=2))
    if not detector.npu_enabled:
        print("error: detector loaded but npu_enabled is false", file=sys.stderr)
        return 6
    return 0


def _read_one_frame(config: dict):
    camera = CameraSource(config["camera"])
    try:
        for _ in range(10):
            packet = camera.read()
            if packet is not None:
                return packet.frame, packet.source
            time.sleep(0.1)
    finally:
        camera.release()
    height = int(config.get("camera", {}).get("height") or 720)
    width = int(config.get("camera", {}).get("width") or 1280)
    return np.full((height, width, 3), 114, dtype=np.uint8), "blank-smoke-frame"


if __name__ == "__main__":
    raise SystemExit(main())
