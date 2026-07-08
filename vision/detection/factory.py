"""Detector factory for demo and future model-backed deployments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vision.detection.detector import Detector
from vision.detection.motion_detector import MotionPersonDetector
from vision.detection.yolo_detector import OpenCVDNNYoloDetector


def create_detector(model_path: Optional[str] = None) -> Detector:
    """Create a detector from a local model path, or use the demo fallback."""

    if model_path:
        path = Path(model_path).expanduser()
        if path.exists() and path.suffix.lower() == ".onnx":
            return OpenCVDNNYoloDetector(str(path))
    return MotionPersonDetector()
