#!/usr/bin/env python3
"""Verify that RetinaFace detections, rather than fixed head boxes, drive mosaic."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.privacy import FaceMosaicProcessor, RetinaFaceONNXDetector, apply_privacy_mosaic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Test real face detection and face-box mosaic.")
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "data" / "retinaface_test.jpg")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "RetinaFace_mobile320.onnx")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"error: could not read test image: {args.image}", file=sys.stderr)
        return 2

    detector = RetinaFaceONNXDetector(args.model, confidence_threshold=0.6, num_threads=2)
    face_boxes = detector.detect(frame)
    if not face_boxes:
        print("error: RetinaFace returned no face boxes", file=sys.stderr)
        return 3

    mosaiced = apply_privacy_mosaic(
        frame,
        face_boxes=face_boxes,
        face_mosaic_enabled=True,
        head_fallback_enabled=False,
        mosaic_strength=14,
    )
    changed = np.any(frame != mosaiced, axis=2)
    allowed = np.zeros(frame.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in face_boxes:
        allowed[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)] = True

    changed_inside = int(np.count_nonzero(changed & allowed))
    changed_outside = int(np.count_nonzero(changed & ~allowed))
    if changed_inside == 0 or changed_outside != 0:
        print(
            f"error: unexpected mosaic diff inside={changed_inside} outside={changed_outside}",
            file=sys.stderr,
        )
        return 4

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), mosaiced):
            print(f"error: could not write output: {args.output}", file=sys.stderr)
            return 5

    processor = FaceMosaicProcessor(
        {
            "face_mosaic_enabled": True,
            "face_detector": "retinaface-onnx",
            "face_model_path": str(args.model),
            "face_detect_every_n_frames": 1,
            "head_fallback_enabled": True,
        }
    )
    processor.start()
    async_result = frame.copy()
    try:
        for _ in range(50):
            async_result = processor.process(frame, [(0, 0, frame.shape[1], frame.shape[0])])
            if processor.snapshot()["faces_detected"] > 0:
                async_result = processor.process(frame, [(0, 0, frame.shape[1], frame.shape[0])])
                break
            time.sleep(0.02)
    finally:
        processor.stop()
    async_status = processor.snapshot()
    if (
        async_status["faces_detected"] == 0
        or async_status["face_privacy_mode"] != "face-detected"
        or not np.any(frame != async_result)
    ):
        print(f"error: asynchronous face processor did not mosaic a detected face: {async_status}", file=sys.stderr)
        return 6

    print(f"face_detector: retinaface-mobile320-onnx")
    print(f"faces_detected: {len(face_boxes)}")
    print(f"face_boxes: {face_boxes}")
    print(f"face_detection_ms: {detector.last_inference_ms:.1f}")
    print(f"mosaic_changed_pixels: {changed_inside}")
    print("fixed_head_fallback_used: false")
    print(f"async_face_privacy_mode: {async_status['face_privacy_mode']}")
    print(f"async_faces_detected: {async_status['faces_detected']}")
    print("status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
