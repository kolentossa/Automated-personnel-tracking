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
            "face_mosaic_padding": 0.1,
            "face_tracking_enabled": True,
            "head_fallback_enabled": False,
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
        or async_status["face_privacy_mode"] != "face-tracked"
        or not np.any(frame != async_result)
    ):
        print(f"error: asynchronous face processor did not mosaic a detected face: {async_status}", file=sys.stderr)
        return 6

    expected_x = (face_boxes[0][0] + face_boxes[0][2]) / 2.0
    expected_y = (face_boxes[0][1] + face_boxes[0][3]) / 2.0
    motion_errors = []
    for step in range(1, 9):
        shift_x = step * 12
        transform = np.asarray(((1.0, 0.0, shift_x), (0.0, 1.0, 0.0)), dtype=np.float32)
        moving_frame = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))
        moving_result = processor.process(moving_frame, [(0, 0, frame.shape[1], frame.shape[0])])
        moving_diff = np.any(moving_frame != moving_result, axis=2).astype(np.uint8)
        changed_points = cv2.findNonZero(moving_diff)
        if changed_points is None:
            print(f"error: optical-flow mosaic disappeared at movement step {step}", file=sys.stderr)
            return 7
        x, y, width, height = cv2.boundingRect(changed_points)
        actual_center = np.asarray((x + width / 2.0, y + height / 2.0))
        expected_center = np.asarray((expected_x + shift_x, expected_y))
        motion_errors.append(float(np.linalg.norm(actual_center - expected_center)))
    moving_status = processor.snapshot()
    if max(motion_errors) > 20.0 or moving_status["face_tracked_boxes"] < 1:
        print(
            f"error: optical-flow face tracking drifted: max_error={max(motion_errors):.1f}, status={moving_status}",
            file=sys.stderr,
        )
        return 8

    print(f"face_detector: retinaface-mobile320-onnx")
    print(f"faces_detected: {len(face_boxes)}")
    print(f"face_boxes: {face_boxes}")
    print(f"face_detection_ms: {detector.last_inference_ms:.1f}")
    print(f"mosaic_changed_pixels: {changed_inside}")
    print("fixed_head_fallback_used: false")
    print(f"async_face_privacy_mode: {async_status['face_privacy_mode']}")
    print(f"async_faces_detected: {async_status['faces_detected']}")
    print(f"motion_steps: {len(motion_errors)}")
    print(f"motion_shift_px: {len(motion_errors) * 12}")
    print(f"motion_max_center_error_px: {max(motion_errors):.1f}")
    print(f"face_tracking_ms: {moving_status['face_tracking_ms']:.1f}")
    print("status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
