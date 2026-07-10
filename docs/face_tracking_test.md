# Face Mosaic Motion Tracking Test

Date: 2026-07-10
Target: RK3588 / LubanCat, branch `optimize-rknn-yolo-npu`

## Functional Verification

The Rockchip RetinaFace test image was shifted horizontally by 12 pixels per
frame for eight frames after the initial real face detection. Head fallback was
disabled for this check so only the optical-flow face box could produce mosaic.

```text
faces_detected: 1
motion_steps: 8
motion_shift_px: 96
motion_max_center_error_px: 0.0
face_tracking_ms: 4.8
status: ok
```

## Full Pipeline Benchmark

A 360-frame, 30 FPS local video moved the same real-person image by 14 pixels
per frame. The benchmark included RKNN YOLO person detection, person tracking,
RetinaFace refresh every five frames, per-frame optical flow, mosaic, drawing,
resize, and JPEG encoding.

```text
frames: 300
average_fps: 42.46
avg_total_latency_ms: 20.4
avg_privacy_ms: 9.3
face_detection_ms: 37.4
face_tracking_ms: 9.4
face_tracked_boxes: 1
face_fallback_regions: 0
```

The ignored input video was `data/privacy_motion_test.mp4`. No image, video,
ONNX, or RKNN artifact is included in Git.
