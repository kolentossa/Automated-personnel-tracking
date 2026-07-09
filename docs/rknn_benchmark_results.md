# RKNN YOLOv8n Benchmark Results

Date: 2026-07-09

## Model

```text
model: models/yolov8n.rknn
detector: rknn-yolov8n
npu_enabled: true
```

## Summary

| Metric | Before optimization | After optimization |
| --- | ---: | ---: |
| Average FPS | 18.00 | 28.07 |
| Average total latency | 54.9 ms | 35.2 ms |
| P50 latency | not recorded | 33.3 ms |
| P95 latency | not recorded | 50.0 ms |
| Inference | 23.3 ms | 23.0 ms |
| Postprocess | 17.1 ms | 0.3 ms |
| Encode | 9.2 ms | 6.6 ms |

## Optimized 300-frame Run

```text
frames: 300
detector: rknn-yolov8n
npu_enabled: true
average_fps: 28.07
p50_latency_ms: 33.3
p95_latency_ms: 50.0
min_latency_ms: 27.1
max_latency_ms: 69.8
avg_capture_ms: 2.1
avg_queue_wait_ms: 0.0
avg_preprocess_ms: 2.7
avg_inference_ms: 23.0
avg_postprocess_ms: 0.3
avg_tracking_ms: 0.0
avg_privacy_ms: 0.6
avg_draw_ms: 0.0
avg_encode_ms: 6.6
avg_total_latency_ms: 35.2
```

## Notes

- YOLOv8 postprocess now filters COCO class `person` first and decodes only candidate positions.
- Web streaming keeps detection on the camera frame but resizes the outgoing MJPEG frame to `960x540`.
- `queue_wait_ms` stayed at `0.0`, confirming the latest-frame pipeline is not accumulating frame backlog during the benchmark.
- Remaining dominant cost is RKNN inference, followed by MJPEG resize/encode.
