# Counting Configuration Test

Date: 2026-07-09

Branch: `optimize-rknn-yolo-npu`

## Scope

Added live Web/API configuration for the counting line and ENTER direction.
This change does not modify RKNN detector logic, YOLO post-processing, or the
camera capture pipeline.

## API Test

Command:

```bash
.venv/bin/python scripts/test_counting_config.py --base-url http://127.0.0.1:8001
```

Result:

```text
GET config: ok
POST config: ok
GET confirms saved config: ok
config.yaml reload: ok
restore original config: ok
```

The test updates `/api/config/counting`, verifies that the value is saved in
`config.yaml`, reloads the config through `load_config()`, and restores the
original counting configuration afterwards.

## Health Check

Endpoint:

```bash
curl http://127.0.0.1:8001/api/health
```

Key result:

```text
status: ok
camera_status: online
source: /dev/video12
detector: rknn-yolov8n
npu_enabled: true
```

## Benchmark

Command:

```bash
.venv/bin/python scripts/benchmark_pipeline.py --frames 300
```

Result:

```text
frames: 300
detector: rknn-yolov8n
npu_enabled: true
average_fps: 27.29
p50_latency_ms: 33.6
p95_latency_ms: 51.3
min_latency_ms: 27.2
max_latency_ms: 106.1
avg_capture_ms: 2.4
avg_queue_wait_ms: 0.0
avg_preprocess_ms: 2.8
avg_inference_ms: 23.3
avg_postprocess_ms: 0.2
avg_tracking_ms: 0.0
avg_privacy_ms: 0.6
avg_draw_ms: 0.0
avg_encode_ms: 7.1
avg_total_latency_ms: 36.2
```

Previous RKNN benchmark:

```text
average_fps: 28.07
avg_inference_ms: 23.0
avg_total_latency_ms: 35.2
```

The small difference is within normal live camera variation while 8000 remains
running. The counting configuration feature does not add work to the detector
or NPU inference path.

## Service Note

Port 8000 was not replaced. Port 8001 was restarted as the test service to load
the new FastAPI routes. A stale old 8001 process was found holding
`/dev/video12`; it was stopped so the restarted 8001 service could read the
camera again.
