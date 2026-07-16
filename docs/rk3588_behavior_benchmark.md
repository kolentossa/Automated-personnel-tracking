# RK3588 No-Mosaic Behavior Benchmark

Measured on 2026-07-16 on the LubanCat RK3588 with the final feature branch.
Model hashes and configuration are fixed in `config.yaml` and
`models/behavior_model_manifest.json`.

## Configuration

```text
platform: aarch64 Linux 5.10.209-rk3588
camera: /dev/video11, GStreamer, 1280x720 NV12 -> BGR
stream: 800x450 JPEG quality 74
capture queue: latest frame only, capacity 1
primary: yolov8n.rknn, every 3 frames, NPU cores 0_1_2
phone context: yolo11n.rknn, scheduled person ROI, NPU cores 0_1_2
behavior: behavior_damoyolo_cigarette_int8.rknn, every 5 frames
face detection: disabled
mosaic: disabled
```

## Online Service

After restart, `/api/health` reported:

```text
status: ok
camera_status: online
detector: rknn-yolov8n
npu_enabled: true
phone_context_model_enabled: true
phone_context_model_name: rknn-yolo11n
behavior_model: rknn-damoyolo-cigarette-int8
behavior_model_npu_enabled: true
fps: 27.99
primary inference_ms: 8.8
behavior inference_ms: 6.0
total_latency_ms: 25.0
privacy_mode: no_mosaic
face_detection_enabled: false
mosaic_enabled: false
```

The metrics are frame-normalized rolling values. Camera capture overlaps the
processing worker and both buffers retain only the newest frame.

## 300-Frame Primary Pipeline

Command:

```bash
python scripts/benchmark_pipeline.py --frames 300
```

```text
average FPS: 28.45
P50 latency: 6.0 ms
P95 latency: 71.4 ms
minimum / maximum latency: 3.9 / 78.8 ms

capture: 13.2 ms
queue wait: 0.0 ms
preprocess: 1.0 ms
frame-normalized inference: 13.3 ms
postprocess: 0.6 ms
tracking: 0.0 ms
privacy: 0.0 ms
draw: 0.0 ms
encode: 5.5 ms
average total latency: 22.0 ms
```

The camera was already owned by the online service, so the benchmark's initial
auto-detection probes logged expected V4L2 busy warnings. It still completed
all 300 frames through the configured camera path.

## 300-Frame Full Behavior Pipeline

Command:

```bash
python scripts/benchmark_behavior_pipeline.py --frames 300 \
  --json-output data/private_accuracy_validation/behavior_pipeline_final.json
```

```text
frames: 300
average FPS: 26.21
average latency: 37.64 ms
P50 latency: 28.85 ms
P95 latency: 97.98 ms
process CPU: 137.22 percent
peak RSS: 295.78 MB
NPU load: 100 percent average
```

Average frame-normalized stages:

| Stage | Milliseconds |
| --- | ---: |
| capture | 10.19 |
| preprocess | 1.07 |
| primary inference | 10.06 |
| postprocess | 0.59 |
| tracking | 0.05 |
| behavior inference | 6.73 |
| behavior analysis | 0.23 |
| privacy | 0.00 |
| encode | 6.82 |
| total | 37.64 |

Actual inference runs, excluding reused frames:

```text
primary: 100 runs, mean 30.19 ms, P50 26.0 ms, P95 53.0 ms
behavior: 60 runs, mean 33.63 ms, P50 32.0 ms, P95 42.8 ms
```

## Acceptance

The full pipeline passes the required average FPS >=20 and P95 latency <=120
ms while all NPU models and behavior logic are enabled. Privacy time is zero
because face detection and masking are intentionally absent, not because the
benchmark disabled a normally active stage.

## 30-Minute Stability

Command:

```bash
python scripts/monitor_behavior_stability.py \
  --duration-seconds 1800 --interval-seconds 10 \
  --url http://127.0.0.1:8001/api/health \
  --pid-file run/behavior-8001-supervisor.pid \
  --output data/private_accuracy_validation/stability_final_30m.json
```

```text
observed duration / samples: 1800.01 seconds / 181
supervisor / worker PID: 34541 / 34545, unchanged
API errors / camera offline / restarts: 0 / 0 / 0

FPS mean / min / P50 / P95 / max:
28.118 / 23.03 / 28.34 / 29.68 / 30.99
latency mean / P50 / P95 / max:
26.688 / 25.7 / 39.9 / 42.5 ms
primary inference mean / P95: 9.18 / 13.4 ms
behavior inference mean / P95: 6.093 / 8.5 ms
CPU mean / P95: 129.37 / 132.05 percent
NPU load: 100 percent average

RSS mean / min / P95 / max:
261.316 / 236.207 / 287.207 / 304.645 MB
RSS start-to-end growth: -20.528 MB
RSS fitted slope: 6.886 MB/hour
memory_stable: true

captured / processed / latest-frame dropped:
53,455 / 50,542 / 2,913
queue maximum depth: 1
camera failures / reconnects: 0 / 0
event dropped / write failures / max queue: 0 / 0 / 0
```

The run included the online Web service plus short periods of concurrent NPU
accuracy auditing, so its latency distribution is a conservative stress result.
All acceptance conditions passed: average FPS >=20, P95 <=120 ms, no crash or
restart, no camera failure, bounded queue, and no sustained RSS growth. Raw
samples remain in the ignored stability JSON file.
