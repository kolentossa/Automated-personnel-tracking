# RK3588 Behavior Pipeline Benchmark Results

Measured on 2026-07-15 on the target LubanCat RK3588. These values apply only
to the model hashes and configuration below.

## Test Configuration

```text
platform: aarch64 Linux 5.10.209-rk3588
camera: /dev/video11 via GStreamer
input: 1280x720 NV12 converted to BGR
Web stream: 800x450 JPEG quality 74
primary detector: yolov8n.rknn, NPU, every 3 frames
primary SHA-256: ff3a64e6fe180203128c8d42456b458d208d3a1e2217d63683af00d6194e82ea
primary classes: person, cell phone
face privacy: RetinaFace ONNX plus optical flow
custom smoking model: disabled and not present
frames: 300
```

## Full Web Runtime

The actual two-thread latest-frame Web runtime was started on temporary port
8002 and sampled from `/api/health` after reaching a stable online state:

```text
fps: 29.99
capture_ms: 33.3 (reported separately)
frame-normalized inference_ms: 7.4
behavior_analysis_ms: 0.2
privacy_ms: 1.0
encode_ms: 5.4
rolling processing total_latency_ms: 16.7
detector: rknn-yolov8n
npu_enabled: true
phone_detection_available: true
smoking_detection_available: false
behavior_status: degraded
```

The Web runtime `total_latency_ms` starts at the timestamp of the frame
delivered by the camera and excludes the blocking capture wait. Capture and
processing overlap in separate latest-frame threads.

## Sequential 300-Frame Benchmark

Command:

```bash
python scripts/benchmark_behavior_pipeline.py --frames 300 \
  --json-output data/benchmark-behavior-rk3588.json
```

Measured output:

```text
average FPS: 27.33
average end-to-end latency: 36.16 ms
P95 end-to-end latency: 71.85 ms
process CPU: 119.01 percent
peak memory: 260.42 MB
NPU load: 100.0 percent average, 30 readable samples
```

Average stages:

```text
capture: 20.31 ms
preprocess: 0.94 ms
frame-normalized primary inference: 8.74 ms
postprocess: 0.10 ms
tracking: 0.03 ms
privacy: 0.57 ms
behavior analysis: 0.11 ms
auxiliary behavior inference: 0.00 ms (model disabled)
stream resize and encode: 4.99 ms
```

The sequential benchmark includes camera waiting and cannot overlap capture
with processing, so its FPS and latency are not interchangeable with the Web
runtime metrics. Both are recorded to keep the measurement definitions clear.

## Interpretation

Adding phone target association and both temporal state machines costs about
0.1 to 0.2 ms per frame in the tested empty-camera scene. The primary YOLO NPU
inference cost is frame-normalized because inference runs every third frame and
detections are reused between runs. No performance claim is made for the absent
custom smoking model. Re-run the same benchmark after installing that model,
and record its class order, hash, core mask, interval, and representative scene.
