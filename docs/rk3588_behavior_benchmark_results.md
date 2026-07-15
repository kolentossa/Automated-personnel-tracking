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
behavior detector: behavior_damoyolo_cigarette_int8.rknn, NPU, every 5 frames
behavior SHA-256: d04c43a3a695c9985fbd03db1e0a2956763374fd686d949b8cd96cabdc7c5941
behavior classes: cigarette; second output score channel is explicitly unused
frames: 300
```

## Full Web Runtime

The actual supervised two-thread latest-frame Web runtime on port 8001 was
sampled from `/api/health` after reaching a stable online state:

```text
fps: 30.86
frame-normalized primary inference_ms: 7.3
frame-normalized behavior inference_ms: 5.8
rolling processing total_latency_ms: 21.9
detector: rknn-yolov8n
npu_enabled: true
phone_detection_available: true
behavior_model: rknn-damoyolo-cigarette-int8
behavior_model_npu_enabled: true
behavior_model_loaded: true
smoking_detection_available: true
behavior_status: ready
capture queue capacity/depth: 1 / 0
event queue depth/dropped/write failures: 0 / 0 / 0
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
average FPS: 26.43
average end-to-end latency: 37.26 ms
P50 end-to-end latency: 33.41 ms
P95 end-to-end latency: 92.79 ms
process CPU: 133.92 percent
peak memory: 304.61 MB
NPU load: 100.0 percent average, 30 readable samples
```

Average stages:

```text
capture: 13.44 ms
preprocess: 0.86 ms
frame-normalized primary inference: 8.64 ms
postprocess: 0.15 ms
tracking: 0.04 ms
privacy: 0.66 ms
behavior analysis: 0.15 ms
frame-normalized behavior inference: 5.96 ms
stream resize and encode: 6.38 ms
```

Actual NPU runs, excluding skipped frames:

```text
primary person/phone: 100 runs, mean 25.93 ms, P50 25.40 ms, P95 32.30 ms
behavior cigarette:    60 runs, mean 29.79 ms, P50 30.10 ms, P95 32.50 ms
```

All 300 frames were captured and processed in the sequential harness, with
queue depth 0, dropped frames 0, one successful camera open, and no read
failure. Separately, a real `/dev/video11` release/reopen cycle returned valid
frames before and after release and reported one reconnect.

The sequential benchmark includes camera waiting and cannot overlap capture
with processing, so its FPS and latency are not interchangeable with the Web
runtime metrics. Both are recorded to keep the measurement definitions clear.

## Interpretation

Both NPU models, phone/smoking association, face privacy, temporal state
machines, and JPEG encoding were enabled. Primary and behavior inference costs
are frame-normalized because detections are reused between NPU runs. The result
passes the deployment targets of at least 20 FPS and no more than 120 ms P95.

## Stability Monitor

Run the supervised service continuously while collecting process and pipeline
metrics:

```bash
python scripts/monitor_behavior_stability.py \
  --duration-seconds 1800 --interval-seconds 10 \
  --output data/stability-behavior-rk3588.json
```

The monitor exits nonzero on API errors, camera-offline samples, dead
supervisor/worker processes, or a worker restart. Raw samples remain under the
ignored `data/` directory. The completed 30-minute summary is recorded below.

## Final 30-Minute Stability Result

The first 30-minute run exposed real memory instability: RSS grew by 233.32 MB
with a fitted 434.27 MB/hour slope. Component isolation showed capture/JPEG,
each RKNN detector, and RetinaFace/optical flow were stable separately, while
the combined two-RKNN path accumulated cyclic wrapper objects between default
GC passes. This failed run was not accepted.

The production runtime now performs explicit GC and glibc heap trimming every
300 processed frames, while the project-local supervisor sets
`MALLOC_ARENA_MAX=2`. A 3000-frame combined profile then changed RSS by only
0.40 MB and anonymous memory by 0.10 MB. The final supervised run used the same
already-warmed worker for the full window:

```text
duration: 1800.01 seconds
samples: 181 at 10-second intervals
supervisor / worker: 21416 / 21420, unchanged
API errors / camera-offline samples / worker restarts: 0 / 0 / 0

Web FPS: mean 27.798, min 25.27, P50 27.89, P95 29.31, max 30.85
application latency: mean 27.333 ms, P50 26.9 ms, P95 32.3 ms, max 33.9 ms
frame-normalized primary inference: mean 9.447 ms, P95 10.9 ms
frame-normalized behavior inference: mean 5.986 ms, P95 6.3 ms
process CPU: mean 131.514 percent, P95 133.66 percent
NPU load: 100 percent average across 181 readable samples

RSS: mean 266.875 MB, min 243.258 MB, P95 291.918 MB, max 299.891 MB
RSS start-to-end change: -11.23 MB
RSS fitted slope: -0.812 MB/hour
memory_stable: true

captured / processed frames in window: 52,750 / 50,019
latest-frame drops: 2,731; capture queue maximum depth: 1
camera read failures / reconnects during window: 0 / 0
event queue maximum / dropped / write failures: 0 / 0 / 0
GC runs / cyclic objects collected: 167 / 93,932
heap trims reporting released pages: 167 / 167
```

The 2,731 frame drops are deliberate latest-frame replacement (5.18 percent),
not queue accumulation or output loss. The final run passes at least 20 FPS,
P95 below 120 ms, no crash/restart, and stable-memory requirements.
