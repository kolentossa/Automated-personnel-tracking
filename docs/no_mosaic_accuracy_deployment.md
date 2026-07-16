# No-Mosaic Accuracy Deployment

## Scope And Safety

`feature/rk3588-no-mosaic-accuracy` combines two independent changes:

1. Face detection and all face masking are removed from the active runtime.
2. Phone recall and cigarette false-positive handling are improved with model,
   ROI, conflict, person-context, and temporal changes.

Removing mosaic did not cause the accuracy improvement. The MJPEG stream,
event snapshots, optional clips, and debug images are unredacted and may show
identifiable people. Run this service only on a controlled LAN. Do not expose
port 8001 to the public Internet and do not commit ignored evidence files.

## Runtime Flow

```text
raw camera frame
  -> YOLOv8n person/phone detection
  -> scheduled YOLOv8n/YOLO11n phone ROI refinement
  -> DAMO-YOLO cigarette detection
  -> person tracking and object association
  -> phone and smoking temporal state machines
  -> boxes, IDs, behavior overlays, and counting line
  -> unredacted event writer and MJPEG encoder
```

There is no face detector, face worker, face queue, optical-flow face tracker,
mosaic, blur, pixelation, anonymization, or redaction step in this path.

Phone-call and smoking geometry uses configurable head and mouth regions
estimated from a person bounding box. Those regions are geometric proxies, not
face detections. Person detection, track IDs, counting, phone association,
smoking verification, and event persistence remain enabled.

## Configuration

The deployment profile is `configs/rk3588_no_mosaic_accuracy.yaml`:

```yaml
privacy:
  face_detection_enabled: false
  mosaic_enabled: false
  unredacted_video_enabled: true
  unredacted_evidence_enabled: true

behavior:
  phone_detection_enabled: true
  smoking_detection_enabled: true
```

`config.yaml` carries the same runtime privacy flags. The service does not
check for a RetinaFace model and a missing face model cannot degrade health.

## Model Stack

| Purpose | Runtime file | SHA-256 |
| --- | --- | --- |
| Person and baseline phone | `models/yolov8n.rknn` | `ff3a64e6fe180203128c8d42456b458d208d3a1e2217d63683af00d6194e82ea` |
| Phone context ROI | `models/yolo11n.rknn` | `8853507c88c777f39e574b730abdd2a151be102830c570cee0fd31a6bb8010fc` |
| Cigarette | `models/behavior_damoyolo_cigarette_int8.rknn` | `d04c43a3a695c9985fbd03db1e0a2956763374fd686d949b8cd96cabdc7c5941` |

Model binaries are ignored by Git. Sources, licenses, conversion inputs,
sizes, hashes, tensor contracts, and quantization details are recorded in
`models/behavior_model_manifest.json` and `docs/behavior_model_selection.md`.

## Deploy And Operate

Run as the normal `cat` user from the feature worktree or the checked-out
project root. No system files or services are required.

```bash
cd /home/cat/projects/person-tracking/.venv/codex-worktrees/phone-smoking
./scripts/manage_behavior_service.sh restart
./scripts/manage_behavior_service.sh status
```

The project-local supervisor keeps Uvicorn on `0.0.0.0:8001`, restarts a
failed child, stores its PID under `run/`, and logs under `logs/`. It is used
instead of systemd because this deployment is intentionally project-local and
does not modify `/etc` or system services. After a board reboot, run `start`.

Open:

```text
http://RK3588_IP:8001
```

The Dashboard displays a warning that face masking is disabled.

## Verification

```bash
curl -fsS http://127.0.0.1:8001/api/health
curl -fsS http://127.0.0.1:8001/api/stats
curl -fsS 'http://127.0.0.1:8001/api/events?limit=1'
curl --max-time 3 --output /dev/null http://127.0.0.1:8001/video
```

Required health fields:

```text
camera_status=online
detector=rknn-yolov8n
npu_enabled=true
phone_context_model_enabled=true
phone_context_model_name=rknn-yolo11n
behavior_model=rknn-damoyolo-cigarette-int8
behavior_model_npu_enabled=true
phone_detection_available=true
smoking_detection_available=true
face_detection_enabled=false
face_model_loaded=false
mosaic_enabled=false
privacy_mode=no_mosaic
behavior_input_frame=raw
```

Required startup log lines:

```text
FACE DETECTION DISABLED
FACE MOSAIC DISABLED
VIDEO AND EVENT EVIDENCE ARE UNREDACTED
```

Run regression and performance checks with:

```bash
python -m pytest -q
python -m compileall -q app scripts tests vision
python scripts/benchmark_pipeline.py --frames 300
python scripts/benchmark_behavior_pipeline.py --frames 300
python scripts/monitor_behavior_stability.py --duration-seconds 1800
```

## Known Limits

- Phone and cigarette models still miss small, heavily occluded, blurred, or
  poorly lit objects. Temporal memory reduces brief misses but does not create
  evidence when no detector sees the object.
- Head and mouth regions are person-box estimates; there is no pose or gaze
  model. Phone behavior labels are geometric temporal proxies.
- The locked static evidence set is useful for reproducible regression but is
  not a substitute for a scene-disjoint annotated video benchmark.
- Event screenshots are intentionally unredacted. Access control and retention
  are deployment responsibilities.
