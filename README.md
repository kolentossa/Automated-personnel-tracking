# RK3588 Personnel Tracking

## Project overview

This project counts people entering and leaving on an RK3588 board. A local web page shows live video, entry and exit counts, and recent events. Faces are pixelated before display. The page supports English and Chinese.

## Processing flow

Camera → Person detection → Tracking → Crossing count → Face masking → Web page

YOLO finds people, ByteTrack assigns temporary IDs, and the counter records a passage when a person crosses the selected line.

## Models and code sources

- **Person detection:** pretrained YOLOv5m CrowdHuman weights from [yakhyo/yolov5-crowdhuman-onnx](https://github.com/yakhyo/yolov5-crowdhuman-onnx), converted to INT8 RKNN for the NPU.
- **Face detection:** [RetinaFace Mobile320 ONNX](https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/RetinaFace/RetinaFace_mobile320.onnx), distributed by Rockchip and run on the CPU.
- **Tracking:** ByteTrack from [Supervision 0.25.0](https://github.com/roboflow/supervision/tree/eaeb51a8dbe4bef5ef2c1e7e733907226d863e2f/supervision/tracker/byte_tracker), included in `app/bytetrack/`. Third-party notices are in `LICENSE`.

Our work includes model conversion and deployment, tracking with actual frame times, crossing confirmation and state recovery, face masking, processing threads, and the web interface.

## Installation and running

Use Debian 11 ARM64, Python 3.9, a working RK3588 NPU driver, and a V4L2 camera. Python dependencies are listed in `requirements.txt`. Both models are included:

- `models/crowdhuman448_sigmoid_yolov5m_int8.rknn`
- `models/RetinaFace_mobile320.onnx`

Download the RKNN Toolkit Lite2 2.3.0 wheel for CPython 3.9 and ARM64 into `~/Downloads`. From the project directory on the board:

```bash
python3 -m venv ~/.venvs/person-tracking
source ~/.venvs/person-tracking/bin/activate
python -m pip install -r requirements.txt
python -m pip install ~/Downloads/rknn_toolkit_lite2-2.3.0-cp39-cp39-*aarch64.whl
```

Set `camera.camera_device` in `config.yaml`, then start:

```bash
python -B -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Open `http://<board-ip>:8001` in a browser. Select the counting line and entry direction.

If the service is already running, use `systemctl --user stop person-tracking.service` before starting manually.
