# Windows + WSL2 Guide: Export YOLOv8n ONNX To RKNN For RK3588

This guide is for converting a YOLOv8n model on a Windows PC through WSL2 Ubuntu 22.04. Do not run RKNN model conversion on the RK3588 ARM64 board.

Target RK3588 project path:

```text
/home/cat/projects/person-tracking
```

Target RKNN model file:

```text
/home/cat/projects/person-tracking/models/yolov8n.rknn
```

## A. Prepare Windows

Confirm the main Windows computer is an Intel or AMD x86_64 PC. RKNN Toolkit 2 conversion should run on x86_64 Linux, not on the RK3588 board.

Install WSL2 Ubuntu 22.04 from an Administrator PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart Windows if prompted, then open Ubuntu 22.04.

Inside Ubuntu, install basic tools:

```bash
sudo apt update
sudo apt install -y git wget curl python3 python3-pip python3-venv build-essential
```

## B. Prepare Python Environment In WSL2 Ubuntu

Use a dedicated virtual environment:

```bash
python3 -m venv ~/venvs/rknn
source ~/venvs/rknn/bin/activate
python -m pip install --upgrade pip
```

Keep this environment only for conversion tooling.

## C. Download Official Rockchip Tools And Examples

Clone the official Rockchip repositories:

```bash
mkdir -p ~/rk3588-rknn
cd ~/rk3588-rknn
git clone https://github.com/airockchip/rknn-toolkit2.git
git clone https://github.com/airockchip/rknn_model_zoo.git
```

Install the `rknn-toolkit2` wheel that matches your Ubuntu Python version. The wheel files are usually under `rknn-toolkit2/packages/`.

Example:

```bash
cd ~/rk3588-rknn/rknn-toolkit2
find packages -name '*.whl' -maxdepth 2
python -m pip install packages/<matching-rknn-toolkit2-wheel>.whl
```

If no wheel matches your Python version, switch to a Python version supported by the official release you downloaded, or follow the release notes from the Rockchip repository. Do not force-install a mismatched wheel.

## D. Prepare YOLOv8n ONNX

Prefer the official model expected by `rknn_model_zoo/examples/yolov8`.

Expected location:

```text
~/rk3588-rknn/rknn_model_zoo/examples/yolov8/model/yolov8n.onnx
```

If the model directory does not exist, create it:

```bash
mkdir -p ~/rk3588-rknn/rknn_model_zoo/examples/yolov8/model
```

If the model zoo README provides a download script for YOLOv8n ONNX, use that script. If you already have a trusted `yolov8n.onnx`, copy it to:

```bash
cp /path/to/yolov8n.onnx ~/rk3588-rknn/rknn_model_zoo/examples/yolov8/model/yolov8n.onnx
```

The project does not require training a new model.

## E. Convert ONNX To RKNN

Enter the YOLOv8 Python example directory:

```bash
cd ~/rk3588-rknn/rknn_model_zoo/examples/yolov8/python
```

Run conversion for RK3588:

```bash
python3 convert.py ../model/yolov8n.onnx rk3588
```

Depending on the exact model zoo version, the output may be named `yolov8.rknn` or similar. Rename the final RK3588 model to:

```bash
mv yolov8.rknn yolov8n.rknn
```

If the script output path is different, locate it with:

```bash
find .. -name '*.rknn' -type f -print
```

The final file name copied to RK3588 must be:

```text
yolov8n.rknn
```

## F. Copy The RKNN Model To RK3588

From WSL2, copy with `scp`:

```bash
scp yolov8n.rknn cat@192.168.1.213:/home/cat/projects/person-tracking/models/yolov8n.rknn
```

You can also use WinSCP from Windows:

```text
Host: 192.168.1.213
User: cat
Remote directory: /home/cat/projects/person-tracking/models/
Remote file name: yolov8n.rknn
```

Do not upload the private SSH key anywhere. Do not commit the `.rknn` file to Git.

## G. Verify On RK3588

SSH into the RK3588 and run:

```bash
cd /home/cat/projects/person-tracking
ls -lh models/yolov8n.rknn
sha256sum models/yolov8n.rknn
source .venv/bin/activate
python scripts/check_rknn_model.py
```

Expected success indicators:

```text
detector name: rknn-yolov8n
npu_enabled: true
inference_ms: <reasonable number>
boxes count: <0 or more>
```

Only after this script succeeds should the RKNN Web service be tested on port 8001.
