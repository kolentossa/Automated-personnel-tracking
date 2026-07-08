#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ] || [ ! -f .venv/bin/activate ]; then
  python3 -m venv --system-site-packages .venv || python3 -m venv --system-site-packages --without-pip .venv
fi

. .venv/bin/activate
SITE_PACKAGES="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

if ! python -m pip --version >/dev/null 2>&1; then
  python3 -m pip install --upgrade --target "$SITE_PACKAGES" pip
fi

python -m pip install -r requirements.txt

python - <<'PY'
import sys
import cv2
import fastapi
import uvicorn
print(sys.executable)
print(f"opencv={cv2.__version__}")
print(f"fastapi={fastapi.__version__}")
print(f"uvicorn={uvicorn.__version__}")
if "/.venv/" not in sys.executable:
    raise SystemExit("Python is not running from .venv")
PY
