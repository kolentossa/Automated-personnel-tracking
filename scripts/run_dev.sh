#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv --system-site-packages .venv
fi

. .venv/bin/activate
python -m pip install -r requirements.txt

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

echo "RK3588 tracking UI: http://127.0.0.1:${PORT}"
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
