#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv --system-site-packages .venv
fi

. .venv/bin/activate
python -m pip install -r requirements.txt

exec uvicorn app.main:app --host 0.0.0.0 --port 8000