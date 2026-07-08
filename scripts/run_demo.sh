#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  ./scripts/setup.sh
fi

. .venv/bin/activate
python scripts/generate_sample_video.py >/dev/null

export PERSON_TRACKING_VIDEO="${1:-data/sample.mp4}"
export PERSON_TRACKING_MODEL="${PERSON_TRACKING_MODEL:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "API:       http://127.0.0.1:${PORT}/"
echo "Dashboard: http://127.0.0.1:${PORT}/dashboard/"
exec python -m uvicorn backend.main:app --host "$HOST" --port "$PORT"
