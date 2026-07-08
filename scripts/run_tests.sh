#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then
  . .venv/bin/activate
fi
python -m unittest discover -s tests -v
