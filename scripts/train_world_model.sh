#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python3.11 src/train_world_model.py \
  --config configs/fourrooms_world_model.yaml \
  --output-dir artifacts/world_model \
  "$@"

