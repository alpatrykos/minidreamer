#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python3.11 src/minidreamer/data/collect_random.py \
  --config configs/fourrooms_world_model.yaml \
  --output-dir artifacts/bootstrap_replay \
  "$@"

