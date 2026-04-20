#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python3.11 src/minidreamer/baselines/train_ppo.py \
  --config configs/fourrooms_ppo.yaml \
  --output-dir artifacts/ppo \
  "$@"

