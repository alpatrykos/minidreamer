#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 CHECKPOINT REPLAY_DIR [extra evaluate args]" >&2
  exit 1
fi

checkpoint="$1"
replay_dir="$2"
shift 2

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python3.11 src/evaluate.py \
  planner \
  --config configs/fourrooms_world_model.yaml \
  --checkpoint "${checkpoint}" \
  "$@"

python3.11 src/evaluate.py \
  world-model \
  --config configs/fourrooms_world_model.yaml \
  --checkpoint "${checkpoint}" \
  --replay-dir "${replay_dir}" \
  "$@"

