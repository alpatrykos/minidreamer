# MiniDreamer

MiniDreamer is a PlaNet-style world model project for `MiniGrid-FourRooms-v0`. It learns a recurrent latent dynamics model from partial RGB observations, predicts reward and episode termination, and uses discrete CEM planning in latent space.

The repository contains:

- MiniGrid RGB environment wrappers and bootstrap trajectory collection
- Episode-aware replay buffer with reproducible train/val/test splits
- CNN encoder, Gaussian RSSM, reward/done heads, optional decoder
- Discrete CEM planner with termination-aware return scoring
- PPO baseline entrypoint with a MiniGrid-compatible CNN feature extractor
- Evaluation code, configs, scripts, tests, and project documentation

A complete training run has been executed. A summary is recorded in [results.md](/Users/patryktargosinski/minidreamer/results.md), while run artifacts remain gitignored under `artifacts/world_model/`.

## Layout

```text
configs/
docs/
notebooks/
scripts/
src/
tests/
```

Core code lives under `src/minidreamer/`, with CLI entrypoints at `src/train_world_model.py` and `src/evaluate.py`.

## Setup

Use Python 3.11 or 3.12. The project metadata is defined in [pyproject.toml](/Users/patryktargosinski/minidreamer/pyproject.toml).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Main Commands

Bootstrap replay collection:

```bash
./scripts/collect_random.sh
```

World-model pipeline:

```bash
./scripts/train_world_model.sh
```

Resume an interrupted world-model run from a checkpoint:

```bash
python3.11 src/train_world_model.py \
  --config configs/fourrooms_world_model.yaml \
  --output-dir artifacts/world_model \
  --replay-dir artifacts/world_model/replay \
  --resume-checkpoint artifacts/world_model/checkpoints/world_model_env_steps_90021.pt
```

Planner evaluation from a checkpoint:

```bash
./scripts/eval_planner.sh /path/to/checkpoint.pt /path/to/replay
```

PPO baseline:

```bash
./scripts/train_ppo.sh
```

## Notes

- The latest completed run summary is in [results.md](/Users/patryktargosinski/minidreamer/results.md).
- Metrics, replay snapshots, and checkpoints are written to `artifacts/world_model/` and are intentionally gitignored.
