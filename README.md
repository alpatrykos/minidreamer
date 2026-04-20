# MiniDreamer

MiniDreamer is a research-grade PlaNet-style world model project for `MiniGrid-FourRooms-v0`. It learns a recurrent latent dynamics model from partial RGB observations, predicts reward and episode termination, and uses discrete CEM planning in latent space.

The repository now contains the full implementation scaffold:

- MiniGrid RGB environment wrappers and bootstrap trajectory collection
- Episode-aware replay buffer with reproducible train/val/test splits
- CNN encoder, Gaussian RSSM, reward/done heads, optional decoder
- Discrete CEM planner with termination-aware return scoring
- PPO baseline entrypoint with a MiniGrid-compatible CNN feature extractor
- Evaluation code, configs, scripts, tests, and project documentation

Training has not been run yet, per the project requirement.

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

Planner evaluation from a checkpoint:

```bash
./scripts/eval_planner.sh /path/to/checkpoint.pt /path/to/replay
```

PPO baseline:

```bash
./scripts/train_ppo.sh
```

## Notes

- The implementation follows the supplied project spec closely, with explicit clarifications captured in [docs/spec_clarifications.md](/Users/patryktargosinski/minidreamer/docs/spec_clarifications.md).
- Result placeholders and reporting structure are in [results.md](/Users/patryktargosinski/minidreamer/results.md).

