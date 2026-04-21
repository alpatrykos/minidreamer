# Results

## Status

A full world-model training run completed on `2026-04-21` on Apple Silicon using the `mps` backend. Final world-model summaries now come from `artifacts/world_model/metrics/run_summary.json` and `artifacts/world_model/metrics/final_eval_latest.json`, the PPO baseline summary comes from `artifacts/ppo/metrics/run_summary.json`, and generated figures live under `plots/`.

## Problem Statement

Train a PlaNet-style world model for `MiniGrid-FourRooms-v0` from partial RGB observations, then evaluate a latent-space discrete CEM planner against a random baseline.

## Method

- CNN encoder + Gaussian RSSM world model with reward, done, and reconstruction heads.
- Discrete CEM planning in latent space.
- Replay-buffer training with episode-aware train/val/test splits.
- Config: `configs/fourrooms_world_model.yaml`.

## Environment Setup

- Device: `mps`
- Target environment steps: `100000`
- Final realized environment steps: `100004`
- Total gradient updates completed: `10000`
- Final replay summary:
  - Episodes: `1022`
  - Success episodes: `42`
  - Train/val/test episodes: `819 / 116 / 87`

## Baselines

Persisted comparison metrics were recorded at `97124` env steps:

- Random baseline success rate: `0.0`
- Random baseline mean return: `0.0`
- Random baseline mean episode length: `100.0`

Completed PPO baseline after `100000` training env steps:

- PPO success rate: `0.14`
- PPO mean return: `0.1336`
- PPO median return: `0.0`
- PPO return std: `0.3313`
- PPO mean episode length: `86.71`

## Main Metrics

Persisted evaluation metrics at `97124` env steps:

- Planner success rate: `0.10`
- Planner mean return: `0.0568`
- Planner median return: `0.0`
- Planner mean episode length: `94.8`
- Planner action entropy: `0.6569`

Held-out world-model metrics at the same checkpoint:

- Reward MSE: `1.98e-6`
- Done BCE: `0.1808`
- KL loss: `0.8996`
- Reconstruction MSE: `0.0109`

Open-loop rollout quality:

- Done accuracy @1/@5/@10: `0.9880 / 0.9960 / 0.9973`
- Reward error @1/@5/@10: `1.98e-6 / 1.83e-6 / 1.69e-6`

Final explicit evaluation at `world_model_latest.pt`:

- Evaluation budget: `5` planner episodes, `5` random episodes, `5` held-out world-model episodes
- Planner success rate: `0.0`
- Planner mean return: `0.0`
- Planner median return: `0.0`
- Planner mean episode length: `100.0`
- Planner action entropy: `0.9468`
- Final random baseline success rate: `0.0`
- Final random baseline mean return: `0.0`
- Final random baseline mean episode length: `100.0`
- Final reward MSE: `4.95e-6`
- Final done BCE: `0.1513`
- Final KL loss: `0.8833`
- Final reconstruction MSE: `0.0096`
- Final done accuracy @1/@5/@10: `0.9880 / 0.9971 / 0.9978`
- Final reward error @1/@5/@10: `4.98e-6 / 5.09e-6 / 5.12e-6`

## Comparison

At roughly matched data budgets, the PPO baseline produced stronger direct control performance than the world-model planner:

- PPO at `100000` env steps: `0.14` success rate, `0.1336` mean return, `86.71` mean episode length over `100` evaluation episodes.
- Planner at `97124` env steps: `0.10` success rate, `0.0568` mean return, `94.8` mean episode length.
- Final explicit planner eval at `100004` env steps: `0.0` success rate and `0.0` mean return over `5` episodes.

The predictive model stayed numerically strong through the end of training, but that did not translate into a robust planner win on FourRooms at this training budget. The final latest-checkpoint planner evaluation should be treated as high-variance because it uses only `5` episodes.

## Ablations

No ablation runs have been recorded yet.

## Failure Cases / Operational Notes

- The initial long run stopped near `90021` env steps when the local machine hit severe disk pressure.
- Training was resumed successfully from `artifacts/world_model/checkpoints/world_model_env_steps_90021.pt` after adding checkpoint-resume support to the trainer.
- Only one scheduled planner evaluation row is persisted in `eval_metrics.jsonl`. The resumed segment completed without crossing another configured evaluation boundary before the final save, so the end-of-run planner metrics are stored separately in `final_eval_latest.json`.

## Visualizations

- Generated from `artifacts/world_model/metrics/train_metrics.jsonl` and `artifacts/world_model/metrics/eval_metrics.jsonl` using `scripts/generate_results_plots.py`.
- `plots/learning_curves.png`
- `plots/success_rate_vs_env_steps.png`
- `plots/model_error_vs_rollout_horizon.png`

## Artifact Locations

- Metrics: `artifacts/world_model/metrics/`
- Replay: `artifacts/world_model/replay/`
- Checkpoints: `artifacts/world_model/checkpoints/`
- Final checkpoint: `artifacts/world_model/checkpoints/world_model_latest.pt`
- PPO metrics: `artifacts/ppo/metrics/`
- PPO checkpoint: `artifacts/ppo/checkpoints/ppo_latest.zip`
- Generated plots: `plots/`
