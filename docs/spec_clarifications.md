# Spec Clarifications

This document records the implementation choices the spec left open enough that code needed an explicit default.

## Implemented choices

1. The Python package lives under `src/minidreamer/`, with thin CLI entrypoints at `src/train_world_model.py` and `src/evaluate.py`. That keeps imports stable while still matching the spec's requested top-level scripts.
2. The encoder and decoder use `padding=1` in all `4x4, stride=2` convolutions so `64x64` inputs shrink cleanly to `4x4` and decode symmetrically back to `64x64`.
3. CEM planning uses the prior mean during latent imagination instead of sampling the stochastic latent. That reduces planner variance and makes candidate ranking deterministic given the current model parameters.
4. Replay sampling pads only at the tail of an in-episode chunk and applies a transition mask so padded steps do not contribute to loss terms.
5. Bootstrap and online training both default to `train_collect_ratio = 1.0`, so the initial bootstrap replay produces one gradient update per collected environment step unless an explicit `gradient_updates_per_iteration` override is set.
6. Evaluation computes one-step held-out metrics over full episodes and open-loop rollout metrics for horizons `1`, `5`, and `10` using actual held-out action sequences.

## Remaining optional extensions

These are intentionally not implemented in v1 because the spec marked them as later improvements or ablations:

- KL balancing beyond free nats
- uncertainty penalties or ensembles in the planner
- actor-critic imagination learning
- richer sparse-reward heads beyond scalar reward regression

## Budget semantics

Collection currently finishes complete episodes rather than cutting trajectories mid-episode. That keeps replay episodes semantically clean for recurrent training, but it means a run can land slightly above a requested step target when the last episode crosses the threshold. If exact step-matched checkpoints become mandatory for reporting, the next refinement should snapshot metrics at the first checkpoint at or above each budget and report the realized environment step count alongside the nominal target.

