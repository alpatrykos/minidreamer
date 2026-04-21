from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from minidreamer.config import ensure_run_dirs, load_config, merge_dicts, save_config
from minidreamer.data.collect_random import collect_bootstrap_dataset
from minidreamer.data.replay_buffer import ReplayBuffer
from minidreamer.evaluation import evaluate_random_policy, evaluate_world_model
from minidreamer.envs.make_env import make_env_from_config
from minidreamer.models.world_model import WorldModel
from minidreamer.planning.cem import DiscreteCEMPlanner
from minidreamer.planning.evaluate_planner import evaluate_planner
from minidreamer.serialization import save_world_model_checkpoint
from minidreamer.utils.common import get_device, seed_everything, write_json, write_jsonl


def train_world_model_updates(
    model: WorldModel,
    replay: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    num_updates: int,
    device: torch.device,
) -> list[dict[str, float]]:
    if num_updates <= 0:
        return []
    model.train()
    logs: list[dict[str, float]] = []
    progress = trange(num_updates, desc="world-model-updates", leave=False)
    for _ in progress:
        batch = ReplayBuffer.batch_to_torch(replay.sample_sequences(split="train"), device=device)
        losses = model.compute_losses(batch, config)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip_norm", 100.0)))
        optimizer.step()
        log_row = {
            "loss": float(losses["loss"].detach().cpu()),
            "reward_loss": float(losses["reward_loss"].cpu()),
            "done_loss": float(losses["done_loss"].cpu()),
            "kl_loss": float(losses["kl_loss"].cpu()),
            "recon_loss": float(losses["recon_loss"].cpu()),
        }
        logs.append(log_row)
        progress.set_postfix({key: f"{value:.3f}" for key, value in log_row.items()})
    return logs


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_training_state(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    action_dim: int,
    device: torch.device,
) -> tuple[dict[str, Any], WorldModel, torch.optim.Optimizer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_config = merge_dicts(payload["config"], config)
    model = WorldModel.from_config(resolved_config, action_dim=action_dim).to(device)
    model.load_state_dict(payload["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(resolved_config["training"]["lr"]))
    optimizer_state = payload.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        optimizer_to_device(optimizer, device)
    return resolved_config, model, optimizer, payload.get("metadata", {})


def collect_planner_steps(
    env,
    replay: ReplayBuffer,
    model: WorldModel,
    planner: DiscreteCEMPlanner,
    num_steps: int,
    random_action_fraction: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    collected_steps = 0
    episodes = 0
    success_episodes = 0
    model.eval()
    while collected_steps < num_steps:
        obs, _ = env.reset()
        observations = [obs]
        actions: list[int] = []
        rewards: list[float] = []
        terminated_flags: list[float] = []
        truncated_flags: list[float] = []
        done_flags: list[float] = []
        terminated = False
        truncated = False

        with torch.no_grad():
            state = model.posterior_step(model.initial_state(1), None, obs, sample=False)
            while not (terminated or truncated):
                if rng.random() < random_action_fraction:
                    action = int(env.action_space.sample())
                else:
                    action = planner.plan(state).action
                obs, reward, terminated, truncated, _ = env.step(action)
                actions.append(action)
                rewards.append(float(reward))
                terminated_flags.append(float(terminated))
                truncated_flags.append(float(truncated))
                done_flags.append(float(terminated or truncated))
                observations.append(obs)
                collected_steps += 1
                if terminated or truncated:
                    break
                state = model.posterior_step(state, action, obs, sample=False)

        replay.add_episode(
            obs=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated_flags, dtype=np.float32),
            truncated=np.asarray(truncated_flags, dtype=np.float32),
            done=np.asarray(done_flags, dtype=np.float32),
        )
        episodes += 1
        success_episodes += int(bool(terminated and np.sum(rewards) > 0.0))
    return {
        "env_steps": collected_steps,
        "episodes": episodes,
        "success_episodes": success_episodes,
    }


def run_training(
    config: dict[str, Any],
    output_dir: str | Path,
    replay_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    seed = config.get("project", {}).get("seed", 0)
    seed_everything(seed)
    run_dirs = ensure_run_dirs(output_dir)
    device = get_device(config.get("training", {}).get("device"))

    env = make_env_from_config(config, seed=seed)
    action_dim = env.action_space.n
    env.close()

    if replay_dir is not None and Path(replay_dir).exists():
        replay = ReplayBuffer.load(replay_dir)
        collection_summary = {"replay_loaded": replay.summary()}
    else:
        replay, collection_summary = collect_bootstrap_dataset(config, output_dir=run_dirs["replay"], seed=seed)

    resume_metadata: dict[str, Any] = {}
    if resume_checkpoint is not None:
        config, model, optimizer, resume_metadata = load_training_state(
            checkpoint_path=resume_checkpoint,
            config=config,
            action_dim=action_dim,
            device=device,
        )
    else:
        model = WorldModel.from_config(config, action_dim=action_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))

    save_config(config, run_dirs["base"] / "resolved_config.yaml")
    training_logs: list[dict[str, float]] = []
    evaluation_logs: list[dict[str, float]] = []

    train_collect_ratio = float(config["collection"].get("train_collect_ratio", 1.0))
    total_updates_budget = int(config["training"]["train_steps"])
    if resume_checkpoint is not None:
        updates_done = int(resume_metadata.get("updates_done", 0))
        checkpoint_env_steps = int(resume_metadata.get("env_steps", 0))
        if replay.env_steps > checkpoint_env_steps and updates_done < total_updates_budget:
            collect_steps_per_iteration = max(1, int(config["collection"].get("collect_steps_per_iteration", 1)))
            per_iteration_updates = int(
                config["collection"].get(
                    "gradient_updates_per_iteration",
                    round(collect_steps_per_iteration * train_collect_ratio),
                )
            )
            missed_iterations = max(0, round((replay.env_steps - checkpoint_env_steps) / collect_steps_per_iteration))
            catch_up_updates = min(total_updates_budget - updates_done, per_iteration_updates * missed_iterations)
            catch_up_logs = train_world_model_updates(model, replay, optimizer, config, catch_up_updates, device)
            training_logs.extend(catch_up_logs)
            updates_done += len(catch_up_logs)
    else:
        initial_updates = min(total_updates_budget, max(1, int(round(replay.env_steps * train_collect_ratio))))
        training_logs.extend(train_world_model_updates(model, replay, optimizer, config, initial_updates, device))
        updates_done = len(training_logs)

    comparison_budgets = config.get("comparison", {}).get("env_steps", [replay.env_steps])
    target_env_steps = int(max(comparison_budgets))
    rng = np.random.default_rng(seed)
    env = make_env_from_config(config, seed=seed)
    planner = DiscreteCEMPlanner.from_config(model, env.action_space.n, config)
    eval_every_steps = int(config["evaluation"].get("eval_every_env_steps", target_env_steps))
    next_eval_step = replay.env_steps

    while replay.env_steps < target_env_steps and updates_done < total_updates_budget:
        collect_steps = min(
            int(config["collection"]["collect_steps_per_iteration"]),
            target_env_steps - replay.env_steps,
        )
        collection_row = collect_planner_steps(
            env,
            replay,
            model,
            planner,
            num_steps=collect_steps,
            random_action_fraction=float(config["collection"].get("random_action_fraction_after_planner", 0.0)),
            rng=rng,
        )
        updates = int(config["collection"].get("gradient_updates_per_iteration", round(collection_row["env_steps"] * train_collect_ratio)))
        updates = min(updates, total_updates_budget - updates_done)
        training_logs.extend(train_world_model_updates(model, replay, optimizer, config, updates, device))
        updates_done = len(training_logs)
        replay.save(run_dirs["replay"])

        if replay.env_steps >= next_eval_step:
            world_model_metrics = evaluate_world_model(config, model, replay, split="val", max_episodes=10)
            planner_metrics = evaluate_planner(config, model, episodes=min(10, config["evaluation"]["episodes"]), seed=seed)
            random_metrics = evaluate_random_policy(config, episodes=min(10, config["evaluation"]["episodes"]), seed=seed)
            eval_row = {
                "env_steps": replay.env_steps,
                "updates_done": updates_done,
                **{f"world_model/{key}": value for key, value in world_model_metrics.items()},
                **{f"planner/{key}": value for key, value in planner_metrics.items()},
                **{f"random/{key}": value for key, value in random_metrics.items()},
            }
            evaluation_logs.append(eval_row)
            next_eval_step += eval_every_steps
            save_world_model_checkpoint(
                run_dirs["checkpoints"] / f"world_model_env_steps_{replay.env_steps}.pt",
                model,
                config,
                optimizer=optimizer,
                metadata={"env_steps": replay.env_steps, "updates_done": updates_done},
            )

    env.close()
    save_world_model_checkpoint(
        run_dirs["checkpoints"] / "world_model_latest.pt",
        model,
        config,
        optimizer=optimizer,
        metadata={"env_steps": replay.env_steps, "updates_done": updates_done},
    )
    write_json(run_dirs["metrics"] / "collection_summary.json", collection_summary)
    write_jsonl(run_dirs["metrics"] / "train_metrics.jsonl", training_logs)
    write_jsonl(run_dirs["metrics"] / "eval_metrics.jsonl", evaluation_logs)
    summary = {
        "replay": replay.summary(),
        "updates_done": updates_done,
        "device": str(device),
    }
    write_json(run_dirs["metrics"] / "run_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MiniDreamer world model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, default=None, help="Optional existing replay directory.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Optional checkpoint to resume from.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_training(
        config,
        args.output_dir,
        replay_dir=args.replay_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(summary)


if __name__ == "__main__":
    main()
