from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from minidreamer.data.replay_buffer import Episode, ReplayBuffer
from minidreamer.envs.make_env import make_env_from_config
from minidreamer.models.world_model import WorldModel


def evaluate_random_policy(config: dict, episodes: int | None = None, seed: int | None = None) -> dict[str, float]:
    eval_cfg = config["evaluation"]
    episodes = episodes or eval_cfg["episodes"]
    seed = config.get("project", {}).get("seed", 0) if seed is None else seed
    env = make_env_from_config(config, seed=seed)
    rng = np.random.default_rng(seed)
    returns = []
    lengths = []
    successes = []
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        total_return = 0.0
        terminated = False
        truncated = False
        length = 0
        while not (terminated or truncated):
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            length += 1
        returns.append(total_return)
        lengths.append(length)
        successes.append(float(terminated and total_return > 0.0))
    env.close()
    returns_array = np.asarray(returns, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.float32)
    successes_array = np.asarray(successes, dtype=np.float32)
    return {
        "success_rate": float(successes_array.mean()),
        "mean_return": float(returns_array.mean()),
        "median_return": float(np.median(returns_array)),
        "mean_episode_length": float(lengths_array.mean()),
    }


def _episode_to_batch(episode: Episode, device: torch.device) -> dict[str, torch.Tensor]:
    batch = {
        "obs": torch.from_numpy(episode.obs[None]).permute(0, 1, 4, 2, 3).float().to(device),
        "actions": torch.from_numpy(episode.actions[None]).long().to(device),
        "rewards": torch.from_numpy(episode.rewards[None]).float().to(device),
        "terminated": torch.from_numpy(episode.terminated[None]).float().to(device),
        "truncated": torch.from_numpy(episode.truncated[None]).float().to(device),
        "done": torch.from_numpy(episode.done[None]).float().to(device),
        "mask": torch.ones((1, episode.length), dtype=torch.float32, device=device),
    }
    return batch


def _sequence_state(model: WorldModel, episode: Episode, start_idx: int) -> Any:
    state = model.posterior_step(model.initial_state(1), None, episode.obs[0], sample=False)
    for idx in range(start_idx):
        state = model.posterior_step(state, int(episode.actions[idx]), episode.obs[idx + 1], sample=False)
    return state


def _discounted_return(rewards: np.ndarray, done: np.ndarray, discount: float) -> float:
    total = 0.0
    alive = 1.0
    for step, reward in enumerate(rewards):
        total += (discount**step) * alive * float(reward)
        alive *= 1.0 - float(done[step])
    return total


def evaluate_world_model(
    config: dict,
    model: WorldModel,
    replay: ReplayBuffer,
    split: str = "val",
    max_episodes: int | None = None,
) -> dict[str, float]:
    device = model.device
    model.eval()
    metrics: dict[str, list[float]] = defaultdict(list)
    horizons = [1, 5, 10]
    discount = float(config["planner"]["discount"])
    episodes = replay.episode_ids(split)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    with torch.no_grad():
        for episode_id in episodes:
            episode = replay.episodes[episode_id]
            batch = _episode_to_batch(episode, device)
            outputs = model.observe_sequence(batch["obs"], batch["actions"], sample=False)
            reward_mse = F.mse_loss(outputs.reward_pred, batch["rewards"], reduction="none").mean()
            done_bce = F.binary_cross_entropy_with_logits(outputs.done_logits, batch["done"], reduction="none").mean()
            kl = model.rssm.kl_divergence(
                outputs.post_mean,
                outputs.post_std,
                outputs.prior_mean,
                outputs.prior_std,
            ).mean()
            metrics["reward_mse"].append(float(reward_mse.cpu()))
            metrics["done_bce"].append(float(done_bce.cpu()))
            metrics["kl_loss"].append(float(kl.cpu()))
            if outputs.reconstructions is not None:
                recon_mse = F.mse_loss(outputs.reconstructions, batch["obs"][:, 1:], reduction="none").mean()
                metrics["reconstruction_mse"].append(float(recon_mse.cpu()))

            for horizon in horizons:
                if episode.length < horizon:
                    continue
                reward_errors = []
                done_correct = []
                imagined_returns = []
                real_returns = []
                for start_idx in range(episode.length - horizon + 1):
                    state = _sequence_state(model, episode, start_idx)
                    actions = torch.from_numpy(episode.actions[start_idx : start_idx + horizon]).long().to(device)
                    rollout = model.score_action_sequences(
                        state,
                        actions.unsqueeze(0),
                        discount=discount,
                        use_done_mask=True,
                    )
                    reward_pred = rollout["reward_pred"].squeeze(0).cpu().numpy()
                    done_prob = rollout["done_prob"].squeeze(0).cpu().numpy()
                    done_pred = (done_prob >= 0.5).astype(np.float32)
                    real_rewards = episode.rewards[start_idx : start_idx + horizon]
                    real_done = episode.done[start_idx : start_idx + horizon]
                    reward_errors.append(np.mean((reward_pred - real_rewards) ** 2))
                    done_correct.append(np.mean(done_pred == real_done))
                    imagined_returns.append(float(rollout["scores"].squeeze(0).cpu()))
                    real_returns.append(_discounted_return(real_rewards, real_done, discount))
                metrics[f"open_loop_reward_error_h{horizon}"].append(float(np.mean(reward_errors)))
                metrics[f"open_loop_done_accuracy_h{horizon}"].append(float(np.mean(done_correct)))
                if len(imagined_returns) > 1 and np.std(real_returns) > 0.0 and np.std(imagined_returns) > 0.0:
                    correlation = np.corrcoef(imagined_returns, real_returns)[0, 1]
                    metrics["imagined_return_vs_real_return_correlation"].append(float(correlation))

    return {name: float(np.mean(values)) for name, values in metrics.items() if values}

