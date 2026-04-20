from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from minidreamer.envs.make_env import make_env_from_config
from minidreamer.planning.cem import DiscreteCEMPlanner


@dataclass
class PlannerEpisode:
    success: bool
    total_return: float
    length: int
    terminated: bool
    truncated: bool
    planner_entropy: float


def run_planner_episode(
    env,
    world_model,
    planner: DiscreteCEMPlanner,
    rng: np.random.Generator,
    seed: int | None = None,
    random_action_fraction: float = 0.0,
) -> PlannerEpisode:
    obs, _ = env.reset(seed=seed)
    world_model.eval()
    with torch.no_grad():
        state = world_model.posterior_step(world_model.initial_state(1), None, obs, sample=False)
        total_return = 0.0
        length = 0
        terminated = False
        truncated = False
        entropies: list[float] = []

        while not (terminated or truncated):
            if rng.random() < random_action_fraction:
                action = int(env.action_space.sample())
            else:
                plan = planner.plan(state)
                action = plan.action
                entropies.append(plan.entropy)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            length += 1
            if not (terminated or truncated):
                state = world_model.posterior_step(state, action, obs, sample=False)

    return PlannerEpisode(
        success=bool(terminated and total_return > 0.0),
        total_return=total_return,
        length=length,
        terminated=bool(terminated),
        truncated=bool(truncated),
        planner_entropy=float(np.mean(entropies)) if entropies else float("nan"),
    )


def evaluate_planner(
    config: dict,
    world_model,
    episodes: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    eval_cfg = config["evaluation"]
    collection_cfg = config["collection"]
    episodes = episodes or eval_cfg["episodes"]
    seed = config.get("project", {}).get("seed", 0) if seed is None else seed
    env = make_env_from_config(config, seed=seed)
    planner = DiscreteCEMPlanner.from_config(world_model, env.action_space.n, config)
    rng = np.random.default_rng(seed)
    results = [
        run_planner_episode(
            env,
            world_model,
            planner,
            rng,
            seed=seed + episode_idx,
            random_action_fraction=collection_cfg.get("random_action_fraction_after_planner", 0.0),
        )
        for episode_idx in range(episodes)
    ]
    env.close()

    returns = np.asarray([result.total_return for result in results], dtype=np.float32)
    lengths = np.asarray([result.length for result in results], dtype=np.float32)
    successes = np.asarray([result.success for result in results], dtype=np.float32)
    entropies = np.asarray([result.planner_entropy for result in results], dtype=np.float32)
    return {
        "success_rate": float(successes.mean()),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "mean_episode_length": float(lengths.mean()),
        "planner_action_entropy": float(np.nanmean(entropies)),
    }
