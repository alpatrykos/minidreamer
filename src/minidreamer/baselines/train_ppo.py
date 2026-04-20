from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn

from minidreamer.config import ensure_run_dirs, load_config
from minidreamer.envs.make_env import make_env_from_config
from minidreamer.utils.common import seed_everything

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    PPO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class MiniGridCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 3:
            raise ValueError(f"Expected 3D image observations, got {observation_space.shape}.")
        self.channel_first = observation_space.shape[0] in (1, 3)
        channels = observation_space.shape[0] if self.channel_first else observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            if not self.channel_first:
                sample = sample.permute(0, 3, 1, 2)
            flattened_dim = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(flattened_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 4 and observations.shape[1] not in (1, 3):
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations.float()))


def build_env(config: dict, seed: int, rank: int):
    def _make():
        env = make_env_from_config(config, seed=seed + rank)
        return Monitor(env)

    return _make


def train_ppo(config: dict, output_dir: str | Path) -> dict[str, float]:
    if PPO is None:
        raise ImportError(
            "stable-baselines3 is required for PPO training."
        ) from IMPORT_ERROR
    ppo_cfg = config["ppo"]
    seed = config.get("project", {}).get("seed", 0)
    seed_everything(seed)
    run_dirs = ensure_run_dirs(output_dir)

    env_fns = [build_env(config, seed, rank) for rank in range(ppo_cfg.get("num_envs", 4))]
    vec_env = DummyVecEnv(env_fns)
    policy_kwargs = {
        "features_extractor_class": MiniGridCNNExtractor,
        "features_extractor_kwargs": {"features_dim": ppo_cfg.get("features_dim", 256)},
    }
    model = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 256),
        batch_size=ppo_cfg.get("batch_size", 256),
        n_epochs=ppo_cfg.get("n_epochs", 4),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        seed=seed,
        device=ppo_cfg.get("device", "auto"),
        verbose=1,
    )
    model.learn(total_timesteps=ppo_cfg["total_timesteps"])
    model.save(Path(run_dirs["checkpoints"]) / "ppo_latest")
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=config["evaluation"]["episodes"],
        deterministic=True,
    )
    vec_env.close()
    return {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO baseline on MiniGrid pixels.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    summary = train_ppo(config, args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
