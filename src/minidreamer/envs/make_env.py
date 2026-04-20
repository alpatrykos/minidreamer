from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    resize: tuple[int, int] = (64, 64)
    normalize_obs: bool = True
    rgb_partial_obs: bool = True
    image_only: bool = True


class ResizeNormalizeObservation(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        resize: tuple[int, int] | None = (64, 64),
        normalize: bool = True,
    ) -> None:
        super().__init__(env)
        self.resize = resize
        self.normalize = normalize
        base_space = env.observation_space
        if not isinstance(base_space, spaces.Box):
            raise TypeError("MiniDreamer expects a Box observation space after wrappers.")
        channels = base_space.shape[-1]
        if resize is None:
            height, width = base_space.shape[:2]
        else:
            height, width = resize
        low, high = (0.0, 1.0) if normalize else (0, 255)
        dtype = np.float32 if normalize else np.uint8
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(height, width, channels),
            dtype=dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation
        if self.resize is not None and tuple(obs.shape[:2]) != self.resize:
            pil_image = Image.fromarray(obs.astype(np.uint8))
            pil_image = pil_image.resize((self.resize[1], self.resize[0]), Image.Resampling.BILINEAR)
            obs = np.asarray(pil_image)
        if self.normalize:
            return obs.astype(np.float32) / 255.0
        return obs.astype(np.uint8)


def make_env(
    env_id: str = "MiniGrid-FourRooms-v0",
    seed: int | None = None,
    resize: tuple[int, int] = (64, 64),
    normalize_obs: bool = True,
    rgb_partial_obs: bool = True,
    image_only: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if rgb_partial_obs:
        env = RGBImgPartialObsWrapper(env)
    if image_only:
        env = ImgObsWrapper(env)
    env = ResizeNormalizeObservation(env, resize=resize, normalize=normalize_obs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_env_from_config(config: dict, seed: int | None = None) -> gym.Env:
    env_cfg = config["env"]
    return make_env(
        env_id=env_cfg["id"],
        seed=seed,
        resize=tuple(env_cfg.get("resize", (64, 64))),
        normalize_obs=env_cfg.get("normalize_obs", True),
        rgb_partial_obs=env_cfg.get("rgb_partial_obs", True),
        image_only=env_cfg.get("image_only", True),
    )


def observation_to_tensor(observation: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    if observation.ndim != 3:
        raise ValueError(f"Expected HWC observation, got shape {observation.shape}.")
    tensor = torch.from_numpy(observation).permute(2, 0, 1).float()
    return tensor.to(device) if device is not None else tensor


def batch_observations_to_tensor(
    observations: np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    if observations.ndim != 5:
        raise ValueError(f"Expected BT HWC observations, got shape {observations.shape}.")
    tensor = torch.from_numpy(observations).permute(0, 1, 4, 2, 3).float()
    return tensor.to(device) if device is not None else tensor


def action_subset(action_space_n: int, names: Iterable[str] | None = None) -> list[int]:
    if names is None:
        return list(range(action_space_n))
    lookup = {
        "left": 0,
        "right": 1,
        "forward": 2,
        "pickup": 3,
        "drop": 4,
        "toggle": 5,
        "done": 6,
    }
    return [lookup[name] for name in names]

