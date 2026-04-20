from pathlib import Path

import numpy as np
import torch

from minidreamer.data.replay_buffer import ReplayBuffer


def make_episode(length: int, reward: float = 0.0):
    obs = np.random.rand(length + 1, 64, 64, 3).astype(np.float32)
    actions = np.arange(length, dtype=np.int64) % 7
    rewards = np.full(length, reward, dtype=np.float32)
    terminated = np.zeros(length, dtype=np.float32)
    truncated = np.zeros(length, dtype=np.float32)
    done = np.zeros(length, dtype=np.float32)
    terminated[-1] = 1.0
    done[-1] = 1.0
    return obs, actions, rewards, terminated, truncated, done


def test_replay_buffer_sampling_and_padding(tmp_path: Path):
    buffer = ReplayBuffer(capacity_episodes=10, sequence_length=8, batch_size=4)
    for episode_id, length in enumerate((3, 5, 9)):
        obs, actions, rewards, terminated, truncated, done = make_episode(length, reward=float(episode_id))
        buffer.add_episode(obs, actions, rewards, terminated, truncated, done, episode_id=episode_id)

    available_split = next(split for split in ("train", "val", "test") if buffer.episode_ids(split))
    batch = buffer.sample_sequences(split=available_split, batch_size=2, rng=np.random.default_rng(0))
    assert batch["obs"].shape == (2, 9, 64, 64, 3)
    assert batch["actions"].shape == (2, 8)
    assert batch["mask"].shape == (2, 8)
    assert np.all(batch["mask"].sum(axis=1) >= 1)

    save_dir = tmp_path / "replay"
    buffer.save(save_dir)
    loaded = ReplayBuffer.load(save_dir)
    assert loaded.summary()["episodes"] == buffer.summary()["episodes"]
    assert loaded.summary()["env_steps"] == buffer.summary()["env_steps"]


def test_replay_buffer_torch_batch_shapes():
    buffer = ReplayBuffer(capacity_episodes=4, sequence_length=4, batch_size=2)
    obs, actions, rewards, terminated, truncated, done = make_episode(5, reward=1.0)
    buffer.add_episode(obs, actions, rewards, terminated, truncated, done)
    available_split = next(split for split in ("train", "val", "test") if buffer.episode_ids(split))
    batch = buffer.sample_sequences(split=available_split, batch_size=2, rng=np.random.default_rng(1))
    tensor_batch = ReplayBuffer.batch_to_torch(batch)
    assert tensor_batch["obs"].shape == (2, 5, 3, 64, 64)
    assert tensor_batch["actions"].dtype == torch.int64

