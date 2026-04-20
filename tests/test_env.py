import numpy as np

from minidreamer.envs.make_env import make_env


def test_make_env_returns_normalized_rgb_observation():
    env = make_env(seed=0)
    obs, _ = env.reset()
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.float32
    assert 0.0 <= float(obs.min()) <= float(obs.max()) <= 1.0

    next_obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert next_obs.shape == (64, 64, 3)
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()

