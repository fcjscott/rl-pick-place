"""M0: the simulator is installed and behaves as the plan assumes."""

import numpy as np
import pytest

from rlpp import ENV_IDS, OBS_LAYOUT, make_env


@pytest.mark.parametrize("name", list(ENV_IDS))
def test_env_steps(name):
    env = make_env(name)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isfinite(reward)
    assert "is_success" in info
    env.close()


def test_pick_and_place_spec():
    env = make_env("pick")
    obs, _ = env.reset(seed=0)
    assert env.action_space.shape == (4,)
    assert obs["observation"].shape == (25,)
    assert OBS_LAYOUT[-1][2] == 25
    assert env.spec.max_episode_steps == 50
    env.close()


def test_episode_truncates_at_50_steps():
    env = make_env("pick")
    env.reset(seed=0)
    for step in range(1, 51):
        _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
        assert not terminated
    assert truncated and step == 50
    env.close()


def test_seeding_is_reproducible():
    env = make_env("pick")
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.allclose(a["observation"], b["observation"])
    assert np.allclose(a["desired_goal"], b["desired_goal"])
    env.close()


def test_sparse_reward_matches_threshold():
    env = make_env("pick").unwrapped
    goal = np.array([1.3, 0.7, 0.5])
    assert env.compute_reward(goal + [0.04, 0, 0], goal, {}) == 0.0
    assert env.compute_reward(goal + [0.06, 0, 0], goal, {}) == -1.0
