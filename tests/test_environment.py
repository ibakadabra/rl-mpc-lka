"""Tests for the CarlaMPCEnv (internal bicycle-simulation backend).

These are the "does the env behave like a proper Gym env" checks: valid spaces,
in-bounds observations, reproducibility under a seed, graceful episode endings,
and — the meaningful one — that a sensible fixed policy can actually keep the
car on the road (otherwise the RL problem would be unsolvable).
"""

import numpy as np
import pytest

from src.rl.environment import CarlaMPCEnv, EnvConfig

GOOD_ACTION = np.array([0.65, 0.0, -0.3])  # ~ q1=300, q2=30, r1=0.5


def make_env(**kw):
    return CarlaMPCEnv(config=EnvConfig(**kw))


def test_spaces():
    env = make_env()
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (7,)


def test_reset_returns_valid_obs():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_seed_reproducibility():
    env = make_env(kappa_max=0.04)
    o1, _ = env.reset(seed=42)
    a = env.action_space.sample()
    s1 = env.step(a)[0]
    o2, _ = env.reset(seed=42)
    s2 = env.step(a)[0]
    np.testing.assert_allclose(o1, o2)
    np.testing.assert_allclose(s1, s2)


def test_random_actions_never_crash_and_stay_in_bounds():
    env = make_env(kappa_max=0.04)
    obs, _ = env.reset(seed=1)
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        assert set(["q1", "q2", "r1", "ey", "mpc_solve_ms"]).issubset(info)
        if term or trunc:
            obs, _ = env.reset()


def test_good_policy_survives_straight_road():
    env = make_env(kappa_max=0.0, max_episode_steps=500)
    obs, _ = env.reset(seed=3)
    total, done = 0.0, False
    for _ in range(60):
        obs, r, term, trunc, info = env.step(GOOD_ACTION)
        total += r
        if term or trunc:
            done = trunc  # should truncate (survive), not terminate (crash)
            break
    assert done is True
    assert abs(info["ey"]) < 0.5


def test_good_policy_survives_curve():
    """A feasible (speed-coupled) curve must be trackable by a sensible policy."""
    env = make_env(kappa_max=0.05, max_episode_steps=800)
    obs, _ = env.reset(seed=2)
    terminated = False
    for _ in range(80):
        obs, r, term, trunc, info = env.step(GOOD_ACTION)
        terminated = term
        if term or trunc:
            break
    assert terminated is False          # never leaves the lane
    assert abs(info["ey"]) < 1.0


def test_termination_on_lane_departure():
    """Force a tiny lane and a bad policy -> must terminate, not run forever."""
    env = make_env(kappa_max=0.05, ey_terminate=0.3)
    env.reset(seed=7)
    ended = False
    for _ in range(200):
        # constant extreme action -> poor tracking
        _, _, term, trunc, _ = env.step(np.array([-1.0, -1.0, 1.0]))
        if term or trunc:
            ended = True
            break
    assert ended
