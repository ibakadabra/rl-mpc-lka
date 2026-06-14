"""Tests for the reward function."""

import pytest

from src.rl.reward import RewardWeights, compute_reward


def test_perfect_tracking_gives_alive_bonus():
    w = RewardWeights()
    r = compute_reward(0.0, 0.0, 0.0, 0.0, w)
    assert r == pytest.approx(w.alive_bonus)


def test_error_reduces_reward():
    r0 = compute_reward(0.0, 0.0, 0.0, 0.0)
    r1 = compute_reward(0.5, 0.0, 0.0, 0.0)
    r2 = compute_reward(1.0, 0.0, 0.0, 0.0)
    assert r0 > r1 > r2


def test_quadratic_penalty_values():
    w = RewardWeights(w_ey=1.0, w_epsi=0.5, w_delta_rate=0.1, w_ay=0.05, alive_bonus=1.0)
    # ey=2 -> -1*4 ; epsi=2 -> -0.5*4=-2 ; ddelta=2 -> -0.1*4=-0.4 ; ay=2 -> -0.05*4=-0.2
    r = compute_reward(2.0, 2.0, 2.0, 2.0, w)
    assert r == pytest.approx(-(4.0 + 2.0 + 0.4 + 0.2) + 1.0)


def test_components_sum_to_reward():
    w = RewardWeights()
    r, comp = compute_reward(0.3, 0.1, 0.05, 1.2, w, return_components=True)
    assert sum(comp.values()) == pytest.approx(r)


def test_from_yaml(tmp_path):
    f = tmp_path / "rl.yaml"
    f.write_text(
        "reward:\n  w_ey: 2.0\n  w_epsi: 0.5\n  w_delta_rate: 0.1\n"
        "  w_ay: 0.05\n  alive_bonus: 1.0\n"
    )
    w = RewardWeights.from_yaml(f)
    assert w.w_ey == 2.0
