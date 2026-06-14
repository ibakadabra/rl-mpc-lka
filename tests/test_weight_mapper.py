"""Tests for the log-scale action -> (Q, R) mapping."""

import numpy as np
import pytest

from src.rl.weight_mapper import WeightRanges, action_to_weights


def test_extremes_hit_range_bounds():
    Q_lo, r_lo = action_to_weights([-1, -1, -1])
    Q_hi, r_hi = action_to_weights([1, 1, 1])
    assert Q_lo[0] == pytest.approx(1.0)
    assert Q_lo[1] == pytest.approx(1.0)
    assert r_lo == pytest.approx(0.1)
    assert Q_hi[0] == pytest.approx(1000.0)
    assert Q_hi[1] == pytest.approx(1000.0)
    assert r_hi == pytest.approx(10.0)


def test_midpoint_is_geometric_mean():
    """Log-uniform: a=0 maps to sqrt(lo*hi), not the arithmetic mean."""
    Q, r = action_to_weights([0, 0, 0])
    assert Q[0] == pytest.approx(np.sqrt(1.0 * 1000.0), rel=1e-6)   # ~31.62
    assert r == pytest.approx(np.sqrt(0.1 * 10.0), rel=1e-6)        # ~1.0


def test_monotonic_increasing():
    q_prev = -np.inf
    for a in np.linspace(-1, 1, 11):
        Q, _ = action_to_weights([a, a, a])
        assert Q[0] > q_prev
        q_prev = Q[0]


def test_fixed_entries_passthrough():
    Q, _ = action_to_weights([0.3, -0.4, 0.1])
    assert Q[2] == pytest.approx(0.1)  # q3_fixed
    assert Q[3] == pytest.approx(0.1)  # q4_fixed


def test_action_is_clipped():
    Q_over, r_over = action_to_weights([5.0, 5.0, 5.0])
    Q_hi, r_hi = action_to_weights([1.0, 1.0, 1.0])
    np.testing.assert_allclose(Q_over, Q_hi)
    assert r_over == pytest.approx(r_hi)


def test_wrong_action_size_raises():
    with pytest.raises(ValueError):
        action_to_weights([0.0, 0.0])


def test_custom_ranges():
    rng = WeightRanges(q1_range=(10.0, 100.0), q2_range=(1.0, 1.0), r1_range=(1.0, 2.0))
    Q, r = action_to_weights([-1, 0, 1], rng)
    assert Q[0] == pytest.approx(10.0)
    assert Q[1] == pytest.approx(1.0)  # degenerate range stays put
    assert r == pytest.approx(2.0)
