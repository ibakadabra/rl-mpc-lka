"""Unit / closed-loop tests for the linear MPC.

The decisive test (`test_closed_loop_regulation`) is the controls equivalent of
a step-response check: start off the lane, close the loop with the same bicycle
model as the plant, and confirm the error is driven to zero while respecting the
actuator limits.
"""

import numpy as np
import pytest

from src.mpc.linear_mpc import LinearMPC, MPCParams
from src.vehicle_model.bicycle_model import VehicleParams, get_discrete_bicycle

P = VehicleParams(mass=1500.0, yaw_inertia=2500.0, lf=1.2, lr=1.4, Cf=80000.0, Cr=85000.0)
Q_NOM = np.array([100.0, 10.0, 0.1, 0.1])
R_NOM = 1.0


def make_mpc():
    return LinearMPC(P, MPCParams())


def test_solver_feasible_and_timed():
    mpc = make_mpc()
    x0 = np.array([0.5, 0.02, 0.0, 0.0])
    res = mpc.solve(x0, vx=20.0, Q=Q_NOM, R=R_NOM, delta_prev=0.0)
    assert res.feasible
    assert np.isfinite(res.delta)
    assert res.solve_time_ms > 0.0


def test_steering_respects_hard_limits():
    mpc = make_mpc()
    # Big error to provoke aggressive steering; limits must still hold.
    x0 = np.array([1.5, 0.2, 0.0, 0.0])
    res = mpc.solve(x0, vx=25.0, Q=np.array([1000.0, 100.0, 0.1, 0.1]), R=0.1, delta_prev=0.0)
    p = mpc.p
    assert np.all(np.abs(res.u_sequence) <= p.delta_max + 1e-6)
    # rate limit on the first move (vs delta_prev=0)
    assert abs(res.u_sequence[0] - 0.0) <= p.delta_rate_max + 1e-6


def test_steering_rate_limit_with_prev():
    mpc = make_mpc()
    x0 = np.array([1.0, 0.1, 0.0, 0.0])
    delta_prev = 0.3
    res = mpc.solve(x0, vx=20.0, Q=Q_NOM, R=R_NOM, delta_prev=delta_prev)
    assert abs(res.u_sequence[0] - delta_prev) <= mpc.p.delta_rate_max + 1e-6


def test_closed_loop_regulation():
    """Start off-center; the loop should bring ey -> 0 (a stable step response)."""
    mpc = make_mpc()
    vx, Ts = 20.0, 0.02
    A_d, B_d = get_discrete_bicycle(vx, P, Ts)

    x = np.array([0.5, 0.0, 0.0, 0.0])  # 0.5 m off the lane center
    delta_prev = 0.0
    ey_hist = [x[0]]
    for _ in range(200):  # 4 seconds
        res = mpc.solve(x, vx=vx, Q=Q_NOM, R=R_NOM, delta_prev=delta_prev)
        assert res.feasible
        delta = res.delta
        x = A_d @ x + (B_d @ np.array([delta])).ravel()
        delta_prev = delta
        ey_hist.append(x[0])

    ey_hist = np.array(ey_hist)
    assert abs(ey_hist[-1]) < 0.05            # converged near lane center
    assert abs(ey_hist[-1]) < abs(ey_hist[0])  # monotone improvement overall
    assert np.max(np.abs(ey_hist)) <= 0.6      # no large overshoot past start


def test_higher_q1_steers_harder():
    """Raising the ey-weight q1 should make MPC fight error more aggressively.

    Use a relaxed steering-rate limit so the first move is NOT rate-saturated;
    otherwise both weightings clip to delta_rate_max and the Q effect is hidden.
    """
    relaxed = MPCParams(delta_rate_max=0.5)
    x0 = np.array([0.3, 0.0, 0.0, 0.0])
    res_lo = LinearMPC(P, relaxed).solve(x0, vx=20.0, Q=np.array([1.0, 10.0, 0.1, 0.1]), R=1.0)
    res_hi = LinearMPC(P, relaxed).solve(x0, vx=20.0, Q=np.array([1000.0, 10.0, 0.1, 0.1]), R=1.0)
    assert abs(res_hi.delta) > abs(res_lo.delta)


def test_low_speed_no_singularity():
    """vx≈0 must still yield a finite, bounded command (vx_min guard)."""
    mpc = make_mpc()
    res = mpc.solve(np.array([0.3, 0.0, 0.0, 0.0]), vx=0.0, Q=Q_NOM, R=R_NOM)
    assert np.isfinite(res.delta)
    assert abs(res.delta) <= mpc.p.delta_max + 1e-6


def test_from_yaml_loads_constraints(tmp_path):
    yaml_text = (
        "mpc:\n  prediction_horizon: 20\n  control_horizon: 10\n  sampling_time: 0.02\n"
        "  solver: OSQP\n  warm_start: true\n"
        "constraints:\n  delta_max: 0.5\n  delta_rate_max: 0.1\n  ey_max: 1.8\n  ay_max: 3.0\n"
        "fixed_weights:\n  Q_diag: [100.0, 10.0, 0.1, 0.1]\n  R: 1.0\n  R_delta: 10.0\n"
        "  q3_fixed: 0.1\n  q4_fixed: 0.1\n"
    )
    f = tmp_path / "mpc.yaml"
    f.write_text(yaml_text)
    p = MPCParams.from_yaml(f)
    assert p.Np == 20 and p.ey_max == 1.8 and p.R_delta == 10.0
