"""Unit tests for the linear bicycle model.

We check three things a control engineer would check by hand:
  1. The state-space matrices have the right *structure* (the spec's entries).
  2. The LPV speed-dependence and the 1/vx singularity guard behave.
  3. ZOH discretization agrees with the matrix exponential definition.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from src.vehicle_model.bicycle_model import (
    VehicleParams,
    continuous_bicycle_ss,
    discretize_zoh,
    get_discrete_bicycle,
)

# Nominal plant used across tests (matches config/vehicle_params.yaml).
P = VehicleParams(mass=1500.0, yaw_inertia=2500.0, lf=1.2, lr=1.4, Cf=80000.0, Cr=85000.0)
TS = 0.02


def test_matrix_shapes():
    A, B = continuous_bicycle_ss(20.0, P)
    assert A.shape == (4, 4)
    assert B.shape == (4, 1)


def test_structural_entries():
    """The 'kinematic' rows of A and the input matrix B are speed/param exact."""
    vx = 25.0
    A, B = continuous_bicycle_ss(vx, P)

    # Row 0: ey_dot = vx * e_psi + vy  -> A[0,1]=vx, A[0,2]=1
    assert A[0, 1] == pytest.approx(vx)
    assert A[0, 2] == pytest.approx(1.0)
    # Row 1: e_psi_dot = psi_dot       -> A[1,3]=1
    assert A[1, 3] == pytest.approx(1.0)
    # First two rows have no steering input (kinematics, not forces)
    assert B[0, 0] == 0.0 and B[1, 0] == 0.0
    # Input matrix from front tire force
    assert B[2, 0] == pytest.approx(P.Cf / P.mass)
    assert B[3, 0] == pytest.approx(P.lf * P.Cf / P.yaw_inertia)


def test_tire_terms_numeric():
    """Spot-check a dynamics entry against the closed-form expression."""
    vx = 20.0
    A, _ = continuous_bicycle_ss(vx, P)
    assert A[2, 2] == pytest.approx(-(P.Cf + P.Cr) / (P.mass * vx))
    assert A[2, 3] == pytest.approx((P.lr * P.Cr - P.lf * P.Cf) / (P.mass * vx) - vx)
    assert A[3, 2] == pytest.approx((P.lr * P.Cr - P.lf * P.Cf) / (P.yaw_inertia * vx))
    assert A[3, 3] == pytest.approx(-(P.lf**2 * P.Cf + P.lr**2 * P.Cr) / (P.yaw_inertia * vx))


def test_lpv_speed_dependence():
    """Tire terms scale as 1/vx -> doubling speed halves their magnitude."""
    A_slow, _ = continuous_bicycle_ss(10.0, P)
    A_fast, _ = continuous_bicycle_ss(20.0, P)
    assert abs(A_fast[2, 2]) == pytest.approx(abs(A_slow[2, 2]) / 2.0)


def test_singularity_guard_at_zero_speed():
    """vx=0 must not produce inf/nan thanks to the vx_min clamp."""
    A, B = continuous_bicycle_ss(0.0, P, vx_min=1.0)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))


def test_zoh_matches_matrix_exponential():
    """A_d must equal expm(A_c * Ts); B_d must equal the ZOH integral form."""
    A_c, B_c = continuous_bicycle_ss(20.0, P)
    A_d, B_d = discretize_zoh(A_c, B_c, TS)

    # A_d = exp(A_c Ts)
    np.testing.assert_allclose(A_d, expm(A_c * TS), rtol=1e-9, atol=1e-12)

    # B_d = A_c^{-1} (A_d - I) B_c  when A_c invertible; here A_c is singular
    # (ey is an integrator), so verify via the augmented-matrix definition instead.
    n, m = A_c.shape[0], B_c.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_c
    M[:n, n:] = B_c
    Md = expm(M * TS)
    np.testing.assert_allclose(A_d, Md[:n, :n], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(B_d, Md[:n, n:], rtol=1e-9, atol=1e-12)


def test_discrete_has_integrator_eigenvalue():
    """ey is a pure integrator -> the discrete A_d keeps an eigenvalue at z=1."""
    A_d, _ = get_discrete_bicycle(20.0, P, TS)
    eig = np.linalg.eigvals(A_d)
    assert np.min(np.abs(eig - 1.0)) < 1e-6


def test_from_yaml(tmp_path):
    yaml_text = (
        "vehicle:\n"
        "  mass: 1500.0\n"
        "  yaw_inertia: 2500.0\n"
        "  lf: 1.2\n"
        "  lr: 1.4\n"
        "  Cf: 80000.0\n"
        "  Cr: 85000.0\n"
    )
    f = tmp_path / "veh.yaml"
    f.write_text(yaml_text)
    p = VehicleParams.from_yaml(f)
    assert p.mass == 1500.0 and p.Cf == 80000.0
