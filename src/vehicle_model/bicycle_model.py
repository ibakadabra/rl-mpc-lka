"""Linear bicycle model — lateral error dynamics (LPV, speed-dependent).

This is the *prediction model* the MPC uses to forecast how the vehicle's
tracking error evolves over the horizon. In control terms it is the plant
model G(s) that the controller carries inside its head: feed it a steering
input and it tells you where the lateral/heading error will be next.

State vector (continuous):

    x = [ey, e_psi, vy, psi_dot]^T

    ey       lateral error from lane center      [m]
    e_psi    heading (yaw) error                 [rad]
    vy       lateral velocity                    [m/s]   (spec labels this ey_dot)
    psi_dot  yaw rate                            [rad/s] (spec labels this epsi_dot)

NOTE on labels: the project spec writes the state as [ey, e_psi, ey_dot, epsi_dot].
Rows 2-3 of the dynamics are physically the lateral-velocity / yaw-rate states
(classic Rajamani error dynamics). The coupling term in row 0 (ey_dot = vy + vx*e_psi)
is exactly why they are not literally the time-derivatives of rows 0-1. The matrices
below are implemented verbatim from the spec so they cross-check 1:1 against the
MATLAB validation model.

Input:  u = delta  (front steering angle) [rad]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import cont2discrete


@dataclass
class VehicleParams:
    """Physical constants of the bicycle model (the plant's nameplate data)."""

    mass: float        # m   total mass                       [kg]
    yaw_inertia: float # Iz  moment of inertia about z (yaw)  [kg m^2]
    lf: float          # lf  CG -> front axle distance        [m]
    lr: float          # lr  CG -> rear axle distance         [m]
    Cf: float          # Cf  front cornering stiffness         [N/rad]
    Cr: float          # Cr  rear cornering stiffness          [N/rad]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleParams":
        """Load nominal parameters from config/vehicle_params.yaml."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)["vehicle"]
        return cls(
            mass=data["mass"],
            yaw_inertia=data["yaw_inertia"],
            lf=data["lf"],
            lr=data["lr"],
            Cf=data["Cf"],
            Cr=data["Cr"],
        )


def continuous_bicycle_ss(
    vx: float,
    p: VehicleParams,
    vx_min: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the continuous LPV state-space (A_c, B_c) at longitudinal speed vx.

    LPV = Linear Parameter-Varying: the system is linear in the state, but its
    A matrix *depends on vx*. Think of it as a different linear plant for every
    speed — the tire-force terms get divided by vx, so the vehicle "feels"
    different at 30 km/h vs 120 km/h. This is why A,B are rebuilt every MPC call.

    vx_min clamps the speed away from zero: the 1/vx terms blow up at standstill
    (a 0 m/s "lateral error dynamics" is physically meaningless — you can't steer
    a parked car back to the lane). Clamping keeps the model well-posed.
    """
    m, Iz = p.mass, p.yaw_inertia
    lf, lr = p.lf, p.lr
    Cf, Cr = p.Cf, p.Cr

    v = max(float(vx), vx_min)  # guard the 1/vx singularity

    A_c = np.array(
        [
            [0.0, vx,  1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -(Cf + Cr) / (m * v),       (lr * Cr - lf * Cf) / (m * v) - vx],
            [0.0, 0.0, (lr * Cr - lf * Cf) / (Iz * v), -(lf**2 * Cf + lr**2 * Cr) / (Iz * v)],
        ],
        dtype=float,
    )

    B_c = np.array(
        [
            [0.0],
            [0.0],
            [Cf / m],
            [lf * Cf / Iz],
        ],
        dtype=float,
    )

    return A_c, B_c


def discretize_zoh(
    A_c: np.ndarray,
    B_c: np.ndarray,
    Ts: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous -> discrete via Zero-Order Hold (ZOH).

    The MPC runs in discrete time (one QP every Ts = 0.02 s). ZOH assumes the
    steering input is held constant between samples — exactly how a real actuator
    behaves between control updates. Mathematically:

        A_d = exp(A_c * Ts),     B_d = (integral_0^Ts exp(A_c*t) dt) B_c

    scipy does this exact matrix-exponential integration for us. Compare to a
    crude Euler step (A_d ~= I + A_c*Ts): ZOH is exact for piecewise-constant
    inputs, so the prediction model matches reality far better at Ts = 20 ms.
    """
    n = A_c.shape[0]
    C = np.eye(n)
    D = np.zeros((n, B_c.shape[1]))
    A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, C, D), Ts, method="zoh")
    return A_d, B_d


def get_discrete_bicycle(
    vx: float,
    p: VehicleParams,
    Ts: float,
    vx_min: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: continuous LPV model at vx, already discretized for the MPC."""
    A_c, B_c = continuous_bicycle_ss(vx, p, vx_min=vx_min)
    return discretize_zoh(A_c, B_c, Ts)
