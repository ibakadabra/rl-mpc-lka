"""Baseline #1 — Linear MPC with hand-tuned constant Q, R.

The "is RL even necessary?" control. Same MPC code as the RL agent uses, just
frozen at one weighting from config/mpc_params.yaml. If RL-MPC beats this
across the evaluation matrix, the thesis hypothesis is supported.
"""

from __future__ import annotations

import numpy as np

from src.mpc.linear_mpc import LinearMPC, MPCParams
from src.vehicle_model.bicycle_model import VehicleParams


class FixedMPC:
    """MPC with a constant, manually-tuned Q,R (the spec's nominal values)."""

    name = "fixed_mpc"

    def __init__(self, vehicle: VehicleParams, mpc_params: MPCParams | None = None,
                 Q_diag=(100.0, 10.0, 0.1, 0.1), R: float = 1.0):
        self.mpc = LinearMPC(vehicle, mpc_params or MPCParams())
        self.Q = np.asarray(Q_diag, dtype=float)
        self.R = float(R)

    def compute(self, x, vx, kappa_preview, delta_prev=0.0):
        res = self.mpc.solve(x, vx, self.Q, self.R, delta_prev,
                             kappa_preview=kappa_preview)
        return res.delta

    def reset(self):
        # forget warm-start so each rollout is independent
        self.mpc._last_z = None
