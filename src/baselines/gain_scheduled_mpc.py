"""Baseline #2 — Gain-scheduled MPC: Q, R chosen from a vx-indexed table.

Classic control engineering answer to "the plant feels different at different
speeds": precompute a lookup table of weights for a handful of operating speeds
and interpolate between them. Each row is what you would hand-tune as the best
Q,R for that one speed.

Why this is the right baseline for the RL claim
-----------------------------------------------
RL-tuned MPC's main argument over fixed MPC is "context-dependent gains". A
1-D table (vx only) is the simplest such scheme. If RL-MPC beats this baseline,
the claim must be that the schedule needs MORE than just vx — i.e. RL is
implicitly indexing on curvature, lateral acceleration, plant state, etc. That
is exactly the multi-dim scheduling argument the thesis wants to make.

Linear interpolation between adjacent rows lets the schedule remain smooth (no
sudden gain jumps at table edges, which would otherwise jolt the actuator).
"""

from __future__ import annotations

import numpy as np

from src.mpc.linear_mpc import LinearMPC, MPCParams
from src.vehicle_model.bicycle_model import VehicleParams

# Hand-tuned table: low-speed needs softer R (more authority); high-speed
# tolerates less aggressive q1 (avoid lateral-accel spikes).
#   vx [m/s]    q1     q2     r1
_DEFAULT_TABLE = np.array([
    [ 5.0,    50.0,  20.0,  0.3],
    [10.0,   150.0,  30.0,  0.5],
    [15.0,   250.0,  30.0,  0.7],
    [20.0,   300.0,  30.0,  1.0],
    [25.0,   200.0,  20.0,  1.5],
    [30.0,   120.0,  15.0,  2.0],
])


class GainScheduledMPC:
    """MPC whose Q,R are linearly interpolated from a vx-indexed lookup."""

    name = "gain_scheduled_mpc"

    def __init__(self, vehicle: VehicleParams, mpc_params: MPCParams | None = None,
                 table: np.ndarray | None = None):
        self.mpc = LinearMPC(vehicle, mpc_params or MPCParams())
        self.table = _DEFAULT_TABLE if table is None else np.asarray(table)
        self._vx_grid = self.table[:, 0]
        self._q3_fixed = self.mpc.p.q3_fixed
        self._q4_fixed = self.mpc.p.q4_fixed

    def _weights_at(self, vx: float):
        """Linear interp of (q1, q2, r1) at the requested vx (clamped to grid)."""
        q1 = float(np.interp(vx, self._vx_grid, self.table[:, 1]))
        q2 = float(np.interp(vx, self._vx_grid, self.table[:, 2]))
        r1 = float(np.interp(vx, self._vx_grid, self.table[:, 3]))
        Q = np.array([q1, q2, self._q3_fixed, self._q4_fixed])
        return Q, r1

    def compute(self, x, vx, kappa_preview, delta_prev=0.0):
        Q, R = self._weights_at(vx)
        res = self.mpc.solve(x, vx, Q, R, delta_prev, kappa_preview=kappa_preview)
        return res.delta

    def reset(self):
        self.mpc._last_z = None
