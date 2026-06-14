"""Baseline #4 — Stanley controller (front-axle geometric controller).

The Stanford DARPA Grand Challenge winner. No prediction, no optimization, just
a closed-form geometric law:

    delta = e_psi + atan2(k * ey, k_soft + vx)

In words: "align the front wheel with the path tangent (e_psi), then add an
extra steer that scales with cross-track error (ey)". The atan2 saturates
gracefully so a large ey at low vx does not blow up.

Why include it
--------------
It is the canonical "no model, no MPC, no learning" lateral controller. If it
holds the lane on simple scenarios, MPC's added complexity must justify itself
on the HARD scenarios — exactly the story the thesis wants. Conversely, if
MPC + RL beats Stanley by a small margin only on easy scenarios, the thesis
contribution is weaker; the contrast on hard scenarios (low-mu, sharp turns)
is where Stanley loses badly.
"""

from __future__ import annotations

import numpy as np


class Stanley:
    """Closed-form geometric lateral controller (Hoffmann et al., 2007)."""

    name = "stanley"

    def __init__(self, k: float = 1.5, k_soft: float = 1.0,
                 delta_max: float = 0.5, delta_rate_max: float = 0.1):
        self.k = k                  # cross-track gain
        self.k_soft = k_soft        # softening (prevents 1/vx blowup at low speed)
        self.delta_max = delta_max
        self.delta_rate_max = delta_rate_max
        self._dprev = 0.0

    def compute(self, x, vx, kappa_preview=0.0, delta_prev=0.0):
        ey, e_psi = float(x[0]), float(x[1])
        # Sign convention: in our bicycle model positive delta INCREASES ey, so
        # to drive ey>0 back to zero we need delta<0. The standard Stanley
        # formula assumes the opposite sign for delta — flip it.
        delta = -e_psi - np.arctan2(self.k * ey, self.k_soft + vx)
        # respect the same actuator limits as MPC for a fair comparison
        delta = float(np.clip(delta, -self.delta_max, self.delta_max))
        # rate limit vs the previous applied command
        dmax = self.delta_rate_max
        delta = float(np.clip(delta, delta_prev - dmax, delta_prev + dmax))
        return delta

    def reset(self):
        self._dprev = 0.0
