"""Baseline #5 — Pure Pursuit (geometric, look-ahead point).

Picks a "carrot" point on the path L_d meters ahead and computes the steering
that would put the rear axle on a circle through that point:

    delta = atan2(2 * L * sin(alpha), L_d)

with L the wheelbase and alpha the angle from the vehicle heading to the
carrot. We do not have a literal point on a parametric path here (we observe
lateral/heading error in a Frenet frame), so we reconstruct an equivalent
geometric target from (ey, e_psi, kappa_preview):

    carrot.lateral_offset  = ey + L_d * sin(e_psi)
    alpha approx           = atan2(carrot.lateral_offset, L_d)

This is the cleanest form that fits our state without inventing a path
representation. The look-ahead distance L_d scales with vx — at higher speed
you must look further ahead, otherwise you "over-correct" and oscillate.
"""

from __future__ import annotations

import numpy as np


class PurePursuit:
    """Geometric look-ahead lateral controller."""

    name = "pure_pursuit"

    def __init__(self, wheelbase: float = 2.6, k_ld: float = 0.5, Ld_min: float = 3.0,
                 delta_max: float = 0.5, delta_rate_max: float = 0.1):
        self.L = wheelbase
        self.k_ld = k_ld            # look-ahead grows with speed: L_d = max(Ld_min, k_ld*vx)
        self.Ld_min = Ld_min
        self.delta_max = delta_max
        self.delta_rate_max = delta_rate_max

    def compute(self, x, vx, kappa_preview=0.0, delta_prev=0.0):
        ey, e_psi = float(x[0]), float(x[1])
        Ld = max(self.Ld_min, self.k_ld * vx)
        # In the vehicle frame, the carrot point on the path Ld ahead has
        # lateral coordinate y_target = -(ey + Ld*sin(e_psi)) (negative because
        # if vehicle is at +ey relative to path, the path centerline is at -ey
        # in the vehicle frame). Sign flip matches our bicycle-model convention.
        y_target = -(ey + Ld * np.sin(e_psi))
        alpha = np.arctan2(y_target, Ld)
        delta = np.arctan2(2.0 * self.L * np.sin(alpha), Ld)

        delta = float(np.clip(delta, -self.delta_max, self.delta_max))
        dmax = self.delta_rate_max
        delta = float(np.clip(delta, delta_prev - dmax, delta_prev + dmax))
        return delta

    def reset(self):
        pass
