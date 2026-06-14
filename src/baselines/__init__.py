"""Baseline lateral controllers, all exposing the same `compute(...)` interface.

Common interface
----------------
    controller.compute(
        x: np.ndarray,            # error state [ey, e_psi, vy, psi_dot]
        vx: float,                # longitudinal speed
        kappa_preview: np.ndarray | float,   # road curvature (preview length Np)
        delta_prev: float,        # last applied steering
    ) -> float                    # commanded delta [rad]

The trained RL-MPC agent is wrapped with the same signature in `evaluate.py` so
the comparison loop treats every controller identically — a single rollout
function works for all 6.
"""

from .fixed_mpc import FixedMPC
from .gain_scheduled_mpc import GainScheduledMPC
from .pure_pursuit import PurePursuit
from .pure_rl import PureRL
from .stanley import Stanley

__all__ = ["FixedMPC", "GainScheduledMPC", "PurePursuit", "PureRL", "Stanley"]
