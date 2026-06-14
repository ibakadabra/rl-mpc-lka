"""Map the RL agent's action to MPC cost weights (Q, R) on a LOG scale.

The agent outputs a bounded action a in [-1, 1]^3 (the natural output range of a
tanh-squashed SAC policy). We must turn that into:

    Q = diag(q1, q2, q3_fixed, q4_fixed),   R = [[r1]]

with q1 (ey weight), q2 (e_psi weight) and r1 (steering weight) chosen online.

Why a log scale (this is the important part)
--------------------------------------------
q1 spans 1 .. 1000 — three orders of magnitude. With a LINEAR map, half the
action range (a in [0,1]) would land between q1=500 and q1=1000, where the
controller barely notices the difference, while the controls-relevant region
q1=1..10 would get almost no resolution. On a log scale each equal step in the
action multiplies the weight by a constant factor, so the agent gets uniform
"control authority" across every decade. Same reasoning a control engineer uses
when tuning gains on a logarithmic (dB / decade) axis rather than linearly.

    norm = (a + 1) / 2 in [0, 1]
    value = lo * (hi/lo)**norm   ==   10**(log10(lo) + norm*(log10(hi)-log10(lo)))

This reproduces the spec exactly:
    q1 = 10**(norm*3)      for [1, 1000]
    r1 = 10**(norm*2 - 1)  for [0.1, 10]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class WeightRanges:
    """Lower/upper bounds (inclusive) for each RL-tuned weight, + fixed entries."""

    q1_range: tuple[float, float] = (1.0, 1000.0)
    q2_range: tuple[float, float] = (1.0, 1000.0)
    r1_range: tuple[float, float] = (0.1, 10.0)
    q3_fixed: float = 0.1
    q4_fixed: float = 0.1

    @classmethod
    def from_yaml(cls, rl_path: str | Path, mpc_path: str | Path) -> "WeightRanges":
        with open(rl_path, "r") as f:
            wm = yaml.safe_load(f)["weight_mapping"]
        with open(mpc_path, "r") as f:
            fw = yaml.safe_load(f)["fixed_weights"]
        return cls(
            q1_range=tuple(wm["q1_range"]),
            q2_range=tuple(wm["q2_range"]),
            r1_range=tuple(wm["r1_range"]),
            q3_fixed=fw["q3_fixed"],
            q4_fixed=fw["q4_fixed"],
        )


def _log_interp(a: float, lo: float, hi: float) -> float:
    """Map a in [-1, 1] to [lo, hi] geometrically (log-uniform)."""
    norm = (np.clip(a, -1.0, 1.0) + 1.0) / 2.0
    return float(lo * (hi / lo) ** norm)


def action_to_weights(action, ranges: WeightRanges | None = None):
    """RL action a in [-1,1]^3  ->  (Q_diag (4,), R scalar).

    Returns Q as a length-4 diagonal vector [q1, q2, q3_fixed, q4_fixed] and R as
    a float, ready to hand to LinearMPC.solve(...).
    """
    ranges = ranges or WeightRanges()
    a = np.asarray(action, dtype=float).ravel()
    if a.shape[0] != 3:
        raise ValueError(f"action must have 3 elements, got {a.shape[0]}")

    q1 = _log_interp(a[0], *ranges.q1_range)
    q2 = _log_interp(a[1], *ranges.q2_range)
    r1 = _log_interp(a[2], *ranges.r1_range)

    Q_diag = np.array([q1, q2, ranges.q3_fixed, ranges.q4_fixed], dtype=float)
    return Q_diag, r1
