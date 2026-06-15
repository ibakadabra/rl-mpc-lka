"""ISO performance criteria for the lateral-control evaluation.

This module turns a closed-loop rollout's time series (lateral error, lateral
acceleration, ...) into the quantitative PASS/FAIL criteria defined by the
relevant ISO standards, so the thesis can state "evaluated against ISO 11270 /
ISO 3888-2 / ISO 4138" rather than only ad-hoc RMSE numbers.

Implemented standards
---------------------
ISO 11270:2014  — Lane Keeping Assistance Systems (LKAS): performance limits.
    Quantitative dynamic bounds we can check on any rollout:
      * commanded lateral acceleration   |a_y|   <= 3.0 m/s^2
      * lateral jerk                      |da_y/dt| <= 5.0 m/s^3
      * no lane departure: the vehicle body stays inside the lane markings,
        i.e.  |e_y| + half_vehicle_width <= half_lane_width.
    (ISO 11270 also fixes an operating-speed envelope and a minimum curve
     radius; those are *applicability* conditions, reported separately, not a
     per-step dynamic fail.)

ISO 3888-2  — severe double lane change ("moose test") cone geometry. The
    reference-path generator lives in `iso_maneuvers.py`; here we only score the
    pass/fail (did the vehicle stay within the gated corridor).

ISO 4138    — steady-state circular driving (understeer gradient). Scoring of
    the constant-radius speed ramp also lives in `iso_maneuvers.py`.

Everything here is pure NumPy with no simulator dependency, so it runs the same
way on the internal-sim rollout and the CARLA rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

G = 9.81  # m/s^2


# ISO 11270 nominal limits (SI units).
ISO11270_AY_MAX = 3.0       # m/s^2  max commanded lateral acceleration
ISO11270_JERK_MAX = 5.0     # m/s^3  max lateral jerk
# Vehicle / lane geometry defaults (Tesla Model 3 body width ~1.85 m).
DEFAULT_VEHICLE_HALF_WIDTH = 0.925   # m
DEFAULT_LANE_HALF_WIDTH = 1.8        # m  (matches MPC ey_max)


@dataclass
class ISO11270Result:
    """Per-rollout ISO 11270 scoring."""
    max_ay: float              # m/s^2
    max_jerk: float            # m/s^3
    max_abs_ey: float          # m
    lane_departure_margin: float  # m  (half_lane - half_veh - max|ey|); <0 == departed
    pass_ay: bool
    pass_jerk: bool
    pass_lane: bool
    overall_pass: bool         # all three

    def as_dict(self, prefix: str = "iso11270_") -> dict:
        d = asdict(self)
        return {prefix + k: (int(v) if isinstance(v, bool) else v) for k, v in d.items()}


def lateral_jerk(ay: np.ndarray, Ts: float) -> np.ndarray:
    """Lateral jerk time series = d(a_y)/dt, finite-differenced.

    np.gradient gives a centered difference (second-order accurate interior,
    one-sided at the ends), which is the standard way to estimate jerk from a
    uniformly sampled acceleration trace.
    """
    ay = np.asarray(ay, dtype=float)
    if ay.size < 2:
        return np.zeros_like(ay)
    return np.gradient(ay, Ts)


def evaluate_iso11270(
    ey: np.ndarray,
    ay: np.ndarray,
    Ts: float,
    vehicle_half_width: float = DEFAULT_VEHICLE_HALF_WIDTH,
    lane_half_width: float = DEFAULT_LANE_HALF_WIDTH,
    ay_max: float = ISO11270_AY_MAX,
    jerk_max: float = ISO11270_JERK_MAX,
) -> ISO11270Result:
    """Score one rollout against ISO 11270 dynamic limits.

    ey, ay   lateral-error and lateral-acceleration time series (same length)
    Ts       sample time [s] (for the jerk derivative)
    """
    ey = np.asarray(ey, dtype=float)
    ay = np.asarray(ay, dtype=float)

    max_ay = float(np.max(np.abs(ay))) if ay.size else float("nan")
    jerk = lateral_jerk(ay, Ts)
    max_jerk = float(np.max(np.abs(jerk))) if jerk.size else float("nan")
    max_abs_ey = float(np.max(np.abs(ey))) if ey.size else float("nan")

    # The vehicle's outer edge must stay inside the lane marking.
    departure_limit = lane_half_width - vehicle_half_width
    margin = float(departure_limit - max_abs_ey)

    pass_ay = bool(max_ay <= ay_max)
    pass_jerk = bool(max_jerk <= jerk_max)
    pass_lane = bool(margin >= 0.0)
    return ISO11270Result(
        max_ay=max_ay,
        max_jerk=max_jerk,
        max_abs_ey=max_abs_ey,
        lane_departure_margin=margin,
        pass_ay=pass_ay,
        pass_jerk=pass_jerk,
        pass_lane=pass_lane,
        overall_pass=bool(pass_ay and pass_jerk and pass_lane),
    )
