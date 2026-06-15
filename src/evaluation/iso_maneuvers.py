"""ISO standardized lateral maneuvers as reference paths for the internal sim.

These generators turn an ISO test geometry into the inputs our bicycle-model
rollout already understands: a curvature profile `kappa(s)` (the reference
centerline the controller must track, with e_y measured from it) and, for the
constant-radius test, a longitudinal-speed ramp.

ISO 3888-2  — severe double lane change ("moose test").
    A straight road with a gated lateral displacement: enter, shift one lane
    over, hold, shift back. We model the reference *centerline* as the ISO path
    itself, so tracking it perfectly means e_y == 0. The curvature is the second
    derivative of lateral position w.r.t. arc length (small-slope approximation
    kappa ~= y'').

ISO 4138    — steady-state circular driving (understeer characterisation).
    Constant radius R, speed ramped slowly from v_start to v_end. Curvature is
    constant 1/R; the test reads off the steering needed as lateral
    acceleration a_y = v^2 / R rises, giving the understeer gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- ISO 3888-2 --
# Standard section lengths [m] (entry, transition-1, side lane, transition-2,
# exit) for the severe (obstacle-avoidance) double lane change.
ISO3888_2_SECTIONS = (12.0, 13.5, 11.0, 12.5, 12.0)
# Lateral offset between the entry lane centre and the side-lane centre [m].
ISO3888_2_LATERAL_OFFSET = 3.5


@dataclass
class ManeuverProfile:
    """A reference maneuver sampled over arc length."""
    s: np.ndarray            # arc length [m]
    kappa: np.ndarray        # reference curvature [1/m]
    y_ref: np.ndarray        # lateral displacement of the centerline [m]
    vx: np.ndarray           # longitudinal speed setpoint [m/s]
    name: str


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """C1-smooth 0->1 ramp on t in [0,1] (raised-cosine).

    Using a smooth transition (rather than a step) keeps the second derivative
    finite, so the curvature we differentiate out is bounded and physically
    drivable — the same reason real lane-change paths are clothoid-like.
    """
    t = np.clip(t, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def iso3888_2_double_lane_change(
    vx: float,
    ds: float = 0.5,
    lateral_offset: float = ISO3888_2_LATERAL_OFFSET,
    sections: tuple = ISO3888_2_SECTIONS,
) -> ManeuverProfile:
    """Build the ISO 3888-2 double-lane-change reference at constant speed `vx`.

    Returns a ManeuverProfile whose y_ref goes 0 -> +offset -> 0 across the five
    standard sections, with smooth transitions, and whose kappa is y''(s).
    """
    L_entry, L_tr1, L_side, L_tr2, L_exit = sections
    total = sum(sections)
    s = np.arange(0.0, total, ds)

    y = np.zeros_like(s)
    s0 = L_entry
    s1 = s0 + L_tr1
    s2 = s1 + L_side
    s3 = s2 + L_tr2
    # entry: y=0 (already). transition 1: 0 -> offset
    m = (s >= s0) & (s < s1)
    y[m] = lateral_offset * _smoothstep((s[m] - s0) / L_tr1)
    # side lane: hold offset
    m = (s >= s1) & (s < s2)
    y[m] = lateral_offset
    # transition 2: offset -> 0
    m = (s >= s2) & (s < s3)
    y[m] = lateral_offset * (1.0 - _smoothstep((s[m] - s2) / L_tr2))
    # exit: y=0 (already)

    # curvature ~= y'' (small-slope); differentiate twice over arc length.
    dyds = np.gradient(y, ds)
    d2yds2 = np.gradient(dyds, ds)
    kappa = d2yds2 / np.power(1.0 + dyds**2, 1.5)

    return ManeuverProfile(
        s=s, kappa=kappa, y_ref=y,
        vx=np.full_like(s, float(vx)),
        name="iso3888_2_dlc",
    )


# ------------------------------------------------------------------ ISO 4138 --
def iso4138_constant_radius(
    radius: float = 100.0,
    v_start: float = 5.0,
    v_end: float = 25.0,
    ds: float = 0.5,
    ramp_distance: float | None = None,
) -> ManeuverProfile:
    """Build an ISO 4138 steady-state circular-driving speed ramp.

    Constant curvature 1/radius; speed increases linearly with arc length from
    v_start to v_end so lateral acceleration a_y = v^2/radius sweeps up to the
    grip limit. The understeer gradient is read from steering-vs-a_y afterwards.
    """
    if ramp_distance is None:
        # full circle-ish length; long enough for a slow, quasi-static ramp
        ramp_distance = 2.0 * np.pi * radius
    s = np.arange(0.0, ramp_distance, ds)
    frac = s / max(ramp_distance, 1e-9)
    vx = v_start + (v_end - v_start) * frac
    kappa = np.full_like(s, 1.0 / radius)
    # lateral displacement is not meaningful for a circle reference; report 0.
    return ManeuverProfile(
        s=s, kappa=kappa, y_ref=np.zeros_like(s), vx=vx,
        name="iso4138_const_radius",
    )


def understeer_gradient(steer: np.ndarray, ay: np.ndarray) -> float:
    """ISO 4138 understeer gradient K [rad/(m/s^2)] = slope of steer vs a_y.

    Positive K => understeer (more steering needed as a_y rises), the normal,
    stable passenger-car behaviour. Computed by a least-squares line fit over
    the quasi-static ramp.
    """
    steer = np.asarray(steer, dtype=float)
    ay = np.asarray(ay, dtype=float)
    if steer.size < 2 or ay.size < 2:
        return float("nan")
    A = np.vstack([ay, np.ones_like(ay)]).T
    slope, _ = np.linalg.lstsq(A, steer, rcond=None)[0]
    return float(slope)
