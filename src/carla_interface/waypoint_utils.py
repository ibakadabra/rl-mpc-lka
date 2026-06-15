"""CARLA waypoint / lane-frame utilities (the 'perception layer' substitute).

These functions take the CARLA world + vehicle and return the same 7-D signal
our controllers were trained on — (ey, e_psi, kappa_preview, vx, ...) — by
querying CARLA's ground-truth map API. In a real car these numbers would come
from a lane-detection NN; here we get them analytically from the simulator,
matching the thesis' "perfect perception" assumption.

How CARLA exposes the lane
--------------------------
`world.get_map().get_waypoint(location, project_to_road=True, lane_type=Driving)`
returns the closest centerline waypoint to any 3-D point. Each waypoint also
has `.next(distance)` to walk forward along the lane and `.transform` for its
pose. We walk a small chain forward from the vehicle to build a discrete
preview of the centerline, then numerically differentiate twice to get kappa.

Lane-frame error definitions (match the bicycle model)
------------------------------------------------------
Given vehicle pose (x_v, y_v, yaw_v) and the nearest waypoint pose (x_w, y_w,
yaw_w), with the lane tangent unit vector  t = (cos yaw_w, sin yaw_w) and the
left-normal  n = (-sin yaw_w, cos yaw_w):

    ey      = n . (vehicle_pos - waypoint_pos)     # signed lateral offset
    e_psi   = wrap_to_pi(yaw_v - yaw_w)            # heading vs lane tangent

The sign of ey is chosen so that POSITIVE ey means the vehicle is offset along
the lane's LEFT normal — consistent with our bicycle-model convention where
positive delta increases ey.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import carla  # type: ignore
except Exception:  # pragma: no cover
    carla = None  # imports fail on machines without CARLA — env file guards this


def wrap_to_pi(angle: float) -> float:
    """Normalize an angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class LaneFrameSignals:
    ey: float           # lateral error [m]
    e_psi: float        # heading error [rad]
    kappa_preview: np.ndarray  # curvature over the preview window [1/m]
    vx: float           # longitudinal speed in the vehicle's body frame [m/s]
    psi_dot: float      # yaw rate [rad/s]
    ay: float           # lateral acceleration estimate [m/s^2]
    vy: float = 0.0     # lateral speed in the vehicle's body frame [m/s]


def _preview_centerline(world_map, start_wp, n_points: int, ds: float):
    """Walk forward along the lane, sampling centerline points every ds metres.

    CARLA's waypoint graph branches at intersections; we always take .next(ds)[0]
    (the first successor) so the preview is single-threaded. Returns lists of
    (x, y, yaw) tuples of length <= n_points.
    """
    pts = []
    wp = start_wp
    for _ in range(n_points):
        tr = wp.transform
        pts.append((tr.location.x, tr.location.y, math.radians(tr.rotation.yaw)))
        nxt = wp.next(ds)
        if not nxt:
            break
        wp = nxt[0]
    return pts


def _curvature_from_yaw_series(yaws, ds: float) -> np.ndarray:
    """Curvature kappa = d(yaw)/ds, finite-differenced with yaw unwrapping."""
    if len(yaws) < 2:
        return np.zeros(1)
    yaws = np.unwrap(np.asarray(yaws, dtype=float))
    kappa = np.gradient(yaws, ds)
    return kappa


def compute_lane_signals(
    world,
    vehicle,
    Np: int = 20,
    Ts: float = 0.02,
    ds_min: float = 0.5,
) -> LaneFrameSignals:
    """Read CARLA ground-truth + IMU-like signals into our control signal vector.

    Np, Ts must match the MPC horizon and sampling time. The preview spacing is
    ds = max(ds_min, vx*Ts) so the preview window covers the same time horizon
    the MPC is reasoning about, regardless of speed.
    """
    assert carla is not None, "CARLA python package not importable on this machine"
    world_map = world.get_map()
    tr = vehicle.get_transform()
    vel = vehicle.get_velocity()              # m/s in world frame
    ang = vehicle.get_angular_velocity()      # deg/s
    acc = vehicle.get_acceleration()          # m/s^2 in world frame

    # 1) project vehicle onto the nearest driving lane to find the lane frame
    wp0 = world_map.get_waypoint(tr.location, project_to_road=True,
                                 lane_type=carla.LaneType.Driving)
    if wp0 is None:
        raise RuntimeError("vehicle is off the road map")

    # 2) lane-frame conversion at the nearest waypoint
    yaw_w = math.radians(wp0.transform.rotation.yaw)
    yaw_v = math.radians(tr.rotation.yaw)
    dx = tr.location.x - wp0.transform.location.x
    dy = tr.location.y - wp0.transform.location.y
    # left-normal of the lane tangent (positive ey = left of lane)
    nx = -math.sin(yaw_w); ny = math.cos(yaw_w)
    ey = nx * dx + ny * dy
    e_psi = wrap_to_pi(yaw_v - yaw_w)

    # 3) speed in the vehicle body frame (forward component)
    cosY, sinY = math.cos(yaw_v), math.sin(yaw_v)
    vx_body = vel.x * cosY + vel.y * sinY
    vy_body = -vel.x * sinY + vel.y * cosY
    psi_dot = math.radians(ang.z)
    ay_body = -acc.x * sinY + acc.y * cosY

    # 4) preview curvature: walk Np waypoints forward at spacing ds
    ds = max(ds_min, abs(vx_body) * Ts)
    pts = _preview_centerline(world_map, wp0, Np, ds)
    if len(pts) >= 2:
        yaws = [p[2] for p in pts]
        kappa_preview = _curvature_from_yaw_series(yaws, ds)
    else:
        kappa_preview = np.zeros(Np)
    if kappa_preview.size < Np:
        last = kappa_preview[-1] if kappa_preview.size else 0.0
        kappa_preview = np.concatenate(
            [kappa_preview, np.full(Np - kappa_preview.size, last)])

    return LaneFrameSignals(
        ey=float(ey),
        e_psi=float(e_psi),
        kappa_preview=kappa_preview.astype(float),
        vx=float(vx_body),
        psi_dot=float(psi_dot),
        ay=float(ay_body),
        vy=float(vy_body),
    )


def lane_signals_to_state(sig: LaneFrameSignals, vy_estimate: float | None = None):
    """Pack the lane-frame signals into the controllers' expected x state.

    The bicycle-model state we condition on is [ey, e_psi, vy, psi_dot]. When
    vy_estimate is None (default), the CARLA-measured vy_body is used.
    """
    vy = sig.vy if vy_estimate is None else vy_estimate
    return np.array([sig.ey, sig.e_psi, vy, sig.psi_dot], dtype=float)
