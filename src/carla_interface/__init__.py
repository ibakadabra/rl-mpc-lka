"""CARLA interface package: validation backend, waypoint utilities.

Only the lightweight imports are available without the CARLA Python package;
the functions inside raise at runtime if `import carla` fails. This lets the
rest of the project run on machines without CARLA (e.g. the training PC),
while the CARLA pass runs on the host that owns the simulator.
"""

from .waypoint_utils import (
    LaneFrameSignals,
    compute_lane_signals,
    lane_signals_to_state,
    wrap_to_pi,
)

__all__ = [
    "LaneFrameSignals",
    "compute_lane_signals",
    "lane_signals_to_state",
    "wrap_to_pi",
]
