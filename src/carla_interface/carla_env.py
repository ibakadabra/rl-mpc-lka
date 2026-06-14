"""CARLA backend for validation rollouts (no training here — Phase 5 'B' path).

Mirrors the interface of CarlaMPCEnv just enough for evaluate_carla.py to drive
any of our controllers in CARLA Town04. Key differences from the internal-sim
env:

  * The plant is no longer a 4-state bicycle ODE we step ourselves; CARLA's
    full vehicle physics integrates the actuator command. We only WRITE the
    `VehicleControl.steer` (and a fixed throttle to hold target speed) and READ
    `world.tick()` to advance the simulator.
  * Lane errors come from waypoint_utils (CARLA ground-truth map API).
  * Synchronous mode + fixed_delta_seconds=Ts keeps the simulation step locked
    to the controller's 20 ms tick (no wall-clock drift).

Speed control
-------------
This thesis is lateral-only; longitudinal control is a separate problem. We
keep speed near a target with a simple PI on the throttle, leaving the steering
entirely to the controller under test — that is, the experimental variable.

Connecting to a remote CARLA server
-----------------------------------
The config file `config/carla_params.yaml` carries host/port. Override via
CarlaConnectConfig(host="192.168.X.Y") if you run the code on a different
machine from the simulator. CARLA is a TCP/RPC API; the network round-trip
adds a few ms but does not change the simulation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

try:
    import carla  # type: ignore
except Exception:  # pragma: no cover
    carla = None

from src.carla_interface.waypoint_utils import (
    LaneFrameSignals,
    compute_lane_signals,
    lane_signals_to_state,
)


@dataclass
class CarlaConnectConfig:
    """Network + simulator settings (load from config/carla_params.yaml)."""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    map: str = "Town04"
    fixed_delta_seconds: float = 0.02
    synchronous_mode: bool = True
    vehicle_blueprint: str = "vehicle.tesla.model3"
    weather: str = "ClearNoon"
    spawn_index: int | None = None       # if None, picked randomly from valid spawns
    target_speed: float = 20.0           # [m/s]   longitudinal setpoint
    episode_seconds: float = 30.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CarlaConnectConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)["carla"]
        return cls(
            host=data.get("host", "localhost"),
            port=data.get("port", 2000),
            timeout=data.get("timeout", 10.0),
            map=data.get("map", "Town04"),
            fixed_delta_seconds=data.get("fixed_delta_seconds", 0.02),
            synchronous_mode=data.get("synchronous_mode", True),
            vehicle_blueprint=data.get("vehicle_blueprint", "vehicle.tesla.model3"),
        )


class CarlaValidationEnv:
    """Step a CARLA vehicle with a black-box steering controller.

    Usage:
        with CarlaValidationEnv(cfg) as env:
            sig, x = env.reset(seed=0)
            for _ in range(N):
                delta = controller.compute(x, sig.vx, sig.kappa_preview, delta_prev)
                sig, x, done, info = env.step(delta)
    """

    def __init__(self, cfg: CarlaConnectConfig, mpc_horizon: int = 20, Ts: float = 0.02):
        if carla is None:
            raise ImportError(
                "carla python package not importable. Install the CARLA PythonAPI "
                "wheel matching your simulator (e.g. carla==0.9.15) on the machine "
                "that runs this code."
            )
        self.cfg = cfg
        self.Np = mpc_horizon
        self.Ts = Ts
        self._client = None
        self._world = None
        self._vehicle = None
        self._collision_sensor = None
        self._collided = False
        self._sim_step = 0
        # simple longitudinal PI controller state
        self._spd_int = 0.0

    # ----- lifecycle --------------------------------------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        self._client = carla.Client(self.cfg.host, self.cfg.port)
        self._client.set_timeout(self.cfg.timeout)
        # load map only if not already current — switching maps is slow
        if self._client.get_world().get_map().name.split("/")[-1] != self.cfg.map:
            self._world = self._client.load_world(self.cfg.map)
        else:
            self._world = self._client.get_world()
        settings = self._world.get_settings()
        settings.synchronous_mode = self.cfg.synchronous_mode
        settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
        self._world.apply_settings(settings)
        # weather preset
        wp = getattr(carla.WeatherParameters, self.cfg.weather, None)
        if wp is not None:
            self._world.set_weather(wp)

    def close(self):
        try:
            if self._collision_sensor is not None:
                self._collision_sensor.stop(); self._collision_sensor.destroy()
            if self._vehicle is not None:
                self._vehicle.destroy()
        finally:
            if self._world is not None:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self._world.apply_settings(settings)
        self._collision_sensor = None
        self._vehicle = None

    # ----- one episode ------------------------------------------------------
    def reset(self, seed: int = 0, spawn_index: int | None = None,
              target_speed: float | None = None,
              cf_mult: float | None = None, cr_mult: float | None = None):
        """Spawn a fresh vehicle at a (seed-deterministic) lane location."""
        rng = random.Random(seed)
        np.random.seed(seed)

        if self._vehicle is not None:
            self._vehicle.destroy(); self._vehicle = None
        if self._collision_sensor is not None:
            self._collision_sensor.stop(); self._collision_sensor.destroy()
            self._collision_sensor = None
        self._collided = False
        self._sim_step = 0
        self._spd_int = 0.0
        self._target_speed = target_speed if target_speed is not None else self.cfg.target_speed

        bp_lib = self._world.get_blueprint_library()
        veh_bp = bp_lib.filter(self.cfg.vehicle_blueprint)[0]
        spawn_points = self._world.get_map().get_spawn_points()
        if spawn_index is None:
            spawn_index = self.cfg.spawn_index
        sp = spawn_points[spawn_index % len(spawn_points)] if spawn_index is not None \
             else rng.choice(spawn_points)
        self._vehicle = self._world.spawn_actor(veh_bp, sp)

        # optional friction tweak (low-mu scenario): CARLA exposes WheelPhysicsControl
        if cf_mult is not None or cr_mult is not None:
            phys = self._vehicle.get_physics_control()
            wheels = list(phys.wheels)
            for i, w in enumerate(wheels):
                base = w.tire_friction
                mult = cf_mult if i < 2 else cr_mult
                if mult is not None:
                    w.tire_friction = float(base * mult)
            phys.wheels = wheels
            self._vehicle.apply_physics_control(phys)

        # collision sensor for safety termination
        col_bp = bp_lib.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._vehicle)
        self._collision_sensor.listen(lambda _e: self._on_collision())

        # tick a few steps so initial transforms settle
        for _ in range(5):
            self._world.tick()
        sig = compute_lane_signals(self._world, self._vehicle, Np=self.Np, Ts=self.Ts)
        x = lane_signals_to_state(sig, vy_estimate=0.0)
        return sig, x

    def _on_collision(self):
        self._collided = True

    def _longitudinal_pi(self, sig: LaneFrameSignals):
        """Hold target speed with a tiny PI on throttle (or brake when over)."""
        err = self._target_speed - sig.vx
        self._spd_int = float(np.clip(self._spd_int + err * self.Ts, -5.0, 5.0))
        u = 0.15 * err + 0.05 * self._spd_int
        throttle = float(np.clip(u, 0.0, 1.0))
        brake = float(np.clip(-u, 0.0, 1.0))
        return throttle, brake

    def step(self, delta: float):
        """Apply steering, tick CARLA once, return new signals + done flag."""
        sig_now = compute_lane_signals(self._world, self._vehicle, Np=self.Np, Ts=self.Ts)
        throttle, brake = self._longitudinal_pi(sig_now)
        # CARLA's steer input is normalized to [-1, 1] mapping ~[-70 deg, +70 deg]
        # of the front wheels. For lane keeping the relevant range is ~[-0.3, 0.3]
        # of physical delta; we send delta / max_steer_rad.
        max_steer_rad = math.radians(70.0)
        steer_cmd = float(np.clip(delta / max_steer_rad, -1.0, 1.0))
        ctrl = carla.VehicleControl(throttle=throttle, steer=steer_cmd,
                                    brake=brake, hand_brake=False, reverse=False)
        self._vehicle.apply_control(ctrl)
        self._world.tick()
        self._sim_step += 1

        sig = compute_lane_signals(self._world, self._vehicle, Np=self.Np, Ts=self.Ts)
        x = lane_signals_to_state(sig, vy_estimate=0.0)
        done = (self._collided
                or abs(sig.ey) > 2.5
                or self._sim_step >= int(self.cfg.episode_seconds / self.Ts))
        info = {
            "collided": self._collided,
            "lane_violation": abs(sig.ey) > 1.8,
            "sim_step": self._sim_step,
        }
        return sig, x, done, info
