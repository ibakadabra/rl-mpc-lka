"""Gymnasium environment: RL agent tunes MPC weights; MPC drives a bicycle plant.

This is the Phase-1/2 environment that runs WITHOUT CARLA. It wires together the
three pieces we already built:

    RL action a in [-1,1]^3  --(weight_mapper)-->  Q, R
    Q, R + error state x     --(LinearMPC)----->   steering delta   (x10, at 50 Hz)
    delta                    --(bicycle plant)-->  next error state
    next state               --(reward)--------->  scalar reward

Hierarchical timing (the key structural idea)
---------------------------------------------
One env.step() = ONE RL decision = TEN MPC/plant steps = 0.5 s of driving.
The agent picks Q,R once, the MPC then runs ten 20 ms ticks with those weights,
and we sum the reward over the ten ticks. This is the 2 Hz / 50 Hz split.

Plant vs model (sim-to-real in miniature)
-----------------------------------------
The MPC's internal model uses the NOMINAL vehicle parameters. The simulated
"real" plant can use RANDOMIZED parameters (Cf, Cr, mass, actuator delay). The
gap between them is exactly the model mismatch a robust controller must reject,
and is what domain randomization trains the agent against.

Road curvature as a disturbance
-------------------------------
A curving lane rotates the reference heading at rate vx*kappa, so the heading
error obeys  e_psi_dot = psi_dot - vx*kappa. We inject this as an additive
disturbance on the plant. The MPC has NO curvature feed-forward (per spec), so a
constant curve leaves a small steady-state error that better Q-tuning shrinks —
which is precisely the performance the RL agent is meant to improve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError("gymnasium is required for the RL environment") from e

from src.mpc.linear_mpc import LinearMPC, MPCParams
from src.rl.reward import RewardWeights, compute_reward
from src.rl.weight_mapper import WeightRanges, action_to_weights
from src.vehicle_model.bicycle_model import (
    VehicleParams,
    continuous_bicycle_ss,
    discretize_zoh,
)

# Observation bounds [vx, kappa, ey, e_psi, psi_dot, ay, delta_prev] (from spec).
_OBS_LOW = np.array([0.0, -0.2, -3.0, -1.0, -1.0, -10.0, -0.5], dtype=np.float32)
_OBS_HIGH = np.array([50.0, 0.2, 3.0, 1.0, 1.0, 10.0, 0.5], dtype=np.float32)


@dataclass
class EnvConfig:
    """Episode / timing / scenario settings for the internal-sim environment."""

    Ts: float = 0.02              # MPC & plant sampling time [s]
    mpc_steps_per_rl: int = 10    # inner ticks per env.step (=> 0.5 s, 2 Hz RL)
    max_episode_steps: int = 2000  # truncation horizon, in SIM steps (=40 s)
    ey_terminate: float = 2.5     # |ey| past this => lane departure (terminate)
    vx_range: tuple[float, float] = (8.0, 30.0)  # episode speed sample [m/s]
    kappa_max: float = 0.05       # peak |curvature| cap of the generated road [1/m]
    a_lat_max: float = 3.0        # couples speed<->curvature: |kappa| <= a_lat_max/vx^2
                                  # (a 30 m/s + kappa=0.05 corner would be ~4.6 g — infeasible;
                                  #  this keeps generated roads physically trackable)
    action_rate_limit: float = 2.0  # max |Δaction| per RL step (2.0 = effectively off)
    randomize: bool = False       # domain randomization of the plant
    # randomization ranges (multipliers / ms), used only if randomize=True
    cf_range: tuple[float, float] = (0.7, 1.3)
    cr_range: tuple[float, float] = (0.7, 1.3)
    mass_range: tuple[float, float] = (0.9, 1.1)
    delay_ms_range: tuple[float, float] = (0.0, 50.0)

    # --- within-episode tire/grip transition (the "dry->wet mid-corner" case) --
    # With probability mu_step_prob an episode applies an extra Cf,Cr multiplier
    # partway through (a sudden grip drop). A vx-only schedule cannot anticipate
    # this; an online tuner can react to the resulting tracking/yaw error. This
    # is the scenario class that structurally separates RL-MPC from fixed/
    # gain-scheduled MPC. Defaults (prob=0) preserve the original behaviour.
    mu_step_prob: float = 0.0
    mu_step_frac_range: tuple[float, float] = (0.3, 0.7)   # WHEN (episode fraction)
    mu_step_mult_range: tuple[float, float] = (0.5, 0.8)   # post-step Cf,Cr multiplier


class CarlaMPCEnv(gym.Env):
    """RL-tunes-MPC environment (internal bicycle simulation backend)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        vehicle: VehicleParams | None = None,
        mpc_params: MPCParams | None = None,
        weight_ranges: WeightRanges | None = None,
        reward_weights: RewardWeights | None = None,
        config: EnvConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.veh_nominal = vehicle or VehicleParams(1500.0, 2500.0, 1.2, 1.4, 80000.0, 85000.0)
        self.mpc = LinearMPC(self.veh_nominal, mpc_params or MPCParams())
        self.wranges = weight_ranges or WeightRanges()
        self.rweights = reward_weights or RewardWeights()
        self.cfg = config or EnvConfig()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=_OBS_LOW, high=_OBS_HIGH, dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        # nominal (fallback) weights when an MPC solve is infeasible
        self._Q_nom = np.array([100.0, 10.0, self.mpc.p.q3_fixed, self.mpc.p.q4_fixed])
        self._R_nom = 1.0
        self._reset_internal_state()

    # ----- scenario generation ---------------------------------------------
    def _make_curvature_profile(self, n_points: int) -> np.ndarray:
        """Smooth random road curvature kappa(s) bounded by cfg.kappa_max.

        Sum of a few sinusoids with random wavelengths -> a wavy but smooth road.
        kappa_max≈0 yields an essentially straight road (curriculum stage 1).
        """
        # couple curvature to speed: a curve is only kept if vx^2*kappa is a
        # comfortable/feasible lateral acceleration (see EnvConfig.a_lat_max).
        km = min(self.cfg.kappa_max, self.cfg.a_lat_max / max(self._vx, 1.0) ** 2)
        if km <= 1e-6:
            return np.zeros(n_points)
        s = np.arange(n_points) * self.cfg.Ts  # proxy arc index (vx folded in later)
        prof = np.zeros(n_points)
        for _ in range(3):
            wavelength = self._rng.uniform(4.0, 20.0)  # seconds-scale undulation
            phase = self._rng.uniform(0, 2 * np.pi)
            amp = self._rng.uniform(0.3, 1.0)
            prof += amp * np.sin(2 * np.pi * s / wavelength + phase)
        prof = prof / np.max(np.abs(prof)) * km
        return prof

    def _sample_plant(self):
        """Build the simulated plant's vehicle params (optionally randomized)."""
        n = self.veh_nominal
        if not self.cfg.randomize:
            self._delay_steps = 0
            return n
        r = self._rng.uniform
        veh = VehicleParams(
            mass=n.mass * r(*self.cfg.mass_range),
            yaw_inertia=n.yaw_inertia,
            lf=n.lf, lr=n.lr,
            Cf=n.Cf * r(*self.cfg.cf_range),
            Cr=n.Cr * r(*self.cfg.cr_range),
        )
        delay_ms = r(*self.cfg.delay_ms_range)
        self._delay_steps = int(round(delay_ms / 1000.0 / self.cfg.Ts))
        return veh

    def _plant_matrices(self, veh: VehicleParams, vx: float):
        """Discrete plant with curvature disturbance: x+ = A_d x + B_d δ + E_d κ."""
        A_c, B_c = continuous_bicycle_ss(vx, veh, vx_min=self.mpc.p.vx_min)
        E_c = np.array([[0.0], [-vx], [0.0], [0.0]])      # disturbance on e_psi_dot
        B_aug = np.hstack([B_c, E_c])
        A_d, Bd_aug = discretize_zoh(A_c, B_aug, self.cfg.Ts)
        return A_d, Bd_aug[:, :1], Bd_aug[:, 1:2]

    # ----- gym API ----------------------------------------------------------
    def _reset_internal_state(self):
        self._x = np.zeros(4)
        self._delta_prev = 0.0
        self._sim_step = 0
        self._prev_action = np.zeros(3)
        self._vx = 20.0
        self._kappa_profile = np.zeros(1)
        self._delay_steps = 0
        self._delay_buf: list[float] = []
        self._mu_step_at = None
        self._veh_plant_after = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._vx = float(self._rng.uniform(*self.cfg.vx_range))
        self.veh_plant = self._sample_plant()
        self._kappa_profile = self._make_curvature_profile(self.cfg.max_episode_steps + 50)
        self._delay_buf = [0.0] * self._delay_steps

        # schedule an optional within-episode grip drop (dry->wet mid-corner)
        self._mu_step_at = None
        self._veh_plant_after = None
        if self.cfg.randomize and self._rng.uniform() < self.cfg.mu_step_prob:
            frac = self._rng.uniform(*self.cfg.mu_step_frac_range)
            self._mu_step_at = int(frac * self.cfg.max_episode_steps)
            m = self._rng.uniform(*self.cfg.mu_step_mult_range)
            self._veh_plant_after = VehicleParams(
                mass=self.veh_plant.mass, yaw_inertia=self.veh_plant.yaw_inertia,
                lf=self.veh_plant.lf, lr=self.veh_plant.lr,
                Cf=self.veh_plant.Cf * m, Cr=self.veh_plant.Cr * m,
            )

        # small random initial error so the agent sees varied starts
        self._x = np.array([
            self._rng.uniform(-0.5, 0.5),   # ey
            self._rng.uniform(-0.05, 0.05),  # e_psi
            0.0, 0.0,
        ])
        self._delta_prev = 0.0
        self._sim_step = 0
        self._prev_action = np.zeros(3)
        return self._get_obs(kappa=self._kappa_profile[0]), {}

    def _get_obs(self, kappa: float) -> np.ndarray:
        ey, e_psi, _vy, psi_dot = self._x
        ay = self._vx * psi_dot
        obs = np.array([self._vx, kappa, ey, e_psi, psi_dot, ay, self._delta_prev],
                       dtype=np.float32)
        return np.clip(obs, _OBS_LOW, _OBS_HIGH)

    def step(self, action):
        cfg = self.cfg
        action = np.asarray(action, dtype=float).ravel()

        # Q,R rate limiting: bound how fast the agent may swing the knobs.
        delta_a = np.clip(action - self._prev_action,
                          -cfg.action_rate_limit, cfg.action_rate_limit)
        action = np.clip(self._prev_action + delta_a, -1.0, 1.0)
        self._prev_action = action
        Q, R = action_to_weights(action, self.wranges)

        # apply a scheduled within-episode grip drop once we pass its trigger step
        if self._mu_step_at is not None and self._sim_step >= self._mu_step_at:
            self.veh_plant = self._veh_plant_after
            self._mu_step_at = None  # one-shot

        A_d, B_d, E_d = self._plant_matrices(self.veh_plant, self._vx)

        total_reward = 0.0
        infeasible = 0
        solve_ms = []
        terminated = False
        Np = self.mpc.p.Np
        for _ in range(cfg.mpc_steps_per_rl):
            i0 = self._sim_step
            kappa = float(self._kappa_profile[min(i0, len(self._kappa_profile) - 1)])
            # curvature preview over the horizon (perfect-perception lookahead)
            kpre = self._kappa_profile[i0:i0 + Np]
            if kpre.size < Np:
                kpre = np.concatenate([kpre, np.full(Np - kpre.size, kpre[-1] if kpre.size else 0.0)])

            res = self.mpc.solve(self._x, self._vx, Q, R, self._delta_prev, kappa_preview=kpre)
            if not res.feasible:
                infeasible += 1
                res = self.mpc.solve(self._x, self._vx, self._Q_nom, self._R_nom,
                                     self._delta_prev, kappa_preview=kpre)  # nominal fallback
            solve_ms.append(res.solve_time_ms)

            delta_cmd = res.delta
            # optional actuator delay: command takes effect after _delay_steps
            if self._delay_steps > 0:
                self._delay_buf.append(delta_cmd)
                delta_applied = self._delay_buf.pop(0)
            else:
                delta_applied = delta_cmd

            ddelta = delta_applied - self._delta_prev
            # advance the plant one tick
            self._x = (A_d @ self._x
                       + (B_d @ np.array([delta_applied])).ravel()
                       + (E_d @ np.array([kappa])).ravel())
            self._delta_prev = delta_applied
            self._sim_step += 1

            ey, e_psi, _vy, psi_dot = self._x
            ay = self._vx * psi_dot
            total_reward += compute_reward(ey, e_psi, ddelta, ay, self.rweights)

            if abs(ey) > cfg.ey_terminate:
                terminated = True
                break

        truncated = self._sim_step >= cfg.max_episode_steps
        next_kappa = float(self._kappa_profile[min(self._sim_step, len(self._kappa_profile) - 1)])
        obs = self._get_obs(kappa=next_kappa)
        info = {
            "q1": Q[0], "q2": Q[1], "r1": R,
            "ey": float(self._x[0]), "e_psi": float(self._x[1]),
            "mpc_solve_ms": float(np.mean(solve_ms)) if solve_ms else 0.0,
            "infeasible": infeasible,
            "sim_step": self._sim_step,
        }
        return obs, float(total_reward), terminated, truncated, info
