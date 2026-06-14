"""Baseline #3 — Pure RL: SAC outputs the steering delta directly (no MPC).

This is the strongest argument for the thesis: by training a separate SAC agent
whose action IS the steering, we show what happens when you take the safety
floor away. The expected outcome is exactly what the thesis claims:

  * sample-inefficient (learns much more slowly)
  * unstable in the long tail of randomization
  * potentially violates lane / steering limits during exploration

This file provides:
  1. A Gym environment whose action space is delta (rad), reward unchanged
  2. A trainer (SAC) on top of that env
  3. A controller wrapper that loads the trained policy and returns delta

The architecture intentionally mirrors the RL-MPC env so the comparison is
apples-to-apples on every other axis except "is there an MPC in the loop".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError("gymnasium is required") from e

from src.rl.environment import EnvConfig, _OBS_HIGH, _OBS_LOW
from src.rl.reward import RewardWeights, compute_reward
from src.vehicle_model.bicycle_model import (
    VehicleParams,
    continuous_bicycle_ss,
    discretize_zoh,
)


# ---------------- Pure-RL environment (action = steering delta) ----------------
@dataclass
class PureRLEnvConfig(EnvConfig):
    delta_max: float = 0.5
    delta_rate_max: float = 0.1


class PureRLEnv(gym.Env):
    """Same plant / reward as CarlaMPCEnv, but the agent commands delta directly."""

    metadata = {"render_modes": []}

    def __init__(self, vehicle: VehicleParams | None = None,
                 reward_weights: RewardWeights | None = None,
                 config: PureRLEnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.veh_nominal = vehicle or VehicleParams(1500.0, 2500.0, 1.2, 1.4, 80000.0, 85000.0)
        self.rweights = reward_weights or RewardWeights()
        self.cfg = config or PureRLEnvConfig()
        # action: a single steering command in [-1, 1], rescaled to [-delta_max, delta_max]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=_OBS_LOW, high=_OBS_HIGH, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._reset_internal()

    def _reset_internal(self):
        self._x = np.zeros(4)
        self._delta_prev = 0.0
        self._sim_step = 0
        self._vx = 20.0
        self._kappa_profile = np.zeros(1)

    def _make_kappa(self, n):
        km = min(self.cfg.kappa_max, self.cfg.a_lat_max / max(self._vx, 1.0) ** 2)
        if km <= 1e-6:
            return np.zeros(n)
        s = np.arange(n) * self.cfg.Ts
        prof = np.zeros(n)
        for _ in range(3):
            wl = self._rng.uniform(4.0, 20.0); ph = self._rng.uniform(0, 2*np.pi)
            prof += self._rng.uniform(0.3, 1.0) * np.sin(2*np.pi*s/wl + ph)
        prof = prof / np.max(np.abs(prof)) * km
        return prof

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._vx = float(self._rng.uniform(*self.cfg.vx_range))
        self._kappa_profile = self._make_kappa(self.cfg.max_episode_steps + 50)
        self._x = np.array([self._rng.uniform(-0.5, 0.5),
                            self._rng.uniform(-0.05, 0.05), 0.0, 0.0])
        self._delta_prev = 0.0; self._sim_step = 0
        return self._obs(self._kappa_profile[0]), {}

    def _obs(self, kappa):
        ey, e_psi, _vy, psi_dot = self._x
        ay = self._vx * psi_dot
        return np.clip(
            np.array([self._vx, kappa, ey, e_psi, psi_dot, ay, self._delta_prev],
                     dtype=np.float32),
            _OBS_LOW, _OBS_HIGH,
        )

    def step(self, action):
        cfg = self.cfg
        # rescale action -> physical delta + rate-limit (apply the same actuator
        # constraint MPC enforces, so the comparison is fair)
        delta_cmd = float(np.clip(action[0], -1.0, 1.0)) * cfg.delta_max
        delta = float(np.clip(delta_cmd, self._delta_prev - cfg.delta_rate_max,
                              self._delta_prev + cfg.delta_rate_max))
        delta = float(np.clip(delta, -cfg.delta_max, cfg.delta_max))

        # plant + curvature disturbance (nominal vehicle here for simplicity)
        A_c, B_c = continuous_bicycle_ss(self._vx, self.veh_nominal)
        E_c = np.array([[0.0], [-self._vx], [0.0], [0.0]])
        A_d, BE_d = discretize_zoh(A_c, np.hstack([B_c, E_c]), cfg.Ts)
        B_d, E_d = BE_d[:, :1], BE_d[:, 1:2]
        kappa = float(self._kappa_profile[min(self._sim_step, len(self._kappa_profile)-1)])
        self._x = A_d @ self._x + (B_d @ np.array([delta])).ravel() + (E_d @ np.array([kappa])).ravel()
        ddelta = delta - self._delta_prev
        self._delta_prev = delta; self._sim_step += 1

        ey, e_psi, _vy, psi_dot = self._x
        ay = self._vx * psi_dot
        r = compute_reward(ey, e_psi, ddelta, ay, self.rweights)
        terminated = abs(ey) > cfg.ey_terminate
        truncated = self._sim_step >= cfg.max_episode_steps
        info = {"ey": float(ey), "e_psi": float(e_psi)}
        return self._obs(kappa), float(r), terminated, truncated, info


# ---------------- training entry point ----------------
def train_pure_rl(total_timesteps: int = 25_000, kappa_max: float = 0.05,
                  out_dir: Path | None = None, seed: int = 0):
    """Train the no-MPC SAC baseline on the PureRLEnv."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ROOT = Path(__file__).resolve().parents[2]
    out_dir = out_dir or (ROOT / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / "pure_rl"
    log_dir.mkdir(parents=True, exist_ok=True)

    def _mk():
        env = PureRLEnv(config=PureRLEnvConfig(kappa_max=kappa_max), seed=seed)
        return Monitor(env, filename=str(log_dir / "monitor"))
    venv = DummyVecEnv([_mk])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = SAC("MlpPolicy", venv, learning_rate=3e-4, batch_size=256,
                buffer_size=200_000, gamma=0.99, tau=0.005,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_starts=1000, verbose=0, seed=seed,
                tensorboard_log=str(log_dir))
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(out_dir / "sac_pure_rl")
    venv.save(str(out_dir / "vecnormalize_pure_rl.pkl"))
    return model


# ---------------- controller wrapper (matches baseline interface) ----------------
class PureRL:
    """Loads a trained pure-RL policy and returns delta given an obs vector."""

    name = "pure_rl"

    def __init__(self, model_path: str | Path, vecnorm_path: str | Path,
                 delta_max: float = 0.5, delta_rate_max: float = 0.1):
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        venv = DummyVecEnv([lambda: PureRLEnv()])
        self._venv = VecNormalize.load(str(vecnorm_path), venv)
        self._venv.training = False; self._venv.norm_reward = False
        self._model = SAC.load(str(model_path), env=self._venv)
        self.delta_max = delta_max
        self.delta_rate_max = delta_rate_max

    def compute(self, x, vx, kappa_preview, delta_prev=0.0):
        """Build the same 7-D obs the env uses, run the policy, return delta."""
        ey, e_psi, _vy, psi_dot = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        ay = vx * psi_dot
        k0 = float(kappa_preview if np.isscalar(kappa_preview) else kappa_preview[0])
        obs = np.array([vx, k0, ey, e_psi, psi_dot, ay, delta_prev], dtype=np.float32)
        obs_n = self._venv.normalize_obs(obs.reshape(1, -1))
        a, _ = self._model.predict(obs_n, deterministic=True)
        delta = float(a[0, 0]) * self.delta_max
        delta = float(np.clip(delta, delta_prev - self.delta_rate_max,
                              delta_prev + self.delta_rate_max))
        return float(np.clip(delta, -self.delta_max, self.delta_max))

    def reset(self):
        pass
