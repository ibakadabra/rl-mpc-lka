"""Wrap the trained SAC RL-MPC agent in the same `compute(...)` interface as the
baselines, so `evaluate.py` can treat all 6 controllers identically."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.mpc.linear_mpc import LinearMPC, MPCParams
from src.rl.weight_mapper import WeightRanges, action_to_weights
from src.vehicle_model.bicycle_model import VehicleParams


class RLMPC:
    """Trained RL agent picks Q,R; the same LinearMPC then solves for delta."""

    name = "rl_mpc"

    def __init__(self, vehicle: VehicleParams, mpc_params: MPCParams | None,
                 model_path: str | Path, vecnorm_path: str | Path,
                 weight_ranges: WeightRanges | None = None,
                 mpc_steps_per_rl: int = 10):
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from src.rl.environment import CarlaMPCEnv, EnvConfig
        venv = DummyVecEnv([lambda: CarlaMPCEnv(config=EnvConfig())])
        self._venv = VecNormalize.load(str(vecnorm_path), venv)
        self._venv.training = False; self._venv.norm_reward = False
        self._model = SAC.load(str(model_path), env=self._venv)

        self.mpc = LinearMPC(vehicle, mpc_params or MPCParams())
        self.wranges = weight_ranges or WeightRanges()
        self.mpc_steps_per_rl = mpc_steps_per_rl
        self._tick = 0
        self._Q = np.array([100.0, 10.0, self.mpc.p.q3_fixed, self.mpc.p.q4_fixed])
        self._R = 1.0

    def reset(self):
        self._tick = 0
        self.mpc._last_z = None

    def _refresh_weights(self, obs7):
        obs_n = self._venv.normalize_obs(obs7.reshape(1, -1))
        a, _ = self._model.predict(obs_n, deterministic=True)
        self._Q, self._R = action_to_weights(a[0], self.wranges)

    def compute(self, x, vx, kappa_preview, delta_prev=0.0):
        """Refresh Q,R every `mpc_steps_per_rl` ticks (2 Hz over 50 Hz MPC)."""
        if self._tick % self.mpc_steps_per_rl == 0:
            ey, e_psi, _vy, psi_dot = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            ay = vx * psi_dot
            k0 = float(kappa_preview if np.isscalar(kappa_preview) else kappa_preview[0])
            obs7 = np.array([vx, k0, ey, e_psi, psi_dot, ay, delta_prev], dtype=np.float32)
            self._refresh_weights(obs7)
        self._tick += 1
        res = self.mpc.solve(x, vx, self._Q, self._R, delta_prev,
                             kappa_preview=kappa_preview)
        return res.delta
