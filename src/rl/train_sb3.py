"""SAC training (Stable-Baselines3) for the RL-tuned-MPC agent.

What the agent is actually learning
-----------------------------------
The policy pi_theta(s) maps a 7-D driving context s = [vx, kappa, ey, e_psi,
psi_dot, ay, delta_prev] to a 3-D action that the weight-mapper turns into the
MPC weights (q1, q2, r1). So we are NOT learning to steer — we are learning a
*gain-scheduling law* for the MPC, where the "schedule" is a neural net trained
by trial and error instead of hand-tuned lookup tables.

Why SAC (Soft Actor-Critic)
---------------------------
- Continuous 3-D action (the weights are continuous knobs) -> need a continuous
  policy; SAC is built for that.
- Off-policy: it reuses every past transition from a replay buffer, so it is
  sample-efficient. That matters because each environment step costs ten MPC QP
  solves — samples are expensive.
- Maximum-entropy objective: SAC maximizes reward PLUS policy entropy, i.e. it
  is rewarded for staying as random as possible while still performing well.
  In control terms this is automatic, self-annealing exploration — it keeps
  "dithering" the weights to discover better schedules and stops once confident.

Observation normalization (VecNormalize)
----------------------------------------
The observation channels live on wildly different scales (vx ~ 20, kappa ~ 0.03,
delta ~ 0.1). A neural net learns badly when inputs are unscaled, just like an
ill-conditioned plant is hard to control. VecNormalize keeps a running
mean/std per channel and whitens the observations online.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from src.mpc.linear_mpc import MPCParams
from src.rl.environment import CarlaMPCEnv, EnvConfig
from src.rl.reward import RewardWeights
from src.rl.weight_mapper import WeightRanges
from src.vehicle_model.bicycle_model import VehicleParams

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"


def load_sac_hyperparams(path: Path | None = None) -> dict:
    """Read the SAC block from config/rl_params.yaml into SB3 kwargs."""
    path = path or (CONFIG / "rl_params.yaml")
    with open(path, "r") as f:
        sac = yaml.safe_load(f)["sac"]
    return dict(
        learning_rate=sac["learning_rate"],
        batch_size=sac["batch_size"],
        buffer_size=sac["buffer_size"],
        gamma=sac["gamma"],
        tau=sac["tau"],
        policy_kwargs=dict(net_arch=list(sac["hidden_layers"])),
        ent_coef="auto" if sac["auto_entropy_tuning"] else 0.1,
    )


def make_env_fn(env_config: EnvConfig, monitor_path: Path | None, seed: int):
    """Factory that builds one Monitor-wrapped environment instance.

    Each parallel worker gets its own `seed` so the workers explore different
    roads/plants instead of identical episodes.
    """
    def _init():
        veh = VehicleParams.from_yaml(CONFIG / "vehicle_params.yaml")
        env = CarlaMPCEnv(
            vehicle=veh,
            mpc_params=MPCParams.from_yaml(CONFIG / "mpc_params.yaml"),
            weight_ranges=WeightRanges.from_yaml(
                CONFIG / "rl_params.yaml", CONFIG / "mpc_params.yaml"),
            reward_weights=RewardWeights.from_yaml(CONFIG / "rl_params.yaml"),
            config=env_config,
            seed=seed,
        )
        return Monitor(env, filename=str(monitor_path) if monitor_path else None)
    return _init


def train(
    total_timesteps: int = 200_000,
    env_config: EnvConfig | None = None,
    out_dir: Path | None = None,
    learning_starts: int = 1000,
    seed: int = 0,
    verbose: int = 1,
    n_envs: int = 1,
    checkpoint_freq: int = 50_000,
) -> SAC:
    """Train a SAC agent that tunes the MPC weights, with obs normalization.

    n_envs           number of parallel rollout workers. The bottleneck is the
                     CPU-bound MPC QP (10 solves per env step), so running N
                     envs in separate processes scales throughput ~N-fold.
    checkpoint_freq  save an intermediate model every this many TOTAL env steps,
                     so a long run survives an interruption.
    """
    env_config = env_config or EnvConfig(kappa_max=0.05, randomize=True)
    out_dir = out_dir or (ROOT / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # one env factory per worker, each with its own seed and monitor file
    env_fns = [make_env_fn(env_config, log_dir / f"monitor_{i}", seed + i)
               for i in range(max(1, n_envs))]
    venv = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    # whiten observations online; keep rewards raw so logged returns are real.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # TensorBoard logging is optional: only enable it if the package is present,
    # otherwise SB3 raises at learn() time. The monitor CSV + learning-curve plot
    # already capture the training return without it.
    try:
        import tensorboard  # noqa: F401
        tb_log = str(log_dir)
    except Exception:
        tb_log = None
        print("[info] tensorboard not installed -> skipping TB logs "
              "(pip install tensorboard to enable)")

    model = SAC(
        "MlpPolicy",
        venv,
        learning_starts=learning_starts,
        tensorboard_log=tb_log,
        seed=seed,
        verbose=verbose,
        **load_sac_hyperparams(),
    )

    # periodic checkpoint so a multi-hour run survives an interruption. save_freq
    # is per-worker, so divide by n_envs to land on the requested TOTAL-step cadence.
    ckpt_dir = out_dir / "checkpoints"
    callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // max(1, n_envs), 1),
        save_path=str(ckpt_dir),
        name_prefix="sac_rlmpc",
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callback)

    model.save(out_dir / "sac_rlmpc")
    venv.save(str(out_dir / "vecnormalize.pkl"))  # need the obs stats at eval time
    print(f"saved model -> {out_dir / 'sac_rlmpc.zip'}")
    return model


def plot_learning_curve(out_png: Path | None = None):
    """Plot the rolling-mean episode return from the Monitor log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3.common.results_plotter import load_results, ts2xy

    log_dir = ROOT / "logs"
    out_png = out_png or (ROOT / "results" / "phase3_learning_curve.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x, y = ts2xy(load_results(str(log_dir)), "timesteps")
    if len(y) == 0:
        print("no monitor data yet")
        return
    window = max(1, min(20, len(y) // 5))
    y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
    x_smooth = x[window - 1:]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, alpha=0.25, color="tab:blue", label="episode return")
    ax.plot(x_smooth, y_smooth, color="tab:red", lw=2, label=f"rolling mean ({window})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("SAC learning curve — RL-tuned MPC (internal sim)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"saved learning curve -> {out_png}")
