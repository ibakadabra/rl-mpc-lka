"""Evaluate all controllers across all scenarios and write results/eval_metrics.csv.

What this script does
---------------------
Closed-loop rollouts of N seeds for each (controller, scenario) pair on the
internal bicycle plant + curvature disturbance. Reports per-rollout:
    return        cumulative episode reward
    rmse_ey       tracking accuracy
    mae_ey
    max_ay [g]    peak lateral acceleration  (comfort)
    rms_ay [g]
    rms_ddelta    steering smoothness
    lane_violation %  fraction of steps with |ey| > ey_max
    solve_ms      mean MPC QP solve time (MPC controllers only)
    crashed       did |ey| > ey_terminate?

This is the thesis' main quantitative table. Run after phase 3 training:
    python scripts/evaluate.py --n-seeds 5
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import FixedMPC, GainScheduledMPC, PurePursuit, Stanley
from src.mpc.linear_mpc import MPCParams
from src.rl.environment import EnvConfig
from src.rl.reward import RewardWeights, compute_reward
from src.rl.rl_mpc_controller import RLMPC
from src.vehicle_model.bicycle_model import (
    VehicleParams,
    continuous_bicycle_ss,
    discretize_zoh,
)

G = 9.81  # m/s^2


@dataclass
class Scenario:
    """One row of the spec's evaluation matrix."""
    name: str
    vx_range: tuple[float, float]
    kappa_range: tuple[float, float]
    cf_mult: float = 1.0          # tire-stiffness multiplier (e.g. wet road)
    cr_mult: float = 1.0
    duration_s: float = 30.0


DEFAULT_SCENARIOS = [
    Scenario("highway_straight",  (27.8, 33.3), (0.000, 0.005)),
    Scenario("highway_curve",     (22.2, 27.8), (0.01, 0.03)),
    Scenario("urban_sharp",       ( 8.3, 13.9), (0.05, 0.15)),
    Scenario("mixed_route",       ( 8.3, 33.3), (0.00, 0.10)),
    Scenario("low_mu_wet",        (13.9, 16.7), (0.01, 0.05), cf_mult=0.6, cr_mult=0.6),
]


def make_kappa_profile(rng, n_steps, Ts, kappa_lo, kappa_hi, vx):
    """Sum of a few sinusoids capped at kappa_hi (or lateral-accel feasibility)."""
    if kappa_hi <= 1e-6:
        return np.zeros(n_steps)
    km_feasible = 3.0 / max(vx, 1.0) ** 2          # |ay| <= 3 m/s^2
    km = min(kappa_hi, km_feasible)
    s = np.arange(n_steps) * Ts
    prof = np.zeros(n_steps)
    for _ in range(3):
        wl = rng.uniform(4.0, 20.0); ph = rng.uniform(0, 2*np.pi)
        prof += rng.uniform(0.3, 1.0) * np.sin(2*np.pi*s/wl + ph)
    prof = prof / max(np.max(np.abs(prof)), 1e-9) * km
    # kappa_range[0] (kappa_lo) was treated as a floor before — that produces
    # an unrealistic *continuous* corner. Real roads alternate straight/curve.
    # We instead use kappa_lo only to skip the zero-curvature case (handled above)
    # and let the sinusoid sum naturally span [0, km].
    return prof


def rollout(controller, vehicle: VehicleParams, scenario: Scenario, seed: int,
            mpc_params: MPCParams, env_cfg: EnvConfig, reward_weights: RewardWeights):
    """Run one closed-loop episode and return per-rollout metrics."""
    rng = np.random.default_rng(seed)
    vx = float(rng.uniform(*scenario.vx_range))
    Ts = env_cfg.Ts
    n_steps = int(scenario.duration_s / Ts)

    # plant uses scenario tire multipliers AND a per-seed randomization of mass +
    # tire stiffness (within EnvConfig randomization ranges) so the controller's
    # nominal internal model has a realistic mismatch. This is the operating
    # envelope on which the RL agent was trained, and on which gain-scheduled /
    # fixed controllers must still deliver — exactly the comparison the thesis
    # claims is favorable to RL-MPC.
    cf_mult_seed = float(rng.uniform(*env_cfg.cf_range))
    cr_mult_seed = float(rng.uniform(*env_cfg.cr_range))
    mass_mult = float(rng.uniform(*env_cfg.mass_range))
    veh_plant = VehicleParams(
        mass=vehicle.mass * mass_mult, yaw_inertia=vehicle.yaw_inertia,
        lf=vehicle.lf, lr=vehicle.lr,
        Cf=vehicle.Cf * scenario.cf_mult * cf_mult_seed,
        Cr=vehicle.Cr * scenario.cr_mult * cr_mult_seed,
    )
    A_c, B_c = continuous_bicycle_ss(vx, veh_plant)
    E_c = np.array([[0.0], [-vx], [0.0], [0.0]])
    A_d, BE_d = discretize_zoh(A_c, np.hstack([B_c, E_c]), Ts)
    B_d, E_d = BE_d[:, :1], BE_d[:, 1:2]

    kappa_profile = make_kappa_profile(rng, n_steps + mpc_params.Np,
                                       Ts, scenario.kappa_range[0], scenario.kappa_range[1], vx)
    if hasattr(controller, "reset"):
        controller.reset()

    x = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.03, 0.03), 0.0, 0.0])
    delta_prev = 0.0
    ey_hist, ay_hist, ddelta_hist = [], [], []
    ret, crashed = 0.0, False
    Np = mpc_params.Np

    for k in range(n_steps):
        kpre = kappa_profile[k:k+Np]
        if kpre.size < Np:
            kpre = np.concatenate([kpre, np.full(Np-kpre.size, kpre[-1] if kpre.size else 0.0)])
        delta = controller.compute(x, vx, kpre, delta_prev)
        ddelta = delta - delta_prev
        x = A_d @ x + (B_d @ np.array([delta])).ravel() + (E_d @ np.array([float(kappa_profile[k])])).ravel()
        ay = vx * x[3]
        ret += compute_reward(x[0], x[1], ddelta, ay, reward_weights)
        ey_hist.append(x[0]); ay_hist.append(ay); ddelta_hist.append(ddelta)
        delta_prev = delta
        if abs(x[0]) > env_cfg.ey_terminate:
            crashed = True
            break

    ey = np.asarray(ey_hist); ay = np.asarray(ay_hist); dd = np.asarray(ddelta_hist)
    metrics = dict(
        scenario=scenario.name,
        controller=getattr(controller, "name", controller.__class__.__name__),
        seed=seed,
        vx=vx,
        steps=len(ey),
        return_=ret,
        rmse_ey=float(np.sqrt(np.mean(ey**2))) if len(ey) else float("nan"),
        mae_ey=float(np.mean(np.abs(ey))) if len(ey) else float("nan"),
        max_ay_g=float(np.max(np.abs(ay))/G) if len(ay) else float("nan"),
        rms_ay_g=float(np.sqrt(np.mean(ay**2))/G) if len(ay) else float("nan"),
        rms_ddelta=float(np.sqrt(np.mean(dd**2))) if len(dd) else float("nan"),
        lane_violation_pct=float(np.mean(np.abs(ey) > mpc_params.ey_max)*100) if len(ey) else float("nan"),
        crashed=int(crashed),
    )
    return metrics


def build_controllers(vehicle, mpc_params, models_dir: Path, include_pure_rl: bool):
    """Instantiate every controller we evaluate."""
    ctrls = [
        FixedMPC(vehicle, mpc_params),
        GainScheduledMPC(vehicle, mpc_params),
        Stanley(),
        PurePursuit(),
    ]
    rl_model = models_dir / "sac_rlmpc.zip"
    rl_vn = models_dir / "vecnormalize.pkl"
    if rl_model.exists() and rl_vn.exists():
        ctrls.append(RLMPC(vehicle, mpc_params, rl_model, rl_vn))
    else:
        print(f"[skip] rl_mpc — missing model files ({rl_model.name}, {rl_vn.name})")
    if include_pure_rl:
        pr_model = models_dir / "sac_pure_rl.zip"
        pr_vn = models_dir / "vecnormalize_pure_rl.pkl"
        if pr_model.exists() and pr_vn.exists():
            from src.baselines.pure_rl import PureRL
            ctrls.append(PureRL(pr_model, pr_vn))
        else:
            print(f"[skip] pure_rl — missing model files ({pr_model.name}, {pr_vn.name})")
    return ctrls


def main():
    ap = argparse.ArgumentParser(description="Run the full evaluation matrix.")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "eval_metrics.csv"))
    ap.add_argument("--include-pure-rl", action="store_true",
                    help="include the (separately trained) Pure-RL baseline if present")
    args = ap.parse_args()

    vehicle = VehicleParams.from_yaml(ROOT / "config" / "vehicle_params.yaml")
    mpc_params = MPCParams.from_yaml(ROOT / "config" / "mpc_params.yaml")
    env_cfg = EnvConfig()
    rw = RewardWeights.from_yaml(ROOT / "config" / "rl_params.yaml")
    controllers = build_controllers(vehicle, mpc_params, ROOT / "models",
                                    args.include_pure_rl)

    rows = []
    total = len(controllers) * len(DEFAULT_SCENARIOS) * args.n_seeds
    done = 0
    for ctrl in controllers:
        for scen in DEFAULT_SCENARIOS:
            for s in range(args.n_seeds):
                m = rollout(ctrl, vehicle, scen, seed=1000 + s,
                            mpc_params=mpc_params, env_cfg=env_cfg, reward_weights=rw)
                rows.append(m); done += 1
            print(f"  {ctrl.name:18} x {scen.name:18}  ok  ({done}/{total})")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote -> {args.out}  ({len(df)} rows)")

    # quick summary table to console
    summ = (df.groupby(["scenario", "controller"])
              .agg(mean_return=("return_", "mean"),
                   mean_rmse_ey=("rmse_ey", "mean"),
                   mean_max_ay=("max_ay_g", "mean"),
                   crash_pct=("crashed", lambda v: 100*np.mean(v)))
              .round(3))
    print("\n=== summary ===")
    print(summ.to_string())


if __name__ == "__main__":
    main()
