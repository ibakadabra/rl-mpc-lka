"""Hard-scenario evaluation: RL-MPC vs gain-scheduled vs fixed MPC.

Purpose
-------
The easy operating points make every MPC look identical (sub-cm error), so they
cannot test the thesis claim. This script runs the scenario classes that an
online weight-tuner SHOULD win and a vx-only schedule (or a single fixed Q,R)
structurally CANNOT cover:

  * strong low-mu (both axles)          -- nominal-tuned schedule mismatched
  * mu-split front / rear               -- understeer/oversteer balance shifts
  * within-episode dry->wet mid-corner  -- a static schedule must compromise;
                                           RL retunes every 0.5 s on the error
  * high speed + sharp curvature         -- limit handling, tracking/comfort
                                           trade-off becomes state-dependent

The MPC's internal model stays NOMINAL for every controller; the plant carries
the designed mismatch. Geometric controllers (Stanley/Pure Pursuit) are run only
as an optional context floor (--with-geometric), not the headline comparison.

Output: results/eval_hard_metrics.csv + console summary (mean +/- std over seeds).

    python scripts/evaluate_hard.py --n-seeds 12
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
from src.evaluation import evaluate_iso11270
from src.mpc.linear_mpc import MPCParams
from src.rl.environment import EnvConfig
from src.rl.rl_mpc_controller import RLMPC
from src.vehicle_model.bicycle_model import (
    VehicleParams,
    continuous_bicycle_ss,
    discretize_zoh,
)

G = 9.81


@dataclass
class HardScenario:
    name: str
    vx_range: tuple[float, float]
    kappa_range: tuple[float, float]
    cf_mult: float = 1.0
    cr_mult: float = 1.0
    mu_step_frac: float | None = None   # if set, grip drops at this episode fraction
    mu_step_mult: float = 1.0           # post-step extra multiplier on Cf,Cr
    duration_s: float = 30.0


# Each scenario targets an axis a vx-only schedule cannot see.
HARD_SCENARIOS = [
    HardScenario("highspeed_lowmu",  (27.0, 30.0), (0.005, 0.02), cf_mult=0.5, cr_mult=0.5),
    HardScenario("musplit_front",    (22.0, 27.0), (0.01, 0.03),  cf_mult=0.55, cr_mult=1.0),
    HardScenario("musplit_rear",     (22.0, 27.0), (0.01, 0.03),  cf_mult=1.0,  cr_mult=0.55),
    HardScenario("dry2wet_corner",   (22.0, 27.0), (0.01, 0.03),  cf_mult=1.0,  cr_mult=1.0,
                 mu_step_frac=0.5, mu_step_mult=0.5),
    HardScenario("highspeed_sharp",  (28.0, 30.0), (0.02, 0.03),  cf_mult=0.8,  cr_mult=0.8),
]


def make_kappa_profile(rng, n_steps, Ts, kappa_lo, kappa_hi, vx):
    """Sum of a few sinusoids capped at kappa_hi (and lateral-accel feasibility)."""
    if kappa_hi <= 1e-6:
        return np.zeros(n_steps)
    km = min(kappa_hi, 5.0 / max(vx, 1.0) ** 2)   # |ay| <= ~5 m/s^2 generated road
    s = np.arange(n_steps) * Ts
    prof = np.zeros(n_steps)
    for _ in range(3):
        wl = rng.uniform(4.0, 20.0); ph = rng.uniform(0, 2 * np.pi)
        prof += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * s / wl + ph)
    prof = prof / max(np.max(np.abs(prof)), 1e-9) * km
    return prof


def _plant(vehicle, cf_mult, cr_mult, mass_mult, vx, Ts):
    veh = VehicleParams(mass=vehicle.mass * mass_mult, yaw_inertia=vehicle.yaw_inertia,
                        lf=vehicle.lf, lr=vehicle.lr,
                        Cf=vehicle.Cf * cf_mult, Cr=vehicle.Cr * cr_mult)
    A_c, B_c = continuous_bicycle_ss(vx, veh)
    E_c = np.array([[0.0], [-vx], [0.0], [0.0]])
    A_d, BE = discretize_zoh(A_c, np.hstack([B_c, E_c]), Ts)
    return A_d, BE[:, :1], BE[:, 1:2]


def rollout(controller, vehicle, scen: HardScenario, seed, mpc_params, env_cfg):
    """One closed-loop episode on the (mismatched) plant; returns metrics."""
    rng = np.random.default_rng(seed)
    vx = float(rng.uniform(*scen.vx_range))
    Ts = env_cfg.Ts
    n_steps = int(scen.duration_s / Ts)
    Np = mpc_params.Np

    # modest per-seed noise on top of the scenario's designed mismatch
    mass_mult = float(rng.uniform(*env_cfg.mass_range))
    cf0 = scen.cf_mult * float(rng.uniform(0.95, 1.05))
    cr0 = scen.cr_mult * float(rng.uniform(0.95, 1.05))

    A_d, B_d, E_d = _plant(vehicle, cf0, cr0, mass_mult, vx, Ts)
    # optional within-episode grip drop
    step_at = int(scen.mu_step_frac * n_steps) if scen.mu_step_frac is not None else None
    A_d2, B_d2, E_d2 = (None, None, None)
    if step_at is not None:
        A_d2, B_d2, E_d2 = _plant(vehicle, cf0 * scen.mu_step_mult,
                                  cr0 * scen.mu_step_mult, mass_mult, vx, Ts)

    kappa_profile = make_kappa_profile(rng, n_steps + Np, Ts,
                                       scen.kappa_range[0], scen.kappa_range[1], vx)
    if hasattr(controller, "reset"):
        controller.reset()

    x = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.03, 0.03), 0.0, 0.0])
    delta_prev = 0.0
    ey_h, ay_h, dd_h = [], [], []
    crashed = False
    for k in range(n_steps):
        if step_at is not None and k == step_at:
            A_d, B_d, E_d = A_d2, B_d2, E_d2   # grip drops here
        kpre = kappa_profile[k:k + Np]
        if kpre.size < Np:
            kpre = np.concatenate([kpre, np.full(Np - kpre.size, kpre[-1] if kpre.size else 0.0)])
        delta = controller.compute(x, vx, kpre, delta_prev)
        ddelta = delta - delta_prev
        x = A_d @ x + (B_d @ np.array([delta])).ravel() \
            + (E_d @ np.array([float(kappa_profile[k])])).ravel()
        ay = vx * x[3]
        ey_h.append(x[0]); ay_h.append(ay); dd_h.append(ddelta)
        delta_prev = delta
        if abs(x[0]) > env_cfg.ey_terminate:
            crashed = True
            break

    ey = np.asarray(ey_h); ay = np.asarray(ay_h); dd = np.asarray(dd_h)
    m = dict(
        scenario=scen.name,
        controller=getattr(controller, "name", controller.__class__.__name__),
        seed=seed, vx=vx, steps=len(ey),
        rmse_ey=float(np.sqrt(np.mean(ey**2))) if len(ey) else float("nan"),
        max_abs_ey=float(np.max(np.abs(ey))) if len(ey) else float("nan"),
        max_ay_g=float(np.max(np.abs(ay)) / G) if len(ay) else float("nan"),
        rms_ddelta=float(np.sqrt(np.mean(dd**2))) if len(dd) else float("nan"),
        crashed=int(crashed),
    )
    m.update(evaluate_iso11270(ey, ay, Ts, lane_half_width=mpc_params.ey_max).as_dict())
    return m


def build_controllers(vehicle, mpc_params, models_dir, with_geometric):
    ctrls = [FixedMPC(vehicle, mpc_params), GainScheduledMPC(vehicle, mpc_params)]
    rl_m = models_dir / "sac_rlmpc.zip"; rl_v = models_dir / "vecnormalize.pkl"
    if rl_m.exists() and rl_v.exists():
        ctrls.append(RLMPC(vehicle, mpc_params, rl_m, rl_v))
    else:
        print(f"[skip] rl_mpc — models not found in {models_dir}")
    if with_geometric:
        ctrls += [Stanley(), PurePursuit()]
    return ctrls


def main():
    ap = argparse.ArgumentParser(description="Hard-scenario eval: RL vs scheduled/fixed MPC.")
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--with-geometric", action="store_true",
                    help="also run Stanley/Pure Pursuit as a context floor")
    ap.add_argument("--models-dir", default=str(ROOT / "models"),
                    help="where sac_rlmpc.zip lives (e.g. models/hard for the 500k model)")
    ap.add_argument("--out", default=str(ROOT / "results" / "eval_hard_metrics.csv"))
    args = ap.parse_args()

    vehicle = VehicleParams.from_yaml(ROOT / "config" / "vehicle_params.yaml")
    mpc_params = MPCParams.from_yaml(ROOT / "config" / "mpc_params.yaml")
    env_cfg = EnvConfig()
    controllers = build_controllers(vehicle, mpc_params, Path(args.models_dir),
                                    args.with_geometric)

    rows = []
    total = len(controllers) * len(HARD_SCENARIOS) * args.n_seeds
    done = 0
    for ctrl in controllers:
        for scen in HARD_SCENARIOS:
            for s in range(args.n_seeds):
                rows.append(rollout(ctrl, vehicle, scen, seed=3000 + s,
                                    mpc_params=mpc_params, env_cfg=env_cfg))
                done += 1
            print(f"  {ctrl.name:18} x {scen.name:18}  ok ({done}/{total})")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote -> {args.out}  ({len(df)} rows)")

    # mean +/- std over seeds, headline metrics
    def mstd(v):
        return f"{np.nanmean(v):.4f}+/-{np.nanstd(v):.4f}"
    summ = (df.groupby(["scenario", "controller"])
              .agg(rmse_ey=("rmse_ey", mstd),
                   max_ay_g=("max_ay_g", lambda v: f"{np.nanmean(v):.3f}"),
                   crash_pct=("crashed", lambda v: f"{100*np.mean(v):.0f}"),
                   iso_pass=("iso11270_overall_pass", lambda v: f"{100*np.mean(v):.0f}")))
    print("\n=== HARD scenarios: mean+/-std rmse_ey [m], crash%, ISO11270 pass% ===")
    print(summ.to_string())

    # headline delta: RL vs gain-scheduled per scenario
    piv = df.groupby(["scenario", "controller"])["rmse_ey"].mean().unstack()
    if {"rl_mpc", "gain_scheduled_mpc"}.issubset(piv.columns):
        print("\n=== RL-MPC vs gain-scheduled (mean rmse_ey, lower=better) ===")
        cmp = piv[["fixed_mpc", "gain_scheduled_mpc", "rl_mpc"]].copy()
        cmp["rl_improve_%"] = 100 * (piv["gain_scheduled_mpc"] - piv["rl_mpc"]) / piv["gain_scheduled_mpc"]
        print(cmp.round(4).to_string())


if __name__ == "__main__":
    main()
