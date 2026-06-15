"""ISO standardized-maneuver evaluation on the internal bicycle plant.

Complements scripts/evaluate.py (random operating-envelope scenarios) and
scripts/evaluate_carla.py (photo-realistic road geometry) with the two ISO
closed-course maneuvers whose exact reference path can only be reproduced in
the controllable internal sim:

  ISO 3888-2  severe double lane change ("moose test")
      The reference centerline is the ISO cone-gate path; tracking it means
      e_y == 0. Run at several entry speeds; score ISO 11270 limits + whether
      the body stayed inside the gated corridor.

  ISO 4138    steady-state circular driving (understeer characterisation)
      Constant radius, speed ramped up so a_y sweeps to the grip limit.
      Report the understeer gradient K = d(steer)/d(a_y).

Output: results/eval_iso_metrics.csv + console summary.

Usage:
    python scripts/evaluate_iso.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import FixedMPC, GainScheduledMPC, PurePursuit, Stanley
from src.evaluation import (
    evaluate_iso11270,
    iso3888_2_double_lane_change,
    iso4138_constant_radius,
    understeer_gradient,
)
from src.mpc.linear_mpc import MPCParams
from src.rl.rl_mpc_controller import RLMPC
from src.vehicle_model.bicycle_model import (
    VehicleParams,
    continuous_bicycle_ss,
    discretize_zoh,
)

G = 9.81
# ISO 3888-2 entry speeds (~30 / 40 / 50 km/h). The standard cone path packs a
# 3.5 m shift into a 13.5 m transition (curvature radius ~10 m), so the lateral
# acceleration it demands grows as vx^2 and crosses the grip/comfort envelope
# around 50 km/h. Sweeping these speeds yields a clean pass->fail gradient.
# (The linear bicycle plant has no tire saturation, so beyond-grip demand shows
#  up as an honest ISO 11270 a_y-limit failure rather than a slide.)
DLC_ENTRY_SPEEDS = (8.3, 11.1, 13.9)
DLC_CORRIDOR_HALF = 1.8                 # m   max |e_y| before a gate is hit


def _discrete(vehicle, vx, Ts):
    """ZOH-discretized [A_d | B_d | E_d] for the LPV model at speed vx."""
    A_c, B_c = continuous_bicycle_ss(vx, vehicle)
    E_c = np.array([[0.0], [-vx], [0.0], [0.0]])
    A_d, BE_d = discretize_zoh(A_c, np.hstack([B_c, E_c]), Ts)
    return A_d, BE_d[:, :1], BE_d[:, 1:2]


def run_dlc(controller, vehicle, mpc_params, vx, Ts=0.02):
    """ISO 3888-2 double lane change at constant entry speed vx."""
    # Sample the maneuver at exactly the per-step travelled distance ds=vx*Ts so
    # that arc-length index k lines up with how far the car has driven by time
    # step k (s[k] = k*vx*Ts). A coarser ds would compress the maneuver in time.
    man = iso3888_2_double_lane_change(vx=vx, ds=vx * Ts)
    n_steps = len(man.s)
    kappa_profile = man.kappa
    Np = mpc_params.Np

    A_d, B_d, E_d = _discrete(vehicle, vx, Ts)
    if hasattr(controller, "reset"):
        controller.reset()

    x = np.zeros(4)               # start centered on the reference path
    delta_prev = 0.0
    ey_h, ay_h, dd_h = [], [], []
    for k in range(n_steps):
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

    ey = np.asarray(ey_h); ay = np.asarray(ay_h); dd = np.asarray(dd_h)
    iso = evaluate_iso11270(ey, ay, Ts, lane_half_width=mpc_params.ey_max)
    corridor_ok = bool(np.max(np.abs(ey)) <= DLC_CORRIDOR_HALF) if len(ey) else False
    row = dict(
        maneuver="iso3888_2_dlc",
        entry_speed=vx,
        controller=getattr(controller, "name", controller.__class__.__name__),
        rmse_ey=float(np.sqrt(np.mean(ey**2))) if len(ey) else float("nan"),
        max_abs_ey=float(np.max(np.abs(ey))) if len(ey) else float("nan"),
        max_ay_g=float(np.max(np.abs(ay)) / G) if len(ay) else float("nan"),
        rms_ddelta=float(np.sqrt(np.mean(dd**2))) if len(dd) else float("nan"),
        corridor_pass=int(corridor_ok),
    )
    row.update(iso.as_dict())
    return row


def run_4138(controller, vehicle, mpc_params, radius=100.0,
             v_start=5.0, v_end=25.0, Ts=0.02):
    """ISO 4138 steady-state circular driving with a slow speed ramp."""
    man = iso4138_constant_radius(radius=radius, v_start=v_start, v_end=v_end,
                                  ds=0.5)
    n_steps = len(man.s)
    Np = mpc_params.Np
    kappa = float(man.kappa[0])
    if hasattr(controller, "reset"):
        controller.reset()

    x = np.zeros(4)
    delta_prev = 0.0
    steer_h, ay_h, vx_h = [], [], []
    for k in range(n_steps):
        vx = float(man.vx[k])
        A_d, B_d, E_d = _discrete(vehicle, vx, Ts)   # LPV: re-discretize as vx ramps
        kpre = np.full(Np, kappa)
        delta = controller.compute(x, vx, kpre, delta_prev)
        x = A_d @ x + (B_d @ np.array([delta])).ravel() \
            + (E_d @ np.array([kappa])).ravel()
        ay = vx * x[3]
        steer_h.append(delta); ay_h.append(ay); vx_h.append(vx)
        delta_prev = delta

    steer = np.asarray(steer_h); ay = np.asarray(ay_h)
    K = understeer_gradient(steer, ay)
    row = dict(
        maneuver="iso4138_const_radius",
        entry_speed=float("nan"),
        controller=getattr(controller, "name", controller.__class__.__name__),
        radius_m=radius,
        understeer_gradient=K,
        behaviour=("understeer" if K > 1e-4 else "oversteer" if K < -1e-4 else "neutral"),
        max_ay_g=float(np.max(np.abs(ay)) / G) if len(ay) else float("nan"),
    )
    return row


def build_controllers(vehicle, mpc_params, models_dir: Path):
    ctrls = [FixedMPC(vehicle, mpc_params), GainScheduledMPC(vehicle, mpc_params),
             Stanley(), PurePursuit()]
    rl_m = models_dir / "sac_rlmpc.zip"; rl_v = models_dir / "vecnormalize.pkl"
    if rl_m.exists() and rl_v.exists():
        ctrls.append(RLMPC(vehicle, mpc_params, rl_m, rl_v))
    else:
        print(f"[skip] rl_mpc — models not found in {models_dir}")
    return ctrls


def main():
    ap = argparse.ArgumentParser(description="ISO closed-course maneuvers (internal sim).")
    ap.add_argument("--out", default=str(ROOT / "results" / "eval_iso_metrics.csv"))
    ap.add_argument("--radius", type=float, default=100.0, help="ISO 4138 radius [m]")
    args = ap.parse_args()

    vehicle = VehicleParams.from_yaml(ROOT / "config" / "vehicle_params.yaml")
    mpc_params = MPCParams.from_yaml(ROOT / "config" / "mpc_params.yaml")
    controllers = build_controllers(vehicle, mpc_params, ROOT / "models")

    rows = []
    for ctrl in controllers:
        for vx in DLC_ENTRY_SPEEDS:
            rows.append(run_dlc(ctrl, vehicle, mpc_params, vx))
        rows.append(run_4138(ctrl, vehicle, mpc_params, radius=args.radius))
        print(f"  {ctrl.name:18}  ok")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote -> {args.out}  ({len(df)} rows)")

    # ISO 3888-2 summary
    dlc = df[df.maneuver == "iso3888_2_dlc"]
    if len(dlc):
        print("\n=== ISO 3888-2 double lane change ===")
        print(dlc.groupby(["controller", "entry_speed"])
                .agg(max_abs_ey=("max_abs_ey", "mean"),
                     max_ay_g=("max_ay_g", "mean"),
                     corridor_pass=("corridor_pass", "max"),
                     iso11270_pass=("iso11270_overall_pass", "max"))
                .round(3).to_string())

    # ISO 4138 summary
    c4138 = df[df.maneuver == "iso4138_const_radius"]
    if len(c4138):
        print("\n=== ISO 4138 steady-state cornering ===")
        print(c4138[["controller", "understeer_gradient", "behaviour", "max_ay_g"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
