"""Phase-5 CARLA validation pass.

Runs the same controllers from Phase 4 inside CARLA Town04 and writes
results/eval_carla_metrics.csv. Compare side-by-side with eval_metrics.csv to
quantify the sim-to-sim gap.

Usage (run on the machine that owns CARLA, or with --host set):
    # 1) start the simulator
    ./CarlaUE4.sh -carla-rpc-port=2000 -RenderOffScreen

    # 2) run the validation pass
    python scripts/evaluate_carla.py --n-seeds 3 --scenarios highway_curve mixed_route

Notes
-----
* No training here. Phase-3 models (`models/sac_rlmpc.zip` etc.) are loaded
  read-only.
* Scenario targets (speed range, tire mults) mirror the internal-sim matrix
  so the comparison is apples-to-apples. CARLA's full vehicle dynamics
  substitute for our bicycle ODE; that mismatch IS the validation we want to
  measure.
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
from src.rl.rl_mpc_controller import RLMPC
from src.vehicle_model.bicycle_model import VehicleParams

G = 9.81


@dataclass
class CarlaScenario:
    name: str
    target_speed: float          # [m/s]
    cf_mult: float = 1.0
    cr_mult: float = 1.0
    spawn_index: int | None = None
    duration_s: float = 30.0


# Town04 has a big highway oval, so highway/curve scenarios map naturally.
# We do not pick spawn points by-hand here — `spawn_index=None` lets the env
# pick a deterministic-per-seed one.
DEFAULT_CARLA_SCENARIOS = [
    CarlaScenario("highway_straight",  target_speed=30.0),
    CarlaScenario("highway_curve",     target_speed=25.0),
    CarlaScenario("urban_sharp",       target_speed=10.0),
    CarlaScenario("mixed_route",       target_speed=20.0),
    CarlaScenario("low_mu_wet",        target_speed=15.0, cf_mult=0.6, cr_mult=0.6),
]


def rollout_carla(env, controller, scenario: CarlaScenario, seed: int,
                  debug: bool = False):
    """Drive one CARLA episode with the given controller; return metrics."""
    sig, x = env.reset(seed=seed, target_speed=scenario.target_speed,
                       cf_mult=scenario.cf_mult, cr_mult=scenario.cr_mult)
    if hasattr(controller, "reset"):
        controller.reset()

    delta_prev = 0.0
    ey_h, ay_h, dd_h, vx_h = [], [], [], []
    trace = []
    crashed = False
    info = {}
    step_i = 0

    while True:
        delta = controller.compute(x, sig.vx, sig.kappa_preview, delta_prev)
        ddelta = delta - delta_prev
        if debug:
            trace.append(dict(step=step_i, ey=sig.ey, e_psi=sig.e_psi,
                              vx=sig.vx, delta=delta, kappa0=float(sig.kappa_preview[0]),
                              psi_dot=sig.psi_dot))
        sig, x, done, info = env.step(delta)
        ey_h.append(sig.ey); ay_h.append(sig.ay); dd_h.append(ddelta); vx_h.append(sig.vx)
        delta_prev = delta
        step_i += 1
        if done:
            crashed = info["collided"] or abs(sig.ey) > 2.5
            if debug and crashed:
                print(f"\n  *** CRASH at step {step_i} | collided={info['collided']} "
                      f"ey={sig.ey:.3f} ***")
                n = min(20, len(trace))
                print(f"  last {n} steps before crash:")
                for t in trace[-n:]:
                    print(f"    step={t['step']:5d}  ey={t['ey']:+.3f}  e_psi={t['e_psi']:+.4f}"
                          f"  kappa={t['kappa0']:+.5f}  delta={t['delta']:+.4f}  vx={t['vx']:.1f}")
            break

    ey = np.asarray(ey_h); ay = np.asarray(ay_h); dd = np.asarray(dd_h)
    metrics = dict(
        scenario=scenario.name,
        controller=getattr(controller, "name", controller.__class__.__name__),
        seed=seed,
        steps=len(ey),
        mean_vx=float(np.mean(vx_h)) if vx_h else float("nan"),
        rmse_ey=float(np.sqrt(np.mean(ey**2))) if len(ey) else float("nan"),
        mae_ey=float(np.mean(np.abs(ey))) if len(ey) else float("nan"),
        max_ay_g=float(np.max(np.abs(ay))/G) if len(ay) else float("nan"),
        rms_ay_g=float(np.sqrt(np.mean(ay**2))/G) if len(ay) else float("nan"),
        rms_ddelta=float(np.sqrt(np.mean(dd**2))) if len(dd) else float("nan"),
        lane_violation_pct=float(np.mean(np.abs(ey) > 1.8)*100) if len(ey) else float("nan"),
        crashed=int(crashed),
    )
    return metrics


def build_controllers(vehicle, mpc_params, models_dir: Path):
    ctrls = [
        FixedMPC(vehicle, mpc_params),
        GainScheduledMPC(vehicle, mpc_params),
        Stanley(),
        PurePursuit(),
    ]
    rl_m = models_dir / "sac_rlmpc.zip"
    rl_v = models_dir / "vecnormalize.pkl"
    if rl_m.exists() and rl_v.exists():
        ctrls.append(RLMPC(vehicle, mpc_params, rl_m, rl_v))
    else:
        print(f"[skip] rl_mpc — models not found in {models_dir}")
    return ctrls


def main():
    from src.carla_interface.carla_env import CarlaConnectConfig, CarlaValidationEnv

    ap = argparse.ArgumentParser(description="CARLA validation pass.")
    ap.add_argument("--host", default=None, help="override CARLA host (default: config yaml)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="subset of scenario names; default = all 5")
    ap.add_argument("--out", default=str(ROOT / "results" / "eval_carla_metrics.csv"))
    ap.add_argument("--debug", action="store_true", help="print state trace on crash")
    args = ap.parse_args()

    cfg = CarlaConnectConfig.from_yaml(ROOT / "config" / "carla_params.yaml")
    if args.host: cfg.host = args.host
    if args.port: cfg.port = args.port
    if args.map: cfg.map = args.map

    scenarios = DEFAULT_CARLA_SCENARIOS
    if args.scenarios:
        scenarios = [s for s in DEFAULT_CARLA_SCENARIOS if s.name in set(args.scenarios)]

    vehicle = VehicleParams.from_yaml(ROOT / "config" / "vehicle_params.yaml")
    mpc_params = MPCParams.from_yaml(ROOT / "config" / "mpc_params.yaml")
    controllers = build_controllers(vehicle, mpc_params, ROOT / "models")

    print(f"connecting to CARLA at {cfg.host}:{cfg.port}, map={cfg.map}")
    rows = []
    total = len(controllers) * len(scenarios) * args.n_seeds
    done = 0
    with CarlaValidationEnv(cfg, mpc_horizon=mpc_params.Np, Ts=mpc_params.Ts) as env:
        for ctrl in controllers:
            for scen in scenarios:
                for s in range(args.n_seeds):
                    try:
                        m = rollout_carla(env, ctrl, scen, seed=2000+s,
                                         debug=args.debug)
                    except Exception as e:
                        print(f"  !! {ctrl.name} x {scen.name} seed={s}: {e}")
                        m = dict(scenario=scen.name, controller=ctrl.name, seed=s,
                                 steps=0, rmse_ey=float("nan"), mae_ey=float("nan"),
                                 max_ay_g=float("nan"), rms_ay_g=float("nan"),
                                 rms_ddelta=float("nan"), lane_violation_pct=float("nan"),
                                 crashed=1, mean_vx=float("nan"))
                    rows.append(m); done += 1
                print(f"  {ctrl.name:18} x {scen.name:18}  ok ({done}/{total})")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote -> {args.out}  ({len(df)} rows)")

    summ = (df.groupby(["scenario", "controller"])
              .agg(mean_rmse_ey=("rmse_ey", "mean"),
                   mean_max_ay=("max_ay_g", "mean"),
                   crash_pct=("crashed", lambda v: 100*np.mean(v)))
              .round(3))
    print("\n=== CARLA summary ===")
    print(summ.to_string())


if __name__ == "__main__":
    main()
