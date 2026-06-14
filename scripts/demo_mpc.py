"""Phase-1 sanity demo: close the loop (MPC + bicycle plant) and watch ey -> 0.

We run a lateral step response (start 0.5 m off-center) for three different
ey-weights q1. This is exactly the knob the RL agent will learn to turn:
  - small q1  -> MPC tolerates error, steers gently (sluggish recovery)
  - large q1  -> MPC punishes error hard, steers aggressively (fast recovery)

Both the prediction model and the "real" plant are the same discrete bicycle
model here (a perfect-model assumption); Phase 2+ adds mismatch and disturbance.

Run:  python scripts/demo_mpc.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save to file, no display needed
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mpc.linear_mpc import LinearMPC, MPCParams
from src.vehicle_model.bicycle_model import VehicleParams, get_discrete_bicycle

ROOT = Path(__file__).resolve().parents[1]


def run_step_response(q1: float, vx: float = 20.0, n_steps: int = 250):
    """Closed-loop step response for a given ey-weight q1."""
    veh = VehicleParams.from_yaml(ROOT / "config" / "vehicle_params.yaml")
    mpc = LinearMPC(veh, MPCParams.from_yaml(ROOT / "config" / "mpc_params.yaml"))
    A_d, B_d = get_discrete_bicycle(vx, veh, mpc.p.Ts)

    x = np.array([0.5, 0.0, 0.0, 0.0])  # 0.5 m lateral error, everything else 0
    delta_prev = 0.0
    Q = np.array([q1, 10.0, mpc.p.q3_fixed, mpc.p.q4_fixed])
    R = 1.0

    t, ey, delta, solve_ms = [], [], [], []
    for k in range(n_steps):
        res = mpc.solve(x, vx=vx, Q=Q, R=R, delta_prev=delta_prev)
        x = A_d @ x + (B_d @ np.array([res.delta])).ravel()
        delta_prev = res.delta
        t.append(k * mpc.p.Ts)
        ey.append(x[0])
        delta.append(res.delta)
        solve_ms.append(res.solve_time_ms)
    return np.array(t), np.array(ey), np.array(delta), np.array(solve_ms)


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    all_ms = []
    for q1, color in [(1.0, "tab:blue"), (100.0, "tab:orange"), (1000.0, "tab:red")]:
        t, ey, delta, ms = run_step_response(q1)
        all_ms.append(ms)
        ax1.plot(t, ey, color=color, label=f"q1 = {q1:.0f}")
        ax2.plot(t, delta, color=color, label=f"q1 = {q1:.0f}")

    ax1.axhline(0.0, color="k", lw=0.6, ls=":")
    ax1.set_ylabel("lateral error  ey  [m]")
    ax1.set_title("Phase-1 closed loop: MPC + bicycle model — step response vs ey-weight q1")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.axhline(0.5, color="k", lw=0.4, ls=":"); ax2.axhline(-0.5, color="k", lw=0.4, ls=":")
    ax2.set_ylabel("steering  delta  [rad]")
    ax2.set_xlabel("time [s]")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "phase1_mpc_demo.png"
    fig.savefig(out, dpi=120)
    print(f"saved figure -> {out}")
    print(f"median MPC solve time: {np.median(np.concatenate(all_ms)):.2f} ms/step")


if __name__ == "__main__":
    main()
