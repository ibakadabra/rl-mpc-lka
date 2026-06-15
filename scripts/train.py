"""Main training entry point.

Examples
--------
    # Phase 1-3: train with the internal bicycle simulation (no CARLA)
    python scripts/train.py --env internal --algo sac --timesteps 200000

    # Phase 4: train inside CARLA (not yet wired up)
    python scripts/train.py --env carla --algo sac --timesteps 500000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.environment import EnvConfig
from src.rl.train_sb3 import plot_learning_curve, train


def main():
    ap = argparse.ArgumentParser(description="Train the RL-tuned MPC agent.")
    ap.add_argument("--env", choices=["internal", "carla"], default="internal")
    ap.add_argument("--algo", choices=["sac"], default="sac")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa-max", type=float, default=0.05)
    ap.add_argument("--no-randomize", action="store_true",
                    help="disable domain randomization of the plant")
    ap.add_argument("--hard", action="store_true",
                    help="widen domain randomization (low-mu + within-episode grip "
                         "drop) so the agent learns the regimes where a vx-only "
                         "schedule fails — pair with --timesteps 500000")
    ap.add_argument("--out-subdir", default=None,
                    help="save under models/<subdir>/ instead of models/ "
                         "(keeps the existing demo model for before/after compare)")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="parallel rollout workers; the MPC QP is CPU-bound so "
                         "this scales throughput ~N-fold (try 8 on an 8-core box)")
    ap.add_argument("--plot", action="store_true", help="save learning curve when done")
    args = ap.parse_args()

    if args.env == "carla":
        raise NotImplementedError(
            "CARLA backend is Phase 4. Use --env internal for now.")

    if args.hard:
        cfg = EnvConfig(
            kappa_max=args.kappa_max, randomize=True,
            cf_range=(0.5, 1.3), cr_range=(0.5, 1.3),   # stronger + asymmetric low-mu
            mu_step_prob=0.3,                            # 30% episodes get a grip drop
        )
    else:
        cfg = EnvConfig(kappa_max=args.kappa_max, randomize=not args.no_randomize)

    from pathlib import Path as _Path
    out_dir = None
    if args.out_subdir:
        out_dir = _Path(__file__).resolve().parents[1] / "models" / args.out_subdir
    train(total_timesteps=args.timesteps, env_config=cfg, seed=args.seed,
          out_dir=out_dir, n_envs=args.n_envs)
    if args.plot:
        plot_learning_curve()


if __name__ == "__main__":
    main()
