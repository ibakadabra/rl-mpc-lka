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
    ap.add_argument("--plot", action="store_true", help="save learning curve when done")
    args = ap.parse_args()

    if args.env == "carla":
        raise NotImplementedError(
            "CARLA backend is Phase 4. Use --env internal for now.")

    cfg = EnvConfig(kappa_max=args.kappa_max, randomize=not args.no_randomize)
    train(total_timesteps=args.timesteps, env_config=cfg, seed=args.seed)
    if args.plot:
        plot_learning_curve()


if __name__ == "__main__":
    main()
