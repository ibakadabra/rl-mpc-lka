# RL-Tuned MPC for Autonomous Vehicle Lateral Control

MSc Thesis Project — Hacettepe University, System Dynamics and Control

## Overview

A Soft Actor-Critic (SAC) reinforcement learning agent adaptively tunes the weight matrices (Q, R) of a linear Model Predictive Controller for vehicle lane keeping. MPC guarantees constraint satisfaction; RL optimizes performance across varying driving conditions.

## Architecture

```
Driving Context [vx, κ, ey, eψ, ...] → SAC Policy → [Q, R] → MPC (QP) → δ steering
```

- **RL at 2 Hz:** selects MPC cost weights based on driving situation
- **MPC at 50 Hz:** solves constrained QP for optimal steering

## Tech Stack

- **Simulation:** CARLA + ROS2
- **MPC:** cvxpy + OSQP
- **RL:** Stable-Baselines3 (prototype), CleanRL (thesis)
- **Vehicle Model:** Linear bicycle model (LPV, speed-dependent)

## Quick Start

```bash
# Install
pip install -e .

# Train (internal simulation, no CARLA needed)
python scripts/train.py --env internal --timesteps 200000

# Train (CARLA)
./scripts/carla_setup.sh
python scripts/train.py --env carla --timesteps 500000

# Evaluate
python scripts/evaluate.py --model models/sac_best.zip --scenarios all
```

## Project Structure

See [CLAUDE_CODE_PROMPT.md](CLAUDE_CODE_PROMPT.md) for full project specification.

## License

MIT
