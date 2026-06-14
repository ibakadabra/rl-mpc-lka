"""Reward for the RL agent.

    r_t = -(w1*ey^2 + w2*e_psi^2 + w3*ddelta^2 + w4*ay^2) + r_alive

Control-systems reading
-----------------------
The quadratic penalty is literally an LQR-style running cost J = x^T Q_r x +
u^T R_r u, but evaluated on the REAL closed-loop trajectory instead of the
controller's internal model. Maximizing -J is the same as minimizing tracking
error, heading error, steering jerk, and lateral acceleration. So the agent is
rewarded for producing good *closed-loop* behavior, and it can only influence
that behavior through the MPC weights it chooses.

The r_alive term is a survival bonus: it makes "stay on the road for one more
step" intrinsically valuable, which (a) gives dense positive signal early in
training before the agent tracks well, and (b) makes lane departure (episode
termination, losing all future alive bonuses) genuinely costly — the classic
trick to discourage early-termination shortcuts.

Each weight trades off a competing objective:
    w1  tracking accuracy   (stay centered)
    w2  heading accuracy    (point down the lane)
    w3  smoothness          (don't saw the wheel -> comfort + actuator wear)
    w4  comfort             (limit lateral g)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RewardWeights:
    """Relative importance of each penalty term (from config/rl_params.yaml)."""

    w_ey: float = 1.0          # tracking accuracy
    w_epsi: float = 0.5        # heading accuracy
    w_delta_rate: float = 0.1  # smoothness (steering rate)
    w_ay: float = 0.05         # comfort (lateral acceleration)
    alive_bonus: float = 1.0   # survival bonus per step

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RewardWeights":
        with open(path, "r") as f:
            r = yaml.safe_load(f)["reward"]
        return cls(
            w_ey=r["w_ey"],
            w_epsi=r["w_epsi"],
            w_delta_rate=r["w_delta_rate"],
            w_ay=r["w_ay"],
            alive_bonus=r["alive_bonus"],
        )


def compute_reward(
    ey: float,
    e_psi: float,
    delta_rate: float,
    ay: float,
    weights: RewardWeights | None = None,
    return_components: bool = False,
):
    """One-step reward. `delta_rate` is Δδ between consecutive applied commands.

    With return_components=True also returns a dict of the individual terms,
    handy for TensorBoard / debugging which objective dominates.
    """
    w = weights or RewardWeights()
    pen_ey = w.w_ey * ey * ey
    pen_epsi = w.w_epsi * e_psi * e_psi
    pen_ddelta = w.w_delta_rate * delta_rate * delta_rate
    pen_ay = w.w_ay * ay * ay
    reward = -(pen_ey + pen_epsi + pen_ddelta + pen_ay) + w.alive_bonus

    if return_components:
        return reward, {
            "pen_ey": -pen_ey,
            "pen_epsi": -pen_epsi,
            "pen_ddelta": -pen_ddelta,
            "pen_ay": -pen_ay,
            "alive": w.alive_bonus,
        }
    return reward
