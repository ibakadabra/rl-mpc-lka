"""Single-file SAC (CleanRL-style) — the thesis implementation.

Why a from-scratch version when SB3 already works?
For the thesis we must *demonstrate understanding of the algorithm internals* and
be able to modify them (e.g. custom critics, reward shaping experiments). This
file lays SAC bare: the twin critics, the squashed-Gaussian actor, the entropy
temperature auto-tuning, and the soft target updates — nothing hidden in a
library.

Control-systems map of the moving parts:
  * QNetwork        -> learned cost-to-go surface Q(s,a) (data-driven, nonlinear)
  * Actor           -> the gain-scheduling policy: state -> (mean, std) of action
  * target networks -> slowly-tracking reference models; tau is the filter pole
                       of a first-order low-pass that smooths the bootstrap target
  * alpha (entropy) -> automatically-tuned exploration "dither" amplitude
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.environment import CarlaMPCEnv, EnvConfig

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# ---------------------------------------------------------------------------
@dataclass
class SACConfig:
    total_timesteps: int = 200_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005           # soft target-update rate (target LPF pole)
    lr: float = 3e-4
    hidden: int = 256
    learning_starts: int = 1000
    seed: int = 0


class ReplayBuffer:
    """Fixed-size ring buffer of transitions (s, a, r, s', done)."""

    def __init__(self, size, obs_dim, act_dim, device):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros((size, 1), np.float32)
        self.max, self.ptr, self.n = size, 0, 0
        self.device = device

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.max
        self.n = min(self.n + 1, self.max)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


class RunningNorm:
    """Online observation whitening (Welford mean/var) — input conditioning."""

    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x):
        self.count += 1
        d = x - self.mean
        self.mean += d / self.count
        self.var += (d * (x - self.mean) - self.var) / self.count

    def __call__(self, x):
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


class QNetwork(nn.Module):
    """Critic: estimates the value (negative cost-to-go) of taking a in s."""

    def __init__(self, obs_dim, act_dim, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class Actor(nn.Module):
    """Squashed-Gaussian policy: outputs tanh(N(mu, sigma)) in [-1, 1]^act."""

    def __init__(self, obs_dim, act_dim, h):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(),
                                   nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, act_dim)
        self.log_std = nn.Linear(h, act_dim)

    def forward(self, s):
        x = self.trunk(s)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, s):
        """Reparameterized sample + tanh-corrected log-prob (for entropy term)."""
        mu, log_std = self(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()                 # reparameterization trick
        a = torch.tanh(x)                    # squash into [-1, 1]
        logp = normal.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True)


def train_cleanrl(cfg: SACConfig | None = None, env_config: EnvConfig | None = None,
                  out_dir: Path | None = None):
    cfg = cfg or SACConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    env = CarlaMPCEnv(config=env_config or EnvConfig(kappa_max=0.05, randomize=True),
                      seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    norm = RunningNorm(obs_dim)

    actor = Actor(obs_dim, act_dim, cfg.hidden).to(device)
    q1, q2 = QNetwork(obs_dim, act_dim, cfg.hidden).to(device), QNetwork(obs_dim, act_dim, cfg.hidden).to(device)
    q1t, q2t = QNetwork(obs_dim, act_dim, cfg.hidden).to(device), QNetwork(obs_dim, act_dim, cfg.hidden).to(device)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())

    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr)
    pi_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)

    # automatic entropy tuning: target entropy = -|A| (SAC heuristic)
    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    a_opt = torch.optim.Adam([log_alpha], lr=cfg.lr)

    buf = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim, device)
    o, _ = env.reset(seed=cfg.seed)
    norm.update(o); o_n = norm(o)
    ep_ret, ep_rets = 0.0, []

    for step in range(cfg.total_timesteps):
        if step < cfg.learning_starts:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor.sample(torch.as_tensor(o_n, device=device).unsqueeze(0))
                a = a.cpu().numpy()[0]

        o2, r, term, trunc, _ = env.step(a)
        norm.update(o2); o2_n = norm(o2)
        buf.add(o_n, a, r, o2_n, float(term))
        o_n = o2_n; ep_ret += r

        if term or trunc:
            ep_rets.append(ep_ret); ep_ret = 0.0
            o, _ = env.reset(); norm.update(o); o_n = norm(o)
            if len(ep_rets) % 10 == 0:
                print(f"step {step}  mean_ep_ret(last10) {np.mean(ep_rets[-10:]):.1f}")

        # ----- SAC gradient step -----
        if buf.n >= cfg.batch_size and step >= cfg.learning_starts:
            s, ac, rew, s2, d = buf.sample(cfg.batch_size)
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                a2, logp2 = actor.sample(s2)
                q_next = torch.min(q1t(s2, a2), q2t(s2, a2)) - alpha * logp2
                target = rew + cfg.gamma * (1 - d) * q_next   # entropy-augmented TD

            q1_loss = F.mse_loss(q1(s, ac), target)
            q2_loss = F.mse_loss(q2(s, ac), target)
            q_opt.zero_grad(); (q1_loss + q2_loss).backward(); q_opt.step()

            ap, logp = actor.sample(s)
            q_pi = torch.min(q1(s, ap), q2(s, ap))
            pi_loss = (alpha * logp - q_pi).mean()            # climb the value surface
            pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()

            alpha_loss = -(log_alpha.exp() * (logp + target_entropy).detach()).mean()
            a_opt.zero_grad(); alpha_loss.backward(); a_opt.step()

            # soft target update: theta_target <- (1-tau) theta_target + tau theta
            for p, pt in zip(q1.parameters(), q1t.parameters()):
                pt.data.mul_(1 - cfg.tau); pt.data.add_(cfg.tau * p.data)
            for p, pt in zip(q2.parameters(), q2t.parameters()):
                pt.data.mul_(1 - cfg.tau); pt.data.add_(cfg.tau * p.data)

    out_dir = out_dir or (Path(__file__).resolve().parents[2] / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), out_dir / "cleanrl_actor.pt")
    print(f"saved actor -> {out_dir / 'cleanrl_actor.pt'}")
    return actor, ep_rets


if __name__ == "__main__":
    train_cleanrl(SACConfig(total_timesteps=20000))
