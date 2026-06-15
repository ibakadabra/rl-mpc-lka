"""Linear MPC for lateral control (condensed QP solved directly with OSQP).

Control-systems picture
-----------------------
MPC is a "rolling optimal regulator". At every 20 ms tick it:
  1. takes the current error state x0 (where am I vs the lane),
  2. uses the bicycle model to *predict* the next Np steps for any candidate
     steering sequence,
  3. picks the steering sequence that minimizes a cost J (stay on the lane,
     don't steer wildly) subject to hard limits (steering saturation/rate),
  4. applies ONLY the first steering command, then repeats next tick
     (receding horizon).

The Q,R weights are the "knobs" trading tracking accuracy (Q) against control
effort/smoothness (R). The thesis lets an RL agent turn these knobs online.

Why a hand-built condensed QP instead of cvxpy
----------------------------------------------
We eliminate the predicted states and write everything as a quadratic in the
input sequence U:

    X = Phi x0 + Gamma U   ->   J(U) = (1/2) U^T H U + f^T U
    H = 2 (Gamma^T Qbar Gamma + Rbar),   f = 2 Gamma^T Qbar Phi x0

This is exactly the spec's formulation (so it cross-checks against MATLAB), and
handing the dense H,f straight to OSQP is ~100x faster than re-canonicalizing a
cvxpy problem every tick -- essential because RL training runs millions of
solves. We also keep full control of solver tolerances, which a high penalty on
soft constraints otherwise wrecks.

Soft state constraints
----------------------
Steering magnitude and rate are HARD (physical actuator limits). Lane boundary
and lateral-acceleration limits are SOFT (slack variable + penalty): a hard
state constraint can make the QP infeasible exactly when an aggressive RL Q,R
drives a transient over the line. A soft constraint always returns a command
and only gives up the comfort limit by the minimum necessary amount, so the RL
loop never stalls on an infeasible solve.

Decision vector:  z = [ U (Nc) ;  s_lane (Np) ;  s_ay (Np) ]
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp
import yaml
from scipy.linalg import solve_discrete_are

from src.vehicle_model.bicycle_model import (
    VehicleParams,
    continuous_bicycle_ss,
    discretize_zoh,
)

N_STATES = 4  # [ey, e_psi, vy, psi_dot]


@dataclass
class MPCParams:
    """MPC horizon, timing, and constraint limits (from config/mpc_params.yaml)."""

    Np: int = 20            # prediction horizon
    Nc: int = 10            # control horizon (free moves; held constant after)
    Ts: float = 0.02        # sampling time [s]
    delta_max: float = 0.5  # steering saturation [rad]            (HARD)
    delta_rate_max: float = 0.1  # steering rate per step [rad]    (HARD)
    ey_max: float = 1.8     # lane half-width [m]                  (SOFT)
    ay_max: float = 3.0     # lateral accel comfort [m/s^2]        (SOFT)
    R_delta: float = 10.0   # steering-rate weight (comfort)       (FIXED)
    q3_fixed: float = 0.1   # weight on vy   (RL never touches)
    q4_fixed: float = 0.1   # weight on psi_dot
    soft_penalty: float = 1.0e3  # rho: price of violating a soft constraint
    vx_min: float = 1.0     # speed clamp for the LPV model

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MPCParams":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        mpc = cfg["mpc"]
        con = cfg["constraints"]
        fw = cfg["fixed_weights"]
        return cls(
            Np=mpc["prediction_horizon"],
            Nc=mpc["control_horizon"],
            Ts=mpc["sampling_time"],
            delta_max=con["delta_max"],
            delta_rate_max=con["delta_rate_max"],
            ey_max=con["ey_max"],
            ay_max=con["ay_max"],
            R_delta=fw["R_delta"],
            q3_fixed=fw["q3_fixed"],
            q4_fixed=fw["q4_fixed"],
        )


@dataclass
class MPCResult:
    """What the controller hands back each tick."""

    delta: float            # steering command actually applied (first move)
    u_sequence: np.ndarray  # full optimized steering plan over the horizon
    feasible: bool          # did the solver converge?
    soft_violation: float   # total slack used (0 => all comfort limits met)
    solve_time_ms: float
    cost: float = field(default=float("nan"))


def _coerce_Q(Q) -> np.ndarray:
    """Accept Q as a 4-vector (diagonal) or a 4x4 matrix; return 4x4."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        Q = np.diag(Q)
    return Q


class LinearMPC:
    """Receding-horizon lateral controller with RL-settable Q, R."""

    def __init__(self, vehicle: VehicleParams, params: MPCParams | None = None):
        self.veh = vehicle
        self.p = params or MPCParams()
        n, Np, Nc = N_STATES, self.p.Np, self.p.Nc

        # move-blocking lift S: maps Nc free moves -> Np-long input sequence
        S = np.zeros((Np, Nc))
        for k in range(Np):
            S[k, min(k, Nc - 1)] = 1.0
        self._S = S

        # difference operator on the Np-long input (for steering-rate term)
        D = np.eye(Np) - np.eye(Np, k=-1)
        self._DS = D @ S                       # (Np x Nc)
        self._e0 = np.zeros(Np); self._e0[0] = 1.0

        # row indices that pick ey and psi_dot out of the stacked state X
        self._idx_ey = np.array([k * n + 0 for k in range(Np)])
        self._idx_yaw = np.array([k * n + 3 for k in range(Np)])

        self.nz = Nc + 2 * Np                  # [U ; s_lane ; s_ay]
        self._last_z: np.ndarray | None = None  # warm-start memory

    # ----- condensed prediction matrices ------------------------------------
    def _prediction_matrices(
        self, A_d: np.ndarray, B_d: np.ndarray, E_d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phi, Gamma (input), Gamma_d (curvature disturbance) for the horizon.

        X = Phi x0 + Gamma U_full + Gamma_d Kappa, where Kappa is the previewed
        road curvature over the horizon. Each block row i (predicting x_{i+1}):
            Phi_i    = A^{i+1}
            Gamma_ij = A^{i-j} B   (discrete convolution of steering -> state)
            Gd_ij    = A^{i-j} E   (same convolution for the known disturbance)
        Including Gamma_d is the curvature *feed-forward*: the MPC plans steering
        to cancel curvature it already knows is coming, instead of only reacting
        after the error builds up (which at highway speed integrates fast).
        """
        n, Np = N_STATES, self.p.Np
        Phi = np.zeros((Np * n, n))
        Gamma = np.zeros((Np * n, Np))
        Gamma_d = np.zeros((Np * n, Np))
        Apow = [np.eye(n)]
        for _ in range(Np):
            Apow.append(Apow[-1] @ A_d)        # Apow[k] = A^k
        for i in range(Np):
            Phi[i * n:(i + 1) * n, :] = Apow[i + 1]
            for j in range(i + 1):
                AB = Apow[i - j]
                Gamma[i * n:(i + 1) * n, j:j + 1] = AB @ B_d
                Gamma_d[i * n:(i + 1) * n, j:j + 1] = AB @ E_d
        return Phi, Gamma, Gamma_d

    def _terminal_cost(self, A_d, B_d, Q, R) -> np.ndarray:
        """Terminal weight Qf = infinite-horizon LQR cost-to-go (DARE solution).

        Putting the LQR cost-to-go at the end of the horizon is the textbook way
        to give a finite-horizon MPC the stability of the infinite-horizon
        problem. Fall back to Q if the DARE can't be solved.
        """
        try:
            Qf = solve_discrete_are(A_d, B_d, Q, R)
            return 0.5 * (Qf + Qf.T)
        except Exception:
            return Q.copy()

    # ----- QP assembly ------------------------------------------------------
    def _build_qp(self, x0, vx, Q, R, dp, kappa):
        """Assemble the condensed QP: returns (P, q, A, l, u) for OSQP.

        OSQP solves  min (1/2) z^T P z + q^T z  s.t.  l <= A z <= u.
        `kappa` is the previewed curvature over the horizon (length Np).
        """
        p, n, Np, Nc = self.p, N_STATES, self.p.Np, self.p.Nc

        # LPV model + curvature disturbance E_c, both discretized together.
        A_c, B_c = continuous_bicycle_ss(vx, self.veh, vx_min=p.vx_min)
        E_c = np.array([[0.0], [-vx], [0.0], [0.0]])   # curvature acts on e_psi_dot
        A_d, BE_d = discretize_zoh(A_c, np.hstack([B_c, E_c]), p.Ts)
        B_d, E_d = BE_d[:, :1], BE_d[:, 1:2]

        Phi, Gamma, Gamma_d = self._prediction_matrices(A_d, B_d, E_d)
        Qf = self._terminal_cost(A_d, B_d, Q, np.array([[R]]))
        Gb = Gamma @ self._S                    # (Np*n x Nc) with move-blocking

        # Free response = unforced state + known curvature disturbance (feed-fwd).
        free = Phi @ x0 + Gamma_d @ kappa       # (Np*n,)

        # Stacked state weight Qbar = blkdiag(Q,...,Q, Qf).
        Qbar = np.zeros((Np * n, Np * n))
        for k in range(Np - 1):
            Qbar[k * n:(k + 1) * n, k * n:(k + 1) * n] = Q
        Qbar[(Np - 1) * n:, (Np - 1) * n:] = Qf

        # Cost in U:  U^T M U + 2 c^T U  (state + effort + rate terms).
        M = (Gb.T @ Qbar @ Gb) + R * (self._S.T @ self._S) \
            + p.R_delta * (self._DS.T @ self._DS)
        c = Gb.T @ Qbar @ free - p.R_delta * dp * (self._DS.T @ self._e0)

        # OSQP form: P_uu = 2M, q_u = 2c.
        P = np.zeros((self.nz, self.nz))
        P[:Nc, :Nc] = 2.0 * M + 1e-8 * np.eye(Nc)
        # Regularize slacks so P is strictly PD (OSQP convergence)
        for i in range(Nc, self.nz):
            P[i, i] = 1e-6
        q = np.zeros(self.nz)
        q[:Nc] = 2.0 * c
        q[Nc:] = p.soft_penalty                       # linear price on slacks

        # Constraints  l <= A z <= u.
        Gb_ey, Gb_yaw = Gb[self._idx_ey], Gb[self._idx_yaw]
        ey0 = free[self._idx_ey]
        yaw0 = free[self._idx_yaw]
        INF = np.inf
        rows, l, u = [], [], []

        def add(block_u, block_sl, block_sa, lo, hi):
            r = np.zeros((block_u.shape[0], self.nz))
            r[:, :Nc] = block_u
            if block_sl is not None:
                r[:, Nc:Nc + Np] = block_sl
            if block_sa is not None:
                r[:, Nc + Np:] = block_sa
            rows.append(r); l.append(lo); u.append(hi)

        I_np = np.eye(Np)
        # (1) HARD steering box on U
        add(np.eye(Nc), None, None, np.full(Nc, -p.delta_max), np.full(Nc, p.delta_max))
        # (2) HARD steering rate: DS U in [-dmax+e0*dp, dmax+e0*dp]
        add(self._DS, None, None,
            -p.delta_rate_max + self._e0 * dp, p.delta_rate_max + self._e0 * dp)
        # (3,4) SOFT lane: |ey| <= ey_max + s_lane
        add(Gb_ey, -I_np, None, np.full(Np, -INF), p.ey_max - ey0)
        add(-Gb_ey, -I_np, None, np.full(Np, -INF), p.ey_max + ey0)
        # (5,6) SOFT lateral accel ~ vx*yaw_rate: |ay| <= ay_max + s_ay
        add(vx * Gb_yaw, None, -I_np, np.full(Np, -INF), p.ay_max - vx * yaw0)
        add(-vx * Gb_yaw, None, -I_np, np.full(Np, -INF), p.ay_max + vx * yaw0)
        # (7) slacks >= 0
        sl = np.zeros((2 * Np, self.nz)); sl[:, Nc:] = np.eye(2 * Np)
        rows.append(sl); l.append(np.zeros(2 * Np)); u.append(np.full(2 * Np, INF))

        A = np.vstack(rows)
        l = np.concatenate(l); u = np.concatenate(u)
        return P, q, A, l, u

    # ----- main solve -------------------------------------------------------
    def solve(self, x0, vx: float, Q, R, delta_prev: float = 0.0,
              kappa_preview=0.0) -> MPCResult:
        """Solve one MPC step.

        x0            current error state [ey, e_psi, vy, psi_dot]
        vx            current longitudinal speed (sets the LPV model)
        Q, R          cost weights chosen by the RL agent this 0.5 s window
        delta_prev    last applied steering (steering-rate term/constraint)
        kappa_preview road curvature over the horizon: scalar (held constant) or
                      array of length Np. Enables curvature feed-forward.
        """
        p, Np, Nc = self.p, self.p.Np, self.p.Nc
        x0 = np.asarray(x0, dtype=float).ravel()
        Q = _coerce_Q(Q)
        R = float(np.asarray(R).ravel()[0])
        dp = float(delta_prev)
        kappa = np.asarray(kappa_preview, dtype=float).ravel()
        if kappa.size == 1:
            kappa = np.full(Np, float(kappa[0]))
        elif kappa.size < Np:
            kappa = np.concatenate([kappa, np.full(Np - kappa.size, kappa[-1])])
        else:
            kappa = kappa[:Np]

        P, q, A, l, u = self._build_qp(x0, vx, Q, R, dp, kappa)

        # Solve with OSQP (upper-triangular P, warm-started from last z).
        P_csc = sp.triu(sp.csc_matrix(P), format="csc")
        A_csc = sp.csc_matrix(A)
        m = osqp.OSQP()
        m.setup(P=P_csc, q=q, A=A_csc, l=l, u=u, verbose=False,
                eps_abs=1e-4, eps_rel=1e-4, max_iter=4000, polishing=True,
                adaptive_rho=True, warm_starting=True, scaling=20)
        if self._last_z is not None and self._last_z.shape == (self.nz,):
            m.warm_start(x=self._last_z)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            r = m.solve()
        solve_ms = (time.perf_counter() - t0) * 1e3

        status = r.info.status
        feasible = status in ("solved", "solved inaccurate")
        if not feasible:
            print(f"  [MPC] status={status} iter={r.info.iter} "
                  f"ey={x0[0]:.3f} epsi={x0[1]:.4f} vx={vx:.1f} dp={dp:.4f}")
        if feasible and r.x is not None and np.all(np.isfinite(r.x)):
            z = np.asarray(r.x).ravel()
            self._last_z = z
            U = z[:Nc]
            u_seq = (self._S @ U).ravel()
            delta = float(np.clip(u_seq[0], -p.delta_max, p.delta_max))
            slack = float(np.sum(z[Nc:]))
            cost_val = float(r.info.obj_val)
        else:
            # Safety net: hold previous steering, report infeasible.
            u_seq = np.full(Np, dp)
            delta = float(dp)
            slack = float("nan")
            cost_val = float("nan")

        return MPCResult(
            delta=delta, u_sequence=u_seq, feasible=feasible,
            soft_violation=slack, solve_time_ms=solve_ms, cost=cost_val,
        )
