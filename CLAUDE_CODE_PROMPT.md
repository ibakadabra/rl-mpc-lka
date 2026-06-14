# RL-Tuned MPC for Autonomous Vehicle Lateral Control

## Project Overview

This is an MSc thesis project: **Reinforcement Learning-based Adaptive Model Predictive Control for Autonomous Vehicle Lane Keeping**.

The core idea: a SAC (Soft Actor-Critic) RL agent **tunes the MPC weight matrices Q and R online** based on driving context (speed, curvature, tracking error). MPC retains constraint satisfaction and stability guarantees; RL optimizes performance across diverse driving conditions.

**Tech stack:** Python, CARLA simulator, ROS2, Gymnasium, Stable-Baselines3 (prototype) → CleanRL (thesis), cvxpy + OSQP, NumPy/SciPy, matplotlib.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CARLA + ROS2                       │
│                                                      │
│  Sensors → Perception → Reference Trajectory         │
│       ↓              ↓              ↓                │
│  ┌────────────────────────────────────────────┐      │
│  │         RL-MPC Controller                  │      │
│  │                                            │      │
│  │  Observation     RL Policy     MPC         │      │
│  │  [vx,κ,ey,eψ] → π_θ(s) → [Q,R] → QP → δ │      │
│  │       ↑                           ↓        │      │
│  │       └──── state feedback ───────┘        │      │
│  └────────────────────────────────────────────┘      │
│                      ↓                               │
│              Vehicle Actuator (δ)                    │
└─────────────────────────────────────────────────────┘
```

### Hierarchical timing
- **RL operates at 2 Hz** (every 0.5s = every 10 MPC steps): selects Q, R
- **MPC operates at 50 Hz** (every 0.02s): solves QP with current Q, R, applies steering

---

## Vehicle Model (MPC Prediction Model)

### Lateral error dynamics — Linear bicycle model

**State vector:**
```
x = [ey, eψ, ėy, ėψ]^T ∈ ℝ⁴
```
- ey: lateral error from lane center (m)
- eψ: heading error (rad)
- ėy: lateral error rate (m/s)
- ėψ: heading error rate (rad/s)

**Input:** u = δ (front steering angle, rad)

**Continuous state-space (speed-dependent, LPV):**

```
A_c(vx) = [
    [0,    vx,   1,    0                              ],
    [0,    0,    0,    1                              ],
    [0,    0,    -(Cf+Cr)/(m*vx),    (lr*Cr-lf*Cf)/(m*vx) - vx],
    [0,    0,    (lr*Cr-lf*Cf)/(Iz*vx),  -(lf²*Cf+lr²*Cr)/(Iz*vx)]
]

B_c = [
    [0      ],
    [0      ],
    [Cf/m   ],
    [lf*Cf/Iz]
]
```

**Discretization:** ZOH (Zero-Order Hold) via `scipy.signal.cont2discrete`, Ts = 0.02s.

**A and B are updated every MPC call** based on current vx measurement.

### Vehicle parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Mass | m | 1500 | kg |
| Yaw inertia | Iz | 2500 | kg·m² |
| CG to front axle | lf | 1.2 | m |
| CG to rear axle | lr | 1.4 | m |
| Front cornering stiffness | Cf | 80000 | N/rad |
| Rear cornering stiffness | Cr | 85000 | N/rad |

**Note:** These are nominal values. For robustness testing, randomize: Cf,Cr ±30%, m ±10%, actuator delay 0-50ms uniform.

---

## MPC Formulation

### Cost function

```
J = Σ(k=0..Np-1) [x_k^T Q x_k + u_k^T R u_k + Δu_k^T R_Δ Δu_k] + x_Np^T Qf x_Np
```

- **Q = diag(q1, q2, q3, q4):** state weights — **q1, q2 set by RL; q3=0.1, q4=0.1 fixed**
- **R = [[r1]]:** steering effort weight — **set by RL**
- **R_Δ = [[10.0]]:** steering rate weight — **fixed** (comfort)
- **Qf:** terminal cost from DARE solution

### Constraints

| Constraint | Expression | Value |
|-----------|-----------|-------|
| Steering saturation | \|δ_k\| ≤ δ_max | 0.5 rad |
| Steering rate | \|Δδ_k\| ≤ Δδ_max | 0.1 rad/step |
| Lane boundary | \|ey_k\| ≤ ey_max | 1.8 m |
| Lateral acceleration (comfort) | \|ay_k\| ≤ ay_max | 3.0 m/s² |

### QP formulation

Standard stacking: X = Φ x₀ + Γ U

```
min_U  (1/2) U^T H U + f^T U
s.t.   A_ineq U ≤ b_ineq

H = 2(Γ^T Q̄ Γ + R̄)
f = 2 Γ^T Q̄ Φ x₀
```

**Solver:** OSQP via cvxpy, warm-start enabled.

### MPC parameters

| Parameter | Value |
|-----------|-------|
| Prediction horizon Np | 20 |
| Control horizon Nc | 10 |
| Sampling time Ts | 0.02 s |

---

## RL Formulation

### MDP Definition

**Observation (RL state) s_t ∈ ℝ⁷:**

```
s_t = [vx, κ, ey, eψ, ψ̇, ay, δ_prev]
```

| Element | Description | Range |
|---------|-------------|-------|
| vx | Longitudinal velocity | [0, 50] m/s |
| κ | Road curvature | [-0.2, 0.2] 1/m |
| ey | Lateral error | [-3.0, 3.0] m |
| eψ | Heading error | [-1.0, 1.0] rad |
| ψ̇ | Yaw rate | [-1.0, 1.0] rad/s |
| ay | Lateral acceleration | [-10, 10] m/s² |
| δ_prev | Previous steering | [-0.5, 0.5] rad |

**Action (RL output) a_t ∈ [-1, 1]³:**

```
a_t = [α1, α2, α3]  →  mapped to Q, R via log scale:

q1 = 10^(α1_normalized * 3)      # ey weight:  1 ~ 1000
q2 = 10^(α2_normalized * 3)      # eψ weight:  1 ~ 1000
r1 = 10^(α3_normalized * 2 - 1)  # δ weight:   0.1 ~ 10

Q = diag(q1, q2, 0.1, 0.1)
R = [[r1]]
```

**Log scale is critical** — linear mapping cannot distinguish between Q=1 and Q=10 as effectively.

**Reward r_t:**

```
r_t = -(w1 * ey² + w2 * eψ² + w3 * Δδ² + w4 * ay²) + r_alive

w1 = 1.0   (tracking accuracy)
w2 = 0.5   (heading accuracy)
w3 = 0.1   (smoothness)
w4 = 0.05  (comfort)
r_alive = 1.0  (survival bonus)
```

**Termination conditions:**
- |ey| > 2.5 m (lane departure)
- Collision detected (CARLA collision sensor)
- Episode step count > 2000

### Algorithm: SAC (Soft Actor-Critic)

**Why SAC:**
- Continuous action space (Q, R are continuous)
- Off-policy → sample efficient (CARLA simulation is slow)
- Entropy regularization → automatic exploration
- Well-suited for 3D continuous action

**SAC Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Batch size | 256 |
| Replay buffer size | 1,000,000 |
| Discount γ | 0.99 |
| Soft update τ | 0.005 |
| Hidden layers | [256, 256] |
| Activation | ReLU |
| Auto-tune entropy | True |

### Training pipeline

```
1. Spawn vehicle at random CARLA location
2. Every RL step (0.5s):
   a) Observe s_t = [vx, κ, ey, eψ, ψ̇, ay, δ_prev]
   b) a_t = SAC.policy(s_t)
   c) Q, R = log_scale_map(a_t)
   d) Run 10 MPC steps (0.02s each) with this Q, R
   e) Accumulate reward over 10 steps
   f) Store (s, a, r, s') in replay buffer
   g) SAC.update(random_batch from buffer)
3. Episode ends on termination/truncation → new episode
4. Every 50 episodes → evaluate on 5 fixed scenarios
```

**Domain randomization during training:**
- Cf, Cr: ±30% uniform
- m: ±10% uniform
- Actuator delay: 0-50ms uniform
- Weather: random (clear, rain, fog)
- Spawn location: random

**Estimated training:** ~500-1000 episodes, ~8-15 hours on GPU machine.

---

## Gymnasium Environment

The Gymnasium wrapper connects RL ↔ MPC ↔ CARLA:

```
env.reset()  →  Spawn vehicle, return initial observation
env.step(action)  →  action → Q,R → 10x MPC → reward, next obs
```

**1 RL step = 10 MPC steps = 0.5 seconds of simulation**

**Key design decisions:**
- MPC infeasibility → fallback to nominal Q, R (safety net)
- Q, R rate limiting: |Q_{k+1} - Q_k| < ε to prevent erratic behavior
- observation normalization: running mean/std via VecNormalize (SB3)

### Development phases:
1. **Phase 1 (no CARLA):** Use internal bicycle model simulation for plant dynamics. Validate RL+MPC pipeline works, reward converges, Q/R values are sensible.
2. **Phase 2 (CARLA):** Replace internal simulation with CARLA API calls. Same Gymnasium interface.
3. **Phase 3 (ROS2):** Wrap as ROS2 node for real-time architecture.

---

## ROS2 Node Architecture

```
/carla_bridge              ← CARLA-ROS2 bridge (existing package)
    pub: /vehicle/state         [vx, vy, ψ, ψ̇, ax, ay]
    pub: /camera/image
    pub: /lidar/points
    sub: /vehicle/control       [δ, throttle, brake]

/perception_node           ← Lane detection + object tracking
    sub: /camera/image
    sub: /lidar/points
    pub: /lane/centerline       [polynomial coefficients]
    pub: /objects/tracked        [x, y, vx, vy, class]

/rl_mpc_node               ← Main controller node
    sub: /vehicle/state
    sub: /lane/centerline
    pub: /vehicle/control

    Internal:
        rl_agent.get_action(obs)  → Q, R   (2 Hz)
        mpc_solver.solve(x, Q, R) → δ      (50 Hz)
```

**Perception simplification:** For thesis scope, use CARLA ground-truth API for lane centerline and object positions (perfect perception assumption). Inject Gaussian noise + outliers for robustness testing.

---

## Baselines (Required for thesis)

| # | Baseline | Description |
|---|----------|-------------|
| 1 | Fixed-Q,R MPC | Manually tuned constant weights |
| 2 | Gain-scheduled MPC | vx-based lookup table for Q, R |
| 3 | Pure RL (SAC) | Direct δ output, no MPC |
| 4 | Stanley controller | Classic geometric controller |
| 5 | Pure Pursuit | Classic geometric controller |

### Evaluation metrics

| Metric | What it measures |
|--------|-----------------|
| MAE(ey), RMSE(ey) | Tracking accuracy |
| max\|ay\|, RMS(ay) | Comfort |
| RMS(Δδ) | Smoothness |
| MPC solve time (ms) | Real-time feasibility |
| Lane violation % | Safety |
| Episode reward (cumulative) | Overall RL performance |

### Evaluation scenarios

| Scenario | vx | Road type |
|----------|-----|-----------|
| Highway straight | 100-120 km/h | κ ≈ 0 |
| Highway gentle curve | 80-100 km/h | κ = 0.01-0.03 |
| Urban sharp turn | 30-50 km/h | κ = 0.05-0.15 |
| Mixed route | varying | combination |
| Low-μ (wet road) | 60 km/h | reduced Cf, Cr by 40% |

---

## Project Structure

```
rl-mpc-autonomous-vehicle/
├── README.md
├── CLAUDE_CODE_PROMPT.md          ← This file
├── requirements.txt
├── setup.py
├── .gitignore
│
├── config/
│   ├── vehicle_params.yaml        ← m, Iz, lf, lr, Cf, Cr
│   ├── mpc_params.yaml            ← Np, Nc, Ts, constraints
│   ├── rl_params.yaml             ← SAC hyperparams, reward weights
│   └── carla_params.yaml          ← server, map, spawn points
│
├── src/
│   ├── __init__.py
│   ├── vehicle_model/
│   │   ├── __init__.py
│   │   ├── bicycle_model.py       ← get_bicycle_ss(), discretize()
│   │   └── test_bicycle_model.py
│   │
│   ├── mpc/
│   │   ├── __init__.py
│   │   ├── linear_mpc.py          ← LinearMPC class (cvxpy + OSQP)
│   │   └── test_mpc.py
│   │
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── environment.py         ← CarlaMPCEnv (Gymnasium wrapper)
│   │   ├── weight_mapper.py       ← rl_action_to_weights (log scale)
│   │   ├── reward.py              ← reward computation
│   │   ├── train_sb3.py           ← SB3 SAC training script
│   │   └── train_cleanrl.py       ← CleanRL SAC (thesis version)
│   │
│   ├── carla_interface/
│   │   ├── __init__.py
│   │   ├── carla_env.py           ← CARLA API wrapper
│   │   ├── sensors.py             ← sensor setup & data extraction
│   │   └── waypoint_utils.py      ← lane centerline, κ extraction
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── fixed_mpc.py           ← Fixed Q,R MPC
│   │   ├── gain_scheduled_mpc.py  ← vx-based Q,R lookup
│   │   ├── pure_rl.py             ← SAC direct control (no MPC)
│   │   ├── stanley.py             ← Stanley controller
│   │   └── pure_pursuit.py        ← Pure Pursuit controller
│   │
│   └── ros2_nodes/
│       ├── __init__.py
│       └── rl_mpc_node.py         ← ROS2 node (Phase 3)
│
├── scripts/
│   ├── train.py                   ← Main training entry point
│   ├── evaluate.py                ← Run evaluation scenarios
│   ├── plot_results.py            ← Generate thesis figures
│   └── carla_setup.sh             ← CARLA server launch script
│
├── tests/
│   ├── test_bicycle_model.py
│   ├── test_mpc_solver.py
│   ├── test_environment.py
│   ├── test_weight_mapper.py
│   └── test_reward.py
│
├── notebooks/
│   ├── 01_model_validation.ipynb  ← Compare Python vs MATLAB model
│   ├── 02_mpc_tuning.ipynb        ← Manual MPC tuning experiments
│   ├── 03_training_analysis.ipynb ← RL training curves, Q/R evolution
│   └── 04_results_comparison.ipynb← Baseline comparison plots
│
├── matlab_validation/
│   ├── bicycle_model_validation.m ← MATLAB cross-check
│   ├── dare_terminal_cost.m       ← Qf computation
│   └── mpc_qp_validation.m       ← QP solution cross-check
│
├── logs/                          ← TensorBoard / W&B logs
├── models/                        ← Saved RL models
├── results/                       ← Evaluation CSVs, plots
└── docs/
    └── architecture.md            ← System architecture documentation
```

---

## Development Roadmap

### Phase 1: Core MPC (Week 1-2)
- [ ] Implement bicycle_model.py with unit tests
- [ ] Implement linear_mpc.py with cvxpy + OSQP
- [ ] MATLAB cross-validation (same A, B, Q, R, x0 → same δ)
- [ ] Simple simulation loop: MPC + bicycle model, plot ey convergence

### Phase 2: Gymnasium Environment (Week 3-4)
- [ ] Implement CarlaMPCEnv with internal bicycle simulation (no CARLA yet)
- [ ] Implement weight_mapper.py (log scale action → Q, R)
- [ ] Implement reward.py
- [ ] Test with random actions → verify env doesn't crash
- [ ] Test with SB3 SAC → verify reward converges

### Phase 3: RL Training & Tuning (Week 5-8)
- [ ] Reward shaping iterations
- [ ] Observation normalization (VecNormalize)
- [ ] Curriculum learning: straight → curved → mixed
- [ ] Hyperparameter sweep (learning rate, hidden size, buffer size)
- [ ] Q, R evolution analysis: do learned weights make physical sense?

### Phase 4: CARLA Integration (Week 9-12)
- [ ] CARLA server setup, carla_env.py
- [ ] Replace internal simulation with CARLA API
- [ ] Sensor data extraction (IMU, GNSS, waypoints)
- [ ] Retrain RL in CARLA environment
- [ ] Domain randomization (mass, tire stiffness, delay, weather)

### Phase 5: Baselines & Evaluation (Week 13-16)
- [ ] Implement all 5 baselines
- [ ] Run evaluation scenarios (5 scenarios × 6 controllers × 10 seeds)
- [ ] Statistical comparison (mean ± std for all metrics)
- [ ] Generate thesis figures

### Phase 6: ROS2 Integration (Week 17-18)
- [ ] ROS2 node architecture
- [ ] CARLA-ROS2 bridge setup
- [ ] Real-time performance validation

### Phase 7: Thesis Writing (Week 19-24)
- [ ] Literature review
- [ ] Methodology chapter
- [ ] Results chapter
- [ ] Discussion & conclusion

---

## Key Technical Decisions & Rationale

1. **Why RL tunes Q,R instead of directly outputting δ:** MPC retains constraint satisfaction (lane boundaries, steering limits). Pure RL has no such guarantees. RL only adjusts performance tuning — safety comes from MPC.

2. **Why SAC over PPO:** Off-policy = sample efficient. CARLA is slow, each sample is expensive. PPO (on-policy) would need 3-5x more simulation time.

3. **Why log scale for action mapping:** Q ranges from 1 to 1000. Linear mapping wastes action resolution on high values where differences don't matter. Log scale gives equal resolution across orders of magnitude.

4. **Why hierarchical timing (2 Hz RL, 50 Hz MPC):** Q, R don't need to change every 20ms. Changing them too fast makes MPC behavior erratic. 0.5s update rate gives MPC time to settle.

5. **Why bicycle model (not full vehicle):** 4 states, linear → QP, <2ms solve time. Full nonlinear model → NLP, 10-50ms. For lateral control at normal driving (not handling limit), bicycle model is sufficient.

6. **Why perfect perception assumption:** Thesis contribution is the RL-MPC controller, not perception. Using CARLA ground-truth with injected noise is standard practice and keeps scope manageable.

7. **Why cvxpy + OSQP:** cvxpy for readable formulation during development, OSQP for fast QP solving. If performance is insufficient, switch to acados for C code generation.

8. **Why SB3 first, then CleanRL:** SB3 for rapid prototyping and validation. CleanRL for thesis — demonstrates understanding of algorithm internals, enables custom modifications.

---

## Potential Issues & Mitigations

| Issue | Mitigation |
|-------|-----------|
| RL doesn't converge | Normalize reward (running mean/std), narrow action range first, curriculum learning |
| MPC infeasible with RL-chosen Q,R | Fallback to nominal Q,R, add Q/R rate limiting |
| CARLA simulation too slow | Phase 1-3 without CARLA, use internal bicycle sim |
| Q,R values don't make physical sense | Log Q,R evolution, add reward term penalizing extreme values |
| Sim-to-real gap | Domain randomization on m, Cf, Cr, delay |
| Reviewer asks "stability guarantee?" | Show Lyapunov analysis for MPC with bounded Q,R range |
| Reviewer asks "why not gain scheduling?" | Gain scheduling is a baseline — show RL outperforms it |

---

## Commands Reference

```bash
# Install dependencies
pip install -e .

# Phase 1-2: Train with internal simulation
python scripts/train.py --env internal --algo sac --timesteps 200000

# Phase 4: Train with CARLA
./scripts/carla_setup.sh  # Start CARLA server
python scripts/train.py --env carla --algo sac --timesteps 500000

# Evaluate
python scripts/evaluate.py --model models/sac_best.zip --scenarios all

# Plot
python scripts/plot_results.py --results results/ --output docs/figures/
```
