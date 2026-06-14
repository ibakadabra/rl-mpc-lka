# CARLA Validation Pass — Remote PC Setup

Phase-5 "B" path: drop the trained model + baselines into CARLA Town04, run the
5 evaluation scenarios, compare metrics with the internal-sim run. **No
retraining.** All work happens on the PC that owns the CARLA simulator.

## 0. Prereqs (on the remote PC)

- CARLA 0.9.15 installed (you already have it).
- Python 3.10–3.12 venv. CARLA's PythonAPI ships a `.whl` matching its version.
  We are using Python 3.14 on the dev PC, but for the remote PC where CARLA
  lives, **use 3.10–3.12** because the CARLA wheel is version-locked.
- Git clone of this repo on the remote PC.

## 1. Sync code

```bash
# on the dev PC: commit/push everything
git push

# on the remote PC: clone or pull
git clone <repo-url> rl-mpc
cd rl-mpc
git pull
```

Copy the trained model + obs normalizer over (they live in `models/`, not
tracked by git):

```bash
scp dev-pc:.../models/sac_rlmpc.zip      models/
scp dev-pc:.../models/vecnormalize.pkl   models/
```

## 2. Set up the remote Python env

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# CARLA PythonAPI wheel (path is inside the CARLA install)
pip install "$CARLA_ROOT"/PythonAPI/carla/dist/carla-0.9.15-*-py3*.whl
```

Verify:
```bash
python -c "import carla, stable_baselines3, torch; print('ok')"
```

## 3. Launch CARLA

```bash
# off-screen (recommended for headless servers):
CARLA_ROOT=/opt/carla-0.9.15 ./scripts/carla_setup.sh

# windowed (for visualizing while debugging):
./scripts/carla_setup.sh --windowed
```

Wait until you see "Bringing World ... up" before continuing.

## 4. Run the validation pass

In a second terminal on the same PC:

```bash
source .venv/bin/activate
python scripts/evaluate_carla.py --n-seeds 3
```

Output: `results/eval_carla_metrics.csv` + console summary table.

### Subset run (smoke test before the full sweep)

```bash
python scripts/evaluate_carla.py --n-seeds 1 \
    --scenarios highway_curve mixed_route
```

### Connecting from a different machine

If you prefer to keep CARLA on the remote PC and run the evaluation script
locally:

```bash
python scripts/evaluate_carla.py --host 192.168.X.Y --n-seeds 3
```

Make sure port 2000 (RPC) and 2001 (streaming) are open on the remote PC's
firewall and reachable from your dev PC.

## 5. Side-by-side comparison

After both `eval_metrics.csv` (internal sim) and `eval_carla_metrics.csv`
(CARLA) exist, the thesis comparison table comes from joining them on
`(scenario, controller)`. A small helper script — `compare_envs.py` — can be
added if needed; for now `pandas.merge` on those two CSVs does the job.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `carla.Client.connect timeout` | server not up yet | wait 10 s after `CarlaUE4.sh` start |
| `RPCError: timeout` mid-run | server hang under load | lower `--quality-level=Low`, fewer NPCs |
| `Bad allocation` after a few episodes | actor leak | the env does cleanup in `__exit__`; ensure the script terminates cleanly |
| no waypoints found | vehicle off-road | check `spawn_index` — pick a road-side spawn for that map |
| CARLA wheel install fails | python version mismatch | wheels are per-Python-minor; use 3.10/3.11/3.12 to match the shipped wheel |
