"""Turn results/eval_metrics.csv into thesis-ready comparison plots.

Generates:
  results/plots/rmse_ey_by_scenario.png      grouped bar (controller x scenario)
  results/plots/max_ay_by_scenario.png       grouped bar (controller x scenario)
  results/plots/return_by_scenario.png       grouped bar (controller x scenario)
  results/plots/violations.png               crash + lane-violation rates
  results/plots/distribution_rmse_ey.png     box plot (variance across seeds)

Run after evaluate.py:
    python scripts/plot_results.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONTROLLER_ORDER = ["fixed_mpc", "gain_scheduled_mpc", "stanley", "pure_pursuit",
                    "pure_rl", "rl_mpc"]
SCENARIO_ORDER = ["highway_straight", "highway_curve", "urban_sharp",
                  "mixed_route", "low_mu_wet"]


def _ordered(df, col, order):
    """Reindex df by the desired order, dropping any missing entries."""
    present = [c for c in order if c in df[col].unique()]
    df = df[df[col].isin(present)].copy()
    df[col] = pd.Categorical(df[col], categories=present, ordered=True)
    return df


def grouped_bar(df, metric, ylabel, title, out_png, lower_is_better=True):
    """One bar per (scenario, controller) pair, with stddev error bars."""
    df = _ordered(_ordered(df, "scenario", SCENARIO_ORDER), "controller", CONTROLLER_ORDER)
    g = df.groupby(["scenario", "controller"])[metric].agg(["mean", "std"]).reset_index()
    scen = list(dict.fromkeys(g["scenario"]))
    ctrls = list(dict.fromkeys(g["controller"]))
    width = 0.8 / max(len(ctrls), 1)
    x_idx = np.arange(len(scen))

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(ctrls):
        gc = g[g["controller"] == c].set_index("scenario").reindex(scen)
        ax.bar(x_idx + i * width, gc["mean"], width=width,
               yerr=gc["std"].fillna(0), capsize=3,
               label=c, color=cmap(i))
    ax.set_xticks(x_idx + width * (len(ctrls) - 1) / 2)
    ax.set_xticklabels(scen, rotation=15)
    ax.set_ylabel(ylabel)
    arrow = " (lower = better)" if lower_is_better else " (higher = better)"
    ax.set_title(title + arrow)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def violation_plot(df, out_png):
    """Stack bar of crash % and lane-violation % per controller."""
    df = _ordered(df, "controller", CONTROLLER_ORDER)
    g = df.groupby("controller").agg(
        crash_pct=("crashed", lambda v: 100*np.mean(v)),
        viol_pct=("lane_violation_pct", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["crash_pct"], width=0.4, label="crash %", color="tab:red")
    ax.bar(x + 0.2, g["viol_pct"], width=0.4, label="lane violation %", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(g["controller"], rotation=15)
    ax.set_ylabel("%")
    ax.set_title("Safety — crash and lane-violation rates (lower = better)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def box_plot(df, metric, ylabel, title, out_png):
    """Box plot per controller across all seeds & scenarios (shows variance)."""
    df = _ordered(df, "controller", CONTROLLER_ORDER)
    data = [df[df["controller"] == c][metric].dropna().values
            for c in df["controller"].cat.categories]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(data, tick_labels=list(df["controller"].cat.categories), showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "eval_metrics.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "plots"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    grouped_bar(df, "rmse_ey", "RMSE(ey) [m]",
                "Tracking accuracy",
                out / "rmse_ey_by_scenario.png", lower_is_better=True)
    grouped_bar(df, "max_ay_g", "max |ay| [g]",
                "Peak lateral acceleration (comfort + tire-grip headroom)",
                out / "max_ay_by_scenario.png", lower_is_better=True)
    grouped_bar(df, "return_", "Episode return",
                "Cumulative reward",
                out / "return_by_scenario.png", lower_is_better=False)
    violation_plot(df, out / "violations.png")
    box_plot(df, "rmse_ey", "RMSE(ey) [m]",
             "Tracking accuracy — distribution across all (seed, scenario)",
             out / "distribution_rmse_ey.png")

    print(f"wrote 5 plots -> {out}")


if __name__ == "__main__":
    main()
