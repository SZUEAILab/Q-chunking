#!/usr/bin/env python3
"""Regenerate all DS experiment plots from ds_experiments data."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path(__file__).resolve().parent / "data/ds_experiments/cube-triple-play-singletask"
IMG  = Path(__file__).resolve().parent / "images"
IMG.mkdir(parents=True, exist_ok=True)

TASKS = ["task1", "task2", "task3", "task4", "task5"]
DS_ORDER = ["none", "posthoc", "stereographic", "spherical"]
DS_LABELS = {"none": "none (TanhNormal)", "posthoc": "posthoc (D+1)",
             "stereographic": "stereographic", "spherical": "spherical"}
DS_COLORS = {"none": "#2166ac", "posthoc": "#f4a582",
             "stereographic": "#4dac26", "spherical": "#d73027"}
DS_LINESTYLE = {"none": "--", "posthoc": "-", "stereographic": "-", "spherical": "-"}

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 11})


def load_eval(task, h, ds, seed):
    p = BASE / task / f"h{h}" / "rlpd" / ds / f"seed{seed}" / "eval.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["seed"] = seed
    return df


def load_all(task, h, ds):
    """Load all seeds for a task/horizon/ds_mode, return combined DataFrame."""
    frames = []
    for s in range(10):
        df = load_eval(task, h, ds, s)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def smooth_curves(df, window=3):
    """Compute rolling-window mean & std of success rate grouped by step."""
    agg = df.groupby("step")["success"].agg(["mean", "std"]).reset_index()
    agg.columns = ["step", "mean", "std"]
    agg["mean"] = agg["mean"].rolling(window, center=True, min_periods=1).mean()
    agg["std"]  = agg["std"].rolling(window, center=True, min_periods=1).mean()
    return agg["step"].values, agg["mean"].values * 100, agg["std"].values * 100


def final_success(task, h, ds):
    """Tail-averaged success (last 3 eval points, all seeds)."""
    df = load_all(task, h, ds)
    if df is None or df.empty:
        return 0
    max_step = df["step"].max()
    tail = df[df["step"] >= max_step - 200000]
    return tail["success"].mean() * 100


def final_success_per_seed(task, h, ds):
    """Per-seed tail-averaged success rates (for std computation)."""
    vals = []
    for s in range(10):
        p = BASE / task / f"h{h}" / "rlpd" / ds / f"seed{s}" / "eval.csv"
        if p.exists():
            df = pd.read_csv(p)
            max_step = df["step"].max()
            tail = df[df["step"] >= max_step - 200000]
            vals.append(tail["success"].mean() * 100)
    return vals


# ═══════════════════════════════════════════════════════════════════════════
# Per-task full curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_task(task):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, h in enumerate([1, 5]):
        ax = axes[idx]
        for ds in DS_ORDER:
            df = load_all(task, h, ds)
            if df is None or df.empty:
                continue
            steps, mean, std = smooth_curves(df)
            ax.plot(steps, mean, color=DS_COLORS[ds], linestyle=DS_LINESTYLE[ds],
                    linewidth=2, label=DS_LABELS[ds])
            ax.fill_between(steps, mean - std, mean + std, color=DS_COLORS[ds],
                            alpha=0.12)
        ax.set_title(f"{task}  H={h}", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Success %")
        ax.legend(loc="best", framealpha=0.85, ncol=2)
        ax.set_ylim(-5, 105)
    fig.suptitle(f"cube-triple-{task} — DS Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = IMG / f"cube-triple-{task}_DS_Curves_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Cross-task bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_cross_task_bar(h):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(TASKS))
    n = len(DS_ORDER)
    width = 0.8 / n
    for i, ds in enumerate(DS_ORDER):
        means = [final_success(t, h, ds) for t in TASKS]
        stds = [np.std(final_success_per_seed(t, h, ds)) for t in TASKS]
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      color=DS_COLORS[ds], label=DS_LABELS[ds],
                      edgecolor="white", capsize=4, error_kw={"linewidth": 2})
        for bar, v, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 3,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("task", "T") for t in TASKS])
    ax.set_ylabel("Success %")
    ax.set_title(f"H={h} — Cross-Task Comparison", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.85, ncol=2, fontsize=11)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    out = IMG / f"cross-task_H{h}_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Summary grid
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary():
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    legend_handles = []
    for ds in DS_ORDER:
        legend_handles.append(Line2D([0], [0], color=DS_COLORS[ds],
                                     linestyle=DS_LINESTYLE[ds], linewidth=2,
                                     label=DS_LABELS[ds]))

    for col, task in enumerate(TASKS):
        for row, h in enumerate([1, 5]):
            ax = axes[row][col]
            for ds in DS_ORDER:
                df = load_all(task, h, ds)
                if df is None or df.empty:
                    continue
                steps, mean, std = smooth_curves(df)
                ax.plot(steps, mean, color=DS_COLORS[ds], linestyle=DS_LINESTYLE[ds],
                        linewidth=1.5)
                ax.fill_between(steps, mean - std, mean + std, color=DS_COLORS[ds],
                                alpha=0.08)
            ax.set_title(f"{task} H={h}", fontsize=11, fontweight="bold")
            ax.set_ylim(-5, 110)
            if col == 0:
                ax.set_ylabel("Success %", fontsize=10)
            if row == 1:
                ax.set_xlabel("Step", fontsize=10)

    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               framealpha=0.9, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("cube-triple all tasks — DS ablation summary", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = IMG / "cube-triple_all_tasks_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Data summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n## Final Success Summary\n")
    print("| Task | H | none | posthoc | stereographic | spherical |")
    print("|------|---|:----:|:-------:|:-------------:|:---------:|")
    for task in TASKS:
        for h in [1, 5]:
            vals = [f"{final_success(task, h, ds):.1f}%" for ds in DS_ORDER]
            print(f"| {task} | H={h} | " + " | ".join(vals) + " |")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print_summary()

    print("\nGenerating per-task curves...")
    for task in TASKS:
        out = plot_per_task(task)
        print(f"  ✓ {out}")

    print("\nGenerating cross-task bars...")
    for h in [1, 5]:
        out = plot_cross_task_bar(h)
        print(f"  ✓ {out}")

    print("\nGenerating summary grid...")
    out = plot_summary()
    print(f"  ✓ {out}")

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
