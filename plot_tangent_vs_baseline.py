#!/usr/bin/env python3
"""Plot tangent experiment results vs spherical and stereographic baselines.

Compares success rate learning curves for cube-triple-play tasks 1-3, H=1 and H=5.
"""
import os, json, re, csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.artifact_paths import reports_dir
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
TASKS = ["task1", "task2", "task3"]
HORIZONS = {"h1": 1, "h5": 5}
ENV_PREFIX = "cube-triple-play-singletask"

# Historical raw runs remain on the data disk.
TANGENT_BASE = reports_dir().parents[1] / "archive/consolidated-legacy-storage-20261018/historical-runs/dummy/tangent"
# Previous baselines: docs/data/ds_experiments/
BASELINE_BASE = Path("docs/data/ds_experiments/cube-triple-play-singletask")

# Methods to compare: (label, color, data_key)
# H=1 baselines use "rlpd", H=5 baselines use "rlpd" (not rlpd-4s)
COMPARISONS = [
    ("spherical", "#2ca02c", "spherical"),    # green
    ("stereographic", "#1f77b4", "stereographic"),  # blue
    ("tangent", "#d62728", None),  # red - uses tangent path
]

# ── Helpers ─────────────────────────────────────────────────────────────────

def read_eval_csv(path: Path):
    """Read eval.csv and return list of (step, success) tuples."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(float(row["step"])), float(row["success"])))
    return rows


def collect_data(task: str, horizon: int, tangent: bool = True, mode: str = None):
    """Collect all eval data for a given config.

    Returns: dict mapping seed -> list of (step, success)
    """
    if tangent:
        env_dir = TANGENT_BASE / f"{ENV_PREFIX}-{task}-v0"
        if not env_dir.exists():
            return {}
    else:
        if horizon == 1:
            env_dir = BASELINE_BASE / task / "h1" / "rlpd" / mode
        else:
            env_dir = BASELINE_BASE / task / "h5" / "rlpd" / mode
        if not env_dir.exists():
            return {}

    result = {}
    for seed_dir in sorted(env_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        eval_file = seed_dir / "eval.csv"
        if not eval_file.exists():
            continue

        if tangent:
            flags_file = seed_dir / "flags.json"
            if not flags_file.exists():
                continue
            with open(flags_file) as f:
                flags = json.load(f)
            if flags.get("horizon_length") != horizon:
                continue
            seed = flags.get("seed", "?")
        else:
            seed_name = seed_dir.name
            try:
                seed = int(seed_name.replace("seed", ""))
            except ValueError:
                continue

        rows = read_eval_csv(eval_file)
        if rows:
            result[seed] = rows
    return result


def compute_curve(seed_data: dict):
    """Compute mean and std across seeds for each eval step.

    seed_data: dict mapping seed -> list of (step, success)
    Returns: steps array, means array, stds array
    """
    # Collect all unique steps
    all_steps = set()
    for rows in seed_data.values():
        for step, _ in rows:
            all_steps.add(step)
    all_steps = sorted(all_steps)

    if not all_steps:
        return np.array([]), np.array([]), np.array([])

    # Build per-seed dict of step -> success
    curves = {}
    for seed, rows in seed_data.items():
        curves[seed] = dict(rows)

    means, stds = [], []
    for step in all_steps:
        vals = [curves[s].get(step, np.nan) for s in curves]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    return np.array(all_steps), np.array(means), np.array(stds)


# ── Plotting ────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, (h_label, horizon) in enumerate([("H=1", 1), ("H=5", 5)]):
        for col_idx, task in enumerate(TASKS):
            ax = axes[row_idx][col_idx]
            task_num = task.replace("task", "")

            for label, color, mode in COMPARISONS:
                if mode is None:
                    seed_data = collect_data(task, horizon, tangent=True)
                else:
                    seed_data = collect_data(task, horizon, tangent=False, mode=mode)

                if not seed_data:
                    print(f"  [SKIP] {task} {h_label} {label}: no data")
                    ax.text(0.5, 0.5, f"{label}: no data",
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=9, color='gray')
                    continue

                steps, means, stds = compute_curve(seed_data)
                if len(steps) == 0:
                    print(f"  [SKIP] {task} {h_label} {label}: empty curve")
                    continue

                n_seeds = len(seed_data)
                lbl = f"{label} (n={n_seeds})"
                ax.plot(steps, means, color=color, label=lbl, linewidth=1.5)
                ax.fill_between(steps, means - stds, means + stds,
                                color=color, alpha=0.12)

                # Print final stats
                final_mean = means[-1] if len(means) > 0 else float('nan')
                final_std = stds[-1] if len(stds) > 0 else float('nan')
                final_step = steps[-1] if len(steps) > 0 else 0
                print(f"  {task} {h_label} {label}: "
                      f"final={final_mean:.3f}±{final_std:.3f} "
                      f"@step={final_step} (n={n_seeds})")

            ax.set_title(f"Task {task_num} — {h_label}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Steps", fontsize=10)
            ax.set_ylabel("Success Rate", fontsize=10)
            ax.legend(fontsize=8, loc='lower right')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Tangent vs Spherical vs Stereographic — Cube Triple Play",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = (reports_dir() / "tangent_vs_sph_stereo.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")
    plt.close()

    # ── Also generate summary table ─────────────────────────────────────────
    print("\n=== Final Success Rate Summary ===")
    print(f"{'Task':<10} {'H':<5} {'Method':<16} {'Final Success':<20} {'Seeds':<8}")
    print("-" * 60)
    rows = []
    for horizon in [1, 5]:
        for task in TASKS:
            for label, _, mode in COMPARISONS:
                if mode is None:
                    seed_data = collect_data(task, horizon, tangent=True)
                else:
                    seed_data = collect_data(task, horizon, tangent=False, mode=mode)
                if not seed_data:
                    continue
                steps, means, stds = compute_curve(seed_data)
                if len(means) == 0:
                    continue
                final = means[-1]
                final_std = stds[-1]
                n = len(seed_data)
                print(f"{task:<10} {horizon:<5} {label:<16} {final:.3f} ± {final_std:.3f}     {n:<8}")


if __name__ == "__main__":
    main()
