#!/usr/bin/env python3
"""Summarize and plot the controlled direction-speed screening experiment."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "baseline": "Baseline",
    "cartesian_l2_projection": "Cartesian + XYZ L2 projection",
    "adapter_cartesian_log_cube": "Adapter Cartesian / log / cube",
    "adapter_cartesian_direct_cube": "Adapter Cartesian / direct / cube",
    "adapter_cartesian_direct_sphere": "Adapter Cartesian / direct / sphere",
    "adapter_spherical_direct_cube": "Adapter spherical / direct / cube",
    "integrated_stereographic_cube": "Integrated stereographic / cube",
}


def condition(flags):
    mode = flags["ds_mode"]
    if mode == "none":
        return "baseline"
    if mode == "projection":
        return "cartesian_l2_projection"
    if mode == "integrated":
        return (
            f"integrated_{flags['ds_integrated_type']}_{flags['ds_speed_bound']}"
        )
    return "adapter_{}_{}_{}".format(
        flags["ds_adapter_representation"],
        flags["ds_adapter_speed_interpretation"],
        flags["ds_speed_bound"],
    )


def read_curve(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = np.asarray([int(float(row["step"])) for row in rows])
    success = np.asarray([float(row["success"]) for row in rows])
    return steps, success


def load_runs(root):
    runs = defaultdict(list)
    for flags_path in sorted(root.rglob("flags.json")):
        eval_path = flags_path.with_name("eval.csv")
        if not eval_path.exists() or eval_path.stat().st_size == 0:
            continue
        flags = json.loads(flags_path.read_text())
        steps, success = read_curve(eval_path)
        if not len(steps):
            continue
        runs[condition(flags)].append(
            {"seed": int(flags["seed"]), "steps": steps, "success": success}
        )
    return runs


def summarize(runs, output_csv):
    rows = []
    for name, condition_runs in sorted(runs.items()):
        complete_steps = min(run["steps"][-1] for run in condition_runs)
        aligned = []
        aucs = []
        for run in condition_runs:
            mask = run["steps"] <= complete_steps
            steps = run["steps"][mask]
            values = run["success"][mask]
            aligned.append(values[-1])
            auc_steps = np.concatenate(([0], steps))
            auc_values = np.concatenate(([0.0], values))
            aucs.append(np.trapezoid(auc_values, auc_steps) / complete_steps)
        rows.append(
            {
                "condition": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "num_seeds": len(condition_runs),
                "complete_steps": int(complete_steps),
                "final_mean": float(np.mean(aligned)),
                "final_std": float(np.std(aligned, ddof=1)) if len(aligned) > 1 else 0.0,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            }
        )
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot(runs, output_path):
    fig, axis = plt.subplots(figsize=(9.2, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    for color, (name, condition_runs) in zip(colors, sorted(runs.items())):
        common_steps = sorted(set.intersection(*[set(r["steps"]) for r in condition_runs]))
        if not common_steps:
            continue
        values = np.asarray(
            [
                [r["success"][np.where(r["steps"] == step)[0][0]] for step in common_steps]
                for r in condition_runs
            ]
        )
        for run in condition_runs:
            axis.plot(run["steps"] / 1000, run["success"] * 100, color=color, alpha=0.18)
        mean = values.mean(axis=0) * 100
        axis.plot(
            np.asarray(common_steps) / 1000,
            mean,
            color=color,
            linewidth=2.3,
            marker="o",
            label=DISPLAY_NAMES.get(name, name),
        )
    axis.set(xlabel="Environment steps (k)", ylabel="Success rate (%)", ylim=(-2, 102))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No completed eval.csv files below {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.root.name
    summary_path = args.output_dir / f"{run_name}_summary.csv"
    plot_path = args.output_dir / f"{run_name}_curves.png"
    rows = summarize(runs, summary_path)
    plot(runs, plot_path)
    for row in sorted(rows, key=lambda item: item["final_mean"], reverse=True):
        print(
            f"{row['display_name']:<43} n={row['num_seeds']} "
            f"final={100 * row['final_mean']:.1f}+-{100 * row['final_std']:.1f} "
            f"AUC={100 * row['auc_mean']:.1f}+-{100 * row['auc_std']:.1f}"
        )
    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()
