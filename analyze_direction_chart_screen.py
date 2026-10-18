#!/usr/bin/env python3
"""Compare spherical and stereographic charts under one semantic reference."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from analyze_klbudget_screen import (
    exact_sign_flip_pvalue,
    load_runs,
    write_csv,
)


ORDER = ("S5", "G5")
DISPLAY = {
    "S5": "Spherical / semantic / KL 5.32",
    "G5": "Stereographic / semantic / KL 5.32",
}
COLORS = {"S5": "#2ca02c", "G5": "#17becf"}


def relabel(control_root, treatment_root):
    control = [run for run in load_runs(control_root) if run["condition"] == "S5"]
    treatment = load_runs(treatment_root)
    for run in treatment:
        run["condition"] = "G5"
    runs = control + treatment
    counts = {name: sum(run["condition"] == name for run in runs) for name in ORDER}
    if any(count != 4 for count in counts.values()):
        raise SystemExit(f"Expected four complete seeds per chart, found {counts}")
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control_root", type=Path)
    parser.add_argument("treatment_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()

    runs = relabel(args.control_root, args.treatment_root)
    grouped = {
        name: sorted(
            [run for run in runs if run["condition"] == name],
            key=lambda run: run["seed"],
        )
        for name in ORDER
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    for run in sorted(runs, key=lambda item: (ORDER.index(item["condition"]), item["seed"])):
        row = {key: value for key, value in run.items() if key not in ("steps", "success")}
        row.update(
            {
                f"success_{step // 1000}k": value
                for step, value in zip(run["steps"], run["success"], strict=True)
            }
        )
        per_run.append(row)
    write_csv(args.output_dir / "direction_chart_screen_per_run.csv", per_run)

    metrics = (
        "final_success",
        "tail3_success",
        "auc",
        "fixed_success",
        "norm_mean",
        "norm_p95",
        "tail_alpha",
        "tail_entropy",
    )
    summary = []
    for name in ORDER:
        row = {"condition": name, "display_name": DISPLAY[name], "num_seeds": 4}
        for metric in metrics:
            values = np.asarray([run[metric] for run in grouped[name]])
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        summary.append(row)
    write_csv(args.output_dir / "direction_chart_screen_summary.csv", summary)

    rng = np.random.default_rng(20260925)
    paired = []
    for metric in ("final_success", "tail3_success", "auc", "fixed_success"):
        control = np.asarray([run[metric] for run in grouped["S5"]])
        treatment = np.asarray([run[metric] for run in grouped["G5"]])
        if not np.all(np.isfinite(control)) or not np.all(np.isfinite(treatment)):
            continue
        differences = treatment - control
        indices = rng.integers(0, len(differences), size=(20000, len(differences)))
        bootstrap = differences[indices].mean(axis=1)
        paired.append(
            {
                "contrast": "G5-S5",
                "metric": metric,
                "mean_delta": float(differences.mean()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                "paired_differences": ";".join(f"{x:.6g}" for x in differences),
            }
        )
    write_csv(args.output_dir / "direction_chart_screen_paired.csv", paired)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for name in ORDER:
        curves = np.asarray([run["success"] for run in grouped[name]]) * 100
        steps = grouped[name][0]["steps"] / 1000
        axes[0].plot(
            steps,
            curves.mean(axis=0),
            marker="o",
            linewidth=2.2,
            color=COLORS[name],
            label=DISPLAY[name],
        )
        axes[0].fill_between(
            steps,
            curves.mean(axis=0) - curves.std(axis=0, ddof=1),
            curves.mean(axis=0) + curves.std(axis=0, ddof=1),
            color=COLORS[name],
            alpha=0.16,
        )
        norms = np.asarray([run["norm_mean"] for run in grouped[name]])
        x = ORDER.index(name)
        axes[1].scatter(
            x + rng.uniform(-0.05, 0.05, len(norms)), norms, color=COLORS[name]
        )
        axes[1].errorbar(
            x,
            norms.mean(),
            yerr=norms.std(ddof=1),
            marker="o",
            capsize=4,
            linewidth=2,
            color=COLORS[name],
        )
    axes[0].set_xlabel("Environment steps (k)")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend()
    axes[1].set_xticks(range(len(ORDER)), ORDER)
    axes[1].set_ylabel("Mean translational norm")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Direction chart under a matched semantic reference")
    figure.tight_layout()
    figure.savefig(args.output_dir / "direction_chart_screen.png", dpi=220)
    plt.close(figure)

    for row in summary:
        print(row)
    for row in paired:
        print(row)


if __name__ == "__main__":
    main()
