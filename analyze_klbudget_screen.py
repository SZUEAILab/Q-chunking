#!/usr/bin/env python3
"""Analyze the matched-reference KL-budget calibration screen."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


ORDER = ("V5", "S5")
DISPLAY = {
    "V5": "Action volume / KL 5.32",
    "S5": "Semantic product / KL 5.32",
}
COLORS = {"V5": "#9467bd", "S5": "#2ca02c"}


def condition(flags):
    if bool(flags["ds_separate_alpha"]):
        raise ValueError("KL-budget screen expects a single temperature")
    reference = flags["ds_regularizer"]
    if reference == "action_volume":
        return "V5"
    if reference == "semantic_product":
        return "S5"
    raise ValueError(f"Unexpected reference: {reference}")


def read_curve(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([int(float(row["step"])) for row in rows]),
        np.asarray([float(row["success"]) for row in rows]),
    )


def read_action_norm(path, start_training):
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(float(row["step"])) > start_training
        ]
    norms = np.asarray([float(row["action_xyz_norm"]) for row in rows])
    return float(norms.mean()), float(np.quantile(norms, 0.95))


def read_training_tail(path, count=10):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[-count:]
    return {
        "tail_alpha": float(np.mean([float(row["actor/alpha"]) for row in rows])),
        "tail_entropy": float(
            np.mean([float(row["actor/entropy"]) for row in rows])
        ),
    }


def read_fixed_success(path):
    if not path.exists():
        return np.nan
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return float(rows[-1]["success"]) if rows else np.nan


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        eval_path = flags_path.with_name("eval.csv")
        if not eval_path.exists():
            continue
        steps, success = read_curve(eval_path)
        if not len(steps) or steps[-1] != 500000:
            continue
        auc = np.trapezoid(
            np.concatenate(([0.0], success)), np.concatenate(([0], steps))
        ) / steps[-1]
        norm_mean, norm_p95 = read_action_norm(
            flags_path.with_name("env.csv"), int(flags["start_training"])
        )
        runs.append(
            {
                "condition": condition(flags),
                "seed": int(flags["seed"]),
                "steps": steps,
                "success": success,
                "final_success": float(success[-1]),
                "tail3_success": float(success[-3:].mean()),
                "auc": float(auc),
                "fixed_success": read_fixed_success(
                    flags_path.with_name("eval_fixed_seed.csv")
                ),
                "norm_mean": norm_mean,
                "norm_p95": norm_p95,
                **read_training_tail(flags_path.with_name("online_agent.csv")),
            }
        )
    return runs


def exact_sign_flip_pvalue(differences):
    observed = abs(float(np.mean(differences)))
    null = [
        abs(float(np.mean(differences * signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()

    runs = load_runs(args.root)
    if len(runs) != 8:
        raise SystemExit(f"Expected 8 complete runs, found {len(runs)}")
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    for values in grouped.values():
        values.sort(key=lambda run: run["seed"])

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
    write_csv(args.output_dir / "klbudget_screen_per_run.csv", per_run)

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
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(row)
    write_csv(args.output_dir / "klbudget_screen_summary.csv", summary)

    rng = np.random.default_rng(20260925)
    paired = []
    for metric in ("final_success", "tail3_success", "auc", "fixed_success"):
        v5 = np.asarray([run[metric] for run in grouped["V5"]])
        s5 = np.asarray([run[metric] for run in grouped["S5"]])
        if not np.all(np.isfinite(v5)) or not np.all(np.isfinite(s5)):
            continue
        differences = s5 - v5
        indices = rng.integers(0, len(differences), size=(20000, len(differences)))
        bootstrap = differences[indices].mean(axis=1)
        paired.append(
            {
                "contrast": "S5-V5",
                "metric": metric,
                "mean_delta": float(differences.mean()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                "paired_differences": ";".join(f"{x:.6g}" for x in differences),
            }
        )
    write_csv(args.output_dir / "klbudget_screen_paired.csv", paired)

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
            alpha=0.16,
            color=COLORS[name],
        )
        norm_values = np.asarray([run["norm_mean"] for run in grouped[name]])
        x = ORDER.index(name)
        axes[1].scatter(
            x + rng.uniform(-0.05, 0.05, len(norm_values)),
            norm_values,
            color=COLORS[name],
        )
        axes[1].errorbar(
            x,
            norm_values.mean(),
            yerr=norm_values.std(ddof=1),
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
    figure.suptitle("Matched reference measures at KL budget 5.32")
    figure.tight_layout()
    figure.savefig(args.output_dir / "klbudget_screen.png", dpi=220)
    plt.close(figure)

    for row in summary:
        print(row)
    for row in paired:
        print(row)


if __name__ == "__main__":
    main()
