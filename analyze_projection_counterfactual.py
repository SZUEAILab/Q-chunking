#!/usr/bin/env python3
"""Paper-oriented analysis for the Cartesian L2 projection counterfactual."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "baseline": "Cartesian baseline",
    "projection": "Cartesian + L2 projection",
    "adapter_cube": "DS direct / cube",
    "adapter_sphere": "DS direct / sphere",
}

CONDITION_COLORS = {
    "baseline": "#d62728",
    "projection": "#e377c2",
    "adapter_cube": "#1f77b4",
    "adapter_sphere": "#ff7f0e",
}


def condition(flags):
    mode = flags["ds_mode"]
    if mode in ("none", "projection"):
        return "baseline" if mode == "none" else "projection"
    if mode == "adapter":
        return f"adapter_{flags['ds_speed_bound']}"
    raise ValueError(f"Unexpected condition: {mode}")


def read_eval(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = np.asarray([int(float(row["step"])) for row in rows])
    success = np.asarray([float(row["success"]) for row in rows])
    return steps, success


def read_action_stats(path, start_training):
    norms = []
    ge_unit = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["step"])) <= start_training:
                continue
            norms.append(float(row["action_xyz_norm"]))
            ge_unit.append(float(row["action_xyz_at_unit_bound"]))
    norms = np.asarray(norms)
    ge_unit = np.asarray(ge_unit)
    if not len(norms):
        return {key: np.nan for key in ("norm_mean", "norm_p50", "norm_p95", "ge_unit_frac")}
    return {
        "norm_mean": float(norms.mean()),
        "norm_p50": float(np.quantile(norms, 0.50)),
        "norm_p95": float(np.quantile(norms, 0.95)),
        "ge_unit_frac": float(ge_unit.mean()),
    }


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        eval_path = flags_path.with_name("eval.csv")
        env_path = flags_path.with_name("env.csv")
        if not eval_path.exists() or not eval_path.stat().st_size:
            continue
        flags = json.loads(flags_path.read_text())
        steps, success = read_eval(eval_path)
        if not len(steps):
            continue
        auc_steps = np.concatenate(([0], steps))
        auc_values = np.concatenate(([0.0], success))
        action_stats = (
            read_action_stats(env_path, int(flags["start_training"]))
            if env_path.exists()
            else {}
        )
        runs.append(
            {
                "condition": condition(flags),
                "seed": int(flags["seed"]),
                "steps": steps,
                "success": success,
                "final_success": float(success[-1]),
                "tail3_success": float(success[-3:].mean()),
                "auc": float(np.trapezoid(auc_values, auc_steps) / steps[-1]),
                **action_stats,
            }
        )
    return runs


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    rows = []
    metrics = (
        "final_success",
        "tail3_success",
        "auc",
        "norm_mean",
        "norm_p95",
        "ge_unit_frac",
    )
    for name, values in sorted(grouped.items()):
        row = {"condition": name, "display_name": DISPLAY_NAMES[name], "num_seeds": len(values)}
        for metric in metrics:
            samples = np.asarray([value.get(metric, np.nan) for value in values])
            row[f"{metric}_mean"] = float(np.nanmean(samples))
            row[f"{metric}_std"] = float(np.nanstd(samples, ddof=1)) if len(samples) > 1 else 0.0
        rows.append(row)
    return rows


def sign_flip_pvalue(differences):
    differences = np.asarray(differences)
    observed = abs(differences.mean())
    permuted = [
        abs(np.mean(differences * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def paired_comparisons(runs, bootstrap_samples=20000):
    by_condition = defaultdict(dict)
    for run in runs:
        by_condition[run["condition"]][run["seed"]] = run
    rng = np.random.default_rng(20260925)
    rows = []
    for reference in ("baseline", "projection", "adapter_sphere"):
        if reference not in by_condition:
            continue
        for name in sorted(by_condition):
            if name == reference:
                continue
            seeds = sorted(set(by_condition[name]) & set(by_condition[reference]))
            for metric in ("final_success", "tail3_success", "auc"):
                diff = np.asarray(
                    [by_condition[name][s][metric] - by_condition[reference][s][metric] for s in seeds]
                )
                indices = rng.integers(0, len(diff), size=(bootstrap_samples, len(diff)))
                boot = diff[indices].mean(axis=1)
                rows.append(
                    {
                        "condition": name,
                        "reference": reference,
                        "metric": metric,
                        "num_pairs": len(diff),
                        "mean_delta": float(diff.mean()),
                        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                        "exact_sign_flip_p": sign_flip_pvalue(diff),
                    }
                )
    return rows


def plot_curves(runs, path):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for name in ("baseline", "projection", "adapter_cube", "adapter_sphere"):
        values = grouped.get(name, [])
        if not values:
            continue
        common_steps = sorted(set.intersection(*[set(run["steps"]) for run in values]))
        curves = np.asarray(
            [[run["success"][np.where(run["steps"] == step)[0][0]] for step in common_steps] for run in values]
        )
        mean = curves.mean(axis=0) * 100
        sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) * 100
        x = np.asarray(common_steps) / 1000
        axis.plot(
            x,
            mean,
            marker="o",
            linewidth=2.3,
            color=CONDITION_COLORS[name],
            label=DISPLAY_NAMES[name],
        )
        axis.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=CONDITION_COLORS[name],
            alpha=0.18,
        )
    axis.set(xlabel="Environment steps (k)", ylabel="Success rate (%)", ylim=(-2, 102))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_action_stats(runs, path):
    order = ("baseline", "projection", "adapter_cube", "adapter_sphere")
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    labels = [DISPLAY_NAMES[name] for name in order]
    colors = [CONDITION_COLORS[name] for name in order]
    means = np.asarray([[run["norm_mean"] for run in grouped[name]] for name in order])
    p95s = np.asarray([[run["norm_p95"] for run in grouped[name]] for name in order])
    ge_unit = np.asarray(
        [[run["ge_unit_frac"] for run in grouped[name]] for name in order]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    x = np.arange(len(order))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        means.mean(axis=1),
        width,
        yerr=means.std(axis=1, ddof=1),
        capsize=3,
        label="Mean norm",
        color=colors,
        alpha=0.78,
    )
    axes[0].bar(
        x + width / 2,
        p95s.mean(axis=1),
        width,
        yerr=p95s.std(axis=1, ddof=1),
        capsize=3,
        label="95th percentile",
        color=colors,
        hatch="//",
        alpha=0.45,
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axes[0].set(ylabel="XYZ action norm", xticks=x, xticklabels=labels)
    axes[0].legend(fontsize=8)

    axes[1].bar(
        x,
        ge_unit.mean(axis=1) * 100,
        yerr=ge_unit.std(axis=1, ddof=1) * 100,
        capsize=3,
        color=colors,
        alpha=0.8,
    )
    axes[1].set(
        ylabel="Actions with XYZ norm >= 1 (%)",
        xticks=x,
        xticklabels=labels,
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=18, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No completed runs below {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.root.name
    per_run = [
        {key: value for key, value in run.items() if key not in ("steps", "success")}
        for run in runs
    ]
    summary = summarize(runs)
    comparisons = paired_comparisons(runs)
    write_csv(args.output_dir / f"{stem}_per_run.csv", per_run, list(per_run[0]))
    write_csv(args.output_dir / f"{stem}_summary.csv", summary, list(summary[0]))
    write_csv(
        args.output_dir / f"{stem}_paired.csv", comparisons, list(comparisons[0])
    )
    plot_curves(runs, args.output_dir / f"{stem}_curves.png")
    plot_action_stats(runs, args.output_dir / f"{stem}_action_norms.png")
    for row in sorted(summary, key=lambda item: item["final_success_mean"], reverse=True):
        print(
            f"{row['display_name']:<30} n={row['num_seeds']} "
            f"final={100 * row['final_success_mean']:.1f}+-{100 * row['final_success_std']:.1f} "
            f"tail3={100 * row['tail3_success_mean']:.1f} "
            f"AUC={100 * row['auc_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
