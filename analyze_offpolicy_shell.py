#!/usr/bin/env python3
"""Paired analysis and frozen decisions for the off-policy shell bridge."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from train_synthetic_shell_offpolicy import OFFPOLICY_CONDITIONS


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(values, rng, count=50_000):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    estimates = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, (0.025, 0.975)))


def per_run_rows(curves, seeds):
    grouped = {}
    for row in curves:
        grouped.setdefault((row["condition"], int(row["seed"])), []).append(row)
    results = []
    for condition in OFFPOLICY_CONDITIONS:
        found = tuple(sorted(seed for name, seed in grouped if name == condition.name))
        if found != tuple(seeds):
            raise ValueError(f"{condition.name}: expected {tuple(seeds)}, found {found}")
        for seed in seeds:
            curve = sorted(
                grouped[(condition.name, seed)],
                key=lambda row: int(row["stage0_interactions"]),
            )
            steps = np.asarray([int(row["stage0_interactions"]) for row in curve])
            success = np.asarray([float(row["success"]) for row in curve])
            reached = steps[success >= 0.5]
            final = curve[-1]
            results.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "final_success": float(final["success"]),
                    "auc": float(np.trapezoid(success, steps) / steps[-1]),
                    "steps_to_50": float(reached[0]) if len(reached) else np.nan,
                    "final_discovery": float(final["discovery"]),
                    "final_precision": float(final["precision"]),
                    "stage0_norm_mean": float(final["stage0_norm_mean"]),
                    "stage0_angle_error_deg": float(final["stage0_angle_error_deg"]),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-path", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    curves = []
    for path in args.curve_path:
        curves.extend(read_csv(path))
    per_run = per_run_rows(curves, seeds)
    write_csv(args.output_dir / f"{args.output_prefix}_per_run.csv", per_run)

    grouped = {
        condition.name: sorted(
            [row for row in per_run if row["condition"] == condition.name],
            key=lambda row: row["seed"],
        )
        for condition in OFFPOLICY_CONDITIONS
    }
    rng = np.random.default_rng(20260926)
    paired = []

    def contrast(treatment, control, metric, lower_is_better=False):
        treated = np.asarray([row[metric] for row in grouped[treatment]])
        baseline = np.asarray([row[metric] for row in grouped[control]])
        differences = baseline - treated if lower_is_better else treated - baseline
        if not np.all(np.isfinite(differences)):
            low = high = np.nan
        else:
            low, high = bootstrap_interval(differences, rng)
        paired.append(
            {
                "contrast": f"{treatment} - {control}",
                "metric": metric,
                "benefit_direction": "lower" if lower_is_better else "higher",
                "mean_benefit": float(np.nanmean(differences)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "num_positive": int(np.sum(differences > 0)),
                "num_tied": int(np.sum(differences == 0)),
                "paired_differences": ";".join(f"{x:.8f}" for x in differences),
            }
        )
        return differences, low, high

    primary_auc, primary_auc_low, _ = contrast(
        "DS stage", "Cartesian full-cov oracle", "auc"
    )
    primary_final, _, _ = contrast(
        "DS stage", "Cartesian full-cov oracle", "final_success"
    )
    primary_steps, _, _ = contrast(
        "DS stage", "Cartesian full-cov oracle", "steps_to_50", True
    )
    cart_auc, _, _ = contrast("DS stage", "Cartesian stage", "auc")
    global_auc, _, _ = contrast("DS stage", "DS global", "auc")
    coupled_auc, _, _ = contrast(
        "DS stage", "DS stage, coupled radius", "auc"
    )
    write_csv(args.output_dir / f"{args.output_prefix}_paired.csv", paired)

    oracle_steps = np.asarray(
        [row["steps_to_50"] for row in grouped["Cartesian full-cov oracle"]]
    )
    ds_steps = np.asarray([row["steps_to_50"] for row in grouped["DS stage"]])
    finite_steps = np.isfinite(oracle_steps) & np.isfinite(ds_steps)
    if finite_steps.all():
        step_reduction = float(
            (oracle_steps.mean() - ds_steps.mean()) / oracle_steps.mean()
        )
    else:
        step_reduction = np.nan
    positive_threshold = int(np.ceil(0.75 * len(seeds)))
    means = {
        name: float(np.mean([row["final_success"] for row in values]))
        for name, values in grouped.items()
    }
    decisions = [
        ("ds_final_success_at_least_50pct", means["DS stage"], 0.50, ">="),
        (
            "oracle_final_success_at_least_50pct",
            means["Cartesian full-cov oracle"],
            0.50,
            ">=",
        ),
        ("primary_auc_delta_at_least_1pp", primary_auc.mean(), 0.01, ">="),
        ("primary_auc_bootstrap_lower_positive", primary_auc_low, 0.0, ">"),
        (
            "primary_auc_at_least_75pct_positive_pairs",
            np.sum(primary_auc > 0),
            positive_threshold,
            ">=",
        ),
        ("steps_to_50_reduction_at_least_5pct", step_reduction, 0.05, ">="),
        (
            "steps_to_50_at_least_75pct_no_later",
            np.sum(primary_steps >= 0),
            positive_threshold,
            ">=",
        ),
        (
            "final_success_noninferiority_margin_15pp",
            primary_final.mean(),
            -0.15,
            ">=",
        ),
        ("cartesian_stage_auc_delta_at_least_15pp", cart_auc.mean(), 0.15, ">="),
        ("global_ablation_auc_delta_at_least_3pp", global_auc.mean(), 0.03, ">="),
        ("coupled_ablation_auc_delta_at_least_5pp", coupled_auc.mean(), 0.05, ">="),
    ]
    decision_rows = []
    for criterion, value, threshold, operator in decisions:
        passed = value > threshold if operator == ">" else value >= threshold
        decision_rows.append(
            {
                "criterion": criterion,
                "value": float(value),
                "threshold": threshold,
                "operator": operator,
                "passed": bool(passed),
            }
        )
    write_csv(args.output_dir / f"{args.output_prefix}_decision.csv", decision_rows)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    pairs = (
        ("auc", "Success AUC (%)", 100.0),
        ("steps_to_50", "Interactions to 50% (k)", 1 / 1000),
        ("final_success", "Final success (%)", 100.0),
    )
    for axis, (metric, label, scale) in zip(axes, pairs):
        left = np.asarray([row[metric] for row in grouped["Cartesian full-cov oracle"]])
        right = np.asarray([row[metric] for row in grouped["DS stage"]])
        for index in range(len(seeds)):
            axis.plot((0, 1), (left[index] * scale, right[index] * scale), color="#999", alpha=0.5)
        axis.scatter(np.zeros(len(seeds)), left * scale, color="#d62728", s=20)
        axis.scatter(np.ones(len(seeds)), right * scale, color="#2ca02c", s=20)
        axis.set_xticks((0, 1), ("Full-cov", "DS-stage"))
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.output_prefix}_paired.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = plt.get_cmap("tab10")
    for index, condition in enumerate(OFFPOLICY_CONDITIONS):
        selected = [row for row in curves if row["condition"] == condition.name]
        steps = sorted({int(row["stage0_interactions"]) for row in selected})
        means, standard_deviations = [], []
        for step in steps:
            values = np.asarray(
                [
                    float(row["success"])
                    for row in selected
                    if int(row["stage0_interactions"]) == step
                ]
            )
            means.append(100.0 * values.mean())
            standard_deviations.append(100.0 * values.std(ddof=1))
        x = np.asarray(steps) / 1000.0
        mean = np.asarray(means)
        deviation = np.asarray(standard_deviations)
        axis.plot(x, mean, label=condition.name, color=colors(index), linewidth=2)
        axis.fill_between(
            x,
            np.maximum(0.0, mean - deviation),
            mean + deviation,
            color=colors(index),
            alpha=0.10,
        )
    axis.set_xlabel("Stage-0 interactions (k)")
    axis.set_ylabel("Success rate (%)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.output_prefix}_curves.png", dpi=200)
    plt.close(figure)

    for row in paired:
        print(row)
    for row in decision_rows:
        print(row)


if __name__ == "__main__":
    main()
