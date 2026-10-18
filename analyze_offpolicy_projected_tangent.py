#!/usr/bin/env python3
"""Equivalence analysis for DS and projected-tangent flow under twin-Q/AWR."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


DS = "DS stage"
FLOW = "Cartesian projected-tangent stage"


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, rng, count=50_000):
    values = np.asarray(values)
    indices = rng.integers(0, len(values), (count, len(values)))
    return tuple(float(x) for x in np.quantile(values[indices].mean(axis=1), (0.025, 0.975)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    rows = read_rows(args.input)
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    if {row["condition"] for row in rows} != {DS, FLOW}:
        raise ValueError("Expected exactly DS stage and projected-tangent stage")
    per_run = []
    curves = {}
    for condition in (FLOW, DS):
        found = tuple(sorted({int(row["seed"]) for row in rows if row["condition"] == condition}))
        if found != seeds:
            raise ValueError(f"{condition}: expected {seeds}, found {found}")
        for seed in seeds:
            curve = sorted(
                [row for row in rows if row["condition"] == condition and int(row["seed"]) == seed],
                key=lambda row: int(row["stage0_interactions"]),
            )
            steps = np.asarray([int(row["stage0_interactions"]) for row in curve])
            success = np.asarray([float(row["success"]) for row in curve])
            reached = steps[success >= 0.5]
            final = curve[-1]
            per_run.append({
                "condition": condition,
                "seed": seed,
                "final_success": float(final["success"]),
                "auc": float(np.trapezoid(success, steps) / steps[-1]),
                "steps_to_50": float(reached[0]) if len(reached) else np.nan,
                "final_discovery": float(final["discovery"]),
                "final_precision": float(final["precision"]),
                "stage0_norm_mean": float(final["stage0_norm_mean"]),
                "stage1_norm_mean": float(final["stage1_norm_mean"]),
            })
            curves[(condition, seed)] = (steps, success)
    if not all(np.isfinite(float(row[key])) for row in per_run for key in ("final_success", "auc", "steps_to_50", "final_precision")):
        raise ValueError("Non-finite required metric")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / f"{args.output_prefix}_per_run.csv", per_run)
    grouped = {
        condition: sorted([row for row in per_run if row["condition"] == condition], key=lambda row: row["seed"])
        for condition in (FLOW, DS)
    }
    rng = np.random.default_rng(20260926)
    paired = []
    diffs = {}
    for metric in ("auc", "final_success", "steps_to_50", "final_precision", "final_discovery"):
        diff = np.asarray([row[metric] for row in grouped[DS]]) - np.asarray([row[metric] for row in grouped[FLOW]])
        low, high = bootstrap(diff, rng)
        diffs[metric] = (diff, low, high)
        paired.append({
            "metric": metric,
            "mean_difference": float(diff.mean()),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "num_positive": int(np.sum(diff > 0)),
            "num_tied": int(np.sum(diff == 0)),
        })
    write_rows(args.output_dir / f"{args.output_prefix}_paired.csv", paired)

    auc, auc_low, auc_high = diffs["auc"]
    final, final_low, final_high = diffs["final_success"]
    t50, t50_low, t50_high = diffs["steps_to_50"]
    precision = diffs["final_precision"][0]
    final_means = {c: np.mean([row["final_success"] for row in grouped[c]]) for c in (FLOW, DS)}
    criteria = (
        ("absolute_auc_difference_at_most_1pp", abs(auc.mean()), 0.01, "<="),
        ("auc_ci_low_above_minus_1p5pp", auc_low, -0.015, ">="),
        ("auc_ci_high_below_plus_1p5pp", auc_high, 0.015, "<="),
        ("absolute_final_difference_at_most_3pp", abs(final.mean()), 0.03, "<="),
        ("final_ci_low_above_minus_4pp", final_low, -0.04, ">="),
        ("final_ci_high_below_plus_4pp", final_high, 0.04, "<="),
        ("flow_final_at_least_60pct", final_means[FLOW], 0.60, ">="),
        ("ds_final_at_least_60pct", final_means[DS], 0.60, ">="),
        ("all_runs_reach_50pct", sum(np.isfinite(row["steps_to_50"]) for row in per_run), 2 * len(seeds), ">="),
        ("absolute_t50_difference_at_most_15k", abs(t50.mean()), 15_000, "<="),
        ("t50_ci_low_above_minus_20k", t50_low, -20_000, ">="),
        ("t50_ci_high_below_plus_20k", t50_high, 20_000, "<="),
        ("absolute_precision_difference_at_most_1pp", abs(precision.mean()), 0.01, "<="),
    )
    decisions = []
    for name, value, threshold, operator in criteria:
        passed = value <= threshold if operator == "<=" else value >= threshold
        decisions.append({"criterion": name, "value": float(value), "threshold": threshold, "operator": operator, "passed": bool(passed)})
    write_rows(args.output_dir / f"{args.output_prefix}_decision.csv", decisions)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    metrics = (("auc", "Success AUC (%)", 100), ("steps_to_50", "Interactions to 50% (k)", 0.001), ("final_success", "Final success (%)", 100))
    for axis, (metric, label, scale) in zip(axes, metrics):
        left = np.asarray([row[metric] for row in grouped[FLOW]]) * scale
        right = np.asarray([row[metric] for row in grouped[DS]]) * scale
        for index in range(len(seeds)):
            axis.plot((0, 1), (left[index], right[index]), color="#999999", alpha=0.45)
        axis.set_xticks((0, 1), ("Projected flow", "DS stage"))
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.output_prefix}_paired.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for condition in (FLOW, DS):
        steps = curves[(condition, seeds[0])][0]
        values = np.asarray([curves[(condition, seed)][1] for seed in seeds])
        axis.plot(steps / 1000, 100 * values.mean(axis=0), label=condition)
        axis.fill_between(steps / 1000, 100 * (values.mean(0) - values.std(0, ddof=1)), 100 * (values.mean(0) + values.std(0, ddof=1)), alpha=0.12)
    axis.set(xlabel="Stage-0 interactions (k)", ylabel="Success rate (%)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.output_prefix}_curves.png", dpi=200)
    plt.close(figure)
    for row in paired + decisions:
        print(row)


if __name__ == "__main__":
    main()
