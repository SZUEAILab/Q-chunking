#!/usr/bin/env python3
"""Equivalence analysis for projected-tangent Cartesian flow versus DS."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


FLOW = "Cartesian projected-tangent flow"
DS = "DS global"


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
    parser.add_argument("--final-update", type=int, default=2400)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    rows = read_rows(args.input)
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    if {row["condition"] for row in rows} != {FLOW, DS}:
        raise ValueError("Expected exactly projected-tangent flow and DS global")
    updates = sorted({int(row["update"]) for row in rows})
    if updates[-1] != args.final_update:
        raise ValueError("Final checkpoint missing")

    per_seed = []
    curves = {}
    for condition in (FLOW, DS):
        found = tuple(sorted({int(row["seed"]) for row in rows if row["condition"] == condition}))
        if found != seeds:
            raise ValueError(f"{condition}: expected {seeds}, found {found}")
        for seed in seeds:
            selected = [row for row in rows if row["condition"] == condition and int(row["seed"]) == seed]
            curve = []
            for update in updates:
                values = [float(row["success"]) for row in selected if row["split"] == "test" and int(row["update"]) == update]
                curve.append(float(np.mean(values)))
            final = [row for row in selected if row["split"] == "test" and int(row["update"]) == args.final_update]
            per_seed.append({
                "condition": condition,
                "seed": seed,
                "test_success": curve[-1],
                "heldout_auc": float(np.trapezoid(curve, updates) / args.final_update),
                "test_mean_error": float(np.mean([float(row["mean_error"]) for row in final])),
            })
            curves[(condition, seed)] = curve
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / f"{args.output_prefix}_per_seed.csv", per_seed)
    grouped = {
        condition: sorted([row for row in per_seed if row["condition"] == condition], key=lambda row: row["seed"])
        for condition in (FLOW, DS)
    }

    rng = np.random.default_rng(20260925)
    paired = []
    diffs = {}
    for metric in ("test_success", "heldout_auc", "test_mean_error"):
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

    angles = sorted({float(row["context_angle_deg"]) for row in rows if row["split"] == "test" and int(row["update"]) == args.final_update})
    angle_rows = []
    for angle in angles:
        item = {"context_angle_deg": angle}
        for condition in (FLOW, DS):
            item[condition] = float(np.mean([
                float(row["success"]) for row in rows
                if row["condition"] == condition and row["split"] == "test"
                and int(row["update"]) == args.final_update
                and float(row["context_angle_deg"]) == angle
            ]))
        item["DS_minus_flow"] = item[DS] - item[FLOW]
        angle_rows.append(item)
    write_rows(args.output_dir / f"{args.output_prefix}_per_angle.csv", angle_rows)

    final_diff, final_low, final_high = diffs["test_success"]
    auc_diff, auc_low, auc_high = diffs["heldout_auc"]
    error_diff = diffs["test_mean_error"][0]
    final_means = {condition: np.mean([row["test_success"] for row in grouped[condition]]) for condition in (FLOW, DS)}
    angle_abs = np.abs([row["DS_minus_flow"] for row in angle_rows])
    criteria = (
        ("absolute_final_mean_at_most_0p5pp", abs(final_diff.mean()), 0.005, "<="),
        ("final_ci_low_above_minus_1pp", final_low, -0.01, ">="),
        ("final_ci_high_below_plus_1pp", final_high, 0.01, "<="),
        ("absolute_auc_mean_at_most_0p5pp", abs(auc_diff.mean()), 0.005, "<="),
        ("auc_ci_low_above_minus_1pp", auc_low, -0.01, ">="),
        ("auc_ci_high_below_plus_1pp", auc_high, 0.01, "<="),
        ("flow_final_at_least_55pct", final_means[FLOW], 0.55, ">="),
        ("ds_final_at_least_55pct", final_means[DS], 0.55, ">="),
        ("absolute_mean_error_difference_at_most_0p003", abs(error_diff.mean()), 0.003, "<="),
        ("mean_absolute_angle_difference_at_most_1p5pp", np.mean(angle_abs), 0.015, "<="),
        ("max_absolute_angle_difference_at_most_4pp", np.max(angle_abs), 0.04, "<="),
    )
    decisions = []
    for name, value, threshold, operator in criteria:
        passed = value <= threshold if operator == "<=" else value >= threshold
        decisions.append({"criterion": name, "value": float(value), "threshold": threshold, "operator": operator, "passed": bool(passed)})
    write_rows(args.output_dir / f"{args.output_prefix}_decision.csv", decisions)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for index in range(len(seeds)):
        axes[0].plot((0, 1), (100 * grouped[FLOW][index]["test_success"], 100 * grouped[DS][index]["test_success"]), color="#999999", alpha=0.45)
    axes[0].set_xticks((0, 1), ("Projected-tangent flow", "DS global"))
    axes[0].set_ylabel("Final held-out success (%)")
    axes[0].grid(alpha=0.2)
    for condition in (FLOW, DS):
        curve = np.mean([curves[(condition, seed)] for seed in seeds], axis=0)
        axes[1].plot(updates, 100 * curve, marker="o", label=condition)
        axes[2].plot(angles, 100 * np.asarray([row[condition] for row in angle_rows]), marker="o", label=condition)
    axes[1].set(xlabel="Policy updates", ylabel="Held-out success (%)")
    axes[2].set(xlabel="Held-out context angle (deg)", ylabel="Success (%)")
    for axis in axes[1:]:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.output_prefix}.png", dpi=200)
    plt.close(figure)
    for row in paired + decisions:
        print(row)


if __name__ == "__main__":
    main()
