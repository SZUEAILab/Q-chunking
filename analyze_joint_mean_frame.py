#!/usr/bin/env python3
"""Paired analysis for the Cartesian mean-frame counterfactual."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("Cartesian mean-frame", "DS global")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def interval(values, rng, count=50_000):
    values = np.asarray(values)
    indices = rng.integers(0, len(values), (count, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), (0.025, 0.975)))


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
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Counterfactual condition set is incomplete")
    updates = sorted({int(row["update"]) for row in rows})
    if updates[-1] != args.final_update:
        raise ValueError("Final checkpoint is incomplete")

    per_seed = []
    curves = {}
    for condition in CONDITIONS:
        found = sorted({int(row["seed"]) for row in rows if row["condition"] == condition})
        if tuple(found) != seeds:
            raise ValueError(f"{condition}: unexpected seeds {found}")
        for seed in seeds:
            selected = [row for row in rows if row["condition"] == condition and int(row["seed"]) == seed]
            trajectory = []
            for update in updates:
                values = [float(row["success"]) for row in selected if row["split"] == "test" and int(row["update"]) == update]
                trajectory.append(float(np.mean(values)))
            final = [row for row in selected if row["split"] == "test" and int(row["update"]) == args.final_update]
            per_seed.append({
                "condition": condition,
                "seed": seed,
                "test_success": trajectory[-1],
                "heldout_auc": float(np.trapezoid(trajectory, updates) / args.final_update),
                "test_mean_error": float(np.mean([float(row["mean_error"]) for row in final])),
            })
            curves[(condition, seed)] = trajectory

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / f"{args.output_prefix}_per_seed.csv", per_seed)
    grouped = {
        condition: sorted([row for row in per_seed if row["condition"] == condition], key=lambda row: row["seed"])
        for condition in CONDITIONS
    }
    rng = np.random.default_rng(20260925)
    paired = []
    differences = {}
    for metric in ("test_success", "heldout_auc"):
        ds = np.asarray([row[metric] for row in grouped["DS global"]])
        control = np.asarray([row[metric] for row in grouped["Cartesian mean-frame"]])
        diff = ds - control
        low, high = interval(diff, rng)
        differences[metric] = (diff, low, high)
        paired.append({
            "metric": metric,
            "mean_difference": float(diff.mean()),
            "bootstrap_ci_low": float(low),
            "bootstrap_ci_high": float(high),
            "num_positive": int(np.sum(diff > 0)),
            "num_tied": int(np.sum(diff == 0)),
        })
    write_rows(args.output_dir / f"{args.output_prefix}_paired.csv", paired)

    angles = sorted({float(row["context_angle_deg"]) for row in rows if row["split"] == "test" and int(row["update"]) == args.final_update})
    per_angle = []
    wins = 0
    for angle in angles:
        item = {"context_angle_deg": angle}
        for condition in CONDITIONS:
            item[condition] = float(np.mean([
                float(row["success"]) for row in rows
                if row["condition"] == condition and row["split"] == "test"
                and int(row["update"]) == args.final_update
                and float(row["context_angle_deg"]) == angle
            ]))
        wins += item["DS global"] > item["Cartesian mean-frame"]
        per_angle.append(item)
    write_rows(args.output_dir / f"{args.output_prefix}_per_angle.csv", per_angle)

    final_diff, final_low, _ = differences["test_success"]
    auc_diff, auc_low, _ = differences["heldout_auc"]
    ds_error = np.mean([row["test_mean_error"] for row in grouped["DS global"]])
    control_success = np.mean([row["test_success"] for row in grouped["Cartesian mean-frame"]])
    criteria = (
        ("final_delta_at_least_10pp", final_diff.mean(), 0.10, ">="),
        ("final_ci_lower_above_8pp", final_low, 0.08, ">"),
        ("positive_pairs_at_least_87p5pct", np.sum(final_diff > 0), int(np.ceil(0.875 * len(seeds))), ">="),
        ("auc_delta_at_least_10pp", auc_diff.mean(), 0.10, ">="),
        ("auc_ci_lower_above_8pp", auc_low, 0.08, ">"),
        ("ds_mean_error_at_most_0p03", ds_error, 0.03, "<="),
        ("angle_wins_at_least_30", wins, 30, ">="),
        ("control_success_at_least_25pct", control_success, 0.25, ">="),
    )
    decisions = []
    for name, value, threshold, operator in criteria:
        passed = value > threshold if operator == ">" else value <= threshold if operator == "<=" else value >= threshold
        decisions.append({"criterion": name, "value": float(value), "threshold": threshold, "operator": operator, "passed": bool(passed)})
    write_rows(args.output_dir / f"{args.output_prefix}_decision.csv", decisions)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for index in range(len(seeds)):
        axes[0].plot((0, 1), (100 * grouped["Cartesian mean-frame"][index]["test_success"], 100 * grouped["DS global"][index]["test_success"]), color="#999999", alpha=0.45)
    axes[0].set_xticks((0, 1), ("Cartesian mean-frame", "DS global"))
    axes[0].set_ylabel("Final held-out success (%)")
    axes[0].grid(alpha=0.2)
    for condition in CONDITIONS:
        mean_curve = np.mean([curves[(condition, seed)] for seed in seeds], axis=0)
        axes[1].plot(updates, 100 * mean_curve, marker="o", label=condition)
        axes[2].plot(angles, 100 * np.asarray([row[condition] for row in per_angle]), marker="o", label=condition)
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
