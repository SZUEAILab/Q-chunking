#!/usr/bin/env python3
"""Formal paired analysis for joint contextual mean--covariance learning."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from train_joint_contextual_policy import CONDITIONS


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, rng, count=50_000):
    values = np.asarray(values)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    estimates = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--final-update", type=int, default=2400)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    rows = read_csv(args.input)
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    expected = {condition.name for condition in CONDITIONS}
    if {row["condition"] for row in rows} != expected:
        raise ValueError("Condition set does not match frozen protocol")
    updates = sorted({int(row["update"]) for row in rows})
    if updates[-1] != args.final_update:
        raise ValueError(f"Expected final update {args.final_update}, found {updates[-1]}")

    per_seed = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition.name]
        found = tuple(sorted({int(row["seed"]) for row in selected}))
        if found != seeds:
            raise ValueError(f"{condition.name}: expected {seeds}, found {found}")
        for seed in seeds:
            seed_rows = [row for row in selected if int(row["seed"]) == seed]
            final_train = [
                row
                for row in seed_rows
                if int(row["update"]) == args.final_update and row["split"] == "train"
            ]
            final_test = [
                row
                for row in seed_rows
                if int(row["update"]) == args.final_update and row["split"] == "test"
            ]
            train_success = float(np.mean([float(row["success"]) for row in final_train]))
            test_success = float(np.mean([float(row["success"]) for row in final_test]))
            trajectory = []
            for update in updates:
                values = [
                    float(row["success"])
                    for row in seed_rows
                    if int(row["update"]) == update and row["split"] == "test"
                ]
                trajectory.append(float(np.mean(values)))
            per_seed.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "train_success": train_success,
                    "test_success": test_success,
                    "heldout_auc": float(
                        np.trapezoid(trajectory, updates) / args.final_update
                    ),
                    "generalization_gap": train_success - test_success,
                    "test_mean_error": float(
                        np.mean([float(row["mean_error"]) for row in final_test])
                    ),
                    "worst_test_success": float(
                        min(float(row["success"]) for row in final_test)
                    ),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{args.output_prefix}_per_seed.csv", per_seed)
    grouped = {
        condition.name: sorted(
            [row for row in per_seed if row["condition"] == condition.name],
            key=lambda row: row["seed"],
        )
        for condition in CONDITIONS
    }
    rng = np.random.default_rng(20260928)
    paired = []

    def contrast(treatment, control, metric="test_success"):
        left = np.asarray([row[metric] for row in grouped[treatment]])
        right = np.asarray([row[metric] for row in grouped[control]])
        differences = left - right
        low, high = bootstrap(differences, rng)
        paired.append(
            {
                "contrast": f"{treatment} - {control}",
                "metric": metric,
                "mean_difference": float(differences.mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "num_positive": int(np.sum(differences > 0)),
                "num_tied": int(np.sum(differences == 0)),
                "paired_differences": ";".join(f"{value:.8f}" for value in differences),
            }
        )
        return differences, low, high

    primary, primary_low, _ = contrast("DS global", "Cartesian full-cov state")
    primary_auc, auc_low, _ = contrast(
        "DS global", "Cartesian full-cov state", "heldout_auc"
    )
    oracle, oracle_low, _ = contrast("DS global", "Cartesian rotating oracle")
    full_diagonal, _, _ = contrast(
        "Cartesian full-cov state", "Cartesian diagonal state"
    )
    state_ablation, _, _ = contrast("DS global", "DS state")
    coupled_ablation, _, _ = contrast("DS global", "DS coupled")
    write_csv(args.output_dir / f"{args.output_prefix}_paired.csv", paired)

    heldout_angles = sorted(
        {
            float(row["context_angle_deg"])
            for row in rows
            if row["split"] == "test" and int(row["update"]) == args.final_update
        }
    )
    angle_rows = []
    angle_wins = 0
    for angle in heldout_angles:
        item = {"context_angle_deg": angle}
        for condition in CONDITIONS:
            values = [
                float(row["success"])
                for row in rows
                if row["condition"] == condition.name
                and row["split"] == "test"
                and int(row["update"]) == args.final_update
                and float(row["context_angle_deg"]) == angle
            ]
            item[condition.name] = float(np.mean(values))
        angle_wins += item["DS global"] > item["Cartesian full-cov state"]
        angle_rows.append(item)
    write_csv(
        args.output_dir / f"{args.output_prefix}_per_angle_summary.csv", angle_rows
    )

    ds_gap = np.mean([row["generalization_gap"] for row in grouped["DS global"]])
    ds_error = np.mean([row["test_mean_error"] for row in grouped["DS global"]])
    positive_threshold = int(np.ceil(0.875 * len(seeds)))
    decisions = [
        ("primary_delta_at_least_10pp", primary.mean(), 0.10, ">="),
        ("primary_ci_lower_above_8pp", primary_low, 0.08, ">"),
        ("primary_at_least_28_of_32_positive", np.sum(primary > 0), positive_threshold, ">="),
        ("auc_delta_at_least_10pp", primary_auc.mean(), 0.10, ">="),
        ("auc_ci_lower_above_8pp", auc_low, 0.08, ">"),
        ("oracle_noninferiority_mean_margin_2pp", oracle.mean(), -0.02, ">="),
        ("oracle_noninferiority_ci_margin_2pp", oracle_low, -0.02, ">="),
        ("ds_test_mean_error_at_most_0p03", ds_error, 0.03, "<="),
        ("ds_generalization_gap_at_most_20pp", ds_gap, 0.20, "<="),
        ("full_cov_beats_diagonal_by_5pp", full_diagonal.mean(), 0.05, ">="),
        ("ds_global_beats_ds_state_by_10pp", state_ablation.mean(), 0.10, ">="),
        ("ds_global_beats_coupled_by_30pp", coupled_ablation.mean(), 0.30, ">="),
        ("ds_wins_at_least_30_of_36_angles", angle_wins, 30, ">="),
    ]
    decision_rows = []
    for criterion, value, threshold, operator in decisions:
        if operator == ">":
            passed = value > threshold
        elif operator == "<=":
            passed = value <= threshold
        else:
            passed = value >= threshold
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

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for index in range(len(seeds)):
        axes[0].plot(
            (0, 1),
            (
                100 * grouped["Cartesian full-cov state"][index]["test_success"],
                100 * grouped["DS global"][index]["test_success"],
            ),
            color="#999999",
            alpha=0.4,
        )
    axes[0].scatter(
        np.zeros(len(seeds)),
        [100 * row["test_success"] for row in grouped["Cartesian full-cov state"]],
        s=18,
    )
    axes[0].scatter(
        np.ones(len(seeds)),
        [100 * row["test_success"] for row in grouped["DS global"]],
        s=18,
    )
    axes[0].set_xticks((0, 1), ("Full-cov state", "DS global"))
    axes[0].set_ylabel("Final held-out success (%)")
    axes[0].grid(alpha=0.2)

    for condition_name in (
        "Cartesian full-cov state",
        "Cartesian rotating oracle",
        "DS global",
        "DS state",
    ):
        curve = []
        for update in updates:
            values = [
                float(row["success"])
                for row in rows
                if row["condition"] == condition_name
                and row["split"] == "test"
                and int(row["update"]) == update
            ]
            curve.append(100 * np.mean(values))
        axes[1].plot(updates, curve, marker="o", label=condition_name)
    axes[1].set(xlabel="Policy updates", ylabel="Held-out success (%)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=7)

    for condition_name in (
        "Cartesian full-cov state",
        "Cartesian rotating oracle",
        "DS global",
    ):
        axes[2].plot(
            heldout_angles,
            [100 * row[condition_name] for row in angle_rows],
            marker="o",
            label=condition_name,
        )
    axes[2].set(xlabel="Held-out context angle (deg)", ylabel="Success (%)")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.output_prefix}_paired.png", dpi=200)
    plt.close(figure)

    for row in paired:
        print(row)
    for row in decision_rows:
        print(row)


if __name__ == "__main__":
    main()
