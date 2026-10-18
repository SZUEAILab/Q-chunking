#!/usr/bin/env python3
"""Paired formal analysis for contextual held-out rotations."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from train_contextual_rotation import CONDITIONS


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
    return tuple(float(x) for x in np.quantile(estimates, (0.025, 0.975)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    rows = read_csv(args.input)
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    expected = {condition.name for condition in CONDITIONS}
    if {row["condition"] for row in rows} != expected:
        raise ValueError("Condition set does not match the frozen protocol")

    per_seed = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition.name]
        found = tuple(sorted({int(row["seed"]) for row in condition_rows}))
        if found != seeds:
            raise ValueError(f"{condition.name}: expected {seeds}, found {found}")
        for seed in seeds:
            selected = [row for row in condition_rows if int(row["seed"]) == seed]
            train = np.asarray([float(row["success"]) for row in selected if row["split"] == "train"])
            test = np.asarray([float(row["success"]) for row in selected if row["split"] == "test"])
            per_seed.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "train_success": float(train.mean()),
                    "test_success": float(test.mean()),
                    "generalization_gap": float(train.mean() - test.mean()),
                    "worst_test_angle": float(test.min()),
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
    rng = np.random.default_rng(20260927)
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
                "paired_differences": ";".join(f"{x:.8f}" for x in differences),
            }
        )
        return differences, low, high

    primary, primary_low, _ = contrast("DS", "Cartesian full-cov Fourier")
    oracle, oracle_low, _ = contrast("DS", "Cartesian rotating oracle")
    fourier, _, _ = contrast("Cartesian full-cov Fourier", "Cartesian full-cov constant")
    coupled, _, _ = contrast("DS", "DS coupled")
    write_csv(args.output_dir / f"{args.output_prefix}_paired.csv", paired)

    ds_gap = np.asarray([row["generalization_gap"] for row in grouped["DS"]])
    heldout_angles = sorted(
        {float(row["context_angle_deg"]) for row in rows if row["split"] == "test"}
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
                and float(row["context_angle_deg"]) == angle
            ]
            item[condition.name] = float(np.mean(values))
        angle_wins += item["DS"] > item["Cartesian full-cov Fourier"]
        angle_rows.append(item)
    write_csv(args.output_dir / f"{args.output_prefix}_per_angle_summary.csv", angle_rows)

    positive_threshold = int(np.ceil(0.875 * len(seeds)))
    decisions = [
        ("primary_test_delta_at_least_3pp", primary.mean(), 0.03, ">="),
        ("primary_bootstrap_lower_positive", primary_low, 0.0, ">"),
        ("primary_at_least_28_of_32_positive", np.sum(primary > 0), positive_threshold, ">="),
        ("ds_abs_generalization_gap_at_most_0p5pp", abs(ds_gap.mean()), 0.005, "<="),
        ("oracle_noninferiority_mean_margin_1pp", oracle.mean(), -0.01, ">="),
        ("oracle_noninferiority_ci_margin_1pp", oracle_low, -0.01, ">="),
        ("fourier_beats_constant_by_3pp", fourier.mean(), 0.03, ">="),
        ("ds_beats_coupled_by_8pp", coupled.mean(), 0.08, ">="),
        ("ds_wins_at_least_10_of_12_angles", angle_wins, 10, ">="),
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

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for index in range(len(seeds)):
        axes[0].plot(
            (0, 1),
            (
                100 * grouped["Cartesian full-cov Fourier"][index]["test_success"],
                100 * grouped["DS"][index]["test_success"],
            ),
            color="#999999",
            alpha=0.45,
        )
    axes[0].scatter(
        np.zeros(len(seeds)),
        [100 * row["test_success"] for row in grouped["Cartesian full-cov Fourier"]],
        color="#2ca02c",
        s=18,
    )
    axes[0].scatter(
        np.ones(len(seeds)),
        [100 * row["test_success"] for row in grouped["DS"]],
        color="#9467bd",
        s=18,
    )
    axes[0].set_xticks((0, 1), ("Full-cov Fourier", "DS"))
    axes[0].set_ylabel("Held-out success (%)")
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        heldout_angles,
        [100 * row["Cartesian full-cov Fourier"] for row in angle_rows],
        marker="o",
        label="Full-cov Fourier",
    )
    axes[1].plot(
        heldout_angles,
        [100 * row["Cartesian rotating oracle"] for row in angle_rows],
        marker="o",
        label="Rotating oracle",
    )
    axes[1].plot(
        heldout_angles,
        [100 * row["DS"] for row in angle_rows],
        marker="o",
        label="DS",
    )
    axes[1].set_xlabel("Held-out context angle (deg)")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.output_prefix}_paired.png", dpi=200)
    plt.close(figure)

    for row in paired:
        print(row)
    for row in decision_rows:
        print(row)


if __name__ == "__main__":
    main()
