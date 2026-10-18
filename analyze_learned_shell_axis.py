#!/usr/bin/env python3
"""Equivalence analysis for the axis-aligned learned-shell counterexample."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from analyze_learned_shell_formal import (
    bootstrap_interval,
    load_per_run,
    write_csv,
)


EXPECTED_SEEDS = tuple(range(200, 232))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve-path",
        type=Path,
        default=(reports_dir() / "learned_shell_axis_formal_per_seed.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_run = load_per_run(args.curve_path, expected_seeds=EXPECTED_SEEDS)
    grouped = {}
    for row in per_run:
        grouped.setdefault(row["condition"], []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["seed"])

    ds_auc = np.asarray([row["auc"] for row in grouped["DS stage"]])
    oracle_auc = np.asarray(
        [row["auc"] for row in grouped["Cartesian full-cov oracle"]]
    )
    cartesian_auc = np.asarray(
        [row["auc"] for row in grouped["Cartesian stage"]]
    )
    ds_steps = np.asarray([row["steps_to_50"] for row in grouped["DS stage"]])
    oracle_steps = np.asarray(
        [row["steps_to_50"] for row in grouped["Cartesian full-cov oracle"]]
    )
    differences = ds_auc - oracle_auc
    low, high = bootstrap_interval(
        differences, np.random.default_rng(20260927), count=50_000
    )
    paired = [
        {
            "contrast": "DS stage - Cartesian full-cov oracle",
            "metric": "auc",
            "mean_delta": float(differences.mean()),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "num_positive": int(np.sum(differences > 0)),
            "num_negative": int(np.sum(differences < 0)),
            "paired_differences": ";".join(
                f"{value:.8f}" for value in differences
            ),
        }
    ]
    write_csv(args.output_dir / "learned_shell_axis_formal_paired.csv", paired)

    structured_margin = min(
        float((ds_auc - cartesian_auc).mean()),
        float((oracle_auc - cartesian_auc).mean()),
    )
    mean_step_difference = float(abs(ds_steps.mean() - oracle_steps.mean()))
    decision = [
        {
            "criterion": "absolute_mean_auc_delta_at_most_0_5pp",
            "value": abs(float(differences.mean())),
            "threshold": 0.005,
            "passed": bool(abs(differences.mean()) <= 0.005),
        },
        {
            "criterion": "bootstrap_interval_inside_plus_minus_1pp",
            "value": f"[{low:.8f},{high:.8f}]",
            "threshold": "[-0.01,0.01]",
            "passed": bool(low >= -0.01 and high <= 0.01),
        },
        {
            "criterion": "mean_steps_difference_at_most_one_eval_interval",
            "value": mean_step_difference,
            "threshold": 20480,
            "passed": bool(mean_step_difference <= 20480),
        },
        {
            "criterion": "both_structured_methods_exceed_cartesian_by_15pp",
            "value": structured_margin,
            "threshold": 0.15,
            "passed": bool(structured_margin >= 0.15),
        },
    ]
    write_csv(args.output_dir / "learned_shell_axis_formal_decision.csv", decision)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for index in range(len(EXPECTED_SEEDS)):
        axes[0].plot(
            [0, 1],
            [100 * oracle_auc[index], 100 * ds_auc[index]],
            color="#888888",
            alpha=0.45,
        )
        axes[1].plot(
            [0, 1],
            [oracle_steps[index] / 1000, ds_steps[index] / 1000],
            color="#888888",
            alpha=0.45,
        )
    axes[0].scatter(np.zeros(len(EXPECTED_SEEDS)), 100 * oracle_auc, color="#d62728", s=18)
    axes[0].scatter(np.ones(len(EXPECTED_SEEDS)), 100 * ds_auc, color="#2ca02c", s=18)
    axes[0].set_ylabel("Normalized success AUC (%)")
    axes[1].scatter(np.zeros(len(EXPECTED_SEEDS)), oracle_steps / 1000, color="#d62728", s=18)
    axes[1].scatter(np.ones(len(EXPECTED_SEEDS)), ds_steps / 1000, color="#2ca02c", s=18)
    axes[1].set_ylabel("Stage-0 interactions to 50% (k)")
    for axis in axes:
        axis.set_xticks([0, 1], ("Full-cov oracle", "DS stage"))
        axis.grid(alpha=0.22)
    figure.suptitle("Axis-aligned counterexample: structured policies are equivalent")
    figure.tight_layout()
    figure.savefig(args.output_dir / "learned_shell_axis_formal_paired.png", dpi=220)
    plt.close(figure)

    print(paired[0])
    for row in decision:
        print(row)


if __name__ == "__main__":
    main()
