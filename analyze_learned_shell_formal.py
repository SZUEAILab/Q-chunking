#!/usr/bin/env python3
"""Paired analysis for the formal learned sparse-shell experiment."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from train_synthetic_shell_rl import CONDITIONS, TARGET_ANGLE


EXPECTED_SEEDS = tuple(range(100, 132))


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(differences, rng, count=50_000):
    indices = rng.integers(0, len(differences), size=(count, len(differences)))
    estimates = differences[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(
        np.quantile(estimates, 0.975)
    )


def load_per_run(curve_path, expected_seeds=EXPECTED_SEEDS):
    rows = read_rows(curve_path)
    grouped = {}
    for row in rows:
        key = (row["condition"], int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    per_run = []
    for condition in CONDITIONS:
        seeds = sorted(seed for name, seed in grouped if name == condition.name)
        if tuple(seeds) != tuple(expected_seeds):
            raise ValueError(
                f"Expected {condition.name} seeds {tuple(expected_seeds)}, found {seeds}"
            )
        for seed in seeds:
            curve = sorted(
                grouped[(condition.name, seed)],
                key=lambda row: int(row["stage0_interactions"]),
            )
            steps = np.asarray(
                [int(row["stage0_interactions"]) for row in curve], dtype=float
            )
            success = np.asarray([float(row["success"]) for row in curve])
            reached = steps[success >= 0.5]
            per_run.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "final_success": float(success[-1]),
                    "auc": float(np.trapezoid(success, steps) / steps[-1]),
                    "steps_to_50": float(reached[0]) if len(reached) else np.nan,
                    "final_discovery": float(curve[-1]["discovery"]),
                    "final_precision": float(curve[-1]["precision"]),
                }
            )
    return per_run


def summarize_parameters(parameter_path):
    rows = read_rows(parameter_path)
    numeric_fields = [
        field
        for field in rows[0]
        if field not in ("condition", "seed") and not field.endswith("angular_std")
    ]
    summary = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition.name]
        item = {"condition": condition.name, "num_seeds": len(selected)}
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in selected])
            finite = values[np.isfinite(values)]
            item[f"{field}_mean"] = float(finite.mean()) if len(finite) else np.nan
            item[f"{field}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        angle_errors = np.asarray(
            [abs(float(row["stage0_mean_angle"]) - TARGET_ANGLE) for row in selected]
        )
        item["stage0_mean_angle_error_mean"] = float(angle_errors.mean())
        item["stage0_mean_angle_error_std"] = float(angle_errors.std(ddof=1))
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve-path",
        type=Path,
        default=(reports_dir() / "learned_shell_formal_per_seed.csv"),
    )
    parser.add_argument(
        "--parameter-path",
        type=Path,
        default=(reports_dir() / "learned_shell_formal_parameters.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_run = load_per_run(args.curve_path)
    write_csv(args.output_dir / "learned_shell_formal_per_run.csv", per_run)
    grouped = {
        condition.name: sorted(
            [row for row in per_run if row["condition"] == condition.name],
            key=lambda row: row["seed"],
        )
        for condition in CONDITIONS
    }
    summary = []
    for condition in CONDITIONS:
        item = {
            "condition": condition.name,
            "num_seeds": len(grouped[condition.name]),
        }
        for metric in (
            "final_success",
            "auc",
            "steps_to_50",
            "final_discovery",
            "final_precision",
        ):
            values = np.asarray([row[metric] for row in grouped[condition.name]])
            item[f"{metric}_mean"] = float(np.nanmean(values))
            item[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        summary.append(item)
    write_csv(args.output_dir / "learned_shell_formal_summary.csv", summary)

    rng = np.random.default_rng(20260926)
    paired = []

    def add_contrast(treatment, control, metric, lower_is_better=False):
        treatment_values = np.asarray(
            [row[metric] for row in grouped[treatment]], dtype=float
        )
        control_values = np.asarray(
            [row[metric] for row in grouped[control]], dtype=float
        )
        differences = (
            control_values - treatment_values
            if lower_is_better
            else treatment_values - control_values
        )
        low, high = bootstrap_interval(differences, rng)
        paired.append(
            {
                "contrast": f"{treatment} - {control}",
                "metric": metric,
                "benefit_direction": "lower" if lower_is_better else "higher",
                "mean_benefit": float(differences.mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "num_positive": int(np.sum(differences > 0)),
                "num_tied": int(np.sum(differences == 0)),
                "paired_differences": ";".join(
                    f"{value:.8f}" for value in differences
                ),
            }
        )
        return differences, low, high

    primary_auc, primary_low, _ = add_contrast(
        "DS stage", "Cartesian full-cov oracle", "auc"
    )
    primary_steps, _, _ = add_contrast(
        "DS stage",
        "Cartesian full-cov oracle",
        "steps_to_50",
        lower_is_better=True,
    )
    cartesian_auc, _, _ = add_contrast(
        "DS stage", "Cartesian stage", "auc"
    )
    global_auc, _, _ = add_contrast("DS stage", "DS global", "auc")
    coupled_auc, _, _ = add_contrast(
        "DS stage", "DS stage, coupled radius", "auc"
    )
    write_csv(args.output_dir / "learned_shell_formal_paired.csv", paired)

    oracle_steps = np.asarray(
        [row["steps_to_50"] for row in grouped["Cartesian full-cov oracle"]]
    )
    ds_steps = np.asarray([row["steps_to_50"] for row in grouped["DS stage"]])
    step_reduction = float((oracle_steps.mean() - ds_steps.mean()) / oracle_steps.mean())
    decision = [
        {
            "criterion": "primary_auc_delta_at_least_3pp",
            "value": float(primary_auc.mean()),
            "threshold": 0.03,
            "passed": bool(primary_auc.mean() >= 0.03),
        },
        {
            "criterion": "primary_auc_bootstrap_lower_positive",
            "value": primary_low,
            "threshold": 0.0,
            "passed": bool(primary_low > 0.0),
        },
        {
            "criterion": "primary_auc_at_least_28_positive_pairs",
            "value": int(np.sum(primary_auc > 0)),
            "threshold": 28,
            "passed": bool(np.sum(primary_auc > 0) >= 28),
        },
        {
            "criterion": "steps_to_50_reduction_at_least_15pct",
            "value": step_reduction,
            "threshold": 0.15,
            "passed": bool(step_reduction >= 0.15),
        },
        {
            "criterion": "steps_to_50_at_least_28_no_later",
            "value": int(np.sum(ds_steps <= oracle_steps)),
            "threshold": 28,
            "passed": bool(np.sum(ds_steps <= oracle_steps) >= 28),
        },
        {
            "criterion": "cartesian_stage_auc_delta_at_least_10pp",
            "value": float(cartesian_auc.mean()),
            "threshold": 0.10,
            "passed": bool(cartesian_auc.mean() >= 0.10),
        },
        {
            "criterion": "global_ablation_auc_delta_at_least_1pp",
            "value": float(global_auc.mean()),
            "threshold": 0.01,
            "passed": bool(global_auc.mean() >= 0.01),
        },
        {
            "criterion": "global_ablation_at_least_24_positive_pairs",
            "value": int(np.sum(global_auc > 0)),
            "threshold": 24,
            "passed": bool(np.sum(global_auc > 0) >= 24),
        },
        {
            "criterion": "radial_ablation_auc_delta_at_least_3pp",
            "value": float(coupled_auc.mean()),
            "threshold": 0.03,
            "passed": bool(coupled_auc.mean() >= 0.03),
        },
        {
            "criterion": "radial_ablation_at_least_24_positive_pairs",
            "value": int(np.sum(coupled_auc > 0)),
            "threshold": 24,
            "passed": bool(np.sum(coupled_auc > 0) >= 24),
        },
    ]
    write_csv(args.output_dir / "learned_shell_formal_decision.csv", decision)

    parameter_summary = summarize_parameters(args.parameter_path)
    write_csv(
        args.output_dir / "learned_shell_formal_parameter_summary.csv",
        parameter_summary,
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    x = np.arange(2)
    for index in range(len(EXPECTED_SEEDS)):
        axes[0].plot(
            x,
            [
                100 * grouped["Cartesian full-cov oracle"][index]["auc"],
                100 * grouped["DS stage"][index]["auc"],
            ],
            color="#888888",
            alpha=0.45,
        )
        axes[1].plot(
            x,
            [oracle_steps[index] / 1000, ds_steps[index] / 1000],
            color="#888888",
            alpha=0.45,
        )
    axes[0].scatter(
        np.zeros(len(EXPECTED_SEEDS)),
        [100 * row["auc"] for row in grouped["Cartesian full-cov oracle"]],
        color="#d62728",
        s=18,
    )
    axes[0].scatter(
        np.ones(len(EXPECTED_SEEDS)),
        [100 * row["auc"] for row in grouped["DS stage"]],
        color="#2ca02c",
        s=18,
    )
    axes[0].set_xticks(x, ("Full-cov oracle", "DS stage"))
    axes[0].set_ylabel("Normalized success AUC (%)")

    axes[1].scatter(np.zeros(len(EXPECTED_SEEDS)), oracle_steps / 1000, color="#d62728", s=18)
    axes[1].scatter(np.ones(len(EXPECTED_SEEDS)), ds_steps / 1000, color="#2ca02c", s=18)
    axes[1].set_xticks(x, ("Full-cov oracle", "DS stage"))
    axes[1].set_ylabel("Stage-0 interactions to 50% (k)")

    labels = [row["condition"] for row in parameter_summary]
    positions = np.arange(len(labels))
    axes[2].bar(
        positions - 0.18,
        [row["stage0_radial_std_mean"] for row in parameter_summary],
        0.36,
        label="Radial std",
    )
    axes[2].bar(
        positions + 0.18,
        [row["stage0_tangential_action_std_mean"] for row in parameter_summary],
        0.36,
        label="Tangential std",
    )
    axes[2].set_xticks(positions, labels, rotation=30, ha="right")
    axes[2].set_ylabel("Learned stage-0 local action std")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle("Formal learned-shell paired and mechanism diagnostics")
    figure.tight_layout()
    figure.savefig(args.output_dir / "learned_shell_formal_paired.png", dpi=220)
    plt.close(figure)

    for row in paired:
        print(row)
    for row in decision:
        print(row)


if __name__ == "__main__":
    main()
