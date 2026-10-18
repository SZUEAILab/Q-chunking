#!/usr/bin/env python3
"""Analyze the frozen task1 executed-variance-matched discovery study."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from analyze_robot_manifold_task1_300k import (
    EVAL_STEPS,
    load_runs,
    paired_bootstrap,
    write_rows,
)


RUN_GROUP = "robot_variance_matched_task1_300k_20260926"
CONDITIONS = ("cart_projection", "exp_tangent", "exp_variance_matched")
SEEDS = (56, 57, 58, 59)


def comparison(candidate, reference, *, fixed_complete, bootstrap_seed):
    auc_delta = np.asarray([
        candidate[seed]["normalized_return_auc"]
        - reference[seed]["normalized_return_auc"]
        for seed in SEEDS
    ])
    fixed_delta = np.asarray([
        candidate[seed]["fixed_reeval_success"]
        - reference[seed]["fixed_reeval_success"]
        for seed in SEEDS
    ])
    finite = all(candidate[seed]["finite_training_metrics"] for seed in SEEDS)
    auc_gate = bool(auc_delta.mean() >= 0.02)
    pair_gate = bool(np.sum(auc_delta > 0.0) >= 3)
    final_gate = bool(fixed_complete and fixed_delta.mean() >= -0.02)
    return {
        "normalized_return_auc_differences": auc_delta.tolist(),
        "mean_auc_difference": float(auc_delta.mean()),
        "mean_auc_difference_ci": paired_bootstrap(
            auc_delta, seed=bootstrap_seed
        ),
        "num_positive_pairs": int(np.sum(auc_delta > 0.0)),
        "fixed_success_differences": fixed_delta.tolist(),
        "mean_fixed_success_difference": (
            float(fixed_delta.mean()) if fixed_complete else None
        ),
        "finite": bool(finite),
        "auc_gate": auc_gate,
        "pair_gate": pair_gate,
        "final_noninferiority_gate": final_gate,
        "performance_gates_pass": bool(
            auc_gate and pair_gate and finite and final_gate
        ),
    }


def geometry_decision(path):
    if path is None or not path.exists():
        return {"available": False, "match_gate": False}
    rows = {row["condition"]: row for row in csv.DictReader(path.open())}
    cart = rows["cart_projection"]
    matched = rows["exp_variance_matched"]
    angular_error = abs(
        float(matched["angular_deviation_rad_mean"])
        - float(cart["angular_deviation_rad_mean"])
    )
    radius_error = abs(
        float(matched["radius_mean_mean"])
        - float(cart["radius_mean_mean"])
    )
    radial_std_error = abs(
        float(matched["radius_std_within_state_mean"])
        - float(cart["radius_std_within_state_mean"])
    )
    return {
        "available": True,
        "angular_deviation_error": angular_error,
        "mean_radius_error": radius_error,
        "radial_std_error": radial_std_error,
        "match_gate": bool(
            angular_error <= 0.05
            and radius_error <= 0.03
            and radial_std_error <= 0.02
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    parser.add_argument("--require-fixed-reeval", action="store_true")
    parser.add_argument("--geometry-summary", type=Path)
    args = parser.parse_args()

    runs = load_runs(args.root, run_group=RUN_GROUP)
    grouped = {
        condition: sorted(
            [run for run in runs if run["condition"] == condition],
            key=lambda run: run["seed"],
        )
        for condition in CONDITIONS
    }
    for condition, values in grouped.items():
        found = tuple(run["seed"] for run in values)
        if found != SEEDS:
            raise SystemExit(f"Expected {condition} seeds {SEEDS}, found {found}")
    fixed_complete = all(
        np.isfinite(run["fixed_reeval_episodes"]) for run in runs
    )
    if args.require_fixed_reeval and not fixed_complete:
        raise SystemExit("Fixed reevaluation is incomplete")

    runs = [run for condition in CONDITIONS for run in grouped[condition]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "robot_variance_matched_task1_300k"
    write_rows(args.output_dir / f"{prefix}_per_run.csv", runs)

    checkpoint_keys = {
        f"normalized_return_{step // 1000}k" for step in EVAL_STEPS
    } | {f"success_{step // 1000}k" for step in EVAL_STEPS}
    excluded = {"condition", "seed", "finite_training_metrics"} | checkpoint_keys
    numeric = [key for key in runs[0] if key not in excluded]
    summary = []
    for condition in CONDITIONS:
        item = {"condition": condition, "num_seeds": len(SEEDS)}
        for metric in numeric:
            values = np.asarray([run[metric] for run in grouped[condition]])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(item)
    write_rows(args.output_dir / f"{prefix}_summary.csv", summary)

    by_seed = {
        condition: {run["seed"]: run for run in grouped[condition]}
        for condition in CONDITIONS
    }
    matched_vs_default = comparison(
        by_seed["exp_variance_matched"],
        by_seed["exp_tangent"],
        fixed_complete=fixed_complete,
        bootstrap_seed=20260926,
    )
    matched_vs_cart = comparison(
        by_seed["exp_variance_matched"],
        by_seed["cart_projection"],
        fixed_complete=fixed_complete,
        bootstrap_seed=20260927,
    )
    geometry = geometry_decision(args.geometry_summary)
    decisions = {
        "matched_vs_default": matched_vs_default,
        "matched_vs_cartesian": matched_vs_cart,
        "geometry": geometry,
        "overwide_variance_hypothesis_passes": bool(
            matched_vs_default["performance_gates_pass"]
            and geometry["match_gate"]
        ),
        "matched_support_beats_cartesian": bool(
            matched_vs_cart["performance_gates_pass"]
            and geometry["match_gate"]
        ),
    }
    (args.output_dir / f"{prefix}_decision.json").write_text(
        json.dumps(decisions, indent=2) + "\n"
    )

    steps_k = EVAL_STEPS / 1000
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for condition in CONDITIONS:
        normalized = np.asarray([
            [run[f"normalized_return_{int(step)}k"] for step in steps_k]
            for run in grouped[condition]
        ])
        success = np.asarray([
            [run[f"success_{int(step)}k"] for step in steps_k]
            for run in grouped[condition]
        ])
        axes[0].plot(steps_k, 100 * normalized.mean(0), marker="o", label=condition)
        axes[0].fill_between(
            steps_k, 100 * normalized.min(0), 100 * normalized.max(0), alpha=0.12
        )
        axes[1].plot(steps_k, 100 * success.mean(0), marker="o", label=condition)
    axes[0].set_ylabel("Normalized return (%)")
    axes[1].set_ylabel("Success rate (%)")
    for axis in axes:
        axis.set_xlabel("Environment steps (k)")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Task1 300k executed-variance-matched discovery")
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{prefix}.png", dpi=190)
    plt.close(figure)
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
