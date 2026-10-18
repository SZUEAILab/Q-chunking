#!/usr/bin/env python3
"""Analyze the frozen task1 reward-blind moment-control pilot."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


RUN_GROUP = "robot_moment_control_task1_pilot_20260926"
CONDITIONS = ("cart_projection", "exp_tangent", "moment_tangent")
SEEDS = (60, 61)
EVAL_STEPS = np.arange(50_000, 300_001, 50_000)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def identify(flags):
    if flags["ds_mode"] == "projection":
        return "cart_projection"
    if flags["agent"].get("ds_moment_controlled", False):
        return "moment_tangent"
    return "exp_tangent"


def load_complete_runs(root, run_group=RUN_GROUP):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("run_group") != run_group:
            continue
        eval_path = flags_path.with_name("eval.csv")
        train_path = flags_path.with_name("online_agent.csv")
        if not eval_path.exists() or not train_path.exists():
            continue
        eval_rows = read_rows(eval_path)
        if not eval_rows or int(eval_rows[-1]["step"]) != 300_000:
            continue
        steps = np.asarray([int(row["step"]) for row in eval_rows])
        if not np.array_equal(steps, EVAL_STEPS):
            raise ValueError(f"unexpected evaluation steps in {eval_path}: {steps}")
        returns = np.asarray([float(row["episode.return"]) for row in eval_rows])
        success = np.asarray([float(row["success"]) for row in eval_rows])
        normalized = np.clip((returns + 3000.0) / 3000.0, 0.0, 1.0)
        x = np.r_[0, steps]
        train_rows = read_rows(train_path)
        finite = all(
            np.isfinite(float(value))
            for row in train_rows
            for key, value in row.items()
            if key != "step"
        )
        run = {
            "path": str(flags_path.parent),
            "condition": identify(flags),
            "seed": int(flags["seed"]),
            "normalized_return_auc": float(
                np.trapezoid(np.r_[0.0, normalized], x) / steps[-1]
            ),
            "success_auc": float(
                np.trapezoid(np.r_[0.0, success], x) / steps[-1]
            ),
            "final_success": float(success[-1]),
            "finite_training_metrics": bool(finite),
        }
        for step, value in zip(steps, normalized, strict=True):
            run[f"normalized_return_{step // 1000}k"] = float(value)
        for step, value in zip(steps, success, strict=True):
            run[f"success_{step // 1000}k"] = float(value)
        if run["condition"] == "moment_tangent":
            control_radius_mean = bool(
                flags["agent"].get("ds_moment_control_speed_mean", True)
            )
            controller_path = flags_path.with_name("moment_controller.csv")
            controller = {
                int(row["step"]): row for row in read_rows(controller_path)
            }
            passes = []
            for step in EVAL_STEPS:
                row = controller[int(step)]
                passed = (
                    float(row["angle_abs_error"]) <= 0.05
                    and float(row["radius_std_abs_error"]) <= 0.02
                    and (
                        not control_radius_mean
                        or float(row["radius_mean_abs_error"]) <= 0.03
                    )
                )
                passes.append(bool(passed))
                run[f"manipulation_pass_{step // 1000}k"] = bool(passed)
            run["manipulation_pass_count"] = int(sum(passes))
            run["manipulation_gate"] = bool(sum(passes) >= 5)
            run["control_radius_mean"] = control_radius_mean
            run["radius_mean_target"] = float(
                flags["agent"].get("ds_moment_target_radius_mean", 0.989)
            )
            run["radius_std_target"] = float(
                flags["agent"].get("ds_moment_target_radius_std", 0.022)
            )
            run["angle_target"] = float(
                flags["agent"].get("ds_moment_target_angle", 0.161)
            )
        runs.append(run)
    return runs


def write_rows(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    parser.add_argument("--run-group", default=RUN_GROUP)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--prefix", default="robot_moment_control_task1_pilot")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    runs = load_complete_runs(args.root, run_group=args.run_group)
    grouped = {
        condition: sorted(
            [run for run in runs if run["condition"] == condition],
            key=lambda run: run["seed"],
        )
        for condition in CONDITIONS
    }
    for condition, values in grouped.items():
        found = tuple(run["seed"] for run in values)
        if found != seeds:
            raise SystemExit(f"expected {condition} seeds {seeds}, found {found}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    write_rows(args.output_dir / f"{prefix}_per_run.csv", runs)
    summary = []
    for condition in CONDITIONS:
        values = grouped[condition]
        summary.append(
            {
                "condition": condition,
                "normalized_return_auc_mean": float(
                    np.mean([run["normalized_return_auc"] for run in values])
                ),
                "normalized_return_auc_std": float(
                    np.std(
                        [run["normalized_return_auc"] for run in values], ddof=1
                    )
                ),
                "success_auc_mean": float(
                    np.mean([run["success_auc"] for run in values])
                ),
                "final_success_mean": float(
                    np.mean([run["final_success"] for run in values])
                ),
            }
        )
    write_rows(args.output_dir / f"{prefix}_summary.csv", summary)
    by_seed = {
        condition: {run["seed"]: run for run in values}
        for condition, values in grouped.items()
    }
    auc_differences = {
        str(seed): (
            by_seed["moment_tangent"][seed]["normalized_return_auc"]
            - by_seed["cart_projection"][seed]["normalized_return_auc"]
        )
        for seed in seeds
    }
    decision = {
        "manipulation_pass_counts": {
            str(seed): by_seed["moment_tangent"][seed][
                "manipulation_pass_count"
            ]
            for seed in seeds
        },
        "manipulation_gate": all(
            by_seed["moment_tangent"][seed]["manipulation_gate"]
            for seed in seeds
        ),
        "paired_auc_differences": auc_differences,
        "both_auc_nonnegative": all(
            difference >= 0.0 for difference in auc_differences.values()
        ),
        "launch_confirmation": False,
    }
    decision["launch_confirmation"] = bool(
        decision["manipulation_gate"] and decision["both_auc_nonnegative"]
    )
    (args.output_dir / f"{prefix}_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
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
    figure.suptitle("Task1 reward-blind executed-moment-control pilot")
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{prefix}.png", dpi=190)
    plt.close(figure)

    geometry_rows = []
    for run in grouped["moment_tangent"]:
        for row in read_rows(Path(run["path"]) / "moment_controller.csv"):
            geometry_rows.append(
                {
                    "seed": run["seed"],
                    "step": int(row["step"]),
                    "radius_mean": float(row["radius_mean"]),
                    "radius_std_within_state": float(
                        row["radius_std_within_state"]
                    ),
                    "angular_deviation_rad": float(
                        row["angular_deviation_rad"]
                    ),
                }
            )
    write_rows(args.output_dir / f"{prefix}_geometry.csv", geometry_rows)
    geometry_figure, geometry_axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    controlled_reference = grouped["moment_tangent"][0]
    radius_mean_target = (
        controlled_reference["radius_mean_target"]
        if controlled_reference["control_radius_mean"]
        else None
    )
    geometry_specs = (
        ("radius_mean", radius_mean_target, 0.03, "Mean translation radius"),
        (
            "radius_std_within_state",
            controlled_reference["radius_std_target"],
            0.02,
            "Within-state radius SD",
        ),
        (
            "angular_deviation_rad",
            controlled_reference["angle_target"],
            0.05,
            "Angular deviation (rad)",
        ),
    )
    for axis, (key, target, tolerance, label) in zip(
        geometry_axes, geometry_specs, strict=True
    ):
        for seed in seeds:
            selected = [row for row in geometry_rows if row["seed"] == seed]
            axis.plot(
                [row["step"] / 1000 for row in selected],
                [row[key] for row in selected],
                label=f"seed {seed}",
            )
        if target is not None:
            axis.axhline(target, color="black", linestyle="--", linewidth=1)
            axis.fill_between(
                [0, 300],
                target - tolerance,
                target + tolerance,
                color="gray",
                alpha=0.12,
            )
        axis.set_xlabel("Environment steps (k)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    geometry_axes[0].legend(fontsize=8)
    geometry_figure.suptitle("Reward-blind executed-moment manipulation check")
    geometry_figure.tight_layout()
    geometry_figure.savefig(
        args.output_dir / f"{prefix}_geometry.png", dpi=190
    )
    plt.close(geometry_figure)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
