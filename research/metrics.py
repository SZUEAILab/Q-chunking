"""Tested analysis primitives retained from historical batch scripts.

These loaders read outcome metrics, so they are not reward-blind validators.
Callers must supply the registered seeds, evaluation grid and run group; no
historical batch is selected implicitly.
"""

import csv
import json

import numpy as np


def _read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_mean(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def bootstrap_interval(differences, rng, count=50_000):
    indices = rng.integers(0, len(differences), size=(count, len(differences)))
    estimates = differences[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(
        np.quantile(estimates, 0.975)
    )


def load_shell_runs(curve_path, *, conditions, expected_seeds):
    """Load shell curves using the historical final/AUC/threshold definitions."""
    rows = _read_rows(curve_path)
    grouped = {}
    for row in rows:
        key = (row["condition"], int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    per_run = []
    for condition in conditions:
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


def load_moment_runs(root, *, run_group, eval_steps):
    """Load robot moment-control runs with the original manipulation gates.

    Return normalization assumes the historical task's [-3000, 0] bounds.
    This is not a general return normalizer or a reward-blind admission check.
    """
    eval_steps = np.asarray(eval_steps)
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("run_group") != run_group:
            continue
        eval_path = flags_path.with_name("eval.csv")
        train_path = flags_path.with_name("online_agent.csv")
        if not eval_path.exists() or not train_path.exists():
            continue
        eval_rows = _read_rows(eval_path)
        if not eval_rows or int(eval_rows[-1]["step"]) != eval_steps[-1]:
            continue
        steps = np.asarray([int(row["step"]) for row in eval_rows])
        if not np.array_equal(steps, eval_steps):
            raise ValueError(f"unexpected evaluation steps in {eval_path}: {steps}")
        returns = np.asarray([float(row["episode.return"]) for row in eval_rows])
        success = np.asarray([float(row["success"]) for row in eval_rows])
        normalized = np.clip((returns + 3000.0) / 3000.0, 0.0, 1.0)
        x = np.r_[0, steps]
        train_rows = _read_rows(train_path)
        finite = all(
            np.isfinite(float(value))
            for row in train_rows
            for key, value in row.items()
            if key != "step"
        )
        if flags["ds_mode"] == "projection":
            condition = "cart_projection"
        elif flags["agent"].get("ds_moment_controlled", False):
            condition = "moment_tangent"
        else:
            condition = "exp_tangent"
        run = {
            "path": str(flags_path.parent),
            "condition": condition,
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
        if condition == "moment_tangent":
            control_radius_mean = bool(
                flags["agent"].get("ds_moment_control_speed_mean", True)
            )
            controller_path = flags_path.with_name("moment_controller.csv")
            controller = {
                int(row["step"]): row for row in _read_rows(controller_path)
            }
            passes = []
            for step in eval_steps:
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
