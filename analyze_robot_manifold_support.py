#!/usr/bin/env python3
"""Analyze the task1/task2 robot manifold-support discovery pilot."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


RUN_GROUP = "robot_manifold_support_pilot_100k_20260925"
CONDITIONS = ("cart_projection", "exp_tangent", "projected_tangent")
TASKS = ("task1", "task2")
# OGBench's cube relabeler uses reward = completed_cubes - num_cubes.
# cube-triple therefore has the same [-3, 0] per-step range for every task ID.
MIN_RETURN = {"task1": -3000.0, "task2": -3000.0}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def identify(flags):
    task = "task1" if "task1" in flags["env_name"] else "task2"
    if flags["ds_mode"] == "projection":
        condition = "cart_projection"
    elif flags["ds_integrated_type"] == "projected_tangent":
        condition = "projected_tangent"
    else:
        condition = "exp_tangent"
    return task, condition


def finite_tail(rows, key, count=10):
    values = np.asarray([float(row[key]) for row in rows[-count:]], dtype=float)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else np.nan


def wilson_interval(success_rate, episodes, z=1.959963984540054):
    """Return a 95% Wilson interval for one checkpoint's success rate."""
    if not np.isfinite(success_rate) or episodes <= 0:
        return np.nan, np.nan
    denominator = 1.0 + z**2 / episodes
    center = (success_rate + z**2 / (2.0 * episodes)) / denominator
    half_width = (
        z
        * np.sqrt(
            success_rate * (1.0 - success_rate) / episodes
            + z**2 / (4.0 * episodes**2)
        )
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("run_group") != RUN_GROUP:
            continue
        eval_path = flags_path.with_name("eval.csv")
        env_path = flags_path.with_name("env.csv")
        train_path = flags_path.with_name("online_agent.csv")
        if not all(path.exists() for path in (eval_path, env_path, train_path)):
            continue
        eval_rows = read_rows(eval_path)
        if not eval_rows or int(eval_rows[-1]["step"]) != 100000:
            continue
        task, condition = identify(flags)
        steps = np.asarray([int(row["step"]) for row in eval_rows])
        success = np.asarray([float(row["success"]) for row in eval_rows])
        returns = np.asarray([float(row["episode.return"]) for row in eval_rows])
        normalized = np.clip((returns - MIN_RETURN[task]) / -MIN_RETURN[task], 0.0, 1.0)
        x = np.r_[0, steps]
        run = {
            "task": task,
            "condition": condition,
            "seed": int(flags["seed"]),
            "final_success": float(success[-1]),
            "success_auc": float(np.trapezoid(np.r_[0.0, success], x) / steps[-1]),
            "final_return": float(returns[-1]),
            "normalized_return_auc": float(np.trapezoid(np.r_[0.0, normalized], x) / steps[-1]),
        }
        for step, value in zip(steps, normalized, strict=True):
            run[f"normalized_return_{step // 1000}k"] = float(value)
        for step, value in zip(steps, success, strict=True):
            run[f"success_{step // 1000}k"] = float(value)

        env_rows = [row for row in read_rows(env_path) if int(row["step"]) >= int(flags["start_training"])]
        stage = np.asarray([float(row["stage_reward"]) for row in env_rows])
        norm = np.asarray([float(row["action_xyz_norm"]) for row in env_rows])
        train_rows = read_rows(train_path)
        finite = all(
            np.isfinite(float(value))
            for row in train_rows
            for key, value in row.items()
            if key != "step"
        )
        run.update({
            "stage_beyond_zero_fraction": float(np.mean(stage > -3.0)),
            "stage_two_fraction": float(np.mean(stage > -2.0)),
            "translation_norm": float(norm.mean()),
            "translation_norm_p95": float(np.quantile(norm, 0.95)),
            "finite_training_metrics": bool(finite),
            "tail_entropy": finite_tail(train_rows, "actor/entropy"),
            "tail_alpha": finite_tail(train_rows, "actor/alpha"),
            "tail_critic_loss": finite_tail(train_rows, "critic/critic_loss"),
        })
        fixed_path = flags_path.with_name("eval_fixed_seed.csv")
        if fixed_path.exists():
            fixed = read_rows(fixed_path)[0]
            fixed_success = float(fixed["success"])
            fixed_episodes = int(fixed["episodes"])
            fixed_low, fixed_high = wilson_interval(fixed_success, fixed_episodes)
            run.update({
                "fixed_reeval_episodes": fixed_episodes,
                "fixed_reeval_success": fixed_success,
                "fixed_reeval_success_ci_low": fixed_low,
                "fixed_reeval_success_ci_high": fixed_high,
                "fixed_reeval_return": float(fixed["episode_return"]),
            })
        else:
            run.update({
                "fixed_reeval_episodes": np.nan,
                "fixed_reeval_success": np.nan,
                "fixed_reeval_success_ci_low": np.nan,
                "fixed_reeval_success_ci_high": np.nan,
                "fixed_reeval_return": np.nan,
            })
        runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    parser.add_argument("--require-fixed-reeval", action="store_true")
    args = parser.parse_args()
    runs = load_runs(args.root)
    grouped = {
        (task, condition): sorted(
            [run for run in runs if run["task"] == task and run["condition"] == condition],
            key=lambda row: row["seed"],
        )
        for task in TASKS for condition in CONDITIONS
    }
    for key, values in grouped.items():
        found = [row["seed"] for row in values]
        if found != [50, 51]:
            raise SystemExit(f"Expected {key} seeds [50, 51], found {found}")
    if args.require_fixed_reeval and not all(
        np.isfinite(run["fixed_reeval_episodes"]) for run in runs
    ):
        raise SystemExit("Fixed reevaluation is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(runs, key=lambda row: (TASKS.index(row["task"]), CONDITIONS.index(row["condition"]), row["seed"]))
    write_rows(args.output_dir / "robot_manifold_support_pilot_per_run.csv", runs)

    summary = []
    numeric = [
        key
        for key in runs[0]
        if key not in ("task", "condition", "seed", "finite_training_metrics")
        and not (key.startswith("normalized_return_") and key.endswith("k"))
        and not (key.startswith("success_") and key.endswith("k"))
    ]
    for task in TASKS:
        for condition in CONDITIONS:
            item = {"task": task, "condition": condition, "num_seeds": 2}
            for metric in numeric:
                values = np.asarray([row[metric] for row in grouped[(task, condition)]])
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std(ddof=1))
            summary.append(item)
    write_rows(args.output_dir / "robot_manifold_support_pilot_summary.csv", summary)

    cart = {(row["task"], row["seed"]): row for row in runs if row["condition"] == "cart_projection"}
    decisions = {}
    for condition in ("exp_tangent", "projected_tangent"):
        candidate = {(row["task"], row["seed"]): row for row in runs if row["condition"] == condition}
        keys = sorted(cart)
        auc_delta = np.asarray([candidate[key]["normalized_return_auc"] - cart[key]["normalized_return_auc"] for key in keys])
        norm_ratio = np.asarray([candidate[key]["translation_norm"] / cart[key]["translation_norm"] for key in keys])
        finite = all(candidate[key]["finite_training_metrics"] for key in keys)
        decisions[condition] = {
            "normalized_return_auc_differences": auc_delta.tolist(),
            "mean_auc_difference": float(auc_delta.mean()),
            "num_positive_pairs": int(np.sum(auc_delta > 0)),
            "mean_translation_norm_ratio": float(norm_ratio.mean()),
            "finite": bool(finite),
            "auc_gate": bool(auc_delta.mean() >= 0.02),
            "pair_gate": bool(np.sum(auc_delta > 0) >= 3),
            "norm_gate": bool(norm_ratio.mean() >= 0.90),
            "launch_confirmation": bool(auc_delta.mean() >= 0.02 and np.sum(auc_delta > 0) >= 3 and norm_ratio.mean() >= 0.90 and finite),
        }
    exp = {(row["task"], row["seed"]): row for row in runs if row["condition"] == "exp_tangent"}
    projected = {(row["task"], row["seed"]): row for row in runs if row["condition"] == "projected_tangent"}
    support_delta = np.asarray([projected[key]["normalized_return_auc"] - exp[key]["normalized_return_auc"] for key in sorted(exp)])
    decisions["projected_vs_exp"] = {
        "auc_differences": support_delta.tolist(),
        "mean_absolute_difference": float(abs(support_delta.mean())),
        "num_pair_losses_over_0p05": int(np.sum(support_delta < -0.05)),
        "practically_similar": bool(abs(support_delta.mean()) <= 0.02 and np.sum(support_delta < -0.05) <= 1),
    }
    if all(np.isfinite(run["fixed_reeval_episodes"]) for run in runs):
        decisions["fixed_reevaluation"] = {
            condition: {
                "success_rates": [
                    run["fixed_reeval_success"]
                    for run in runs
                    if run["condition"] == condition
                ],
                "mean_success": float(
                    np.mean([
                        run["fixed_reeval_success"]
                        for run in runs
                        if run["condition"] == condition
                    ])
                ),
                "mean_return": float(
                    np.mean([
                        run["fixed_reeval_return"]
                        for run in runs
                        if run["condition"] == condition
                    ])
                ),
            }
            for condition in CONDITIONS
        }
    (args.output_dir / "robot_manifold_support_pilot_decision.json").write_text(json.dumps(decisions, indent=2) + "\n")

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.0), sharex="col")
    steps = np.asarray((25, 50, 75, 100))
    for row_index, task in enumerate(TASKS):
        for condition in CONDITIONS:
            curves = np.asarray([[run[f"normalized_return_{step}k"] for step in steps] for run in grouped[(task, condition)]])
            axes[row_index, 0].plot(steps, 100 * curves.mean(0), marker="o", label=condition)
        x = np.arange(len(CONDITIONS))
        axes[row_index, 1].bar(x, [100 * np.mean([run["normalized_return_auc"] for run in grouped[(task, condition)]]) for condition in CONDITIONS])
        stage_metric = (
            "stage_two_fraction" if task == "task1" else "stage_beyond_zero_fraction"
        )
        axes[row_index, 2].bar(x, [100 * np.mean([run[stage_metric] for run in grouped[(task, condition)]]) for condition in CONDITIONS])
        axes[row_index, 0].set_ylabel(f"{task}: normalized return (%)")
        axes[row_index, 1].set_ylabel("Return AUC (%)")
        axes[row_index, 2].set_ylabel(
            "At least two cubes complete (%)"
            if task == "task1"
            else "At least one cube complete (%)"
        )
        for column in (1, 2):
            axes[row_index, column].set_xticks(x, CONDITIONS, rotation=25, ha="right")
        for axis in axes[row_index]:
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("Environment steps (k)")
    figure.suptitle("Robot manifold-support pilot (two discovery seeds)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "robot_manifold_support_pilot.png", dpi=180)
    plt.close(figure)
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
