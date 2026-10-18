#!/usr/bin/env python3
"""Analyze the frozen task1 300k manifold-support discovery study."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


RUN_GROUP = "robot_manifold_task1_300k_20260925"
CONDITIONS = ("cart_projection", "exp_tangent", "projected_tangent")
SEEDS = (52, 53, 54, 55)
EVAL_STEPS = np.arange(50_000, 300_001, 50_000)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def identify(flags):
    if flags["ds_mode"] == "projection":
        return "cart_projection"
    if flags["ds_integrated_type"] == "projected_tangent":
        return "projected_tangent"
    if flags["agent"].get("ds_tangent_speed_mean_offset", 0.0) != 0.0:
        return "exp_variance_matched"
    return "exp_tangent"


def finite_tail(rows, key, count=10):
    values = np.asarray([float(row[key]) for row in rows[-count:]], dtype=float)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else np.nan


def wilson_interval(success_rate, episodes, z=1.959963984540054):
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


def paired_bootstrap(values, seed=20260925, samples=100_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def load_runs(root, run_group=RUN_GROUP):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("run_group") != run_group:
            continue
        eval_path = flags_path.with_name("eval.csv")
        env_path = flags_path.with_name("env.csv")
        train_path = flags_path.with_name("online_agent.csv")
        if not all(path.exists() for path in (eval_path, env_path, train_path)):
            continue
        eval_rows = read_rows(eval_path)
        if not eval_rows or int(eval_rows[-1]["step"]) != 300_000:
            continue
        steps = np.asarray([int(row["step"]) for row in eval_rows])
        if not np.array_equal(steps, EVAL_STEPS):
            raise ValueError(f"Unexpected evaluation steps in {eval_path}: {steps}")
        success = np.asarray([float(row["success"]) for row in eval_rows])
        returns = np.asarray([float(row["episode.return"]) for row in eval_rows])
        normalized = np.clip((returns + 3000.0) / 3000.0, 0.0, 1.0)
        x = np.r_[0, steps]
        env_rows = [
            row
            for row in read_rows(env_path)
            if int(row["step"]) >= int(flags["start_training"])
        ]
        stages = np.asarray([float(row["stage_reward"]) for row in env_rows])
        norms = np.asarray([float(row["action_xyz_norm"]) for row in env_rows])
        train_rows = read_rows(train_path)
        finite = all(
            np.isfinite(float(value))
            for row in train_rows
            for key, value in row.items()
            if key != "step"
        )
        run = {
            "condition": identify(flags),
            "seed": int(flags["seed"]),
            "final_success": float(success[-1]),
            "success_auc": float(np.trapezoid(np.r_[0.0, success], x) / steps[-1]),
            "final_return": float(returns[-1]),
            "normalized_return_auc": float(
                np.trapezoid(np.r_[0.0, normalized], x) / steps[-1]
            ),
            "stage_beyond_zero_fraction": float(np.mean(stages > -3.0)),
            "stage_two_fraction": float(np.mean(stages > -2.0)),
            "translation_norm": float(norms.mean()),
            "translation_norm_p95": float(np.quantile(norms, 0.95)),
            "finite_training_metrics": bool(finite),
            "tail_entropy": finite_tail(train_rows, "actor/entropy"),
            "tail_alpha": finite_tail(train_rows, "actor/alpha"),
            "tail_critic_loss": finite_tail(train_rows, "critic/critic_loss"),
        }
        for step, value in zip(steps, normalized, strict=True):
            run[f"normalized_return_{step // 1000}k"] = float(value)
        for step, value in zip(steps, success, strict=True):
            run[f"success_{step // 1000}k"] = float(value)

        fixed_path = flags_path.with_name("eval_fixed_seed.csv")
        if fixed_path.exists():
            fixed = read_rows(fixed_path)[0]
            fixed_success = float(fixed["success"])
            fixed_episodes = int(fixed["episodes"])
            low, high = wilson_interval(fixed_success, fixed_episodes)
            run.update({
                "fixed_reeval_episodes": fixed_episodes,
                "fixed_reeval_success": fixed_success,
                "fixed_reeval_success_ci_low": low,
                "fixed_reeval_success_ci_high": high,
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
    fixed_complete = all(np.isfinite(run["fixed_reeval_episodes"]) for run in runs)
    if args.require_fixed_reeval and not fixed_complete:
        raise SystemExit("Fixed reevaluation is incomplete")

    runs = [run for condition in CONDITIONS for run in grouped[condition]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "robot_manifold_task1_300k_per_run.csv", runs)

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
    write_rows(args.output_dir / "robot_manifold_task1_300k_summary.csv", summary)

    cart = {run["seed"]: run for run in grouped["cart_projection"]}
    decisions = {}
    for condition in ("exp_tangent", "projected_tangent"):
        candidate = {run["seed"]: run for run in grouped[condition]}
        auc_delta = np.asarray([
            candidate[seed]["normalized_return_auc"]
            - cart[seed]["normalized_return_auc"]
            for seed in SEEDS
        ])
        norm_ratio = np.asarray([
            candidate[seed]["translation_norm"] / cart[seed]["translation_norm"]
            for seed in SEEDS
        ])
        fixed_delta = np.asarray([
            candidate[seed]["fixed_reeval_success"]
            - cart[seed]["fixed_reeval_success"]
            for seed in SEEDS
        ])
        finite = all(candidate[seed]["finite_training_metrics"] for seed in SEEDS)
        return_gate = bool(auc_delta.mean() >= 0.02)
        pair_gate = bool(np.sum(auc_delta > 0.0) >= 3)
        norm_gate = bool(norm_ratio.mean() >= 0.90)
        final_gate = bool(fixed_complete and fixed_delta.mean() >= -0.02)
        decisions[condition] = {
            "normalized_return_auc_differences": auc_delta.tolist(),
            "mean_auc_difference": float(auc_delta.mean()),
            "mean_auc_difference_ci": paired_bootstrap(
                auc_delta, seed=20260925 + CONDITIONS.index(condition)
            ),
            "num_positive_pairs": int(np.sum(auc_delta > 0.0)),
            "mean_translation_norm_ratio": float(norm_ratio.mean()),
            "fixed_success_differences": fixed_delta.tolist(),
            "mean_fixed_success_difference": (
                float(fixed_delta.mean()) if fixed_complete else None
            ),
            "finite": bool(finite),
            "auc_gate": return_gate,
            "pair_gate": pair_gate,
            "norm_gate": norm_gate,
            "final_noninferiority_gate": final_gate,
            "qualifies_cross_task": bool(
                return_gate and pair_gate and norm_gate and finite and final_gate
            ),
        }
    exp = {run["seed"]: run for run in grouped["exp_tangent"]}
    projected = {run["seed"]: run for run in grouped["projected_tangent"]}
    support_delta = np.asarray([
        projected[seed]["normalized_return_auc"]
        - exp[seed]["normalized_return_auc"]
        for seed in SEEDS
    ])
    decisions["projected_vs_exp"] = {
        "auc_differences": support_delta.tolist(),
        "mean_difference": float(support_delta.mean()),
        "mean_difference_ci": paired_bootstrap(support_delta, seed=20260928),
        "num_absolute_pair_differences_over_0p05": int(
            np.sum(np.abs(support_delta) > 0.05)
        ),
        "practically_similar": bool(
            abs(support_delta.mean()) <= 0.02
            and np.sum(np.abs(support_delta) > 0.05) <= 1
        ),
    }
    decisions["support_hypothesis_passes"] = bool(
        decisions["exp_tangent"]["qualifies_cross_task"]
        and decisions["projected_tangent"]["qualifies_cross_task"]
        and decisions["projected_vs_exp"]["practically_similar"]
    )
    (args.output_dir / "robot_manifold_task1_300k_decision.json").write_text(
        json.dumps(decisions, indent=2) + "\n"
    )

    steps_k = EVAL_STEPS / 1000
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 7.5))
    rng = np.random.default_rng(20260925)
    for condition_index, condition in enumerate(CONDITIONS):
        normalized = np.asarray([
            [run[f"normalized_return_{int(step)}k"] for step in steps_k]
            for run in grouped[condition]
        ])
        success = np.asarray([
            [run[f"success_{int(step)}k"] for step in steps_k]
            for run in grouped[condition]
        ])
        axes[0, 0].plot(steps_k, 100 * normalized.mean(0), marker="o", label=condition)
        axes[0, 0].fill_between(
            steps_k,
            100 * normalized.min(0),
            100 * normalized.max(0),
            alpha=0.12,
        )
        axes[0, 1].plot(steps_k, 100 * success.mean(0), marker="o", label=condition)
        for column, metric, ylabel, scale in (
            (2, "normalized_return_auc", "Normalized return AUC (%)", 100),
            (0, "stage_two_fraction", "At least two cubes complete (%)", 100),
            (1, "translation_norm", "Executed XYZ norm", 1),
        ):
            axis = axes[1, column]
            values = np.asarray([run[metric] for run in grouped[condition]]) * scale
            axis.scatter(
                condition_index + rng.uniform(-0.05, 0.05, len(values)),
                values,
                color=f"C{condition_index}",
                alpha=0.8,
            )
            axis.plot(condition_index, values.mean(), marker="D", color="black")
            axis.set_ylabel(ylabel)
    axes[0, 2].axis("off")
    axes[0, 0].set_ylabel("Normalized return (%)")
    axes[0, 1].set_ylabel("Success rate (%)")
    axes[0, 0].legend(fontsize=8)
    for axis in axes[0, :2]:
        axis.set_xlabel("Environment steps (k)")
        axis.grid(alpha=0.2)
    for axis in axes[1]:
        axis.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Task1 300k manifold-support discovery (four fresh seeds)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "robot_manifold_task1_300k.png", dpi=190)
    plt.close(figure)
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
