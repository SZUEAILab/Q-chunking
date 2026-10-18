#!/usr/bin/env python3
"""Analyze global versus reward-stage semantic regularization."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from analyze_klbudget_screen import (
    exact_sign_flip_pvalue,
    read_curve,
    read_fixed_success,
    write_csv,
)


ORDER = ("global", "stage")
DISPLAY = {
    "global": "Global semantic KL",
    "stage": "Reward-stage translation KL",
}
COLORS = {"global": "#2ca02c", "stage": "#d62728"}


def condition(flags):
    mode = flags.get("ds_phase_regularization", "none")
    if mode == "none":
        return "global"
    if mode == "reward_stage_translation":
        return "stage"
    raise ValueError(f"Unexpected phase mode: {mode}")


def finite_mean(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def read_env_metrics(path, start_training):
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(float(row["step"])) > start_training
        ]
    rewards = np.asarray([float(row["stage_reward"]) for row in rows])
    norms = np.asarray([float(row["action_xyz_norm"]) for row in rows])
    result = {
        "stage1_occupancy": float(np.mean(rewards >= -2.0)),
        "stage2_occupancy": float(np.mean(rewards >= -1.0)),
        "norm_mean": float(norms.mean()),
    }
    for stage, reward in enumerate((-3.0, -2.0, -1.0)):
        selected = norms[np.isclose(rewards, reward)]
        result[f"norm_stage{stage}"] = (
            float(selected.mean()) if len(selected) else np.nan
        )
    return result


def read_training_metrics(path, condition_name, count=10):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[-count:]
    result = {
        "tail_alpha": finite_mean(float(row["actor/alpha"]) for row in rows),
        "tail_entropy": finite_mean(float(row["actor/entropy"]) for row in rows),
    }
    if condition_name == "stage":
        result["tail_phase_weight"] = finite_mean(
            float(row["actor/phase_weight"]) for row in rows
        )
        result["tail_late_fraction"] = finite_mean(
            float(row["actor/phase_late_fraction"]) for row in rows
        )
    else:
        result["tail_phase_weight"] = 1.0
        result["tail_late_fraction"] = np.nan
    return result


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        eval_path = flags_path.with_name("eval.csv")
        if not eval_path.exists():
            continue
        steps, success = read_curve(eval_path)
        if not len(steps) or steps[-1] != 500000:
            continue
        with eval_path.open(newline="") as handle:
            eval_rows = list(csv.DictReader(handle))
        name = condition(flags)
        auc = np.trapezoid(
            np.concatenate(([0.0], success)), np.concatenate(([0], steps))
        ) / steps[-1]
        runs.append(
            {
                "condition": name,
                "seed": int(flags["seed"]),
                "steps": steps,
                "success": success,
                "final_success": float(success[-1]),
                "tail3_success": float(success[-3:].mean()),
                "auc": float(auc),
                "fixed_success": read_fixed_success(
                    flags_path.with_name("eval_fixed_seed.csv")
                ),
                "final_return": float(eval_rows[-1]["episode.return"]),
                **read_env_metrics(
                    flags_path.with_name("env.csv"), int(flags["start_training"])
                ),
                **read_training_metrics(
                    flags_path.with_name("online_agent.csv"), name
                ),
            }
        )
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()

    runs = load_runs(args.root)
    grouped = {
        name: sorted(
            [run for run in runs if run["condition"] == name],
            key=lambda run: run["seed"],
        )
        for name in ORDER
    }
    expected_seeds = [8, 9, 10, 11]
    for name in ORDER:
        seeds = [run["seed"] for run in grouped[name]]
        if seeds != expected_seeds:
            raise SystemExit(f"Expected {name} seeds {expected_seeds}, found {seeds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run = []
    for run in sorted(runs, key=lambda item: (ORDER.index(item["condition"]), item["seed"])):
        row = {key: value for key, value in run.items() if key not in ("steps", "success")}
        row.update(
            {
                f"success_{step // 1000}k": value
                for step, value in zip(run["steps"], run["success"], strict=True)
            }
        )
        per_run.append(row)
    write_csv(args.output_dir / "phase_regularization_per_run.csv", per_run)

    metrics = (
        "final_success",
        "tail3_success",
        "auc",
        "fixed_success",
        "final_return",
        "stage1_occupancy",
        "stage2_occupancy",
        "norm_mean",
        "norm_stage0",
        "norm_stage1",
        "norm_stage2",
        "tail_alpha",
        "tail_entropy",
        "tail_phase_weight",
        "tail_late_fraction",
    )
    summary = []
    for name in ORDER:
        row = {"condition": name, "display_name": DISPLAY[name], "num_seeds": 4}
        for metric in metrics:
            values = np.asarray([run[metric] for run in grouped[name]], dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        summary.append(row)
    write_csv(args.output_dir / "phase_regularization_summary.csv", summary)

    rng = np.random.default_rng(20260925)
    paired = []
    for metric in (
        "final_success",
        "tail3_success",
        "auc",
        "fixed_success",
        "final_return",
        "stage1_occupancy",
        "stage2_occupancy",
    ):
        control = np.asarray([run[metric] for run in grouped["global"]])
        treatment = np.asarray([run[metric] for run in grouped["stage"]])
        if not np.all(np.isfinite(control)) or not np.all(np.isfinite(treatment)):
            continue
        differences = treatment - control
        indices = rng.integers(0, len(differences), size=(20000, len(differences)))
        bootstrap = differences[indices].mean(axis=1)
        paired.append(
            {
                "contrast": "stage-global",
                "metric": metric,
                "mean_delta": float(differences.mean()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                "num_positive": int(np.sum(differences > 0)),
                "num_tied": int(np.sum(differences == 0)),
                "paired_differences": ";".join(f"{x:.6g}" for x in differences),
            }
        )
    write_csv(args.output_dir / "phase_regularization_paired.csv", paired)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes = axes.ravel()
    for name in ORDER:
        curves = np.asarray([run["success"] for run in grouped[name]]) * 100
        steps = grouped[name][0]["steps"] / 1000
        for curve in curves:
            axes[0].plot(
                steps,
                curve,
                linewidth=1.0,
                color=COLORS[name],
                alpha=0.22,
            )
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1)
        axes[0].plot(
            steps,
            mean,
            marker="o",
            linewidth=2.2,
            color=COLORS[name],
            label=DISPLAY[name],
        )
        axes[0].fill_between(
            steps,
            np.maximum(0.0, mean - std),
            mean + std,
            color=COLORS[name],
            alpha=0.16,
        )
    axes[0].set_xlabel("Environment steps (k)")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    fixed = {
        name: np.asarray([run["fixed_success"] for run in grouped[name]]) * 100
        for name in ORDER
    }
    for seed_index in range(len(grouped["global"])):
        axes[1].plot(
            [0, 1],
            [fixed["global"][seed_index], fixed["stage"][seed_index]],
            color="#888888",
            alpha=0.55,
            linewidth=1.2,
        )
    for index, name in enumerate(ORDER):
        axes[1].scatter(
            np.full(len(fixed[name]), index),
            fixed[name],
            color=COLORS[name],
            s=38,
            zorder=3,
        )
        axes[1].scatter(
            index,
            fixed[name].mean(),
            color="black",
            marker="D",
            s=42,
            zorder=4,
        )
    axes[1].set_xticks(range(2), ("Global", "Reward-stage"))
    axes[1].set_ylabel("Fixed-seed success rate (%)")
    axes[1].set_ylim(bottom=0)

    x = np.arange(2)
    width = 0.34
    for index, name in enumerate(ORDER):
        means = [
            finite_mean(run[metric] for run in grouped[name]) * 100
            for metric in ("stage1_occupancy", "stage2_occupancy")
        ]
        axes[2].bar(
            x + (index - 0.5) * width,
            means,
            width,
            color=COLORS[name],
            label=DISPLAY[name],
        )
    axes[2].set_xticks(x, ("At least 1 cube", "At least 2 cubes"))
    axes[2].set_ylabel("Online-state occupancy (%)")

    for name in ORDER:
        means = [
            finite_mean(run[f"norm_stage{stage}"] for run in grouped[name])
            for stage in range(3)
        ]
        axes[3].plot(
            range(3), means, marker="o", linewidth=2.2, color=COLORS[name], label=name
        )
    axes[3].set_xticks(range(3), ("0 cubes", "1 cube", "2 cubes"))
    axes[3].set_ylabel("Mean translation norm")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Reward-stage allocation of semantic direction--speed regularization")
    figure.tight_layout()
    figure.savefig(args.output_dir / "phase_regularization_screen.png", dpi=220)
    plt.close(figure)

    for row in summary:
        print(row)
    for row in paired:
        print(row)


if __name__ == "__main__":
    main()
