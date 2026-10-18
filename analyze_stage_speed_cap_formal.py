#!/usr/bin/env python3
"""Analyze the fresh-seed 500k reward-stage speed-cap confirmation."""

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
    write_csv,
)
from analyze_phase_regularization_screen import finite_mean, read_env_metrics


ORDER = ("control", "speed_cap_10")
DISPLAY = {
    "control": "Zero-penalty control",
    "speed_cap_10": "Reward-stage speed cap (lambda=10)",
}
COLORS = {"control": "#777777", "speed_cap_10": "#1f77b4"}


def condition(flags):
    coefficient = float(flags["ds_stage_speed_coefficient"])
    if coefficient == 0.0:
        return "control"
    if coefficient == 10.0:
        return "speed_cap_10"
    raise ValueError(f"Unexpected speed-cap coefficient: {coefficient}")


def read_fixed_metrics(path):
    if not path.exists():
        return np.nan, np.nan
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one fixed-evaluation row in {path}")
    return float(rows[0]["success"]), float(rows[0]["episode_return"])


def read_training_metrics(path, count=10):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[-count:]
    result = {
        "tail_translation_norm": finite_mean(
            float(row["actor/translation_norm"]) for row in rows
        ),
        "tail_cap": finite_mean(float(row["actor/speed_cap"]) for row in rows),
        "tail_violation_fraction": finite_mean(
            float(row["actor/speed_cap_violation_fraction"]) for row in rows
        ),
        "tail_entropy": finite_mean(float(row["actor/entropy"]) for row in rows),
        "tail_q": finite_mean(float(row["actor/q"]) for row in rows),
    }
    for stage in range(3):
        stage_norms = [
            float(row[f"actor/translation_norm_stage{stage}"]) for row in rows
        ]
        # The training code used by this confirmation writes an exact zero when
        # a minibatch contains no sample from a late stage.  A continuous DS
        # policy has zero probability of producing an exact zero norm, so treat
        # those sentinel values as missing for stages 1 and 2.
        if stage > 0:
            stage_norms = [value for value in stage_norms if value > 0.0]
        result[f"tail_norm_stage{stage}"] = finite_mean(stage_norms)
        result[f"tail_violation_stage{stage}"] = finite_mean(
            float(row[f"actor/speed_cap_violation_stage{stage}"]) for row in rows
        )
    return result


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("ds_stage_speed_constraint") != "reward_stage_cap":
            continue
        eval_path = flags_path.with_name("eval.csv")
        training_path = flags_path.with_name("online_agent.csv")
        env_path = flags_path.with_name("env.csv")
        if not (eval_path.exists() and training_path.exists() and env_path.exists()):
            continue
        steps, success = read_curve(eval_path)
        if not len(steps) or steps[-1] != 500000:
            continue
        with eval_path.open(newline="") as handle:
            eval_rows = list(csv.DictReader(handle))
        fixed_success, fixed_return = read_fixed_metrics(
            flags_path.with_name("eval_fixed_seed.csv")
        )
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
                "fixed_success": fixed_success,
                "final_return": float(eval_rows[-1]["episode.return"]),
                "fixed_return": fixed_return,
                **read_env_metrics(env_path, int(flags["start_training"])),
                **read_training_metrics(training_path),
            }
        )
    return runs


def bootstrap_paired(differences, rng, samples=20000):
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    draws = differences[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


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
    expected_seeds = [14, 15, 16, 17]
    for name in ORDER:
        seeds = [run["seed"] for run in grouped[name]]
        if seeds != expected_seeds:
            raise SystemExit(f"Expected {name} seeds {expected_seeds}, found {seeds}")
    if any(not np.isfinite(run["fixed_success"]) for run in runs):
        raise SystemExit("Run reevaluate_checkpoint.py for every checkpoint first")

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
    write_csv(args.output_dir / "stage_speed_cap_formal_per_run.csv", per_run)

    metrics = tuple(
        key
        for key in per_run[0]
        if key not in ("condition", "seed") and not key.startswith("success_")
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
    write_csv(args.output_dir / "stage_speed_cap_formal_summary.csv", summary)

    rng = np.random.default_rng(20260925)
    paired = []
    paired_values = {}
    for metric in metrics:
        control = np.asarray([run[metric] for run in grouped["control"]])
        treatment = np.asarray([run[metric] for run in grouped["speed_cap_10"]])
        if not np.all(np.isfinite(control)) or not np.all(np.isfinite(treatment)):
            continue
        differences = treatment - control
        low, high = bootstrap_paired(differences, rng)
        paired_values[metric] = differences
        paired.append(
            {
                "contrast": "speed_cap_10-control",
                "metric": metric,
                "mean_delta": float(differences.mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                "num_positive": int(np.sum(differences > 0)),
                "num_negative": int(np.sum(differences < 0)),
                "num_tied": int(np.sum(differences == 0)),
                "paired_differences": ";".join(f"{x:.6g}" for x in differences),
            }
        )
    write_csv(args.output_dir / "stage_speed_cap_formal_paired.csv", paired)

    fixed_delta = paired_values["fixed_success"]
    auc_delta = paired_values["auc"]
    stage0_delta = paired_values["norm_stage0"]
    stage1_delta = paired_values["norm_stage1"]
    stage2_delta = paired_values["norm_stage2"]
    checks = {
        "fixed_success_gain_at_least_5pp": float(fixed_delta.mean()) >= 0.05,
        "at_least_3_of_4_fixed_pairs_improve": int(np.sum(fixed_delta > 0)) >= 3,
        "no_fixed_pair_drops_more_than_5pp": float(fixed_delta.min()) >= -0.05,
        "mean_success_auc_nonnegative": float(auc_delta.mean()) >= 0.0,
        "stage1_executed_norm_decreases": float(stage1_delta.mean()) < 0.0,
        "stage2_executed_norm_decreases": float(stage2_delta.mean()) < 0.0,
        "stage0_executed_norm_drop_below_0_05": float(stage0_delta.mean()) >= -0.05,
        "all_required_metrics_finite": all(
            np.all(np.isfinite(values))
            for values in (
                fixed_delta,
                auc_delta,
                stage0_delta,
                stage1_delta,
                stage2_delta,
            )
        ),
    }
    decision = {
        "decision": "promising" if all(checks.values()) else "not_promising",
        "checks": checks,
        "fixed_success_paired_differences": fixed_delta.tolist(),
        "auc_paired_differences": auc_delta.tolist(),
        "executed_norm_paired_differences": {
            "stage0": stage0_delta.tolist(),
            "stage1": stage1_delta.tolist(),
            "stage2": stage2_delta.tolist(),
        },
    }
    (args.output_dir / "stage_speed_cap_formal_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes = axes.ravel()
    for name in ORDER:
        curves = np.asarray([run["success"] for run in grouped[name]]) * 100
        steps = grouped[name][0]["steps"] / 1000
        for curve in curves:
            axes[0].plot(steps, curve, color=COLORS[name], alpha=0.2, linewidth=1)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1)
        axes[0].plot(
            steps, mean, marker="o", color=COLORS[name], linewidth=2.2,
            label=DISPLAY[name],
        )
        axes[0].fill_between(
            steps, np.maximum(0, mean - std), mean + std,
            color=COLORS[name], alpha=0.15,
        )
    axes[0].set(xlabel="Environment steps (k)", ylabel="Success rate (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    fixed = {
        name: np.asarray([run["fixed_success"] for run in grouped[name]]) * 100
        for name in ORDER
    }
    for index in range(4):
        axes[1].plot(
            [0, 1], [fixed["control"][index], fixed["speed_cap_10"][index]],
            color="#999999", alpha=0.65,
        )
    for index, name in enumerate(ORDER):
        axes[1].scatter(np.full(4, index), fixed[name], color=COLORS[name], s=45)
        axes[1].scatter(index, fixed[name].mean(), color="black", marker="D", s=50)
    axes[1].set_xticks([0, 1], ["Control", "Speed cap"])
    axes[1].set_ylabel("Fixed reevaluation success (%)")

    x = np.arange(3)
    width = 0.35
    for index, name in enumerate(ORDER):
        values = [summary[index][f"norm_stage{stage}_mean"] for stage in range(3)]
        axes[2].bar(
            x + (index - 0.5) * width, values, width,
            color=COLORS[name], label=DISPLAY[name],
        )
    axes[2].set_xticks(x, ["0 complete", "1 complete", "2 complete"])
    axes[2].set_ylabel("Executed translation norm")
    axes[2].legend()

    for name in ORDER:
        index = ORDER.index(name)
        values = [summary[index][f"tail_norm_stage{stage}_mean"] for stage in range(3)]
        axes[3].plot(x, values, marker="o", color=COLORS[name], label=DISPLAY[name])
    axes[3].set_xticks(x, ["0 complete", "1 complete", "2 complete"])
    axes[3].set_ylabel("Actor-batch translation norm")
    axes[3].legend()
    figure.suptitle("Reward-stage speed-cap confirmation (4 fresh paired seeds)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "stage_speed_cap_formal.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
