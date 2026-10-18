#!/usr/bin/env python3
"""Analyze the reward-stage translation speed-cap coefficient pilot."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from analyze_klbudget_screen import read_curve, write_csv
from analyze_phase_regularization_screen import finite_mean, read_env_metrics


COEFFICIENTS = (0.0, 10.0, 30.0, 100.0)
COLORS = dict(zip(COEFFICIENTS, ("#777777", "#1f77b4", "#ff7f0e", "#d62728")))


def read_training_metrics(path, count=10):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[-count:]
    return {
        "tail_translation_norm": finite_mean(
            float(row["actor/translation_norm"]) for row in rows
        ),
        "tail_cap": finite_mean(float(row["actor/speed_cap"]) for row in rows),
        "tail_violation_fraction": finite_mean(
            float(row["actor/speed_cap_violation_fraction"]) for row in rows
        ),
        "tail_weighted_penalty": finite_mean(
            float(row["actor/speed_cap_penalty"]) for row in rows
        ),
        "tail_entropy": finite_mean(float(row["actor/entropy"]) for row in rows),
        "tail_q": finite_mean(float(row["actor/q"]) for row in rows),
    }


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
        if not len(steps) or steps[-1] != 200000:
            continue
        with eval_path.open(newline="") as handle:
            eval_rows = list(csv.DictReader(handle))
        coefficient = float(flags["ds_stage_speed_coefficient"])
        auc = np.trapezoid(
            np.concatenate(([0.0], success)), np.concatenate(([0], steps))
        ) / steps[-1]
        runs.append(
            {
                "coefficient": coefficient,
                "seed": int(flags["seed"]),
                "steps": steps,
                "success": success,
                "final_success": float(success[-1]),
                "auc": float(auc),
                "final_return": float(eval_rows[-1]["episode.return"]),
                **read_env_metrics(env_path, int(flags["start_training"])),
                **read_training_metrics(training_path),
            }
        )
    return runs


def paired_rows(grouped):
    control = {run["seed"]: run for run in grouped[0.0]}
    metrics = (
        "final_success",
        "auc",
        "final_return",
        "norm_stage0",
        "norm_stage1",
        "norm_stage2",
        "tail_translation_norm",
        "tail_violation_fraction",
    )
    rows = []
    for coefficient in COEFFICIENTS[1:]:
        treatment = {run["seed"]: run for run in grouped[coefficient]}
        for metric in metrics:
            differences = np.asarray(
                [treatment[seed][metric] - control[seed][metric] for seed in sorted(control)]
            )
            finite = differences[np.isfinite(differences)]
            rows.append(
                {
                    "coefficient": coefficient,
                    "contrast": f"coefficient_{coefficient:g}-coefficient_0",
                    "metric": metric,
                    "mean_delta": float(finite.mean()) if len(finite) else np.nan,
                    "num_lower": int(np.sum(finite < 0)),
                    "num_equal": int(np.sum(finite == 0)),
                    "paired_differences": ";".join(f"{x:.6g}" for x in finite),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()

    runs = load_runs(args.root)
    grouped = {
        coefficient: sorted(
            [run for run in runs if run["coefficient"] == coefficient],
            key=lambda run: run["seed"],
        )
        for coefficient in COEFFICIENTS
    }
    expected_seeds = [12, 13]
    for coefficient in COEFFICIENTS:
        seeds = [run["seed"] for run in grouped[coefficient]]
        if seeds != expected_seeds:
            raise SystemExit(
                f"Expected coefficient {coefficient:g} seeds {expected_seeds}, found {seeds}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run = []
    for run in sorted(runs, key=lambda item: (item["coefficient"], item["seed"])):
        row = {key: value for key, value in run.items() if key not in ("steps", "success")}
        row.update(
            {
                f"success_{step // 1000}k": value
                for step, value in zip(run["steps"], run["success"], strict=True)
            }
        )
        per_run.append(row)
    write_csv(args.output_dir / "stage_speed_cap_pilot_per_run.csv", per_run)

    metrics = tuple(
        key
        for key in per_run[0]
        if key not in ("coefficient", "seed") and not key.startswith("success_")
    )
    summary = []
    for coefficient in COEFFICIENTS:
        row = {"coefficient": coefficient, "num_seeds": len(expected_seeds)}
        for metric in metrics:
            values = np.asarray([run[metric] for run in grouped[coefficient]], dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        summary.append(row)
    write_csv(args.output_dir / "stage_speed_cap_pilot_summary.csv", summary)
    paired = paired_rows(grouped)
    write_csv(args.output_dir / "stage_speed_cap_pilot_paired.csv", paired)

    # This is a tuning diagnostic, not a hypothesis test.  Choose the smallest
    # coefficient that preserves mean final success (within 5 pp) and achieves
    # at least 80% of the largest observed late-stage norm reduction.
    control_success = summary[0]["final_success_mean"]
    control_by_seed = {run["seed"]: run for run in grouped[0.0]}
    candidates = []
    for row in summary[1:]:
        treatment_by_seed = {
            run["seed"]: run for run in grouped[row["coefficient"]]
        }
        stage_reductions = []
        stage_pair_counts = []
        for stage in (1, 2):
            reductions = np.asarray(
                [
                    control_by_seed[seed][f"norm_stage{stage}"]
                    - treatment_by_seed[seed][f"norm_stage{stage}"]
                    for seed in sorted(control_by_seed)
                ]
            )
            finite = reductions[np.isfinite(reductions)]
            stage_pair_counts.append(int(len(finite)))
            if len(finite):
                stage_reductions.append(float(finite.mean()))
        late_reduction = (
            float(np.mean(stage_reductions)) if stage_reductions else -np.inf
        )
        preserves_success = row["final_success_mean"] >= control_success - 0.05
        candidates.append(
            (
                row["coefficient"],
                late_reduction,
                preserves_success,
                stage_pair_counts,
            )
        )
    viable_reductions = [
        reduction for _, reduction, viable, _ in candidates if viable
    ]
    max_reduction = max(viable_reductions) if viable_reductions else -np.inf
    selected = next(
        (
            coefficient
            for coefficient, reduction, viable, _ in candidates
            if viable and reduction >= 0.8 * max_reduction and reduction > 0.0
        ),
        None,
    )
    decision = {
        "purpose": "pilot_hyperparameter_selection_not_formal_evidence",
        "rule": "smallest coefficient preserving mean success within 5 pp and reaching 80% of the best mean stage-1/2 norm reduction",
        "selected_coefficient": selected,
        "control_final_success": control_success,
        "candidates": [
            {
                "coefficient": coefficient,
                "late_norm_reduction": reduction,
                "preserves_success": viable,
                "stage1_finite_pairs": pair_counts[0],
                "stage2_finite_pairs": pair_counts[1],
            }
            for coefficient, reduction, viable, pair_counts in candidates
        ],
    }
    (args.output_dir / "stage_speed_cap_pilot_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes = axes.ravel()
    for coefficient in COEFFICIENTS:
        curves = np.asarray([run["success"] for run in grouped[coefficient]]) * 100
        steps = grouped[coefficient][0]["steps"] / 1000
        axes[0].plot(
            steps,
            curves.mean(axis=0),
            marker="o",
            color=COLORS[coefficient],
            label=f"coefficient={coefficient:g}",
        )
    axes[0].set(xlabel="Environment steps (k)", ylabel="Success rate (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    x = np.arange(3)
    width = 0.19
    for index, coefficient in enumerate(COEFFICIENTS):
        values = [summary[index][f"norm_stage{stage}_mean"] for stage in range(3)]
        axes[1].bar(
            x + (index - 1.5) * width,
            values,
            width,
            color=COLORS[coefficient],
            label=f"{coefficient:g}",
        )
    axes[1].set_xticks(x, ["0 complete", "1 complete", "2 complete"])
    axes[1].set_ylabel("Executed translation norm")
    axes[1].set_title("Online actions by task stage")

    axes[2].plot(
        COEFFICIENTS,
        [row["tail_violation_fraction_mean"] * 100 for row in summary],
        marker="o",
    )
    axes[2].set(xlabel="Penalty coefficient", ylabel="Cap violation fraction (%)")
    axes[2].set_xscale("symlog", linthresh=1.0)

    axes[3].plot(
        COEFFICIENTS,
        [row["final_return_mean"] for row in summary],
        marker="o",
    )
    axes[3].set(xlabel="Penalty coefficient", ylabel="Final episode return")
    axes[3].set_xscale("symlog", linthresh=1.0)
    figure.suptitle("Reward-stage speed-cap pilot (2 paired seeds; tuning only)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "stage_speed_cap_pilot.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
