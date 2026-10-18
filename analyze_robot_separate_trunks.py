#!/usr/bin/env python3
"""Analyze the parameter-isolated direction-support robot pilot."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from analyze_robot_direction_support import finite_tail, read_rows, write_csv


CONDITIONS = {
    ("sac", 0.0): "ds_separate_sac",
    ("sac_direction_awr", 0.1): "ds_separate_direction_awr_c01",
    ("sac_direction_awr", 1.0): "ds_separate_direction_awr_c10",
}
EXPECTED = tuple(CONDITIONS.values())


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("run_group") != "robot_separate_trunks_pilot_100k_20260925":
            continue
        condition = CONDITIONS.get(
            (flags["actor_update_mode"], float(flags["direction_awr_coefficient"]))
        )
        if condition is None:
            continue
        eval_path = flags_path.with_name("eval.csv")
        env_path = flags_path.with_name("env.csv")
        train_path = flags_path.with_name("online_agent.csv")
        if not all(path.exists() for path in (eval_path, env_path, train_path)):
            continue
        eval_rows = read_rows(eval_path)
        if not eval_rows or int(eval_rows[-1]["step"]) != 100000:
            continue
        steps = np.asarray([int(row["step"]) for row in eval_rows])
        success = np.asarray([float(row["success"]) for row in eval_rows])
        returns = np.asarray([float(row["episode.return"]) for row in eval_rows])
        normalized = np.clip((returns + 3000.0) / 3000.0, 0.0, 1.0)
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
            "condition": condition,
            "seed": int(flags["seed"]),
            "final_success": float(success[-1]),
            "return_auc": float(
                np.trapezoid(np.r_[0.0, normalized], np.r_[0, steps]) / steps[-1]
            ),
            "stage_beyond_zero_fraction": float(np.mean(stages > -3.0)),
            "translation_norm": float(norms.mean()),
            "finite_training_metrics": finite,
            "tail_critic_loss": finite_tail(train_rows, "critic/critic_loss"),
            "tail_awr_weight_mean": (
                finite_tail(train_rows, "actor/awr_weight_mean")
                if flags["actor_update_mode"] == "sac_direction_awr"
                else np.nan
            ),
        }
        for step, value in zip(steps, normalized, strict=True):
            run[f"normalized_return_{step // 1000}k"] = float(value)
        runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    loaded = load_runs(args.root)
    grouped = {
        condition: sorted(
            [run for run in loaded if run["condition"] == condition],
            key=lambda run: run["seed"],
        )
        for condition in EXPECTED
    }
    for condition in EXPECTED:
        seeds = [run["seed"] for run in grouped[condition]]
        if seeds != [40, 41]:
            raise SystemExit(f"Expected {condition} seeds [40, 41], found {seeds}")
    runs = [run for condition in EXPECTED for run in grouped[condition]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "robot_separate_trunks_per_run.csv", runs)

    control = {run["seed"]: run for run in grouped["ds_separate_sac"]}
    candidates = (
        (0.1, "ds_separate_direction_awr_c01"),
        (1.0, "ds_separate_direction_awr_c10"),
    )
    valid, validity = [], {}
    for coefficient, condition in candidates:
        treatment = {run["seed"]: run for run in grouped[condition]}
        checks = {
            "finite": all(run["finite_training_metrics"] for run in treatment.values()),
            "weights_ok": all(
                0.99 <= run["tail_awr_weight_mean"] <= 1.01
                for run in treatment.values()
            ),
            "translation_norm_ok": all(
                treatment[seed]["translation_norm"]
                >= 0.9 * control[seed]["translation_norm"]
                for seed in control
            ),
        }
        validity[condition] = checks
        if all(checks.values()):
            valid.append((coefficient, condition))
    if valid:
        best = max(
            np.mean([run["return_auc"] for run in grouped[condition]])
            for _, condition in valid
        )
        tied = [
            pair
            for pair in valid
            if np.mean([run["return_auc"] for run in grouped[pair[1]]])
            >= 0.98 * best
        ]
        coefficient, selected = min(tied)
        treatment = {run["seed"]: run for run in grouped[selected]}
        success_delta = np.asarray(
            [treatment[s]["final_success"] - control[s]["final_success"] for s in control]
        )
        return_delta = np.asarray(
            [treatment[s]["return_auc"] - control[s]["return_auc"] for s in control]
        )
        stage_delta = np.asarray(
            [
                treatment[s]["stage_beyond_zero_fraction"]
                - control[s]["stage_beyond_zero_fraction"]
                for s in control
            ]
        )
    else:
        coefficient = selected = None
        success_delta = return_delta = stage_delta = np.asarray([])
    gates = {
        "success_gate": bool(len(success_delta) and success_delta.mean() >= 0.02),
        "return_auc_gate": bool(
            len(return_delta) and return_delta.mean() >= 0.02 and np.all(return_delta > 0)
        ),
        "stage_progression_gate": bool(
            len(stage_delta) and stage_delta.mean() >= 0.01 and np.all(stage_delta > 0)
        ),
    }
    decision = {
        "purpose": "pilot_selection_not_confirmatory_evidence",
        "selected_condition": selected,
        "selected_coefficient": coefficient,
        "validity": validity,
        "success_differences": success_delta.tolist(),
        "return_auc_differences": return_delta.tolist(),
        "stage_progression_differences": stage_delta.tolist(),
        **gates,
        "launch_confirmation": bool(any(gates.values())),
    }
    (args.output_dir / "robot_separate_trunks_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )

    steps = np.asarray((25, 50, 75, 100))
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for condition in EXPECTED:
        curves = np.asarray(
            [
                [run[f"normalized_return_{step}k"] for step in steps]
                for run in grouped[condition]
            ]
        )
        axes[0].plot(steps, 100 * curves.mean(axis=0), marker="o", label=condition)
    axes[0].set(xlabel="Environment steps (k)", ylabel="Normalized return (%)")
    axes[0].legend(fontsize=7)
    x = np.arange(len(EXPECTED))
    axes[1].bar(x, [np.mean([r["return_auc"] for r in grouped[c]]) for c in EXPECTED])
    axes[1].set_ylabel("Normalized-return AUC")
    axes[2].bar(
        x,
        [100 * np.mean([r["stage_beyond_zero_fraction"] for r in grouped[c]]) for c in EXPECTED],
    )
    axes[2].set_ylabel("Online states beyond stage zero (%)")
    for axis in axes[1:]:
        axis.set_xticks(x, EXPECTED, rotation=35, ha="right", fontsize=7)
    figure.suptitle("Parameter-isolated direction-support pilot")
    figure.tight_layout()
    figure.savefig(args.output_dir / "robot_separate_trunks_pilot.png", dpi=180)
    plt.close(figure)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
