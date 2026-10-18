#!/usr/bin/env python3
"""Analyze the frozen direction-only replay-support discovery pilot."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


CANDIDATES = {
    0.1: "ds_sac_direction_awr_c01",
    1.0: "ds_sac_direction_awr_c10",
    10.0: "ds_sac_direction_awr_c100",
}
EXPECTED = ("ds_sac_balanced", *CANDIDATES.values())


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def condition_name(flags):
    if flags["run_group"] == "robot_awr_balanced_pilot_100k_20260925":
        if (
            flags["ds_mode"] == "integrated"
            and flags["actor_update_mode"] == "sac"
            and flags["stage_balanced_sampling"]
        ):
            return "ds_sac_balanced"
        return None
    if flags["run_group"] == "robot_direction_support_pilot_100k_20260925":
        coefficient = float(flags["direction_awr_coefficient"])
        return CANDIDATES.get(coefficient)
    return None


def finite_tail(rows, key, count=10):
    values = np.asarray([float(row[key]) for row in rows[-count:]], dtype=float)
    return float(values.mean()) if np.all(np.isfinite(values)) else np.nan


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        condition = condition_name(flags)
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
        normalized_return = np.clip((returns + 3000.0) / 3000.0, 0.0, 1.0)
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
            "success_auc": float(
                np.trapezoid(np.r_[0.0, success], np.r_[0, steps]) / steps[-1]
            ),
            "final_return": float(returns[-1]),
            "return_auc": float(
                np.trapezoid(np.r_[0.0, normalized_return], np.r_[0, steps])
                / steps[-1]
            ),
            "stage_beyond_zero_fraction": float(np.mean(stages > -3.0)),
            "translation_norm": float(norms.mean()),
            "finite_training_metrics": finite,
            "tail_critic_loss": finite_tail(train_rows, "critic/critic_loss"),
            "tail_q": finite_tail(train_rows, "actor/q"),
            "tail_awr_weight_mean": (
                finite_tail(train_rows, "actor/awr_weight_mean")
                if condition != "ds_sac_balanced"
                else np.nan
            ),
        }
        for step, value in zip(steps, success, strict=True):
            run[f"success_{step // 1000}k"] = float(value)
        for step, value in zip(steps, normalized_return, strict=True):
            run[f"normalized_return_{step // 1000}k"] = float(value)
        runs.append(run)
    return runs


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    grouped = {
        condition: sorted(
            [run for run in load_runs(args.root) if run["condition"] == condition],
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
    write_csv(args.output_dir / "robot_direction_support_per_run.csv", runs)
    control = {run["seed"]: run for run in grouped["ds_sac_balanced"]}
    valid = []
    validity = {}
    for coefficient, condition in CANDIDATES.items():
        candidate = {run["seed"]: run for run in grouped[condition]}
        finite = all(run["finite_training_metrics"] for run in candidate.values())
        weights_ok = all(
            0.99 <= run["tail_awr_weight_mean"] <= 1.01
            for run in candidate.values()
        )
        norm_ok = all(
            candidate[seed]["translation_norm"]
            >= 0.9 * control[seed]["translation_norm"]
            for seed in control
        )
        validity[condition] = {
            "finite": finite,
            "weights_ok": weights_ok,
            "translation_norm_ok": norm_ok,
        }
        if finite and weights_ok and norm_ok:
            valid.append((coefficient, condition))

    if valid:
        best_auc = max(
            np.mean([run["return_auc"] for run in grouped[condition]])
            for _, condition in valid
        )
        tied = [
            (coefficient, condition)
            for coefficient, condition in valid
            if np.mean([run["return_auc"] for run in grouped[condition]])
            >= best_auc * 0.98
        ]
        selected_coefficient, selected = min(tied)
    else:
        selected_coefficient, selected = None, None

    if selected is not None:
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
        "selected_coefficient": selected_coefficient,
        "validity": validity,
        "success_differences": success_delta.tolist(),
        "return_auc_differences": return_delta.tolist(),
        "stage_progression_differences": stage_delta.tolist(),
        **gates,
        "launch_confirmation": bool(any(gates.values())),
    }
    (args.output_dir / "robot_direction_support_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    steps = np.asarray((25, 50, 75, 100))
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
    axes[1].bar(
        x,
        [np.mean([run["return_auc"] for run in grouped[c]]) for c in EXPECTED],
    )
    axes[1].set_ylabel("Normalized-return AUC")
    axes[2].bar(
        x,
        [
            100 * np.mean([run["stage_beyond_zero_fraction"] for run in grouped[c]])
            for c in EXPECTED
        ],
    )
    axes[2].set_ylabel("Online states beyond stage zero (%)")
    for axis in axes[1:]:
        axis.set_xticks(x, EXPECTED, rotation=45, ha="right", fontsize=7)
    figure.suptitle("Direction-only replay support discovery pilot")
    figure.tight_layout()
    figure.savefig(args.output_dir / "robot_direction_support_pilot.png", dpi=180)
    plt.close(figure)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
