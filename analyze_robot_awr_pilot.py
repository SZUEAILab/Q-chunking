#!/usr/bin/env python3
"""Analyze the robot replay-AWR and stage-balanced discovery pilot."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


EXPECTED = (
    "cart_sac_uniform",
    "cart_awr_balanced_t03",
    "ds_sac_uniform",
    "ds_awr_uniform_t03",
    "ds_sac_balanced",
    "ds_awr_balanced_t01",
    "ds_awr_balanced_t03",
    "ds_awr_balanced_t10",
)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def condition_name(flags):
    ds = flags["ds_mode"] == "integrated"
    awr = flags["actor_update_mode"] == "replay_awr"
    balanced = bool(flags["stage_balanced_sampling"])
    if not ds:
        return "cart_awr_balanced_t03" if awr else "cart_sac_uniform"
    if not awr:
        return "ds_sac_balanced" if balanced else "ds_sac_uniform"
    temperature = float(flags["awr_temperature"])
    if not balanced:
        return "ds_awr_uniform_t03"
    return f"ds_awr_balanced_t{int(round(temperature * 10)):02d}"


def finite_tail(rows, key, count=10):
    values = np.asarray([float(row[key]) for row in rows[-count:]], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("run_group") != "robot_awr_balanced_pilot_100k_20260925":
            continue
        eval_path = flags_path.with_name("eval.csv")
        env_path = flags_path.with_name("env.csv")
        train_path = flags_path.with_name("online_agent.csv")
        if not (eval_path.exists() and env_path.exists() and train_path.exists()):
            continue
        eval_rows = read_rows(eval_path)
        if not eval_rows or int(eval_rows[-1]["step"]) != 100000:
            continue
        steps = np.asarray([int(row["step"]) for row in eval_rows])
        success = np.asarray([float(row["success"]) for row in eval_rows])
        returns = np.asarray([float(row["episode.return"]) for row in eval_rows])
        normalized_return = np.clip((returns + 3000.0) / 3000.0, 0.0, 1.0)
        success_auc = float(
            np.trapezoid(np.r_[0.0, success], np.r_[0, steps]) / steps[-1]
        )
        return_auc = float(
            np.trapezoid(np.r_[0.0, normalized_return], np.r_[0, steps])
            / steps[-1]
        )
        env_rows = read_rows(env_path)
        warmup = int(flags["start_training"])
        env_rows = [row for row in env_rows if int(row["step"]) >= warmup]
        stage_rewards = np.asarray([float(row["stage_reward"]) for row in env_rows])
        norms = np.asarray([float(row["action_xyz_norm"]) for row in env_rows])
        train_rows = read_rows(train_path)
        awr = flags["actor_update_mode"] == "replay_awr"
        run = {
            "condition": condition_name(flags),
            "seed": int(flags["seed"]),
            "final_success": float(success[-1]),
            "success_auc": success_auc,
            "final_return": float(returns[-1]),
            "return_auc": return_auc,
            "stage_beyond_zero_fraction": float(np.mean(stage_rewards > -3.0)),
            "stage_two_fraction": float(np.mean(stage_rewards > -2.0)),
            "translation_norm": float(norms.mean()),
            "finite_training_metrics": bool(
                all(
                    np.isfinite(float(value))
                    for row in train_rows
                    for key, value in row.items()
                    if key != "step"
                )
            ),
            "tail_critic_loss": finite_tail(train_rows, "critic/critic_loss"),
            "tail_q": finite_tail(train_rows, "actor/q"),
            "tail_awr_weight_mean": (
                finite_tail(train_rows, "actor/awr_weight_mean") if awr else np.nan
            ),
            "tail_awr_weight_max": (
                finite_tail(train_rows, "actor/awr_weight_max") if awr else np.nan
            ),
        }
        for step, value in zip(steps, success, strict=True):
            run[f"success_{step // 1000}k"] = float(value)
        for step, value in zip(steps, normalized_return, strict=True):
            run[f"normalized_return_{step // 1000}k"] = float(value)
        runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    runs = load_runs(args.root)
    grouped = {
        condition: sorted(
            [run for run in runs if run["condition"] == condition],
            key=lambda run: run["seed"],
        )
        for condition in EXPECTED
    }
    for condition in EXPECTED:
        found = [run["seed"] for run in grouped[condition]]
        if found != [40, 41]:
            raise SystemExit(f"Expected {condition} seeds [40, 41], found {found}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(runs, key=lambda run: (EXPECTED.index(run["condition"]), run["seed"]))
    write_csv(args.output_dir / "robot_awr_balanced_pilot_per_run.csv", runs)

    numeric = [
        key
        for key in runs[0]
        if key not in ("condition", "seed", "finite_training_metrics")
        and not key.startswith("success_")
    ]
    summary = []
    for condition in EXPECTED:
        item = {"condition": condition, "num_seeds": 2}
        for metric in numeric:
            values = np.asarray([run[metric] for run in grouped[condition]], dtype=float)
            finite = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            item[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
        summary.append(item)
    write_csv(args.output_dir / "robot_awr_balanced_pilot_summary.csv", summary)

    candidates = (
        "ds_awr_balanced_t01",
        "ds_awr_balanced_t03",
        "ds_awr_balanced_t10",
    )
    valid = []
    for condition in candidates:
        mean_weights = np.asarray(
            [run["tail_awr_weight_mean"] for run in grouped[condition]]
        )
        finite = all(run["finite_training_metrics"] for run in grouped[condition])
        if finite and np.all((mean_weights >= 0.99) & (mean_weights <= 1.01)):
            valid.append(condition)
    best_auc = max(
        np.mean([run["return_auc"] for run in grouped[condition]])
        for condition in valid
    )
    tied = [
        condition
        for condition in valid
        if np.mean([run["return_auc"] for run in grouped[condition]])
        >= best_auc * 0.98
    ]
    selected = max(tied, key=lambda name: int(name[-2:])) if tied else None

    control = {run["seed"]: run for run in grouped["ds_sac_uniform"]}
    treatment = {run["seed"]: run for run in grouped[selected]} if selected else {}
    success_delta = np.asarray(
        [treatment[seed]["final_success"] - control[seed]["final_success"] for seed in sorted(control)]
    ) if selected else np.asarray([])
    return_delta = np.asarray(
        [treatment[seed]["return_auc"] - control[seed]["return_auc"] for seed in sorted(control)]
    ) if selected else np.asarray([])
    stage_delta = np.asarray(
        [treatment[seed]["stage_beyond_zero_fraction"] - control[seed]["stage_beyond_zero_fraction"] for seed in sorted(control)]
    ) if selected else np.asarray([])
    gates = {
        "success_gate": bool(len(success_delta) and success_delta.mean() >= 0.02),
        "return_auc_gate": bool(
            len(return_delta)
            and return_delta.mean() >= 0.05
            and np.all(return_delta > 0)
        ),
        "stage_progression_gate": bool(
            len(stage_delta)
            and stage_delta.mean() >= 0.05
            and np.all(stage_delta > 0)
        ),
    }
    decision = {
        "purpose": "pilot_selection_not_confirmatory_evidence",
        "selected_condition": selected,
        "selected_temperature": (
            float(selected[-2:]) / 10.0 if selected is not None else None
        ),
        "valid_candidates": valid,
        "success_differences": success_delta.tolist(),
        "return_auc_differences": return_delta.tolist(),
        "stage_progression_differences": stage_delta.tolist(),
        **gates,
        "launch_confirmation": bool(any(gates.values())),
    }
    (args.output_dir / "robot_awr_balanced_pilot_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
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
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=6)
    x = np.arange(len(EXPECTED))
    axes[1].bar(x, [row["return_auc_mean"] for row in summary])
    axes[1].set_xticks(x, EXPECTED, rotation=55, ha="right", fontsize=7)
    axes[1].set_ylabel("Normalized-return AUC")
    axes[2].bar(
        x,
        [100 * row["stage_beyond_zero_fraction_mean"] for row in summary],
    )
    axes[2].set_xticks(x, EXPECTED, rotation=55, ha="right", fontsize=7)
    axes[2].set_ylabel("Online states beyond stage zero (%)")
    figure.suptitle("Robot AWR + stage-balanced replay pilot (2 tuning seeds)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "robot_awr_balanced_pilot.png", dpi=180)
    plt.close(figure)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
