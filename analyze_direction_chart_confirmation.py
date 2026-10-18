#!/usr/bin/env python3
"""Analyze fresh-seed confirmation and pooled direction-chart evidence."""

import argparse
import csv
import json
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np

from analyze_klbudget_screen import (
    exact_sign_flip_pvalue,
    read_action_norm,
    read_curve,
    read_fixed_success,
    read_training_tail,
    write_csv,
)


ORDER = ("spherical", "stereographic")
DISPLAY = {"spherical": "Spherical", "stereographic": "Stereographic"}
COLORS = {"spherical": "#2ca02c", "stereographic": "#17becf"}
METRICS = (
    "final_success",
    "tail3_success",
    "auc",
    "fixed_success",
    "norm_mean",
    "norm_p95",
    "tail_alpha",
    "tail_entropy",
)


def load_chart_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        chart = flags.get("ds_integrated_type")
        if chart not in ORDER or flags.get("ds_regularizer") != "semantic_product":
            continue
        eval_path = flags_path.with_name("eval.csv")
        if not eval_path.exists():
            continue
        steps, success = read_curve(eval_path)
        if not len(steps) or steps[-1] != 500000:
            continue
        auc = np.trapezoid(
            np.concatenate(([0.0], success)), np.concatenate(([0], steps))
        ) / steps[-1]
        norm_mean, norm_p95 = read_action_norm(
            flags_path.with_name("env.csv"), int(flags["start_training"])
        )
        runs.append(
            {
                "chart": chart,
                "seed": int(flags["seed"]),
                "steps": steps,
                "success": success,
                "final_success": float(success[-1]),
                "tail3_success": float(success[-3:].mean()),
                "auc": float(auc),
                "fixed_success": read_fixed_success(
                    flags_path.with_name("eval_fixed_seed.csv")
                ),
                "norm_mean": norm_mean,
                "norm_p95": norm_p95,
                **read_training_tail(flags_path.with_name("online_agent.csv")),
            }
        )
    return runs


def validate_pairs(runs, expected_seeds, label):
    grouped = {
        chart: sorted(
            [run for run in runs if run["chart"] == chart],
            key=lambda run: run["seed"],
        )
        for chart in ORDER
    }
    for chart in ORDER:
        seeds = [run["seed"] for run in grouped[chart]]
        if seeds != list(expected_seeds):
            raise SystemExit(
                f"{label}: expected {chart} seeds {list(expected_seeds)}, found {seeds}"
            )
    return grouped


def summarize(grouped, cohort):
    rows = []
    for chart in ORDER:
        row = {
            "cohort": cohort,
            "chart": chart,
            "num_seeds": len(grouped[chart]),
        }
        for metric in METRICS:
            values = np.asarray([run[metric] for run in grouped[chart]])
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        rows.append(row)
    return rows


def paired_statistics(grouped, cohort, rng):
    rows = []
    for metric in ("final_success", "tail3_success", "auc", "fixed_success"):
        control = np.asarray([run[metric] for run in grouped["spherical"]])
        treatment = np.asarray([run[metric] for run in grouped["stereographic"]])
        if not np.all(np.isfinite(control)) or not np.all(np.isfinite(treatment)):
            continue
        differences = treatment - control
        indices = rng.integers(0, len(differences), size=(20000, len(differences)))
        bootstrap = differences[indices].mean(axis=1)
        rows.append(
            {
                "cohort": cohort,
                "contrast": "stereographic-spherical",
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
    return rows


def per_run_rows(runs, cohort):
    rows = []
    for run in sorted(runs, key=lambda item: (ORDER.index(item["chart"]), item["seed"])):
        row = {
            "cohort": cohort,
            **{key: value for key, value in run.items() if key not in ("steps", "success")},
        }
        row.update(
            {
                f"success_{step // 1000}k": value
                for step, value in zip(run["steps"], run["success"], strict=True)
            }
        )
        rows.append(row)
    return rows


def plot_cohorts(discovery, confirmation, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, (cohort, grouped) in zip(
        axes,
        (("Discovery: seeds 0--3", discovery), ("Confirmation: seeds 4--7", confirmation)),
        strict=True,
    ):
        for chart in ORDER:
            curves = np.asarray([run["success"] for run in grouped[chart]]) * 100
            steps = grouped[chart][0]["steps"] / 1000
            axis.plot(
                steps,
                curves.mean(axis=0),
                marker="o",
                linewidth=2.2,
                color=COLORS[chart],
                label=DISPLAY[chart],
            )
            axis.fill_between(
                steps,
                curves.mean(axis=0) - curves.std(axis=0, ddof=1),
                curves.mean(axis=0) + curves.std(axis=0, ddof=1),
                color=COLORS[chart],
                alpha=0.16,
            )
        axis.set_title(cohort)
        axis.set_xlabel("Environment steps (k)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend()
    figure.suptitle("Fresh-seed confirmation of the direction-chart effect")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("discovery_spherical_root", type=Path)
    parser.add_argument("discovery_stereographic_root", type=Path)
    parser.add_argument("confirmation_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()

    discovery_runs = (
        load_chart_runs(args.discovery_spherical_root)
        + load_chart_runs(args.discovery_stereographic_root)
    )
    confirmation_runs = load_chart_runs(args.confirmation_root)
    discovery = validate_pairs(discovery_runs, range(4), "discovery")
    confirmation = validate_pairs(confirmation_runs, range(4, 8), "confirmation")
    pooled_runs = discovery_runs + confirmation_runs
    pooled = validate_pairs(pooled_runs, range(8), "pooled")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "direction_chart_confirmation_per_run.csv",
        per_run_rows(discovery_runs, "discovery")
        + per_run_rows(confirmation_runs, "confirmation"),
    )
    summary = (
        summarize(discovery, "discovery")
        + summarize(confirmation, "confirmation")
        + summarize(pooled, "pooled")
    )
    write_csv(args.output_dir / "direction_chart_confirmation_summary.csv", summary)

    rng = np.random.default_rng(20260925)
    paired = (
        paired_statistics(discovery, "discovery", rng)
        + paired_statistics(confirmation, "confirmation", rng)
        + paired_statistics(pooled, "pooled", rng)
    )
    write_csv(args.output_dir / "direction_chart_confirmation_paired.csv", paired)
    plot_cohorts(
        discovery,
        confirmation,
        args.output_dir / "direction_chart_confirmation.png",
    )

    for row in summary:
        print(row)
    for row in paired:
        print(row)


if __name__ == "__main__":
    main()
