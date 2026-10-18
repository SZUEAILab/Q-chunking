#!/usr/bin/env python3
"""Analyze the matched-forward reference-measure and temperature ablations."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


ORDER = ("V1", "S1", "S3")
DISPLAY = {
    "V1": "Action volume / single alpha",
    "S1": "Semantic product / single alpha",
    "S3": "Semantic product / separate alpha",
}
COLORS = {"V1": "#9467bd", "S1": "#2ca02c", "S3": "#ff7f0e"}


def condition(flags):
    reference = flags["ds_regularizer"]
    separate = bool(flags["ds_separate_alpha"])
    if reference == "action_volume" and not separate:
        return "V1"
    if reference == "semantic_product":
        return "S3" if separate else "S1"
    raise ValueError(f"Unexpected matched-reference condition: {reference}/{separate}")


def read_curve(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([int(float(row["step"])) for row in rows]),
        np.asarray([float(row["success"]) for row in rows]),
    )


def read_action_stats(path, start_training):
    norms = []
    at_bound = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["step"])) <= start_training:
                continue
            norms.append(float(row["action_xyz_norm"]))
            at_bound.append(float(row["action_xyz_at_unit_bound"]))
    norms = np.asarray(norms)
    return {
        "norm_mean": float(norms.mean()),
        "norm_p50": float(np.quantile(norms, 0.50)),
        "norm_p95": float(np.quantile(norms, 0.95)),
        "ge_unit_frac": float(np.mean(at_bound)),
    }


def read_training_tail(path, tail=10):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[-tail:]
    keys = (
        "actor/alpha",
        "actor/entropy",
        "actor/alpha_direction",
        "actor/alpha_speed",
        "actor/alpha_scalar",
        "actor/entropy_direction",
        "actor/entropy_speed",
        "actor/entropy_scalar",
    )
    stats = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and row[key] != ""]
        stats[f"tail_{key.removeprefix('actor/').replace('/', '_')}"] = (
            float(np.mean(values)) if values else np.nan
        )
    return stats


def read_fixed_success(path):
    if not path.exists():
        return np.nan
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return float(rows[-1]["success"]) if rows else np.nan


def load_runs(root):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("ds_mode") != "integrated":
            continue
        if flags.get("ds_integrated_type") != "spherical":
            continue
        eval_path = flags_path.with_name("eval.csv")
        env_path = flags_path.with_name("env.csv")
        train_path = flags_path.with_name("online_agent.csv")
        if not eval_path.exists() or not env_path.exists() or not train_path.exists():
            continue
        steps, success = read_curve(eval_path)
        if not len(steps) or steps[-1] != 500000:
            continue
        auc = np.trapezoid(
            np.concatenate(([0.0], success)), np.concatenate(([0], steps))
        ) / steps[-1]
        runs.append(
            {
                "condition": condition(flags),
                "seed": int(flags["seed"]),
                "steps": steps,
                "success": success,
                "final_success": float(success[-1]),
                "tail3_success": float(success[-3:].mean()),
                "auc": float(auc),
                "fixed_success": read_fixed_success(
                    flags_path.with_name("eval_fixed_seed.csv")
                ),
                **read_action_stats(env_path, int(flags["start_training"])),
                **read_training_tail(train_path),
            }
        )
    return runs


def sign_flip_pvalue(diff):
    observed = abs(np.mean(diff))
    values = [
        abs(np.mean(diff * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(diff))
    ]
    return float(np.mean(np.asarray(values) >= observed - 1e-12))


def summarize(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    metrics = (
        "final_success",
        "tail3_success",
        "auc",
        "fixed_success",
        "norm_mean",
        "norm_p95",
        "ge_unit_frac",
        "tail_alpha",
        "tail_entropy",
        "tail_alpha_direction",
        "tail_alpha_speed",
        "tail_alpha_scalar",
        "tail_entropy_direction",
        "tail_entropy_speed",
        "tail_entropy_scalar",
    )
    rows = []
    for name in ORDER:
        values = grouped[name]
        row = {"condition": name, "display_name": DISPLAY[name], "num_seeds": len(values)}
        for metric in metrics:
            samples = np.asarray([run[metric] for run in values], dtype=float)
            finite = samples[np.isfinite(samples)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        rows.append(row)
    return rows


def paired(runs, bootstrap_samples=20000):
    lookup = {(run["condition"], run["seed"]): run for run in runs}
    rng = np.random.default_rng(20260925)
    rows = []
    for condition_name, reference_name in (("S1", "V1"), ("S3", "S1")):
        seeds = sorted(
            set(run["seed"] for run in runs if run["condition"] == condition_name)
            & set(run["seed"] for run in runs if run["condition"] == reference_name)
        )
        for metric in ("final_success", "tail3_success", "auc", "fixed_success"):
            diff = np.asarray(
                [
                    lookup[(condition_name, seed)][metric]
                    - lookup[(reference_name, seed)][metric]
                    for seed in seeds
                ]
            )
            if not np.all(np.isfinite(diff)):
                continue
            indices = rng.integers(0, len(diff), size=(bootstrap_samples, len(diff)))
            boot = diff[indices].mean(axis=1)
            rows.append(
                {
                    "condition": condition_name,
                    "reference": reference_name,
                    "metric": metric,
                    "num_pairs": len(diff),
                    "mean_delta": float(diff.mean()),
                    "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                    "exact_sign_flip_p": sign_flip_pvalue(diff),
                    "paired_differences": ";".join(f"{value:.6g}" for value in diff),
                }
            )
    return rows


def per_run_rows(runs):
    rows = []
    for run in sorted(runs, key=lambda item: (ORDER.index(item["condition"]), item["seed"])):
        row = {key: value for key, value in run.items() if key not in ("steps", "success")}
        row.update(
            {
                f"success_{step // 1000}k": value
                for step, value in zip(run["steps"], run["success"], strict=True)
            }
        )
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(runs, path, projection_root=None):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    if projection_root is not None:
        projection_curves = []
        projection_steps = None
        for flags_path in projection_root.rglob("flags.json"):
            flags = json.loads(flags_path.read_text())
            if flags.get("ds_mode") != "projection":
                continue
            steps, success = read_curve(flags_path.with_name("eval.csv"))
            projection_steps = steps
            projection_curves.append(success)
        if projection_curves:
            curves = np.asarray(projection_curves) * 100
            axis.plot(
                projection_steps / 1000,
                curves.mean(axis=0),
                marker="o",
                linewidth=1.8,
                linestyle=":",
                color="#777777",
                label="Cartesian + L2 projection (context)",
            )
    for name in ORDER:
        values = grouped[name]
        curves = np.asarray([run["success"] for run in values]) * 100
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
        x = values[0]["steps"] / 1000
        axis.plot(
            x,
            mean,
            marker="o",
            linewidth=2.3,
            color=COLORS[name],
            label=DISPLAY[name],
        )
        axis.fill_between(x, mean - sem, mean + sem, color=COLORS[name], alpha=0.15)
    axis.set(xlabel="Environment steps (k)", ylabel="Success rate (%)", ylim=(-2, 102))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_action_norms(runs, path):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)
    rng = np.random.default_rng(20260925)
    for axis, metric, ylabel in (
        (axes[0], "norm_mean", "Mean executed XYZ norm"),
        (axes[1], "norm_p95", "95th-percentile XYZ norm"),
    ):
        for index, name in enumerate(ORDER):
            values = np.asarray([run[metric] for run in grouped[name]])
            jitter = rng.uniform(-0.055, 0.055, size=len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                color=COLORS[name],
                alpha=0.75,
                s=28,
            )
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                color=COLORS[name],
                marker="o",
                markersize=7,
                capsize=4,
                linewidth=2,
            )
        axis.axhline(1.0, color="#777777", linestyle=":", linewidth=1)
        axis.set_xticks(range(len(ORDER)), ORDER)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Executed actions after the 5k random warm-up")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--projection-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    runs = load_runs(args.root)
    counts = {name: sum(run["condition"] == name for run in runs) for name in ORDER}
    if any(count != 4 for count in counts.values()):
        raise SystemExit(f"Expected four complete seeds per condition, got {counts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs)
    write_csv(args.output_dir / "matched_reference_500k_summary.csv", summary)
    write_csv(args.output_dir / "matched_reference_500k_paired.csv", paired(runs))
    write_csv(args.output_dir / "matched_reference_500k_per_run.csv", per_run_rows(runs))
    plot(
        runs,
        args.output_dir / "matched_reference_500k_curves.png",
        projection_root=args.projection_root,
    )
    plot_action_norms(
        runs, args.output_dir / "matched_reference_500k_action_norms.png"
    )
    for row in summary:
        print(
            f"{row['display_name']:<36} "
            f"final={100 * row['final_success_mean']:.1f}+-"
            f"{100 * row['final_success_std']:.1f} "
            f"tail3={100 * row['tail3_success_mean']:.1f} "
            f"AUC={100 * row['auc_mean']:.1f} "
            f"fixed100={100 * row['fixed_success_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
