#!/usr/bin/env python3
"""Compare latent-action and environment-action critics for DS adapters."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


ORDER = ("latent_cube", "env_cube", "latent_sphere", "env_sphere")
DISPLAY = {
    "latent_cube": "DS cube / latent-Q",
    "env_cube": "DS cube / env-Q",
    "latent_sphere": "DS sphere / latent-Q",
    "env_sphere": "DS sphere / env-Q",
}
COLORS = {
    "latent_cube": "#1f77b4",
    "env_cube": "#17becf",
    "latent_sphere": "#ff7f0e",
    "env_sphere": "#2ca02c",
}


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


def load(root, critic_space):
    runs = []
    for flags_path in sorted(root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("ds_mode") != "adapter":
            continue
        if flags.get("ds_adapter_speed_interpretation") != "direct":
            continue
        if flags.get("ds_adapter_critic_space", "latent") != critic_space:
            continue
        bound = flags.get("ds_speed_bound")
        if bound not in ("cube", "sphere"):
            continue
        eval_path = flags_path.with_name("eval.csv")
        env_path = flags_path.with_name("env.csv")
        if not eval_path.exists() or not eval_path.stat().st_size:
            continue
        steps, success = read_curve(eval_path)
        if not len(steps) or steps[-1] != 500000:
            continue
        auc = np.trapezoid(
            np.concatenate(([0.0], success)),
            np.concatenate(([0], steps)),
        ) / steps[-1]
        runs.append(
            {
                "condition": f"{critic_space}_{bound}",
                "critic_space": critic_space,
                "bound": bound,
                "seed": int(flags["seed"]),
                "steps": steps,
                "success": success,
                "final_success": float(success[-1]),
                "tail3_success": float(success[-3:].mean()),
                "auc": float(auc),
                **read_action_stats(env_path, int(flags["start_training"])),
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
    rows = []
    for name in ORDER:
        values = grouped[name]
        row = {"condition": name, "display_name": DISPLAY[name], "num_seeds": len(values)}
        for metric in (
            "final_success",
            "tail3_success",
            "auc",
            "norm_mean",
            "norm_p95",
            "ge_unit_frac",
        ):
            samples = np.asarray([run[metric] for run in values])
            row[f"{metric}_mean"] = float(samples.mean())
            row[f"{metric}_std"] = float(samples.std(ddof=1))
        rows.append(row)
    return rows


def paired(runs, bootstrap_samples=20000):
    lookup = {(run["condition"], run["seed"]): run for run in runs}
    rng = np.random.default_rng(20260925)
    rows = []
    for bound in ("cube", "sphere"):
        seeds = sorted(
            set(run["seed"] for run in runs if run["condition"] == f"latent_{bound}")
            & set(run["seed"] for run in runs if run["condition"] == f"env_{bound}")
        )
        for metric in ("final_success", "tail3_success", "auc"):
            diff = np.asarray(
                [
                    lookup[(f"env_{bound}", seed)][metric]
                    - lookup[(f"latent_{bound}", seed)][metric]
                    for seed in seeds
                ]
            )
            indices = rng.integers(0, len(diff), size=(bootstrap_samples, len(diff)))
            boot = diff[indices].mean(axis=1)
            rows.append(
                {
                    "bound": bound,
                    "metric": metric,
                    "num_pairs": len(diff),
                    "env_minus_latent": float(diff.mean()),
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
        row = {
            "condition": run["condition"],
            "display_name": DISPLAY[run["condition"]],
            "seed": run["seed"],
            "final_success": run["final_success"],
            "tail3_success": run["tail3_success"],
            "auc": run["auc"],
            "norm_mean": run["norm_mean"],
            "norm_p50": run["norm_p50"],
            "norm_p95": run["norm_p95"],
            "ge_unit_frac": run["ge_unit_frac"],
        }
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


def plot(runs, path):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for name in ORDER:
        values = grouped[name]
        curves = np.asarray([run["success"] for run in values]) * 100
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
        x = values[0]["steps"] / 1000
        linestyle = "-" if name.startswith("env_") else "--"
        axis.plot(
            x,
            mean,
            marker="o",
            linewidth=2.3,
            linestyle=linestyle,
            color=COLORS[name],
            label=DISPLAY[name],
        )
        axis.fill_between(x, mean - sem, mean + sem, color=COLORS[name], alpha=0.14)
    axis.set(xlabel="Environment steps (k)", ylabel="Success rate (%)", ylim=(-2, 102))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("latent_root", type=Path)
    parser.add_argument("env_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    runs = load(args.latent_root, "latent") + load(args.env_root, "env")
    counts = {name: sum(run["condition"] == name for run in runs) for name in ORDER}
    if any(count != 4 for count in counts.values()):
        raise SystemExit(f"Expected four complete seeds per condition, got {counts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs)
    comparisons = paired(runs)
    write_csv(args.output_dir / "envq_counterfactual_500k_per_run.csv", per_run_rows(runs))
    write_csv(args.output_dir / "envq_counterfactual_500k_summary.csv", summary)
    write_csv(args.output_dir / "envq_counterfactual_500k_paired.csv", comparisons)
    plot(runs, args.output_dir / "envq_counterfactual_500k_curves.png")
    for row in summary:
        print(
            f"{row['display_name']:<24} "
            f"final={100 * row['final_success_mean']:.1f}+-{100 * row['final_success_std']:.1f} "
            f"tail3={100 * row['tail3_success_mean']:.1f} "
            f"AUC={100 * row['auc_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
