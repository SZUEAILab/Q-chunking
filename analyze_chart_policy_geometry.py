#!/usr/bin/env python3
"""Fixed-state conditional policy geometry for spherical vs stereographic."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_reference_policy_geometry import (
    checkpoint_path,
    create_agent,
    diagnose,
)
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent_with_file


ORDER = ("spherical", "stereographic")
COLORS = {"spherical": "#2ca02c", "stereographic": "#17becf"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-observations", type=int, default=1024)
    parser.add_argument("--samples-per-observation", type=int, default=32)
    args = parser.parse_args()

    complete = []
    for root in args.roots:
        for flag_path in sorted(root.rglob("flags.json")):
            flags = json.loads(flag_path.read_text())
            chart = flags.get("ds_integrated_type")
            if chart not in ORDER or flags.get("ds_regularizer") != "semantic_product":
                continue
            ckpt = checkpoint_path(flag_path.parent)
            if ckpt is not None:
                complete.append((flag_path, ckpt, chart))
    counts = {chart: sum(item[2] == chart for item in complete) for chart in ORDER}
    if counts != {"spherical": 4, "stereographic": 4}:
        raise SystemExit(f"Expected four checkpoints per chart, found {counts}")

    first_flags = json.loads(complete[0][0].read_text())
    env, eval_env, dataset, _ = make_env_and_datasets(first_flags["env_name"])
    try:
        all_observations = np.asarray(dataset["observations"])
        indices = np.linspace(
            0, len(all_observations) - 1, args.num_observations, dtype=np.int64
        )
        observations = all_observations[indices]
        obs_dim = observations.shape[-1]
        action_dim = dataset["actions"].shape[-1]

        rows = []
        for flag_path, ckpt, chart in complete:
            flags = json.loads(flag_path.read_text())
            agent = create_agent(flags, obs_dim, action_dim)
            agent = restore_agent_with_file(agent, str(ckpt))
            row = {
                "chart": chart,
                "seed": int(flags["seed"]),
                **diagnose(
                    agent,
                    observations,
                    args.samples_per_observation,
                    seed=20260925 + int(flags["seed"]),
                ),
            }
            rows.append(row)
            print(
                f"{chart} seed={row['seed']}: KL={row['semantic_kl']:.3f}, "
                f"radius={row['radius_mean']:.3f}, "
                f"angle={row['angular_deviation_rad']:.3f}"
            )
            del agent
    finally:
        env.close()
        eval_env.close()

    rows.sort(key=lambda row: (ORDER.index(row["chart"]), row["seed"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metrics = (
        "semantic_kl",
        "direction_kl",
        "speed_kl",
        "radius_mean",
        "angular_deviation_rad",
        "direction_resultant",
    )
    summary = []
    for chart in ORDER:
        item = {"chart": chart, "num_seeds": 4}
        chart_rows = [row for row in rows if row["chart"] == chart]
        for metric in metrics:
            values = np.asarray([row[metric] for row in chart_rows])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(item)
    with args.output.with_name(f"{args.output.stem}_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    rng = np.random.default_rng(20260925)
    for axis, metric, ylabel in (
        (axes[0], "direction_kl", "Direction KL to semantic reference"),
        (axes[1], "radius_mean", "Conditional mean radius"),
        (axes[2], "angular_deviation_rad", "Mean angular deviation (rad)"),
    ):
        for index, chart in enumerate(ORDER):
            values = np.asarray(
                [row[metric] for row in rows if row["chart"] == chart]
            )
            axis.scatter(
                index + rng.uniform(-0.05, 0.05, len(values)),
                values,
                color=COLORS[chart],
            )
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                marker="o",
                capsize=4,
                linewidth=2,
                color=COLORS[chart],
            )
        axis.set_xticks(range(len(ORDER)), ("Spherical", "Stereographic"))
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Direction charts on a shared bank of prior-data states")
    figure.tight_layout()
    figure.savefig(args.output.with_suffix(".png"), dpi=220)
    plt.close(figure)

    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
