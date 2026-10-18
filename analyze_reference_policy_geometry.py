#!/usr/bin/env python3
"""Post-hoc exploration diagnostics for matched-reference checkpoints.

Every policy is evaluated on the same deterministic bank of prior-data states.
This avoids confusing a changed state visitation distribution with a changed
conditional action distribution.  The script is intentionally separate from
the training loop so adding diagnostics cannot change any formal run.
"""

import argparse
import copy
import csv
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from agents import agents
from direction_speed import configure_direction_speed
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent_with_file


def condition(flags):
    reference = flags["ds_regularizer"]
    if reference == "action_volume":
        return "V1"
    if reference == "semantic_product":
        return "S3" if flags["ds_separate_alpha"] else "S1"
    raise ValueError(f"Unexpected reference: {reference}")


def checkpoint_path(run_dir):
    candidates = sorted(
        run_dir.glob("params_*.pkl"),
        key=lambda path: int(re.search(r"params_(\d+)\.pkl", path.name).group(1)),
    )
    if not candidates:
        return None
    return candidates[-1]


def create_agent(flags, obs_dim, action_dim):
    config = copy.deepcopy(flags["agent"])
    config["horizon_length"] = flags["horizon_length"]
    configure_direction_speed(
        ds_mode=flags["ds_mode"],
        ds_speed_bound=flags["ds_speed_bound"],
        ds_integrated_type=flags["ds_integrated_type"],
        ds_adapter_representation=flags["ds_adapter_representation"],
        ds_adapter_speed_squash=flags["ds_adapter_speed_squash"],
        ds_adapter_speed_interpretation=flags["ds_adapter_speed_interpretation"],
        ds_adapter_critic_space=flags["ds_adapter_critic_space"],
        ds_regularizer=flags["ds_regularizer"],
        ds_separate_alpha=flags["ds_separate_alpha"],
        config=config,
        agent_name=config["agent_name"],
        ds_speed_bound_default="cube",
    )
    config["env_action_dim"] = action_dim
    config["policy_action_dim"] = action_dim
    config["critic_action_dim"] = action_dim
    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_action = np.zeros((1, action_dim), dtype=np.float32)
    return agents[config["agent_name"]].create(
        flags["seed"], dummy_obs, dummy_action, config
    )


def scalar(value):
    return float(np.asarray(jax.device_get(value)))


def diagnose(agent, observations, num_samples, seed):
    dist = agent.network.select("actor")(jnp.asarray(observations))
    actions = dist.sample(num_samples, seed=jax.random.PRNGKey(seed))
    semantic_components = dist.semantic_component_log_ratios(actions)
    semantic_log_ratio = dist.reference_log_prob(actions, "semantic_product")
    volume_log_ratio = dist.reference_log_prob(actions, "action_volume")

    xyz = actions[..., :3]
    radii = jnp.linalg.norm(xyz, axis=-1)
    directions = xyz / jnp.clip(radii[..., None], 1e-8)
    conditional_mean_direction = directions.mean(axis=0)
    resultant = jnp.linalg.norm(conditional_mean_direction, axis=-1)
    center = conditional_mean_direction / jnp.clip(resultant[..., None], 1e-8)
    angular_deviation = jnp.arccos(
        jnp.clip(jnp.sum(directions * center[None, ...], axis=-1), -1.0, 1.0)
    )

    result = {
        "semantic_kl": scalar(semantic_log_ratio.mean()),
        "action_volume_kl": scalar(volume_log_ratio.mean()),
        "direction_kl": scalar(semantic_components["direction"].mean()),
        "speed_kl": scalar(semantic_components["speed"].mean()),
        "scalar_kl": scalar(semantic_components["scalar"].mean()),
        "radius_mean": scalar(radii.mean()),
        "radius_std_within_state": scalar(radii.std(axis=0).mean()),
        "radius_p10": scalar(jnp.quantile(radii, 0.10)),
        "radius_p50": scalar(jnp.quantile(radii, 0.50)),
        "radius_p90": scalar(jnp.quantile(radii, 0.90)),
        "radius_below_025": scalar((radii < 0.25).mean()),
        "radius_above_090": scalar((radii > 0.90).mean()),
        "direction_resultant": scalar(resultant.mean()),
        "angular_deviation_rad": scalar(angular_deviation.mean()),
    }
    if agent.config.get("ds_separate_alpha", False):
        for name in ("direction", "speed", "scalar"):
            result[f"alpha_{name}"] = scalar(
                agent.network.select(f"alpha_{name}")()
            )
        result["alpha"] = np.nan
    else:
        result["alpha"] = scalar(agent.network.select("alpha")())
        for name in ("direction", "speed", "scalar"):
            result[f"alpha_{name}"] = np.nan
    return result


def summarize_and_plot(rows, output):
    order = ("V1", "S1", "S3")
    colors = {"V1": "#9467bd", "S1": "#2ca02c", "S3": "#ff7f0e"}
    summary = []
    for condition_name in order:
        condition_rows = [row for row in rows if row["condition"] == condition_name]
        item = {"condition": condition_name, "num_seeds": len(condition_rows)}
        for metric in (
            "semantic_kl",
            "action_volume_kl",
            "direction_kl",
            "speed_kl",
            "scalar_kl",
            "radius_mean",
            "radius_std_within_state",
            "angular_deviation_rad",
            "direction_resultant",
        ):
            values = np.asarray([row[metric] for row in condition_rows])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(item)

    summary_path = output.with_name(f"{output.stem}_summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    component_metrics = ("direction_kl", "speed_kl", "scalar_kl")
    width = 0.24
    x = np.arange(len(component_metrics))
    for index, condition_name in enumerate(order):
        condition_rows = [row for row in rows if row["condition"] == condition_name]
        values = np.asarray(
            [[row[metric] for metric in component_metrics] for row in condition_rows]
        )
        axes[0].bar(
            x + (index - 1) * width,
            values.mean(axis=0),
            width,
            yerr=values.std(axis=0, ddof=1),
            capsize=3,
            color=colors[condition_name],
            label=condition_name,
        )
    axes[0].set_xticks(x, ("Direction", "Speed", "Scalars"))
    axes[0].set_ylabel("KL to semantic reference")
    axes[0].legend()
    rng = np.random.default_rng(20260925)
    for axis, metric, ylabel in (
        (axes[1], "radius_mean", "Conditional mean radius"),
        (axes[2], "angular_deviation_rad", "Mean angular deviation (rad)"),
    ):
        for index, condition_name in enumerate(order):
            values = np.asarray(
                [row[metric] for row in rows if row["condition"] == condition_name]
            )
            axis.scatter(
                np.full(len(values), index) + rng.uniform(-0.05, 0.05, len(values)),
                values,
                color=colors[condition_name],
                alpha=0.75,
            )
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                marker="o",
                capsize=4,
                linewidth=2,
                color=colors[condition_name],
            )
        axis.set_xticks(range(len(order)), order)
        axis.set_ylabel(ylabel)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Final policies on a shared bank of prior-data states")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-observations", type=int, default=1024)
    parser.add_argument("--samples-per-observation", type=int, default=32)
    args = parser.parse_args()

    flag_paths = sorted(args.root.rglob("flags.json"))
    complete = []
    for flag_path in flag_paths:
        ckpt = checkpoint_path(flag_path.parent)
        if ckpt is not None:
            complete.append((flag_path, ckpt))
    if not complete:
        raise SystemExit(f"No checkpoints under {args.root}")

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
        for flag_path, ckpt in complete:
            flags = json.loads(flag_path.read_text())
            agent = create_agent(flags, obs_dim, action_dim)
            agent = restore_agent_with_file(agent, str(ckpt))
            row = {
                "condition": condition(flags),
                "seed": flags["seed"],
                "checkpoint_step": int(
                    re.search(r"params_(\d+)\.pkl", ckpt.name).group(1)
                ),
                **diagnose(
                    agent,
                    observations,
                    args.samples_per_observation,
                    seed=20260925 + int(flags["seed"]),
                ),
            }
            rows.append(row)
            print(
                f"{row['condition']} seed={row['seed']}: "
                f"semantic_KL={row['semantic_kl']:.3f}, "
                f"radius={row['radius_mean']:.3f}, "
                f"angle={row['angular_deviation_rad']:.3f} rad"
            )
            del agent

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summarize_and_plot(rows, args.output)
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
