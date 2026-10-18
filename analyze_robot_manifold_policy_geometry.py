#!/usr/bin/env python3
"""Diagnose conditional executed-action geometry for the robot pilot."""

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


DEFAULT_RUN_GROUP = "robot_manifold_support_pilot_100k_20260925"
TASKS = ("task1", "task2")
CONDITIONS = (
    "cart_projection",
    "exp_tangent",
    "projected_tangent",
    "exp_variance_matched",
)


def identify(flags):
    task = "task1" if "task1" in flags["env_name"] else "task2"
    if flags["ds_mode"] == "projection":
        condition = "cart_projection"
    elif flags["ds_integrated_type"] == "projected_tangent":
        condition = "projected_tangent"
    elif flags["agent"].get("ds_tangent_speed_mean_offset", 0.0) != 0.0:
        condition = "exp_variance_matched"
    else:
        condition = "exp_tangent"
    return task, condition


def latest_checkpoint(run_dir):
    paths = sorted(
        run_dir.glob("params_*.pkl"),
        key=lambda path: int(re.search(r"params_(\d+)\.pkl", path.name).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"No checkpoint under {run_dir}")
    return paths[-1]


def create_agent(flags, obs_dim, env_action_dim):
    config = copy.deepcopy(flags["agent"])
    config["horizon_length"] = flags["horizon_length"]
    ds_config = configure_direction_speed(
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
    policy_action_dim = (
        ds_config.adapter.policy_action_dim(env_action_dim)
        if ds_config.use_adapter
        else env_action_dim
    )
    critic_action_dim = (
        env_action_dim if ds_config.use_env_action_critic else policy_action_dim
    )
    config["env_action_dim"] = env_action_dim
    config["policy_action_dim"] = policy_action_dim
    config["critic_action_dim"] = critic_action_dim
    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_action = np.zeros((1, critic_action_dim), dtype=np.float32)
    agent = agents[config["agent_name"]].create(
        flags["seed"], dummy_obs, dummy_action, config
    )
    return agent, ds_config.postprocessor, policy_action_dim


def scalar(value):
    return float(np.asarray(jax.device_get(value)))


def diagnose(agent, postprocessor, policy_action_dim, observations, samples, seed):
    dist = agent.network.select("actor")(jnp.asarray(observations))
    policy_actions = dist.sample(samples, seed=jax.random.PRNGKey(seed))
    flat = policy_actions.reshape((-1, policy_action_dim))
    env_actions = postprocessor.compose_jax(flat).reshape(
        (samples, len(observations), -1)
    )
    xyz = env_actions[..., :3]
    radii = jnp.linalg.norm(xyz, axis=-1)
    directions = xyz / jnp.clip(radii[..., None], 1e-8)
    mean_direction = directions.mean(axis=0)
    resultant = jnp.linalg.norm(mean_direction, axis=-1)
    center = mean_direction / jnp.clip(resultant[..., None], 1e-8)
    angles = jnp.arccos(
        jnp.clip(jnp.sum(directions * center[None, ...], axis=-1), -1.0, 1.0)
    )
    mean_xyz = xyz.mean(axis=0)
    tangential_std = jnp.sqrt(jnp.sum(jnp.var(xyz, axis=0), axis=-1))
    return {
        "radius_mean": scalar(radii.mean()),
        "radius_std_within_state": scalar(radii.std(axis=0).mean()),
        "radius_p10": scalar(jnp.quantile(radii, 0.10)),
        "radius_p50": scalar(jnp.quantile(radii, 0.50)),
        "radius_p90": scalar(jnp.quantile(radii, 0.90)),
        "radius_at_bound": scalar((radii >= 1.0 - 1e-6).mean()),
        "mean_xyz_radius": scalar(jnp.linalg.norm(mean_xyz, axis=-1).mean()),
        "xyz_total_std": scalar(tangential_std.mean()),
        "direction_resultant": scalar(resultant.mean()),
        "angular_deviation_rad": scalar(angles.mean()),
    }


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize_and_plot(rows, output, tasks, conditions):
    metrics = (
        "radius_mean",
        "radius_std_within_state",
        "radius_at_bound",
        "mean_xyz_radius",
        "xyz_total_std",
        "direction_resultant",
        "angular_deviation_rad",
    )
    summary = []
    for task in tasks:
        for condition in conditions:
            selected = [
                row
                for row in rows
                if row["task"] == task and row["condition"] == condition
            ]
            item = {"task": task, "condition": condition, "num_seeds": len(selected)}
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected])
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std(ddof=1))
            summary.append(item)
    write_rows(output.with_name(f"{output.stem}_summary.csv"), summary)

    figure, axes = plt.subplots(
        len(tasks), 3, figsize=(13.5, 3.6 * len(tasks)), squeeze=False
    )
    plotted = (
        ("radius_mean", "Mean XYZ radius"),
        ("radius_std_within_state", "Within-state radius std"),
        ("angular_deviation_rad", "Angular deviation (rad)"),
    )
    rng = np.random.default_rng(20260925)
    for row_index, task in enumerate(tasks):
        for axis, (metric, label) in zip(axes[row_index], plotted, strict=True):
            for condition_index, condition in enumerate(conditions):
                values = np.asarray([
                    row[metric]
                    for row in rows
                    if row["task"] == task and row["condition"] == condition
                ])
                axis.scatter(
                    condition_index + rng.uniform(-0.04, 0.04, len(values)),
                    values,
                    color=f"C{condition_index}",
                    alpha=0.8,
                )
                axis.plot(condition_index, values.mean(), marker="D", color="black")
            axis.set_xticks(
                range(len(conditions)), conditions, rotation=25, ha="right"
            )
            axis.set_ylabel(f"{task}: {label}")
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Conditional executed-action geometry on shared prior-data states")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-group", default=DEFAULT_RUN_GROUP)
    parser.add_argument("--expected-runs", type=int, default=12)
    parser.add_argument("--num-observations", type=int, default=512)
    parser.add_argument("--samples-per-observation", type=int, default=32)
    parser.add_argument("--override-angular-offset", type=float)
    parser.add_argument("--override-angular-max", type=float)
    parser.add_argument("--override-speed-offset", type=float)
    parser.add_argument("--override-speed-max", type=float)
    parser.add_argument("--override-speed-mean-offset", type=float)
    args = parser.parse_args()

    runs = []
    for flags_path in sorted(args.root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("run_group") != args.run_group:
            continue
        task, condition = identify(flags)
        runs.append((task, condition, int(flags["seed"]), flags_path, flags))
    if len(runs) != args.expected_runs:
        raise SystemExit(f"Expected {args.expected_runs} runs, found {len(runs)}")
    tasks = tuple(task for task in TASKS if any(run[0] == task for run in runs))
    conditions = tuple(
        condition for condition in CONDITIONS if any(run[1] == condition for run in runs)
    )
    if not tasks:
        raise SystemExit(f"No supported tasks found for run group {args.run_group}")

    rows = []
    for task in tasks:
        task_runs = [run for run in runs if run[0] == task]
        env, eval_env, dataset, _ = make_env_and_datasets(task_runs[0][4]["env_name"])
        try:
            all_observations = np.asarray(dataset["observations"])
            indices = np.linspace(
                0, len(all_observations) - 1, args.num_observations, dtype=np.int64
            )
            observations = all_observations[indices]
            obs_dim = observations.shape[-1]
            env_action_dim = dataset["actions"].shape[-1]
            for _, condition, seed, flags_path, flags in task_runs:
                if condition != "cart_projection":
                    overrides = {
                        "ds_tangent_angular_log_std_offset": args.override_angular_offset,
                        "ds_tangent_angular_log_std_max": args.override_angular_max,
                        "ds_tangent_speed_log_std_offset": args.override_speed_offset,
                        "ds_tangent_speed_log_std_max": args.override_speed_max,
                        "ds_tangent_speed_mean_offset": args.override_speed_mean_offset,
                    }
                    for key, value in overrides.items():
                        if value is not None:
                            flags["agent"][key] = value
                agent, postprocessor, policy_action_dim = create_agent(
                    flags, obs_dim, env_action_dim
                )
                checkpoint = latest_checkpoint(flags_path.parent)
                agent = restore_agent_with_file(agent, str(checkpoint))
                row = {
                    "task": task,
                    "condition": condition,
                    "seed": seed,
                    "checkpoint_step": int(
                        re.search(r"params_(\d+)\.pkl", checkpoint.name).group(1)
                    ),
                    **diagnose(
                        agent,
                        postprocessor,
                        policy_action_dim,
                        observations,
                        args.samples_per_observation,
                        seed=20260925 + seed,
                    ),
                }
                rows.append(row)
                print(
                    f"{task} {condition} seed={seed}: "
                    f"radius={row['radius_mean']:.3f}, "
                    f"radial_std={row['radius_std_within_state']:.3f}, "
                    f"angle={row['angular_deviation_rad']:.3f}"
                )
                del agent
                jax.clear_caches()
        finally:
            env.close()
            eval_env.close()

    rows.sort(key=lambda row: (TASKS.index(row["task"]), CONDITIONS.index(row["condition"]), row["seed"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, rows)
    summarize_and_plot(rows, args.output, tasks, conditions)


if __name__ == "__main__":
    main()
