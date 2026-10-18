#!/usr/bin/env python3
"""Contextual held-out rotation test for exploration covariance families."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from utils.artifact_paths import reports_dir

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax


RADIUS = 0.75
RADIAL_TOLERANCE = 0.025
ANGLE_TOLERANCE = np.deg2rad(7.5)


@dataclass(frozen=True)
class RotationCondition:
    name: str
    family: str
    parameter_count: int


CONDITIONS = (
    RotationCondition("Cartesian diagonal", "cartesian_diagonal", 2),
    RotationCondition("Cartesian full-cov constant", "cartesian_full", 3),
    RotationCondition("Cartesian full-cov Fourier", "cartesian_fourier", 9),
    RotationCondition("Cartesian rotating oracle", "cartesian_oracle", 2),
    RotationCondition("DS", "direction_speed", 2),
    RotationCondition("DS coupled", "direction_speed_coupled", 1),
)


def wrap_angle(value):
    return jnp.arctan2(jnp.sin(value), jnp.cos(value))


def angle_features(angles):
    return jnp.stack(
        (jnp.ones_like(angles), jnp.cos(2.0 * angles), jnp.sin(2.0 * angles)),
        axis=-1,
    )


def initial_parameters(condition, num_seeds):
    if condition.family == "cartesian_diagonal":
        values = np.log([0.27, 0.27])
        return {"values": jnp.asarray(np.broadcast_to(values, (num_seeds, 2)).copy())}
    if condition.family == "cartesian_full":
        values = np.asarray([np.log(0.27), np.log(0.27), 0.0])
        return {"values": jnp.asarray(np.broadcast_to(values, (num_seeds, 3)).copy())}
    if condition.family == "cartesian_fourier":
        weights = np.zeros((3, 3), dtype=np.float32)
        weights[0, :2] = np.log(0.27)
        return {
            "weights": jnp.asarray(
                np.broadcast_to(weights, (num_seeds, 3, 3)).copy()
            )
        }
    if condition.family == "cartesian_oracle":
        values = np.log([0.27, 0.03])
        return {"values": jnp.asarray(np.broadcast_to(values, (num_seeds, 2)).copy())}
    if condition.family == "direction_speed":
        values = np.log([0.25 / RADIUS, 0.008192])
        return {"values": jnp.asarray(np.broadcast_to(values, (num_seeds, 2)).copy())}
    values = np.log([0.08])
    return {"values": jnp.asarray(np.broadcast_to(values, (num_seeds, 1)).copy())}


def project_parameters(params, condition):
    params = dict(params)
    if "values" in params:
        values = params["values"]
        if condition.family == "cartesian_full":
            values = values.at[..., :2].set(jnp.clip(values[..., :2], -6.0, 0.0))
            values = values.at[..., 2].set(jnp.clip(values[..., 2], -1.0, 1.0))
        else:
            values = jnp.clip(values, -6.0, 0.0)
        params["values"] = values
    else:
        weights = params["weights"]
        weights = weights.at[..., 0, :2].set(jnp.clip(weights[..., 0, :2], -6.0, 0.0))
        weights = jnp.clip(weights, -6.0, 6.0)
        params["weights"] = weights
    return params


def shell_hit(actions, target_angles):
    radii = jnp.linalg.norm(actions, axis=-1)
    action_angles = jnp.arctan2(actions[..., 1], actions[..., 0])
    return (
        (jnp.abs(radii - RADIUS) < RADIAL_TOLERANCE)
        & (jnp.abs(wrap_angle(action_angles - target_angles[:, None])) < ANGLE_TOLERANCE)
    ).astype(jnp.float32)


def sample_single(params, key, condition, context_angles, target_offset, batch_size):
    noise = jax.random.normal(key, (len(context_angles), batch_size, 2))
    means = RADIUS * jnp.stack((jnp.cos(context_angles), jnp.sin(context_angles)), axis=-1)
    means = means[:, None, :]

    if condition.family == "cartesian_diagonal":
        log_scales = jnp.clip(params["values"], -6.0, 0.0)
        scales = jnp.exp(log_scales)
        actions = means + scales[None, None, :] * noise
        fixed = jax.lax.stop_gradient(actions)
        standardized = (fixed - means) / scales[None, None, :]
        log_probs = -0.5 * jnp.sum(standardized**2, axis=-1) - jnp.sum(log_scales) - np.log(2 * np.pi)
        return actions, log_probs

    if condition.family in ("cartesian_full", "cartesian_fourier"):
        if condition.family == "cartesian_full":
            outputs = jnp.broadcast_to(params["values"], (len(context_angles), 3))
        else:
            outputs = angle_features(context_angles) @ params["weights"]
        log_scales = jnp.clip(outputs[..., :2], -6.0, 0.0)
        scales = jnp.exp(log_scales)
        off = jnp.clip(outputs[..., 2], -1.0, 1.0)
        first = scales[:, None, 0] * noise[..., 0]
        second = off[:, None] * noise[..., 0] + scales[:, None, 1] * noise[..., 1]
        actions = means + jnp.stack((first, second), axis=-1)
        fixed = jax.lax.stop_gradient(actions) - means
        z0 = fixed[..., 0] / scales[:, None, 0]
        z1 = (fixed[..., 1] - off[:, None] * z0) / scales[:, None, 1]
        log_probs = -0.5 * (z0**2 + z1**2) - jnp.sum(log_scales, axis=-1)[:, None] - np.log(2 * np.pi)
        return actions, log_probs

    if condition.family == "cartesian_oracle":
        log_scales = jnp.clip(params["values"], -6.0, 0.0)
        scales = jnp.exp(log_scales)
        target_angles = context_angles + target_offset
        targets = RADIUS * jnp.stack((jnp.cos(target_angles), jnp.sin(target_angles)), axis=-1)
        chord = targets - means[:, 0]
        along = chord / jnp.linalg.norm(chord, axis=-1, keepdims=True)
        across = jnp.stack((-along[:, 1], along[:, 0]), axis=-1)
        residual = scales[0] * noise[..., 0, None] * along[:, None, :] + scales[1] * noise[..., 1, None] * across[:, None, :]
        actions = means + residual
        fixed = jax.lax.stop_gradient(actions) - means
        z0 = jnp.sum(fixed * along[:, None, :], axis=-1) / scales[0]
        z1 = jnp.sum(fixed * across[:, None, :], axis=-1) / scales[1]
        log_probs = -0.5 * (z0**2 + z1**2) - jnp.sum(log_scales) - np.log(2 * np.pi)
        return actions, log_probs

    if condition.family == "direction_speed":
        log_scales = jnp.clip(params["values"], -6.0, 0.0)
        scales = jnp.exp(log_scales)
    else:
        local_scale = jnp.exp(jnp.clip(params["values"][0], -6.0, 0.0))
        scales = jnp.stack((local_scale / RADIUS, local_scale))
        log_scales = jnp.log(scales)
    latent_angles = context_angles[:, None] + scales[0] * noise[..., 0]
    latent_radii = RADIUS + scales[1] * noise[..., 1]
    actions = jnp.stack(
        (latent_radii * jnp.cos(latent_angles), latent_radii * jnp.sin(latent_angles)),
        axis=-1,
    )
    fixed_angle = jax.lax.stop_gradient(latent_angles)
    fixed_radius = jax.lax.stop_gradient(latent_radii)
    z0 = wrap_angle(fixed_angle - context_angles[:, None]) / scales[0]
    z1 = (fixed_radius - RADIUS) / scales[1]
    log_probs = -0.5 * (z0**2 + z1**2) - jnp.sum(log_scales) - np.log(2 * np.pi)
    return actions, log_probs


def make_train_step(condition, angles, target_offset, batch_size, learning_rate):
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(learning_rate))

    def loss_fn(params, keys):
        def one_seed(seed_params, seed_key):
            actions, log_probs = sample_single(seed_params, seed_key, condition, angles, target_offset, batch_size)
            rewards = shell_hit(actions, angles + target_offset)
            advantages = jax.lax.stop_gradient(rewards - rewards.mean(axis=1, keepdims=True))
            return -jnp.mean(advantages * log_probs), rewards.mean()

        losses, success = jax.vmap(one_seed)(params, keys)
        return losses.mean(), success

    @jax.jit
    def step(params, state, keys):
        (loss, success), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params, keys)
        updates, state = optimizer.update(gradients, state, params)
        params = project_parameters(optax.apply_updates(params, updates), condition)
        return params, state, loss, success

    return optimizer, step


def make_evaluator(condition, angles, target_offset, batch_size):
    @jax.jit
    def evaluate(params, keys):
        def one_seed(seed_params, seed_key):
            actions, _ = sample_single(seed_params, seed_key, condition, angles, target_offset, batch_size)
            return shell_hit(actions, angles + target_offset).mean(axis=1)

        return jax.vmap(one_seed)(params, keys)

    return evaluate


def train_condition(condition, seeds, train_angles, test_angles, args):
    params = initial_parameters(condition, len(seeds))
    optimizer, train_step = make_train_step(condition, train_angles, args.target_offset, args.batch_size, args.learning_rate)
    state = optimizer.init(params)
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    rows = []
    for update in range(1, args.updates + 1):
        split = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
        keys, step_keys = split[:, 0], split[:, 1]
        params, state, _, _ = train_step(params, state, step_keys)

    all_angles = jnp.concatenate((train_angles, test_angles))
    evaluator = make_evaluator(condition, all_angles, args.target_offset, args.evaluation_size)
    eval_keys = jnp.stack([jax.random.PRNGKey(10_000_000 + seed) for seed in seeds])
    success = np.asarray(jax.device_get(evaluator(params, eval_keys)))
    split_labels = ["train"] * len(train_angles) + ["test"] * len(test_angles)
    for seed_index, seed in enumerate(seeds):
        for angle_index, angle in enumerate(np.asarray(all_angles)):
            rows.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "split": split_labels[angle_index],
                    "context_angle_deg": float(np.rad2deg(angle) % 360.0),
                    "target_angle_deg": float(np.rad2deg(angle + args.target_offset) % 360.0),
                    "success": float(success[seed_index, angle_index]),
                    "parameter_count": condition.parameter_count,
                }
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, seeds):
    output = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition.name]
        item = {"condition": condition.name, "num_seeds": len(seeds), "parameter_count": condition.parameter_count}
        for split in ("train", "test"):
            per_seed = []
            for seed in seeds:
                values = [row["success"] for row in selected if row["seed"] == seed and row["split"] == split]
                per_seed.append(float(np.mean(values)))
            item[f"{split}_success_mean"] = float(np.mean(per_seed))
            item[f"{split}_success_std"] = float(np.std(per_seed, ddof=1))
        item["generalization_gap"] = item["train_success_mean"] - item["test_success_mean"]
        output.append(item)
    return output


def plot(rows, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
    colors = plt.get_cmap("tab10")
    for index, condition in enumerate(CONDITIONS):
        selected = [row for row in rows if row["condition"] == condition.name]
        test_angles = sorted({row["context_angle_deg"] for row in selected if row["split"] == "test"})
        means = [np.mean([row["success"] for row in selected if row["split"] == "test" and row["context_angle_deg"] == angle]) for angle in test_angles]
        axes[0].plot(test_angles, np.asarray(means) * 100.0, marker="o", label=condition.name, color=colors(index))
    axes[0].set_xlabel("Held-out context angle (deg)")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    x = np.arange(len(CONDITIONS))
    train = []
    test = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition.name]
        train.append(100 * np.mean([row["success"] for row in selected if row["split"] == "train"]))
        test.append(100 * np.mean([row["success"] for row in selected if row["split"] == "test"]))
    axes[1].bar(x - 0.18, train, 0.36, label="Train angles")
    axes[1].bar(x + 0.18, test, 0.36, label="Held-out angles")
    axes[1].set_xticks(x, [condition.name for condition in CONDITIONS], rotation=25, ha="right")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=500)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--evaluation-size", type=int, default=20000)
    parser.add_argument("--target-offset-deg", type=float, default=30.0)
    parser.add_argument("--output-prefix", default="contextual_rotation_pilot")
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    args.target_offset = np.deg2rad(args.target_offset_deg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    train_angles = jnp.deg2rad(jnp.arange(0.0, 360.0, 30.0))
    test_angles = jnp.deg2rad(jnp.arange(15.0, 360.0, 30.0))
    rows = []
    for condition in CONDITIONS:
        rows.extend(train_condition(condition, seeds, train_angles, test_angles, args))
        print(f"finished {condition.name}", flush=True)
    summaries = summarize(rows, seeds)
    write_csv(args.output_dir / f"{args.output_prefix}_per_angle.csv", rows)
    write_csv(args.output_dir / f"{args.output_prefix}_summary.csv", summaries)
    plot(rows, args.output_dir / f"{args.output_prefix}.png")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
