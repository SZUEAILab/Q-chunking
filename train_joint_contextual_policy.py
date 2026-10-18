#!/usr/bin/env python3
"""Joint state-to-mean and covariance learning on held-out rotations."""

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
FEATURE_DIM = 32
HIDDEN_DIM = 64
MAX_RESIDUAL = 0.6
MIN_LOG_SCALE = -4.0


@dataclass(frozen=True)
class JointCondition:
    name: str
    family: str
    covariance: str
    covariance_outputs: int

    @property
    def parameter_count(self):
        mean = FEATURE_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * 2 + 2
        if self.covariance == "state":
            covariance = (
                FEATURE_DIM * HIDDEN_DIM
                + HIDDEN_DIM
                + HIDDEN_DIM * self.covariance_outputs
                + self.covariance_outputs
            )
        else:
            covariance = self.covariance_outputs
        return mean + covariance


CONDITIONS = (
    JointCondition("Cartesian diagonal state", "cartesian_diagonal", "state", 2),
    JointCondition("Cartesian full-cov state", "cartesian_full", "state", 3),
    JointCondition("Cartesian rotating oracle", "cartesian_oracle", "global", 2),
    JointCondition("DS global", "direction_speed", "global", 2),
    JointCondition("DS state", "direction_speed", "state", 2),
    JointCondition("DS coupled", "direction_speed_coupled", "global", 1),
)

# Optional counterfactual kept outside CONDITIONS so the frozen six-condition
# confirmation set and its analyzer remain immutable.
MEAN_FRAME_CONDITION = JointCondition(
    "Cartesian mean-frame", "cartesian_mean_frame", "global", 2
)
PROJECTED_TANGENT_CONDITION = JointCondition(
    "Cartesian projected-tangent flow", "cartesian_projected_tangent", "global", 2
)


def wrap_angle(value):
    return jnp.arctan2(jnp.sin(value), jnp.cos(value))


def context_features(angles):
    harmonics = jnp.arange(1, FEATURE_DIM // 2 + 1, dtype=angles.dtype)
    phases = angles[:, None] * harmonics[None, :]
    return jnp.concatenate((jnp.cos(phases), jnp.sin(phases)), axis=-1)


def anchors(angles):
    return RADIUS * jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)


def targets(angles, target_offset):
    return anchors(angles + target_offset)


def init_mlp(rng, output_dim, output_bias, num_seeds):
    keys = jax.random.split(rng, num_seeds)

    def one(key):
        first, second = jax.random.split(key)
        return {
            "w1": jax.random.normal(first, (FEATURE_DIM, HIDDEN_DIM))
            / np.sqrt(FEATURE_DIM),
            "b1": jnp.zeros(HIDDEN_DIM),
            "w2": jnp.zeros((HIDDEN_DIM, output_dim)),
            "b2": jnp.asarray(output_bias),
        }

    return jax.vmap(one)(keys)


def initial_parameters(condition, seeds):
    key = jax.random.PRNGKey(91_000 + sum(seeds))
    mean_key, covariance_key = jax.random.split(key)
    params = {
        "mean": init_mlp(mean_key, 2, np.zeros(2, dtype=np.float32), len(seeds))
    }
    if condition.covariance == "state":
        if condition.family == "cartesian_full":
            bias = np.asarray([np.log(0.27), np.log(0.27), 0.0], np.float32)
        elif condition.family == "cartesian_diagonal":
            bias = np.asarray([np.log(0.27), np.log(0.27)], np.float32)
        else:
            bias = np.asarray([np.log(0.25 / RADIUS), np.log(0.008192)], np.float32)
        params["covariance"] = init_mlp(
            covariance_key, condition.covariance_outputs, bias, len(seeds)
        )
    elif condition.family == "cartesian_oracle":
        values = np.log([0.27, 0.03])
        params["covariance"] = jnp.asarray(
            np.broadcast_to(values, (len(seeds), 2)).copy()
        )
    elif condition.family == "cartesian_mean_frame":
        # Tangential and radial Cartesian standard deviations.  These match
        # the first-order local covariance of the DS initialization.  The raw
        # radial value is below MIN_LOG_SCALE, so both execute at exp(-4).
        values = np.log([0.25, 0.008192])
        params["covariance"] = jnp.asarray(
            np.broadcast_to(values, (len(seeds), 2)).copy()
        )
    elif condition.family == "cartesian_projected_tangent":
        # Cartesian tangent displacement and radial displacement.  The former
        # is normalized back onto the mean-radius circle before composition;
        # its radial scale shares the same exp(-4) effective floor as DS.
        values = np.log([0.25, 0.008192])
        params["covariance"] = jnp.asarray(
            np.broadcast_to(values, (len(seeds), 2)).copy()
        )
    elif condition.family == "direction_speed":
        values = np.log([0.25 / RADIUS, 0.008192])
        params["covariance"] = jnp.asarray(
            np.broadcast_to(values, (len(seeds), 2)).copy()
        )
    else:
        params["covariance"] = jnp.full((len(seeds), 1), np.log(0.08))
    return params


def mlp_apply(params, features):
    hidden = jnp.tanh(features @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def policy_means(params, features, angles):
    residual = MAX_RESIDUAL * jnp.tanh(mlp_apply(params["mean"], features))
    return anchors(angles) + residual


def covariance_outputs(params, condition, features, num_contexts):
    if condition.covariance == "state":
        return mlp_apply(params["covariance"], features)
    return jnp.broadcast_to(params["covariance"], (num_contexts, condition.covariance_outputs))


def shell_hit(actions, target_angles):
    radii = jnp.linalg.norm(actions, axis=-1)
    action_angles = jnp.arctan2(actions[..., 1], actions[..., 0])
    return (
        (jnp.abs(radii - RADIUS) < RADIAL_TOLERANCE)
        & (
            jnp.abs(wrap_angle(action_angles - target_angles[:, None]))
            < ANGLE_TOLERANCE
        )
    ).astype(jnp.float32)


def sample_single(
    params, key, condition, features, angles, target_offset, batch_size
):
    noise = jax.random.normal(key, (len(angles), batch_size, 2))
    means = policy_means(params, features, angles)
    outputs = covariance_outputs(params, condition, features, len(angles))

    if condition.family == "cartesian_diagonal":
        log_scales = jnp.clip(outputs[..., :2], MIN_LOG_SCALE, 0.0)
        scales = jnp.exp(log_scales)
        actions = means[:, None, :] + scales[:, None, :] * noise
        fixed = jax.lax.stop_gradient(actions)
        standardized = (fixed - means[:, None, :]) / scales[:, None, :]
        log_probs = (
            -0.5 * jnp.sum(standardized**2, axis=-1)
            - jnp.sum(log_scales, axis=-1)[:, None]
            - np.log(2 * np.pi)
        )
        return actions, log_probs, means

    if condition.family == "cartesian_full":
        log_scales = jnp.clip(outputs[..., :2], MIN_LOG_SCALE, 0.0)
        scales = jnp.exp(log_scales)
        off = jnp.clip(outputs[..., 2], -0.75, 0.75)
        first = scales[:, None, 0] * noise[..., 0]
        second = off[:, None] * noise[..., 0] + scales[:, None, 1] * noise[..., 1]
        actions = means[:, None, :] + jnp.stack((first, second), axis=-1)
        fixed = jax.lax.stop_gradient(actions) - means[:, None, :]
        z0 = fixed[..., 0] / scales[:, None, 0]
        z1 = (fixed[..., 1] - off[:, None] * z0) / scales[:, None, 1]
        log_probs = (
            -0.5 * (z0**2 + z1**2)
            - jnp.sum(log_scales, axis=-1)[:, None]
            - np.log(2 * np.pi)
        )
        return actions, log_probs, means

    if condition.family == "cartesian_oracle":
        log_scales = jnp.clip(outputs, MIN_LOG_SCALE, 0.0)
        scales = jnp.exp(log_scales)
        target_points = targets(angles, target_offset)
        chord = target_points - means
        chord_norm = jnp.linalg.norm(chord, axis=-1, keepdims=True)
        fallback = jnp.stack(
            (jnp.cos(angles + target_offset), jnp.sin(angles + target_offset)),
            axis=-1,
        )
        along = jnp.where(chord_norm > 1e-5, chord / jnp.maximum(chord_norm, 1e-5), fallback)
        across = jnp.stack((-along[:, 1], along[:, 0]), axis=-1)
        residual = (
            scales[:, None, 0, None] * noise[..., 0, None] * along[:, None, :]
            + scales[:, None, 1, None] * noise[..., 1, None] * across[:, None, :]
        )
        actions = means[:, None, :] + residual
        fixed = jax.lax.stop_gradient(actions) - means[:, None, :]
        z0 = jnp.sum(fixed * along[:, None, :], axis=-1) / scales[:, None, 0]
        z1 = jnp.sum(fixed * across[:, None, :], axis=-1) / scales[:, None, 1]
        log_probs = (
            -0.5 * (z0**2 + z1**2)
            - jnp.sum(log_scales, axis=-1)[:, None]
            - np.log(2 * np.pi)
        )
        return actions, log_probs, means

    if condition.family == "cartesian_mean_frame":
        log_scales = jnp.clip(outputs, MIN_LOG_SCALE, 0.0)
        scales = jnp.exp(log_scales)
        mean_radii = jnp.maximum(jnp.linalg.norm(means, axis=-1, keepdims=True), 1e-5)
        radial = means / mean_radii
        tangent = jnp.stack((-radial[:, 1], radial[:, 0]), axis=-1)
        residual = (
            scales[:, None, 0, None] * noise[..., 0, None] * tangent[:, None, :]
            + scales[:, None, 1, None] * noise[..., 1, None] * radial[:, None, :]
        )
        actions = means[:, None, :] + residual
        fixed = jax.lax.stop_gradient(actions) - means[:, None, :]
        z0 = jnp.sum(fixed * tangent[:, None, :], axis=-1) / scales[:, None, 0]
        z1 = jnp.sum(fixed * radial[:, None, :], axis=-1) / scales[:, None, 1]
        log_probs = (
            -0.5 * (z0**2 + z1**2)
            - jnp.sum(log_scales, axis=-1)[:, None]
            - np.log(2 * np.pi)
        )
        return actions, log_probs, means

    if condition.family == "cartesian_projected_tangent":
        log_scales = jnp.clip(outputs, MIN_LOG_SCALE, 0.0)
        scales = jnp.exp(log_scales)
        mean_radii = jnp.maximum(jnp.linalg.norm(means, axis=-1), 1e-5)
        radial = means / mean_radii[:, None]
        tangent = jnp.stack((-radial[:, 1], radial[:, 0]), axis=-1)
        tangent_offsets = scales[:, None, 0] * noise[..., 0]
        proposal = means[:, None, :] + tangent_offsets[..., None] * tangent[:, None, :]
        proposal /= jnp.maximum(jnp.linalg.norm(proposal, axis=-1, keepdims=True), 1e-6)
        sampled_radii = mean_radii[:, None] + scales[:, None, 1] * noise[..., 1]
        actions = sampled_radii[..., None] * proposal

        # Exact density of (tangent offset, radial offset) -> Cartesian action.
        # With delta=atan(t/r), the absolute Jacobian is
        # |rho| * r / (r^2 + t^2).
        fixed_actions = jax.lax.stop_gradient(actions)
        fixed_radii = jnp.maximum(jnp.linalg.norm(fixed_actions, axis=-1), 1e-6)
        fixed_angles = jnp.arctan2(fixed_actions[..., 1], fixed_actions[..., 0])
        mean_angles = jnp.arctan2(means[:, 1], means[:, 0])
        angle_offsets = wrap_angle(fixed_angles - mean_angles[:, None])
        inverse_tangent = mean_radii[:, None] * jnp.tan(angle_offsets)
        inverse_radial = fixed_radii - mean_radii[:, None]
        z0 = inverse_tangent / scales[:, None, 0]
        z1 = inverse_radial / scales[:, None, 1]
        log_jacobian = (
            jnp.log(fixed_radii)
            + jnp.log(mean_radii)[:, None]
            - jnp.log(mean_radii[:, None] ** 2 + inverse_tangent**2)
        )
        log_probs = (
            -0.5 * (z0**2 + z1**2)
            - jnp.sum(log_scales, axis=-1)[:, None]
            - np.log(2 * np.pi)
            - log_jacobian
        )
        return actions, log_probs, means

    if condition.family == "direction_speed":
        log_scales = jnp.clip(outputs, MIN_LOG_SCALE, 0.0)
        scales = jnp.exp(log_scales)
    else:
        local_scale = jnp.exp(jnp.clip(outputs[..., 0], MIN_LOG_SCALE, 0.0))
        mean_radii = jnp.maximum(jnp.linalg.norm(means, axis=-1), 1e-3)
        scales = jnp.stack((local_scale / mean_radii, local_scale), axis=-1)
        log_scales = jnp.log(scales)
    mean_angles = jnp.arctan2(means[:, 1], means[:, 0])
    mean_radii = jnp.maximum(jnp.linalg.norm(means, axis=-1), 1e-3)
    latent_angles = mean_angles[:, None] + scales[:, None, 0] * noise[..., 0]
    latent_radii = mean_radii[:, None] + scales[:, None, 1] * noise[..., 1]
    actions = jnp.stack(
        (latent_radii * jnp.cos(latent_angles), latent_radii * jnp.sin(latent_angles)),
        axis=-1,
    )
    fixed_actions = jax.lax.stop_gradient(actions)
    fixed_angles = jnp.arctan2(fixed_actions[..., 1], fixed_actions[..., 0])
    fixed_radii = jnp.maximum(jnp.linalg.norm(fixed_actions, axis=-1), 1e-6)
    z0 = wrap_angle(fixed_angles - mean_angles[:, None]) / scales[:, None, 0]
    z1 = (fixed_radii - mean_radii[:, None]) / scales[:, None, 1]
    log_probs = (
        -0.5 * (z0**2 + z1**2)
        - jnp.sum(log_scales, axis=-1)[:, None]
        - np.log(2 * np.pi)
        - jnp.log(fixed_radii)
    )
    return actions, log_probs, means


def make_train_step(condition, features, angles, target_offset, batch_size, lr):
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))

    def loss_fn(params, keys):
        def one(seed_params, seed_key):
            actions, log_probs, _ = sample_single(
                seed_params,
                seed_key,
                condition,
                features,
                angles,
                target_offset,
                batch_size,
            )
            rewards = shell_hit(actions, angles + target_offset)
            advantages = jax.lax.stop_gradient(
                rewards - rewards.mean(axis=1, keepdims=True)
            )
            return -jnp.mean(advantages * log_probs), rewards.mean()

        losses, success = jax.vmap(one)(params, keys)
        return losses.mean(), success

    @jax.jit
    def step(params, state, keys):
        (loss, success), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            params, keys
        )
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state, loss, success

    return optimizer, step


def make_evaluator(condition, features, angles, target_offset, batch_size):
    @jax.jit
    def evaluate(params, keys):
        def one(seed_params, seed_key):
            actions, _, means = sample_single(
                seed_params,
                seed_key,
                condition,
                features,
                angles,
                target_offset,
                batch_size,
            )
            success = shell_hit(actions, angles + target_offset).mean(axis=1)
            mean_error = jnp.linalg.norm(means - targets(angles, target_offset), axis=-1)
            return success, mean_error

        return jax.vmap(one)(params, keys)

    return evaluate


def train_condition(condition, seeds, train_angles, test_angles, args):
    params = initial_parameters(condition, seeds)
    train_features = context_features(train_angles)
    optimizer, train_step = make_train_step(
        condition,
        train_features,
        train_angles,
        args.target_offset,
        args.batch_size,
        args.learning_rate,
    )
    state = optimizer.init(params)
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    checkpoints = sorted(set((0, *args.checkpoints, args.updates)))
    all_angles = jnp.concatenate((train_angles, test_angles))
    all_features = context_features(all_angles)
    evaluator = make_evaluator(
        condition,
        all_features,
        all_angles,
        args.target_offset,
        args.evaluation_size,
    )
    rows = []

    def record(update):
        eval_keys = jnp.stack(
            [jax.random.PRNGKey(10_000_000 + update * 1000 + seed) for seed in seeds]
        )
        success, mean_error = jax.device_get(evaluator(params, eval_keys))
        split_labels = ["train"] * len(train_angles) + ["test"] * len(test_angles)
        for seed_index, seed in enumerate(seeds):
            for angle_index, angle in enumerate(np.asarray(all_angles)):
                rows.append(
                    {
                        "condition": condition.name,
                        "seed": seed,
                        "update": update,
                        "split": split_labels[angle_index],
                        "context_angle_deg": float(np.rad2deg(angle) % 360.0),
                        "success": float(success[seed_index, angle_index]),
                        "mean_error": float(mean_error[seed_index, angle_index]),
                        "parameter_count": condition.parameter_count,
                    }
                )

    record(0)
    for update in range(1, args.updates + 1):
        split = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
        keys, step_keys = split[:, 0], split[:, 1]
        params, state, _, _ = train_step(params, state, step_keys)
        if update in checkpoints:
            record(update)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_path, updates, conditions):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for condition in conditions:
        means = []
        for update in updates:
            selected = [
                row["success"]
                for row in rows
                if row["condition"] == condition.name
                and row["split"] == "test"
                and row["update"] == update
            ]
            means.append(100 * np.mean(selected))
        axes[0].plot(updates, means, marker="o", label=condition.name)
    axes[0].set(xlabel="Policy updates", ylabel="Held-out success (%)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    final = max(updates)
    x = np.arange(len(conditions))
    success = []
    error = []
    for condition in conditions:
        selected = [
            row
            for row in rows
            if row["condition"] == condition.name
            and row["split"] == "test"
            and row["update"] == final
        ]
        success.append(100 * np.mean([row["success"] for row in selected]))
        error.append(np.mean([row["mean_error"] for row in selected]))
    axes[1].bar(x, success)
    axes[1].set_xticks(
        x, [condition.name for condition in conditions], rotation=25, ha="right"
    )
    axes[1].set_ylabel("Final held-out success (%)")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=700)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[100, 300, 600])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--evaluation-size", type=int, default=5000)
    parser.add_argument("--target-offset-deg", type=float, default=30.0)
    parser.add_argument(
        "--context-step-deg",
        type=float,
        default=30.0,
        help="Spacing of training rotations; held-out contexts use midpoints.",
    )
    parser.add_argument("--output-prefix", default="joint_contextual_policy_pilot")
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    parser.add_argument(
        "--conditions",
        nargs="+",
        help=(
            "Optional condition names. The Cartesian mean-frame counterfactual "
            "is available here but is not part of the frozen formal set."
        ),
    )
    args = parser.parse_args()
    args.target_offset = np.deg2rad(args.target_offset_deg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    train_angles = jnp.deg2rad(jnp.arange(0.0, 360.0, args.context_step_deg))
    test_angles = jnp.deg2rad(
        jnp.arange(
            args.context_step_deg / 2.0,
            360.0,
            args.context_step_deg,
        )
    )
    available_conditions = (
        *CONDITIONS,
        MEAN_FRAME_CONDITION,
        PROJECTED_TANGENT_CONDITION,
    )
    if args.conditions:
        by_name = {condition.name: condition for condition in available_conditions}
        unknown = sorted(set(args.conditions) - set(by_name))
        if unknown:
            parser.error(f"Unknown conditions: {unknown}")
        conditions = tuple(by_name[name] for name in args.conditions)
    else:
        conditions = CONDITIONS
    rows = []
    for condition in conditions:
        print(f"training {condition.name} ({condition.parameter_count} params)")
        rows.extend(train_condition(condition, seeds, train_angles, test_angles, args))
    csv_path = args.output_dir / f"{args.output_prefix}_rows.csv"
    write_csv(csv_path, rows)
    updates = sorted({row["update"] for row in rows})
    plot(rows, args.output_dir / f"{args.output_prefix}.png", updates, conditions)
    print(csv_path)


if __name__ == "__main__":
    main()
