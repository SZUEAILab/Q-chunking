#!/usr/bin/env python3
"""Learn a sparse two-stage shell task with REINFORCE policy families."""

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


TARGET_ANGLE = np.deg2rad(30.0)
ANGLE_TOLERANCE = np.deg2rad(7.5)
APPROACH_RADIUS = 0.75
APPROACH_RADIAL_TOLERANCE = 0.025
PRECISION_RADIUS = 0.15
PRECISION_RADIAL_TOLERANCE = 0.015
BASE_SIGMA = 0.08
DS_TANGENTIAL_STD = 0.25
DS_RADIAL_STD = BASE_SIGMA**2 / DS_TANGENTIAL_STD
CARTESIAN_ORACLE_SIGMA = 0.27
PRECISION_STD = 0.01


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    shared_scale: bool = False
    structured_covariance: bool = False
    coupled_radius: bool = False


CONDITIONS = (
    Condition("Cartesian global", "cartesian_diag", shared_scale=True),
    Condition("Cartesian stage", "cartesian_diag"),
    Condition("Cartesian full-cov", "cartesian_full"),
    Condition(
        "Cartesian full-cov oracle",
        "cartesian_full",
        structured_covariance=True,
    ),
    Condition("DS global", "direction_speed", shared_scale=True),
    Condition("DS stage", "direction_speed"),
    Condition(
        "DS stage, coupled radius",
        "direction_speed",
        coupled_radius=True,
    ),
)

# Optional strong control kept outside the frozen on-policy CONDITIONS tuple.
PROJECTED_TANGENT_CONDITION = Condition(
    "Cartesian projected-tangent stage",
    "cartesian_projected_tangent",
)


def set_target_angle(degrees):
    global TARGET_ANGLE
    TARGET_ANGLE = np.deg2rad(float(degrees))


def wrap_angle(value):
    return jnp.arctan2(jnp.sin(value), jnp.cos(value))


def shell_hit(actions, target_radius, radial_tolerance):
    radii = jnp.linalg.norm(actions, axis=-1)
    angles = jnp.arctan2(actions[..., 1], actions[..., 0])
    return (
        (jnp.abs(wrap_angle(angles - TARGET_ANGLE)) < ANGLE_TOLERANCE)
        & (jnp.abs(radii - target_radius) < radial_tolerance)
    ).astype(jnp.float32)


def initial_parameters(condition, num_seeds):
    target_direction = np.asarray([np.cos(TARGET_ANGLE), np.sin(TARGET_ANGLE)])
    if condition.family.startswith("cartesian"):
        mean = np.stack(
            (
                np.asarray([APPROACH_RADIUS, 0.0]),
                PRECISION_RADIUS * target_direction,
            )
        )
        params = {"mean": np.broadcast_to(mean, (num_seeds, 2, 2)).copy()}
        if condition.family == "cartesian_projected_tangent":
            scales = np.asarray(
                [
                    [DS_TANGENTIAL_STD, DS_RADIAL_STD],
                    [PRECISION_STD, PRECISION_STD],
                ]
            )
        elif condition.shared_scale:
            scales = np.asarray([[CARTESIAN_ORACLE_SIGMA] * 2])
        elif condition.structured_covariance:
            scales = np.asarray(
                [
                    [DS_RADIAL_STD, DS_TANGENTIAL_STD],
                    [PRECISION_STD, PRECISION_STD],
                ]
            )
        else:
            scales = np.asarray(
                [
                    [CARTESIAN_ORACLE_SIGMA] * 2,
                    [PRECISION_STD] * 2,
                ]
            )
        params["log_scale"] = np.broadcast_to(
            np.log(scales), (num_seeds,) + scales.shape
        ).copy()
        if condition.family == "cartesian_full":
            params["off_diagonal"] = np.zeros((num_seeds, 2))
        return jax.tree_util.tree_map(jnp.asarray, params)

    means = np.asarray(
        [
            [0.0, APPROACH_RADIUS],
            [TARGET_ANGLE, PRECISION_RADIUS],
        ]
    )
    if condition.shared_scale:
        scales = np.asarray(
            [[DS_TANGENTIAL_STD / APPROACH_RADIUS, DS_RADIAL_STD]]
        )
    else:
        approach_radial = BASE_SIGMA if condition.coupled_radius else DS_RADIAL_STD
        scales = np.asarray(
            [
                [DS_TANGENTIAL_STD / APPROACH_RADIUS, approach_radial],
                [PRECISION_STD / PRECISION_RADIUS, PRECISION_STD],
            ]
        )
    return {
        "mean": jnp.asarray(np.broadcast_to(means, (num_seeds, 2, 2)).copy()),
        "log_scale": jnp.asarray(
            np.broadcast_to(np.log(scales), (num_seeds,) + scales.shape).copy()
        ),
    }


def parameter_count(condition):
    params = initial_parameters(condition, 1)
    return int(sum(np.prod(value.shape[1:]) for value in jax.tree.leaves(params)))


def expand_stage_scales(log_scale):
    if log_scale.shape[0] == 1:
        log_scale = jnp.repeat(log_scale, 2, axis=0)
    return jnp.exp(jnp.clip(log_scale, -6.0, 0.0))


def sample_policy(params, key, condition, batch_size):
    noise = jax.random.normal(key, (2, batch_size, 2))
    scales = expand_stage_scales(params["log_scale"])
    if condition.family == "cartesian_diag":
        actions = params["mean"][:, None, :] + scales[:, None, :] * noise
        fixed_actions = jax.lax.stop_gradient(actions)
        standardized = (fixed_actions - params["mean"][:, None, :]) / scales[:, None, :]
        log_probs = (
            -0.5 * jnp.sum(standardized**2, axis=-1)
            - jnp.sum(jnp.log(scales), axis=-1)[:, None]
            - np.log(2.0 * np.pi)
        )
        return actions, log_probs

    if condition.family == "cartesian_full":
        off_diagonal = jnp.clip(params["off_diagonal"], -1.0, 1.0)
        first = scales[:, 0, None] * noise[..., 0]
        second = (
            off_diagonal[:, None] * noise[..., 0]
            + scales[:, 1, None] * noise[..., 1]
        )
        actions = params["mean"][:, None, :] + jnp.stack((first, second), axis=-1)
        fixed_actions = jax.lax.stop_gradient(actions)
        residual = fixed_actions - params["mean"][:, None, :]
        whitened_first = residual[..., 0] / scales[:, 0, None]
        whitened_second = (
            residual[..., 1] - off_diagonal[:, None] * whitened_first
        ) / scales[:, 1, None]
        log_probs = (
            -0.5 * (whitened_first**2 + whitened_second**2)
            - jnp.sum(jnp.log(scales), axis=-1)[:, None]
            - np.log(2.0 * np.pi)
        )
        return actions, log_probs

    if condition.family == "cartesian_projected_tangent":
        means = params["mean"]
        mean_radii = jnp.maximum(jnp.linalg.norm(means, axis=-1), 1e-5)
        radial = means / mean_radii[:, None]
        tangent = jnp.stack((-radial[:, 1], radial[:, 0]), axis=-1)
        tangent_offsets = scales[:, None, 0] * noise[..., 0]
        proposal = means[:, None, :] + tangent_offsets[..., None] * tangent[:, None, :]
        proposal /= jnp.maximum(jnp.linalg.norm(proposal, axis=-1, keepdims=True), 1e-6)
        sampled_radii = mean_radii[:, None] + scales[:, None, 1] * noise[..., 1]
        actions = sampled_radii[..., None] * proposal

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
            - jnp.sum(jnp.log(scales), axis=-1)[:, None]
            - np.log(2.0 * np.pi)
            - log_jacobian
        )
        return actions, log_probs

    latent = params["mean"][:, None, :] + scales[:, None, :] * noise
    fixed_latent = jax.lax.stop_gradient(latent)
    standardized = (fixed_latent - params["mean"][:, None, :]) / scales[:, None, :]
    log_probs = (
        -0.5 * jnp.sum(standardized**2, axis=-1)
        - jnp.sum(jnp.log(scales), axis=-1)[:, None]
        - np.log(2.0 * np.pi)
    )
    actions = jnp.stack(
        (
            latent[..., 1] * jnp.cos(latent[..., 0]),
            latent[..., 1] * jnp.sin(latent[..., 0]),
        ),
        axis=-1,
    )
    return actions, log_probs


def single_seed_loss(params, key, condition, batch_size):
    actions, log_probs = sample_policy(params, key, condition, batch_size)
    discovery = shell_hit(
        actions[0], APPROACH_RADIUS, APPROACH_RADIAL_TOLERANCE
    )
    precision_hit = shell_hit(
        actions[1], PRECISION_RADIUS, PRECISION_RADIAL_TOLERANCE
    )
    terminal = discovery * precision_hit

    approach_return = discovery + terminal
    approach_advantage = jax.lax.stop_gradient(
        approach_return - approach_return.mean()
    )
    reached = jnp.maximum(discovery.sum(), 1.0)
    precision_baseline = terminal.sum() / reached
    precision_advantage = jax.lax.stop_gradient(
        discovery * (terminal - precision_baseline)
    )
    loss = -(
        jnp.mean(approach_advantage * log_probs[0])
        + jnp.sum(precision_advantage * log_probs[1]) / reached
    )
    return loss, {
        "discovery": discovery.mean(),
        "precision": terminal.sum() / reached,
        "success": terminal.mean(),
    }


def make_train_step(condition, batch_size, learning_rate):
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate),
    )

    def all_losses(params, keys):
        losses, metrics = jax.vmap(
            lambda seed_params, seed_key: single_seed_loss(
                seed_params, seed_key, condition, batch_size
            )
        )(params, keys)
        return losses.mean(), metrics

    @jax.jit
    def train_step(params, optimizer_state, keys):
        (loss, metrics), gradients = jax.value_and_grad(
            all_losses, has_aux=True
        )(params, keys)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss, metrics

    return optimizer, train_step


def make_evaluator(condition, evaluation_size):
    @jax.jit
    def evaluate(params, keys):
        def one(seed_params, seed_key):
            actions, _ = sample_policy(
                seed_params, seed_key, condition, evaluation_size
            )
            discovery = shell_hit(
                actions[0], APPROACH_RADIUS, APPROACH_RADIAL_TOLERANCE
            )
            precision_hit = shell_hit(
                actions[1], PRECISION_RADIUS, PRECISION_RADIAL_TOLERANCE
            )
            terminal = discovery * precision_hit
            return {
                "discovery": discovery.mean(),
                "precision": terminal.sum() / jnp.maximum(discovery.sum(), 1.0),
                "success": terminal.mean(),
            }

        return jax.vmap(one)(params, keys)

    return evaluate


def train_condition(
    condition,
    seeds,
    iterations,
    batch_size,
    evaluation_size,
    evaluation_interval,
    learning_rate,
):
    params = initial_parameters(condition, len(seeds))
    optimizer, train_step = make_train_step(
        condition, batch_size, learning_rate
    )
    optimizer_state = optimizer.init(params)
    train_keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    evaluation_keys = jnp.stack(
        [jax.random.PRNGKey(10_000_000 + seed) for seed in seeds]
    )
    evaluator = make_evaluator(condition, evaluation_size)
    rows = []

    def record(iteration):
        metrics = jax.device_get(evaluator(params, evaluation_keys))
        for index, seed in enumerate(seeds):
            rows.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "iteration": iteration,
                    "stage0_interactions": iteration * batch_size,
                    "target_angle_deg": float(np.rad2deg(TARGET_ANGLE)),
                    "discovery": float(metrics["discovery"][index]),
                    "precision": float(metrics["precision"][index]),
                    "success": float(metrics["success"][index]),
                }
            )

    record(0)
    for iteration in range(1, iterations + 1):
        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(train_keys)
        train_keys = split_keys[:, 0]
        step_keys = split_keys[:, 1]
        params, optimizer_state, _, _ = train_step(
            params, optimizer_state, step_keys
        )
        if iteration % evaluation_interval == 0 or iteration == iterations:
            record(iteration)
    return rows, jax.device_get(params)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, conditions, seeds):
    summary = []
    run_summary = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition.name]
        steps = sorted({row["stage0_interactions"] for row in selected})
        for step in steps:
            checkpoint = [row for row in selected if row["stage0_interactions"] == step]
            item = {
                "condition": condition.name,
                "parameter_count": parameter_count(condition),
                "stage0_interactions": step,
            }
            for metric in ("discovery", "precision", "success"):
                values = np.asarray([row[metric] for row in checkpoint])
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std(ddof=1))
            summary.append(item)

        per_seed = []
        for seed in seeds:
            curve = sorted(
                [row for row in selected if row["seed"] == seed],
                key=lambda row: row["stage0_interactions"],
            )
            x = np.asarray([row["stage0_interactions"] for row in curve])
            y = np.asarray([row["success"] for row in curve])
            reached = x[y >= 0.5]
            per_seed.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "final_success": float(y[-1]),
                    "auc": float(np.trapezoid(y, x) / x[-1]),
                    "steps_to_50": float(reached[0]) if len(reached) else np.nan,
                }
            )
        for metric in ("final_success", "auc", "steps_to_50"):
            values = np.asarray([row[metric] for row in per_seed])
            finite = values[np.isfinite(values)]
            value_mean = float(finite.mean()) if len(finite) else np.nan
            value_std = float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            if not run_summary or run_summary[-1]["condition"] != condition.name:
                run_summary.append(
                    {
                        "condition": condition.name,
                        "parameter_count": parameter_count(condition),
                    }
                )
            run_summary[-1][f"{metric}_mean"] = value_mean
            run_summary[-1][f"{metric}_std"] = value_std
            if metric == "steps_to_50":
                run_summary[-1]["fraction_reaching_50"] = float(
                    np.mean(np.isfinite(values))
                )
        run_summary[-1]["per_seed_final_success"] = ";".join(
            f"{row['final_success']:.6f}" for row in per_seed
        )
        run_summary[-1]["per_seed_auc"] = ";".join(
            f"{row['auc']:.6f}" for row in per_seed
        )
    return summary, run_summary


def parameter_diagnostics(params, condition, seeds):
    params = jax.tree_util.tree_map(np.asarray, params)
    rows = []
    for seed_index, seed in enumerate(seeds):
        means = params["mean"][seed_index]
        log_scales = params["log_scale"][seed_index]
        if len(log_scales) == 1:
            log_scales = np.repeat(log_scales, 2, axis=0)
        scales = np.exp(np.clip(log_scales, -6.0, 0.0))
        row = {"condition": condition.name, "seed": seed}
        if condition.family == "direction_speed":
            for stage in range(2):
                row[f"stage{stage}_mean_angle"] = float(means[stage, 0])
                row[f"stage{stage}_mean_radius"] = float(means[stage, 1])
                row[f"stage{stage}_angular_std"] = float(scales[stage, 0])
                row[f"stage{stage}_radial_std"] = float(scales[stage, 1])
                row[f"stage{stage}_tangential_action_std"] = float(
                    abs(means[stage, 1]) * scales[stage, 0]
                )
                row[f"stage{stage}_off_diagonal"] = 0.0
        else:
            off_diagonal = (
                np.clip(params["off_diagonal"][seed_index], -1.0, 1.0)
                if condition.family == "cartesian_full"
                else np.zeros(2)
            )
            for stage in range(2):
                mean = means[stage]
                radius = max(float(np.linalg.norm(mean)), 1e-12)
                direction = mean / radius
                tangent = np.asarray([-direction[1], direction[0]])
                matrix = np.asarray(
                    [
                        [scales[stage, 0], 0.0],
                        [off_diagonal[stage], scales[stage, 1]],
                    ]
                )
                covariance = matrix @ matrix.T
                row[f"stage{stage}_mean_angle"] = float(
                    np.arctan2(mean[1], mean[0])
                )
                row[f"stage{stage}_mean_radius"] = radius
                row[f"stage{stage}_angular_std"] = np.nan
                row[f"stage{stage}_radial_std"] = float(
                    np.sqrt(direction @ covariance @ direction)
                )
                row[f"stage{stage}_tangential_action_std"] = float(
                    np.sqrt(tangent @ covariance @ tangent)
                )
                row[f"stage{stage}_off_diagonal"] = float(off_diagonal[stage])
        rows.append(row)
    return rows


def plot_results(summary, run_summary, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    colors = plt.get_cmap("tab10")
    for index, condition in enumerate(CONDITIONS):
        curve = [row for row in summary if row["condition"] == condition.name]
        x = np.asarray([row["stage0_interactions"] for row in curve]) / 1000
        mean = np.asarray([row["success_mean"] for row in curve]) * 100
        std = np.asarray([row["success_std"] for row in curve]) * 100
        axes[0, 0].plot(x, mean, label=condition.name, color=colors(index))
        axes[0, 0].fill_between(
            x,
            np.maximum(0.0, mean - std),
            np.minimum(100.0, mean + std),
            color=colors(index),
            alpha=0.10,
        )
    axes[0, 0].set_xlabel("Stage-0 interactions (k)")
    axes[0, 0].set_ylabel("One-shot end-to-end success (%)")
    axes[0, 0].legend(fontsize=7, ncol=2)

    x = np.arange(len(run_summary))
    labels = [row["condition"] for row in run_summary]
    axes[0, 1].bar(
        x,
        [100 * row["final_success_mean"] for row in run_summary],
        yerr=[100 * row["final_success_std"] for row in run_summary],
        color=[colors(index) for index in range(len(run_summary))],
        capsize=3,
    )
    axes[0, 1].set_ylabel("Final success (%)")
    axes[0, 1].set_xticks(x, labels, rotation=30, ha="right")

    axes[1, 0].bar(
        x,
        [100 * row["auc_mean"] for row in run_summary],
        yerr=[100 * row["auc_std"] for row in run_summary],
        color=[colors(index) for index in range(len(run_summary))],
        capsize=3,
    )
    axes[1, 0].set_ylabel("Normalized success AUC (%)")
    axes[1, 0].set_xticks(x, labels, rotation=30, ha="right")

    final_step = max(row["stage0_interactions"] for row in summary)
    threshold_values = [
        row["steps_to_50_mean"] / 1000
        if np.isfinite(row["steps_to_50_mean"])
        else final_step / 1000
        for row in run_summary
    ]
    bars = axes[1, 1].bar(
        x,
        threshold_values,
        color=[colors(index) for index in range(len(run_summary))],
    )
    for bar, row in zip(bars, run_summary, strict=True):
        if row["fraction_reaching_50"] < 1.0:
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{100 * row['fraction_reaching_50']:.0f}% reached",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=7,
            )
    axes[1, 1].set_ylabel("Interactions to 50% success (k)")
    axes[1, 1].set_xticks(x, labels, rotation=30, ha="right")
    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    figure.suptitle("Sparse two-stage shell MDP: learned policy distributions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-size", type=int, default=20_000)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--target-angle-deg", type=float, default=30.0)
    parser.add_argument("--output-prefix", default="learned_shell_pilot")
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    set_target_angle(args.target_angle_deg)
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    parameter_rows = []
    for condition in CONDITIONS:
        condition_rows, final_params = train_condition(
            condition,
            seeds,
            args.iterations,
            args.batch_size,
            args.evaluation_size,
            args.evaluation_interval,
            args.learning_rate,
        )
        rows.extend(condition_rows)
        parameter_rows.extend(parameter_diagnostics(final_params, condition, seeds))
        print(f"finished {condition.name}")
    summary, run_summary = summarize(rows, CONDITIONS, seeds)
    write_csv(args.output_dir / f"{args.output_prefix}_per_seed.csv", rows)
    write_csv(args.output_dir / f"{args.output_prefix}_curves.csv", summary)
    write_csv(args.output_dir / f"{args.output_prefix}_summary.csv", run_summary)
    write_csv(
        args.output_dir / f"{args.output_prefix}_parameters.csv", parameter_rows
    )
    plot_results(
        summary,
        run_summary,
        args.output_dir / f"{args.output_prefix}.png",
    )
    for row in run_summary:
        print(row)


if __name__ == "__main__":
    main()
