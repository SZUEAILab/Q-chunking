#!/usr/bin/env python3
"""Off-policy twin-Q bridge for the sparse two-stage shell MDP."""

import argparse
import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from train_synthetic_shell_rl import (
    ANGLE_TOLERANCE,
    APPROACH_RADIAL_TOLERANCE,
    APPROACH_RADIUS,
    CONDITIONS,
    PRECISION_RADIAL_TOLERANCE,
    PRECISION_RADIUS,
    PROJECTED_TANGENT_CONDITION,
    Condition,
    initial_parameters,
    parameter_count,
    sample_policy,
    set_target_angle,
    shell_hit,
)


OFFPOLICY_CONDITIONS = tuple(
    condition
    for condition in CONDITIONS
    if condition.name
    in (
        "Cartesian stage",
        "Cartesian full-cov oracle",
        "DS global",
        "DS stage",
        "DS stage, coupled radius",
    )
) + (PROJECTED_TANGENT_CONDITION,)


def init_single_q(key, hidden_dim, output_bias):
    keys = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(keys[0], (2, hidden_dim)) * np.sqrt(2.0 / 2.0),
        "b1": jnp.zeros((hidden_dim,)),
        "w2": jax.random.normal(keys[1], (hidden_dim, hidden_dim))
        * np.sqrt(2.0 / hidden_dim),
        "b2": jnp.zeros((hidden_dim,)),
        # Start pessimistically and without arbitrary action-space peaks.  The
        # hidden layers still break symmetry as soon as supervised TD errors
        # reach the output layer.
        "w3": jnp.zeros((hidden_dim, 1)),
        "b3": jnp.full((1,), output_bias),
    }


def init_critics(key, num_seeds, hidden_dim, output_bias):
    keys = jax.random.split(key, num_seeds * 2 * 2).reshape(
        num_seeds, 2, 2, 2
    )
    return jax.vmap(
        jax.vmap(
            jax.vmap(
                lambda q_key: init_single_q(q_key, hidden_dim, output_bias)
            )
        )
    )(keys)


def single_q_logits(params, actions):
    hidden = jax.nn.relu(actions @ params["w1"] + params["b1"])
    hidden = jax.nn.relu(hidden @ params["w2"] + params["b2"])
    return (hidden @ params["w3"] + params["b3"])[..., 0]


def all_q_logits(params, actions):
    """Evaluate env-action critics; output is (seed, stage, twin, batch)."""

    def one_seed(seed_params, seed_actions):
        return jax.vmap(
            lambda stage_params, stage_actions: jax.vmap(
                lambda twin_params: single_q_logits(twin_params, stage_actions)
            )(stage_params)
        )(seed_params, seed_actions)

    return jax.vmap(one_seed)(params, actions)


def stage_targets(discovery, terminal, next_stage_value, discount):
    """Return bounded Q targets for stages 0 and 1.

    Stage-0 return is normalized by ``1 + discount`` so both critics predict a
    number in [0, 1].  The stage-1 target is observed only for episodes that
    actually discovered the approach shell; its loss is masked separately.
    """
    stage0 = discovery * (1.0 + discount * next_stage_value) / (1.0 + discount)
    return stage0, terminal


def project_actor(params, condition):
    params = dict(params)
    params["log_scale"] = jnp.clip(params["log_scale"], -6.0, 0.0)
    mean = params["mean"]
    if condition.family == "direction_speed":
        angle = jnp.arctan2(jnp.sin(mean[..., 0]), jnp.cos(mean[..., 0]))
        radius = jnp.clip(mean[..., 1], -1.5, 1.5)
        params["mean"] = jnp.stack((angle, radius), axis=-1)
    else:
        params["mean"] = jnp.clip(mean, -1.5, 1.5)
    if "off_diagonal" in params:
        params["off_diagonal"] = jnp.clip(params["off_diagonal"], -1.0, 1.0)
    return params


def sample_all(actor_params, keys, condition, batch_size):
    return jax.vmap(
        lambda seed_params, seed_key: sample_policy(
            seed_params, seed_key, condition, batch_size
        )[0]
    )(actor_params, keys)


def policy_log_prob(actor_params, actions, condition):
    """Log density of replayed env actions for AWR actor fitting."""
    log_scales = jnp.clip(actor_params["log_scale"], -6.0, 0.0)
    if condition.shared_scale:
        log_scales = jnp.repeat(log_scales, 2, axis=1)
    scales = jnp.exp(log_scales)
    means = actor_params["mean"]

    if condition.family == "direction_speed":
        radii = jnp.linalg.norm(actions, axis=-1)
        angles = jnp.arctan2(actions[..., 1], actions[..., 0])
        angle_residual = jnp.arctan2(
            jnp.sin(angles - means[..., 0, None]),
            jnp.cos(angles - means[..., 0, None]),
        )
        radius_residual = radii - means[..., 1, None]
        residual = jnp.stack((angle_residual, radius_residual), axis=-1)
        standardized = residual / scales[:, :, None, :]
    elif condition.family == "cartesian_projected_tangent":
        mean_radii = jnp.maximum(jnp.linalg.norm(means, axis=-1), 1e-5)
        fixed_radii = jnp.maximum(jnp.linalg.norm(actions, axis=-1), 1e-6)
        fixed_angles = jnp.arctan2(actions[..., 1], actions[..., 0])
        mean_angles = jnp.arctan2(means[..., 1], means[..., 0])
        angle_offsets = jnp.arctan2(
            jnp.sin(fixed_angles - mean_angles[..., None]),
            jnp.cos(fixed_angles - mean_angles[..., None]),
        )
        inverse_tangent = mean_radii[..., None] * jnp.tan(angle_offsets)
        inverse_radial = fixed_radii - mean_radii[..., None]
        standardized = jnp.stack(
            (
                inverse_tangent / scales[..., 0, None],
                inverse_radial / scales[..., 1, None],
            ),
            axis=-1,
        )
        log_jacobian = (
            jnp.log(fixed_radii)
            + jnp.log(mean_radii)[..., None]
            - jnp.log(mean_radii[..., None] ** 2 + inverse_tangent**2)
        )
    elif condition.family == "cartesian_full":
        residual = actions - means[:, :, None, :]
        off_diagonal = jnp.clip(actor_params["off_diagonal"], -1.0, 1.0)
        first = residual[..., 0] / scales[..., 0, None]
        second = (
            residual[..., 1] - off_diagonal[..., None] * first
        ) / scales[..., 1, None]
        standardized = jnp.stack((first, second), axis=-1)
    else:
        residual = actions - means[:, :, None, :]
        standardized = residual / scales[:, :, None, :]

    result = (
        -0.5 * jnp.sum(jnp.square(standardized), axis=-1)
        - jnp.sum(log_scales, axis=-1)[..., None]
        - np.log(2.0 * np.pi)
    )
    if condition.family == "cartesian_projected_tangent":
        result = result - log_jacobian
    return result


def make_updates(
    condition,
    actor_batch_size,
    discount,
    actor_lr,
    critic_lr,
    tau,
    actor_entropy_coefficient,
    actor_update_mode,
    awr_temperature,
):
    actor_optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(actor_lr))
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(10.0), optax.adam(critic_lr)
    )

    @jax.jit
    def critic_update(
        critic_params,
        target_params,
        critic_state,
        actor_params,
        batch,
        target_keys,
    ):
        next_actions = sample_all(
            actor_params, target_keys, condition, batch["actions"].shape[2]
        )
        target_logits = all_q_logits(target_params, next_actions)
        next_stage_value = jax.nn.sigmoid(target_logits[:, 1]).min(axis=1)
        target0, target1 = stage_targets(
            batch["discovery"],
            batch["terminal"],
            jax.lax.stop_gradient(next_stage_value),
            discount,
        )

        def loss_fn(params):
            logits = all_q_logits(params, batch["actions"])
            probabilities = jax.nn.sigmoid(logits)
            error0 = optax.sigmoid_binary_cross_entropy(
                logits[:, 0], target0[:, None, :]
            )
            error1 = optax.sigmoid_binary_cross_entropy(
                logits[:, 1], target1[:, None, :]
            )
            loss0 = error0.mean(axis=(1, 2))
            mask = batch["discovery"][:, None, :]
            loss1 = (error1 * mask).sum(axis=(1, 2)) / jnp.maximum(
                mask.sum(axis=(1, 2)) * 2.0, 1.0
            )
            loss = (loss0 + loss1).mean()
            return loss, {
                "critic_loss": loss,
                "critic_stage0_loss": loss0.mean(),
                "critic_stage1_loss": loss1.mean(),
                "q0": probabilities[:, 0].mean(),
                "q1": probabilities[:, 1].mean(),
            }

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            critic_params
        )
        updates, critic_state = critic_optimizer.update(
            gradients, critic_state, critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)
        target_params = jax.tree_util.tree_map(
            lambda target, current: (1.0 - tau) * target + tau * current,
            target_params,
            critic_params,
        )
        return critic_params, target_params, critic_state, metrics

    @jax.jit
    def actor_update(actor_params, actor_state, critic_params, keys, batch):
        def loss_fn(params):
            if actor_update_mode == "q_gradient":
                actions = sample_all(params, keys, condition, actor_batch_size)
                probabilities = jax.nn.sigmoid(
                    all_q_logits(critic_params, actions)
                )
                conservative_q = probabilities.min(axis=2)
                stage1_valid = (batch["discovery"].sum(axis=1) > 0).astype(
                    jnp.float32
                )
                exploitation_loss = -(
                    conservative_q[:, 0].mean(axis=-1)
                    + stage1_valid * conservative_q[:, 1].mean(axis=-1)
                )
            else:
                actions = batch["actions"]
                probabilities = jax.nn.sigmoid(
                    all_q_logits(critic_params, actions)
                )
                conservative_q = jax.lax.stop_gradient(
                    probabilities.min(axis=2)
                )
                log_probs = policy_log_prob(params, actions, condition)
                batch_size = actions.shape[2]
                weights0 = (
                    jax.nn.softmax(
                        conservative_q[:, 0] / awr_temperature, axis=-1
                    )
                    * batch_size
                )
                mask1 = batch["discovery"] > 0.5
                masked_q1 = jnp.where(
                    mask1, conservative_q[:, 1] / awr_temperature, -1e9
                )
                weights1 = jax.nn.softmax(masked_q1, axis=-1) * batch_size
                stage1_valid = mask1.any(axis=1).astype(jnp.float32)
                exploitation_loss = -(
                    (weights0 * log_probs[:, 0]).mean(axis=-1)
                    + stage1_valid
                    * (weights1 * log_probs[:, 1]).mean(axis=-1)
                )
            log_scales = jnp.clip(params["log_scale"], -6.0, 0.0)
            if condition.shared_scale:
                log_scales = jnp.repeat(log_scales, 2, axis=1)
            entropy_proxy = log_scales.mean(axis=(1, 2))
            per_seed = (
                exploitation_loss
                - actor_entropy_coefficient * entropy_proxy
            )
            return per_seed.mean(), {
                "actor_loss": per_seed.mean(),
                "actor_q0": conservative_q[:, 0].mean(),
                "actor_q1": conservative_q[:, 1].mean(),
            }

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            actor_params
        )
        updates, actor_state = actor_optimizer.update(
            gradients, actor_state, actor_params
        )
        actor_params = project_actor(
            optax.apply_updates(actor_params, updates), condition
        )
        return actor_params, actor_state, metrics

    return actor_optimizer, critic_optimizer, critic_update, actor_update


class Replay:
    def __init__(self, num_seeds, capacity):
        self.actions = np.zeros((num_seeds, capacity, 2, 2), dtype=np.float32)
        self.discovery = np.zeros((num_seeds, capacity), dtype=np.float32)
        self.terminal = np.zeros((num_seeds, capacity), dtype=np.float32)
        self.capacity = capacity
        self.pointer = 0
        self.size = 0

    def add(self, actions, discovery, terminal):
        count = actions.shape[2]
        indices = (np.arange(count) + self.pointer) % self.capacity
        self.actions[:, indices] = np.transpose(actions, (0, 2, 1, 3))
        self.discovery[:, indices] = discovery
        self.terminal[:, indices] = terminal
        self.pointer = (self.pointer + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, rng, batch_size, discovery_fraction=0.0):
        if discovery_fraction <= 0.0:
            indices = np.broadcast_to(
                rng.integers(0, self.size, size=batch_size),
                (self.actions.shape[0], batch_size),
            )
        else:
            positive_count = int(round(batch_size * discovery_fraction))
            negative_count = batch_size - positive_count
            per_seed = []
            for seed_index in range(self.actions.shape[0]):
                positives = np.flatnonzero(
                    self.discovery[seed_index, : self.size] > 0.5
                )
                negatives = np.flatnonzero(
                    self.discovery[seed_index, : self.size] <= 0.5
                )
                if len(positives) == 0 or len(negatives) == 0:
                    selected = rng.integers(0, self.size, size=batch_size)
                else:
                    selected = np.concatenate(
                        (
                            rng.choice(positives, size=positive_count, replace=True),
                            rng.choice(negatives, size=negative_count, replace=True),
                        )
                    )
                    rng.shuffle(selected)
                per_seed.append(selected)
            indices = np.stack(per_seed)
        seed_indices = np.arange(self.actions.shape[0])[:, None]
        return {
            "actions": jnp.asarray(
                np.transpose(self.actions[seed_indices, indices], (0, 2, 1, 3))
            ),
            "discovery": jnp.asarray(self.discovery[seed_indices, indices]),
            "terminal": jnp.asarray(self.terminal[seed_indices, indices]),
        }


def make_evaluator(condition, evaluation_size, target_angle_deg):
    target_angle = np.deg2rad(target_angle_deg)

    @jax.jit
    def evaluate(actor_params, keys):
        actions = sample_all(actor_params, keys, condition, evaluation_size)
        discovery = shell_hit(
            actions[:, 0], APPROACH_RADIUS, APPROACH_RADIAL_TOLERANCE
        )
        precision = shell_hit(
            actions[:, 1], PRECISION_RADIUS, PRECISION_RADIAL_TOLERANCE
        )
        terminal = discovery * precision
        norms = jnp.linalg.norm(actions, axis=-1)
        angles = jnp.arctan2(actions[..., 1], actions[..., 0])
        angle_errors = jnp.abs(
            jnp.arctan2(
                jnp.sin(angles - target_angle),
                jnp.cos(angles - target_angle),
            )
        )
        return {
            "discovery": discovery.mean(axis=1),
            "precision": terminal.sum(axis=1) / jnp.maximum(discovery.sum(axis=1), 1.0),
            "success": terminal.mean(axis=1),
            "stage0_norm_mean": norms[:, 0].mean(axis=1),
            "stage1_norm_mean": norms[:, 1].mean(axis=1),
            "stage0_angle_error_deg": jnp.rad2deg(angle_errors[:, 0].mean(axis=1)),
            "stage1_angle_error_deg": jnp.rad2deg(angle_errors[:, 1].mean(axis=1)),
        }

    return evaluate


def train_condition(condition, seeds, args):
    num_seeds = len(seeds)
    actor_params = initial_parameters(condition, num_seeds)
    critic_params = init_critics(
        jax.random.PRNGKey(args.critic_init_seed),
        num_seeds,
        args.hidden_dim,
        args.critic_output_bias,
    )
    target_params = jax.tree_util.tree_map(lambda value: value.copy(), critic_params)
    actor_optimizer, critic_optimizer, critic_update, actor_update = make_updates(
        condition,
        args.actor_batch_size,
        args.discount,
        args.actor_learning_rate,
        args.critic_learning_rate,
        args.tau,
        args.actor_entropy_coefficient,
        args.actor_update,
        args.awr_temperature,
    )
    actor_state = actor_optimizer.init(actor_params)
    critic_state = critic_optimizer.init(critic_params)
    replay = Replay(num_seeds, args.replay_capacity)
    replay_rng = np.random.default_rng(args.replay_seed)
    train_keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    evaluation_keys = jnp.stack(
        [jax.random.PRNGKey(20_000_000 + seed) for seed in seeds]
    )
    evaluator = make_evaluator(
        condition, args.evaluation_size, args.target_angle_deg
    )
    rows = []

    def record(iteration):
        metrics = jax.device_get(evaluator(actor_params, evaluation_keys))
        for index, seed in enumerate(seeds):
            rows.append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "iteration": iteration,
                    "stage0_interactions": iteration * args.collection_size,
                    "target_angle_deg": args.target_angle_deg,
                    "discovery": float(metrics["discovery"][index]),
                    "precision": float(metrics["precision"][index]),
                    "success": float(metrics["success"][index]),
                    "stage0_norm_mean": float(metrics["stage0_norm_mean"][index]),
                    "stage1_norm_mean": float(metrics["stage1_norm_mean"][index]),
                    "stage0_angle_error_deg": float(
                        metrics["stage0_angle_error_deg"][index]
                    ),
                    "stage1_angle_error_deg": float(
                        metrics["stage1_angle_error_deg"][index]
                    ),
                }
            )

    record(0)
    update_count = 0
    for iteration in range(1, args.iterations + 1):
        split = jax.vmap(lambda key: jax.random.split(key, 2))(train_keys)
        train_keys, collection_keys = split[:, 0], split[:, 1]
        actions = sample_all(
            actor_params, collection_keys, condition, args.collection_size
        )
        discovery = shell_hit(
            actions[:, 0], APPROACH_RADIUS, APPROACH_RADIAL_TOLERANCE
        )
        precision = shell_hit(
            actions[:, 1], PRECISION_RADIUS, PRECISION_RADIAL_TOLERANCE
        )
        terminal = discovery * precision
        replay.add(
            np.asarray(jax.device_get(actions)),
            np.asarray(jax.device_get(discovery)),
            np.asarray(jax.device_get(terminal)),
        )

        if iteration >= args.learning_starts:
            for _ in range(args.updates_per_iteration):
                batch = replay.sample(
                    replay_rng,
                    args.update_batch_size,
                    args.discovery_replay_fraction,
                )
                split = jax.vmap(lambda key: jax.random.split(key, 3))(train_keys)
                train_keys = split[:, 0]
                target_keys = split[:, 1]
                actor_keys = split[:, 2]
                critic_params, target_params, critic_state, _ = critic_update(
                    critic_params,
                    target_params,
                    critic_state,
                    actor_params,
                    batch,
                    target_keys,
                )
                update_count += 1
                if (
                    iteration >= args.actor_starts
                    and update_count % args.policy_delay == 0
                ):
                    actor_params, actor_state, _ = actor_update(
                        actor_params,
                        actor_state,
                        critic_params,
                        actor_keys,
                        batch,
                    )
        if iteration % args.evaluation_interval == 0 or iteration == args.iterations:
            record(iteration)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, conditions, seeds):
    summaries = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition.name]
        per_seed = []
        for seed in seeds:
            curve = sorted(
                [row for row in selected if row["seed"] == seed],
                key=lambda row: row["stage0_interactions"],
            )
            steps = np.asarray([row["stage0_interactions"] for row in curve])
            success = np.asarray([row["success"] for row in curve])
            reached = steps[success >= 0.5]
            per_seed.append(
                {
                    "final_success": float(success[-1]),
                    "auc": float(np.trapezoid(success, steps) / steps[-1]),
                    "steps_to_50": float(reached[0]) if len(reached) else np.nan,
                }
            )
        item = {
            "condition": condition.name,
            "actor_parameter_count": parameter_count(condition),
            "num_seeds": len(seeds),
        }
        for metric in ("final_success", "auc", "steps_to_50"):
            values = np.asarray([row[metric] for row in per_seed])
            finite = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            item[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
            if metric == "steps_to_50":
                item["fraction_reaching_50"] = float(np.mean(np.isfinite(values)))
        summaries.append(item)
    return summaries


def plot(rows, summaries, conditions, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    colors = plt.get_cmap("tab10")
    for index, condition in enumerate(conditions):
        selected = [row for row in rows if row["condition"] == condition.name]
        steps = sorted({row["stage0_interactions"] for row in selected})
        means, stds = [], []
        for step in steps:
            values = np.asarray(
                [row["success"] for row in selected if row["stage0_interactions"] == step]
            )
            means.append(100 * values.mean())
            stds.append(100 * values.std(ddof=1))
        x = np.asarray(steps) / 1000
        means, stds = np.asarray(means), np.asarray(stds)
        axes[0].plot(x, means, label=condition.name, color=colors(index))
        axes[0].fill_between(x, np.maximum(0, means - stds), means + stds, color=colors(index), alpha=0.12)
    axes[0].set(xlabel="Stage-0 interactions (k)", ylabel="Success rate (%)")
    axes[0].legend(fontsize=8)
    x = np.arange(len(summaries))
    axes[1].bar(x, [100 * row["auc_mean"] for row in summaries], color=[colors(i) for i in range(len(summaries))])
    axes[1].set_ylabel("Normalized success AUC (%)")
    axes[1].set_xticks(x, [row["condition"] for row in summaries], rotation=25, ha="right")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle("Sparse shell off-policy twin-Q bridge")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=300)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--collection-size", type=int, default=512)
    parser.add_argument("--learning-starts", type=int, default=10)
    parser.add_argument("--updates-per-iteration", type=int, default=4)
    parser.add_argument("--update-batch-size", type=int, default=512)
    parser.add_argument("--actor-batch-size", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=65536)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-3)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-3)
    parser.add_argument("--critic-output-bias", type=float, default=-2.0)
    parser.add_argument("--actor-starts", type=int, default=100)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--actor-entropy-coefficient", type=float, default=0.002)
    parser.add_argument(
        "--actor-update",
        choices=("q_gradient", "awr"),
        default="awr",
    )
    parser.add_argument("--awr-temperature", type=float, default=0.10)
    parser.add_argument("--discovery-replay-fraction", type=float, default=0.5)
    parser.add_argument("--evaluation-size", type=int, default=20000)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--target-angle-deg", type=float, default=30.0)
    parser.add_argument("--critic-init-seed", type=int, default=20260926)
    parser.add_argument("--replay-seed", type=int, default=20260926)
    parser.add_argument(
        "--condition",
        action="append",
        choices=[condition.name for condition in OFFPOLICY_CONDITIONS],
        help="Run only this condition; repeat the flag to select several.",
    )
    parser.add_argument("--output-prefix", default="offpolicy_shell_pilot")
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    set_target_angle(args.target_angle_deg)
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    conditions = tuple(
        condition
        for condition in OFFPOLICY_CONDITIONS
        if args.condition is None or condition.name in args.condition
    )
    for condition in conditions:
        rows.extend(train_condition(condition, seeds, args))
        print(f"finished {condition.name}", flush=True)
    summaries = summarize(rows, conditions, seeds)
    write_csv(args.output_dir / f"{args.output_prefix}_per_seed.csv", rows)
    write_csv(args.output_dir / f"{args.output_prefix}_summary.csv", summaries)
    plot(rows, summaries, conditions, args.output_dir / f"{args.output_prefix}.png")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
