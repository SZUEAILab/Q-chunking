"""Post-training tests of critic invariance to redundant adapter coordinates.

This tool never changes a policy or training data. It is not a reward-blind
calibrator: it reads learned Q values and must run in the analysis stage.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_training_source(source_root, manifest):
    """Check actual training files rather than trusting the current Git HEAD."""
    for item in manifest["python_sources"]:
        if file_digest(source_root / item["path"]) != item["sha256"]:
            raise ValueError(f"Training source mismatch: {item['path']}")
    return manifest["source_commit"]


def measure_aliases(agent, observations, *, seed=20261020, scale=0.5, actions=None):
    """Compare Q and dQ/dlog||u|| for behaviorally equivalent policy actions.

    The gradient is for Q alone. Policy-coordinate entropy can still change
    along this redundant direction even when the critic is invariant.
    """
    import jax
    import jax.numpy as jnp
    from agents.acrlpd import _adapter_policy_to_env

    if agent.config.get("adapter_representation") != "cartesian":
        raise ValueError("Radial alias test requires the D+1 Cartesian adapter")
    if not 0 < scale < 1:
        raise ValueError("Alias scale must lie strictly between zero and one")
    observations = jnp.asarray(observations)
    if observations.ndim != 2 or not len(observations):
        raise ValueError("A nonempty two-dimensional observation panel is required")
    policy_dim = int(agent.config["action_dim"])
    repetitions = int(agent.config["horizon_length"]) if agent.config["action_chunking"] else 1
    expected_shape = (len(observations), policy_dim * repetitions)
    if actions is None:
        actions = agent.sample_actions(observations, rng=jax.random.PRNGKey(seed))
    actions = jnp.asarray(actions)
    if actions.shape != expected_shape:
        raise ValueError(f"Expected policy action shape {expected_shape}, got {actions.shape}")
    chunks = actions.reshape(len(observations), repetitions, policy_dim)
    aliases = chunks.at[..., :3].multiply(scale).reshape(expected_shape)
    env_actions = _adapter_policy_to_env(actions, agent.config)
    env_aliases = _adapter_policy_to_env(aliases, agent.config)
    env_error = np.max(np.abs(np.asarray(env_actions - env_aliases)), axis=-1)
    direction_norm = np.linalg.norm(np.asarray(chunks[..., :3]), axis=-1)
    # Below the normalization floor, positive radial scaling can change the
    # executed action. Such samples are reported and excluded from alias claims.
    valid = (env_error <= 1e-6) & np.all(direction_norm * scale > 1e-6, axis=-1)

    def q_value(raw):
        return agent.network.select("critic")(
            observations, agent._critic_action(raw)
        ).mean(axis=0)

    q = np.asarray(q_value(actions))
    alias_q = np.asarray(q_value(aliases))
    gradient = np.asarray(jax.grad(lambda raw: q_value(raw).sum())(actions))
    features = np.asarray(agent._critic_action(actions))
    alias_features = np.asarray(agent._critic_action(aliases))
    for value in (q, alias_q, gradient, features, alias_features):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite critic prediction, feature, or value gradient")
    raw_direction = np.asarray(chunks[..., :3])
    gradient_direction = gradient.reshape(len(observations), repetitions, policy_dim)[..., :3]
    # Scaling u to exp(t)u gives derivative u dot grad_u Q at t=0.
    radial_derivative = np.sum(raw_direction * gradient_direction, axis=-1)
    gradient_norm = np.linalg.norm(gradient_direction, axis=-1)
    radial_fraction = np.abs(radial_derivative) / np.maximum(direction_norm * gradient_norm, 1e-12)
    record = dict(samples=len(observations), valid_alias_pairs=int(valid.sum()),
                  excluded_pairs=int((~valid).sum()), alias_scale=scale,
                  max_executed_action_difference=float(env_error.max()),
                  critic_space=agent.config["adapter_critic_space"],
                  value_gradient_only=True, diagnostic_seed=seed,
                  raw_direction_norm_mean=float(direction_norm.mean()),
                  raw_direction_norm_std=float(direction_norm.std()),
                  finite_predictions_and_gradients=True)
    metrics = ("q_alias_abs_mean", "q_alias_abs_p95", "q_alias_relative_mean",
               "critic_feature_alias_abs_max", "radial_q_derivative_rms",
               "radial_q_gradient_fraction_mean")
    if not valid.any():
        record.update({name: None for name in metrics})
        return record
    differences = np.abs(q[valid] - alias_q[valid])
    record.update(
        q_alias_abs_mean=float(differences.mean()),
        q_alias_abs_p95=float(np.quantile(differences, 0.95)),
        q_alias_relative_mean=float(differences.mean() / max(np.abs(q[valid]).mean(), 1e-8)),
        critic_feature_alias_abs_max=float(np.abs(features[valid] - alias_features[valid]).max()),
        radial_q_derivative_rms=float(np.sqrt(np.mean(radial_derivative[valid] ** 2))),
        radial_q_gradient_fraction_mean=float(radial_fraction[valid].mean()),
    )
    return record


def observation_panel(dataset_path, *, count, seed, expected_digest):
    """Choose fixed validation-prior states without reading reward or success."""
    if file_digest(dataset_path) != expected_digest:
        raise ValueError(f"Validation-prior data mismatch: {dataset_path}")
    if count <= 0:
        raise ValueError("Panel size must be positive")
    with np.load(dataset_path) as data:
        observations = data["observations"]
        indices = np.flatnonzero(np.asarray(data["terminals"]) == 0)
        if len(indices) < count:
            raise ValueError("Insufficient nonterminal validation observations")
        chosen = np.random.default_rng(seed).choice(indices, size=count, replace=False)
        panel = np.asarray(observations[chosen], dtype=np.float32)
    if not np.isfinite(panel).all():
        raise ValueError("Nonfinite state panel")
    identity = hashlib.sha256(panel.tobytes() + chosen.astype(np.int64).tobytes()).hexdigest()
    return panel, dict(dataset_path=str(dataset_path), dataset_sha256=expected_digest,
                       panel_seed=seed, panel_size=count, indices=chosen.tolist(),
                       panel_array_sha256=identity, source="validation-prior nonterminal observations")


def reference_policy_actions(config, *, count, seed=20261021):
    """A common probe bank independent of learned actor or critic parameters."""
    import jax
    from direction_speed.adapter import DirectionSpeedAdapter
    from direction_speed.distributions.mixed_squash import MixedSquashBijector

    if count <= 0:
        raise ValueError("Reference probe count must be positive")
    adapter = DirectionSpeedAdapter(
        representation=config["adapter_representation"], speed_squash=config["adapter_speed_squash"],
        speed_interpretation=config["adapter_speed_interpretation"], speed_bound=config["ds_speed_bound"],
    )
    dimension = int(config["action_dim"])
    repetitions = int(config["horizon_length"]) if config["action_chunking"] else 1
    raw = jax.random.normal(jax.random.PRNGKey(seed), (count, dimension * repetitions))
    return MixedSquashBijector(adapter.squash_types(dimension, repetitions)).forward(raw)


def diagnose_run(directory, observations, panel_identity):
    """Restore a trusted owned checkpoint using its effective runtime config."""
    import jax
    from agents.acrlpd import ACRLPDAgent
    from direction_speed.adapter import DirectionSpeedAdapter
    from utils.flax_utils import restore_agent_with_file

    flags_path = directory / "flags.json"
    config_path = directory / "effective_agent_config.json"
    flags = json.loads(flags_path.read_text())
    config = json.loads(config_path.read_text())
    if config.get("adapter_representation", "none") != "cartesian":
        return dict(status="not_applicable", reason="Not a Cartesian D+1 adapter", run_dir=str(directory))
    checkpoint = directory / f"params_{flags['online_steps']}.pkl"
    adapter = DirectionSpeedAdapter(
        representation=config["adapter_representation"], speed_squash=config["adapter_speed_squash"],
        speed_interpretation=config["adapter_speed_interpretation"], speed_bound=config["ds_speed_bound"],
    )
    example_env_action = np.zeros(int(config["env_action_dim"]), dtype=np.float32)
    example = example_env_action if config["adapter_critic_space"] == "env" else adapter.decompose(example_env_action)
    agent = ACRLPDAgent.create(flags["seed"], observations[0], example, dict(config))
    initial_step = int(agent.network.step)
    agent = restore_agent_with_file(agent, str(checkpoint))
    before_rng = np.asarray(agent.rng).copy()
    result = measure_aliases(agent, observations)
    probe = reference_policy_actions(config, count=len(observations))
    reference_result = measure_aliases(agent, observations, actions=probe)
    np.testing.assert_array_equal(agent.rng, before_rng)
    # Force asynchronous device work to finish before recording success.
    jax.block_until_ready(agent.rng)
    return dict(status="complete", run_dir=str(directory), training_seed=flags["seed"],
                budget=flags["online_steps"], env_name=flags["env_name"], run_group=flags["run_group"],
                optimizer_step_counter=int(agent.network.step),
                optimizer_updates=int(agent.network.step) - initial_step,
                checkpoint_sha256=file_digest(checkpoint), config_sha256=file_digest(config_path),
                flags_sha256=file_digest(flags_path), panel=panel_identity, metrics=result,
                shared_probe_metrics=reference_result, shared_probe_seed=20261021,
                shared_probe_definition="standard normal followed by declared policy squash; independent of learned parameters",
                interpretation="post-training critic diagnostic, not a task-performance comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="The .ogbench mount root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=128)
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    manifest = json.loads(args.release_manifest.read_text())
    source_commit = verify_training_source(args.source_root, manifest)
    sys.path.insert(0, str(args.source_root.resolve()))
    import agents.acrlpd as implementation
    if not Path(implementation.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise ValueError("Imported learner is outside the verified training source")
    data_files = {row["relative_path"]: row for row in json.loads(args.data_manifest.read_text())}
    panels, records = {}, []
    for directory in args.run_dir:
        try:
            flags = json.loads((directory / "flags.json").read_text())
            family = flags["env_name"].split("-singletask", 1)[0]
            relative = f"data/{family}-v0-val.npz"
            if relative not in data_files:
                raise ValueError(f"No frozen validation-data identity for {family}")
            if relative not in panels:
                panels[relative] = observation_panel(args.data_root / relative, count=args.panel_size,
                                                      seed=20261020, expected_digest=data_files[relative]["sha256"])
            observations, identity = panels[relative]
            records.append(diagnose_run(directory, observations, identity))
        except Exception as error:
            records.append(dict(status="needs_review", run_dir=str(directory), error=repr(error)))
    report = dict(training_source_commit=source_commit, diagnostic_tool_sha256=file_digest(Path(__file__)),
                  gradient_scope="ensemble-mean Q; excludes policy entropy and all other actor terms",
                  records=records, status="needs_review" if any(r["status"] == "needs_review" for r in records) else "complete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], runs=len(records), output=str(args.output))))
    raise SystemExit(0 if report["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
