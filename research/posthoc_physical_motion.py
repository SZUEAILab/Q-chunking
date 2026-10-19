"""Frozen-policy single-step motion probes from matched MuJoCo integration states.

This measures controller/physics responses, not learning, return or success.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

from research.posthoc_initial_policy import digest, verify_source


def array_digest(array):
    array = np.asarray(array)
    return hashlib.sha256(str((array.shape, str(array.dtype))).encode() + array.tobytes()).hexdigest()


def motion_measurements(before, target, after, action, action_low, action_high):
    before, target, after = [np.asarray(x, dtype=float) for x in (before, target, after)]
    # Preserve the dtype used by the environment's unnormalize_action; casting
    # before (action + 1) can turn float32 rounding into apparent clipping.
    action = np.asarray(action)
    low, high = np.asarray(action_low), np.asarray(action_high)
    if (any(x.shape != (3,) for x in (before, target, after)) or action.shape != (5,)
            or low.shape != (5,) or high.shape != (5,) or np.any(high <= low)
            or not all(np.isfinite(x).all() for x in (before, target, after, action, low, high))
            or np.max(np.abs(action)) > 1 + 1e-6):
        raise ValueError("Expected finite positions, bounds and a legal five-dimensional action")
    request = (.5 * (action + 1) * (high - low) + low)[:3]
    target_delta, delta = target - before, after - before
    return dict(action_xyz_norm=float(np.linalg.norm(action[:3])), requested_displacement_m=request.tolist(),
                requested_norm_m=float(np.linalg.norm(request)), target_displacement_m=target_delta.tolist(),
                target_norm_m=float(np.linalg.norm(target_delta)), actual_displacement_m=delta.tolist(),
                actual_norm_m=float(np.linalg.norm(delta)),
                workspace_clipped=bool(np.max(np.abs(request - target_delta)) > 1e-10))


def assert_same_start(observation, state, expected_observation, expected_state):
    if not all(np.isfinite(x).all() for x in (observation, state, expected_observation, expected_state)):
        raise ValueError("Nonfinite starting observation or integration state")
    np.testing.assert_array_equal(observation, expected_observation)
    np.testing.assert_array_equal(state, expected_state)


def summarize(records, seeds, scales, reset_seeds, draws):
    expected = {(s, k, r, d, a) for s in seeds for k in scales for r in reset_seeds for d in range(draws)
                for a in ("native", "posthoc", "same_head_cartesian", "zero_xyz")}
    indexed = {}
    for row in records:
        key = (row["seed"], row["noise_scale"], row["reset_seed"], row["draw"], row["arm"])
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate motion-probe identity")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Incomplete physical probe grid; no partial aggregates")
    output = []
    for seed in seeds:
        for scale in scales:
            groups = {}
            for arm in ("native", "posthoc", "same_head_cartesian", "zero_xyz"):
                rows = [indexed[seed, scale, r, d, arm] for r in reset_seeds for d in range(draws)]
                groups[arm] = {metric: float(np.mean([row[metric] for row in rows])) for metric in
                    ("action_xyz_norm", "requested_norm_m", "target_norm_m", "actual_norm_m", "workspace_clipped")}
            corrected = {}
            for arm in ("posthoc", "same_head_cartesian"):
                values = []
                for r in reset_seeds:
                    for d in range(draws):
                        a, zero = indexed[seed, scale, r, d, arm], indexed[seed, scale, r, d, "zero_xyz"]
                        if a["start_state_sha256"] != zero["start_state_sha256"]:
                            raise ValueError("Paired physical states differ")
                        values.append(float(np.linalg.norm(np.asarray(a["actual_displacement_m"]) - zero["actual_displacement_m"])))
                corrected[arm] = float(np.mean(values))
            output.append(dict(seed=seed, noise_scale=scale, arms=groups, mean_zero_xyz_corrected_norm_m=corrected,
                paired_corrected_norm_difference_m=corrected["posthoc"] - corrected["same_head_cartesian"]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".jsonl").exists():
        raise FileExistsError("Preserve previous physical probe attempt")
    plan = json.loads(args.plan.read_text())
    source = Path(plan["source_root"])
    manifest = json.loads(Path(plan["release_manifest"]).read_text())
    captured = json.loads(Path(plan["inputs"]).read_text())
    reference = json.loads(Path(plan["initial_policy_report"]).read_text())
    contract_paths = [args.plan, Path(plan["release_manifest"]), Path(plan["inputs"]), Path(plan["initial_policy_report"])]
    input_hashes = {str(p): digest(p) for p in contract_paths}
    if (manifest["source_commit"] != plan["source_commit"] or reference["training_source_commit"] != plan["source_commit"]
            or reference["inputs_sha256"] != digest(Path(plan["inputs"]))
            or digest(Path(sys.modules["research.posthoc_initial_policy"].__file__)) != plan["helper_sha256"]):
        raise ValueError("Wrong frozen actor provenance")
    verify_source(source, manifest)
    sys.path.insert(0, str(source.resolve()))
    import jax
    import jax.numpy as jnp
    import ogbench
    import mujoco
    import importlib.metadata
    from agents.acrlpd import ACRLPDAgent
    from direction_speed.adapter import DirectionSpeedAdapter
    if jax.default_backend() != "cpu":
        raise ValueError("Physical diagnostics require CPU-only JAX")
    if (mujoco.__version__ != plan["mujoco_version"]
            or importlib.metadata.version("ogbench") != plan["ogbench_version"]):
        raise ValueError("Simulator/package version differs from the registered probe")
    for name in ("agents.acrlpd", "direction_speed.adapter"):
        Path(sys.modules[name].__file__).resolve().relative_to(source.resolve())
    package = Path(ogbench.__file__).resolve().parent
    for path, expected in plan["environment_python_sources"].items():
        if digest(package / path) != expected:
            raise ValueError("OGBench source changed: " + path)
    env = ogbench.make_env_and_datasets(plan["env_name"], env_only=True)
    base = env.unwrapped
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    counts = dict(explicit_resets=0, explicit_probe_steps=0, repeated_action_checks=0)

    def state():
        value = np.empty(mujoco.mj_stateSize(base._model, kind))
        mujoco.mj_getState(base._model, base._data, value, kind)
        return value

    def reset(seed):
        random.seed(seed); np.random.seed(seed)
        observation = np.asarray(env.reset(seed=seed)[0])
        counts["explicit_resets"] += 1
        return observation

    def position():
        return base._data.site_xpos[base._pinch_site_id].copy()

    states, observations, canonical, models = [], [], [], []
    records, actor_records = [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for seed in plan["reset_seeds"]:
            observation = reset(seed)
            saved = state()
            model = np.empty(mujoco.mj_sizeModel(base._model), dtype="uint8")
            mujoco.mj_saveModel(base._model, buffer=model)
            reset(seed); mujoco.mj_setState(base._model, base._data, saved, kind); mujoco.mj_forward(base._model, base._data)
            np.testing.assert_array_equal(base.compute_observation(), observation)
            states.append(saved); observations.append(observation); canonical.append(state()); models.append(model)
        panel_path = args.output.with_suffix(".states.npz")
        if panel_path.exists():
            raise FileExistsError("Preserve existing physics-state panel")
        np.savez_compressed(panel_path, reference_states=np.asarray(states), canonical_states=np.asarray(canonical),
                            observations=np.asarray(observations), models=np.asarray(models))
        adapter = DirectionSpeedAdapter(speed_squash="tanh", speed_interpretation="log", speed_bound="sphere")
        actor_observations = np.asarray(observations, dtype="float32")
        noise = np.random.default_rng(plan["noise_seed"]).standard_normal((len(states), plan["draws"], 6)).astype("float32")
        actions = {}
        old = {(r["seed"], r["method"]): r for r in reference["records"]}
        for seed in plan["seeds"]:
            for method, group in [("native", "aa_triple2_native_20261019"), ("posthoc", "aa_triple2_legacy_latent_20261019")]:
                config = copy.deepcopy(captured[group]["effective_config"])
                if hashlib.sha256(json.dumps(config, indent=2, sort_keys=True).encode()).hexdigest() != captured[group]["effective_config_sha256"]:
                    raise ValueError("Captured configuration changed")
                example = np.zeros(5, dtype="float32")
                if method == "posthoc": example = adapter.decompose(example)
                agent = ACRLPDAgent.create(seed, actor_observations[0], example, config)
                before_rng = np.asarray(agent.rng).copy()
                actor_sha = hashlib.sha256(b"".join(np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(agent.network.params["modules_actor"]))).hexdigest()
                if actor_sha != old[seed, method]["actor_parameter_sha256"]:
                    raise ValueError("Not the previously measured frozen actor")
                dist = agent.network.select("actor")(jnp.asarray(actor_observations))
                loc, std = np.asarray(dist.raw_distribution.mean()), np.asarray(dist.raw_distribution.stddev())
                for scale in plan["noise_scales"]:
                    eps = noise if method == "posthoc" else noise[:, :, [0, 1, 2, 4, 5]]
                    raw = loc[:, None] + scale * std[:, None] * eps
                    policy = np.asarray(dist.raw_to_policy(jnp.asarray(raw.reshape(-1, raw.shape[-1])))).reshape(raw.shape)
                    if method == "native": actions[seed, scale, method] = policy
                    else:
                        actions[seed, scale, method] = adapter.compose(policy.reshape(-1, 6)).reshape(*policy.shape[:2], 5)
                        actions[seed, scale, "same_head_cartesian"] = policy[:, :, [0, 1, 2, 4, 5]]
                        zero = actions[seed, scale, method].copy(); zero[:, :, :3] = 0
                        actions[seed, scale, "zero_xyz"] = zero
                        np.testing.assert_array_equal(actions[seed, scale, method][:, :, 3:], actions[seed, scale, "same_head_cartesian"][:, :, 3:])
                np.testing.assert_array_equal(agent.rng, before_rng)
                if int(agent.network.step) != 1: raise ValueError("Analysis updated the actor")
                actor_records.append(dict(seed=seed, method=method, actor_parameter_sha256=actor_sha))

        def run_action(index, action):
            reset(plan["reset_seeds"][index])
            mujoco.mj_setState(base._model, base._data, states[index], kind)
            mujoco.mj_forward(base._model, base._data)
            assert_same_start(base.compute_observation(), state(), observations[index], canonical[index])
            before = position()
            result = env.step(action)  # Only observation is accessed; reward/success are ignored.
            counts["explicit_probe_steps"] += 1
            if not np.isfinite(np.asarray(result[0])).all(): raise ValueError("Invalid resulting observation")
            values = motion_measurements(before, base._target_effector_pose.translation(), position(), action,
                                         base.action_low, base.action_high)
            return values, state()

        with args.output.with_suffix(".jsonl").open("x") as stream:
            for seed in plan["seeds"]:
                for scale in plan["noise_scales"]:
                    for i, reset_seed in enumerate(plan["reset_seeds"]):
                        for draw in range(plan["draws"]):
                            reference_post = None
                            for arm in ("native", "posthoc", "same_head_cartesian", "zero_xyz"):
                                values, after = run_action(i, actions[seed, scale, arm][i, draw])
                                row = dict(seed=seed, noise_scale=scale, reset_seed=reset_seed, draw=draw, arm=arm,
                                           start_state_sha256=array_digest(canonical[i]), after_state_sha256=array_digest(after), **values)
                                records.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n")
                                if arm == "posthoc": reference_post = after
                            _, repeated = run_action(i, actions[seed, scale, "posthoc"][i, draw])
                            np.testing.assert_array_equal(repeated, reference_post)
                            counts["repeated_action_checks"] += 1
                    print(json.dumps(dict(seed=seed, noise_scale=scale, finished=True)), flush=True)
        groups = summarize(records, plan["seeds"], plan["noise_scales"], plan["reset_seeds"], plan["draws"])
        if (len(records) != plan["expected_records"] or counts["explicit_probe_steps"] != plan["expected_probe_steps"]
                or counts["repeated_action_checks"] != plan["expected_repeated_actions"] or len(actor_records) != 8):
            raise ValueError("Physical probe did not execute the complete registered grid")
        verify_source(source, manifest)
        for path, expected in plan["environment_python_sources"].items():
            if digest(package / path) != expected: raise ValueError("OGBench source changed during probe")
        if any(digest(Path(p)) != value for p, value in input_hashes.items()): raise ValueError("Probe inputs changed")
        report = dict(status="complete", source_commit=plan["source_commit"], tool_sha256=digest(Path(__file__)),
            input_hashes=input_hashes, actors=actor_records, groups=groups, counts=counts, control_timestep_s=float(base.control_timestep()),
            state_panel_sha256=digest(panel_path), raw_records_sha256=digest(args.output.with_suffix(".jsonl")),
            environment_python_sources=plan["environment_python_sources"], mujoco=mujoco.__version__, backend="cpu", optimizer_updates=0,
            scope="Frozen initialization, one 50ms physics step per action from matched full integration states. No learning, task return/success analysis, rollout coverage or trained-policy claim.",
            limitations=["Fresh task2 reset states, not the validation-prior panel or learned-policy visitation distribution.",
                         "Noise multipliers hold learned Gaussian parameters fixed; training can adapt them.",
                         "Zero-XYZ comparison retains posthoc yaw/gripper; native raw displacements have different scalar heads.",
                         "Model files and integration states are preserved; exact matching is within this installed simulator/runtime.",
                         "Reset calls internally perform goal stabilization steps, outside explicit probe-step accounting."])
        with args.output.open("x") as handle: handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(status="complete", counts=counts)))
    except Exception as error:
        args.output.with_suffix(".incident.json").write_text(json.dumps(dict(error=repr(error), counts=counts, complete_records=len(records)), indent=2) + "\n")
        raise
    finally:
        env.close()


if __name__ == "__main__": main()
