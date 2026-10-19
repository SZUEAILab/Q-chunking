"""Small-action probability bounds for the original posthoc execution map.

Bounds refer to independent continuous Gaussian heads and real arithmetic.
Finite float32 draws are a separate descriptive check, away from exp(-1).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from research.posthoc_initial_policy import digest, verify_source


def gaussian_box_log_bounds(location, scale, half_width):
    """Bound a diagonal Gaussian box mass using minimum/maximum densities."""
    loc, std = np.asarray(location, dtype=float), np.asarray(scale, dtype=float)
    if (loc.shape != std.shape or loc.ndim != 2 or loc.shape[1] != 3
            or not np.isfinite(loc).all() or not np.isfinite(std).all() or np.any(std <= 0)
            or not np.isfinite(half_width) or half_width <= 0):
        raise ValueError("Expected finite state-by-three independent Gaussian parameters and positive width")
    common = np.log(2 * half_width) - np.log(std) - .5 * np.log(2 * np.pi)
    near = np.maximum(np.abs(loc) - half_width, 0) / std
    far = (np.abs(loc) + half_width) / std
    lower = np.sum(common - .5 * far**2, axis=-1)
    upper = np.sum(np.minimum(0., common - .5 * near**2), axis=-1)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Density bounds exceed numeric range")
    return lower, upper


def logmeanexp(values):
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).mean()))


def small_action_bounds(location, scale, threshold, eps=1e-6):
    """Bound posthoc small-action mass and a same-head Cartesian comparator.

    If ||u|| >= eps, clipping leaves the executed norm >= min(exp(v),1)
    >= exp(-1). For r <= exp(-1), a norm < r therefore requires
    ||u|| < eps*r*e. Each raw direction coordinate then lies inside
    +/- atanh(eps*r*e). Conversely, a Cartesian tanh action inside the
    coordinate box +/- r/sqrt(3) has norm < r. Speed-head parameters are
    not needed for this conservative posthoc upper bound.
    """
    if (not np.isfinite(threshold) or not 0 < threshold <= np.exp(-1)
            or not np.isfinite(eps) or eps <= 0 or eps * threshold * np.e >= 1):
        raise ValueError("Threshold must be in (0, exp(-1)] with a valid normalization floor")
    _, upper = gaussian_box_log_bounds(location, scale, np.arctanh(eps * threshold * np.e))
    lower, _ = gaussian_box_log_bounds(location, scale, np.arctanh(threshold / np.sqrt(3)))
    log_upper, log_lower = logmeanexp(upper), logmeanexp(lower)
    return dict(threshold=threshold, log10_upper_posthoc_probability_per_state=(upper / np.log(10)).tolist(),
                log10_lower_same_head_cartesian_probability_per_state=(lower / np.log(10)).tolist(),
                log10_upper_posthoc_probability_mean=log_upper / np.log(10),
                log10_lower_same_head_cartesian_probability_mean=log_lower / np.log(10),
                log10_upper_posthoc_to_cartesian_probability_ratio=(log_upper - log_lower) / np.log(10))


def event_summary(actions, thresholds):
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 3 or actions.shape[-1] != 5 or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1 + 1e-6:
        raise ValueError("Expected finite states-by-draws-by-five executed actions")
    radius = np.linalg.norm(actions[..., :3], axis=-1)
    return [dict(threshold=r, count=int((radius < r).sum()), draws=radius.size,
                 fraction=float((radius < r).mean()), per_state_fraction=(radius < r).mean(axis=1).tolist())
            for r in thresholds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing small-action analysis")
    plan = json.loads(args.plan.read_text())
    source = Path(plan["source_root"])
    inputs = Path(plan["inputs"])
    manifest_path = Path(plan["release_manifest"])
    reference_path = Path(plan["initial_policy_report"])
    paths = [args.plan, inputs, manifest_path, reference_path, *[Path(r["path"]) for r in plan["datasets"].values()]]
    hashes = {str(p): digest(p) for p in paths}
    manifest = json.loads(manifest_path.read_text())
    helper_path = Path(sys.modules["research.posthoc_initial_policy"].__file__)
    if digest(helper_path) != plan["helper_sha256"]:
        raise ValueError("Initialization helper differs from the registered analysis")
    if manifest["source_commit"] != plan["source_commit"]:
        raise ValueError("Wrong frozen training source")
    verify_source(source, manifest)
    captured = json.loads(inputs.read_text())
    reference = json.loads(reference_path.read_text())
    if (reference["training_source_commit"] != plan["source_commit"] or reference["inputs_sha256"] != digest(inputs)
            or reference["tool_sha256"] != plan["helper_sha256"]):
        raise ValueError("Initialization reference/input identity differs")
    priors = []
    for name, item in plan["datasets"].items():
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("Prior data changed")
        with np.load(item["path"]) as data:
            radii = np.linalg.norm(np.asarray(data["actions"][:, :3], dtype=float), axis=-1)
            priors.append(dict(dataset=name, actions=len(radii), thresholds=[dict(threshold=r,
                count=int((radii < r).sum()), fraction=float((radii < r).mean())) for r in plan["thresholds"]]))
            if name == "triple":
                chosen = np.random.default_rng(plan["panel_seed"]).choice(np.flatnonzero(data["terminals"] == 0),
                                                                         size=plan["states"], replace=False)
                observations = np.asarray(data["observations"][chosen], dtype="float32")
    panel_sha = hashlib.sha256(observations.tobytes() + chosen.astype("int64").tobytes()).hexdigest()
    noise = np.random.default_rng(plan["noise_seed"]).standard_normal((plan["states"], plan["draws"], 6)).astype("float32")
    if (panel_sha != reference["panel"]["sha256"]
            or hashlib.sha256(noise.tobytes()).hexdigest() != reference["noise"]["sha256"]):
        raise ValueError("Panel/noise differ from the previous initialization audit")
    sys.path.insert(0, str(source.resolve()))
    import jax
    import jax.numpy as jnp
    from agents.acrlpd import ACRLPDAgent
    from direction_speed.adapter import DirectionSpeedAdapter
    if jax.default_backend() != "cpu":
        raise ValueError("Run with JAX_PLATFORMS=cpu")
    for name in ("agents.acrlpd", "direction_speed.adapter", "utils.flax_utils"):
        Path(sys.modules[name].__file__).resolve().relative_to(source.resolve())
    adapter = DirectionSpeedAdapter(speed_squash="tanh", speed_interpretation="log", speed_bound="sphere")
    if adapter.eps != plan["eps"]:
        raise ValueError("Normalization floor differs from theorem inputs")
    reference_rows = {(r["seed"], r["method"]): r for r in reference["records"]}
    if set(reference_rows) != {(s, m) for s in plan["seeds"] for m in ("native", "posthoc")}:
        raise ValueError("Expected all previously measured initialization identities")
    rows = []
    for seed in plan["seeds"]:
        for method, group in [("native", "aa_triple2_native_20261019"), ("posthoc", "aa_triple2_legacy_latent_20261019")]:
            config = copy.deepcopy(captured[group]["effective_config"])
            if hashlib.sha256(json.dumps(config, indent=2, sort_keys=True).encode()).hexdigest() != captured[group]["effective_config_sha256"]:
                raise ValueError("Captured effective configuration changed")
            example = np.zeros(5, dtype="float32")
            if method == "posthoc":
                example = adapter.decompose(example)
            agent = ACRLPDAgent.create(seed, observations[0], example, config)
            before = np.asarray(agent.rng).copy()
            actor_sha = hashlib.sha256(b"".join(np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(agent.network.params["modules_actor"]))).hexdigest()
            if actor_sha != reference_rows[seed, method]["actor_parameter_sha256"] or int(agent.network.step) != 1:
                raise ValueError("Actor differs from the previously measured fresh initialization")
            dist = agent.network.select("actor")(jnp.asarray(observations))
            loc, std = np.asarray(dist.raw_distribution.mean()), np.asarray(dist.raw_distribution.stddev())
            covariance = np.asarray(dist.raw_distribution.covariance())
            diagonal = np.eye(covariance.shape[-1]) * covariance
            if np.max(np.abs(covariance - diagonal)) != 0:
                raise ValueError("Bounds require conditionally independent direction heads")
            row = dict(seed=seed, method=method, actor_parameter_sha256=actor_sha,
                       raw_direction_location_per_state=loc[:, :3].tolist(),
                       raw_direction_scale_per_state=std[:, :3].tolist(), cases=[])
            for scale in plan["noise_scales"]:
                if method == "native":
                    raw = loc[:, None] + scale * std[:, None] * noise[:, :, [0, 1, 2, 4, 5]]
                    action = np.asarray(dist.raw_to_policy(jnp.asarray(raw.reshape(-1, 5)))).reshape(plan["states"], plan["draws"], 5)
                    case = dict(noise_scale=scale, native_events=event_summary(action, plan["thresholds"]))
                else:
                    raw = loc[:, None] + scale * std[:, None] * noise
                    policy = np.asarray(dist.raw_to_policy(jnp.asarray(raw.reshape(-1, 6)))).reshape(raw.shape)
                    action = adapter.compose(policy.reshape(-1, 6)).reshape(plan["states"], plan["draws"], 5)
                    cartesian = policy[:, :, [0, 1, 2, 4, 5]]
                    case = dict(noise_scale=scale, posthoc_events=event_summary(action, plan["thresholds"]),
                        same_head_cartesian_events=event_summary(cartesian, plan["thresholds"]),
                        bounds=[small_action_bounds(loc[:, :3], scale * std[:, :3], r, adapter.eps) for r in plan["thresholds"]])
                    if any(e["count"] for e in case["posthoc_events"]):
                        case["unexpected_small_draws"] = "Inspect; zero sampled events were not assumed by the bounds."
                row["cases"].append(case)
            np.testing.assert_array_equal(agent.rng, before)
            if int(agent.network.step) != 1:
                raise ValueError("Analysis updated the model")
            rows.append(row)
            print(json.dumps(dict(seed=seed, method=method, finished=True)), flush=True)
    verify_source(source, manifest)
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise ValueError("Analysis contract changed during execution")
    result = dict(status="complete", source_commit=plan["source_commit"], tool_sha256=digest(Path(__file__)),
        input_hashes=hashes, panel_sha256=panel_sha, records=rows, prior_summaries=priors, backend="cpu", optimizer_updates=0,
        jax_version=jax.__version__, numpy_version=np.__version__, helper_sha256=digest(helper_path),
        scope="Continuous-Gaussian ideal-map probability bounds plus float32 frozen CPU initialization draws; same eight actors/panel/noise as previous audit. No training, environment interaction or success outcome access.",
        limitations=["Analytical bounds use real arithmetic and continuous diagonal Gaussian heads, not a certified finite-RNG/float32 probability law.",
                     "Thresholds 0.1/0.2/0.3 are separated from exp(-1); zero sampled events do not imply zero probability.",
                     "Only Cube Triple initialization was reconstructed; two validation-prior action distributions are descriptive data checks.",
                     "Same-head Cartesian execution is not a retrained baseline; no causal task-benefit or trained-policy claim."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="complete", models=len(rows))))


if __name__ == "__main__":
    main()
