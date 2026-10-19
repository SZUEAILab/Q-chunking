"""Compare the local recipe port to unmodified, commit-pinned upstream code.

Run in a fresh process: JAX_PLATFORMS=cpu python -m
research.check_rlpd_reference --upstream-root /path/to/rlpd --output /path/report.json
Only package initialization is bypassed (it eagerly imports pixel/D4RL agents).
The actual upstream learner, distribution, network and data modules execute from
their original files. Modern Flax dict outputs are frozen at the API boundary.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.rlpd_reference import RLPDReferenceAgent, UPSTREAM_COMMIT, get_config


def load_upstream(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected upstream revision: {commit}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=root, text=True).strip():
        raise ValueError("Upstream tracked files must be unmodified.")
    paths = {}

    def package(name, directory):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module
        return module

    def load(name, relative):
        path = root / relative
        paths[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    package("rlpd", root / "rlpd")
    package("rlpd.agents", root / "rlpd/agents")
    package("rlpd.agents.sac", root / "rlpd/agents/sac")
    package("rlpd.data", root / "rlpd/data")
    distributions = package("rlpd.distributions", root / "rlpd/distributions")
    networks = package("rlpd.networks", root / "rlpd/networks")
    load("rlpd.types", "rlpd/types.py")
    load("rlpd.data.dataset", "rlpd/data/dataset.py")
    for filename, symbols in (
        ("mlp", ("MLP", "default_init")),
        ("ensemble", ("Ensemble", "subsample_ensemble")),
        ("state_action_value", ("StateActionValue",)),
        ("mlp_resnet", ("MLPResNetV2",)),
    ):
        module = load(f"rlpd.networks.{filename}", f"rlpd/networks/{filename}.py")
        for name in symbols:
            setattr(networks, name, getattr(module, name))
    load("rlpd.distributions.tanh_transformed", "rlpd/distributions/tanh_transformed.py")
    distributions.TanhNormal = load("rlpd.distributions.tanh_normal",
                                    "rlpd/distributions/tanh_normal.py").TanhNormal
    load("rlpd.agents.agent", "rlpd/agents/agent.py")
    load("rlpd.agents.sac.temperature", "rlpd/agents/sac/temperature.py")
    learner = load("rlpd.agents.sac.sac_learner", "rlpd/agents/sac/sac_learner.py").SACLearner
    return learner, paths


class ExampleSpace:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape

    def sample(self):
        return self.values.copy()


def assert_same(local, upstream):
    maximum = 0.0
    for name in ("actor", "critic", "target_critic", "temp"):
        left, right = getattr(local, name), getattr(upstream, name)
        for field in ("params", "opt_state", "step"):
            a, b = getattr(left, field), getattr(right, field)
            if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
                raise AssertionError(f"Tree mismatch: {name}/{field}")
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
                x, y = np.asarray(x), np.asarray(y)
                np.testing.assert_allclose(x, y, rtol=3e-6, atol=3e-7,
                                           err_msg=f"{name}/{field}")
                maximum = max(maximum, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
    np.testing.assert_array_equal(local.rng, upstream.rng)
    return maximum


def run_case(learner, *, utd, backup_entropy, subset, full_width=False,
             action_dim=5, agent_class=RLPDReferenceAgent, config_overrides=None):
    rng = np.random.default_rng(20261022)
    obs = np.zeros(7, np.float32)
    actions = np.zeros(action_dim, np.float32)
    hidden = (256, 256) if full_width else (16, 16)
    config = get_config().to_dict()
    config.update(horizon_length=1, actor_hidden_dims=hidden, value_hidden_dims=hidden,
                  backup_entropy=backup_entropy, num_min_qs=subset)
    config.update(config_overrides or {})
    local = agent_class.create(9100, obs, actions, config)
    upstream = learner.create(
        9100, ExampleSpace(obs), ExampleSpace(actions), hidden_dims=hidden,
        num_qs=10, num_min_qs=subset, backup_entropy=backup_entropy, critic_layer_norm=True,
        target_entropy=config["target_entropy"])
    # Flax 0.5 returned FrozenDict from init; current Flax returns dict.
    for name in ("actor", "critic", "target_critic", "temp"):
        state = getattr(upstream, name)
        params = flax.core.freeze(state.params)
        upstream = upstream.replace(**{name: state.replace(params=params, opt_state=state.tx.init(params))})
    errors = [assert_same(local, upstream)]
    for _ in range(3):
        size = 4 * utd
        batch = dict(
            observations=jnp.asarray(rng.normal(size=(size, 7)), dtype=jnp.float32),
            next_observations=jnp.asarray(rng.normal(size=(size, 7)), dtype=jnp.float32),
            actions=jnp.asarray(rng.uniform(-.8, .8, size=(size, action_dim)), dtype=jnp.float32),
            rewards=jnp.asarray(rng.integers(-3, 1, size), dtype=jnp.float32),
            masks=jnp.asarray(rng.integers(0, 2, size), dtype=jnp.float32))
        local, info = local.update(batch, utd_ratio=utd)
        upstream, reference_info = upstream.update(batch, utd_ratio=utd)
        errors.append(assert_same(local, upstream))
        for name in reference_info:
            np.testing.assert_allclose(info[name], reference_info[name], rtol=3e-6, atol=3e-7)
    return dict(utd=utd, backup_entropy=backup_entropy, num_min_qs=subset,
                action_dim=action_dim, local_agent=agent_class.__name__, target_entropy=config["target_entropy"],
                hidden_dims=list(hidden), update_calls=3, critic_updates=int(local.critic.step),
                actor_updates=int(local.actor.step), temperature_updates=int(local.temp.step),
                max_absolute_state_error=max(errors), status="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    learner, files = load_upstream(args.upstream_root)
    cases = [run_case(learner, utd=utd, backup_entropy=backup, subset=subset)
             for utd, backup, subset in ((1, True, 2), (20, True, 2), (1, False, 1),
                                         (20, False, 2), (1, True, None))]
    cases.append(run_case(learner, utd=1, backup_entropy=True, subset=2, full_width=True))
    report = dict(status="passed", upstream_commit=UPSTREAM_COMMIT, upstream_files=files,
                  local_learner_sha256=hashlib.sha256(Path(sys.modules[RLPDReferenceAgent.__module__].__file__).read_bytes()).hexdigest(),
                  oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  compatibility_adaptations=["bypass eager package initialization", "freeze current Flax init dictionaries"],
                  scope="learner updates only; local OGBench driver and external interaction/evaluation RNG differ",
                  jax_version=jax.__version__, flax_version=flax.__version__, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "upstream_commit", "cases")}))


if __name__ == "__main__":
    main()
