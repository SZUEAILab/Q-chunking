"""Compare FQL reference updates with a clean, pinned official checkout.

Use a fresh CPU process per --case-index; the oracle imports official agent,
network, encoder and TrainState modules from their unmodified source files.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.fql_reference import FQLReferenceAgent, UPSTREAM_COMMIT, get_config


CASES = [
    dict(q_agg='mean', normalize_q_loss=False, action_dim=5, full_width=False, flow_steps=10, alpha=10.),
    dict(q_agg='min', normalize_q_loss=False, action_dim=5, full_width=False, flow_steps=1, alpha=300.),
    dict(q_agg='mean', normalize_q_loss=True, action_dim=5, full_width=False, flow_steps=10, alpha=300.),
    dict(q_agg='min', normalize_q_loss=True, action_dim=6, full_width=False, flow_steps=3, alpha=100.),
    dict(q_agg='mean', normalize_q_loss=False, action_dim=5, full_width=True, flow_steps=10, alpha=300.),
]


def load_upstream(root):
    root = Path(root).resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != UPSTREAM_COMMIT:
        raise ValueError('Unexpected official FQL commit')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip():
        raise ValueError('Official tracked files must be unmodified')
    saved = {name: value for name, value in sys.modules.items() if name == 'utils' or name.startswith('utils.')}
    paths = {}
    try:
        for name in saved:
            del sys.modules[name]
        package = types.ModuleType('utils')
        package.__path__ = [str(root / 'utils')]
        sys.modules['utils'] = package
        for name, relative in [('utils.networks', 'utils/networks.py'), ('utils.encoders', 'utils/encoders.py'),
                               ('utils.flax_utils', 'utils/flax_utils.py'), ('_official_fql_agent', 'agents/fql.py')]:
            path = root / relative
            paths[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module.FQLAgent, paths
    finally:
        for name in list(sys.modules):
            if name == 'utils' or name.startswith('utils.'):
                del sys.modules[name]
        sys.modules.update(saved)


def assert_same(local, official):
    maximum = 0.
    for field in ('params', 'opt_state', 'step'):
        a, b = (flax.serialization.to_state_dict(getattr(agent.network, field)) for agent in (local, official))
        if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
            raise AssertionError('State structure mismatch: ' + field)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
            np.testing.assert_allclose(x, y, rtol=3e-6, atol=3e-7, err_msg=field)
            maximum = max(maximum, float(np.max(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)))))
    np.testing.assert_array_equal(local.rng, official.rng)
    return maximum


def run_case(official_class, case):
    rng = np.random.default_rng(20261028)
    hidden = (512, 512, 512, 512) if case['full_width'] else (16, 16)
    config = get_config().to_dict()
    config.update(actor_hidden_dims=hidden, value_hidden_dims=hidden,
                  **{name: case[name] for name in ('q_agg', 'normalize_q_loss', 'flow_steps', 'alpha')})
    example_obs = np.zeros((2, 7), dtype=np.float32)
    example_actions = np.zeros((2, case['action_dim']), dtype=np.float32)
    local = FQLReferenceAgent.create(20261028, example_obs, example_actions, config)
    official = official_class.create(20261028, example_obs, example_actions, dict(config))
    errors = [assert_same(local, official)]
    for update in range(3):
        batch = dict(observations=jnp.asarray(rng.normal(size=(4, 7)), dtype=jnp.float32),
                     actions=jnp.asarray(rng.uniform(-.9, .9, (4, case['action_dim'])), dtype=jnp.float32),
                     next_observations=jnp.asarray(rng.normal(size=(4, 7)), dtype=jnp.float32),
                     rewards=jnp.asarray([-1., 0., -2., 1.]), masks=jnp.asarray([1., 0., 1., 0.]))
        key = jax.random.PRNGKey(850 + update)
        np.testing.assert_array_equal(local.sample_actions(batch['observations'], rng=key),
                                      official.sample_actions(batch['observations'], seed=key))
        noises = jnp.asarray(rng.normal(size=(4, case['action_dim'])), dtype=jnp.float32)
        np.testing.assert_array_equal(local.compute_flow_actions(batch['observations'], noises),
                                      official.compute_flow_actions(batch['observations'], noises))
        local, left_info = local.update(batch)
        official, right_info = official.update(batch)
        errors.append(assert_same(local, official))
        assert set(left_info) == set(right_info)
        for name in left_info:
            np.testing.assert_allclose(left_info[name], right_info[name], rtol=3e-6, atol=3e-7, err_msg=name)
    return dict(**case, hidden_dims=hidden, update_calls=3, train_state_step=int(local.network.step),
                max_absolute_state_error=max(errors), status='passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-index', type=int, required=True, choices=range(len(CASES)))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing numerical evidence')
    if jax.default_backend() != 'cpu':
        raise ValueError('Run official equivalence checks on CPU')
    local_paths = [Path(sys.modules[FQLReferenceAgent.__module__].__file__), Path(__file__),
                   Path('utils/networks.py'), Path('utils/flax_utils.py')]
    input_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in local_paths}
    official, paths = load_upstream(args.upstream_root)
    case = run_case(official, CASES[args.case_index])
    for path, expected in input_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Local source changed during numerical validation: ' + path)
    for path, expected in paths.items():
        if hashlib.sha256((args.upstream_root / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Official source changed during numerical validation: ' + path)
    report = dict(status='passed', upstream_commit=UPSTREAM_COMMIT, upstream_files=paths, case=case,
                  input_hashes=input_hashes,
                  local_learner_sha256=hashlib.sha256(Path(sys.modules[FQLReferenceAgent.__module__].__file__).read_bytes()).hexdigest(),
                  local_networks_sha256=hashlib.sha256(Path('utils/networks.py').read_bytes()).hexdigest(),
                  local_train_state_sha256=hashlib.sha256(Path('utils/flax_utils.py').read_bytes()).hexdigest(),
                  oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  jax=jax.__version__, flax=flax.__version__,
                  scope='Native state-based learner initialization, sampling, BC-flow integration and three updates only. Not a driver, performance or posthoc transfer verification.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
