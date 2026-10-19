"""Validate source-bound FQL runs and describe predeclared paired contrasts.

Online and offline curves use separate axes. Four-seed summaries are descriptive;
missing or failed pairs suppress group estimates. No model updates occur here.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def expected_grid(plan):
    offline, online, interval = (plan[k] for k in ('offline_steps', 'online_steps', 'eval_interval'))
    if min(offline, online) < 0 or interval < 1 or online == 0:
        raise ValueError('Validation requires a nonnegative offline and positive online budget')
    total = offline + online
    return sorted({0, offline, total} | set(range(interval, total + 1, interval)))


def curve_metrics(rows, plan):
    grid = expected_grid(plan)
    if [row['step'] for row in rows] != grid:
        raise ValueError('Incomplete, duplicate or reordered evaluation grid')
    rates = []
    for row in rows:
        episodes = row['episodes']
        if row['episode_count'] != plan['eval_episodes'] or len(episodes) != row['episode_count']:
            raise ValueError('Observed episode count differs')
        for index, episode in enumerate(episodes):
            if (episode['episode'] != index or episode['reset_seed'] != plan['eval_seed'] + index
                    or type(episode['length']) is not int or episode['length'] < 1
                    or episode['success'] not in (0, 1)):
                raise ValueError('Invalid observed episode record')
        count = sum(episode['success'] for episode in episodes)
        if row['success_count'] != count or row['success'] != count / len(episodes):
            raise ValueError('Success aggregate disagrees with observed episodes')
        if row['online_step'] != max(0, row['step'] - plan['offline_steps']):
            raise ValueError('Online evaluation axis differs')
        rates.append(row['success'])
    steps, success = np.asarray(grid), np.asarray(rates, dtype=float)
    if not np.isfinite(success).all() or np.any((success < 0) | (success > 1)):
        raise ValueError('Invalid success curve')
    online = steps >= plan['offline_steps']
    online_steps = steps[online] - plan['offline_steps']
    online_success = success[online]
    reached = online_steps[online_success >= .5]
    offline = steps <= plan['offline_steps']
    return dict(online_auc=float(np.trapezoid(online_success, online_steps) / plan['online_steps']),
                endpoint=float(online_success[-1]), online_tail3=float(online_success[-3:].mean()),
                offline_auc=float(np.trapezoid(success[offline], steps[offline]) / plan['offline_steps']) if plan['offline_steps'] else None,
                offline_endpoint=float(success[offline][-1]), time_to_50=int(reached[0]) if len(reached) else None,
                time_to_50_censored=not bool(len(reached)), online_eval_steps=online_steps.tolist(),
                eval_steps=grid, success=success.tolist())


def assert_state_equal(left, right):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError('Checkpoint state keys differ')
        return sum(assert_state_equal(left[key], right[key]) for key in left)
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        raise ValueError('Checkpoint state values, shapes or dtypes differ')
    return 1


def validate_actions(path, plan, adapter, env_dim, policy_dim, block_size=4096):
    count = 0
    latent, actual = [], []
    def flush():
        if not latent:
            return
        policy, executed = np.asarray(latent, dtype='float32'), np.asarray(actual, dtype='float32')
        if policy.shape != (len(latent), policy_dim) or executed.shape != (len(latent), env_dim):
            raise ValueError('Wrong policy/environment action shape')
        if (not np.isfinite(policy).all() or not np.isfinite(executed).all()
                or np.max(np.abs(policy)) > 1 or np.max(np.abs(executed)) > 1):
            raise ValueError('Illegal policy or environment action')
        expected = policy if adapter is None else adapter.compose(policy)
        if not np.array_equal(expected, executed):
            raise ValueError('Executed action differs from registered map')
        latent.clear(); actual.clear()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line); count += 1
            if row['online_step'] != count or row['step'] != plan['offline_steps'] + count:
                raise ValueError('Missing, duplicate or reordered action step')
            if row['mask'] not in (0., 1.) or row['terminal'] not in (0., 1.) or (row['mask'] == 0 and row['terminal'] != 1):
                raise ValueError('Invalid mask/termination record')
            latent.append(row['policy_action']); actual.append(row['env_action'])
            if len(latent) == block_size: flush()
    flush()
    if count != plan['online_steps']:
        raise ValueError('Wrong actual environment-step count')
    return count


def validate_run(plan_path, directory, process, source_root):
    plan_path, directory, source_root = Path(plan_path), Path(directory), Path(source_root).resolve()
    plan = json.loads(plan_path.read_text())
    if process['returncode'] != 0 or process['plan_sha256'] != digest(plan_path):
        raise ValueError('Real process exit or plan identity differs')
    if process['source_commit'] != plan['source_commit']:
        raise ValueError('Process/source identity differs')
    if (directory / 'incident.json').exists():
        raise ValueError('Run retains an unresolved training incident')
    initial_hashes = {str(path): digest(path) for path in directory.iterdir() if path.is_file()}
    receipt = json.loads((directory / 'receipt.json').read_text())
    if (receipt['status'] != 'complete' or receipt['source_commit'] != plan['source_commit']
            or receipt['source_files'] != plan['source_files'] or receipt['plan_sha256'] != digest(plan_path)
            or receipt['method'] != plan['method']):
        raise ValueError('Driver receipt identity differs')
    total = plan['offline_steps'] + plan['online_steps']
    if (receipt['counts'] != dict(offline_updates=plan['offline_steps'], online_updates=plan['online_steps'], online_environment_steps=plan['online_steps'])
            or receipt['joint_adam_count'] != total or receipt['train_state_step'] != total + 1
            or receipt['evaluation_steps'] != expected_grid(plan)
            or receipt['observed_episode_counts'] != [plan['eval_episodes']] * len(expected_grid(plan))
            or not receipt['checkpoint_full_state_exact'] or receipt['restored_policy_probes'] != 16
            or receipt['policy_output_transform'] != 'clip'):
        raise ValueError('Driver counters, evaluations or checkpoint proof differ')
    sys.path.insert(0, str(source_root))
    import flax
    import jax
    import jax.numpy as jnp
    import main_fql_reference as driver
    from agents.fql_reference import FQLReferenceAgent
    from agents.fql_posthoc_reference import FQLPosthocReferenceAgent
    from direction_speed.adapter import DirectionSpeedAdapter
    if jax.default_backend() != 'cpu':
        raise ValueError('Final FQL validation is CPU only')
    for module in (driver, sys.modules[FQLReferenceAgent.__module__], sys.modules[FQLPosthocReferenceAgent.__module__]):
        Path(module.__file__).resolve().relative_to(source_root)
    driver.validate_plan(plan); driver.verify_inputs(plan, source_root)
    config_path = directory / 'effective_agent_config.json'
    if digest(config_path) != receipt['effective_config_sha256']:
        raise ValueError('Effective configuration hash differs')
    config = json.loads(config_path.read_text())
    expected_config = driver.make_learner_config(plan)
    for name, value in expected_config.items():
        if name in ('ob_dims', 'action_dim'):
            continue
        if isinstance(value, tuple): value = list(value)
        if config.get(name) != value:
            raise ValueError('Effective learner setting differs: ' + name)
    adapter = None if plan['method'] == 'native' else DirectionSpeedAdapter(speed_squash='tanh', speed_interpretation='log', speed_bound='sphere')
    panels = []
    for phase in (('offline', 'online') if plan['offline_steps'] else ('online',)):
        with np.load(directory / f'first-{phase}-batch.npz') as saved:
            raw = {key.removeprefix('raw_'): saved[key] for key in saved if key.startswith('raw_')}
            prepared = driver.prepare_batch(raw, adapter)
            if not np.array_equal(prepared['actions'], saved['encoded_actions']):
                raise ValueError('First batch encoding differs: ' + phase)
            panels.append(prepared)
    observations = panels[0]['observations'][:16]
    for name in ('actor_hidden_dims', 'value_hidden_dims'): config[name] = tuple(config[name])
    cls = FQLReferenceAgent if plan['method'] == 'native' else FQLPosthocReferenceAgent
    template = cls.create(plan['seed'], observations[:1], panels[0]['actions'][:1], config)
    checkpoint = directory / 'final.msgpack'
    if digest(checkpoint) != receipt['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs')
    payload = checkpoint.read_bytes()
    agent = flax.serialization.from_bytes(template, payload)
    fields = assert_state_equal(flax.serialization.msgpack_restore(payload), flax.serialization.to_state_dict(agent))
    if int(agent.network.step) != total + 1 or int(agent.network.opt_state[0].count) != total:
        raise ValueError('Actual optimizer counters differ')
    if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves((agent.network.params, agent.network.opt_state))):
        raise ValueError('Nonfinite parameters or optimizer state')
    probe = np.asarray(agent.sample_actions(jnp.asarray(observations), rng=jax.random.PRNGKey(20261030)))
    if not np.isfinite(probe).all() or np.max(np.abs(probe)) > 1:
        raise ValueError('Invalid restored policy probe')
    env_dim = config['env_action_dim']; policy_dim = env_dim if adapter is None else env_dim + 1
    if (config['policy_action_dim'] != policy_dim or config['critic_action_dim'] != policy_dim
            or config['action_dim'] != policy_dim or probe.shape != (16, policy_dim)):
        raise ValueError('Restored action dimensions differ')
    action_count = validate_actions(directory / 'actions.jsonl', plan, adapter, env_dim, policy_dim)
    rows = [json.loads(line) for line in (directory / 'eval.jsonl').read_text().splitlines()]
    metrics = curve_metrics(rows, plan)
    expected_training = [(step, 'offline') for step in range(plan['log_interval'], plan['offline_steps'] + 1, plan['log_interval'])]
    if plan['offline_steps'] and plan['offline_steps'] % plan['log_interval']:
        expected_training.append((plan['offline_steps'], 'offline'))
    expected_training += [(plan['offline_steps'] + step, 'online') for step in range(plan['log_interval'], plan['online_steps'] + 1, plan['log_interval'])]
    if plan['online_steps'] % plan['log_interval']:
        expected_training.append((total, 'online'))
    training = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()]
    if [(row['step'], row['phase']) for row in training] != expected_training:
        raise ValueError('Incomplete actual training diagnostic grid')
    if not all(np.isfinite(value) for row in training for key, value in row.items() if key not in ('step', 'phase')):
        raise ValueError('Nonfinite training diagnostics')
    driver.verify_inputs(plan, source_root)
    if digest(plan_path) != process['plan_sha256']:
        raise ValueError('Plan changed during validation')
    if any(digest(path) != expected for path, expected in initial_hashes.items()):
        raise ValueError('Run artifacts changed during validation')
    return dict(status='complete', returncode=0, source_commit=plan['source_commit'], seed=plan['seed'], method=plan['method'],
                env_name=plan['env_name'], plan_sha256=digest(plan_path), run_dir=str(directory),
                offline_steps=plan['offline_steps'], online_steps=plan['online_steps'],
                metrics=metrics, effective_config=json.loads(config_path.read_text()),
                actual_joint_optimizer_updates=total, actual_environment_steps=action_count,
                restored_policy_probes=16, exact_checkpoint_state_fields=fields,
                observed_evaluation_episode_counts='verified', artifacts=initial_hashes,
                started=process['started'], ended=process['ended'], runtime=receipt['runtime'])


def summarize_pairs(cohort, records):
    seeds, tasks = cohort['seeds'], cohort['tasks']
    if (len(seeds) != 4 or len(set(seeds)) != 4 or cohort['offline_steps'] <= 0 or cohort['online_steps'] <= 0
            or len({task['env_name'] for task in tasks}) != len(tasks)):
        raise ValueError('Expected registered four-pair offline-to-online development design')
    expected = {(task['env_name'], method, seed) for task in tasks for method in ('native', 'posthoc') for seed in seeds}
    indexed = {}
    for record in records:
        key = (record['env_name'], record['method'], record['seed'])
        if key in indexed or key not in expected:
            raise ValueError('Unexpected or duplicate run identity')
        if record['status'] == 'complete' and (record['source_commit'] != cohort['source_commit'] or record['returncode'] != 0):
            raise ValueError('Unvalidated complete receipt')
        indexed[key] = record
    output = []
    for task in tasks:
        pairs, unavailable = [], []
        for seed in seeds:
            rows = [indexed.get((task['env_name'], method, seed)) for method in ('native', 'posthoc')]
            if any(row is None or row['status'] != 'complete' for row in rows):
                unavailable.append(seed)
            else:
                if any(row['offline_steps'] != cohort['offline_steps'] or row['online_steps'] != cohort['online_steps'] for row in rows):
                    raise ValueError('Paired budgets differ')
                method_keys = {'agent_name', 'action_dim', 'policy_action_dim', 'critic_action_dim',
                               'adapter_representation', 'adapter_speed_squash', 'adapter_speed_interpretation',
                               'ds_speed_bound', 'adapter_critic_space', 'policy_output_transform'}
                common = [{key: value for key, value in row['effective_config'].items() if key not in method_keys} for row in rows]
                if common[0] != common[1]:
                    raise ValueError('Paired learner settings differ beyond registered method coordinates')
                pairs.append(dict(seed=seed, differences={name: rows[1]['metrics'][name] - rows[0]['metrics'][name]
                    for name in ('online_auc', 'endpoint', 'online_tail3', 'offline_auc', 'offline_endpoint')},
                    time_to_50=[{k: row['metrics'][k] for k in ('time_to_50', 'time_to_50_censored')} for row in rows]))
        comparison = dict(task=task['name'], contrast='posthoc - native', status='incomplete' if unavailable else 'complete',
                          pairs=pairs, unavailable_seeds=unavailable, primary_metric='online_auc',
                          interpretation='Four-seed development transfer, descriptive only; not confirmation, pure-coordinate or entropy attribution.')
        if not unavailable:
            indices = np.asarray(list(itertools.product(range(len(seeds)), repeat=len(seeds))))
            comparison['summary'] = {}
            for metric in pairs[0]['differences']:
                values = np.asarray([pair['differences'][metric] for pair in pairs])
                comparison['summary'][metric] = dict(mean=float(values.mean()), median=float(np.median(values)),
                    positive_pairs=int((values > 0).sum()), zero_pairs=int((values == 0).sum()),
                    descriptive_bootstrap_95=np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist())
        output.append(comparison)
    return output
