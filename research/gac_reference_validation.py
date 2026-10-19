"""Audit source-bound GAC runs without updates or partial-cohort inference."""
import hashlib
from dataclasses import asdict
import importlib.metadata
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


def expected_updates(plan):
    budget, learner = plan['online_steps'], plan['learner']
    start, actor, target = (learner[k] for k in ('learning_starts', 'policy_frequency', 'target_network_frequency'))
    last = budget - 1
    return dict(critic=max(0, last - start),
                actor=max(0, last // actor - start // actor) * actor if last > start else 0,
                target=max(0, last // target - start // target) if last > start else 0)


def expected_grid(plan):
    budget, interval = plan['online_steps'], plan['eval_interval']
    if type(budget) is not int or type(interval) is not int or min(budget, interval) < 1:
        raise ValueError('Expected a positive budget and evaluation interval')
    return sorted({0, budget} | set(range(interval, budget + 1, interval)))


def curve_metrics(rows, plan):
    grid = expected_grid(plan)
    if [row['step'] for row in rows] != grid:
        raise ValueError('Incomplete, duplicate or reordered evaluation grid')
    rates, clipped = [], 0
    for row in rows:
        episodes = row['episodes']
        if row['episode_count'] != plan['eval_episodes'] or len(episodes) != row['episode_count']:
            raise ValueError('Actual evaluation episode count differs')
        for index, episode in enumerate(episodes):
            if (episode['episode'] != index or episode['reset_seed'] != plan['eval_seed'] + index
                    or type(episode['length']) is not int or episode['length'] < 1
                    or type(episode['clipped_action_steps']) is not int
                    or not 0 <= episode['clipped_action_steps'] <= episode['length']
                    or episode['success'] not in (0, 1)):
                raise ValueError('Invalid observed evaluation episode')
            if plan.get('action_execution', 'identity') == 'identity' and episode['clipped_action_steps']:
                raise ValueError('Identity evaluation claims clipping')
            clipped += episode['clipped_action_steps']
        success = sum(episode['success'] for episode in episodes)
        if row['success_count'] != success or row['success'] != success / len(episodes):
            raise ValueError('Evaluation success aggregate differs from observed episodes')
        rates.append(row['success'])
    steps, rates = np.asarray(grid), np.asarray(rates, dtype=float)
    reached = steps[rates >= .5]
    return dict(auc=float(np.trapezoid(rates, steps) / plan['online_steps']), endpoint=float(rates[-1]),
                tail3=float(rates[-3:].mean()), tail_evaluations=min(3, len(rates)),
                time_to_50=int(reached[0]) if len(reached) else None, time_to_50_censored=not bool(len(reached)),
                eval_steps=grid, success=rates.tolist(), evaluation_clipped_steps=clipped)


def validate_actions(path, plan, prefix_limit=0):
    radius, start = plan['learner']['radius'], plan['learner']['learning_starts']
    execution = plan.get('action_execution', 'identity')
    if execution not in ('identity', 'clip') or execution == 'identity' and radius != 1:
        raise ValueError('Invalid execution/radius contract')
    count = clipped = 0
    max_request = max_execution = 0.
    prefix = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['step'] != count + 1 or row['zero_based_loop'] != count:
                raise ValueError('Missing, duplicate or reordered collection step')
            warmup = count < start
            if type(row['warmup']) is not bool or row['warmup'] != warmup:
                raise ValueError('Warm-up clock differs')
            if type(row['terminated']) is not bool or type(row['truncated']) is not bool:
                raise ValueError('Termination and truncation must be observed separately')
            requested, actual = np.asarray(row['requested_action']), np.asarray(row['action'])
            if (requested.shape != (5,) or actual.shape != (5,) or not np.isfinite(requested).all()
                    or not np.isfinite(actual).all() or np.max(np.abs(actual)) > 1 + 1e-6):
                raise ValueError('Illegal requested/executed action')
            if warmup:
                if np.max(np.abs(requested)) > 1 + 1e-6:
                    raise ValueError('Warm-up request exceeds Box')
            elif np.linalg.norm(requested) > radius + 1e-6 * max(1., radius):
                raise ValueError('Policy request exceeds full-vector radius')
            expected = np.clip(requested, -1, 1) if execution == 'clip' else requested
            if not np.array_equal(actual, expected):
                raise ValueError('Executed action differs from declared mapping')
            clipped += int(np.any(requested != actual))
            max_request = max(max_request, float(np.max(np.abs(requested))))
            max_execution = max(max_execution, float(np.max(np.abs(actual))))
            if count < prefix_limit:
                prefix.append(row['requested_action'])
            count += 1
    if count != plan['online_steps']:
        raise ValueError('Incomplete action stream')
    return dict(environment_steps=count, execution_clipped_steps=clipped, max_requested_abs=max_request,
                max_executed_abs=max_execution, prefix_requested_actions=prefix)


def validate_training(rows, plan):
    budget, interval, learner = plan['online_steps'], plan['log_interval'], plan['learner']
    start, frequency = learner['learning_starts'], learner['policy_frequency']
    expected = [step for step in range(interval, budget + 1, interval) if step - 1 > start]
    if budget - 1 > start and budget not in expected:
        expected.append(budget)
    if [row['step'] for row in rows] != expected:
        raise ValueError('Incomplete training diagnostic grid')
    for row in rows:
        if row['global_step'] != row['step'] - 1 or not np.isfinite(list(row.values())).all():
            raise ValueError('Nonfinite diagnostics or incorrect update clock')
        latest_actor = row['global_step'] // frequency * frequency
        if latest_actor > start:
            if row.get('actor_global_step') != latest_actor:
                raise ValueError('Actor loss timestamp differs from actual last actor event')
            if not {'actor_loss', 'sac_actor_loss', 'kappa_reg_loss'} <= set(row):
                raise ValueError('Missing actual actor diagnostics')
        elif any(key in row for key in ('actor_global_step', 'actor_loss', 'sac_actor_loss', 'kappa_reg_loss')):
            raise ValueError('Actor diagnostics precede the first actor event')
        if not {'qf_loss', 'qf1_loss', 'qf2_loss'} <= set(row):
            raise ValueError('Missing actual critic diagnostics')
    return len(rows)


def assert_checkpoint_equal(left, right):
    import torch
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError('Checkpoint keys differ')
        return sum(assert_checkpoint_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError('Checkpoint sequence differs')
        return sum(assert_checkpoint_equal(a, b) for a, b in zip(left, right))
    if torch.is_tensor(left):
        if (not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype
                or not torch.equal(left.cpu(), right.cpu()) or not torch.isfinite(left).all()):
            raise ValueError('Checkpoint tensor is nonfinite or differs')
    elif type(left) is not type(right) or left != right:
        raise ValueError('Checkpoint scalar differs')
    return 1


def validate_run(directory, plan_path, process, source_root):
    """Validate a stopped run with a supervisor-observed exit code, on its backend."""
    directory, plan_path, source_root = map(Path, (directory, plan_path, source_root))
    if (type(process.get('returncode')) is not int or process['returncode'] != 0
            or not np.isfinite([process['started'], process['ended']]).all()
            or process['ended'] < process['started'] or (directory / 'incident.json').exists()):
        raise ValueError('Missing successful process lifecycle evidence')
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != process['plan_sha256']:
        raise ValueError('Supervisor plan identity differs')
    names = ['receipt.json', 'effective_config.json', 'actions.jsonl', 'train.jsonl', 'eval.jsonl', 'final.pt']
    if expected_updates(plan)['critic']:
        names += ['first-update-batch.npz', 'first-update-online-replay.npz']
    hashes = {str(directory / name): digest(directory / name) for name in names}
    receipt = json.loads((directory / 'receipt.json').read_text())
    execution = plan.get('action_execution', 'identity')
    if (receipt['status'] != 'complete' or receipt['source_commit'] != plan['source_commit']
            or receipt['source_files'] != plan['source_files'] or receipt['plan_sha256'] != digest(plan_path)
            or receipt['replay_protocol'] != plan['replay_protocol'] or receipt['action_execution'] != execution
            or receipt['replay_action_coordinates'] != 'requested' or receipt['critic_action_coordinates'] != 'requested'):
        raise ValueError('Receipt identity or action-coordinate contract differs')
    updates = expected_updates(plan)
    prior_rows = updates['critic'] * plan['learner']['batch_size'] // 2 if plan['replay_protocol'] == 'prior_balanced' else 0
    online_rows = updates['critic'] * plan['learner']['batch_size'] - prior_rows
    if (receipt['actual_updates'] != updates or receipt['evaluation_steps'] != expected_grid(plan)
            or receipt['observed_eval_episode_counts'] != [plan['eval_episodes']] * len(expected_grid(plan))
            or receipt['counts']['environment_transitions'] != plan['online_steps']
            or receipt['counts']['sampling_events'] != updates['critic']
            or receipt['counts']['sampled_prior_rows'] != prior_rows or receipt['counts']['sampled_online_rows'] != online_rows):
        raise ValueError('Observed update, data-origin or evaluation counts differ')
    sys.path.insert(0, str(source_root))
    import torch
    import main_gac_reference as driver
    from agents.gac_reference import GACReferenceAgent
    for module in (driver, sys.modules[GACReferenceAgent.__module__]):
        Path(module.__file__).resolve().relative_to(source_root.resolve())
    config = driver.validate_plan(plan)
    driver.verify_inputs(plan, source_root)
    effective_path = directory / 'effective_config.json'
    effective = json.loads(effective_path.read_text())
    if (digest(effective_path) != receipt['effective_config_sha256'] or effective['learner'] != asdict(config)
            or effective['replay_protocol'] != plan['replay_protocol'] or effective['online_replay_capacity'] != plan['replay_capacity']
            or effective['action_dim'] != 5 or effective['action_execution'] != execution
            or effective['replay_action_coordinates'] != 'requested' or effective['critic_action_coordinates'] != 'requested'
            or effective['source_commit'] != plan['source_commit']):
        raise ValueError('Effective configuration differs')
    packages = {name: importlib.metadata.version(name) for name in plan['environment_versions']}
    if (torch.__version__ != plan['torch_version'] or packages != plan['environment_versions']
            or receipt['runtime']['torch'] != plan['torch_version'] or receipt['runtime']['packages'] != plan['environment_versions']
            or receipt['runtime']['device'] != plan['device'] or effective['device'] != plan['device']):
        raise ValueError('Actual runtime differs from plan')
    prefix_limit = config.learning_starts + 2 if updates['critic'] else 0
    action_summary = validate_actions(directory / 'actions.jsonl', plan, prefix_limit)
    prefix = np.asarray(action_summary.pop('prefix_requested_actions'), dtype='float32')
    if action_summary['execution_clipped_steps'] != receipt['counts']['execution_clipped_steps']:
        raise ValueError('Clipping count differs from actual actions')
    observations = None
    if updates['critic']:
        with np.load(directory / 'first-update-batch.npz') as saved, np.load(directory / 'first-update-online-replay.npz') as online:
            indices, prior_indices = saved['online_indices'], saved['prior_indices']
            offset = config.batch_size // 2 if plan['replay_protocol'] == 'prior_balanced' else 0
            if len(indices) != config.batch_size - offset or len(prior_indices) != offset:
                raise ValueError('First batch data-origin counts differ')
            expected_size = min(config.learning_starts + 2, plan['replay_capacity'])
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= expected_size)
                    or prior_indices.ndim != 1 or not np.issubdtype(prior_indices.dtype, np.integer)
                    or np.any(prior_indices < 0)):
                raise ValueError('Invalid recorded replay indices')
            shapes = dict(observations=(effective['observation_dim'],), next_observations=(effective['observation_dim'],),
                          actions=(5,), rewards=(1,), dones=(1,))
            for name in driver.FIELDS:
                if (online[name].shape != (expected_size,) + shapes[name]
                        or saved[name].shape != (config.batch_size,) + shapes[name]
                        or online[name].dtype != np.float32 or saved[name].dtype != np.float32
                        or not np.isfinite(online[name]).all() or not np.isfinite(saved[name]).all()):
                    raise ValueError('Invalid first replay snapshot/batch fields')
                np.testing.assert_array_equal(saved[name][offset:], online[name][indices])
            if not np.isin(saved['dones'], [0, 1]).all() or not np.isin(online['dones'], [0, 1]).all():
                raise ValueError('First replay terminal mask is not binary')
            row = np.arange(expected_size)
            if len(prefix) >= plan['replay_capacity']:
                row = (len(prefix) - 1 - row) // plan['replay_capacity'] * plan['replay_capacity'] + row
            np.testing.assert_array_equal(online['actions'], prefix[row])
            if offset:
                from envs.ogbench_utils import make_ogbench_env_and_datasets
                env, eval_env, raw, _ = make_ogbench_env_and_datasets(plan['env_name'], dataset_path=plan['dataset_path'], compact_dataset=False)
                try:
                    prior = {name: np.asarray(raw[name], dtype='float32') for name in ('observations', 'actions', 'next_observations')}
                    prior['rewards'] = np.asarray(raw['rewards'], dtype='float32').reshape(-1, 1)
                    prior['dones'] = (1 - np.asarray(raw['masks'], dtype='float32')).reshape(-1, 1)
                    if effective['prior_transitions'] != len(prior['actions']):
                        raise ValueError('Actual prior size differs')
                    if np.any(prior_indices >= len(prior['actions'])):
                        raise ValueError('Prior index exceeds the verified dataset')
                    for name in driver.FIELDS:
                        np.testing.assert_array_equal(saved[name][:offset], prior[name][prior_indices])
                finally:
                    env.close(); eval_env.close()
            elif effective['prior_transitions'] != 0:
                raise ValueError('Online-only run declares prior data')
            observations = np.asarray(saved['observations'][:min(16, plan['online_steps'])]).copy()
    checkpoint = directory / 'final.pt'
    if digest(checkpoint) != receipt['checkpoint_sha256']:
        raise ValueError('Checkpoint identity differs')
    agent = GACReferenceAgent(plan['seed'], effective['observation_dim'], -np.ones(5), np.ones(5), config, plan['device'])
    if driver.initial_state_digest(agent) != receipt['initial_model_and_rng_sha256']:
        raise ValueError('Initial model/RNG digest differs')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    agent.load_state_dict(state)
    fields = assert_checkpoint_equal(state, agent.state_dict())
    if (agent.critic_updates, agent.actor_updates, agent.target_updates) != tuple(updates[k] for k in ('critic', 'actor', 'target')):
        raise ValueError('Restored learner update counts differ')
    if agent.last_global_step != (plan['online_steps'] - 1 if updates['critic'] else None):
        raise ValueError('Restored learner clock differs')
    for optimizer, key in ((agent.q_optimizer, 'critic'), (agent.actor_optimizer, 'actor')):
        if {int(value['step']) for value in optimizer.state.values()} != ({updates[key]} if updates[key] else set()):
            raise ValueError('Actual Adam count differs')
    probe_count = min(16, plan['online_steps'])
    if fields != receipt['exact_checkpoint_fields'] or receipt['restored_policy_probes'] != probe_count:
        raise ValueError('Checkpoint restoration proof differs')
    if observations is None:
        observations = np.zeros((probe_count, effective['observation_dim']), dtype='float32')
    probe = agent.sample_actions(observations, seed=20261102)
    if probe.shape != (probe_count, 5) or not np.isfinite(probe).all() or np.max(np.linalg.norm(probe, axis=-1)) > config.radius + 1e-6:
        raise ValueError('Invalid restored requested-action probes')
    metrics = curve_metrics([json.loads(line) for line in (directory / 'eval.jsonl').read_text().splitlines()], plan)
    training = validate_training([json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()], plan)
    driver.verify_inputs(plan, source_root)
    if digest(plan_path) != process['plan_sha256'] or any(digest(path) != expected for path, expected in hashes.items()):
        raise ValueError('Inputs or artifacts changed during validation')
    return dict(status='complete', returncode=0, source_commit=plan['source_commit'], env_name=plan['env_name'], seed=plan['seed'],
                replay_protocol=plan['replay_protocol'], radius=config.radius, action_execution=execution,
                plan_sha256=digest(plan_path), online_steps=plan['online_steps'], actual_updates=updates, metrics=metrics,
                action_summary=action_summary, training_diagnostic_rows=training, exact_checkpoint_fields=fields,
                restored_policy_probes=probe_count, artifacts=hashes, effective_config=effective,
                started=process['started'], ended=process['ended'], runtime=receipt['runtime'],
                checkpoint_scope='Complete learner state, not replay/environment/sampler trajectory continuation.')
