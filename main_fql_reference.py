"""Explicit native/posthoc H1 FQL driver for source-bound OGBench experiments.

This is a local driver adaptation, not the official experiment script. Plans
must bind data and source. Outputs are exclusive; failures are never retried.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.fql_reference import FQLReferenceAgent, get_config as native_config
from agents.fql_posthoc_reference import FQLPosthocReferenceAgent, get_config as posthoc_config
from direction_speed.adapter import DirectionSpeedAdapter
from utils.datasets import Dataset, ReplayBuffer

FIELDS = ('observations', 'actions', 'rewards', 'masks', 'terminals', 'next_observations')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_learner_config(plan):
    config = (native_config() if plan['method'] == 'native' else posthoc_config()).to_dict()
    unknown = set(plan['learner']) - set(config)
    if unknown:
        raise ValueError('Unknown learner settings: ' + ', '.join(sorted(unknown)))
    config.update(plan['learner'])
    expected = 'fql_reference' if plan['method'] == 'native' else 'fql_posthoc_reference'
    if config['agent_name'] != expected:
        raise ValueError('Learner identity differs from the selected method')
    for name in ('actor_hidden_dims', 'value_hidden_dims'):
        config[name] = tuple(config[name])
    FQLReferenceAgent._validate_learner_config(config)
    if plan['method'] == 'posthoc':
        FQLPosthocReferenceAgent._validate_method_config(config)
    if (not isinstance(config['batch_size'], int) or config['batch_size'] < 1
            or not np.isfinite([config['lr'], config['alpha'], config['discount'], config['tau']]).all()
            or config['lr'] <= 0 or config['alpha'] < 0
            or not 0 <= config['discount'] <= 1 or not 0 <= config['tau'] <= 1):
        raise ValueError('Invalid learner batch, learning rate, weight, discount or EMA')
    return config


def validate_plan(plan):
    required = {'method', 'seed', 'env_name', 'dataset_path', 'dataset_sha256', 'validation_sha256',
                'offline_steps', 'online_steps', 'buffer_size', 'balanced_sampling', 'eval_seed',
                'eval_interval', 'eval_episodes', 'log_interval', 'learner', 'source_files', 'source_commit'}
    if set(plan) != required:
        raise ValueError('Plan fields differ: ' + str(set(plan) ^ required))
    if plan['method'] not in ('native', 'posthoc'):
        raise ValueError('Unknown FQL method')
    if not plan['env_name'].startswith('cube-') or '-singletask-task' not in plan['env_name']:
        raise ValueError('Current reference driver supports explicit single-task OGBench Cube names')
    for name in ('offline_steps', 'online_steps', 'seed', 'eval_seed'):
        if type(plan[name]) is not int or plan[name] < 0:
            raise ValueError('Expected nonnegative integer: ' + name)
    for name in ('buffer_size', 'eval_interval', 'eval_episodes', 'log_interval'):
        if type(plan[name]) is not int or plan[name] < 1:
            raise ValueError('Expected positive integer: ' + name)
    if plan['offline_steps'] + plan['online_steps'] == 0 or type(plan['balanced_sampling']) is not bool:
        raise ValueError('Need a training budget and explicit replay mode')
    config = make_learner_config(plan)
    if plan['balanced_sampling'] and config['batch_size'] % 2:
        raise ValueError('Balanced replay requires an even batch size')
    required_sources = {'main_fql_reference.py', 'agents/fql_reference.py', 'agents/fql_posthoc_reference.py',
                        'direction_speed/adapter.py', 'direction_speed/speed.py', 'direction_speed/specs.py',
                        'envs/ogbench_utils.py', 'utils/datasets.py', 'utils/networks.py',
                        'utils/encoders.py', 'utils/flax_utils.py'}
    if not required_sources <= set(plan['source_files']) or not plan['source_commit']:
        raise ValueError('Missing required source provenance')
    return config


def verify_inputs(plan, source_root):
    for name, expected in plan['source_files'].items():
        path = (source_root / name).resolve()
        path.relative_to(source_root.resolve())
        if digest(path) != expected:
            raise ValueError('Source identity changed: ' + name)
    for path, expected in [(Path(plan['dataset_path']), plan['dataset_sha256']),
                           (Path(plan['dataset_path']).with_name(Path(plan['dataset_path']).stem + '-val.npz'),
                            plan['validation_sha256'])]:
        if digest(path) != expected:
            raise ValueError('Dataset identity changed: ' + str(path))


def prepare_batch(batch, adapter):
    data = {name: np.asarray(batch[name]).copy() for name in FIELDS if name in batch}
    if adapter is not None:
        data['actions'] = adapter.decompose(data['actions'])
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError('Nonfinite learner input')
    return data


def sample_training_batch(prior, replay, size, rng, balanced):
    def sample(dataset, count):
        return dataset.sample(count, idxs=rng.integers(dataset.size, size=count))
    if balanced:
        left, right = sample(prior, size // 2), sample(replay, size // 2)
        return {name: np.concatenate([left[name], right[name]], axis=0) for name in FIELDS}
    return sample(replay, size)


def environment_action(agent, observation, key, adapter):
    latent = np.asarray(agent.sample_actions(jnp.asarray(observation), rng=key))
    executed = latent if adapter is None else adapter.compose(latent[None])[0]
    if not np.isfinite(executed).all() or np.max(np.abs(executed)) > 1:
        raise ValueError('Invalid executed action')
    return latent, np.asarray(executed).copy()


def evaluate_policy(agent, env, adapter, seed, episodes):
    key = jax.random.PRNGKey(seed)
    learner_rng = np.asarray(agent.rng).copy()
    initial_step = int(agent.network.step)
    records = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done, steps = False, 0
        while not done:
            key, action_key = jax.random.split(key)
            _, action = environment_action(agent, observation, action_key, adapter)
            observation, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            if not np.isfinite(observation).all():
                raise ValueError('Nonfinite evaluation observation')
        success = float(info['success'])
        if success not in (0., 1.):
            raise ValueError('Expected observed binary terminal success')
        records.append(dict(episode=episode, reset_seed=seed + episode, length=steps, success=int(success)))
    np.testing.assert_array_equal(agent.rng, learner_rng)
    assert int(agent.network.step) == initial_step
    return dict(episode_count=len(records), success_count=sum(row['success'] for row in records),
                success=float(np.mean([row['success'] for row in records])), episodes=records)


def run(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text())
    plan_sha = digest(plan_path)
    config = validate_plan(plan)
    source_root = Path(__file__).resolve().parent
    verify_inputs(plan, source_root)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    counts = dict(offline_updates=0, online_updates=0, online_environment_steps=0)
    env = eval_env = None
    try:
        from envs.ogbench_utils import make_ogbench_env_and_datasets
        import importlib.metadata
        import mujoco
        env, eval_env, raw, _ = make_ogbench_env_and_datasets(
            plan['env_name'], dataset_path=plan['dataset_path'], compact_dataset=False)
        prior = Dataset.create(**{name: np.asarray(raw[name]) for name in FIELDS})
        if prior['observations'].ndim != 2 or prior['actions'].ndim != 2:
            raise ValueError('Reference driver requires vector observations and actions')
        adapter = None if plan['method'] == 'native' else DirectionSpeedAdapter(
            representation='cartesian', speed_squash='tanh', speed_interpretation='log', speed_bound='sphere')
        example = prepare_batch(prior.sample(1, idxs=np.asarray([0])), adapter)
        cls = FQLReferenceAgent if adapter is None else FQLPosthocReferenceAgent
        config['env_action_dim'] = prior['actions'].shape[-1]
        agent = cls.create(plan['seed'], example['observations'], example['actions'], config)
        config = dict(agent.config)
        (output / 'effective_agent_config.json').write_text(json.dumps(config, indent=2) + '\n')
        random.seed(plan['seed']); np.random.seed(plan['seed'])
        sampler = np.random.default_rng(plan['seed'])
        online_rng = jax.random.PRNGKey(plan['seed'])
        if plan['balanced_sampling']:
            replay = ReplayBuffer.create({name: prior[name][0] for name in FIELDS}, size=plan['buffer_size'])
        else:
            replay = ReplayBuffer.create_from_initial_dataset(dict(prior), size=max(plan['buffer_size'], prior.size + 1))
        evaluations = []
        total_budget = plan['offline_steps'] + plan['online_steps']
        eval_steps = {0, total_budget, plan['offline_steps']} | set(range(plan['eval_interval'], total_budget + 1, plan['eval_interval']))
        with (output / 'eval.jsonl').open('x') as eval_log, (output / 'train.jsonl').open('x') as train_log, (output / 'actions.jsonl').open('x') as action_log:
            def evaluation(step):
                row = dict(step=step, online_step=max(0, step - plan['offline_steps']),
                           **evaluate_policy(agent, eval_env, adapter, plan['eval_seed'], plan['eval_episodes']))
                evaluations.append(row); eval_log.write(json.dumps(row, allow_nan=False) + '\n'); eval_log.flush()
            evaluation(0)
            for step in range(1, plan['offline_steps'] + 1):
                raw_batch = prior.sample(config['batch_size'], idxs=sampler.integers(prior.size, size=config['batch_size']))
                batch = prepare_batch(raw_batch, adapter)
                if step == 1:
                    np.savez_compressed(output / 'first-offline-batch.npz', **{'raw_' + k: v for k, v in raw_batch.items()}, encoded_actions=batch['actions'])
                agent, info = agent.update(batch)
                counts['offline_updates'] += 1
                values = {name: float(value) for name, value in info.items()}
                if not np.isfinite(list(values.values())).all(): raise ValueError('Nonfinite offline losses')
                if step % plan['log_interval'] == 0 or step == plan['offline_steps']:
                    train_log.write(json.dumps(dict(step=step, phase='offline', **values), allow_nan=False) + '\n'); train_log.flush()
                if step in eval_steps: evaluation(step)
            observation = None
            for online_step in range(1, plan['online_steps'] + 1):
                if observation is None: observation, _ = env.reset(seed=plan['seed'])
                online_rng, key = jax.random.split(online_rng)
                latent, action = environment_action(agent, observation, key, adapter)
                next_observation, reward, terminated, truncated, _ = env.step(action.copy())
                if not np.isfinite(next_observation).all() or not np.isfinite(reward): raise ValueError('Nonfinite online transition')
                done = bool(terminated or truncated)
                transition = dict(observations=np.asarray(observation), actions=action, rewards=float(reward),
                                  terminals=float(done), masks=float(1. - terminated), next_observations=np.asarray(next_observation))
                replay.add_transition(transition)
                for name in FIELDS:
                    np.testing.assert_array_equal(replay[name][(replay.pointer - 1) % replay.max_size], np.asarray(transition[name], dtype=replay[name].dtype))
                counts['online_environment_steps'] += 1
                raw_batch = sample_training_batch(prior, replay, config['batch_size'], sampler, plan['balanced_sampling'])
                batch = prepare_batch(raw_batch, adapter)
                if online_step == 1:
                    np.savez_compressed(output / 'first-online-batch.npz', **{'raw_' + k: v for k, v in raw_batch.items()}, encoded_actions=batch['actions'])
                agent, info = agent.update(batch)
                counts['online_updates'] += 1
                step = plan['offline_steps'] + online_step
                action_log.write(json.dumps(dict(step=step, online_step=online_step, policy_action=latent.tolist(),
                    env_action=action.tolist(), mask=transition['masks'], terminal=transition['terminals']), allow_nan=False) + '\n')
                values = {name: float(value) for name, value in info.items()}
                if not np.isfinite(list(values.values())).all(): raise ValueError('Nonfinite online losses')
                if online_step % plan['log_interval'] == 0 or online_step == plan['online_steps']:
                    train_log.write(json.dumps(dict(step=step, phase='online', **values), allow_nan=False) + '\n'); train_log.flush()
                if step in eval_steps: evaluation(step)
                if done:
                    observation, _ = env.reset()
                else: observation = next_observation
        expected = counts['offline_updates'] + counts['online_updates']
        if expected != total_budget or int(agent.network.step) != 1 + expected or int(agent.network.opt_state[0].count) != expected:
            raise ValueError('Incorrect joint optimizer/TrainState counts')
        if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves((agent.network.params, agent.network.opt_state))):
            raise ValueError('Nonfinite final parameters or optimizer state')
        if sorted(row['step'] for row in evaluations) != sorted(eval_steps): raise ValueError('Incomplete evaluation grid')
        checkpoint = output / 'final.msgpack'
        checkpoint.write_bytes(flax.serialization.to_bytes(agent))
        restored = flax.serialization.from_bytes(agent, checkpoint.read_bytes())
        for left, right in zip(jax.tree_util.tree_leaves(flax.serialization.to_state_dict(agent)),
                               jax.tree_util.tree_leaves(flax.serialization.to_state_dict(restored)), strict=True):
            np.testing.assert_array_equal(left, right)
        probe_obs = jnp.asarray(prior['observations'][:16]); key = jax.random.PRNGKey(20261029)
        np.testing.assert_array_equal(agent.sample_actions(probe_obs, rng=key), restored.sample_actions(probe_obs, rng=key))
        verify_inputs(plan, source_root)
        if digest(plan_path) != plan_sha: raise ValueError('Plan changed while running')
        receipt = dict(status='complete', source_commit=plan['source_commit'], source_files=plan['source_files'], plan_sha256=plan_sha,
            method=plan['method'], counts=counts, joint_adam_count=expected, train_state_step=int(agent.network.step),
            evaluation_steps=sorted(eval_steps), observed_episode_counts=[row['episode_count'] for row in evaluations],
            checkpoint_sha256=digest(checkpoint), effective_config_sha256=digest(output / 'effective_agent_config.json'),
            replay_storage='raw environment actions; encoded on batch fetch for posthoc',
            policy_output_transform='clip', checkpoint_full_state_exact=True, restored_policy_probes=len(probe_obs),
            runtime=dict(jax=jax.__version__, flax=flax.__version__, numpy=np.__version__, mujoco=mujoco.__version__,
                         ogbench=importlib.metadata.version('ogbench'), backend=jax.default_backend()),
            started=started, finished=time.time(), limitations=['Local driver/data/evaluation adaptation, not official benchmark replication.',
                'Fixed episode reset/policy seeds do not certify all internal simulator RNG/cache state.',
                'FQL hard-clipped output distribution differs from SAC tanh-Gaussian posthoc.'])
        with (output / 'receipt.json').open('x') as handle: handle.write(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(dict(status='complete', counts=counts, evaluation_count=len(evaluations))), flush=True)
        return receipt
    except Exception as error:
        with (output / 'incident.json').open('x') as handle: handle.write(json.dumps(dict(error=repr(error), counts=counts, time=time.time()), indent=2) + '\n')
        raise
    finally:
        for environment in (env, eval_env):
            if environment is not None: environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output_dir)


if __name__ == '__main__':
    main()
