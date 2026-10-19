"""Source-bound single-environment GAC, with explicit online/prior protocols.

This adapts the official learner to local OGBench data and evaluation. Every
plan names its replay protocol; the online clock remains zero-based upstream.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch

from agents.gac_reference import GACReferenceAgent, GACReferenceConfig

FIELDS = ('observations', 'actions', 'rewards', 'dones', 'next_observations')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def expected_updates(budget, config):
    last = budget - 1
    critic = max(0, last - config.learning_starts)
    actor = max(0, last // config.policy_frequency - config.learning_starts // config.policy_frequency) * config.policy_frequency if critic else 0
    target = max(0, last // config.target_network_frequency - config.learning_starts // config.target_network_frequency) if critic else 0
    return dict(critic=critic, actor=actor, target=target)


def validate_plan(plan):
    fields = {'seed', 'env_name', 'replay_protocol', 'dataset_path', 'dataset_sha256', 'validation_sha256',
              'online_steps', 'replay_capacity', 'eval_seed', 'eval_interval', 'eval_episodes', 'log_interval',
              'learner', 'device', 'torch_version', 'environment_versions', 'source_commit', 'source_files'}
    if set(plan) != fields:
        raise ValueError('Unexpected or missing GAC plan fields: ' + str(set(plan) ^ fields))
    if plan['replay_protocol'] not in ('online_only', 'prior_balanced') or plan['device'] not in ('cpu', 'cuda'):
        raise ValueError('Unknown replay protocol or backend')
    if not plan['env_name'].startswith('cube-') or '-singletask-task' not in plan['env_name']:
        raise ValueError('Current driver requires an explicit OGBench Cube task')
    for name in ('seed', 'eval_seed'):
        if type(plan[name]) is not int or not 0 <= plan[name] < 2**31:
            raise ValueError('Invalid seed: ' + name)
    for name in ('online_steps', 'replay_capacity', 'eval_interval', 'eval_episodes', 'log_interval'):
        if type(plan[name]) is not int or plan[name] < 1:
            raise ValueError('Expected positive integer: ' + name)
    if set(plan['learner']) - set(asdict(GACReferenceConfig())):
        raise ValueError('Unknown GAC learner settings')
    config = GACReferenceConfig(**plan['learner']); config.validate()
    if config.radius != 1:
        raise ValueError('Cube reference protocol uses the official radius=1 full-action ball')
    if plan['replay_protocol'] == 'prior_balanced':
        if config.batch_size % 2 or not all(plan[k] for k in ('dataset_path', 'dataset_sha256', 'validation_sha256')):
            raise ValueError('Balanced prior requires even batch and explicit dataset identity')
        expected_name = plan['env_name'].split('-singletask')[0] + '-v0.npz'
        if Path(plan['dataset_path']).name != expected_name:
            raise ValueError('Prior dataset family differs from the requested environment')
    elif any(plan[k] is not None for k in ('dataset_path', 'dataset_sha256', 'validation_sha256')):
        raise ValueError('Online-only protocol must not declare or load prior data')
    required = {'main_gac_reference.py', 'agents/__init__.py', 'agents/gac_reference.py', 'envs/ogbench_utils.py'}
    if not required <= set(plan['source_files']):
        raise ValueError('Missing source provenance')
    if len(plan['source_commit']) != 40 or any(c not in '0123456789abcdef' for c in plan['source_commit']):
        raise ValueError('Expected a pinned source commit')
    if set(plan['environment_versions']) != {'ogbench', 'mujoco', 'gymnasium', 'numpy'}:
        raise ValueError('Pin all simulator/runtime package versions')
    return config


def verify_inputs(plan, source_root):
    for name, expected in plan['source_files'].items():
        path = (source_root / name).resolve(); path.relative_to(source_root.resolve())
        if digest(path) != expected:
            raise ValueError('Source changed: ' + name)
    if plan['replay_protocol'] == 'prior_balanced':
        path = Path(plan['dataset_path'])
        validation = path.with_name(path.stem + '-val.npz')
        if digest(path) != plan['dataset_sha256'] or digest(validation) != plan['validation_sha256']:
            raise ValueError('Prior input identity changed')


class OnlineReplay:
    """Raw, fixed-capacity transition ring; prior arrays never enter this ring."""
    def __init__(self, example, capacity):
        self.capacity, self.size, self.pointer = capacity, 0, 0
        self.data = {name: np.empty((capacity,) + np.asarray(example[name]).shape, dtype='float32') for name in FIELDS}

    def add(self, transition):
        for name in FIELDS:
            value = np.asarray(transition[name], dtype='float32')
            if value.shape != self.data[name].shape[1:] or not np.isfinite(value).all():
                raise ValueError('Invalid online transition: ' + name)
        for name in FIELDS: self.data[name][self.pointer] = transition[name]
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size, rng):
        if not self.size:
            raise ValueError('Cannot sample empty online replay')
        indices = rng.integers(self.size, size=size)
        return {name: value[indices] for name, value in self.data.items()}, indices


def sample_batch(replay, prior, batch_size, rng):
    online_size = batch_size if prior is None else batch_size // 2
    online, online_indices = replay.sample(online_size, rng)
    if prior is None:
        return online, dict(prior_rows=0, online_rows=online_size, prior_indices=np.empty(0, dtype='int64'), online_indices=online_indices)
    if batch_size % 2:
        raise ValueError('Balanced replay requires an even batch')
    prior_indices = rng.integers(len(prior['actions']), size=batch_size // 2)
    batch = {name: np.concatenate([prior[name][prior_indices], online[name]]) for name in FIELDS}
    return batch, dict(prior_rows=batch_size // 2, online_rows=online_size, prior_indices=prior_indices, online_indices=online_indices)


def evaluate_policy(agent, env, seed, episodes):
    before_cpu = agent.rng_cpu.clone()
    before_cuda = agent.rng_cuda.clone() if agent.rng_cuda is not None else None
    counters = (agent.critic_updates, agent.actor_updates, agent.target_updates)
    records = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done, length = False, 0
        while not done:
            action = agent.sample_actions(observation, seed=seed + episode * 1000000 + length)
            if not np.isfinite(action).all() or np.linalg.norm(action) > 1 + 1e-6:
                raise ValueError('Illegal full-vector GAC evaluation action')
            observation, _, terminated, truncated, info = env.step(action.copy())
            length += 1; done = bool(terminated or truncated)
        success = float(info['success'])
        if success not in (0., 1.):
            raise ValueError('Expected observed binary terminal success')
        records.append(dict(episode=episode, reset_seed=seed + episode, length=length, success=int(success)))
    if not torch.equal(before_cpu, agent.rng_cpu) or before_cuda is not None and not torch.equal(before_cuda, agent.rng_cuda):
        raise ValueError('Evaluation advanced learner random stream')
    if counters != (agent.critic_updates, agent.actor_updates, agent.target_updates):
        raise ValueError('Evaluation updated the learner')
    return dict(episode_count=len(records), success_count=sum(row['success'] for row in records),
                success=float(np.mean([row['success'] for row in records])), episodes=records)


def compare_checkpoint(left, right):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right): raise ValueError('Checkpoint keys differ')
        return sum(compare_checkpoint(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        if type(left) != type(right) or len(left) != len(right): raise ValueError('Checkpoint sequence differs')
        return sum(compare_checkpoint(a, b) for a, b in zip(left, right))
    if torch.is_tensor(left):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left.cpu(), right.cpu()):
            raise ValueError('Checkpoint tensor differs')
    elif left != right:
        raise ValueError('Checkpoint scalar differs')
    return 1


def initial_state_digest(agent):
    value = hashlib.sha256()
    state = agent.state_dict()
    for name, model in sorted(state['models'].items()):
        for key, tensor in sorted(model.items()):
            array = tensor.cpu().numpy()
            value.update(str((name, key, array.shape, str(array.dtype))).encode())
            value.update(array.tobytes())
    value.update(state['rng_cpu'].numpy().tobytes())
    if state['rng_cuda'] is not None: value.update(state['rng_cuda'].numpy().tobytes())
    return value.hexdigest()


def run(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text()); config = validate_plan(plan)
    plan_sha = digest(plan_path); source_root = Path(__file__).resolve().parent
    verify_inputs(plan, source_root)
    versions = {name: importlib.metadata.version(name) for name in plan['environment_versions']}
    if versions != plan['environment_versions'] or torch.__version__ != plan['torch_version']:
        raise ValueError('Pinned package/runtime versions differ')
    if plan['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Explicit CUDA backend is unavailable; no CPU fallback')
    output.mkdir(parents=True, exist_ok=False)
    started = time.time(); env = eval_env = None
    counts = dict(environment_transitions=0, sampled_prior_rows=0, sampled_online_rows=0, sampling_events=0)
    try:
        if plan['replay_protocol'] == 'prior_balanced':
            from envs.ogbench_utils import make_ogbench_env_and_datasets
            env, eval_env, raw, _ = make_ogbench_env_and_datasets(plan['env_name'], dataset_path=plan['dataset_path'], compact_dataset=False)
            prior = {name: np.asarray(raw[name], dtype='float32') for name in ('observations', 'actions', 'next_observations')}
            prior['rewards'] = np.asarray(raw['rewards'], dtype='float32').reshape(-1, 1)
            prior['dones'] = (1 - np.asarray(raw['masks'], dtype='float32')).reshape(-1, 1)
            for value in prior.values(): value.setflags(write=False)
        else:
            import ogbench
            env = ogbench.make_env_and_datasets(plan['env_name'], env_only=True)
            eval_env = ogbench.make_env_and_datasets(plan['env_name'], env_only=True)
            prior = None
        observation, _ = env.reset(seed=plan['seed'])
        low, high = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
        if not np.array_equal(low, -np.ones(5)) or not np.array_equal(high, np.ones(5)):
            raise ValueError('Expected normalized five-dimensional Cube actions')
        agent = GACReferenceAgent(plan['seed'], len(observation), low, high, config, plan['device'])
        initial_sha = initial_state_digest(agent)
        effective = dict(learner=asdict(config), observation_dim=len(observation), action_dim=5,
                         replay_protocol=plan['replay_protocol'], prior_transitions=0 if prior is None else len(prior['actions']),
                         online_replay_capacity=plan['replay_capacity'], device=str(agent.device), source_commit=plan['source_commit'])
        (output / 'effective_config.json').write_text(json.dumps(effective, indent=2) + '\n')
        example = dict(observations=observation, actions=np.zeros(5), rewards=np.zeros(1), dones=np.zeros(1), next_observations=observation)
        replay = OnlineReplay(example, plan['replay_capacity'])
        warmup_rng, sample_rng = np.random.default_rng(plan['seed']), np.random.default_rng(plan['seed'] ^ 0x6A09E667)
        evaluations = []; first_update = True; observations = []; latest_actor_info = {}
        grid = sorted({0, plan['online_steps']} | set(range(plan['eval_interval'], plan['online_steps'] + 1, plan['eval_interval'])))
        with (output / 'eval.jsonl').open('x') as eval_log, (output / 'train.jsonl').open('x') as train_log, (output / 'actions.jsonl').open('x') as action_log:
            def evaluation(step):
                row = dict(step=step, **evaluate_policy(agent, eval_env, plan['eval_seed'], plan['eval_episodes']))
                evaluations.append(row); eval_log.write(json.dumps(row, allow_nan=False) + '\n'); eval_log.flush()
            evaluation(0)
            for global_step in range(plan['online_steps']):
                if len(observations) < 16: observations.append(np.asarray(observation).copy())
                warmup = global_step < config.learning_starts
                action = warmup_rng.uniform(low, high).astype('float32') if warmup else agent.sample_actions(observation)
                if not np.isfinite(action).all() or np.max(np.abs(action)) > 1 + 1e-6 or not warmup and np.linalg.norm(action) > 1 + 1e-6:
                    raise ValueError('Illegal collected GAC action')
                next_observation, reward, terminated, truncated, _ = env.step(action.copy())
                transition = dict(observations=np.asarray(observation), actions=action, rewards=np.asarray([reward]),
                                  dones=np.asarray([terminated]), next_observations=np.asarray(next_observation))
                replay.add(transition); counts['environment_transitions'] += 1
                step = global_step + 1
                action_log.write(json.dumps(dict(step=step, zero_based_loop=global_step, warmup=warmup, action=action.tolist(),
                    terminated=bool(terminated), truncated=bool(truncated)), allow_nan=False) + '\n')
                if global_step > config.learning_starts:
                    batch, sample = sample_batch(replay, prior, config.batch_size, sample_rng)
                    counts['sampled_prior_rows'] += sample['prior_rows']; counts['sampled_online_rows'] += sample['online_rows']; counts['sampling_events'] += 1
                    if first_update:
                        saved = dict(batch, prior_indices=sample['prior_indices'], online_indices=sample['online_indices'])
                        np.savez_compressed(output / 'first-update-batch.npz', **saved)
                        np.savez_compressed(output / 'first-update-online-replay.npz', **{k:v[:replay.size] for k,v in replay.data.items()})
                        first_update = False
                    info = agent.update(batch, global_step)
                    if not np.isfinite(list(info.values())).all(): raise ValueError('Nonfinite learner diagnostics')
                    if 'actor_loss' in info:
                        latest_actor_info = {name: info[name] for name in ('actor_loss', 'sac_actor_loss', 'kappa_reg_loss')}
                        latest_actor_info['actor_global_step'] = global_step
                    if step % plan['log_interval'] == 0 or step == plan['online_steps']:
                        row = dict(latest_actor_info); row.update(info)
                        row.update(step=step, global_step=global_step)
                        train_log.write(json.dumps(row, allow_nan=False) + '\n'); train_log.flush()
                if step in grid: evaluation(step)
                if terminated or truncated: observation, _ = env.reset()
                else: observation = next_observation
        expected = expected_updates(plan['online_steps'], config)
        actual = dict(critic=agent.critic_updates, actor=agent.actor_updates, target=agent.target_updates)
        if actual != expected: raise ValueError('Update event counts differ')
        for optimizer, key in ((agent.q_optimizer, 'critic'), (agent.actor_optimizer, 'actor')):
            steps = {int(value['step']) for value in optimizer.state.values()}
            if steps != ({expected[key]} if expected[key] else set()): raise ValueError('Actual Adam counters differ')
        expected_prior = expected['critic'] * (config.batch_size // 2 if prior is not None else 0)
        expected_online = expected['critic'] * config.batch_size - expected_prior
        if (counts['sampled_prior_rows'], counts['sampled_online_rows'], counts['sampling_events']) != (expected_prior, expected_online, expected['critic']):
            raise ValueError('Actual prior/online sampled-row counts differ')
        if [row['step'] for row in evaluations] != grid: raise ValueError('Incomplete evaluation grid')
        state = agent.state_dict()
        def finite(value):
            if isinstance(value, dict): return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
            return bool(torch.isfinite(value).all()) if torch.is_tensor(value) else True
        if not finite(state): raise ValueError('Nonfinite model or optimizer state')
        checkpoint = output / 'final.pt'; torch.save(state, checkpoint)
        restored = GACReferenceAgent(0, len(observation), low, high, config, plan['device'])
        restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
        fields = compare_checkpoint(state, restored.state_dict())
        probes = np.asarray(observations)
        np.testing.assert_array_equal(agent.sample_actions(probes, seed=20261101), restored.sample_actions(probes, seed=20261101))
        verify_inputs(plan, source_root)
        if digest(plan_path) != plan_sha: raise ValueError('Plan changed during run')
        receipt = dict(status='complete', source_commit=plan['source_commit'], source_files=plan['source_files'], plan_sha256=plan_sha,
            replay_protocol=plan['replay_protocol'], counts=counts, actual_updates=actual, evaluation_steps=grid,
            initial_model_and_rng_sha256=initial_sha,
            observed_eval_episode_counts=[row['episode_count'] for row in evaluations], exact_checkpoint_fields=fields,
            restored_policy_probes=len(probes), checkpoint_sha256=digest(checkpoint), effective_config_sha256=digest(output / 'effective_config.json'),
            runtime=dict(torch=torch.__version__, compiled_cuda=torch.version.cuda, device=str(agent.device), packages=versions),
            started=started, finished=time.time(), limitations=['Local single-environment driver; official defaults use eight vector environments.',
                'Prior-balanced replay is an explicit GAC adaptation, not the original online-only algorithm.',
                'Sampled-row counts include replacement and actor reuse of the same critic minibatch; they are not unique transition counts.',
                'Fixed evaluation seeds do not certify complete simulator cache/RNG matching.'])
        with (output / 'receipt.json').open('x') as stream: stream.write(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(dict(status='complete', actual_updates=actual, counts=counts)), flush=True)
        return receipt
    except Exception as error:
        with (output / 'incident.json').open('x') as stream: stream.write(json.dumps(dict(error=repr(error), counts=counts), indent=2) + '\n')
        raise
    finally:
        for environment in (env, eval_env):
            if environment is not None: environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1); run(args.plan, args.output_dir)


if __name__ == '__main__': main()
