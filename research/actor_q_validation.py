"""Source-bound actor-Q ablation validation and terminal-only paired summaries."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import pickle

import numpy as np

SOURCE = 'db861f3c1217c642d8138453630f8a2b8f275219'
PLAN_SHA = '0116e96c8ee634fd3567aab2caf59a22d2029f29e4a0ee336a6a7947e9ed737c'
COHORT = 'posthoc_actor_q_ablation_20261020'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def flags_from_argv(argv):
    flags = {}
    for item in argv[2:]:
        if not item.startswith('--') or '=' not in item:
            raise ValueError('Expected explicit name=value training arguments')
        key, value = item[2:].split('=', 1)
        if key in flags:
            raise ValueError('Duplicate training flag')
        flags[key] = value
    return flags


def verify_plan(plan):
    expected = list(itertools.product(('triple2', 'double2'), ('full', 'remove_direction_radial'), range(20270, 20280)))
    if (plan['cohort'] != COHORT or plan['training_source_commit'] != SOURCE or plan['expected_runs'] != 40
            or [(r['task'], r['method'], r['seed']) for r in plan['cases']] != expected
            or [r['index'] for r in plan['cases']] != list(range(40))):
        raise ValueError('Frozen forty-case contract differs')
    for row in plan['cases']:
        args = flags_from_argv(row['argv'])
        if (row['argv'][:2] != ['/workspace/venv/bin/python', 'main_online.py']
                or args['agent'] != 'agents/acrlpd.py' or args['seed'] != str(row['seed'])
                or args['env_name'] != row['env_name'] or args['run_group'] != row['run_group']
                or args['agent.adapter_actor_q_gradient'] != row['method']):
            raise ValueError('Frozen argv identity differs')
    for task, seed in itertools.product(plan['tasks'], plan['training_seeds']):
        pair = [r for r in plan['cases'] if r['task'] == task and r['seed'] == seed]
        cleaned = [{k: v for k, v in flags_from_argv(r['argv']).items()
                    if k not in ('agent.adapter_actor_q_gradient', 'run_group')} for r in pair]
        if len(pair) != 2 or cleaned[0] != cleaned[1]:
            raise ValueError('Pair changes more than its registered gradient path')
    return plan


def resolve_run(row):
    args = flags_from_argv(row['argv']); root = Path(args['save_dir'])
    paths = [p.parent for p in root.rglob('flags.json')
             if p.parent.name.startswith(f"sd{row['seed']:03d}")
             and read(p).get('run_group') == row['run_group']]
    if len(paths) != 1:
        raise ValueError('Expected exactly one run for the registered identity')
    return paths[0]


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    return bool(np.isfinite(np.asarray(value)).all())


def assert_state_equal(left, right):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError('Restored state keys differ')
        return sum(assert_state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        if type(left) != type(right) or len(left) != len(right):
            raise ValueError('Restored state sequence differs')
        return sum(assert_state_equal(a, b) for a, b in zip(left, right, strict=True))
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b) or not np.isfinite(a).all():
        raise ValueError('Restored state leaf differs or is nonfinite')
    return 1


def restore_checkpoint(path, config, seed, source_root=None):
    """Independent template restore, without loading an environment or updating."""
    import flax.serialization
    from agents.acrlpd import ACRLPDAgent
    if source_root is not None:
        import agents.acrlpd as learner
        if Path(learner.__file__).resolve() != Path(source_root).resolve()/'agents/acrlpd.py':
            raise ValueError('Restorer imported a different learner release')
    state = pickle.loads(Path(path).read_bytes())['agent']
    obs_dim = state['network']['params']['modules_actor']['MLP_0']['Dense_0']['kernel'].shape[0]
    template = ACRLPDAgent.create(seed, np.zeros(obs_dim, np.float32), np.zeros(6, np.float32), dict(config))
    restored = flax.serialization.from_state_dict(template, state)
    return assert_state_equal(state, flax.serialization.to_state_dict(restored))


def technical_validate(row, directory, initial, *, expected_device='4090', source_root=None):
    """Inspect counts/actions/state, while leaving success and reward uninterpreted."""
    directory = Path(directory); args = flags_from_argv(row['argv'])
    directory.resolve().relative_to(Path(args['save_dir']).resolve())
    flags, config = (read(directory/name) for name in ('flags.json', 'effective_agent_config.json'))
    for key, text in args.items():
        if key in ('save_dir', 'agent'):
            continue
        actual = flags
        for part in key.split('.'):
            actual = actual[part]
        expected = {'true': True, 'false': False}[text.lower()] if isinstance(actual, bool) else (
            float(text) if isinstance(actual, (int, float)) else text)
        if actual != expected:
            raise ValueError('Explicit argument differs: '+key)
    required = dict(agent_name='acrlpd', adapter_actor_q_gradient=row['method'], adapter_representation='cartesian',
        adapter_execution_map='direction_speed', adapter_critic_space='latent', adapter_speed_squash='tanh',
        adapter_speed_interpretation='log', ds_speed_bound='sphere', action_dim=6, policy_action_dim=6,
        critic_action_dim=6, env_action_dim=5, horizon_length=1, actor_hidden_dims=[512]*4,
        value_hidden_dims=[512]*4, num_qs=10, lr=.0003, batch_size=256, q_agg='mean', discount=.99,
        tau=.005, target_entropy=-3., bc_alpha=0., fixed_alpha=-1., init_temp=1., action_chunking=True,
        layer_norm=True, actor_layer_norm=False, actor_update_mode='sac', ds_regularizer='policy_coordinate',
        ds_phase_regularization='none', ds_stage_speed_constraint='none', ds_moment_controlled=False,
        ds_separate_alpha=False, use_integrated_ds=False, use_cartesian_l2_projection=False)
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Effective learner recipe differs')
    if (initial['source_commit'] != SOURCE or initial['seed'] != row['seed'] or initial['method'] != row['method']
            or initial['case_index'] != row['index']
            or initial['network_step'] != 1 or initial['backend'] != 'gpu'
            or expected_device not in initial['device_kind'] or len(initial['serialized_agent_sha256']) != 64
            or initial['training_flags'] != row['argv'][2:]):
        raise ValueError('Actual driver initialization evidence differs')
    budget, start, interval, episodes = [int(args[k]) for k in ('online_steps', 'start_training', 'eval_interval', 'eval_episodes')]
    evaluations = list(csv.DictReader((directory/'eval.csv').open()))
    grid = list(range(0, budget+1, interval))
    if ([int(r['step']) for r in evaluations] != grid
            or any(int(r['evaluation_episodes_observed']) != episodes for r in evaluations)):
        raise ValueError('Evaluation grid or actual episode counts differ')
    warmup = []; count = 0
    with (directory/'env.csv').open() as stream:
        for row_env in csv.DictReader(stream):
            count += 1
            action = np.array([float(row_env[f'action_{i}']) for i in range(5)], dtype=np.float32)
            # Legacy sphere is a radial reference, not a hard unit-ball support:
            # tanh/log uses exp(speed), then clips each executed component.
            if int(row_env['step']) != count or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.00001:
                raise ValueError('Executed-action stream differs')
            if count <= start:
                warmup.append(action)
    if count != budget or len(warmup) != min(start, budget):
        raise ValueError('Collection budget or warmup count differs')
    training = list(csv.DictReader((directory/'online_agent.csv').open()))
    if not training or int(training[-1]['step']) != budget or not all(
            np.isfinite(float(v)) for r in training for v in r.values() if v):
        raise ValueError('Missing/nonfinite final learner diagnostics')
    checkpoint = directory/f'params_{budget}.pkl'; state = pickle.loads(checkpoint.read_bytes())['agent']['network']
    updates = max(0, budget-max(1, start)+1)*int(args['utd_ratio'])
    if int(state['step']) != updates+1 or int(state['opt_state']['0']['count']) != updates or not _finite_tree(state):
        raise ValueError('Checkpoint counters or finite state differ')
    leaves = restore_checkpoint(checkpoint, config, row['seed'], source_root)
    hashes = {name: digest(directory/name) for name in ('flags.json', 'effective_agent_config.json',
              'env.csv', 'online_agent.csv', 'eval.csv', checkpoint.name)}
    return dict(status='complete', run_dir=str(directory), source_commit=SOURCE, seed=row['seed'],
        task=row['task'], method=row['method'], index=row['index'], identity_hashes=hashes,
        initial_serialized_agent_sha256=initial['serialized_agent_sha256'],
        warmup_actions_sha256=hashlib.sha256(np.asarray(warmup, np.float32).tobytes()).hexdigest(),
        actual_environment_steps=budget, actual_joint_optimizer_updates=updates, network_step=updates+1,
        restored_state_leaves=leaves, observed_eval_episode_counts=[episodes]*len(grid),
        effective_config=config, outcomes_inspected=False)


def outcome_metrics(record, row):
    directory = Path(record['run_dir'])
    for name, sha in record['identity_hashes'].items():
        if digest(directory/name) != sha:
            raise ValueError('Terminal run artifact changed')
    args = flags_from_argv(row['argv']); budget, interval, episodes = [int(args[k]) for k in ('online_steps', 'eval_interval', 'eval_episodes')]
    rows = list(csv.DictReader((directory/'eval.csv').open())); steps = [int(r['step']) for r in rows]
    rates = np.array([float(r['success']) for r in rows])
    if steps != list(range(0, budget+1, interval)) or not np.isfinite(rates).all() or np.any((rates < 0)|(rates > 1)):
        raise ValueError('Terminal success grid differs')
    for row_eval, rate in zip(rows, rates, strict=True):
        count, successes = float(row_eval['evaluation_episodes_observed']), float(row_eval['evaluation_successes_observed'])
        if count != episodes or not successes.is_integer() or not 0 <= successes <= count or abs(rate-successes/count) > 1e-7:
            raise ValueError('Observed success counts differ')
    reached = [s for s, r in zip(steps, rates, strict=True) if r >= .5]
    return dict(success=rates.tolist(), eval_steps=steps, auc_percentage=float(np.trapezoid(rates, steps)/budget*100),
        endpoint_percentage=float(rates[-1]*100), tail3_percentage=float(rates[-3:].mean()*100),
        time_to_50=reached[0] if reached else None, time_to_50_censored=not bool(reached))


def collect(plan, records, outcome=outcome_metrics):
    verify_plan(plan); by_index = {r['index']: r for r in records}
    if len(by_index) != len(records) or set(by_index) != set(range(40)):
        raise ValueError('Missing/duplicate terminal receipt identities')
    for case in plan['cases']:
        r = by_index[case['index']]
        if any(r[k] != case[k] for k in ('task', 'method', 'seed')) or r['source_commit'] != SOURCE:
            raise ValueError('Receipt identity differs')
    released = all(type(r.get('returncode')) is int and r.get('driver_exit_observed') is True
        and (r['returncode'] != 0 or type(r.get('validation_returncode')) is int) for r in records)
    result = []
    for case in plan['cases']:
        record = dict(by_index[case['index']])
        if released and record['status'] == 'complete':
            try:
                if record['returncode'] != 0 or type(record.get('validation_returncode')) is not int or record['validation_returncode'] != 0:
                    raise ValueError('Complete receipt lacks successful actual exits')
                record.update(outcome(record, case))
            except Exception as error:
                record.update(status='needs_review', outcome_error=str(error))
        result.append(record)
    comparisons = []; rng = np.random.default_rng(plan['primary_analysis']['interval']['seed'])
    for task in plan['tasks']:
        pairs = []
        for seed in plan['training_seeds']:
            pair = {r['method']: r for r in result if r['task'] == task and r['seed'] == seed}
            a, b = pair['remove_direction_radial'], pair['full']; valid = released and all(
                r['status']=='complete' and 'auc_percentage' in r for r in (a,b))
            if valid:
                ac, bc = dict(a['effective_config']), dict(b['effective_config'])
                ac.pop('adapter_actor_q_gradient'); bc.pop('adapter_actor_q_gradient')
                if (ac != bc or a['initial_serialized_agent_sha256'] != b['initial_serialized_agent_sha256']
                        or a['warmup_actions_sha256'] != b['warmup_actions_sha256']):
                    valid = False
            pairs.append(dict(seed=seed, status='complete' if valid else 'unavailable',
                auc_difference=a['auc_percentage']-b['auc_percentage'] if valid else None,
                secondary={k: dict(cut=a[k], full=b[k]) for k in ('endpoint_percentage','tail3_percentage','time_to_50','time_to_50_censored')} if valid else None))
        summary = dict(task=task, pairs=pairs, complete_pairs=sum(r['status']=='complete' for r in pairs))
        if summary['complete_pairs'] == 10:
            values = np.array([r['auc_difference'] for r in pairs]); interval = plan['primary_analysis']['interval']
            draws = values[rng.integers(0,10,size=(interval['draws'],10))].mean(axis=1)
            tail = (1-interval['coverage'])/2
            summary.update(mean=float(values.mean()), median=float(np.median(values)),
                positive_pairs=int((values>0).sum()), zero_pairs=int((values==0).sum()), negative_pairs=int((values<0).sum()),
                descriptive_bootstrap_interval=np.quantile(draws,[tail,1-tail]).tolist(), coverage=interval['coverage'])
        comparisons.append(summary)
    complete = sum(r['status']=='complete' for r in result)
    return dict(status='complete' if released and complete==40 and all(r['complete_pairs']==10 for r in comparisons) else 'needs_review',
        source_commit=SOURCE, expected_runs=40, complete_runs=complete, outcomes_released=released,
        records=result, comparisons=comparisons, interpretation=plan['scope'])
