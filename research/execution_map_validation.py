"""Validate matched execution-map runs using the driver's actual initialization.

This does not read success or reward. It establishes agent-state and command
prefix equality, not simulator-state or executed-distribution equality.
"""
import csv
import hashlib
from pathlib import Path
import pickle
import re

import numpy as np

from research import actor_q_validation as common

MAPS = ('direction_speed', 'cartesian_aux', 'cartesian_projection_aux')
TASKS = {
    'triple2': 'cube-triple-play-singletask-task2-v0',
    'double2': 'cube-double-play-singletask-task2-v0',
}


def validate_identity(case, initial, config, *, source_binding_sha, expected_device):
    args = common.flags_from_argv(case['argv'])
    mapping = case['execution_map']
    if (mapping not in MAPS or case['task'] not in TASKS or case['method'] != 'full'
            or args['agent'] != 'agents/acrlpd.py'
            or args['seed'] != str(case['seed'])
            or args['env_name'] != TASKS[case['task']]
            or args['run_group'] != case['run_group']
            or args['ds_adapter_execution_map'] != mapping
            or args['ds_adapter_critic_space'] != 'env'
            or args['agent.adapter_actor_q_gradient'] != 'full'):
        raise ValueError('Execution-map case identity differs')
    required = dict(agent_name='acrlpd', adapter_actor_q_gradient='full',
        adapter_representation='cartesian', adapter_execution_map=mapping,
        adapter_critic_space='env', adapter_speed_squash='tanh',
        adapter_speed_interpretation='log', ds_speed_bound='sphere',
        action_dim=6, policy_action_dim=6, critic_action_dim=5, env_action_dim=5,
        horizon_length=1, actor_hidden_dims=[512]*4, value_hidden_dims=[512]*4,
        num_qs=10, lr=.0003, batch_size=256, q_agg='mean', discount=.99,
        tau=.005, target_entropy=-3., bc_alpha=0., fixed_alpha=-1., init_temp=1.,
        action_chunking=True, layer_norm=True, actor_layer_norm=False,
        actor_update_mode='sac', ds_regularizer='policy_coordinate',
        ds_phase_regularization='none', ds_stage_speed_constraint='none',
        ds_moment_controlled=False, ds_separate_alpha=False,
        ds_separate_actor_trunks=False, use_integrated_ds=False,
        use_cartesian_l2_projection=False)
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Matched execution-map recipe differs')
    if (initial['source_commit'] != common.SOURCE
            or initial['source_binding_sha256'] != source_binding_sha
            or initial['seed'] != case['seed'] or initial['method'] != case['method']
            or initial['case_index'] != case['index'] or initial['network_step'] != 1
            or initial['backend'] != 'gpu' or expected_device not in initial['device_kind']
            or re.fullmatch('[0-9a-f]{64}', initial['serialized_agent_sha256']) is None
            or initial['training_flags'] != case['argv'][2:]):
        raise ValueError('Actual constructor observation differs')
    return args


def validate_run(case, initial, source, binding, *, expected_device='4090'):
    """Check a single observed run; independently restore the saved learner."""
    from research.actor_q_entry import verify_source
    verify_source(source, binding)
    directory = common.resolve_run(case)
    config = common.read(directory/'effective_agent_config.json')
    args = validate_identity(case, initial, config,
        source_binding_sha=common.digest(binding), expected_device=expected_device)
    flags = common.read(directory/'flags.json')
    for key, text in args.items():
        if key in ('save_dir', 'agent'):
            continue
        actual = flags
        for part in key.split('.'):
            actual = actual[part]
        expected = ({'true': True, 'false': False}[text.lower()] if isinstance(actual, bool)
            else float(text) if isinstance(actual, (int, float)) else text)
        if actual != expected:
            raise ValueError('Recorded training argument differs: '+key)
    budget, start, interval, episodes = [int(args[k]) for k in
        ('online_steps', 'start_training', 'eval_interval', 'eval_episodes')]
    if (budget < start or start < 1 or budget % interval or
            args['utd_ratio'] != '1' or args['eval_at_start'].lower() != 'true'):
        raise ValueError('Unsupported execution-map budget or update schedule')
    with (directory/'eval.csv').open() as stream:
        evaluations = list(csv.DictReader(stream))
    grid = list(range(0, budget+1, interval))
    # Deliberately do not interpret success/reward, including partial outcomes.
    if ([int(r['step']) for r in evaluations] != grid or
            any(int(r['evaluation_episodes_observed']) != episodes for r in evaluations)):
        raise ValueError('Evaluation grid or actual episode count differs')
    warmup, count = [], 0
    with (directory/'env.csv').open() as stream:
        for row in csv.DictReader(stream):
            count += 1
            action = np.array([float(row[f'action_{i}']) for i in range(5)], np.float32)
            norm = np.linalg.norm(action[:3])
            if (int(row['step']) != count or not np.isfinite(action).all()
                    or np.max(np.abs(action)) > 1.00001
                    or not np.isclose(float(row['action_xyz_norm']), norm, rtol=2e-6, atol=2e-6)):
                raise ValueError('Executed command stream differs')
            if count <= start:
                warmup.append(action)
            elif case['execution_map'] == 'cartesian_projection_aux' and norm > 1.00001:
                raise ValueError('Projected policy action exceeds unit ball')
    if count != budget or len(warmup) != start:
        raise ValueError('Collection budget or warmup count differs')
    with (directory/'online_agent.csv').open() as stream:
        training = list(csv.DictReader(stream))
    if (not training or int(training[-1]['step']) != budget or
            not all(np.isfinite(float(v)) for row in training for v in row.values() if v)):
        raise ValueError('Final learner diagnostics missing or nonfinite')
    checkpoint = directory/f'params_{budget}.pkl'
    state = pickle.loads(checkpoint.read_bytes())['agent']['network']
    updates = budget-start+1
    if (int(state['step']) != updates+1 or int(state['opt_state']['0']['count']) != updates
            or not common._finite_tree(state)):
        raise ValueError('Checkpoint update counters or finite state differs')
    leaves = restore_environment_checkpoint(checkpoint, config, case['seed'], source)
    return dict(status='complete', source_commit=common.SOURCE,
        task=case['task'], seed=case['seed'], execution_map=case['execution_map'],
        run_dir=str(directory), actual_environment_steps=budget,
        actual_joint_optimizer_updates=updates, network_step=updates+1,
        actual_initial_serialized_agent_sha256=initial['serialized_agent_sha256'],
        initial_receipt_kind='actual_constructor_capture_before_initial_evaluation',
        warmup_actions_sha256=hashlib.sha256(np.asarray(warmup, np.float32).tobytes()).hexdigest(),
        observed_eval_episode_counts=[episodes]*len(grid), eval_steps=grid,
        restored_state_leaves=leaves, expected_device=expected_device,
        config_sha256=common.digest(directory/'effective_agent_config.json'),
        checkpoint_sha256=common.digest(checkpoint), outcomes_inspected=False)


def restore_environment_checkpoint(path, config, seed, source):
    """Restore through the same five-dimensional replay interface as this driver."""
    import flax.serialization
    import agents.acrlpd as learner
    if Path(learner.__file__).resolve() != Path(source).resolve()/'agents/acrlpd.py':
        raise ValueError('Restorer imported a different learner release')
    state = pickle.loads(Path(path).read_bytes())['agent']
    obs_dim = state['network']['params']['modules_actor']['MLP_0']['Dense_0']['kernel'].shape[0]
    template = learner.ACRLPDAgent.create(seed, np.zeros(obs_dim, np.float32),
        np.zeros(5, np.float32), dict(config))
    restored = flax.serialization.from_state_dict(template, state)
    return common.assert_state_equal(state, flax.serialization.to_state_dict(restored))


def validate_groups(records):
    """Require the entire six-case engineering matrix before claiming equality."""
    identities = [(r['task'], r['execution_map']) for r in records]
    expected = [(task, mapping) for task in TASKS for mapping in MAPS]
    if len(records) != 6 or sorted(identities) != sorted(expected):
        raise ValueError('Missing/duplicate matched-map engineering case')
    for task in TASKS:
        group = [r for r in records if r['task'] == task]
        if (any(r['status'] != 'complete' or r.get('driver_exit_observed') is not True
                or type(r.get('driver_returncode')) is not int or r['driver_returncode'] != 0
                or type(r.get('validation_returncode')) is not int or r['validation_returncode'] != 0
                or r.get('outcomes_inspected') is not False
                or r.get('initial_receipt_kind') != 'actual_constructor_capture_before_initial_evaluation'
                or r.get('source_commit') != common.SOURCE
                or re.fullmatch('[0-9a-f]{64}', r['actual_initial_serialized_agent_sha256']) is None
                or re.fullmatch('[0-9a-f]{64}', r['warmup_actions_sha256']) is None for r in group)
                or len({r['seed'] for r in group}) != 1
                or len({r['actual_initial_serialized_agent_sha256'] for r in group}) != 1
                or len({r['warmup_actions_sha256'] for r in group}) != 1):
            raise ValueError('Observed process, initialization or warmup equality not proved')
    return dict(status='all_six_actual_initializations_verified', records=records,
        source_commit=common.SOURCE, cloud_4090_validation=False,
        scientific_training_started=False, initial_execution_distribution_matched=False,
        simulator_state_matched=False,
        limitation='Actual serialized agent and warmup commands only; local GPU engineering, no learning-effect estimate.')
