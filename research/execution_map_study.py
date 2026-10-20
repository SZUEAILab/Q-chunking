"""Frozen 60-run matched-map contract and terminal-only descriptive analysis.

The comparison is conditional on a common env critic and latent entropy. It
does not isolate an amplitude head or certify the original latent-Q recipe.
"""
from copy import deepcopy
import itertools
from pathlib import Path
import re

import numpy as np

from research import execution_map_validation as technical

common = technical.common
COHORT = 'posthoc_matched_map_study_20261020'
PLAN_SHA = '0b03d10bafadb9fa3431194fe3c56f6804604d3e8be4aefd9921e39e84ca5f82'
SOURCE_BINDING_SHA = 'adc09f52635ab6ab1ebcbe707d28e8435e91072cc4ad051eee6158b23cf33622'
SEEDS = tuple(range(20290, 20300))
TASKS = tuple(technical.TASKS)
MAPS = technical.MAPS
GRID = list(range(0, 1000001, 100000))
CONFIG_EXCEPTIONS = {'adapter_execution_map', 'action_pipeline_components'}
OUTCOME_FIELDS = {'metrics', 'success', 'reward', 'auc', 'auc_percentage',
    'endpoint', 'endpoint_percentage', 'tail3', 'tail3_percentage',
    'time_to_50', 'time_to_50_censored'}


def load_plan(path):
    if common.digest(path) != PLAN_SHA:
        raise ValueError('Frozen matched-map plan changed')
    return verify_plan(common.read(path))


def verify_plan(plan):
    expected = list(itertools.product(TASKS, MAPS, SEEDS))
    if (plan['cohort'] != COHORT or plan['expected_runs'] != 60
            or plan['training_source_commit'] != common.SOURCE
            or plan['source_binding_sha256'] != SOURCE_BINDING_SHA
            or plan['tasks'] != list(TASKS) or plan['maps'] != list(MAPS)
            or plan['training_seeds'] != list(SEEDS)
            or [(r['task'], r['execution_map'], r['seed']) for r in plan['cases']] != expected
            or [r['index'] for r in plan['cases']] != list(range(60))):
        raise ValueError('Frozen sixty-case identity contract differs')
    recipe = plan['learning_recipe']
    if (recipe['action_dim'] != 6 or recipe['critic_action_dim'] != 5
            or recipe['env_action_dim'] != 5 or recipe['target_entropy'] != -3.
            or recipe['utd_ratio'] != 1 or recipe['num_qs'] != 10
            or recipe['backup_entropy'] is not False
            or plan['required_effective_updates'] != 995001
            or plan['required_network_step'] != 995002):
        raise ValueError('Frozen learner metadata differs from the env-critic contract')
    fixed = dict(agent='agents/acrlpd.py', **{
        'agent.adapter_actor_q_gradient': 'full', 'agent.target_entropy': '-3.0',
        'agent.num_qs': '10', 'agent.q_agg': 'mean', 'agent.lr': '0.0003',
        'agent.batch_size': '256', 'agent.discount': '0.99', 'agent.tau': '0.005',
        'agent.bc_alpha': '0.0', 'agent.fixed_alpha': '-1.0'},
        horizon_length='1', online_steps='1000000', start_training='5000',
        buffer_size='1000000', utd_ratio='1', eval_at_start='True',
        eval_interval='100000', eval_episodes='50', eval_seed='20261027',
        video_episodes='0', save_interval='1000000', log_interval='5000',
        dataset_proportion='1.0', sparse='False', actor_update_mode='sac',
        stage_balanced_sampling='False', ds_mode='adapter',
        ds_adapter_representation='cartesian', ds_adapter_speed_squash='tanh',
        ds_adapter_speed_interpretation='log', ds_adapter_critic_space='env',
        ds_speed_bound='sphere', ds_regularizer='policy_coordinate',
        save_dir='/workspace/results/'+COHORT+'/science')
    for row in plan['cases']:
        args = common.flags_from_argv(row['argv'])
        expected_args = dict(fixed, env_name=technical.TASKS[row['task']], seed=str(row['seed']),
            run_group='amm_'+row['task']+'_'+row['execution_map']+'_20261020',
            ds_adapter_execution_map=row['execution_map'])
        if (row['argv'][:2] != ['/workspace/venv/bin/python', 'main_online.py']
                or row['method'] != 'full' or row['env_name'] != expected_args['env_name']
                or row['run_group'] != expected_args['run_group'] or args != expected_args):
            raise ValueError('Frozen argv changes a factor outside the execution map')
    primary = [dict(task=t, left='direction_speed', right=m) for t in TASKS for m in MAPS[1:]]
    secondary = [dict(task=t, left=MAPS[2], right=MAPS[1]) for t in TASKS]
    interval = plan['primary_analysis']['interval']
    if (plan['primary_analysis']['comparisons'] != primary
            or plan['secondary_analysis']['comparisons'] != secondary
            or plan['primary_analysis']['complete_pairs_required_per_comparison'] != 10
            or plan['primary_analysis']['complete_three_map_panel_required_per_task'] is not True
            or interval['draws'] != 20000 or interval['seed'] != 20261028
            or interval['coverage'] != .9875
            or plan['evaluation']['steps'] != GRID
            or plan['evaluation']['episodes_per_point'] != 50
            or plan['evaluation']['eval_seed'] != 20261027
            or plan['integrity']['read_outcomes_only_after_all_60_terminal'] is not True):
        raise ValueError('Frozen comparison, evaluation or blinding rule differs')
    return plan


def _technical_record(case, record):
    if (record['status'] != 'complete' or record['returncode'] != 0
            or record['validation_returncode'] != 0
            or record.get('actual_environment_steps') != 1000000
            or record.get('actual_joint_optimizer_updates') != 995001
            or record.get('network_step') != 995002
            or record.get('restored_state_leaves') != 150
            or record.get('observed_eval_episode_counts') != [50]*11
            or record.get('eval_steps') != GRID
            or record.get('source_binding_sha256') != SOURCE_BINDING_SHA
            or record.get('initial_receipt_kind') != 'actual_constructor_capture_before_initial_evaluation'):
        raise ValueError('Technical completion evidence differs')
    for key in ('actual_initial_serialized_agent_sha256', 'warmup_actions_sha256'):
        if re.fullmatch('[0-9a-f]{64}', record.get(key, '')) is None:
            raise ValueError('Missing actual initialization/warmup digest')
    config = record['effective_config']
    required = dict(action_dim=6, policy_action_dim=6, critic_action_dim=5, env_action_dim=5,
        adapter_execution_map=case['execution_map'], adapter_critic_space='env',
        adapter_actor_q_gradient='full', target_entropy=-3.)
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Effective matched-map interface differs')


def verify_terminal_receipts(records):
    """Check every observed wait receipt before reading any scientific result."""
    paths = set()
    for record in records:
        if (not isinstance(record.get('case_sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', record['case_sha256']) is None):
            raise ValueError('Missing scientific case binding')
        kinds = ['driver']+(['validator'] if record['returncode'] == 0 else [])
        for kind in kinds:
            path = Path(record[kind+'_exit_path']).resolve()
            if path in paths:
                raise ValueError('A process exit receipt is reused across scientific cases')
            paths.add(path)
            if common.digest(path) != record[kind+'_exit_sha256']:
                raise ValueError('Observed process receipt changed')
            value = common.read(path)
            expected = record['returncode' if kind == 'driver' else 'validation_returncode']
            if (type(value.get('returncode')) is not int or value['returncode'] != expected
                    or not np.isfinite([value['started'], value['ended']]).all()
                    or value['ended'] < value['started']
                    or type(value.get('case_index')) is not int
                    or value.get('case_index') != record['index']
                    or value.get('case_sha256') != record['case_sha256']
                    or value.get('source_commit') != common.SOURCE
                    or type(value.get('pid')) is not int or value['pid'] <= 0
                    or type(value.get('observed_by_pid')) is not int or value['observed_by_pid'] <= 0):
                raise ValueError('Actual source-bound process exit differs')


def observed_outcome(record, case):
    """Read outcomes only after verifying all bound files and process receipts."""
    directory = Path(record['run_dir'])
    required = {'flags.json', 'effective_agent_config.json', 'env.csv',
                'online_agent.csv', 'eval.csv', 'params_1000000.pkl'}
    if (set(record['identity_hashes']) != required
            or common.read(directory/'effective_agent_config.json') != record['effective_config']):
        raise ValueError('Bound outcome inputs differ')
    for name, digest in record['identity_hashes'].items():
        if common.digest(directory/name) != digest:
            raise ValueError('Terminal artifact changed: '+name)
    initial_path = Path(record['initial_receipt_path'])
    if common.digest(initial_path) != record['initial_receipt_sha256']:
        raise ValueError('Actual initialization receipt changed')
    initial = common.read(initial_path)
    technical.validate_identity(case, initial, record['effective_config'],
        source_binding_sha=SOURCE_BINDING_SHA, expected_device='4090')
    if initial['serialized_agent_sha256'] != record['actual_initial_serialized_agent_sha256']:
        raise ValueError('Actual initialization digest differs')
    verify_terminal_receipts([record])
    return common.outcome_metrics(record, case)


def _check_metrics(metrics):
    rates = np.asarray(metrics['success'], dtype=float)
    if (metrics['eval_steps'] != GRID or rates.shape != (11,) or not np.isfinite(rates).all()
            or np.any((rates < 0) | (rates > 1))):
        raise ValueError('Success curve has a wrong grid or invalid rate')
    expected = dict(auc_percentage=float(np.trapezoid(rates, GRID)/1000000*100),
        endpoint_percentage=float(rates[-1]*100), tail3_percentage=float(rates[-3:].mean()*100))
    if any(not np.isclose(metrics[k], v, rtol=0, atol=1e-7) for k, v in expected.items()):
        raise ValueError('Reported metric disagrees with the actual curve')
    reached = [s for s, r in zip(GRID, rates, strict=True) if r >= .5]
    if (metrics['time_to_50'] != (reached[0] if reached else None)
            or metrics['time_to_50_censored'] is not (not bool(reached))):
        raise ValueError('Time-to-threshold or censoring differs')


def collect(plan, records, outcome=observed_outcome, *, terminal_check=None):
    """Retain all identities; require all exits before any outcome callback."""
    verify_plan(plan)
    indexed = {r['index']: r for r in records}
    if len(records) != 60 or len(indexed) != 60 or set(indexed) != set(range(60)):
        raise ValueError('Missing/duplicate sixty-case receipt identities')
    for case in plan['cases']:
        row = indexed[case['index']]
        if (any(row[k] != case[k] for k in ('task', 'seed', 'execution_map'))
                or row['source_commit'] != common.SOURCE):
            raise ValueError('Terminal receipt belongs to a different scientific case')
    released = all(r.get('driver_exit_observed') is True
        and type(r.get('returncode')) is int
        and (r['returncode'] != 0 or type(r.get('validation_returncode')) is int)
        and r['status'] in ('complete', 'failed', 'needs_review') for r in records)
    terminal_error = None
    if released:
        try:
            (verify_terminal_receipts if terminal_check is None else terminal_check)(records)
        except (ValueError, KeyError, OSError) as error:
            released = False; terminal_error = str(error)
    rows = []
    for case in plan['cases']:
        row = deepcopy({k: v for k, v in indexed[case['index']].items() if k not in OUTCOME_FIELDS})
        if released and row['status'] == 'complete':
            try:
                _technical_record(case, row)
                metrics = outcome(row, case); _check_metrics(metrics)
                row.update(metrics=metrics)
            except (ValueError, KeyError, OSError) as error:
                row.update(status='needs_review', analysis_error=str(error))
        rows.append(row)
    by_identity = {(r['task'], r['execution_map'], r['seed']): r for r in rows}
    panels = []
    for task, seed in itertools.product(TASKS, SEEDS):
        group = [by_identity[task, mapping, seed] for mapping in MAPS]
        good = released and all(r['status'] == 'complete' and 'metrics' in r for r in group)
        if good:
            configs = [{k: v for k, v in r['effective_config'].items() if k not in CONFIG_EXCEPTIONS} for r in group]
            good = (configs[0] == configs[1] == configs[2]
                and len({r['actual_initial_serialized_agent_sha256'] for r in group}) == 1
                and len({r['warmup_actions_sha256'] for r in group}) == 1)
        panels.append(dict(task=task, seed=seed, complete_and_matched=bool(good)))
    task_good = {task: all(p['complete_and_matched'] for p in panels if p['task'] == task) for task in TASKS}
    means = []
    for task, mapping in itertools.product(TASKS, MAPS):
        group = [by_identity[task, mapping, seed] for seed in SEEDS]
        mean = dict(task=task, execution_map=mapping, status='complete' if task_good[task] else 'needs_review',
            complete_curves=sum('metrics' in r for r in group), required_seeds=10)
        if task_good[task]:
            mean.update({key: float(np.mean([r['metrics'][key] for r in group]))
                for key in ('auc_percentage', 'endpoint_percentage', 'tail3_percentage')})
        means.append(mean)
    comparisons = []
    definitions = [(c, True) for c in plan['primary_analysis']['comparisons']]
    definitions += [(c, False) for c in plan['secondary_analysis']['comparisons']]
    for definition, primary in definitions:
        task, left, right = (definition[k] for k in ('task', 'left', 'right'))
        pairs = []
        for seed in SEEDS:
            a, b = by_identity[task, left, seed], by_identity[task, right, seed]
            pair = dict(seed=seed, left_status=a['status'], right_status=b['status'])
            if 'metrics' in a and 'metrics' in b:
                pair['auc_difference'] = a['metrics']['auc_percentage']-b['metrics']['auc_percentage']
                pair['secondary'] = {
                    key: dict(left=a['metrics'][key], right=b['metrics'][key])
                    for key in ('endpoint_percentage', 'tail3_percentage', 'time_to_50', 'time_to_50_censored')}
            pairs.append(pair)
        comparison = dict(**definition, primary=primary, pairs=pairs,
            complete_pairs=sum('auc_difference' in p for p in pairs),
            status='complete' if task_good[task] else 'needs_review',
            complete_three_map_task_required=True)
        if task_good[task]:
            differences = np.array([p['auc_difference'] for p in pairs])
            comparison.update(mean=float(differences.mean()), median=float(np.median(differences)),
                positive_pairs=int((differences > 0).sum()), zero_pairs=int((differences == 0).sum()),
                negative_pairs=int((differences < 0).sum()))
            if primary:
                interval = plan['primary_analysis']['interval']
                rng = np.random.default_rng(interval['seed'])
                draws = differences[rng.integers(0, 10, size=(interval['draws'], 10))].mean(axis=1)
                tail = (1-interval['coverage'])/2
                comparison.update(interval=np.quantile(draws, [tail, 1-tail]).tolist(),
                    interval_coverage=interval['coverage'], interval_type='descriptive paired percentile bootstrap',
                    interval_rng_seed=interval['seed'], exact_coverage_claim=False)
        comparisons.append(comparison)
    complete = released and all(task_good.values())
    return dict(status='complete' if complete else 'needs_review', cohort=COHORT,
        source_commit=common.SOURCE, plan_sha256=PLAN_SHA, expected_runs=60,
        complete_runs=sum(r['status'] == 'complete' for r in rows), outcomes_released=released,
        records=rows, pairing_panels=panels, method_means=means, comparisons=comparisons,
        terminal_evidence_error=terminal_error,
        pooled_primary=False, p_value_or_equivalence_claim=False,
        development_tasks_only=True, causal_scope=plan['estimand'])
