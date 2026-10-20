"""Synthetic protocol fixtures only; none of these curves are scientific data."""
from copy import deepcopy
import hashlib
import itertools
import json

import numpy as np
import pytest

from research import execution_map_study as v


@pytest.fixture
def plan():
    template = '''--agent=agents/acrlpd.py --agent.adapter_actor_q_gradient=full
--agent.target_entropy=-3.0 --agent.num_qs=10 --agent.q_agg=mean --agent.lr=0.0003
--agent.batch_size=256 --agent.discount=0.99 --agent.tau=0.005 --agent.bc_alpha=0.0
--agent.fixed_alpha=-1.0 --horizon_length=1 --online_steps=1000000 --start_training=5000
--buffer_size=1000000 --utd_ratio=1 --eval_at_start=True --eval_interval=100000
--eval_episodes=50 --eval_seed=20261027 --video_episodes=0 --save_interval=1000000
--log_interval=5000 --dataset_proportion=1.0 --sparse=False --actor_update_mode=sac
--stage_balanced_sampling=False --ds_mode=adapter --ds_adapter_representation=cartesian
--ds_adapter_speed_squash=tanh --ds_adapter_speed_interpretation=log
--ds_adapter_critic_space=env --ds_speed_bound=sphere --ds_regularizer=policy_coordinate'''.split()
    cases = []
    for task, mapping, seed in itertools.product(v.TASKS, v.MAPS, v.SEEDS):
        group = 'amm_'+task+'_'+mapping+'_20261020'
        args = template+[f'--env_name={v.technical.TASKS[task]}', f'--seed={seed}',
            '--run_group='+group, '--ds_adapter_execution_map='+mapping,
            '--save_dir=/workspace/results/'+v.COHORT+'/science']
        cases.append(dict(index=len(cases), task=task, execution_map=mapping, seed=seed,
            method='full', env_name=v.technical.TASKS[task], run_group=group,
            argv=['/workspace/venv/bin/python', 'main_online.py', *args]))
    primary = [dict(task=t, left=v.MAPS[0], right=m) for t in v.TASKS for m in v.MAPS[1:]]
    secondary = [dict(task=t, left=v.MAPS[2], right=v.MAPS[1]) for t in v.TASKS]
    return dict(cohort=v.COHORT, expected_runs=60, training_source_commit=v.common.SOURCE,
        source_binding_sha256=v.SOURCE_BINDING_SHA, tasks=list(v.TASKS), maps=list(v.MAPS),
        training_seeds=list(v.SEEDS), cases=cases,
        learning_recipe=dict(action_dim=6, critic_action_dim=5, env_action_dim=5, target_entropy=-3.,
            utd_ratio=1, num_qs=10, backup_entropy=False),
        required_effective_updates=995001, required_network_step=995002,
        primary_analysis=dict(comparisons=primary, complete_pairs_required_per_comparison=10,
            complete_three_map_panel_required_per_task=True,
            interval=dict(draws=20000, seed=20261028, coverage=.9875)),
        secondary_analysis=dict(comparisons=secondary),
        evaluation=dict(steps=v.GRID, episodes_per_point=50, eval_seed=20261027),
        integrity=dict(read_outcomes_only_after_all_60_terminal=True),
        estimand='Synthetic fixture; not a scientific result')


def records(plan):
    result = []
    for c in plan['cases']:
        h = hashlib.sha256(f"{c['task']}:{c['seed']}".encode()).hexdigest()
        result.append(dict(index=c['index'], task=c['task'], execution_map=c['execution_map'], seed=c['seed'],
            source_commit=v.common.SOURCE, status='complete', returncode=0,
            validation_returncode=0, driver_exit_observed=True, actual_environment_steps=1000000,
            actual_joint_optimizer_updates=995001, network_step=995002, restored_state_leaves=150,
            observed_eval_episode_counts=[50]*11, eval_steps=v.GRID,
            source_binding_sha256=v.SOURCE_BINDING_SHA,
            initial_receipt_kind='actual_constructor_capture_before_initial_evaluation',
            actual_initial_serialized_agent_sha256=h, warmup_actions_sha256='c'*64,
            effective_config=dict(action_dim=6, policy_action_dim=6, critic_action_dim=5, env_action_dim=5,
                adapter_execution_map=c['execution_map'], adapter_critic_space='env',
                adapter_actor_q_gradient='full', target_entropy=-3., discount=.99,
                action_pipeline_components=[c['execution_map']])))
    return result


def metric(record, case):
    percent = {'cartesian_aux': 50., 'cartesian_projection_aux': 40.}.get(case['execution_map'])
    if percent is None:
        percent = 50.+(case['seed']-20290)-4.5
    rates = np.full(11, percent/100)
    return dict(success=rates.tolist(), eval_steps=v.GRID,
        auc_percentage=float(np.trapezoid(rates, v.GRID)/1000000*100),
        endpoint_percentage=percent, tail3_percentage=percent,
        time_to_50=0 if percent >= 50 else None, time_to_50_censored=percent < 50)


def synthetic_collect(plan, rows, outcome):
    # Synthetic curves deliberately do not claim real process receipts.
    return v.collect(plan, rows, outcome, terminal_check=lambda _: None)


def test_all_sixty_curves_six_comparisons_and_negative_findings_retained(plan):
    result = synthetic_collect(plan, records(plan), metric)
    assert result['status'] == 'complete' and result['outcomes_released']
    assert result['complete_runs'] == 60 and len(result['records']) == 60
    assert len(result['method_means']) == 6 and len(result['comparisons']) == 6
    assert all(p['complete_and_matched'] for p in result['pairing_panels'])
    for c in result['comparisons']:
        assert c['complete_pairs'] == len(c['pairs']) == 10
        if c['primary']:
            assert c['interval_coverage'] == .9875 and c['exact_coverage_claim'] is False
            assert c['interval_rng_seed'] == 20261028
            if c['right'] == 'cartesian_aux':
                assert c['mean'] == pytest.approx(0) and c['positive_pairs'] == c['negative_pairs'] == 5
                assert c['interval'][0] < 0 < c['interval'][1]
            else:
                assert c['mean'] == pytest.approx(10) and c['positive_pairs'] == 10
        else:
            assert c['mean'] == pytest.approx(-10) and c['negative_pairs'] == 10
            assert 'interval' not in c
    assert result == synthetic_collect(plan, records(plan), metric)
    assert result['pooled_primary'] is result['p_value_or_equivalence_claim'] is False


@pytest.mark.parametrize('field,value', [('returncode', None), ('returncode', False),
    ('driver_exit_observed', False), ('validation_returncode', None),
    ('validation_returncode', False), ('status', 'running')])
def test_no_outcome_call_or_metric_leak_until_every_child_terminal(plan, field, value):
    rr = records(plan); rr[-1][field] = value
    for r in rr:
        r['metrics'] = {'success': ['poisoned partial outcome']}
        r['success'] = ['poisoned partial outcome']
    def forbidden(*_):
        raise AssertionError('An outcome was read before full cohort terminal')
    result = synthetic_collect(plan, rr, forbidden)
    assert result['outcomes_released'] is False
    assert all('metrics' not in r and 'success' not in r for r in result['records'])
    assert all('mean' not in c and all('auc_difference' not in p for p in c['pairs'])
        for c in result['comparisons'])


@pytest.mark.parametrize('issue', ['failed', 'validator_failed', 'initial', 'warmup',
    'config', 'updates', 'episodes', 'bad_curve', 'bad_metric', 'wrong_censoring'])
def test_incomplete_or_unmatched_three_map_task_never_gets_partial_means(plan, issue):
    rr = records(plan)
    if issue == 'failed':rr[0].update(status='failed', returncode=1, validation_returncode=None)
    elif issue == 'validator_failed':rr[0].update(status='needs_review', validation_returncode=1)
    elif issue == 'initial':rr[0]['actual_initial_serialized_agent_sha256'] = 'd'*64
    elif issue == 'warmup':rr[0]['warmup_actions_sha256'] = 'e'*64
    elif issue == 'config':rr[0]['effective_config']['discount'] = .9
    elif issue == 'updates':rr[0]['actual_joint_optimizer_updates'] = 995000
    elif issue == 'episodes':rr[0]['observed_eval_episode_counts'][0] = 49
    def outcome(r, c):
        m = metric(r, c)
        if c['index'] == 0:
            if issue == 'bad_curve':m['success'][0] = float('nan')
            if issue == 'bad_metric':m['auc_percentage'] = 99.
            if issue == 'wrong_censoring':m['time_to_50_censored'] = False
        return m
    result = synthetic_collect(plan, rr, outcome)
    assert result['outcomes_released'] and result['status'] == 'needs_review'
    assert all('auc_percentage' not in x for x in result['method_means'] if x['task'] == 'triple2')
    assert all('mean' not in c for c in result['comparisons'] if c['task'] == 'triple2')
    assert all('mean' in c for c in result['comparisons'] if c['task'] == 'double2')
    assert len(result['records']) == 60 and len(result['comparisons']) == 6


def test_all_zero_success_is_retained_with_censoring_and_zero_interval(plan):
    def zero(r, c):
        return dict(success=[0.]*11, eval_steps=v.GRID, auc_percentage=0.,
            endpoint_percentage=0., tail3_percentage=0., time_to_50=None, time_to_50_censored=True)
    result = synthetic_collect(plan, records(plan), zero)
    assert result['status'] == 'complete'
    assert all(c['zero_pairs'] == 10 for c in result['comparisons'])
    assert all(c['interval'] == [0., 0.] for c in result['comparisons'] if c['primary'])


@pytest.mark.parametrize('issue', ['missing', 'duplicate', 'seed', 'source'])
def test_wrong_or_incomplete_identity_set_is_rejected(plan, issue):
    rr = records(plan)
    if issue == 'missing':rr.pop()
    elif issue == 'duplicate':rr[-1] = deepcopy(rr[-2])
    elif issue == 'seed':rr[-1]['seed'] = 10
    else:rr[-1]['source_commit'] = 'another-source'
    with pytest.raises(ValueError):synthetic_collect(plan, rr, metric)


@pytest.mark.parametrize('issue', ['critic_metadata', 'entropy', 'gradient', 'extra_flag',
    'map', 'seed', 'interval', 'evaluation', 'blinding'])
def test_plan_drift_is_rejected_before_science(plan, issue):
    if issue == 'critic_metadata':plan['learning_recipe']['critic_action_dim'] = 6
    elif issue == 'entropy':plan['cases'][0]['argv'].append('--agent.target_entropy=-2.5')
    elif issue == 'gradient':plan['cases'][0]['argv'].append('--agent.adapter_actor_q_gradient=remove_direction_radial')
    elif issue == 'extra_flag':plan['cases'][0]['argv'].append('--ds_separate_alpha=True')
    elif issue == 'map':plan['cases'][0]['execution_map'] = 'projection'
    elif issue == 'seed':plan['training_seeds'][0] = 10
    elif issue == 'interval':plan['primary_analysis']['interval']['coverage'] = .95
    elif issue == 'evaluation':plan['evaluation']['episodes_per_point'] = 49
    else:plan['integrity']['read_outcomes_only_after_all_60_terminal'] = False
    with pytest.raises(ValueError):v.verify_plan(plan)


@pytest.fixture
def artifact_record(tmp_path, plan, monkeypatch):
    record = records(plan)[0]; record['run_dir'] = str(tmp_path)
    for name in ('flags.json', 'env.csv', 'online_agent.csv', 'eval.csv', 'params_1000000.pkl'):
        (tmp_path/name).write_text('synthetic artifact')
    config = tmp_path/'effective_agent_config.json'
    config.write_text(json.dumps(record['effective_config']))
    record['identity_hashes'] = {p.name: v.common.digest(p) for p in tmp_path.iterdir()}
    initial = tmp_path/'initial.json'
    initial.write_text(json.dumps(dict(serialized_agent_sha256=record['actual_initial_serialized_agent_sha256'])))
    record.update(initial_receipt_path=str(initial), initial_receipt_sha256=v.common.digest(initial))
    record['case_sha256']='f'*64
    for prefix in ('driver', 'validator'):
        value=dict(returncode=0, started=1., ended=2., case_index=0,
            case_sha256=record['case_sha256'], source_commit=v.common.SOURCE,
            pid=123, observed_by_pid=456)
        path = tmp_path/(prefix+'.json');path.write_text(json.dumps(value))
        record[prefix+'_exit_path'] = str(path);record[prefix+'_exit_sha256'] = v.common.digest(path)
    monkeypatch.setattr(v.technical, 'validate_identity', lambda *args, **kwargs: None)
    calls=[]
    def outcome(r, c):calls.append(c['index']);return metric(r, c)
    monkeypatch.setattr(v.common, 'outcome_metrics', outcome)
    return record, plan['cases'][0], calls


def test_default_reader_checks_bound_inputs_then_calls_outcome(artifact_record):
    r, c, calls = artifact_record
    v.observed_outcome(r, c)
    assert calls == [0]


@pytest.mark.parametrize('issue', ['artifact', 'initial_hash', 'initial_identity',
    'driver_hash', 'validator_hash', 'unknown_exit', 'bool_exit', 'backward_clock'])
def test_changed_inputs_or_unobserved_process_evidence_never_reads_outcomes(artifact_record, issue):
    from pathlib import Path
    r, c, calls = artifact_record
    if issue == 'artifact':(Path(r['run_dir'])/'env.csv').write_text('changed')
    elif issue == 'initial_hash':Path(r['initial_receipt_path']).write_text('{}')
    elif issue == 'initial_identity':
        f=Path(r['initial_receipt_path']);f.write_text(json.dumps(dict(serialized_agent_sha256='d'*64)))
        r['initial_receipt_sha256']=v.common.digest(f)
    elif issue.endswith('_hash'):Path(r[issue.replace('_hash','_exit_path')]).write_text('{}')
    else:
        f=Path(r['driver_exit_path']);value=json.loads(f.read_text())
        if issue == 'unknown_exit':value['returncode']=None
        elif issue == 'bool_exit':value['returncode']=False
        else:value['ended']=0.
        f.write_text(json.dumps(value));r['driver_exit_sha256']=v.common.digest(f)
    with pytest.raises(ValueError):v.observed_outcome(r,c)
    assert not calls


def test_global_exit_check_fails_before_even_first_outcome_callback(plan):
    calls=[]
    def bad_last_exit(rows):
        assert len(rows)==60
        raise ValueError('The last process receipt is unverified')
    def forbidden(*_):calls.append(1);raise AssertionError('Outcome read before all actual exits verified')
    result=v.collect(plan, records(plan), forbidden, terminal_check=bad_last_exit)
    assert result['outcomes_released'] is False and not calls
    assert result['terminal_evidence_error']=='The last process receipt is unverified'
    assert all('metrics' not in r for r in result['records'])


def test_duplicate_wait_receipt_cannot_prove_two_actual_exits(artifact_record):
    r, _, _=artifact_record
    with pytest.raises(ValueError,match='reused'):
        v.verify_terminal_receipts([r,deepcopy(r)])


@pytest.mark.parametrize('field,value', [('case_index',59),('case_sha256','e'*64),
    ('source_commit','other'),('pid',False),('observed_by_pid',None),
    ('case_index',False),('started',float('nan')),('ended',float('inf'))])
def test_actual_process_receipt_must_match_case_source_and_observer(artifact_record,field,value):
    from pathlib import Path
    r,_,calls=artifact_record;path=Path(r['driver_exit_path']);d=json.loads(path.read_text());d[field]=value
    path.write_text(json.dumps(d));r['driver_exit_sha256']=v.common.digest(path)
    with pytest.raises(ValueError):v.verify_terminal_receipts([r])
    assert not calls


def test_unbound_case_cannot_use_an_otherwise_valid_exit_receipt(artifact_record):
    r,_,_=artifact_record;r['case_sha256']=''
    with pytest.raises(ValueError,match='case binding'):v.verify_terminal_receipts([r])
