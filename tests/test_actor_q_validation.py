import copy
import json
import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from research import actor_q_validation as v


@pytest.fixture
def plan():
    # Use the registered design's exact identities and a small synthetic curve.
    cases=[]
    for task in ('triple2','double2'):
        for mode in ('full','remove_direction_radial'):
            for seed in range(20270,20280):
                group=task+'_'+mode
                cases.append(dict(index=len(cases),task=task,method=mode,seed=seed,env_name=task,run_group=group,
                    argv=['/workspace/venv/bin/python','main_online.py','--agent=agents/acrlpd.py',f'--seed={seed}',
                          '--env_name='+task,'--run_group='+group,'--agent.adapter_actor_q_gradient='+mode]))
    return dict(cohort=v.COHORT,training_source_commit=v.SOURCE,expected_runs=40,tasks=['triple2','double2'],
        methods=['full','remove_direction_radial'],training_seeds=list(range(20270,20280)),cases=cases,
        primary_analysis=dict(interval=dict(seed=20261025,draws=20000,coverage=.975)),scope='synthetic fixed protocol')


def records(plan):
    return [dict(index=c['index'],task=c['task'],method=c['method'],seed=c['seed'],source_commit=v.SOURCE,
        status='complete',returncode=0,validation_returncode=0,driver_exit_observed=True,
        initial_serialized_agent_sha256='same'+str(c['seed']),warmup_actions_sha256='warmup'+str(c['seed']),
        effective_config=dict(adapter_actor_q_gradient=c['method'],other=1)) for c in plan['cases']]


def metric(record,case):
    delta=(case['seed']-20270)-4.5 if case['method']=='remove_direction_radial' else 0.
    return dict(auc_percentage=50+delta,endpoint_percentage=0.,tail3_percentage=0.,time_to_50=None,time_to_50_censored=True)


def test_all_forty_both_directions_zero_secondary_and_seed_pairs_retained(plan):
    result=v.collect(plan,records(plan),metric)
    assert result['status']=='complete' and result['outcomes_released'] and len(result['records'])==40
    for group in result['comparisons']:
        assert group['complete_pairs']==10 and group['mean']==0.
        assert group['positive_pairs']==group['negative_pairs']==5
        assert len(group['pairs'])==10 and group['pairs'][0]['secondary']['time_to_50_censored']['cut'] is True
    assert result==v.collect(plan,records(plan),metric)


@pytest.mark.parametrize('issue',['missing_exit','unobserved_exit','bool_exit','unknown_validation','bool_validation'])
def test_no_outcomes_callback_until_every_actual_child_exit_is_known(plan,issue):
    rows=records(plan)
    if issue=='missing_exit':rows[-1]['returncode']=None
    elif issue=='unobserved_exit':rows[-1]['driver_exit_observed']=False
    elif issue=='bool_exit':rows[-1]['returncode']=False
    elif issue=='unknown_validation':rows[-1]['validation_returncode']=None
    else:rows[-1]['validation_returncode']=False
    def forbidden(*_):raise AssertionError('Partial scientific outcomes read')
    result=v.collect(plan,rows,forbidden)
    assert result['outcomes_released'] is False and result['status']=='needs_review'
    assert all('mean' not in g for g in result['comparisons'])


@pytest.mark.parametrize('issue',['failed','validator_failed','initial','warmup','config','bad_curve'])
def test_failure_or_pair_mismatch_never_creates_a_partial_ten_pair_mean(plan,issue):
    rows=records(plan)
    if issue=='failed':rows[0].update(status='failed',returncode=1)
    elif issue=='validator_failed':rows[0]['validation_returncode']=1
    elif issue=='initial':rows[0]['initial_serialized_agent_sha256']='changed'
    elif issue=='warmup':rows[0]['warmup_actions_sha256']='changed'
    elif issue=='config':rows[0]['effective_config']['other']=2
    def outcome(r,c):
        if issue=='bad_curve' and c['index']==0:raise ValueError('bad success counts')
        return metric(r,c)
    result=v.collect(plan,rows,outcome)
    assert result['status']=='needs_review' and result['comparisons'][0]['complete_pairs']==9
    assert 'mean' not in result['comparisons'][0] and len(result['records'])==40
    assert result['comparisons'][1]['complete_pairs']==10


@pytest.mark.parametrize('issue',['duplicate','seed','source','method','argv_duplicate','extra_difference'])
def test_wrong_identity_or_changed_pair_contract_rejected(plan,issue):
    rows=records(plan)
    if issue=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    elif issue=='seed':rows[0]['seed']=1
    elif issue=='source':rows[0]['source_commit']='other'
    elif issue=='method':plan['cases'][0]['method']='other'
    elif issue=='argv_duplicate':plan['cases'][0]['argv'].append('--seed=20270')
    else:plan['cases'][0]['argv'].append('--different=value')
    with pytest.raises(ValueError):v.collect(plan,rows,metric)


def test_zero_effects_are_preserved_as_zero(plan):
    def zero(*_):return dict(auc_percentage=0.,endpoint_percentage=0.,tail3_percentage=0.,time_to_50=None,time_to_50_censored=True)
    result=v.collect(plan,records(plan),zero)
    assert all(g['zero_pairs']==10 and g['descriptive_bootstrap_interval']==[0.,0.] for g in result['comparisons'])


def test_state_restore_requires_identical_shapes_dtypes_and_values():
    state=dict(a=np.array([1.,2.],np.float32),b={'0':np.array(2,np.int32)})
    assert v.assert_state_equal(state,copy.deepcopy(state))==2
    for changed in [dict(a=np.array([1.,2.],np.float64),b=state['b']),dict(a=np.array([1.,3.],np.float32),b=state['b'])]:
        with pytest.raises(ValueError):v.assert_state_equal(state,changed)


def test_write_once_preserves_first_receipt(tmp_path):
    path=tmp_path/'receipt.json';v.write_once(path,{'status':'failed'})
    with pytest.raises(FileExistsError):v.write_once(path,{'status':'complete'})
    assert v.read(path)=={'status':'failed'}


def test_technical_validation_leaves_poisoned_success_and_reward_uninterpreted(tmp_path,monkeypatch):
    from tests.test_actor_q_gradient import configured
    config,_=configured();config.update(actor_hidden_dims=[512]*4,value_hidden_dims=[512]*4,action_dim=6,target_entropy=-3.,num_qs=10)
    directory=tmp_path/'run';directory.mkdir()
    argv=['python','main_online.py','--seed=29','--agent.adapter_actor_q_gradient=full',
          '--save_dir='+str(tmp_path),'--online_steps=2','--start_training=1',
          '--eval_interval=2','--eval_episodes=2','--utd_ratio=1']
    row=dict(index=0,task='double2',method='full',seed=29,argv=argv)
    flags=dict(seed=29,agent={'adapter_actor_q_gradient':'full'},online_steps=2,start_training=1,
               eval_interval=2,eval_episodes=2,utd_ratio=1)
    v.write_once(directory/'flags.json',flags);v.write_once(directory/'effective_agent_config.json',config)
    def save(name,rows):
        with (directory/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save('eval.csv',[dict(step=s,evaluation_episodes_observed=2,success='SEALED',evaluation_successes_observed='SEALED') for s in [0,2]])
    save('env.csv',[dict(step=s,action_0=.3,action_1=.4,action_2=0.,action_3=.1,action_4=.2,reward='SEALED') for s in [1,2]])
    save('online_agent.csv',[dict(step=2,loss=.1)])
    with (directory/'params_2.pkl').open('wb') as f:pickle.dump(dict(agent=dict(network=dict(step=3,opt_state={'0':{'count':2}}))),f)
    initial=dict(source_commit=v.SOURCE,seed=29,method='full',case_index=0,network_step=1,backend='gpu',
                 device_kind='NVIDIA RTX 4090',serialized_agent_sha256='a'*64,training_flags=argv[2:])
    monkeypatch.setattr(v,'restore_checkpoint',lambda *args:42)
    report=v.technical_validate(row,directory,initial)
    assert report['status']=='complete' and report['outcomes_inspected'] is False
    with pytest.raises(ValueError):v.outcome_metrics(report,row)
