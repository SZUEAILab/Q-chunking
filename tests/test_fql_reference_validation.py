import copy
import json
import numpy as np
import pytest

from research.fql_reference_validation import curve_metrics, expected_grid, assert_state_equal, validate_actions, summarize_pairs


def plan():
    return dict(offline_steps=100, online_steps=100, eval_interval=50, eval_episodes=2, eval_seed=70)


def curve():
    result=[]
    for step, success in zip([0,50,100,150,200],[0,0,0,1,2]):
        episodes=[dict(episode=i,reset_seed=70+i,length=3,success=int(i<success)) for i in range(2)]
        result.append(dict(step=step,online_step=max(0,step-100),episode_count=2,success_count=success,success=success/2,episodes=episodes))
    return result


def test_online_auc_has_its_own_origin_and_does_not_include_offline_learning():
    result=curve_metrics(curve(),plan())
    assert result['online_auc']==.5 and result['offline_auc']==0
    assert result['online_eval_steps']==[0,50,100]
    assert result['time_to_50']==50 and not result['time_to_50_censored']
    assert result['online_tail3']==.5 and result['endpoint']==1


@pytest.mark.parametrize('alteration',['duplicate','missing','order','count','seed','success','axis'])
def test_curve_rejects_missing_steps_and_unobserved_episode_claims(alteration):
    rows=curve()
    if alteration=='duplicate':rows.append(rows[-1])
    elif alteration=='missing':rows.pop(2)
    elif alteration=='order':rows[0],rows[1]=rows[1],rows[0]
    elif alteration=='count':rows[0]['episode_count']=50
    elif alteration=='seed':rows[0]['episodes'][0]['reset_seed']=99
    elif alteration=='success':rows[0]['success']=.5
    else:rows[2]['online_step']=100
    with pytest.raises(ValueError):curve_metrics(rows,plan())


def test_zero_success_curve_keeps_right_censoring_and_nonmultiple_endpoint():
    p=dict(plan(),offline_steps=4,online_steps=64,eval_interval=64,eval_episodes=1)
    assert expected_grid(p)==[0,4,64,68]
    rows=[]
    for step in expected_grid(p):
        rows.append(dict(step=step,online_step=max(0,step-4),episode_count=1,success_count=0,success=0,
                         episodes=[dict(episode=0,reset_seed=70,length=2,success=0)]))
    result=curve_metrics(rows,p)
    assert result['online_auc']==0 and result['time_to_50'] is None and result['time_to_50_censored']


def test_checkpoint_values_ignore_dictionary_order_but_require_exact_dtype():
    a={'z':np.asarray([1],dtype='float32'),'a':{'x':np.asarray(4,dtype='int32')}}
    b={'a':{'x':np.asarray(4,dtype='int32')},'z':np.asarray([1],dtype='float32')}
    assert assert_state_equal(a,b)==2
    b['z']=np.asarray([1],dtype='float64')
    with pytest.raises(ValueError):assert_state_equal(a,b)


def action_rows():
    return [dict(step=4+i,online_step=i,mask=1.,terminal=0.,policy_action=[.2,.1,0,0,0],env_action=[.2,.1,0,0,0]) for i in range(1,4)]


@pytest.mark.parametrize('kind',['correct','duplicate','missing','map','illegal'])
def test_action_stream_checks_actual_step_count_and_mapping(tmp_path,kind):
    rows=action_rows()
    if kind=='duplicate':rows[1]=rows[0]
    if kind=='missing':rows.pop()
    if kind=='map':rows[0]['env_action'][0]=.5
    if kind=='illegal':rows[0]['env_action'][0]=2
    path=tmp_path/'actions.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    args=(path,dict(offline_steps=4,online_steps=3),None,5,5)
    if kind=='correct':assert validate_actions(*args,block_size=2)==3
    else:
        with pytest.raises(ValueError):validate_actions(*args,block_size=2)


def cohort_and_records():
    cohort=dict(seeds=[10,11,12,13],tasks=[dict(name='triple2',env_name='cube-triple-task2')],source_commit='frozen',offline_steps=100,online_steps=100)
    records=[]
    for seed in cohort['seeds']:
        for method in ('native','posthoc'):
            value=.2 if method=='native' else .3
            metrics={key:value for key in ('online_auc','endpoint','online_tail3','offline_auc','offline_endpoint')}
            metrics.update(time_to_50=None,time_to_50_censored=True)
            records.append(dict(env_name='cube-triple-task2',seed=seed,method=method,status='complete',returncode=0,source_commit='frozen',
                offline_steps=100,online_steps=100,metrics=metrics,effective_config=dict(alpha=300,agent_name=method)))
    return cohort,records


def test_complete_pair_summary_is_equal_seed_weight_and_descriptive():
    cohort,records=cohort_and_records()
    result=summarize_pairs(cohort,records)[0]
    assert result['status']=='complete' and len(result['pairs'])==4
    assert result['summary']['online_auc']['mean']==pytest.approx(.1)
    assert result['summary']['online_auc']['positive_pairs']==4


@pytest.mark.parametrize('kind',['failed','missing'])
def test_incomplete_pairs_never_generate_group_estimates(kind):
    cohort,records=cohort_and_records()
    if kind=='failed':records[-1].update(status='failed',returncode=1)
    else:records.pop()
    result=summarize_pairs(cohort,records)[0]
    assert result['status']=='incomplete' and result['unavailable_seeds']==[13] and 'summary' not in result


@pytest.mark.parametrize('kind',['duplicate','source','budget','learner'])
def test_paired_comparison_rejects_incompatible_identities(kind):
    cohort,records=cohort_and_records()
    if kind=='duplicate':records.append(copy.deepcopy(records[0]))
    elif kind=='source':records[-1]['source_commit']='other'
    elif kind=='budget':records[-1]['online_steps']=101
    else:records[-1]['effective_config']['alpha']=100
    with pytest.raises(ValueError):summarize_pairs(cohort,records)
