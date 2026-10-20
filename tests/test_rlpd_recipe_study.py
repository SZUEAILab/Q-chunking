"""Synthetic contract tests, never scientific experiment outcomes."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from research import rlpd_recipe_study as s


def plan():
    return s.build_plan('a'*40,'b'*64,[],1792509000.)


def test_complete_factorial_commands_are_unique_balanced_and_explicit():
    p=s.verify_plan(plan());assert len(p['cases'])==320
    assert len({(c['task'],c['seed'],c['utd_ratio'],*[c[f] for f in s.FACTORS]) for c in p['cases']})==320
    assert [sum(c['lane']==i for c in p['cases']) for i in range(4)]==[80]*4
    for c in p['cases']:
        f=s.flags(c['argv']);assert c['argv'][2]=='--agent=agents/rlpd_recipe_ablation.py'
        assert f['agent.num_min_qs']=='2' and f['eval_episodes']=='50'
        assert f['agent.actor_hidden_dims']==f['agent.value_hidden_dims']
    assert p['execution']['armed'] is False and p['execution']['performance_gate'] is None


@pytest.mark.parametrize('mutate',[
    lambda p:p['cases'][0]['argv'].append('--agent.num_min_qs=1'),
    lambda p:p['training_seeds'].pop(),
    lambda p:p['execution'].update(armed=True),
    lambda p:p['integrity'].update(scientific_retries=True),
    lambda p:p['primary_analysis']['interval'].update(coverage=.95),
])
def test_frozen_contract_rejects_drift(mutate):
    p=plan();mutate(p)
    with pytest.raises(ValueError):s.verify_plan(p)


def test_factorial_weights_recover_known_main_and_interaction_effects():
    def y(recipe):
        e=int(not recipe[0]);q=int(recipe[1]=='mean');c=int(recipe[2]=='large')
        return 10+2*e+3*q+4*c+5*e*q+6*e*q*c
    def contrast(fs,fixed=None):return sum(w*y(r) for r,w in s.contrast_weights(fs,fixed))
    assert contrast(('entropy_backup',),{'target_reduction':'min','capacity':'small'})==2
    assert contrast(('entropy_backup','target_reduction'),{'capacity':'small'})==5
    assert contrast(s.FACTORS)==6
    assert contrast(('entropy_backup',))==6


def records(tmp_path,p):
    result=[]
    for c in p['cases']:
        r=dict(index=c['index'],source_commit=p['training_source_commit'],case_sha256=s.case_digest(c),
            source_binding_sha256=p['source_binding_sha256'],status='complete',returncode=0,validation_returncode=0,
            warmup_actions_sha256=hashlib.sha256(str((c['task'],c['seed'])).encode()).hexdigest(),
            initial_component_sha256={k:hashlib.sha256(str((c['task'],c['seed'],c['capacity'],k)).encode()).hexdigest() for k in s.COMPONENTS})
        for kind in ('driver','validator'):
            f=tmp_path/f"{c['index']}-{kind}.json";f.write_text(json.dumps(dict(
                case_sha256=s.case_digest(c),case_index=c['index'],source_commit=p['training_source_commit'],
                returncode=0,started=1.,ended=2.,pid=100+c['index'],observed_by_pid=90)))
            r[kind+'_exit_path']=str(f);r[kind+'_exit_sha256']=s.digest(f)
        result.append(r)
    return result


def metrics(record,case,p):
    e=int(not case['entropy_backup']);q=int(case['target_reduction']=='mean');c=int(case['capacity']=='large')
    y=10+2*e+3*q+4*c+5*e*q+7*e*int(case['utd_ratio']==20)
    return dict(eval_steps=s.GRID,success=[y/100]*11,auc_percentage=float(y))


def test_complete_synthetic_panel_retains_all_curves_and_recovers_utd_interaction(tmp_path):
    p=plan();rs=records(tmp_path,p);out=s.analyze(p,rs,metrics)
    assert out['status']=='complete' and len(out['records'])==320 and len(out['cells'])==32
    assert len(out['primary'])==6 and len(out['secondary'])==50
    e=next(x for x in out['primary'] if x['task']=='triple2' and x['factor']=='entropy_backup')
    np.testing.assert_allclose(e['paired_differences'],[9]*10);np.testing.assert_allclose(e['descriptive_interval'],[9,9])
    interaction=next(x for x in out['secondary'] if x['task']=='triple2' and x.get('factors')==['entropy_backup'] and x['contrast']=='utd20_minus_utd1')
    assert interaction['mean']==7


def test_missing_terminal_prevents_every_outcome_read(tmp_path):
    p=plan();rs=records(tmp_path,p)[:-1];calls=[]
    with pytest.raises(ValueError):s.analyze(p,rs,lambda *a:calls.append(a))
    assert calls==[]


def test_wrong_case_exit_prevents_every_outcome_read(tmp_path):
    p=plan();rs=records(tmp_path,p);rs[1]['driver_exit_path']=rs[0]['driver_exit_path'];rs[1]['driver_exit_sha256']=rs[0]['driver_exit_sha256'];calls=[]
    with pytest.raises(ValueError):s.analyze(p,rs,lambda *a:calls.append(a))
    assert calls==[]


def test_failed_case_blocks_entire_task_aggregate_without_dropping_seed(tmp_path):
    p=plan();rs=records(tmp_path,p);case=p['cases'][0];r=rs[0];r['status']='needs_review'
    out=s.analyze(p,rs,metrics)
    assert out['status']=='needs_review' and len(out['primary'])==3
    assert next(t for t in out['tasks'] if t['task']==case['task'])['unavailable_indices']==[0]


def test_warmup_mismatch_rejects_panel_before_metrics(tmp_path):
    p=plan();rs=records(tmp_path,p);rs[0]['warmup_actions_sha256']='c'*64;calls=[]
    with pytest.raises(ValueError):s.analyze(p,rs,lambda *a:calls.append(a))
    assert calls==[]


def test_initial_parameter_matching_is_required_within_not_across_architectures(tmp_path):
    p=plan();rs=records(tmp_path,p);rs[0]['initial_component_sha256']['actor']='c'*64
    with pytest.raises(ValueError):s.analyze(p,rs,metrics)


def test_changed_plan_bytes_fail_before_deserialization(tmp_path):
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan()));expected=s.digest(p);p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError):s.load_plan(p,expected)
