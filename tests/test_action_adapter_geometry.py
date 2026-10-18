"""Geometry summaries must represent actual rows, phase boundaries and missingness."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from research.action_adapter_geometry import analyze_cohort, digest, stream_actions


def csv_rows(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_actions(path,actions,steps=None):
    rows=[]
    for step,action in zip(range(1,len(actions)+1) if steps is None else steps,actions,strict=True):
        rows.append(dict(step=step,**{f'action_{i}':v for i,v in enumerate(action)},
                         action_xyz_norm=np.linalg.norm(action[:3]),action_xyz_linf=np.max(np.abs(action[:3]))))
    csv_rows(path,rows)


def test_streamed_covariance_matches_batch_and_excludes_random_warmup(tmp_path):
    actions=np.array([[1.,1.,1.,0.,0.],[.2,.3,.4,.1,-.1],[-.2,.1,.4,.2,.3],[.5,-.3,.1,-.2,0.]])
    path=tmp_path/'env.csv';write_actions(path,actions)
    result=stream_actions(path,dimensions=5,budget=4,warmup=1,window=2,block_size=2)
    policy=result['learned_policy']
    assert result['rows']==4 and policy['count']==3 and result['random_warmup']['count']==1
    np.testing.assert_allclose(policy['action_mean'],actions[1:].mean(axis=0),atol=1e-15)
    np.testing.assert_allclose(policy['action_sample_covariance'],np.cov(actions[1:].T),atol=1e-15)
    assert result['env_csv_sha256']==digest(path)
    assert [x['count'] for x in result['learning_windows']]==[1,2]
    assert policy['xyz_norm_mean']==pytest.approx(np.linalg.norm(actions[1:,:3],axis=1).mean())


def test_cube_boundary_sphere_boundary_and_outside_ball_are_distinct(tmp_path):
    actions=np.array([[1.,0.,0.,0.,0.],[1.,1.,0.,0.,0.],[.2,0.,0.,1.,0.]])
    path=tmp_path/'env.csv';write_actions(path,actions)
    result=stream_actions(path,dimensions=5,budget=3,warmup=0,block_size=1)['learned_policy']
    assert result['unit_sphere_boundary_fraction']==pytest.approx(1/3)
    assert result['outside_unit_sphere_fraction']==pytest.approx(1/3)
    assert result['xyz_cube_boundary_fraction']==pytest.approx(2/3)
    np.testing.assert_allclose(result['component_cube_boundary_fraction'],[2/3,1/3,0,1/3,0])


def test_empty_policy_and_single_sample_covariance_are_not_fake_zeros(tmp_path):
    path=tmp_path/'env.csv';write_actions(path,np.zeros((1,5)))
    result=stream_actions(path,dimensions=5,budget=1,warmup=1)
    assert result['learned_policy']['count']==0 and result['learned_policy']['xyz_norm_mean'] is None
    assert result['all_actions']['action_sample_covariance'] is None
    assert result['learning_windows'][0]['policy_first_step'] is None


@pytest.mark.parametrize('steps,budget',[([1,3],3),([1,1],2),([2,1],2),([1,2],1),([1,2],3)])
def test_gaps_duplicates_order_and_incomplete_budget_are_rejected(tmp_path,steps,budget):
    path=tmp_path/'env.csv';write_actions(path,np.zeros((2,5)),steps)
    with pytest.raises(ValueError):stream_actions(path,dimensions=5,budget=budget,warmup=0)


@pytest.mark.parametrize('field,value',[('action_0','nan'),('action_4','1.2'),
                                       ('action_xyz_norm','nan'),('action_xyz_norm','1'),('action_xyz_linf','1')])
def test_nonfinite_illegal_and_inconsistent_actions_are_rejected(tmp_path,field,value):
    path=tmp_path/'env.csv';write_actions(path,np.zeros((2,5)))
    with path.open() as stream:rows=list(csv.DictReader(stream))
    rows[0][field]=value;csv_rows(path,rows)
    with pytest.raises(ValueError):stream_actions(path,dimensions=5,budget=2,warmup=0)


@pytest.fixture
def cohort(tmp_path):
    records=[]
    for seed in [10,11]:
        directory=tmp_path/f'sd{seed}';directory.mkdir()
        flags=dict(seed=seed,env_name='cube',run_group='group',online_steps=4,start_training=1)
        config=dict(env_action_dim=5)
        (directory/'flags.json').write_text(json.dumps(flags))
        (directory/'effective_agent_config.json').write_text(json.dumps(config))
        write_actions(directory/'env.csv',np.zeros((4,5)))
        records.append(dict(name='native',seed=seed,status='complete',returncode=0,source_commit='frozen',
                            run_dir=str(directory),budget=4,effective_config=config,started=10.,ended=12.))
    report=dict(source_commit='frozen',status='complete',expected_runs=2,complete_runs=2,records=records)
    tasks=dict(common=dict(online_steps=4,start_training=1),tasks=[dict(name='native',env_name='cube',run_group='group')],seeds=[10,11])
    return report,tasks


def test_complete_group_has_per_seed_values_without_pooling_samples(cohort):
    report,tasks=cohort
    result=analyze_cohort(report,tasks,'frozen')
    assert result['status']=='complete' and result['geometry_complete_runs']==2
    assert result['groups'][0]['per_seed_xyz_norm_mean']==[0.,0.]
    assert result['groups'][0]['per_seed_run_wall_seconds']==[2.,2.]


def test_missing_seed_remains_in_inventory_and_prevents_group_average(cohort):
    report,tasks=cohort;report['records'].pop();report['complete_runs']=1;report['status']='needs_review'
    result=analyze_cohort(report,tasks,'frozen')
    assert result['expected_runs']==2 and result['geometry_complete_runs']==1
    assert result['records'][1]['upstream_status']=='missing'
    assert result['groups'][0]['status']=='incomplete' and 'mean_xyz_norm_mean' not in result['groups'][0]


@pytest.mark.parametrize('problem',['returncode','source','flags','duration'])
def test_invalid_validated_receipt_is_not_promoted_to_geometry_complete(cohort,problem):
    report,tasks=cohort;record=report['records'][0]
    if problem=='returncode':record['returncode']=7
    if problem=='source':record['source_commit']='other'
    if problem=='duration':record['ended']=9
    if problem=='flags':
        path=Path(record['run_dir'])/'flags.json';flags=json.loads(path.read_text());flags['seed']=99;path.write_text(json.dumps(flags))
    result=analyze_cohort(report,tasks,'frozen')
    assert result['status']=='needs_review' and result['geometry_complete_runs']==1
    assert result['records'][0]['status']=='needs_review'


def test_duplicate_report_identity_is_rejected(cohort):
    report,tasks=cohort;report['records'][1]=copy.deepcopy(report['records'][0])
    with pytest.raises(ValueError,match='duplicate upstream'):analyze_cohort(report,tasks,'frozen')


def test_historical_norm_only_log_cannot_invent_coordinate_covariance(tmp_path):
    path=tmp_path/'env.csv';csv_rows(path,[dict(step=1,action_xyz_norm=.5)])
    with pytest.raises(ValueError,match='per-component'):stream_actions(path,dimensions=5,budget=1,warmup=0)


def test_nearly_constant_actions_keep_small_variance_across_blocks(tmp_path):
    rng=np.random.default_rng(20261019)
    actions=.8+1e-8*rng.normal(size=(103,5))
    path=tmp_path/'env.csv';write_actions(path,actions)
    result=stream_actions(path,dimensions=5,budget=103,warmup=0,block_size=7)['learned_policy']
    np.testing.assert_allclose(result['action_sample_covariance'],np.cov(actions.T),atol=1e-23,rtol=1e-6)
