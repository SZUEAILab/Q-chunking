import copy,json
from pathlib import Path
import pytest
from research.posthoc_checkpoint_policy import digest,bind_run,analyze,eligible,contract

@pytest.fixture
def run(tmp_path):
 d=tmp_path/'sd010';d.mkdir();f=dict(seed=10,online_steps=64,env_name='env',run_group='group',ds_mode='adapter',ds_adapter_speed_squash='tanh',ds_adapter_speed_interpretation='log',ds_speed_bound='sphere',horizon_length=1)
 c=dict(env_action_dim=5,horizon_length=1,adapter_representation='cartesian',adapter_speed_squash='tanh',adapter_speed_interpretation='log',ds_speed_bound='sphere')
 (d/'flags.json').write_text(json.dumps(f));(d/'effective_agent_config.json').write_text(json.dumps(c));(d/'params_64.pkl').write_bytes(b'owned-test-checkpoint')
 r=dict(name='posthoc',seed=10,status='complete',returncode=0,source_commit='source',budget=64,run_dir=str(d),effective_config=c,effective_config_sha256=digest(d/'effective_agent_config.json'),checkpoint_sha256=digest(d/'params_64.pkl'))
 expected={k:v for k,v in f.items() if k!='seed'}
 return d,r,expected,tmp_path


def test_hash_bound_validated_run_accepted(run):
 d,r,c,root=run
 assert bind_run(r,c,'source',root)[3]==d/'params_64.pkl'

@pytest.mark.parametrize('problem',['checkpoint','config','flag','seed','exit','source','escape'])
def test_changed_or_unvalidated_input_rejected(run,problem):
 d,r,c,root=run
 if problem=='checkpoint':(d/'params_64.pkl').write_bytes(b'changed')
 elif problem=='config':(d/'effective_agent_config.json').write_text('{}')
 elif problem in ['flag','seed']:
  p=d/'flags.json';f=json.loads(p.read_text());f['run_group' if problem=='flag' else 'seed']='wrong';p.write_text(json.dumps(f))
 elif problem=='exit':r['returncode']=1
 elif problem=='source':r['source_commit']='other'
 elif problem=='escape':root=d/'unrelated'
 with pytest.raises((ValueError,KeyError)):bind_run(r,c,'source',root)


def dataset():
 tasks=dict(common=dict(ds_mode='adapter',ds_adapter_speed_squash='tanh',ds_adapter_speed_interpretation='log',ds_speed_bound='sphere'),tasks=[dict(name='posthoc')],seeds=[10,11])
 rows=[dict(name='posthoc',seed=s,status='complete',returncode=0,source_commit='source',run_dir=str(s)) for s in [10,11]]
 return dict(source_commit='source',expected_runs=2,complete_runs=2,status='complete',records=rows),tasks


def operation(r,c):
 m=dict(radius_mean=.5,mean_within_state_radius_std=.1,mean_within_state_xyz_covariance_trace=.2)
 return dict(name=r['name'],seed=r['seed'],source_commit='source',status='policy_probe_complete',method='posthoc',cases=[dict(noise_scale=1.,posthoc=m)])


def test_missing_seed_has_no_partial_group_mean():
 r,t=dataset();r['records'].pop();r['complete_runs']=1;r['status']='needs_review'
 out=analyze(r,t,'source',{},operation)
 assert out['status']=='needs_review' and out['policy_probe_complete_runs']==1
 assert out['groups'][0]['status']=='incomplete' and 'mean_radius_mean' not in out['groups'][0]
 assert out['records'][1]['upstream_status']=='missing'


def test_complete_group_has_equal_seed_aggregate():
 r,t=dataset();out=analyze(r,t,'source',{},operation)
 assert out['status']=='complete' and out['eligible_runs']==2
 assert out['groups'][0]['mean_radius_mean']==.5


def test_failed_child_not_loaded_and_bad_source_not_accepted():
 r,t=dataset();r['records'][0].update(status='failed',returncode=7);r['records'][1]['source_commit']='other';r['complete_runs']=1;r['status']='needs_review'
 out=analyze(r,t,'source',{},lambda *a:pytest.fail('Invalid record loaded'))
 assert out['records'][0]['status']=='upstream_unavailable'
 assert out['records'][1]['status']=='needs_review'


def test_duplicates_and_unknown_identities_rejected():
 r,t=dataset();r['records'].append(copy.deepcopy(r['records'][0]))
 with pytest.raises(ValueError,match='duplicate'):contract(r,t,'source')
 r,t=dataset();r['records'][0]['name']='other'
 with pytest.raises(ValueError,match='identity'):contract(r,t,'source')


def test_execution_mapping_error_is_retained():
 r,t=dataset()
 def fail(*a):raise ValueError('bad geometry')
 out=analyze(r,t,'source',{},fail)
 assert out['policy_probe_complete_runs']==0 and all(x['status']=='needs_review' for x in out['records'])


def test_direct_and_projection_not_mislabeled_as_original_posthoc():
 assert not eligible(dict(ds_mode='projection'))
 assert not eligible(dict(ds_mode='adapter',ds_adapter_speed_squash='sigmoid',ds_adapter_speed_interpretation='direct',ds_speed_bound='cube'))


@pytest.mark.parametrize('plan',[dict(eligible_identities=[dict(name='posthoc',seed=10)]),dict(expected_runs=3),dict(checkpoint_step=1000000)])
def test_registered_scope_is_checked_before_restoring_any_model(plan):
    r,t=dataset();t['common']['online_steps']=64
    with pytest.raises(ValueError,match='Registered'):
        analyze(r,t,'source',plan,lambda *a:pytest.fail('Model restoration started before contract validation'))
