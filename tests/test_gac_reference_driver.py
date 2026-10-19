from dataclasses import asdict

import numpy as np
import pytest

from agents.gac_reference import GACReferenceConfig
from main_gac_reference import OnlineReplay, expected_updates, sample_batch, validate_plan


def example(value=0):
    return dict(observations=np.full(7,value,np.float32),actions=np.full(5,value,np.float32),
                rewards=np.asarray([value],np.float32),dones=np.zeros(1,np.float32),next_observations=np.full(7,value,np.float32))


def plan(protocol='online_only'):
    return dict(seed=10,eval_seed=20261019,env_name='cube-triple-play-singletask-task2-v0',replay_protocol=protocol,
      dataset_path=None if protocol=='online_only' else '/data/cube-triple-play-v0.npz',
      dataset_sha256=None if protocol=='online_only' else 'a'*64,validation_sha256=None if protocol=='online_only' else 'b'*64,
      online_steps=64,replay_capacity=128,eval_interval=64,eval_episodes=1,log_interval=32,
      learner=asdict(GACReferenceConfig(learning_starts=32)),device='cuda',torch_version='pinned',
      environment_versions={name:'pinned' for name in ('ogbench','mujoco','gymnasium','numpy')},
      source_commit='a'*40,source_files={name:'c'*64 for name in ('main_gac_reference.py','agents/__init__.py','agents/gac_reference.py','envs/ogbench_utils.py')})


@pytest.mark.parametrize('budget,start,actor,target',[(64,32,2,1),(5000,5000,2,1),(5002,5000,2,1),(100,3,3,4)])
def test_expected_counts_match_the_actual_upstream_zero_based_events(budget,start,actor,target):
    config=GACReferenceConfig(learning_starts=start,policy_frequency=actor,target_network_frequency=target)
    updates=[step for step in range(budget) if step>start]
    assert expected_updates(budget,config)==dict(critic=len(updates),actor=sum(actor for step in updates if step%actor==0),target=sum(step%target==0 for step in updates))


def test_ring_overwrite_retains_recent_raw_transitions_and_stable_sample_indices():
    replay=OnlineReplay(example(),3)
    for value in range(5):replay.add(example(value))
    assert replay.size==3 and replay.pointer==2
    assert set(replay.data['actions'][:,0])=={2,3,4}
    rows,indices=replay.sample(20,np.random.default_rng(1))
    np.testing.assert_array_equal(rows['observations'],replay.data['observations'][indices])
    assert set(rows['actions'][:,0])<={2,3,4}


def test_balanced_replay_preserves_immutable_prior_and_counts_each_origin():
    replay=OnlineReplay(example(),3);replay.add(example(-.5))
    prior={name:np.stack([example(.75)[name]]*4) for name in example()}
    for value in prior.values():value.setflags(write=False)
    batch,audit=sample_batch(replay,prior,8,np.random.default_rng(3))
    assert (audit['prior_rows'],audit['online_rows'])==(4,4)
    np.testing.assert_array_equal(batch['actions'][:4],np.full((4,5),.75,np.float32))
    np.testing.assert_array_equal(batch['actions'][4:],np.full((4,5),-.5,np.float32))
    np.testing.assert_array_equal(prior['actions'],np.full((4,5),.75,np.float32))
    _,online=sample_batch(replay,None,8,np.random.default_rng(3))
    assert (online['prior_rows'],online['online_rows'])==(0,8) and len(online['prior_indices'])==0


@pytest.mark.parametrize('field,value',[('replay_protocol','combined'),('online_steps',0),('seed',True),('device','auto'),('eval_interval',0)])
def test_plan_rejects_ambiguous_or_invalid_protocols(field,value):
    value_plan=plan();value_plan[field]=value
    with pytest.raises(ValueError):validate_plan(value_plan)


@pytest.mark.parametrize('kind',['hidden_prior','missing_prior','wrong_family','odd_batch','radius','unknown_learner','missing_source'])
def test_data_and_recipe_changes_fail_before_environment_creation(kind):
    value_plan=plan('prior_balanced')
    if kind=='hidden_prior':value_plan['replay_protocol']='online_only'
    elif kind=='missing_prior':value_plan['dataset_sha256']=None
    elif kind=='wrong_family':value_plan['dataset_path']='/data/cube-double-play-v0.npz'
    elif kind=='odd_batch':value_plan['learner']['batch_size']=255
    elif kind=='radius':value_plan['learner']['radius']=2.5
    elif kind=='unknown_learner':value_plan['learner']['hidden_alpha']=.1
    else:value_plan['source_files'].pop('agents/__init__.py')
    with pytest.raises(ValueError):validate_plan(value_plan)


def test_valid_protocols_are_explicit_and_empty_buffer_is_not_sampled():
    assert validate_plan(plan()).learning_starts==32
    assert validate_plan(plan('prior_balanced')).batch_size==256
    replay=OnlineReplay(example(),3)
    with pytest.raises(ValueError,match='empty'):replay.sample(2,np.random.default_rng(3))
