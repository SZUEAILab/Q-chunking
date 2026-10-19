import numpy as np
import pytest

from main_fql_reference import validate_plan, sample_training_batch
from utils.datasets import Dataset, ReplayBuffer


def plan():
    sources = ['main_fql_reference.py', 'agents/fql_reference.py', 'agents/fql_posthoc_reference.py',
               'direction_speed/adapter.py', 'direction_speed/speed.py', 'direction_speed/specs.py',
               'envs/ogbench_utils.py', 'utils/datasets.py', 'utils/networks.py', 'utils/encoders.py', 'utils/flax_utils.py']
    return dict(method='posthoc', seed=10, env_name='cube-triple-play-singletask-task2-v0',
                dataset_path='/data/fake.npz', dataset_sha256='a'*64, validation_sha256='b'*64,
                offline_steps=4, online_steps=64, buffer_size=128, balanced_sampling=True,
                eval_seed=20261019, eval_interval=64, eval_episodes=1, log_interval=32,
                learner={}, source_files={name:'c'*64 for name in sources}, source_commit='d'*40)


@pytest.mark.parametrize('key,value', [('online_steps',-1),('offline_steps',True),('eval_interval',0),
    ('eval_episodes',0),('balanced_sampling',1),('method','direct'),('env_name','antmaze-large-v2')])
def test_plan_rejects_invalid_budget_or_experiment_identity(key,value):
    value_plan=plan();value_plan[key]=value
    with pytest.raises(ValueError): validate_plan(value_plan)


@pytest.mark.parametrize('key,value', [('adapter_critic_space','env'),('adapter_speed_interpretation','direct'),
    ('ds_speed_bound','cube'),('policy_output_transform','tanh'),('agent_name','fql_reference'),('batch_size',3)])
def test_changed_method_is_rejected_before_data_or_environment_creation(key,value):
    value_plan=plan();value_plan['learner'][key]=value
    with pytest.raises(ValueError): validate_plan(value_plan)


def test_raw_uniform_and_balanced_sampling_follow_registered_buffer_choice():
    fields=dict(observations=np.ones((5,7),np.float32), next_observations=np.ones((5,7),np.float32),
                actions=np.full((5,5),.25,np.float32), rewards=np.zeros(5,np.float32),
                masks=np.ones(5,np.float32), terminals=np.zeros(5,np.float32))
    prior=Dataset.create(**fields)
    replay=ReplayBuffer.create({name:value[0] for name,value in fields.items()},size=8)
    transition={name:value[0].copy() for name,value in fields.items()}
    transition['actions']=np.full(5,-.75,np.float32)
    replay.add_transition(transition)
    balanced=sample_training_batch(prior,replay,4,np.random.default_rng(3),True)
    np.testing.assert_array_equal(balanced['actions'][:2],np.full((2,5),.25,np.float32))
    np.testing.assert_array_equal(balanced['actions'][2:],np.full((2,5),-.75,np.float32))
    uniform=sample_training_batch(prior,replay,4,np.random.default_rng(3),False)
    np.testing.assert_array_equal(uniform['actions'],np.full((4,5),-.75,np.float32))
    np.testing.assert_array_equal(prior['actions'],fields['actions'])
