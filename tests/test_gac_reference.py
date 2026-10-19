import copy
from dataclasses import replace
import io

import numpy as np
import pytest
import torch

from agents.gac_reference import GACReferenceAgent, GACReferenceConfig


torch.set_num_threads(1)


def agent(**overrides):
    return GACReferenceAgent(51,7,-np.ones(5),np.ones(5),GACReferenceConfig(learning_starts=0,**overrides))


def batch(size=8):
    rng=np.random.default_rng(51)
    return dict(observations=rng.normal(size=(size,7)).astype('float32'),
                next_observations=rng.normal(size=(size,7)).astype('float32'),
                actions=rng.uniform(-.8,.8,(size,5)).astype('float32'),
                rewards=np.zeros((size,1),np.float32),dones=np.zeros((size,1),np.float32))


def same(left,right):
    if isinstance(left,dict):
        assert set(left)==set(right)
        for key in left:same(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):same(a,b)
    elif torch.is_tensor(left):torch.testing.assert_close(left,right,atol=0,rtol=0)
    else:assert left==right


def test_initialization_and_external_evaluation_keep_caller_and_training_rng_isolated():
    before=torch.get_rng_state().clone();value=agent()
    torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)
    owned=value.rng_cpu.clone()
    first=value.sample_actions(np.ones(7,np.float32),seed=9)
    second=value.sample_actions(np.ones(7,np.float32),seed=9)
    np.testing.assert_array_equal(first,second)
    torch.testing.assert_close(value.rng_cpu,owned,atol=0,rtol=0)
    torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)
    value.sample_actions(np.ones(7,np.float32))
    assert not torch.equal(value.rng_cpu,owned)
    torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)


def test_full_action_ball_support_targets_and_official_zero_direction():
    value=agent()
    sampled=value.sample_actions(np.ones((32,7),np.float32),seed=1)
    assert sampled.shape==(32,5) and np.max(np.linalg.norm(sampled,axis=-1))<=1+1e-6
    for name in ('qf1','qf2'):
        same(getattr(value,name).state_dict(),getattr(value,name+'_target').state_dict())
    with torch.no_grad():value.actor.mu_head.weight.zero_();value.actor.mu_head.bias.zero_()
    mu,kappa=value.actor(torch.ones((2,7)))
    torch.testing.assert_close(mu,torch.zeros_like(mu),atol=0,rtol=0)
    assert torch.isfinite(kappa).all()
    with pytest.warns(UserWarning):result=value.actor.compute_kappa_regularization(torch.ones((1,1)))
    assert torch.isnan(result)  # Preserve official helper behavior; update explicitly rejects singleton batches.
    with pytest.raises(ValueError,match='singleton'):value.update(batch(1),1)


def test_update_clock_uses_zero_based_steps_delayed_actor_and_post_update_targets():
    value=agent(target_network_frequency=3)
    data=batch()
    for step in range(1,7):
        info=value.update(data,step)
        assert all(np.isfinite(list(info.values())))
    assert (value.critic_updates,value.actor_updates,value.target_updates)==(6,6,2)
    assert {int(state['step']) for state in value.q_optimizer.state.values()}=={6}
    assert {int(state['step']) for state in value.actor_optimizer.state.values()}=={6}
    with pytest.raises(ValueError):value.update(data,6)
    with pytest.raises(ValueError):value.update(data,0)


def test_complete_safe_checkpoint_restores_next_update_and_owned_rng():
    first=agent();data=batch()
    for step in (1,2,3):first.update(data,step)
    stream=io.BytesIO();torch.save(first.state_dict(),stream);stream.seek(0)
    saved=torch.load(stream,weights_only=True,map_location='cpu')
    second=GACReferenceAgent(999,7,-np.ones(5),np.ones(5),GACReferenceConfig(learning_starts=0))
    second.load_state_dict(saved);same(first.state_dict(),second.state_dict())
    np.testing.assert_array_equal(first.sample_actions(np.ones(7,np.float32)),second.sample_actions(np.ones(7,np.float32)))
    info_first=first.update(data,4);info_second=second.update(data,4)
    assert info_first==info_second;same(first.state_dict(),second.state_dict())
    altered=copy.deepcopy(saved);altered['models'].pop('qf2')
    with pytest.raises(ValueError):second.load_state_dict(altered)


@pytest.mark.parametrize('overrides',[dict(batch_size=1),dict(policy_frequency=0),dict(target_network_frequency=0),
    dict(policy_frequency=True),dict(learning_starts=-1),dict(kappa_init=0),dict(kappa_min=20),dict(radius=0),dict(alpha=-1),dict(q_lr=0)])
def test_invalid_reference_recipe_is_rejected(overrides):
    with pytest.raises(ValueError):GACReferenceAgent(1,7,-np.ones(5),np.ones(5),replace(GACReferenceConfig(),**overrides))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires a CUDA runtime')
def test_cpu_agent_does_not_reseed_an_unrelated_visible_cuda_generator():
    before=torch.cuda.get_rng_state().clone()
    value=agent()
    value.sample_actions(np.ones(7,np.float32),seed=123)
    torch.testing.assert_close(torch.cuda.get_rng_state(),before,atol=0,rtol=0)
