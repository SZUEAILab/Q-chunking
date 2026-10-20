"""Checks for the new intervention; original reference code is left untouched."""
import copy

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.rlpd_recipe_ablation import RLPDRecipeAblationAgent, get_config
from agents.rlpd_reference import RLPDReferenceAgent


def config(**overrides):
    c=get_config().to_dict()
    c.update(horizon_length=1,actor_hidden_dims=(8,8),value_hidden_dims=(8,8))
    c.update(overrides)
    return c


def batch(utd=1):
    rng=np.random.default_rng(3100);n=4*utd
    return {k:jnp.asarray(v,np.float32) for k,v in dict(
        observations=rng.normal(size=(n,7)),next_observations=rng.normal(size=(n,7)),
        actions=rng.uniform(-.8,.8,size=(n,5)),rewards=rng.integers(-3,1,size=n),
        masks=rng.integers(0,2,size=n)).items()}


def assert_state_equal(a,b):
    for name in ('actor','critic','target_critic','temp','rng'):
        left=flax.serialization.to_state_dict(getattr(a,name))
        right=flax.serialization.to_state_dict(getattr(b,name))
        assert jax.tree_util.tree_structure(left)==jax.tree_util.tree_structure(right)
        for x,y in zip(jax.tree_util.tree_leaves(left),jax.tree_util.tree_leaves(right),strict=True):
            np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('utd',[1,20])
@pytest.mark.parametrize('backup',[True,False])
def test_min_path_exactly_matches_reference_updates_and_rng(utd,backup):
    c=config(backup_entropy=backup)
    a=RLPDRecipeAblationAgent.create(9365,np.zeros(7,np.float32),np.zeros(5,np.float32),copy.deepcopy(c))
    b=RLPDReferenceAgent.create(9365,np.zeros(7,np.float32),np.zeros(5,np.float32),copy.deepcopy(c))
    for _ in range(2):
        a,ai=a.update(batch(utd),utd_ratio=utd);b,bi=b.update(batch(utd),utd_ratio=utd)
        assert_state_equal(a,b)
        for name in bi:np.testing.assert_array_equal(ai[name],bi[name])
    assert int(a.critic.step)==2*utd and int(a.actor.step)==int(a.temp.step)==2


@pytest.mark.parametrize('backup',[True,False])
def test_mean_target_matches_independent_two_head_formula_and_terminal_mask(backup):
    c=config(recipe_target_reduction='mean',backup_entropy=backup)
    a=RLPDRecipeAblationAgent.create(9366,np.zeros(7,np.float32),np.zeros(5,np.float32),c)
    b=batch(1);d=a.actor.apply_fn({'params':a.actor.params},b['next_observations'])
    key,rng=jax.random.split(a.rng);actions=d.sample(seed=key)
    key,rng=jax.random.split(rng);indices=np.asarray(jax.random.choice(key,jnp.arange(10),shape=(2,),replace=False))
    # Compute all ten heads independently, then gather the same draw.
    qs=a.critic.apply_fn({'params':a.target_critic.params},b['next_observations'],actions,True)
    next_value=np.asarray(qs)[indices].mean(axis=0)
    expected=np.asarray(b['rewards'])+.99*np.asarray(b['masks'])*next_value
    if backup:expected-=.99*np.asarray(b['masks'])*float(a.temp.apply_fn({'params':a.temp.params}))*np.asarray(d.log_prob(actions))
    current=np.asarray(a.critic.apply_fn({'params':a.critic.params},b['observations'],b['actions'],True))
    updated,info=a.update_critic(b)
    np.testing.assert_allclose(info['critic_loss'],np.square(current-expected).mean(),rtol=2e-6,atol=2e-6)
    np.testing.assert_array_equal(expected[np.asarray(b['masks'])==0],np.asarray(b['rewards'])[np.asarray(b['masks'])==0])
    assert int(updated.critic.step)==1 and int(updated.actor.step)==int(updated.temp.step)==0
    assert len(indices)==2 and len(set(indices))==2


def test_factor_changes_preserve_actual_initial_state_within_architecture():
    agents=[]
    for backup in (True,False):
        for reduction in ('min','mean'):
            agents.append(RLPDRecipeAblationAgent.create(9367,np.zeros(7,np.float32),np.zeros(5,np.float32),
                config(backup_entropy=backup,recipe_target_reduction=reduction)))
    for a in agents[1:]:assert_state_equal(a,agents[0])


@pytest.mark.parametrize('overrides',[dict(recipe_target_reduction='median'),dict(num_min_qs=1),dict(num_qs=2)])
def test_unregistered_target_interventions_fail_before_network_creation(overrides):
    with pytest.raises(ValueError):
        RLPDRecipeAblationAgent.create(1,np.zeros(7),np.zeros(5),config(**overrides))
