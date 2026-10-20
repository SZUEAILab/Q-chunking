import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze, unfreeze

from agents.acrlpd import ACRLPDAgent, get_config
from direction_speed.config import configure_direction_speed
from research.actor_radial_response import measure_actor_radial_response, radial_direction_component


@pytest.mark.parametrize('diagonal,expected', [([2.,1.,1.], [.5303300859,-.5303300859,0.]),([1.,1.,1.],[0.,0.,0.])])
def test_action_radial_null_direction_can_turn_execution_after_parameter_pullback(diagonal, expected):
    with jax.enable_x64(True):
        matrix=jnp.diag(jnp.asarray(diagonal));u=jnp.array([.4,.4,0.]);theta=jnp.linalg.solve(matrix,u)
        execute=lambda value:.5*value/jnp.linalg.norm(value)
        critic=lambda value:.5*(jnp.dot(value,value)-1.)
        q_gradient=jax.grad(critic)(u)
        np.testing.assert_allclose(q_gradient,u,atol=1e-15)
        np.testing.assert_allclose(jax.jvp(execute,(u,),(q_gradient,))[1],0.,atol=1e-15)
        actor=lambda parameter:matrix@parameter
        parameter_gradient=jax.grad(lambda p:critic(actor(p)))(theta)
        response=jax.jvp(lambda p:execute(actor(p)),(theta,),(parameter_gradient,))[1]
        np.testing.assert_allclose(response,expected,atol=1e-9)
        step=1e-7
        finite=(execute(actor(theta+step*parameter_gradient))-execute(u))/step
        np.testing.assert_allclose(finite,response,atol=2e-7)
        assert np.max(np.abs(actor(theta+step*parameter_gradient)))<1.


def make_agent(space='latent',speed='tanh'):
    config=get_config().to_dict();config.update(actor_hidden_dims=(8,),value_hidden_dims=(8,),num_qs=2,
        horizon_length=1,env_action_dim=5,policy_action_dim=6,critic_action_dim=5 if space=='env' else 6)
    resolved=configure_direction_speed(ds_mode='adapter',ds_speed_bound='sphere',ds_adapter_speed_squash=speed,
        ds_adapter_speed_interpretation='log' if speed=='tanh' else 'direct',ds_adapter_critic_space=space,
        config=config,agent_name='acrlpd')
    action=np.array([.2,.3,.4,.1,-.1],np.float32)
    example=action if space=='env' else resolved.adapter.decompose(action)
    return ACRLPDAgent.create(17,np.ones(7,np.float32),example,config)


@pytest.mark.parametrize('space',['latent','env','canonical'])
@pytest.mark.parametrize('speed',['tanh','sigmoid'])
def test_actual_actor_gradient_matches_archived_loss_and_preserves_state(space,speed):
    agent=make_agent(space,speed)
    before=[np.asarray(v).copy() for v in jax.tree_util.tree_leaves((agent.network.params,agent.network.opt_state,agent.rng))]
    result=measure_actor_radial_response(agent,np.ones((3,7),np.float32))
    assert result['status']=='complete' and result['optimizer_updates_performed']==0
    assert result['network_step_before']==result['network_step_after']
    assert result['q_chain_rule_error_norm']<1e-5 and result['archived_actor_gradient_error_norm']<1e-5
    assert result['valid_direction_samples']==3
    assert result['direct_radial_action_to_environment_response_rms']<1e-6
    if space=='latent':
        assert result['radial_q_actor_gradient_norm']>1e-5
        assert result['radial_parameter_to_environment_response_rms']>1e-7
    else:
        assert result['radial_q_actor_gradient_norm']<1e-5
    after=jax.tree_util.tree_leaves((agent.network.params,agent.network.opt_state,agent.rng))
    for a,b in zip(before,after,strict=True):np.testing.assert_array_equal(a,b)


def test_zero_critic_is_not_mistaken_for_a_useful_invariant_value_function():
    agent=make_agent();params=unfreeze(agent.network.params)
    params['modules_critic']=jax.tree_util.tree_map(np.zeros_like,params['modules_critic'])
    agent=agent.replace(network=agent.network.replace(params=freeze(params)))
    result=measure_actor_radial_response(agent,np.ones((3,7),np.float32))
    assert result['q_actor_gradient_norm']==result['radial_q_actor_gradient_norm']==0.
    assert result['radial_parameter_to_environment_response_rms']==0.
    assert result['unit_radial_parameter_environment_response_rms'] is None
    assert result['entropy_actor_gradient_norm']>0.


def test_zero_small_direction_is_retained_in_residual_without_a_radial_alias_claim():
    actions=jnp.array([[0.,0.,0.,.5,.2,.3],[.4,.4,0.,.5,.2,.3]])
    gradient=jnp.ones_like(actions)
    radial,valid=radial_direction_component(actions,gradient)
    np.testing.assert_array_equal(valid,[False,True]);np.testing.assert_array_equal(radial[0],0.)
    np.testing.assert_array_equal(radial[:,3:],0.)
    np.testing.assert_allclose(jnp.sum((gradient-radial)[1,:3]*actions[1,:3]),0.,atol=1e-7)


@pytest.mark.parametrize('change',[dict(horizon_length=5),dict(actor_update_mode='replay_awr'),
    dict(bc_alpha=1.),dict(adapter_execution_map='cartesian_aux'),dict(ds_phase_regularization='reward_stage_translation')])
def test_unimplemented_actor_interfaces_are_rejected(change):
    agent=make_agent();config=dict(agent.config);config.update(change);agent=agent.replace(config=config)
    with pytest.raises(ValueError):measure_actor_radial_response(agent,np.ones((3,7),np.float32))
