"""Explicit actor-Q backward intervention; forward actions remain unchanged.

This surrogate derivative is an experiment, not the ordinary derivative of the
identity or a claim that the removed component is harmful. It acts only on the
Q query in actor_loss, after sampling/squashing and before the critic interface.
"""
import jax
import jax.numpy as jnp


@jax.custom_jvp
def remove_direction_radial_q_gradient(actions):
    """Identity forward; remove the first-three-coordinate radial derivative.

    Directions at or below the execution normalization floor (1e-6) retain the
    full derivative. Speed and the two scalar coordinates retain theirs too.
    The projection is symmetric, so reverse AD removes the same cotangent.
    """
    if actions.shape[-1] != 6:
        raise ValueError('Actor-Q radial intervention requires six policy coordinates')
    return actions


@remove_direction_radial_q_gradient.defjvp
def _remove_direction_radial_jvp(primals, tangents):
    (actions,), (tangent,) = primals, tangents
    direction = actions[..., :3]
    squared = jnp.sum(direction**2, axis=-1, keepdims=True)
    floor_squared = jnp.asarray(1e-12, dtype=actions.dtype)
    coefficient = jnp.sum(direction*tangent[..., :3], axis=-1, keepdims=True) / jnp.maximum(squared, floor_squared)
    radial = jnp.where(squared > floor_squared, coefficient*direction, 0.)
    retained = jnp.concatenate((tangent[..., :3]-radial, tangent[..., 3:]), axis=-1)
    return remove_direction_radial_q_gradient(actions), retained


def validate_actor_q_gradient(config):
    """Reject interfaces outside the planned single-path H1 SAC intervention."""
    mode = config.get('adapter_actor_q_gradient', 'full')
    if mode == 'full':
        return
    if mode != 'remove_direction_radial':
        raise ValueError('Unknown adapter_actor_q_gradient: '+str(mode))
    required = dict(agent_name='acrlpd', adapter_representation='cartesian',
                    adapter_execution_map='direction_speed', adapter_critic_space='latent',
                    horizon_length=1, action_dim=6, env_action_dim=5, critic_action_dim=6,
                    actor_update_mode='sac', bc_alpha=0., ds_regularizer='policy_coordinate',
                    ds_phase_regularization='none', ds_stage_speed_constraint='none')
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError('Actor-Q radial intervention requires '+name+'='+repr(expected))
    if any(config.get(name, False) for name in ('use_integrated_ds', 'use_cartesian_l2_projection',
                                               'ds_separate_alpha', 'ds_moment_controlled')):
        raise ValueError('Actor-Q radial intervention requires ordinary adapter SAC without extra controls')
