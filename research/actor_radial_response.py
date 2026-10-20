"""Separate action-coordinate radial derivatives from actor-parameter effects.

These are local derivatives on fixed states/noise, not optimizer steps or task
performance. The original frozen critic diagnostic remains a separate record.
"""
import numpy as np


def radial_direction_component(actions, cotangent, *, eps=1e-6):
    """Project the first three coordinates; retain small-direction counts.

    Below the normalization floor no radial alias claim is made. Such rows
    contribute zero to this component and remain in the residual/full gradient.
    """
    import jax.numpy as jnp
    direction = actions[..., :3]
    squared = jnp.sum(direction**2, axis=-1, keepdims=True)
    valid = squared > eps**2
    coefficient = jnp.sum(direction*cotangent[..., :3], axis=-1, keepdims=True)/jnp.maximum(squared, eps**2)
    radial = jnp.where(valid, coefficient*direction, 0.)
    return jnp.concatenate([radial, jnp.zeros_like(cotangent[..., 3:])], axis=-1), valid[..., 0]


def tree_dot(left, right):
    import jax
    return float(sum(np.sum(np.asarray(a, dtype=np.float64)*np.asarray(b, dtype=np.float64))
                     for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True)))


def tree_norm(tree):
    return float(np.sqrt(max(tree_dot(tree, tree), 0.)))


def _rms(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(value**2, axis=-1))))


def measure_actor_radial_response(agent, observations, *, seed=20261023):
    """VJP radial Q cotangents, then JVP through the actor and execution map.

    The parameter direction is Euclidean expected-Q ascent, not an Adam update.
    Entropy and the archived complete actor loss are differentiated separately.
    """
    import jax
    import jax.numpy as jnp
    from agents.acrlpd import _adapter_policy_to_env

    config = agent.config
    required = dict(agent_name='acrlpd', adapter_representation='cartesian', horizon_length=1,
                    actor_update_mode='sac', bc_alpha=0., ds_phase_regularization='none',
                    ds_stage_speed_constraint='none', ds_regularizer='policy_coordinate')
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError('Actor radial response requires '+name+'='+repr(expected))
    if (config.get('adapter_execution_map', 'direction_speed') != 'direction_speed'
            or config.get('ds_separate_alpha', False) or config.get('ds_moment_controlled', False)
            or config.get('adapter_critic_space') not in ('latent', 'env', 'canonical')):
        raise ValueError('Unsupported execution or actor-regularization interface')
    observations = jnp.asarray(observations)
    if observations.ndim != 2 or not len(observations) or not np.isfinite(observations).all():
        raise ValueError('Expected a finite nonempty observation matrix')
    if config['action_dim'] != 6 or config['env_action_dim'] != 5:
        raise ValueError('Actor radial response currently requires six/five action coordinates')
    before_rng = np.asarray(agent.rng).copy(); before_step = int(agent.network.step)
    actor = agent.network.params['modules_actor']; key = jax.random.PRNGKey(seed)

    def full_params(p):
        params = dict(agent.network.params); params['modules_actor'] = p
        return params

    def distribution(p):
        return agent.network.select('actor')(observations, params=full_params(p))

    def action_fn(p):
        return distribution(p).sample(seed=key)

    def q_rows(actions):
        return agent.network.select('critic')(observations, agent._critic_action(actions)).mean(axis=0)

    def execute(actions):
        return _adapter_policy_to_env(actions, config)

    actions, pullback = jax.vjp(action_fn, actor)
    action_gradient = jax.grad(lambda a: q_rows(a).mean())(actions)
    radial, valid = radial_direction_component(actions, action_gradient)
    q_gradient = pullback(action_gradient)[0]
    radial_gradient = pullback(radial)[0]
    residual_gradient = pullback(action_gradient-radial)[0]
    direct_q_gradient = jax.grad(lambda p: q_rows(action_fn(p)).mean())(actor)
    add = lambda a, b: jax.tree_util.tree_map(lambda x, y: x+y, a, b)
    subtract = lambda a, b: jax.tree_util.tree_map(lambda x, y: x-y, a, b)
    chain_error = tree_norm(subtract(q_gradient, direct_q_gradient))
    decomposition_error = tree_norm(subtract(add(radial_gradient, residual_gradient), q_gradient))

    fixed_alpha = float(config.get('fixed_alpha', -1.))
    alpha = jnp.asarray(fixed_alpha) if fixed_alpha > 0 else agent.network.select('alpha')()

    def entropy_term(p):
        dist = distribution(p); sample = dist.sample(seed=key)
        return alpha*agent._regularizer_log_prob(dist, sample).mean()

    entropy_gradient = jax.grad(entropy_term)(actor)
    batch = dict(observations=observations, actions=jnp.zeros((len(observations), 1, config['critic_action_dim'])))
    archived_gradient = jax.grad(lambda p: agent.actor_loss(batch, full_params(p), key)[0])(actor)
    expected_actor = subtract(entropy_gradient, q_gradient)
    actor_error = tree_norm(subtract(archived_gradient, expected_actor))
    q_norm, radial_norm, residual_norm = map(tree_norm, (q_gradient, radial_gradient, residual_gradient))
    actor_norm = tree_norm(archived_gradient)
    tolerance = 5e-5
    if (chain_error > tolerance*max(1., q_norm)
            or decomposition_error > tolerance*max(1., q_norm)
            or actor_error > tolerance*max(1., actor_norm)):
        raise ValueError('Actor chain rule or complete-loss reconstruction differs')

    # An action-space radial tangent is killed by the execution map, but its
    # parameter pullback need not be killed after pushing through the actor.
    _, direct_environment_response = jax.jvp(execute, (actions,), (radial,))
    _, latent_response = jax.jvp(action_fn, (actor,), (radial_gradient,))
    _, environment_response = jax.jvp(execute, (actions,), (latent_response,))
    stable_unit = radial_norm > max(1e-6, 1e-6*q_norm)
    unit_environment = environment_response/radial_norm if stable_unit else None
    q = np.asarray(q_rows(actions), dtype=np.float64)
    record = dict(status='complete', samples=len(observations), seed=seed, critic_space=config['adapter_critic_space'],
        valid_direction_samples=int(np.asarray(valid).sum()), small_direction_samples=int((~np.asarray(valid)).sum()),
        alpha=float(alpha), q_mean=float(q.mean()), q_std=float(q.std()),
        q_actor_gradient_norm=q_norm, radial_q_actor_gradient_norm=radial_norm,
        residual_q_actor_gradient_norm=residual_norm, entropy_actor_gradient_norm=tree_norm(entropy_gradient),
        full_actor_gradient_norm=actor_norm,
        radial_to_total_q_actor_norm=radial_norm/q_norm if q_norm > 0 else None,
        radial_residual_cosine=tree_dot(radial_gradient, residual_gradient)/(radial_norm*residual_norm) if radial_norm*residual_norm > 0 else None,
        q_chain_rule_error_norm=chain_error, q_decomposition_error_norm=decomposition_error,
        archived_actor_gradient_error_norm=actor_error,
        q_action_gradient_rms_unaveraged=_rms(action_gradient*len(observations)),
        radial_q_action_gradient_rms_unaveraged=_rms(radial*len(observations)),
        direct_radial_action_to_environment_response_rms=_rms(direct_environment_response),
        radial_parameter_to_latent_response_rms=_rms(latent_response),
        radial_parameter_to_latent_other_coordinates_response_rms=_rms(latent_response[..., 3:]),
        radial_parameter_to_environment_response_rms=_rms(environment_response),
        radial_parameter_to_environment_xyz_response_rms=_rms(environment_response[..., :3]),
        radial_parameter_to_environment_scalar_response_rms=_rms(environment_response[..., 3:]),
        unit_radial_parameter_environment_response_rms=_rms(unit_environment) if stable_unit else None,
        unit_response_status='defined' if stable_unit else 'radial_gradient_below_fixed_numerical_floor',
        unit_response_floor=max(1e-6, 1e-6*q_norm),
        actor_parameter_count=sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(actor)),
        optimizer_updates_performed=0, network_step_before=before_step, network_step_after=int(agent.network.step),
        interpretation='Fixed-state/noise local derivatives. Euclidean Q-gradient direction; no Adam step, rollout, or causal return attribution. Norm ratios can exceed one through cancellation; they are not wasted-gradient percentages.')
    for tree in (actions, action_gradient, q_gradient, radial_gradient, residual_gradient,
                 entropy_gradient, archived_gradient, latent_response, environment_response):
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(tree)):
            raise ValueError('Nonfinite gradient or local response')
    np.testing.assert_array_equal(agent.rng, before_rng)
    assert int(agent.network.step) == before_step
    return record
