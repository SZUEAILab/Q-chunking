"""The map intervention must retain its actor, entropy and environment interface."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.acrlpd import ACRLPDAgent, get_config
from direction_speed.adapter import DirectionSpeedAdapter, validate_auxiliary_control
from direction_speed.config import configure_direction_speed
from direction_speed.distributions.factory import resolve_pipeline_recipe


def configured(mapping="cartesian_aux", **overrides):
    config = get_config().to_dict()
    config.update(actor_hidden_dims=(16, 16), value_hidden_dims=(16, 16), num_qs=2,
                  horizon_length=1, env_action_dim=5, policy_action_dim=6, critic_action_dim=5)
    resolved = configure_direction_speed(
        ds_mode="adapter", ds_speed_bound="sphere", ds_adapter_speed_squash="tanh",
        ds_adapter_speed_interpretation="log", ds_adapter_critic_space="env",
        ds_adapter_execution_map=mapping, config=config, agent_name="acrlpd")
    config.update(overrides)
    return config, resolved


def make_agent(mapping="cartesian_aux", **overrides):
    config, resolved = configured(mapping, **overrides)
    return ACRLPDAgent.create(7, np.ones(7, np.float32), np.zeros(5, np.float32), config), resolved


def test_execution_ignores_only_auxiliary_coordinate_and_preserves_scalars():
    _, resolved = configured()
    actions = np.array([[.2, -.3, .4, -.8, .1, -.2], [.2, -.3, .4, .9, .1, -.2]], np.float32)
    expected = actions[:, [0, 1, 2, 4, 5]]
    np.testing.assert_array_equal(resolved.adapter.compose(actions), expected)
    np.testing.assert_array_equal(resolved.postprocessor.compose_policy_output(actions.ravel(), 6), expected)
    np.testing.assert_array_equal(resolved.adapter.compose_jax(jnp.array(actions)), expected)
    np.testing.assert_array_equal(expected[0], expected[1])


def test_numpy_jax_representative_lift_and_chunk_layout_agree():
    _, resolved = configured()
    actions = np.random.default_rng(3).uniform(-.9, .9, (4, 2, 5)).astype(np.float32)
    lifted = resolved.adapter.decompose(actions)
    np.testing.assert_array_equal(lifted[..., 3], 0.)
    np.testing.assert_array_equal(resolved.adapter.decompose_jax(jnp.array(actions)), lifted)
    np.testing.assert_array_equal(resolved.adapter.compose(lifted), actions)
    np.testing.assert_array_equal(resolved.adapter.decompose_chunked(actions, 2), lifted.reshape(4, 12))
    np.testing.assert_array_equal(resolved.adapter.decompose_chunked(actions.reshape(4, 10), 2), lifted.reshape(4, 12))


def test_auxiliary_q_gradient_is_zero_but_entropy_coordinate_is_retained():
    agent, _ = make_agent()
    action = jnp.array([.2, -.3, .4, .4, .1, -.2])
    jacobian = jax.jacrev(agent._critic_action)(action)
    np.testing.assert_array_equal(jacobian[:, 3], 0.)
    np.testing.assert_array_equal(jacobian[:, [0, 1, 2, 4, 5]], np.eye(5))
    dist = agent.network.select("actor")(jnp.ones(7))
    gradient = jax.grad(lambda a: agent._regularizer_log_prob(dist, a))(action)
    assert np.isfinite(gradient).all() and abs(float(gradient[3])) > 1e-5


def test_complete_initial_state_and_policy_distribution_match_legacy_env():
    control, _ = make_agent()
    legacy, _ = make_agent("direction_speed")
    for left, right in zip(jax.tree_util.tree_leaves(control.network), jax.tree_util.tree_leaves(legacy.network), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(control.rng, legacy.rng)
    assert control.config['target_entropy'] == legacy.config['target_entropy'] == -3.
    obs = jnp.ones((4, 7)); key = jax.random.PRNGKey(4)
    dist_a, dist_b = [a.network.select("actor")(obs) for a in (control, legacy)]
    actions = dist_a.sample(seed=key)
    np.testing.assert_array_equal(actions, dist_b.sample(seed=key))
    np.testing.assert_array_equal(dist_a.distribution.loc, dist_b.distribution.loc)
    np.testing.assert_array_equal(dist_a.distribution.stddev(), dist_b.distribution.stddev())
    # Common fresh probes avoid TFP's cached forward vs recomputed inverse path.
    probes = jnp.linspace(-.7, .7, 24).reshape(4, 6)
    np.testing.assert_array_equal(dist_a.log_prob(probes), dist_b.log_prob(probes))
    assert not np.allclose(control._critic_action(actions), legacy._critic_action(actions))


def test_fresh_recipe_reconstruction_uses_same_execution_map():
    config, resolved = configured()
    rebuilt = resolve_pipeline_recipe(config)
    assert rebuilt.components == resolved.components
    assert 'ignored_coordinate=3' in rebuilt.components.postprocessor
    action = np.array([.2, .3, .4, .8, -.2, -.1], np.float32)
    np.testing.assert_array_equal(rebuilt.postprocessor.compose(action), resolved.adapter.compose(action))


@pytest.mark.parametrize('changes', [dict(bc_alpha=1.), dict(horizon_length=5), dict(actor_update_mode='replay_awr'),
                                    dict(adapter_critic_space='latent'), dict(critic_action_dim=6),
                                    dict(policy_action_dim=5), dict(ds_phase_regularization='reward_stage_translation')])
def test_unimplemented_training_interfaces_fail_before_initialization(changes):
    config, _ = configured(**changes)
    with pytest.raises(ValueError, match='Cartesian auxiliary control requires'):
        ACRLPDAgent.create(7, np.ones(7, np.float32), np.zeros(5, np.float32), config)


def test_missing_dimensions_and_wrong_replay_width_are_rejected():
    config, _ = configured(); del config['env_action_dim']
    with pytest.raises(ValueError, match='env_action_dim'): validate_auxiliary_control(config)
    config, _ = configured()
    with pytest.raises(ValueError, match='replay actions'):
        ACRLPDAgent.create(7, np.ones(7, np.float32), np.zeros(6, np.float32), config)


@pytest.mark.parametrize('space', ['latent', 'canonical'])
def test_configuration_rejects_unsupported_critic(space):
    with pytest.raises(ValueError, match='env critic'):
        configure_direction_speed(ds_mode='adapter', ds_speed_bound='sphere', ds_adapter_speed_squash='tanh',
            ds_adapter_speed_interpretation='log', ds_adapter_critic_space=space,
            ds_adapter_execution_map='cartesian_aux', config={}, agent_name='acrlpd')


@pytest.mark.parametrize('changes', [dict(representation='spherical'), dict(speed_squash='sigmoid'),
                                    dict(speed_interpretation='direct'), dict(speed_bound='cube')])
def test_control_does_not_silently_use_another_mapping_recipe(changes):
    args=dict(representation='cartesian', speed_squash='tanh', speed_interpretation='log', speed_bound='sphere',
              execution_map='cartesian_aux'); args.update(changes)
    with pytest.raises(ValueError, match='cartesian/tanh/log/sphere'): DirectionSpeedAdapter(**args)


def test_environment_replay_produces_finite_real_learner_update():
    agent, _ = make_agent()
    actions = jnp.tile(jnp.array([.2, -.3, .4, .1, -.2]), (4, 1, 1))
    batch = dict(observations=jnp.ones((4, 7)), next_observations=jnp.ones((4, 1, 7)), actions=actions,
                 rewards=jnp.zeros((4, 1)), masks=jnp.ones((4, 1)), valid=jnp.ones((4, 1)))
    updated, info = agent.update(batch)
    assert int(updated.network.step) == int(agent.network.step)+1
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((updated.network.params, info)))
