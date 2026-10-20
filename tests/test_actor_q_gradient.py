"""Verify a single actor-Q path intervention against independent derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.acrlpd import ACRLPDAgent, get_config
from direction_speed.actor_q_gradient import remove_direction_radial_q_gradient, validate_actor_q_gradient
from direction_speed.config import configure_direction_speed


def assert_tree_equal(left, right):
    left, right = map(jax.tree_util.tree_leaves, (left, right))
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_identity_forward_and_independent_batched_vjp_jvp(dtype):
    with jax.enable_x64(True):
        actions = jnp.array([[.3, -.4, .5, -.7, .2, -.3], [0., 0., 0., .8, -.1, .7],
                             [2e-6, 0., 0., .3, .4, .5]], dtype=dtype)
        cotangent = jnp.array([[2., -3., 4., 5., -6., 7.], [1., 2., 3., 4., 5., 6.],
                              [-2., 7., 3., 4., 1., 9.]], dtype=dtype)
        expected = np.asarray(cotangent).copy()
        for index in (0, 2):
            unit = np.array(actions[index, :3], dtype=np.float64, copy=True)
            unit /= np.linalg.norm(unit)
            expected[index, :3] -= np.dot(unit, expected[index, :3])*unit
        output, back = jax.vjp(remove_direction_radial_q_gradient, actions)
        assert_tree_equal(actions, output)
        actual = jax.jit(back)(cotangent)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        _, forward = jax.jvp(remove_direction_radial_q_gradient, (actions,), (cotangent,))
        np.testing.assert_allclose(forward, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(actual[..., 3:], cotangent[..., 3:])
        np.testing.assert_allclose(jnp.sum(actual[[0, 2], :3]*actions[[0, 2], :3], axis=-1), 0., atol=1e-6)


def test_zero_and_below_floor_keep_full_gradient_without_nan():
    actions = jnp.array([[0., 0., 0., .1, .2, .3], [5e-7, 0., 0., .4, .5, .6]], jnp.float32)
    weights = jnp.arange(1., 13.).reshape(2, 6)
    gradient = jax.jit(jax.grad(lambda a: jnp.sum(remove_direction_radial_q_gradient(a)*weights)))(actions)
    np.testing.assert_array_equal(gradient, weights)


def configured(speed='tanh', **changes):
    config = get_config().to_dict()
    config.update(actor_hidden_dims=(8,), value_hidden_dims=(8,), num_qs=2,
                  horizon_length=1, env_action_dim=5, policy_action_dim=6, critic_action_dim=6)
    adapter = configure_direction_speed(ds_mode='adapter', ds_speed_bound='sphere',
        ds_adapter_speed_squash=speed, ds_adapter_speed_interpretation='log' if speed=='tanh' else 'direct',
        ds_adapter_critic_space='latent', config=config, agent_name='acrlpd').adapter
    config.update(changes)
    example = adapter.decompose(np.array([.2, .3, .4, -.1, .1], np.float32))
    return config, example


def make_agent(speed='tanh', **changes):
    config, example = configured(speed, **changes)
    return ACRLPDAgent.create(29, np.ones(7, np.float32), example, config)


def batch():
    obs = jnp.arange(21., dtype=jnp.float32).reshape(3, 7)/21.
    return dict(observations=obs, next_observations=(obs+.1)[:, None, :],
                actions=jnp.tile(jnp.array([.2, -.3, .4, -.2, .1, -.1]), (3, 1, 1)),
                rewards=jnp.array([[-1.], [0.], [-2.]]), masks=jnp.ones((3, 1)), valid=jnp.ones((3, 1)))


@pytest.mark.parametrize('speed', ['tanh', 'sigmoid'])
def test_only_actor_q_parameter_gradient_changes_while_all_forward_values_match(speed):
    full = make_agent(speed)
    cut = make_agent(speed, adapter_actor_q_gradient='remove_direction_radial')
    assert_tree_equal(full.network, cut.network)
    assert_tree_equal(full.rng, cut.rng)
    assert full.config['target_entropy'] == cut.config['target_entropy'] == -3.
    data = batch(); key = jax.random.PRNGKey(27)
    assert_tree_equal(full.sample_actions(data['observations'], key), cut.sample_actions(data['observations'], key))
    value_full, info_full = full.actor_loss(data, full.network.params, key)
    value_cut, info_cut = cut.actor_loss(data, cut.network.params, key)
    assert_tree_equal((value_full, info_full), (value_cut, info_cut))
    full_grad = jax.grad(lambda p: full.actor_loss(data, p, key)[0])(full.network.params)
    cut_grad = jax.grad(lambda p: cut.actor_loss(data, p, key)[0])(cut.network.params)

    # Independent reference: compute an ordinary Q action gradient and explicitly
    # pull back its radial component, without calling the custom derivative.
    actor = full.network.params['modules_actor']
    def sample(p):
        params = dict(full.network.params); params['modules_actor'] = p
        return full.network.select('actor')(data['observations'], params=params).sample(seed=key)
    actions, pullback = jax.vjp(sample, actor)
    q_gradient = jax.grad(lambda a: full.network.select('critic')(data['observations'], a).mean())(actions)
    unit = actions[:, :3]/jnp.linalg.norm(actions[:, :3], axis=-1, keepdims=True)
    radial = jnp.concatenate((unit*jnp.sum(q_gradient[:, :3]*unit, axis=-1, keepdims=True),
                              jnp.zeros_like(q_gradient[:, 3:])), axis=-1)
    radial_actor = pullback(radial)[0]
    difference = jax.tree_util.tree_map(lambda c, f: c-f, cut_grad['modules_actor'], full_grad['modules_actor'])
    for actual, expected in zip(jax.tree_util.tree_leaves(difference), jax.tree_util.tree_leaves(radial_actor), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert sum(float(jnp.sum(x*x)) for x in jax.tree_util.tree_leaves(radial_actor)) > 1e-8
    for name in full_grad:
        if name != 'modules_actor':
            assert_tree_equal(full_grad[name], cut_grad[name])
    assert_tree_equal(jax.grad(lambda p: full.critic_loss(data, p, key)[0])(full.network.params),
                      jax.grad(lambda p: cut.critic_loss(data, p, key)[0])(cut.network.params))


def test_actual_adam_update_changes_actor_only_and_missing_flag_retains_legacy_update():
    full = make_agent(); cut = make_agent(adapter_actor_q_gradient='remove_direction_radial')
    config = dict(full.config); config.pop('adapter_actor_q_gradient')
    legacy = full.replace(config=config)
    data = batch()
    updated, info = full.update(data)
    cut_updated, cut_info = cut.update(data)
    legacy_updated, legacy_info = legacy.update(data)
    assert_tree_equal((updated.network, updated.rng, info), (legacy_updated.network, legacy_updated.rng, legacy_info))
    assert_tree_equal(updated.rng, cut_updated.rng)
    assert int(updated.network.step) == int(cut_updated.network.step) == int(full.network.step)+1
    for name in updated.network.params:
        if name != 'modules_actor':
            assert_tree_equal(updated.network.params[name], cut_updated.network.params[name])
            assert_tree_equal(updated.network.opt_state[0].mu[name], cut_updated.network.opt_state[0].mu[name])
            assert_tree_equal(updated.network.opt_state[0].nu[name], cut_updated.network.opt_state[0].nu[name])
    differences = [np.max(np.abs(a-b)) for a,b in zip(jax.tree_util.tree_leaves(updated.network.params['modules_actor']),
                                                    jax.tree_util.tree_leaves(cut_updated.network.params['modules_actor']), strict=True)]
    assert max(differences) > 1e-8
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves((cut_updated.network, cut_info)))


@pytest.mark.parametrize('changes', [dict(adapter_representation='none'), dict(adapter_critic_space='env'),
    dict(adapter_execution_map='cartesian_aux'), dict(horizon_length=5), dict(env_action_dim=7),
    dict(critic_action_dim=5), dict(actor_update_mode='replay_awr'), dict(bc_alpha=1.),
    dict(ds_regularizer='semantic_product'), dict(ds_phase_regularization='reward_stage_translation'),
    dict(ds_stage_speed_constraint='reward_stage'), dict(ds_separate_alpha=True), dict(ds_moment_controlled=True)])
def test_unsupported_interfaces_rejected_before_network_initialization(changes):
    config, _ = configured(adapter_actor_q_gradient='remove_direction_radial', **changes)
    config['action_dim'] = 6
    with pytest.raises(ValueError, match='Actor-Q radial intervention requires'):
        validate_actor_q_gradient(config)


def test_unknown_mode_and_wrong_action_width_rejected():
    with pytest.raises(ValueError, match='Unknown adapter_actor_q_gradient'):
        validate_actor_q_gradient(dict(adapter_actor_q_gradient='typo'))
    with pytest.raises(ValueError, match='six policy coordinates'):
        remove_direction_radial_q_gradient(jnp.ones(5))
    config, example = configured(adapter_actor_q_gradient='remove_direction_radial')
    with pytest.raises(ValueError, match='six-dimensional replay actions'):
        ACRLPDAgent.create(29, np.ones(7, np.float32), example[:5], config)
