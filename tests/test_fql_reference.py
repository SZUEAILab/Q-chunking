import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.fql_reference import FQLReferenceAgent, get_config


def make_agent(**overrides):
    config = get_config().to_dict()
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), flow_steps=2)
    config.update(overrides)
    return FQLReferenceAgent.create(73, np.zeros(7, np.float32), np.zeros(5, np.float32), config)


def batch():
    rng = np.random.default_rng(20261028)
    return dict(observations=jnp.asarray(rng.normal(size=(4, 7)), dtype=jnp.float32),
                next_observations=jnp.asarray(rng.normal(size=(4, 7)), dtype=jnp.float32),
                actions=jnp.asarray(rng.uniform(-.9, .9, (4, 5)), dtype=jnp.float32),
                rewards=jnp.asarray([-1., 0., -2., 1.]), masks=jnp.asarray([1., 0., 1., 0.]))


def assert_state_equal(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(flax.serialization.to_state_dict(a)),
                    jax.tree_util.tree_leaves(flax.serialization.to_state_dict(b)), strict=True):
        np.testing.assert_array_equal(x, y)


def test_sampling_alias_shapes_bounds_and_external_rng_are_preserved():
    agent = make_agent()
    rng = jax.random.PRNGKey(61)
    before = np.asarray(agent.rng).copy()
    single = np.asarray(agent.sample_actions(jnp.zeros(7), rng=rng))
    assert single.shape == (5,) and np.max(np.abs(single)) <= 1
    np.testing.assert_array_equal(single, agent.sample_actions(jnp.zeros(7), seed=rng))
    assert agent.sample_actions(jnp.zeros((3, 7)), rng=rng).shape == (3, 5)
    np.testing.assert_array_equal(agent.rng, before)
    for kwargs in (dict(seed=rng, rng=rng), dict(temperature=.2, rng=rng), {}):
        with pytest.raises(ValueError): agent.sample_actions(jnp.zeros(7), **kwargs)


def test_sequence_updates_match_ordered_one_step_updates_and_checkpoint_resume():
    agent = make_agent(); data = batch()
    sequence = {'observations': jnp.stack([data['observations']] * 2)}
    for name in ('actions', 'next_observations', 'rewards', 'masks'):
        sequence[name] = jnp.stack([data[name][:, None]] * 2)
    sequence['valid'] = jnp.ones_like(sequence['masks'])
    stacked, info = agent.batch_update(sequence)
    sequential, _ = agent.update(data)
    sequential, _ = sequential.update(data)
    # lax.scan and individual jit calls may differ by float roundoff.
    for x, y in zip(jax.tree_util.tree_leaves(stacked.network.params),
                    jax.tree_util.tree_leaves(sequential.network.params), strict=True):
        np.testing.assert_allclose(x, y, atol=3e-7, rtol=3e-6)
    assert int(stacked.network.step) == 3
    assert all(np.isfinite(np.asarray(value)).all() for value in info.values())
    restored = flax.serialization.from_bytes(agent, flax.serialization.to_bytes(stacked))
    assert_state_equal(stacked, restored)
    first, _ = stacked.update(data); second, _ = restored.update(data)
    assert_state_equal(first, second)
    sequence['valid'] = sequence['valid'].at[0, 0, 0].set(0)
    with pytest.raises(ValueError, match='padded'): agent.batch_update(sequence)


def test_target_update_retains_official_pre_update_critic_semantics():
    agent = make_agent(tau=.2)
    changed = jax.tree_util.tree_map(lambda value: value + .7, agent.network.params['modules_target_critic'])
    params = dict(agent.network.params)
    params['modules_target_critic'] = changed
    agent = agent.replace(network=agent.network.replace(params=params))
    updated, _ = agent.update(batch())
    expected = jax.tree_util.tree_map(lambda current, target: .2 * current + .8 * target,
                                     agent.network.params['modules_critic'], changed)
    for x, y in zip(jax.tree_util.tree_leaves(updated.network.params['modules_target_critic']),
                    jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_allclose(x, y, atol=3e-7, rtol=3e-6)


@pytest.mark.parametrize('overrides', [dict(horizon_length=5), dict(adapter_representation='cartesian'),
    dict(use_integrated_ds=True), dict(use_cartesian_l2_projection=True), dict(encoder='impala_small'),
    dict(use_fourier_features=True), dict(weight_decay=.01), dict(actor_type='best-of-n'),
    dict(num_qs=10), dict(q_agg='invalid'), dict(flow_steps=0), dict(policy_action_dim=6)])
def test_unsupported_recipe_is_rejected_before_learning(overrides):
    with pytest.raises(ValueError): make_agent(**overrides)
