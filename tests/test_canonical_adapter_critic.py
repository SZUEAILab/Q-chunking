"""Behavioral equivalence and update checks for the canonical critic ablation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.acrlpd import ACRLPDAgent, get_config
from direction_speed.adapter import DirectionSpeedAdapter
from direction_speed.config import configure_direction_speed


def make_agent(space, horizon=1, speed="log"):
    config = get_config().to_dict()
    config.update(actor_hidden_dims=(16, 16), value_hidden_dims=(16, 16),
                  num_qs=2, horizon_length=horizon, env_action_dim=5,
                  policy_action_dim=6, critic_action_dim=5 if space == "env" else 6)
    resolved = configure_direction_speed(
        ds_mode="adapter", ds_speed_bound="sphere",
        ds_adapter_speed_squash="tanh" if speed == "log" else "sigmoid",
        ds_adapter_speed_interpretation=speed, ds_adapter_critic_space=space,
        config=config, agent_name="acrlpd",
    )
    env_actions = np.array([0.2, -0.3, 0.4, 0.1, -0.2], dtype=np.float32)
    example = env_actions if space == "env" else resolved.adapter.decompose(env_actions)
    return ACRLPDAgent.create(7, np.ones(7, dtype=np.float32), example, config)


@pytest.mark.parametrize("horizon", [1, 5])
@pytest.mark.parametrize("speed", ["log", "direct"])
def test_same_executed_action_has_same_canonical_features(horizon, speed):
    agent = make_agent("canonical", horizon, speed)
    first = np.array([0.2, -0.3, 0.4, -0.5 if speed == "log" else 0.5, 0.1, -0.2])
    second = first.copy()
    second[:3] *= 1.7  # Same direction and executed action, different latent norm.
    a = np.tile(first, horizon)
    b = np.tile(second, horizon)
    np.testing.assert_allclose(agent._critic_action(a), agent._critic_action(b),
                               rtol=2e-6, atol=2e-6)
    gradient = jax.grad(lambda x: jnp.sum(agent._critic_action(x) ** 2))(jnp.asarray(a))
    assert np.isfinite(gradient).all()
    # Radial direction scaling is a null direction of the canonical mapping.
    tangent = np.tile(np.r_[first[:3], np.zeros(3)], horizon)
    assert abs(float(jnp.dot(gradient, tangent))) < 2e-5


def test_canonical_queries_include_actual_execution_clipping():
    agent = make_agent("canonical")
    action = np.array([1., 1., 1., 1., 0.2, -0.2], dtype=np.float32)
    adapter = DirectionSpeedAdapter(speed_squash="tanh", speed_interpretation="log",
                                    speed_bound="sphere")
    expected = adapter.decompose(adapter.compose(action))
    np.testing.assert_allclose(agent._critic_action(action), expected, atol=2e-6)


def test_critic_intervention_preserves_actor_initialization():
    agents = [make_agent(space) for space in ("latent", "canonical", "env")]
    reference = jax.tree_util.tree_leaves(agents[0].network.params["modules_actor"])
    for agent in agents[1:]:
        leaves = jax.tree_util.tree_leaves(agent.network.params["modules_actor"])
        for before, after in zip(reference, leaves, strict=True):
            np.testing.assert_array_equal(before, after)
    first = jax.tree_util.tree_leaves(agents[0].network.params["modules_critic"])
    second = jax.tree_util.tree_leaves(agents[1].network.params["modules_critic"])
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_unimplemented_fql_canonical_interface_is_rejected():
    with pytest.raises(ValueError, match="only implemented for ACRLPD"):
        configure_direction_speed(
            ds_mode="adapter", ds_adapter_speed_squash="tanh",
            ds_adapter_speed_interpretation="log", ds_adapter_critic_space="canonical",
            config={}, agent_name="acfql",
        )


@pytest.mark.parametrize("horizon", [1, 5])
def test_canonical_actor_and_bellman_update_is_finite(horizon):
    agent = make_agent("canonical", horizon)
    adapter = DirectionSpeedAdapter(speed_squash="tanh", speed_interpretation="log",
                                    speed_bound="sphere")
    actions = np.tile(np.array([0.2, -0.3, 0.4, 0.1, -0.2], dtype=np.float32), (4, horizon, 1))
    batch = dict(observations=jnp.ones((4, 7)),
                 next_observations=jnp.ones((4, horizon, 7)),
                 actions=jnp.asarray(adapter.decompose(actions)),
                 rewards=jnp.zeros((4, horizon)), masks=jnp.ones((4, horizon)),
                 valid=jnp.ones((4, horizon)))
    updated, info = agent.update(batch)
    for value in jax.tree_util.tree_leaves(info):
        assert np.isfinite(value).all()
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(agent.network.params),
        jax.tree_util.tree_leaves(updated.network.params), strict=True))
