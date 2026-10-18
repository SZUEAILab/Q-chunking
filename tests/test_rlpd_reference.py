"""Protocol and scheduling checks independent of the optional upstream checkout."""

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.rlpd_reference import RLPDReferenceAgent, get_config


def make_agent(**overrides):
    config = get_config().to_dict()
    config.update(horizon_length=1, actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8))
    config.update(overrides)
    return RLPDReferenceAgent.create(42, jnp.zeros(7), jnp.zeros(5), config)


def batch(size=12):
    rng = np.random.default_rng(510)
    return dict(observations=jnp.asarray(rng.normal(size=(size, 7)), dtype=jnp.float32),
                next_observations=jnp.asarray(rng.normal(size=(size, 7)), dtype=jnp.float32),
                actions=jnp.asarray(rng.uniform(-.5, .5, size=(size, 5)), dtype=jnp.float32),
                rewards=jnp.asarray(rng.integers(-3, 1, size), dtype=jnp.float32),
                masks=jnp.asarray(rng.integers(0, 2, size), dtype=jnp.float32))


def assert_tree_equal(first, second):
    for a, b in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True):
        np.testing.assert_array_equal(a, b)


def test_utd_updates_critic_many_times_and_actor_temperature_once():
    agent = make_agent()
    updated, info = agent.update(batch(), utd_ratio=3)
    assert int(updated.critic.step) == 3
    assert int(updated.actor.step) == int(updated.temp.step) == 1
    for value in info.values():
        assert np.isfinite(value).all()


def test_target_tracks_new_critic_after_update():
    agent = make_agent()
    updated, _ = agent.update_critic(batch())
    for old, online, target in zip(jax.tree_util.tree_leaves(agent.target_critic.params),
                                   jax.tree_util.tree_leaves(updated.critic.params),
                                   jax.tree_util.tree_leaves(updated.target_critic.params), strict=True):
        np.testing.assert_allclose(target, .005 * online + .995 * old, rtol=1e-6, atol=1e-7)


def test_sequence_bridge_matches_flat_batches():
    agent = make_agent()
    flat = batch()
    sequence = {"observations": flat["observations"].reshape(3, 4, 7)}
    for name in ("next_observations", "actions", "rewards", "masks"):
        sequence[name] = flat[name].reshape((3, 4, 1) + flat[name].shape[1:])
    expected, info = agent.update(flat, utd_ratio=3)
    actual, actual_info = agent.batch_update(sequence)
    assert_tree_equal(flax.serialization.to_state_dict(expected), flax.serialization.to_state_dict(actual))
    assert_tree_equal(info, actual_info)


def test_external_sampling_is_reproducible_and_does_not_advance_learner_rng():
    agent = make_agent()
    rng = np.asarray(agent.rng).copy()
    first = agent.sample_actions(jnp.ones(7), rng=jax.random.PRNGKey(83))
    second = agent.sample_actions(jnp.ones(7), rng=jax.random.PRNGKey(83))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(agent.rng, rng)
    assert np.max(np.abs(first)) <= 1


def test_checkpoint_round_trip_preserves_independent_optimizers():
    agent = make_agent()
    updated, _ = agent.update(batch(), utd_ratio=3)
    restored = flax.serialization.from_bytes(agent, flax.serialization.to_bytes(updated))
    assert_tree_equal(flax.serialization.to_state_dict(updated), flax.serialization.to_state_dict(restored))
    again, _ = restored.update(batch(), utd_ratio=3)
    assert int(again.critic.step) == 6
    assert int(again.actor.step) == int(again.temp.step) == 2


@pytest.mark.parametrize("overrides", [dict(horizon_length=5), dict(num_min_qs=11),
                                      dict(adapter_representation="cartesian"),
                                      dict(actor_update_mode="replay_awr"), dict(fixed_alpha=1.)])
def test_unsupported_interventions_fail_before_training(overrides):
    with pytest.raises(ValueError):
        make_agent(**overrides)
