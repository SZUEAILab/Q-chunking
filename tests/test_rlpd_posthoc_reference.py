from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.rlpd_posthoc_reference import RLPDPosthocReferenceAgent, get_config
from direction_speed.adapter import DirectionSpeedAdapter
from direction_speed.wrappers import ActionExecutionPipeline


def make_agent(**overrides):
    config = get_config().to_dict()
    config.update(horizon_length=1, actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8))
    config.update(overrides)
    example = DirectionSpeedAdapter(speed_squash="tanh", speed_interpretation="log", speed_bound="sphere").decompose(np.zeros(5))
    return RLPDPosthocReferenceAgent.create(43, jnp.zeros(7), example, config)


def batch():
    rng = np.random.default_rng(20261024)
    actions = rng.uniform(-1, 1, (12, 5)).astype("float32")
    actions[0] = 0.  # Canonical replay log-speed can lie outside actor tanh support.
    adapter = DirectionSpeedAdapter(speed_squash="tanh", speed_interpretation="log", speed_bound="sphere")
    return dict(observations=jnp.asarray(rng.normal(size=(12, 7)), dtype=jnp.float32),
                next_observations=jnp.asarray(rng.normal(size=(12, 7)), dtype=jnp.float32),
                actions=jnp.asarray(adapter.decompose(actions)), rewards=jnp.zeros(12), masks=jnp.ones(12))


def assert_state_equal(first, second):
    for a, b in zip(jax.tree_util.tree_leaves(flax.serialization.to_state_dict(first)),
                    jax.tree_util.tree_leaves(flax.serialization.to_state_dict(second)), strict=True):
        np.testing.assert_array_equal(a, b)


def test_policy_critic_and_environment_dimensions_remain_distinct():
    agent = make_agent()
    assert isinstance(agent, RLPDPosthocReferenceAgent)
    assert agent.config["env_action_dim"] == 5
    assert agent.config["policy_action_dim"] == agent.config["critic_action_dim"] == 6
    assert agent.config["target_entropy"] == -3.
    adapter = DirectionSpeedAdapter(speed_squash="tanh", speed_interpretation="log", speed_bound="sphere")
    pipeline = ActionExecutionPipeline(agent, adapter, 6)
    key = jax.random.PRNGKey(73)
    latent = np.asarray(agent.sample_actions(jnp.zeros(7), rng=key))
    executed = np.asarray(pipeline.sample_actions(jnp.zeros(7), rng=key))
    assert latent.shape == (6,) and executed.shape == (1, 5)
    np.testing.assert_array_equal(executed, adapter.compose(latent[None]))
    np.testing.assert_array_equal(executed[0, 3:], latent[4:])
    assert np.isfinite(executed).all() and np.max(np.abs(executed)) <= 1


def test_utd_schedule_sequence_bridge_and_checkpoint_with_encoded_replay():
    agent = make_agent(); data = batch()
    assert np.asarray(data["actions"])[0, 3] < -1
    sequence = dict(observations=data["observations"].reshape(3, 4, 7))
    for name in ("actions", "next_observations", "rewards", "masks"):
        sequence[name] = data[name].reshape((3, 4, 1) + data[name].shape[1:])
    updated, info = agent.batch_update(sequence)
    flat, _ = agent.update(data, utd_ratio=3)
    assert_state_equal(updated, flat)
    assert int(updated.critic.step) == 3
    assert int(updated.actor.step) == int(updated.temp.step) == 1
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())
    restored = flax.serialization.from_bytes(agent, flax.serialization.to_bytes(updated))
    assert_state_equal(updated, restored)
    again, _ = restored.batch_update(sequence)
    assert int(again.critic.step) == 6 and int(again.actor.step) == int(again.temp.step) == 2


def test_common_numeric_entropy_target_is_explicit_and_sampling_keeps_learner_rng():
    agent = make_agent(target_entropy=-2.5)
    assert agent.config["target_entropy"] == -2.5
    before = np.asarray(agent.rng).copy()
    first = agent.sample_actions(jnp.zeros(7), rng=jax.random.PRNGKey(84))
    second = agent.sample_actions(jnp.zeros(7), rng=jax.random.PRNGKey(84))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(agent.rng, before)


@pytest.mark.parametrize("overrides", [dict(adapter_speed_squash="sigmoid"), dict(adapter_speed_interpretation="direct"),
    dict(ds_speed_bound="cube"), dict(adapter_critic_space="env"), dict(adapter_critic_space="canonical"),
    dict(env_action_dim=6), dict(policy_action_dim=5), dict(horizon_length=5), dict(ds_moment_controlled=True),
    dict(ds_regularizer="semantic_product"), dict(actor_update_mode="replay_awr"), dict(bc_alpha=1.)])
def test_unsupported_recipe_is_rejected_before_learning(overrides):
    with pytest.raises(ValueError): make_agent(**overrides)


def test_driver_rejects_changed_recipe_before_dataset_or_environment_creation():
    from main_rlpd_posthoc_reference import validate_driver
    flags = SimpleNamespace(agent=get_config(), horizon_length=1, ds_mode="adapter", stage_balanced_sampling=False,
        ds_speed_bound="config", ds_adapter_representation="cartesian", ds_adapter_speed_squash="tanh",
        ds_adapter_speed_interpretation="log", ds_adapter_critic_space="latent", discount=.99)
    validate_driver(flags)
    for name, value in [("ds_mode", "none"), ("ds_speed_bound", "cube"), ("ds_adapter_critic_space", "env"),
                        ("horizon_length", 5), ("discount", .9), ("stage_balanced_sampling", True)]:
        with pytest.raises(ValueError): validate_driver(SimpleNamespace(**dict(vars(flags), **{name: value})))
