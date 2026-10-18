import json

import jax
import numpy as np
import pytest

from agents.acrlpd import ACRLPDAgent, get_config
from direction_speed.config import configure_direction_speed
from research.action_adapter_diagnostics import (
    file_digest, measure_aliases, observation_panel, reference_policy_actions, verify_training_source,
)


def agent_with_space(space, horizon=1):
    config = get_config().to_dict()
    config.update(actor_hidden_dims=(16,), value_hidden_dims=(16,), num_qs=2,
                  horizon_length=horizon, env_action_dim=5, policy_action_dim=6,
                  critic_action_dim=5 if space == "env" else 6)
    resolved = configure_direction_speed(ds_mode="adapter", ds_speed_bound="sphere",
        ds_adapter_speed_squash="tanh", ds_adapter_speed_interpretation="log",
        ds_adapter_critic_space=space, config=config, agent_name="acrlpd")
    env = np.array([.2, -.3, .4, .1, -.2], dtype=np.float32)
    example = env if space == "env" else resolved.adapter.decompose(env)
    return ACRLPDAgent.create(7, np.ones(7, dtype=np.float32), example, config)


@pytest.mark.parametrize("space", ["canonical", "env"])
@pytest.mark.parametrize("horizon", [1, 5])
def test_equivalent_actions_have_invariant_values_and_radial_value_gradient(space, horizon):
    agent = agent_with_space(space, horizon)
    observations = np.random.default_rng(7).normal(size=(8, 7)).astype(np.float32)
    before = jax.tree_util.tree_leaves(agent.network.params)
    record = measure_aliases(agent, observations)
    assert record["valid_alias_pairs"] == 8
    assert record["q_alias_abs_mean"] < 1e-5
    assert record["radial_q_derivative_rms"] < 1e-5
    assert record["critic_feature_alias_abs_max"] < 2e-6
    for a, b in zip(before, jax.tree_util.tree_leaves(agent.network.params), strict=True):
        np.testing.assert_array_equal(a, b)


def test_latent_critic_can_distinguish_behaviorally_identical_actions():
    record = measure_aliases(agent_with_space("latent"), np.ones((8, 7), dtype=np.float32))
    assert record["valid_alias_pairs"] == 8
    assert record["max_executed_action_difference"] < 1e-6
    assert record["q_alias_abs_mean"] > 1e-5
    assert record["radial_q_derivative_rms"] > 1e-5


def test_normalization_floor_cases_do_not_become_false_alias_evidence():
    agent = agent_with_space("canonical")
    actions = np.tile(np.array([0., 0., 0., -.5, .2, -.3], dtype=np.float32), (4, 1))
    record = measure_aliases(agent, np.ones((4, 7), dtype=np.float32), actions=actions)
    assert record["valid_alias_pairs"] == 0
    assert record["excluded_pairs"] == 4
    assert record["q_alias_abs_mean"] is None
    assert record["finite_predictions_and_gradients"]
    json.dumps(record, allow_nan=False)


def test_state_panel_is_fixed_and_uses_only_nonterminal_observations(tmp_path):
    data = tmp_path / "data.npz"
    observations = np.arange(60, dtype=np.float32).reshape(20, 3)
    terminals = np.array([i % 3 == 0 for i in range(20)], dtype=np.float32)
    np.savez(data, observations=observations, terminals=terminals)
    kwargs = dict(count=8, seed=123, expected_digest=file_digest(data))
    first, identity = observation_panel(data, **kwargs)
    second, other = observation_panel(data, **kwargs)
    np.testing.assert_array_equal(first, second)
    assert identity == other
    assert not terminals[identity["indices"]].any()
    with pytest.raises(ValueError, match="data mismatch"):
        observation_panel(data, **{**kwargs, "expected_digest": "wrong"})


def test_source_identity_checks_content_not_a_git_label(tmp_path):
    path = tmp_path / "agent.py"
    path.write_text("original\n")
    manifest = dict(source_commit="frozen", python_sources=[dict(path="agent.py", sha256=file_digest(path))])
    assert verify_training_source(tmp_path, manifest) == "frozen"
    path.write_text("modified\n")
    with pytest.raises(ValueError, match="source mismatch"):
        verify_training_source(tmp_path, manifest)


def test_reference_probes_are_identical_across_critic_interventions():
    agents = [agent_with_space(space) for space in ("latent", "canonical", "env")]
    probes = [reference_policy_actions(agent.config, count=8) for agent in agents]
    for value in probes[1:]:
        np.testing.assert_array_equal(probes[0], value)
    assert np.max(np.abs(probes[0])) < 1
    different = reference_policy_actions(agents[0].config, count=8, seed=7)
    assert not np.array_equal(probes[0], different)
