import jax
import jax.numpy as jnp
import numpy as np

from research.shell_offpolicy import Replay, policy_log_prob, stage_targets
from research.shell import CONDITIONS, initial_parameters, sample_policy


def test_stage_targets_mask_bootstrap_when_discovery_fails():
    discovery = jnp.asarray([[0.0, 1.0]])
    terminal = jnp.asarray([[0.0, 1.0]])
    next_value = jnp.asarray([[0.9, 0.5]])
    stage0, stage1 = stage_targets(discovery, terminal, next_value, 0.99)

    np.testing.assert_allclose(stage0[0, 0], 0.0)
    np.testing.assert_allclose(stage0[0, 1], (1.0 + 0.99 * 0.5) / 1.99)
    np.testing.assert_allclose(stage1, terminal)


def test_replay_preserves_executed_actions_and_stage_labels():
    replay = Replay(num_seeds=2, capacity=8)
    actions = np.arange(2 * 2 * 3 * 2, dtype=np.float32).reshape(2, 2, 3, 2)
    discovery = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    terminal = np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.float32)
    replay.add(actions, discovery, terminal)

    assert replay.size == 3
    np.testing.assert_allclose(replay.actions[:, :3], np.transpose(actions, (0, 2, 1, 3)))
    np.testing.assert_allclose(replay.discovery[:, :3], discovery)
    np.testing.assert_allclose(replay.terminal[:, :3], terminal)


def test_replay_action_log_prob_matches_policy_sampler():
    names = ("Cartesian stage", "Cartesian full-cov oracle", "DS stage")
    for offset, name in enumerate(names):
        condition = next(item for item in CONDITIONS if item.name == name)
        params = initial_parameters(condition, 1)
        actions, expected = sample_policy(
            jax.tree_util.tree_map(lambda value: value[0], params),
            jax.random.PRNGKey(100 + offset),
            condition,
            32,
        )
        actual = policy_log_prob(params, actions[None], condition)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
