import jax.numpy as jnp
import numpy as np

from agents.acrlpd import normalized_advantage_weights
from utils.datasets import Dataset


def make_dataset(stage_rewards):
    size = len(stage_rewards)
    values = np.arange(size, dtype=np.float32)
    return Dataset.create(
        observations=values[:, None],
        actions=np.zeros((size, 2), dtype=np.float32),
        rewards=np.zeros(size, dtype=np.float32),
        masks=np.ones(size, dtype=np.float32),
        terminals=np.zeros(size, dtype=np.float32),
        next_observations=(values + 1)[:, None],
        stage_rewards=np.asarray(stage_rewards, dtype=np.float32),
    )


def test_advantage_weights_are_monotonic_capped_and_mean_one():
    advantages = jnp.asarray([-10.0, 0.0, 100.0, 200.0])
    weights = np.asarray(
        normalized_advantage_weights(
            advantages, temperature=1.0, max_weight=5.0
        )
    )
    assert np.all(np.diff(weights) >= 0.0)
    np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-6)
    np.testing.assert_allclose(weights[-1], weights[-2], atol=1e-6)


def test_stage_balanced_sequence_sampling_oversamples_rare_stages():
    dataset = make_dataset([-3.0] * 90 + [-2.0] * 9 + [-1.0])
    np.random.seed(0)
    batch = dataset.sample_sequence(
        batch_size=96,
        sequence_length=1,
        discount=0.99,
        balance_key="stage_rewards",
    )
    values, counts = np.unique(batch["stage_rewards"], return_counts=True)
    np.testing.assert_allclose(values, [-3.0, -2.0, -1.0])
    np.testing.assert_allclose(counts, [32, 32, 32])
