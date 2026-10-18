import numpy as np

from utils.datasets import Dataset


def test_sample_sequence_keeps_stage_reward_aligned_with_actor_observation():
    size = 8
    observations = np.arange(size, dtype=np.float32)[:, None]
    dataset = Dataset.create(
        observations=observations,
        next_observations=observations + 1,
        actions=np.zeros((size, 2), dtype=np.float32),
        rewards=np.zeros(size, dtype=np.float32),
        stage_rewards=np.arange(size, dtype=np.float32) - 3.0,
        masks=np.ones(size, dtype=np.float32),
        terminals=np.zeros(size, dtype=np.float32),
    )

    batch = dataset.sample_sequence(batch_size=16, sequence_length=2, discount=0.99)

    assert batch["stage_rewards"].shape == (16,)
    np.testing.assert_allclose(
        batch["stage_rewards"], batch["observations"][:, 0] - 3.0
    )
