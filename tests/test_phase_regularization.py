import jax.numpy as jnp
import numpy as np

from agents.acrlpd import reward_stage_weight


def test_reward_stage_weight_halves_translation_regularization_per_stage():
    rewards = jnp.asarray([-3.0, -2.0, -1.0])
    weights = reward_stage_weight(rewards, reward_floor=-3.0, decay=0.5)

    np.testing.assert_allclose(weights, [1.0, 0.5, 0.25])


def test_reward_stage_weight_clips_out_of_range_rewards():
    rewards = jnp.asarray([-5.0, -3.0, -1.0, 1.0])
    weights = reward_stage_weight(
        rewards,
        reward_floor=-3.0,
        decay=0.5,
        max_progress=2.0,
    )

    np.testing.assert_allclose(weights, [1.0, 1.0, 0.25, 0.25])
