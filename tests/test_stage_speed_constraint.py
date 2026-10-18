import jax.numpy as jnp
import numpy as np

from agents.acrlpd import (
    reward_stage_speed_cap,
    translation_speed_cap_statistics,
)


def test_reward_stage_speed_cap_decays_per_completed_stage():
    rewards = jnp.asarray([-3.0, -2.0, -1.0])
    caps = reward_stage_speed_cap(
        rewards,
        base_cap=1.0,
        decay=0.5,
        reward_floor=-3.0,
    )

    np.testing.assert_allclose(caps, [1.0, 0.5, 0.25])


def test_reward_stage_speed_cap_clips_progress():
    rewards = jnp.asarray([-5.0, -3.0, -1.0, 1.0])
    caps = reward_stage_speed_cap(
        rewards,
        base_cap=0.8,
        decay=0.5,
        reward_floor=-3.0,
        max_progress=2.0,
    )

    np.testing.assert_allclose(caps, [0.8, 0.8, 0.2, 0.2])


def test_translation_speed_cap_penalizes_only_excess_norm():
    actions = jnp.asarray(
        [
            [0.8, 0.0, 0.0, 0.2, -0.1],
            [0.8, 0.0, 0.0, 0.2, -0.1],
            [0.8, 0.0, 0.0, 0.2, -0.1],
        ]
    )
    stats = translation_speed_cap_statistics(
        actions,
        jnp.asarray([-3.0, -2.0, -1.0]),
        env_action_dim=5,
    )

    np.testing.assert_allclose(stats["translation_norm"], [0.8, 0.8, 0.8])
    np.testing.assert_allclose(stats["penalty"], [0.0, 0.09, 0.3025])
    np.testing.assert_allclose(stats["violation_fraction"], [0.0, 1.0, 1.0])


def test_translation_speed_cap_averages_over_action_chunk():
    # Two 5D actions with translation norms 0.4 and 0.8; cap is 0.5.
    actions = jnp.asarray([[0.4, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]])
    stats = translation_speed_cap_statistics(
        actions,
        jnp.asarray([-2.0]),
        env_action_dim=5,
        repetitions=2,
    )

    np.testing.assert_allclose(stats["translation_norm"], [0.6])
    np.testing.assert_allclose(stats["penalty"], [0.045])
    np.testing.assert_allclose(stats["violation_fraction"], [0.5])
