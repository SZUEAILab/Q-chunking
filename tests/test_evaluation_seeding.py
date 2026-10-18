import jax
import numpy as np

from evaluation import evaluate


class _Agent:
    def sample_actions(self, observations, rng):
        del observations
        return jax.random.uniform(rng, shape=(1,), minval=-1.0, maxval=1.0)


class _OneStepEnv:
    def __init__(self):
        self.reset_seeds = []
        self.actions = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return (
            np.zeros(2, dtype=np.float32),
            0.0,
            True,
            False,
            {"success": float(action[0] > 0)},
        )


def test_fixed_eval_seed_controls_policy_and_episode_randomness():
    first = _OneStepEnv()
    second = _OneStepEnv()

    evaluate(_Agent(), first, num_eval_episodes=3, action_dim=1, eval_seed=123)
    evaluate(_Agent(), second, num_eval_episodes=3, action_dim=1, eval_seed=123)

    assert first.reset_seeds == [123, 124, 125]
    assert second.reset_seeds == first.reset_seeds
    np.testing.assert_allclose(first.actions, second.actions)
