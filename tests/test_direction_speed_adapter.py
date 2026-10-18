import unittest

import jax.numpy as jnp
import numpy as np

from direction_speed.adapter import DirectionSpeedAdapter


class DirectionSpeedAdapterTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.actions = rng.uniform(-0.9, 0.9, size=(128, 5)).astype(np.float32)
        # Keep XYZ away from the undefined direction at exactly zero.
        self.actions[:, 2] += 0.05

    def test_direct_cube_round_trip_for_all_representations(self):
        for representation in ("cartesian", "spherical", "stereographic"):
            with self.subTest(representation=representation):
                adapter = DirectionSpeedAdapter(
                    representation=representation,
                    speed_squash="sigmoid",
                    speed_interpretation="direct",
                    speed_bound="cube",
                )
                restored = adapter.compose(adapter.decompose(self.actions))
                np.testing.assert_allclose(restored, self.actions, atol=2e-5, rtol=2e-5)

    def test_legacy_log_cube_round_trip(self):
        adapter = DirectionSpeedAdapter(
            representation="cartesian",
            speed_squash="tanh",
            speed_interpretation="log",
            speed_bound="cube",
        )
        restored = adapter.compose(adapter.decompose(self.actions))
        np.testing.assert_allclose(restored, self.actions, atol=2e-6, rtol=2e-6)

    def test_cube_bound_reaches_action_cube_boundary(self):
        directions = np.asarray(
            [[1.0, 1.0, 1.0], [0.2, -0.8, 0.4], [-1.0, 0.1, 0.2]],
            dtype=np.float32,
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        policy_actions = np.concatenate(
            [directions, np.ones((len(directions), 1), dtype=np.float32)], axis=-1
        )
        adapter = DirectionSpeedAdapter(speed_bound="cube")
        xyz = adapter.compose(policy_actions)[:, :3]
        np.testing.assert_allclose(np.max(np.abs(xyz), axis=-1), 1.0, atol=1e-6)

    def test_numpy_and_jax_composition_agree(self):
        adapter = DirectionSpeedAdapter(
            representation="spherical",
            speed_squash="sigmoid",
            speed_interpretation="direct",
            speed_bound="sphere",
        )
        policy_actions = adapter.decompose(self.actions)
        numpy_actions = adapter.compose(policy_actions)
        jax_actions = np.asarray(adapter.compose_jax(jnp.asarray(policy_actions)))
        np.testing.assert_allclose(jax_actions, numpy_actions, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
