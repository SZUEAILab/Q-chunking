import jax.numpy as jnp
import numpy as np

from direction_speed.postprocessors import L2BallProjectionPostprocessor


def test_projection_leaves_inside_ball_and_nonspatial_dims_unchanged():
    projector = L2BallProjectionPostprocessor()
    action = np.array([0.3, 0.4, 0.0, -0.25, 0.75])

    actual = projector.compose(action)

    np.testing.assert_allclose(actual, action)


def test_projection_maps_outside_point_to_unit_sphere():
    projector = L2BallProjectionPostprocessor()
    action = np.array([1.0, 1.0, 1.0, -0.25, 0.75])

    actual = projector.compose(action)

    np.testing.assert_allclose(np.linalg.norm(actual[:3]), 1.0)
    np.testing.assert_allclose(actual[3:], action[3:])
    np.testing.assert_allclose(actual[:3], np.ones(3) / np.sqrt(3.0))


def test_projection_reshapes_action_chunks_and_matches_jax():
    projector = L2BallProjectionPostprocessor()
    chunk = np.array(
        [1.0, 1.0, 0.0, 0.1, -0.1, 0.1, 0.2, 0.3, -0.2, 0.2]
    )

    actual = projector.compose_policy_output(chunk, policy_action_dim=5)
    actual_jax = np.asarray(projector.compose_jax(jnp.asarray(actual)))

    assert actual.shape == (2, 5)
    np.testing.assert_allclose(np.linalg.norm(actual[0, :3]), 1.0)
    np.testing.assert_allclose(actual[1], chunk.reshape(2, 5)[1])
    np.testing.assert_allclose(actual_jax, actual, rtol=1e-6, atol=1e-6)
