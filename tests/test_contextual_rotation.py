import jax
import jax.numpy as jnp
import numpy as np

from train_contextual_rotation import (
    CONDITIONS,
    RADIUS,
    angle_features,
    initial_parameters,
    sample_single,
    shell_hit,
)


def test_second_order_features_are_pi_periodic():
    angles = jnp.asarray([0.1, 1.2])
    np.testing.assert_allclose(
        angle_features(angles), angle_features(angles + np.pi), atol=1e-6
    )


def test_direction_speed_samples_preserve_narrow_radius():
    condition = next(item for item in CONDITIONS if item.name == "DS")
    params = {key: value[0] for key, value in initial_parameters(condition, 1).items()}
    angles = jnp.asarray([0.0, 1.0])
    actions, log_probs = sample_single(
        params, jax.random.PRNGKey(0), condition, angles, np.deg2rad(30), 10000
    )
    radii = np.linalg.norm(np.asarray(actions), axis=-1)
    assert abs(radii.mean() - RADIUS) < 0.002
    assert np.isfinite(np.asarray(log_probs)).all()


def test_shell_hit_is_rotation_equivariant():
    targets = jnp.asarray([0.2, 1.7])
    actions = RADIUS * jnp.stack((jnp.cos(targets), jnp.sin(targets)), axis=-1)
    hits = shell_hit(actions[:, None, :], targets)
    np.testing.assert_allclose(hits, 1.0)
