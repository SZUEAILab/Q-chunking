import jax
import jax.numpy as jnp
import numpy as np

from train_joint_contextual_policy import (
    CONDITIONS,
    FEATURE_DIM,
    MEAN_FRAME_CONDITION,
    MIN_LOG_SCALE,
    PROJECTED_TANGENT_CONDITION,
    RADIUS,
    anchors,
    context_features,
    initial_parameters,
    policy_means,
    sample_single,
)


def test_context_features_are_high_dimensional_and_periodic():
    angles = jnp.asarray([0.1, 1.7])
    assert context_features(angles).shape == (2, FEATURE_DIM)
    np.testing.assert_allclose(
        context_features(angles), context_features(angles + 2 * np.pi), atol=1e-5
    )


def test_all_conditions_share_the_initial_learnable_mean():
    angles = jnp.asarray([0.0, 0.7, 2.1])
    features = context_features(angles)
    expected = anchors(angles)
    for condition in CONDITIONS:
        params = jax.tree_util.tree_map(
            lambda value: value[0], initial_parameters(condition, [800])
        )
        np.testing.assert_allclose(
            policy_means(params, features, angles), expected, atol=1e-7
        )


def test_direction_speed_joint_sampler_is_finite():
    condition = next(item for item in CONDITIONS if item.name == "DS global")
    params = jax.tree_util.tree_map(
        lambda value: value[0], initial_parameters(condition, [800])
    )
    angles = jnp.deg2rad(jnp.arange(0.0, 360.0, 30.0))
    actions, log_probs, means = sample_single(
        params,
        jax.random.PRNGKey(0),
        condition,
        context_features(angles),
        angles,
        np.deg2rad(30.0),
        256,
    )
    assert actions.shape == (12, 256, 2)
    assert means.shape == (12, 2)
    assert np.isfinite(np.asarray(log_probs)).all()
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(means), axis=-1), RADIUS, atol=1e-6
    )


def test_primary_cartesian_control_has_more_parameters_than_ds_global():
    counts = {condition.name: condition.parameter_count for condition in CONDITIONS}
    assert counts["Cartesian full-cov state"] > counts["DS global"]


def test_cartesian_mean_frame_counterfactual_is_finite_and_capacity_matched():
    ds = next(item for item in CONDITIONS if item.name == "DS global")
    assert MEAN_FRAME_CONDITION.parameter_count == ds.parameter_count
    params = jax.tree_util.tree_map(
        lambda value: value[0], initial_parameters(MEAN_FRAME_CONDITION, [900])
    )
    angles = jnp.deg2rad(jnp.arange(0.0, 360.0, 30.0))
    actions, log_probs, means = sample_single(
        params,
        jax.random.PRNGKey(0),
        MEAN_FRAME_CONDITION,
        context_features(angles),
        angles,
        np.deg2rad(30.0),
        512,
    )
    assert actions.shape == (12, 512, 2)
    assert means.shape == (12, 2)
    assert np.isfinite(np.asarray(log_probs)).all()


def test_projected_tangent_flow_is_finite_curved_and_capacity_matched():
    ds = next(item for item in CONDITIONS if item.name == "DS global")
    assert PROJECTED_TANGENT_CONDITION.parameter_count == ds.parameter_count
    params = jax.tree_util.tree_map(
        lambda value: value[0], initial_parameters(PROJECTED_TANGENT_CONDITION, [1100])
    )
    angles = jnp.deg2rad(jnp.arange(0.0, 360.0, 30.0))
    actions, log_probs, means = sample_single(
        params,
        jax.random.PRNGKey(2),
        PROJECTED_TANGENT_CONDITION,
        context_features(angles),
        angles,
        np.deg2rad(30.0),
        1024,
    )
    assert actions.shape == (12, 1024, 2)
    assert np.isfinite(np.asarray(log_probs)).all()
    # Projection bends tangent proposals back to the shell: radial variation
    # stays at the explicitly small radial scale instead of growing with the
    # tangent offset as it does for the straight mean-frame Gaussian.
    radial_deviation = np.linalg.norm(np.asarray(actions), axis=-1) - np.linalg.norm(
        np.asarray(means), axis=-1
    )[:, None]
    np.testing.assert_allclose(
        np.std(radial_deviation), np.exp(MIN_LOG_SCALE), rtol=0.08
    )
