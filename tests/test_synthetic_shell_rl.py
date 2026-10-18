import jax
import jax.numpy as jnp
import numpy as np

from research.shell import (
    CONDITIONS,
    APPROACH_RADIUS,
    DS_RADIAL_STD,
    DS_TANGENTIAL_STD,
    initial_parameters,
    parameter_diagnostics,
    parameter_count,
    sample_policy,
    set_target_angle,
    shell_hit,
)


def test_structured_full_covariance_matches_ds_local_volume():
    full_cov = next(
        condition
        for condition in CONDITIONS
        if condition.name == "Cartesian full-cov oracle"
    )
    params = initial_parameters(full_cov, 1)
    scales = np.exp(np.asarray(params["log_scale"][0, 0]))

    np.testing.assert_allclose(scales, [DS_RADIAL_STD, DS_TANGENTIAL_STD])
    np.testing.assert_allclose(np.prod(scales), 0.08**2)


def test_policy_parameter_counts_expose_stronger_full_covariance_baseline():
    counts = {condition.name: parameter_count(condition) for condition in CONDITIONS}

    assert counts["Cartesian stage"] == counts["DS stage"] == 8
    assert counts["Cartesian full-cov oracle"] == 10


def test_shell_hit_accepts_target_and_rejects_wrong_direction():
    target = APPROACH_RADIUS * jnp.asarray(
        [[jnp.cos(jnp.deg2rad(30.0)), jnp.sin(jnp.deg2rad(30.0))]]
    )
    wrong = jnp.asarray([[APPROACH_RADIUS, 0.0]])

    np.testing.assert_array_equal(shell_hit(target, APPROACH_RADIUS, 0.025), [1.0])
    np.testing.assert_array_equal(shell_hit(wrong, APPROACH_RADIUS, 0.025), [0.0])


def test_all_policy_families_sample_finite_actions_and_log_probs():
    for condition in CONDITIONS:
        params = jax.tree_util.tree_map(lambda value: value[0], initial_parameters(condition, 1))
        actions, log_probs = sample_policy(
            params, jax.random.PRNGKey(123), condition, batch_size=32
        )
        assert actions.shape == (2, 32, 2)
        assert log_probs.shape == (2, 32)
        assert np.all(np.isfinite(np.asarray(actions)))
        assert np.all(np.isfinite(np.asarray(log_probs)))


def test_parameter_diagnostics_are_finite_for_each_family():
    for condition in CONDITIONS:
        params = initial_parameters(condition, 1)
        row = parameter_diagnostics(params, condition, seeds=(123,))[0]
        for key, value in row.items():
            if key.endswith("angular_std") and condition.family != "direction_speed":
                assert np.isnan(value)
            elif key not in ("condition", "seed"):
                assert np.isfinite(value)


def test_target_angle_can_define_axis_aligned_counterexample():
    import research.shell as module

    original = float(np.rad2deg(module.TARGET_ANGLE))
    try:
        set_target_angle(0.0)
        np.testing.assert_allclose(module.TARGET_ANGLE, 0.0)
    finally:
        set_target_angle(original)
