import math
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from direction_speed.distributions.integrated import (
    IntegratedDirectionSpeedBijector,
    IntegratedDirectionSpeedNormal,
    ReferenceMeasureActionDistribution,
)
from direction_speed.config import configure_direction_speed
from direction_speed.specs import (
    ActionGroup,
    ActionGroupKind,
    RadialBound,
)
from rlpd_networks import MLP


tfd = tfp.distributions


def _distribution(
    bound=RadialBound.SPHERE, kind=ActionGroupKind.SPHERICAL
):
    groups = (
        ActionGroup(kind, 3, 1.0, bound),
        ActionGroup(ActionGroupKind.SCALAR, 1),
        ActionGroup(ActionGroupKind.SCALAR, 1),
    )
    base = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(5), scale_diag=jnp.ones(5)
    )
    return ReferenceMeasureActionDistribution(
        distribution=base,
        bijector=IntegratedDirectionSpeedBijector(groups),
    )


def test_semantic_component_log_ratios_sum_to_total():
    distribution = _distribution()
    action = distribution.raw_to_policy(jnp.asarray([0.2, -0.3, -0.7, 0.1, -0.2]))

    components = distribution.semantic_component_log_ratios(action)
    total = distribution.reference_log_prob(action, "semantic_product")

    np.testing.assert_allclose(
        total,
        components["direction"] + components["speed"] + components["scalar"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_separate_trunks_isolate_direction_replay_gradient():
    groups = (
        ActionGroup(ActionGroupKind.SPHERICAL, 3, 1.0, RadialBound.SPHERE),
        ActionGroup(ActionGroupKind.SCALAR, 1),
        ActionGroup(ActionGroupKind.SCALAR, 1),
    )
    actor = IntegratedDirectionSpeedNormal(
        base_cls=partial(MLP, hidden_dims=(8,), activate_final=True),
        action_dim=5,
        action_groups=groups,
        separate_component_trunks=True,
    )
    observations = jnp.ones((4, 6))
    params = actor.init(jax.random.PRNGKey(0), observations)
    replay_actions = IntegratedDirectionSpeedBijector(groups).forward(
        jnp.zeros((4, 5))
    )

    def direction_loss(candidate_params):
        distribution = actor.apply(candidate_params, observations)
        return -distribution.semantic_component_log_ratios(replay_actions)[
            "direction"
        ].mean()

    flat_grads = flax.traverse_util.flatten_dict(
        jax.grad(direction_loss)(params), sep="/"
    )
    direction_norm = sum(
        float(jnp.linalg.norm(value))
        for path, value in flat_grads.items()
        if "Direction" in path
    )
    unrelated_norm = sum(
        float(jnp.linalg.norm(value))
        for path, value in flat_grads.items()
        if "Speed" in path or "Scalar" in path
    )
    assert direction_norm > 0.0
    np.testing.assert_allclose(unrelated_norm, 0.0, atol=1e-10)


def test_sphere_action_volume_and_semantic_references_have_expected_ratio():
    distribution = _distribution()
    raw = jnp.asarray([0.2, -0.3, -0.7, 0.1, -0.2])
    action = distribution.raw_to_policy(raw)
    radial_fraction = float(jnp.linalg.norm(action[:3]))

    action_volume = distribution.reference_log_prob(action, "action_volume")
    semantic = distribution.reference_log_prob(action, "semantic_product")

    expected_difference = -2.0 * math.log(radial_fraction) - math.log(3.0)
    np.testing.assert_allclose(
        action_volume - semantic, expected_difference, rtol=1e-5, atol=1e-5
    )


def test_stereographic_reference_ratios_are_finite_and_have_expected_ratio():
    distribution = _distribution(kind=ActionGroupKind.STEREOGRAPHIC)
    raw = jnp.asarray([0.4, -0.7, -0.2, 0.1, -0.2])
    action = distribution.raw_to_policy(raw)
    radial_fraction = float(jnp.linalg.norm(action[:3]))

    components = distribution.semantic_component_log_ratios(action)
    semantic = distribution.reference_log_prob(action, "semantic_product")
    action_volume = distribution.reference_log_prob(action, "action_volume")

    assert all(np.isfinite(np.asarray(value)) for value in components.values())
    np.testing.assert_allclose(
        semantic, sum(components.values()), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        action_volume - semantic,
        -2.0 * math.log(radial_fraction) - math.log(3.0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_reference_log_volumes_match_action_domains():
    sphere = _distribution(RadialBound.SPHERE).policy_transform
    cube = _distribution(RadialBound.CUBE).policy_transform

    np.testing.assert_allclose(
        sphere.action_log_volume(), math.log(16.0 * math.pi / 3.0), atol=1e-7
    )
    np.testing.assert_allclose(cube.action_log_volume(), math.log(32.0), atol=1e-7)
    semantic = sphere.semantic_log_volume_components()
    np.testing.assert_allclose(semantic["direction"], math.log(4.0 * math.pi))
    np.testing.assert_allclose(semantic["speed"], 0.0)
    np.testing.assert_allclose(semantic["scalar"], math.log(4.0))


def test_semantic_reference_with_separate_temperatures_is_validated():
    config = {"use_integrated_ds": False, "ds_max_speed": 1.0}

    resolved = configure_direction_speed(
        ds_mode="integrated",
        ds_speed_bound="sphere",
        ds_integrated_type="spherical",
        ds_regularizer="semantic_product",
        ds_separate_alpha=True,
        config=config,
        agent_name="acrlpd",
    )

    assert resolved.separate_alpha
    assert config["ds_regularizer"] == "semantic_product"
    assert config["ds_separate_alpha"] is True


def test_projected_tangent_pipeline_resolves_with_action_volume_reference():
    config = {"use_integrated_ds": False, "ds_max_speed": 1.0}
    resolved = configure_direction_speed(
        ds_mode="integrated",
        ds_speed_bound="sphere",
        ds_integrated_type="projected_tangent",
        ds_regularizer="action_volume",
        config=config,
        agent_name="acrlpd",
    )
    assert resolved.use_integrated
    assert config["ds_integrated_type"] == "projected_tangent"
    assert "ProjectedTangentPolicyBijector" in config["action_pipeline_components"][
        "policy_bijector"
    ]


def test_separate_temperatures_reject_action_volume_reference():
    config = {"use_integrated_ds": False, "ds_max_speed": 1.0}

    with np.testing.assert_raises_regex(ValueError, "semantic_product"):
        configure_direction_speed(
            ds_mode="integrated",
            ds_speed_bound="sphere",
            ds_integrated_type="spherical",
            ds_regularizer="action_volume",
            ds_separate_alpha=True,
            config=config,
            agent_name="acrlpd",
        )
