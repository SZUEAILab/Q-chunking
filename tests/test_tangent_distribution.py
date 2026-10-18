import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from direction_speed.distributions.tangent import (
    TangentDirectionSpeedNormal,
    TangentPolicyBijector,
    _normalize_direction_centers,
    _projected_direction_log_det,
    _projected_map,
    _tangent_basis,
)
from direction_speed.specs import build_action_groups


class IdentityBackbone(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        return inputs


class TangentGeometryTest(unittest.TestCase):
    def test_projected_tangent_direction_jacobian_matches_autodiff_area(self):
        center = jnp.asarray([0.0, 0.0, 1.0])
        offset = jnp.asarray([0.4, -0.2])
        jacobian = jax.jacfwd(lambda value: _projected_map(center, value))(offset)
        area = jnp.sqrt(jnp.linalg.det(jacobian.T @ jacobian))
        analytic = jnp.exp(_projected_direction_log_det(offset))
        np.testing.assert_allclose(area, analytic, rtol=2e-5, atol=2e-5)

    def test_direction_centers_have_no_fixed_north_offset(self):
        raw = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -3.0],
            ]
        )
        centers = _normalize_direction_centers(raw)
        np.testing.assert_allclose(
            centers,
            jnp.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            ),
            atol=1e-6,
        )

    def test_tangent_basis_is_orthonormal_near_old_switch(self):
        centers = _normalize_direction_centers(
            jnp.array(
                [
                    [0.44, 0.0, 0.898],
                    [0.43, 0.0, 0.903],
                    [0.01, -0.02, -0.999],
                ]
            )
        )
        basis_x, basis_y = _tangent_basis(centers)
        np.testing.assert_allclose(jnp.sum(basis_x * centers, axis=-1), 0.0, atol=2e-5)
        np.testing.assert_allclose(jnp.sum(basis_y * centers, axis=-1), 0.0, atol=2e-5)
        np.testing.assert_allclose(jnp.sum(basis_x * basis_y, axis=-1), 0.0, atol=2e-5)
        np.testing.assert_allclose(jnp.linalg.norm(basis_x, axis=-1), 1.0, atol=2e-5)
        np.testing.assert_allclose(jnp.linalg.norm(basis_y, axis=-1), 1.0, atol=2e-5)
        self.assertLess(float(jnp.linalg.norm(basis_x[0] - basis_x[1])), 0.05)

    def test_horizon_five_bijector_round_trip_is_finite(self):
        groups = build_action_groups(
            action_dim=5,
            horizon_length=5,
            action_chunking=True,
            max_speed=1.0,
            direction_type="tangent",
            speed_bound="cube",
        )
        centers = _normalize_direction_centers(
            jnp.array(
                [
                    [1.0, 0.2, 0.3],
                    [-0.4, 0.7, 0.5],
                    [0.1, -0.3, 0.9],
                    [-0.5, -0.4, 0.6],
                    [0.2, 0.1, -0.8],
                ]
            )
        )[None, ...]
        bijector = TangentPolicyBijector(centers, groups)
        raw = jnp.linspace(-0.7, 0.7, 25, dtype=jnp.float32)[None, :]
        action = bijector.forward(raw)
        restored = bijector.inverse(action)
        log_det = bijector.forward_log_det_jacobian(raw, event_ndims=1)

        self.assertEqual(action.shape, (1, 25))
        self.assertTrue(bool(jnp.all(jnp.isfinite(action))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(log_det))))
        np.testing.assert_allclose(restored, raw, atol=3e-4, rtol=3e-4)

    def test_projected_tangent_horizon_five_round_trip_is_finite(self):
        groups = build_action_groups(
            action_dim=5,
            horizon_length=5,
            action_chunking=True,
            max_speed=1.0,
            direction_type="tangent",
            speed_bound="sphere",
        )
        centers = _normalize_direction_centers(
            jnp.asarray(
                [[1.0, 0.2, 0.3], [-0.4, 0.7, 0.5], [0.1, -0.3, 0.9], [-0.5, -0.4, 0.6], [0.2, 0.1, -0.8]]
            )
        )[None, ...]
        bijector = TangentPolicyBijector(
            centers, groups, projected_tangent=True
        )
        raw = jnp.linspace(-0.7, 0.7, 25, dtype=jnp.float32)[None, :]
        action = bijector.forward(raw)
        restored = bijector.inverse(action)
        log_det = bijector.forward_log_det_jacobian(raw, event_ndims=1)
        self.assertTrue(bool(jnp.all(jnp.isfinite(action))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(log_det))))
        np.testing.assert_allclose(restored, raw, atol=4e-4, rtol=4e-4)


class TangentDistributionTest(unittest.TestCase):
    def test_speed_log_std_controls_change_only_radial_raw_scale(self):
        groups = build_action_groups(
            action_dim=5,
            horizon_length=1,
            action_chunking=False,
            max_speed=1.0,
            direction_type="tangent",
            speed_bound="sphere",
        )
        default_module = TangentDirectionSpeedNormal(
            base_cls=IdentityBackbone,
            action_dim=5,
            action_groups=groups,
        )
        narrow_module = TangentDirectionSpeedNormal(
            base_cls=IdentityBackbone,
            action_dim=5,
            action_groups=groups,
            speed_log_std_offset=-2.0,
            speed_log_std_max=-1.0,
            speed_mean_offset=1.25,
        )
        inputs = jnp.ones((4, 32), dtype=jnp.float32)
        variables = default_module.init(jax.random.key(42), inputs)
        default_std = default_module.apply(variables, inputs).raw_distribution.stddev()
        narrow_std = narrow_module.apply(variables, inputs).raw_distribution.stddev()

        np.testing.assert_allclose(narrow_std[..., :2], default_std[..., :2])
        np.testing.assert_allclose(narrow_std[..., 3:], default_std[..., 3:])
        self.assertTrue(bool(jnp.all(narrow_std[..., 2] < default_std[..., 2])))
        default_loc = default_module.apply(variables, inputs).raw_distribution.loc
        narrow_loc = narrow_module.apply(variables, inputs).raw_distribution.loc
        np.testing.assert_allclose(narrow_loc[..., :2], default_loc[..., :2])
        np.testing.assert_allclose(narrow_loc[..., 3:], default_loc[..., 3:])
        np.testing.assert_allclose(narrow_loc[..., 2] - default_loc[..., 2], 1.25)

    def test_projected_tangent_distribution_and_reference_are_finite(self):
        groups = build_action_groups(
            action_dim=5,
            horizon_length=1,
            action_chunking=False,
            max_speed=1.0,
            direction_type="tangent",
            speed_bound="sphere",
        )
        module = TangentDirectionSpeedNormal(
            base_cls=IdentityBackbone,
            action_dim=5,
            action_groups=groups,
            projected_tangent=True,
        )
        inputs = jnp.ones((32, 32), dtype=jnp.float32)
        variables = module.init(jax.random.key(40), inputs)
        distribution = module.apply(variables, inputs)
        actions = distribution.sample(seed=jax.random.key(41))
        log_prob = distribution.log_prob(actions)
        semantic = distribution.reference_log_prob(actions, "semantic_product")
        action_volume = distribution.reference_log_prob(actions, "action_volume")
        radial_fraction = jnp.linalg.norm(actions[..., :3], axis=-1)
        self.assertTrue(bool(jnp.all(jnp.isfinite(log_prob))))
        np.testing.assert_allclose(
            action_volume - semantic,
            -2.0 * jnp.log(radial_fraction) - jnp.log(3.0),
            rtol=3e-4,
            atol=3e-4,
        )

    def test_horizon_five_has_independent_angular_scales(self):
        groups = build_action_groups(
            action_dim=5,
            horizon_length=5,
            action_chunking=True,
            max_speed=1.0,
            direction_type="tangent",
            speed_bound="cube",
        )
        module = TangentDirectionSpeedNormal(
            base_cls=IdentityBackbone,
            action_dim=25,
            action_groups=groups,
        )
        inputs = jnp.ones((2, 32), dtype=jnp.float32)
        variables = module.init(jax.random.key(0), inputs)
        distribution = module.apply(variables, inputs)

        self.assertEqual(distribution.event_shape, (25,))
        self.assertEqual(distribution.raw_distribution.loc.shape, (2, 25))
        self.assertEqual(distribution.raw_distribution.stddev().shape, (2, 25))
        log_scale_bias = variables["params"]["ParameterHeads"][
            "LogScaleParameters"
        ]["bias"]
        self.assertEqual(log_scale_bias.shape, (25,))

        sample = distribution.sample(seed=jax.random.key(1))
        log_prob = distribution.log_prob(sample)
        self.assertTrue(bool(jnp.all(jnp.isfinite(sample))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(log_prob))))

    def test_semantic_reference_is_finite_and_matches_volume_ratio(self):
        groups = build_action_groups(
            action_dim=5,
            horizon_length=1,
            action_chunking=False,
            max_speed=1.0,
            direction_type="tangent",
            speed_bound="sphere",
        )
        module = TangentDirectionSpeedNormal(
            base_cls=IdentityBackbone,
            action_dim=5,
            action_groups=groups,
        )
        inputs = jnp.ones((8, 32), dtype=jnp.float32)
        variables = module.init(jax.random.key(4), inputs)
        distribution = module.apply(variables, inputs)
        actions = distribution.sample(seed=jax.random.key(5))
        components = distribution.semantic_component_log_ratios(actions)
        semantic = distribution.reference_log_prob(actions, "semantic_product")
        action_volume = distribution.reference_log_prob(actions, "action_volume")
        radial_fraction = jnp.linalg.norm(actions[..., :3], axis=-1)

        self.assertTrue(bool(jnp.all(jnp.isfinite(semantic))))
        np.testing.assert_allclose(
            semantic,
            components["direction"] + components["speed"] + components["scalar"],
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            action_volume - semantic,
            -2.0 * jnp.log(radial_fraction) - jnp.log(3.0),
            rtol=2e-4,
            atol=2e-4,
        )


if __name__ == "__main__":
    unittest.main()
