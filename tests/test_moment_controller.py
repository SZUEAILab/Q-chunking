import tempfile
import unittest
from dataclasses import fields

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from agents.acrlpd import ACRLPDAgent, get_config
from direction_speed.distributions.tangent import TangentDirectionSpeedNormal
from direction_speed.moment_control import (
    MomentControllerConfig,
    MomentControllerState,
    MomentTargets,
    measure_translation_geometry,
    read_moment_controller_params,
    replace_moment_controller_params,
    update_moment_controller,
)
from direction_speed.distributions.factory import resolve_pipeline_recipe
from direction_speed.specs import build_action_groups
from utils.flax_utils import restore_agent_with_file, save_agent


class IdentityBackbone(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        return inputs


class MomentControllerTest(unittest.TestCase):
    def test_fixed_alpha_is_used_without_temperature_gradient(self):
        config = get_config()
        config.horizon_length = 1
        config.action_chunking = False
        config.target_entropy = -2.5
        config.fixed_alpha = 0.1
        config.env_action_dim = 5
        config.policy_action_dim = 5
        config.critic_action_dim = 5
        agent = ACRLPDAgent.create(
            11,
            np.zeros((1, 6), dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            config,
        )
        batch = {
            "observations": jnp.zeros((4, 6), dtype=jnp.float32),
            "actions": jnp.zeros((4, 1, 5), dtype=jnp.float32),
        }

        def loss_fn(params):
            return agent.actor_loss(batch, params, jax.random.key(12))[0]

        grads = jax.grad(loss_fn)(agent.network.params)
        _, info = agent.actor_loss(
            batch, agent.network.params, jax.random.key(12)
        )

        self.assertAlmostEqual(float(info["alpha"]), 0.1, places=6)
        self.assertEqual(float(info["alpha_loss"]), 0.0)
        np.testing.assert_array_equal(
            grads["modules_alpha"]["log_temp"],
            np.zeros_like(grads["modules_alpha"]["log_temp"]),
        )

    def test_controller_config_exposes_radius_mean_control_switch(self):
        field_names = {field.name for field in fields(MomentControllerConfig)}
        self.assertIn("control_radius_mean", field_names)

    def test_geometry_is_conditional_on_state(self):
        # Shape is [samples, states, action_dim].  The two states deliberately
        # point in different directions; pooling them would report false spread.
        actions = np.asarray(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.8, 0.0, 0.0], [0.0, 1.2, 0.0]],
            ],
            dtype=np.float32,
        )
        metrics = measure_translation_geometry(actions)

        self.assertAlmostEqual(metrics["radius_mean"], 1.0, places=6)
        self.assertAlmostEqual(metrics["radius_std_within_state"], 0.1, places=6)
        self.assertAlmostEqual(metrics["angular_deviation_rad"], 0.0, places=5)

    def test_update_moves_each_control_toward_its_executed_moment(self):
        state = MomentControllerState(
            angular_log_std=-1.0,
            speed_log_std=-1.0,
            speed_mean=0.0,
            num_updates=0,
        )
        targets = MomentTargets(
            radius_mean=0.9,
            radius_std_within_state=0.05,
            angular_deviation_rad=0.2,
        )
        config = MomentControllerConfig(
            angular_gain=0.5,
            radial_std_gain=0.5,
            radius_mean_gain=1.0,
        )
        updated = update_moment_controller(
            state,
            {
                "radius_mean": 0.7,
                "radius_std_within_state": 0.10,
                "angular_deviation_rad": 0.4,
            },
            targets,
            config,
        )

        self.assertLess(updated.angular_log_std, state.angular_log_std)
        self.assertLess(updated.speed_log_std, state.speed_log_std)
        self.assertGreater(updated.speed_mean, state.speed_mean)
        self.assertEqual(updated.num_updates, 1)

    def test_noise_only_update_never_changes_speed_mean(self):
        state = MomentControllerState(
            angular_log_std=-1.0,
            speed_log_std=-1.0,
            speed_mean=0.75,
            num_updates=0,
        )
        updated = update_moment_controller(
            state,
            {
                "radius_mean": 0.1,
                "radius_std_within_state": 0.10,
                "angular_deviation_rad": 0.4,
            },
            MomentTargets(
                radius_mean=0.9,
                radius_std_within_state=0.05,
                angular_deviation_rad=0.2,
            ),
            MomentControllerConfig(
                control_radius_mean=False,
                angular_gain=0.5,
                radial_std_gain=0.5,
                radius_mean_gain=100.0,
            ),
        )

        self.assertLess(updated.angular_log_std, state.angular_log_std)
        self.assertLess(updated.speed_log_std, state.speed_log_std)
        self.assertEqual(updated.speed_mean, state.speed_mean)

    def test_update_api_is_reward_blind(self):
        state = MomentControllerState(-1.0, -1.0, 0.0, 0)
        targets = MomentTargets(0.9, 0.05, 0.2)
        with self.assertRaisesRegex(ValueError, "unexpected metrics"):
            update_moment_controller(
                state,
                {
                    "radius_mean": 0.9,
                    "radius_std_within_state": 0.05,
                    "angular_deviation_rad": 0.2,
                    "reward": 1.0,
                },
                targets,
                MomentControllerConfig(),
            )

    def test_controller_converges_for_monotone_geometry_response(self):
        targets = MomentTargets(0.9, 0.05, 0.2)
        state = MomentControllerState(-0.5, -0.5, -0.5, 0)
        optimum = MomentControllerState(-1.5, -2.0, 0.8, 0)
        config = MomentControllerConfig(
            angular_gain=0.5,
            radial_std_gain=0.5,
            radius_mean_gain=1.0,
        )
        for _ in range(12):
            metrics = {
                "angular_deviation_rad": targets.angular_deviation_rad
                * np.exp(state.angular_log_std - optimum.angular_log_std),
                "radius_std_within_state": targets.radius_std_within_state
                * np.exp(state.speed_log_std - optimum.speed_log_std),
                "radius_mean": targets.radius_mean
                + 0.5 * (state.speed_mean - optimum.speed_mean),
            }
            state = update_moment_controller(state, metrics, targets, config)

        self.assertLess(abs(state.angular_log_std - optimum.angular_log_std), 3e-4)
        self.assertLess(abs(state.speed_log_std - optimum.speed_log_std), 4e-4)
        self.assertLess(abs(state.speed_mean - optimum.speed_mean), 4e-4)

    def test_controlled_tangent_scales_have_zero_actor_gradient(self):
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
            moment_controlled=True,
            moment_angular_log_std_init=-1.7,
            moment_speed_log_std_init=-2.1,
            moment_speed_mean_init=0.4,
        )
        inputs = jnp.ones((4, 32), dtype=jnp.float32)
        variables = module.init(jax.random.key(4), inputs)

        distribution = module.apply(variables, inputs)
        raw_std = distribution.raw_distribution.stddev()
        np.testing.assert_allclose(raw_std[..., :2], np.exp(-1.7), rtol=1e-6)
        np.testing.assert_allclose(raw_std[..., 2], np.exp(-2.1), rtol=1e-6)
        np.testing.assert_allclose(
            distribution.raw_distribution.loc[..., 2], 0.4, rtol=1e-6
        )

        def controller_sensitive_loss(params):
            dist = module.apply({"params": params}, inputs)
            return dist.raw_distribution.loc.sum() + dist.raw_distribution.stddev().sum()

        grads = jax.grad(controller_sensitive_loss)(variables["params"])
        for name in (
            "moment_angular_log_std",
            "moment_speed_log_std",
            "moment_speed_mean",
        ):
            np.testing.assert_array_equal(grads[name], np.zeros_like(grads[name]))
        # The actor's ordinary speed-mean head is also disconnected in this
        # mode, so it cannot learn an equal-and-opposite compensation.
        np.testing.assert_array_equal(
            grads["ParameterHeads"]["LocationParameters"]["bias"][3],
            np.asarray(0.0, dtype=np.float32),
        )

    def test_bounded_speed_residual_retains_actor_gradient_without_global_drift(self):
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
            moment_controlled=True,
            moment_speed_mean_init=1.0,
            moment_speed_residual_scale=0.5,
        )
        inputs = jnp.ones((4, 32), dtype=jnp.float32)
        variables = module.init(jax.random.key(9), inputs)
        distribution = module.apply(variables, inputs)
        speed_locs = distribution.raw_distribution.loc[..., 2]
        self.assertTrue(bool(jnp.all(speed_locs >= 0.5)))
        self.assertTrue(bool(jnp.all(speed_locs <= 1.5)))

        def speed_location_loss(params):
            dist = module.apply({"params": params}, inputs)
            return dist.raw_distribution.loc[..., 2].sum()

        grads = jax.grad(speed_location_loss)(variables["params"])
        self.assertNotEqual(
            float(grads["ParameterHeads"]["LocationParameters"]["bias"][3]),
            0.0,
        )
        np.testing.assert_array_equal(
            grads["moment_speed_mean"],
            np.zeros_like(grads["moment_speed_mean"]),
        )

    def test_noise_only_control_preserves_actor_centers_and_their_gradients(self):
        groups = build_action_groups(
            action_dim=5,
            horizon_length=1,
            action_chunking=False,
            max_speed=1.0,
            direction_type="tangent",
            speed_bound="sphere",
        )
        ordinary = TangentDirectionSpeedNormal(
            base_cls=IdentityBackbone,
            action_dim=5,
            action_groups=groups,
        )
        controlled = TangentDirectionSpeedNormal(
            base_cls=IdentityBackbone,
            action_dim=5,
            action_groups=groups,
            moment_controlled=True,
            moment_control_speed_mean=False,
            moment_speed_mean_init=3.0,
            moment_speed_residual_scale=0.0,
        )
        inputs = jnp.arange(4 * 32, dtype=jnp.float32).reshape((4, 32)) / 100.0
        key = jax.random.key(17)
        ordinary_variables = ordinary.init(key, inputs)
        controlled_variables = controlled.init(key, inputs)
        ordinary_dist = ordinary.apply(ordinary_variables, inputs)
        controlled_dist = controlled.apply(controlled_variables, inputs)

        np.testing.assert_allclose(
            controlled_dist.bijector._direction_centers,
            ordinary_dist.bijector._direction_centers,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            controlled_dist.raw_distribution.loc[..., 2:],
            ordinary_dist.raw_distribution.loc[..., 2:],
            rtol=1e-6,
            atol=1e-6,
        )

        def center_loss(params):
            dist = controlled.apply({"params": params}, inputs)
            directions = dist.bijector._direction_centers
            direction_weights = jnp.asarray([1.0, 2.0, 4.0])
            raw_locations = dist.raw_distribution.loc
            return (
                jnp.sum(directions * direction_weights)
                + jnp.sum(raw_locations[..., 2])
                + jnp.sum(raw_locations[..., 3] * 2.0)
                + jnp.sum(raw_locations[..., 4] * 3.0)
            )

        grads = jax.grad(center_loss)(controlled_variables["params"])
        location_bias_grad = grads["ParameterHeads"]["LocationParameters"]["bias"]
        self.assertGreater(float(jnp.linalg.norm(location_bias_grad[:3])), 0.0)
        self.assertNotEqual(float(location_bias_grad[3]), 0.0)
        self.assertNotEqual(float(location_bias_grad[4]), 0.0)
        self.assertNotEqual(float(location_bias_grad[5]), 0.0)
        np.testing.assert_array_equal(
            grads["moment_speed_mean"],
            np.zeros_like(grads["moment_speed_mean"]),
        )

    def test_controller_parameters_round_trip_through_flax_state_dict(self):
        state = MomentControllerState(-1.7, -2.1, 0.4, 7)
        encoded = flax.serialization.to_state_dict(state)
        restored = flax.serialization.from_state_dict(
            MomentControllerState(0.0, 0.0, 0.0, 0), encoded
        )
        self.assertEqual(restored, state)

    def test_pipeline_forwards_moment_controller_configuration(self):
        recipe = resolve_pipeline_recipe(
            {
                "adapter_representation": "none",
                "use_integrated_ds": True,
                "ds_integrated_type": "tangent",
                "ds_max_speed": 1.0,
                "ds_speed_bound": "sphere",
                "ds_moment_controlled": True,
                "ds_moment_angular_log_std_init": -1.4,
                "ds_moment_speed_log_std_init": -2.2,
                "ds_moment_speed_mean_init": 0.3,
                "ds_moment_speed_residual_scale": 0.5,
                "ds_moment_control_speed_mean": False,
            }
        )
        kwargs = recipe.actor_kwargs
        self.assertTrue(kwargs["moment_controlled"])
        self.assertEqual(kwargs["moment_angular_log_std_init"], -1.4)
        self.assertEqual(kwargs["moment_speed_log_std_init"], -2.2)
        self.assertEqual(kwargs["moment_speed_mean_init"], 0.3)
        self.assertEqual(kwargs["moment_speed_residual_scale"], 0.5)
        self.assertIn("moment_control_speed_mean", kwargs)
        self.assertFalse(kwargs["moment_control_speed_mean"])

    def test_parameter_tree_replacement_is_exact_and_non_destructive(self):
        params = flax.core.freeze(
            {
                "modules_actor": {
                    "moment_angular_log_std": jnp.asarray([[-1.0]]),
                    "moment_speed_log_std": jnp.asarray([[-1.1]]),
                    "moment_speed_mean": jnp.asarray([[0.2]]),
                    "moment_num_updates": jnp.asarray(0.0),
                    "unrelated": jnp.asarray([3.0]),
                },
                "modules_critic": {"kernel": jnp.asarray([4.0])},
            }
        )
        desired = MomentControllerState(-1.7, -2.1, 0.4, 7)
        replaced = replace_moment_controller_params(params, desired)

        observed = read_moment_controller_params(replaced)
        self.assertAlmostEqual(
            observed.angular_log_std, desired.angular_log_std, places=6
        )
        self.assertAlmostEqual(observed.speed_log_std, desired.speed_log_std, places=6)
        self.assertAlmostEqual(
            observed.speed_mean, desired.speed_mean, places=6
        )
        self.assertEqual(observed.num_updates, desired.num_updates)
        np.testing.assert_array_equal(
            replaced["modules_actor"]["unrelated"],
            params["modules_actor"]["unrelated"],
        )
        np.testing.assert_array_equal(
            replaced["modules_critic"]["kernel"],
            params["modules_critic"]["kernel"],
        )
        self.assertEqual(float(params["modules_actor"]["moment_num_updates"]), 0.0)

    def test_real_agent_checkpoint_restores_controller_exactly(self):
        config = get_config()
        config.horizon_length = 1
        config.action_chunking = False
        config.target_entropy = -2.5
        config.env_action_dim = 5
        config.policy_action_dim = 5
        config.critic_action_dim = 5
        config.use_integrated_ds = True
        config.ds_integrated_type = "tangent"
        config.ds_speed_bound = "sphere"
        config.ds_moment_controlled = True
        agent = ACRLPDAgent.create(
            3,
            np.zeros((1, 6), dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            config,
        )
        desired = MomentControllerState(-1.25, -2.5, 0.75, 9)
        changed_params = replace_moment_controller_params(agent.network.params, desired)
        changed = agent.replace(network=agent.network.replace(params=changed_params))

        with tempfile.TemporaryDirectory() as directory:
            save_agent(changed, directory, 123)
            restored = restore_agent_with_file(agent, f"{directory}/params_123.pkl")

        changed_state = read_moment_controller_params(changed.network.params)
        restored_state = read_moment_controller_params(restored.network.params)
        self.assertEqual(
            flax.serialization.to_state_dict(restored_state),
            flax.serialization.to_state_dict(changed_state),
        )


if __name__ == "__main__":
    unittest.main()
