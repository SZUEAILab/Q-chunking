import numpy as np

from agents.acrlpd import _adapter_policy_to_env
from direction_speed.config import configure_direction_speed


def _config(bound="sphere", horizon=1):
    return {
        "adapter_representation": "cartesian",
        "adapter_speed_squash": "sigmoid",
        "adapter_speed_interpretation": "direct",
        "adapter_critic_space": "env",
        "ds_speed_bound": bound,
        "action_dim": 6,
        "env_action_dim": 5,
        "horizon_length": horizon,
        "action_chunking": True,
    }


def test_sphere_adapter_maps_policy_action_to_five_dimensional_env_action():
    policy_action = np.array([1.0, 0.0, 0.0, 0.5, 0.2, -0.3])

    actual = np.asarray(_adapter_policy_to_env(policy_action, _config("sphere")))

    np.testing.assert_allclose(actual, [0.5, 0.0, 0.0, 0.2, -0.3], atol=1e-6)


def test_cube_adapter_uses_fraction_of_directional_cube_radius():
    policy_action = np.array([1.0, 1.0, 1.0, 0.5, 0.2, -0.3])

    actual = np.asarray(_adapter_policy_to_env(policy_action, _config("cube")))

    np.testing.assert_allclose(actual, [0.5, 0.5, 0.5, 0.2, -0.3], atol=1e-6)


def test_env_critic_transform_preserves_action_chunks():
    one = np.array([1.0, 0.0, 0.0, 0.5, 0.2, -0.3])
    two = np.array([0.0, 1.0, 0.0, 0.25, -0.1, 0.4])
    chunk = np.concatenate([one, two])

    actual = np.asarray(_adapter_policy_to_env(chunk, _config("sphere", horizon=2)))

    assert actual.shape == (10,)
    np.testing.assert_allclose(
        actual,
        [0.5, 0.0, 0.0, 0.2, -0.3, 0.0, 0.25, 0.0, -0.1, 0.4],
        atol=1e-6,
    )


def test_config_resolves_env_action_critic():
    config = {
        "use_integrated_ds": False,
        "ds_max_speed": 1.0,
    }

    resolved = configure_direction_speed(
        ds_mode="adapter",
        ds_adapter_representation="cartesian",
        ds_adapter_speed_squash="sigmoid",
        ds_adapter_speed_interpretation="direct",
        ds_adapter_critic_space="env",
        config=config,
        agent_name="acrlpd",
    )

    assert resolved.use_env_action_critic
    assert config["adapter_critic_space"] == "env"
