"""
Direction-Speed action distribution for structured action parameterization.

Decomposes continuous actions into:
- Direction head: unit vectors on the sphere (using spherical coordinates for 3D groups)
- Speed head: magnitude scalar (sigmoid-gated)
- Scalar head: regular tanh-squashed Gaussian (for yaw, gripper, etc.)

Supports proper bijector (forward + inverse) for correct log_prob computation in SAC.
"""

import functools
from typing import Optional, Sequence, Dict

import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

import flax.linen as nn
import jax.numpy as jnp
import jax

from rlpd_networks import default_init


def _safe_log(x, eps=1e-8):
    return jnp.log(jnp.clip(x, eps, 1e8))


def _safe_atanh(x, eps=1e-8):
    """Inverse of tanh: atanh(x) = 0.5 * log((1+x)/(1-x))"""
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * jnp.log((1.0 + x) / (1.0 - x))


class _DirectionSpeedBijector(tfb.Bijector):
    """
    Bijector from raw Gaussian params → structured action via direction-speed decomposition.

    Forward (raw → action):
        For each direction_speed_3d group:
            raw (theta_raw, phi_raw, speed_raw) → action (x, y, z)
        For each scalar group:
            raw → tanh(raw)

    Inverse (action → raw):
        For each direction_speed_3d group:
            action (x, y, z) → raw (theta_raw, phi_raw, speed_raw)
            r = sqrt(x²+y²+z²), θ = acos(z/r), φ = atan2(y,x)
            theta_raw = logit(θ/π), phi_raw = logit(φ/2π), speed_raw = logit(r/max_speed)
        For each scalar group:
            action → atanh(action)
    """

    def __init__(self, action_groups: Sequence[Dict], name="direction_speed_bijector"):
        super().__init__(forward_min_event_ndims=1, name=name)
        self._action_groups = action_groups

        # Pre-compute group slicing info
        self._group_info = []
        raw_idx = 0
        act_idx = 0
        for group in action_groups:
            info = {"type": group["type"]}
            if group["type"] == "direction_speed_3d":
                info["raw_start"] = raw_idx
                info["raw_len"] = 3  # theta_raw, phi_raw, speed_raw
                info["act_start"] = act_idx
                info["act_len"] = 3  # x, y, z
                info["max_speed"] = group.get("max_speed", 1.0)
            elif group["type"] == "scalar":
                n = len(group["dims"])
                info["raw_start"] = raw_idx
                info["raw_len"] = n
                info["act_start"] = act_idx
                info["act_len"] = n
            else:
                raise ValueError(f"Unknown action group type: {group['type']}")
            self._group_info.append(info)
            raw_idx += info["raw_len"]
            act_idx += info["act_len"]

        self._total_raw_dim = raw_idx
        self._total_act_dim = act_idx

    def _forward(self, x):
        """Transform raw → action."""
        actions = []
        for info in self._group_info:
            raw = x[..., info["raw_start"]:info["raw_start"] + info["raw_len"]]
            if info["type"] == "direction_speed_3d":
                a = self._forward_ds3d(raw, info["max_speed"])
            else:
                a = jnp.tanh(raw)
            actions.append(a)
        return jnp.concatenate(actions, axis=-1)

    def _forward_ds3d(self, raw, max_speed):
        """Transform (theta_raw, phi_raw, speed_raw) → (x, y, z)."""
        theta_raw = raw[..., 0:1]
        phi_raw = raw[..., 1:2]
        speed_raw = raw[..., 2:3]

        theta = jnp.pi * jax.nn.sigmoid(theta_raw)
        phi = 2.0 * jnp.pi * jax.nn.sigmoid(phi_raw)
        speed = max_speed * jax.nn.sigmoid(speed_raw)

        sin_theta = jnp.sin(theta)
        x = speed * sin_theta * jnp.cos(phi)
        y = speed * sin_theta * jnp.sin(phi)
        z = speed * jnp.cos(theta)
        return jnp.concatenate([x, y, z], axis=-1)

    def _forward_log_det_jacobian(self, x):
        """Log determinant of forward Jacobian."""
        total_log_det = jnp.zeros(x.shape[:-1])  # batch dims

        for info in self._group_info:
            raw = x[..., info["raw_start"]:info["raw_start"] + info["raw_len"]]
            if info["type"] == "direction_speed_3d":
                log_det = self._forward_log_det_ds3d(raw, info["max_speed"])
            else:
                # tanh: log det = sum log(1 - tanh(raw)²)
                a = jnp.tanh(raw)
                log_det = jnp.sum(_safe_log(1.0 - a ** 2), axis=-1)
            total_log_det = total_log_det + log_det

        return total_log_det

    def _forward_log_det_ds3d(self, raw, max_speed):
        """Log det for spherical coordinate transformation."""
        theta_raw = raw[..., 0]
        phi_raw = raw[..., 1]
        speed_raw = raw[..., 2]

        sig_theta = jax.nn.sigmoid(theta_raw)
        sig_phi = jax.nn.sigmoid(phi_raw)
        sig_speed = jax.nn.sigmoid(speed_raw)

        theta = jnp.pi * sig_theta
        speed = max_speed * sig_speed

        # Spherical Jacobian: r² sin(θ)
        log_r2 = 2.0 * _safe_log(speed)
        log_sin_theta = _safe_log(jnp.abs(jnp.sin(theta)))

        # Sigmoid Jacobians
        log_dtheta = _safe_log(jnp.pi * sig_theta * (1.0 - sig_theta))
        log_dphi = _safe_log(2.0 * jnp.pi * sig_phi * (1.0 - sig_phi))
        log_dspeed = _safe_log(max_speed * sig_speed * (1.0 - sig_speed))

        return log_r2 + log_sin_theta + log_dtheta + log_dphi + log_dspeed

    def _inverse(self, y):
        """Transform action → raw."""
        raws = []
        for info in self._group_info:
            act = y[..., info["act_start"]:info["act_start"] + info["act_len"]]
            if info["type"] == "direction_speed_3d":
                raw = self._inverse_ds3d(act, info["max_speed"])
            else:
                raw = _safe_atanh(act)
            raws.append(raw)
        return jnp.concatenate(raws, axis=-1)

    def _inverse_ds3d(self, action, max_speed):
        """Transform (x, y, z) → (theta_raw, phi_raw, speed_raw)."""
        x = action[..., 0]
        y = action[..., 1]
        z = action[..., 2]

        r = jnp.sqrt(x**2 + y**2 + z**2 + 1e-12)
        eps = 1e-6

        # theta = acos(z/r), clip to valid range
        cos_theta = jnp.clip(z / jnp.clip(r, 1e-12, 1e8), -1.0 + eps, 1.0 - eps)
        theta = jnp.arccos(cos_theta)

        # phi = atan2(y, x), normalize to [0, 2π]
        phi = jnp.arctan2(y, x)
        phi = jnp.where(phi < 0, phi + 2.0 * jnp.pi, phi)

        # Invert sigmoids
        theta_frac = jnp.clip(theta / jnp.pi, eps, 1.0 - eps)
        phi_frac = jnp.clip(phi / (2.0 * jnp.pi), eps, 1.0 - eps)
        speed_frac = jnp.clip(r / max_speed, eps, 1.0 - eps)

        theta_raw = jnp.log(theta_frac / (1.0 - theta_frac))
        phi_raw = jnp.log(phi_frac / (1.0 - phi_frac))
        speed_raw = jnp.log(speed_frac / (1.0 - speed_frac))

        return jnp.stack([theta_raw, phi_raw, speed_raw], axis=-1)

    def _inverse_log_det_jacobian(self, y):
        """Log determinant of inverse Jacobian = -forward_log_det(inverse(y))."""
        x = self._inverse(y)
        return -self._forward_log_det_jacobian(x)

    def _forward_event_shape(self, input_shape):
        return input_shape[:-1] + (self._total_act_dim,)

    def _inverse_event_shape(self, output_shape):
        return output_shape[:-1] + (self._total_raw_dim,)


class DirectionSpeedDistribution(nn.Module):
    """
    Action distribution that decomposes action space via direction-speed parameterization.

    Uses a proper TFP bijector (forward + inverse) for correct log_prob computation.

    Configuration `action_groups` specifies the decomposition:

        # OGBench (act_dim=5: dx, dy, dz, dyaw, gripper)
        action_groups = [
            {"type": "direction_speed_3d", "dims": [0, 1, 2], "max_speed": 1.0},
            {"type": "scalar", "dims": [3]},
            {"type": "scalar", "dims": [4]},
        ]

        # Robomimic (act_dim=7: dx, dy, dz, droll, dpitch, dyaw, gripper)
        action_groups = [
            {"type": "direction_speed_3d", "dims": [0, 1, 2], "max_speed": 1.0},
            {"type": "direction_speed_3d", "dims": [3, 4, 5], "max_speed": 1.0},
            {"type": "scalar", "dims": [6]},
        ]
    """

    base_cls: nn.Module
    action_dim: int
    action_groups: Sequence[Dict]
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param(
                "OutputLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))

        bijector = _DirectionSpeedBijector(self.action_groups)
        return tfd.TransformedDistribution(distribution=base_dist, bijector=bijector)


# Convenience partial
DirectionSpeedNormal = DirectionSpeedDistribution
