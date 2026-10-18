"""Tangent raw Gaussian parameters and their policy-action bijector."""

import math
from typing import Optional, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from direction_speed.geometry import (
    EPS,
    bounded_sigmoid,
    bounded_sigmoid_grad,
    bounded_sigmoid_inverse,
    radial_bound,
)
from direction_speed.specs import ActionGroup, ActionGroupKind
from rlpd_distributions.parameter_heads import DistributionParameterHeads
from direction_speed.distributions.integrated import ReferenceMeasureActionDistribution


tfd = tfp.distributions
tfb = tfp.bijectors
MAX_ANGLE = math.pi - 1e-4
DEFAULT_ANGULAR_LOG_STD_OFFSET = -1.0
DEFAULT_ANGULAR_LOG_STD_MAX = 0.0
DEFAULT_SPEED_LOG_STD_OFFSET = 0.0
DEFAULT_SPEED_LOG_STD_MAX = 2.0
DEFAULT_SPEED_MEAN_OFFSET = 0.0


def _safe_norm(value, axis=-1, keepdims=True):
    return jnp.sqrt(jnp.sum(value**2, axis=axis, keepdims=keepdims) + EPS**2)


def _normalize_direction_centers(raw_centers):
    """Map unconstrained 3D heads to unit directions without a pole bias."""
    norms = jnp.linalg.norm(raw_centers, axis=-1, keepdims=True)
    normalized = raw_centers / jnp.maximum(norms, EPS)
    north = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 1.0], dtype=raw_centers.dtype),
        raw_centers.shape,
    )
    return jnp.where(norms > EPS, normalized, north)


def _tangent_basis(direction_center):
    """Construct a stable orthonormal frame with no threshold-switch seam."""
    x = direction_center[..., 0:1]
    y = direction_center[..., 1:2]
    z = direction_center[..., 2:3]
    south = z < -0.999
    reciprocal = 1.0 / jnp.maximum(1.0 + z, EPS)
    cross_term = -x * y * reciprocal
    basis_x = jnp.concatenate(
        [1.0 - x * x * reciprocal, cross_term, -x],
        axis=-1,
    )
    basis_y = jnp.concatenate(
        [cross_term, 1.0 - y * y * reciprocal, -y],
        axis=-1,
    )
    y_axis = jnp.broadcast_to(
        jnp.array([0.0, 1.0, 0.0], dtype=direction_center.dtype),
        direction_center.shape,
    )
    south_basis_x = jnp.cross(y_axis, direction_center)
    south_basis_x = south_basis_x / _safe_norm(south_basis_x)
    south_basis_y = jnp.cross(direction_center, south_basis_x)
    basis_x = jnp.where(south, south_basis_x, basis_x)
    basis_y = jnp.where(south, south_basis_y, basis_y)
    return basis_x, basis_y


def _bound_local_offset(local_offset):
    """Biject R^2 to the open tangent disk with radius pi."""
    raw_radius = _safe_norm(local_offset)
    angle = MAX_ANGLE * jnp.tanh(raw_radius / MAX_ANGLE)
    scale = jnp.where(raw_radius > EPS, angle / jnp.maximum(raw_radius, EPS), 1.0)
    return local_offset * scale, raw_radius, angle


def _unbound_local_offset(tangent):
    angle = _safe_norm(tangent)
    clipped_angle = jnp.minimum(angle, MAX_ANGLE * (1.0 - EPS))
    raw_radius = MAX_ANGLE * jnp.arctanh(clipped_angle / MAX_ANGLE)
    scale = jnp.where(angle > EPS, raw_radius / jnp.maximum(angle, EPS), 1.0)
    return tangent * scale


def _direction_log_det(raw_radius, angle):
    derivative = 1.0 - jnp.tanh(raw_radius / MAX_ANGLE) ** 2
    ratio = jnp.sin(angle) * derivative / jnp.maximum(raw_radius, EPS)
    ratio = jnp.where(raw_radius > EPS, ratio, jnp.ones_like(ratio))
    return jnp.log(jnp.maximum(ratio, EPS)).squeeze(-1)


def _exp_map(direction_center, tangent):
    angle = _safe_norm(tangent)
    basis_x, basis_y = _tangent_basis(direction_center)
    tangent_3d = basis_x * tangent[..., 0:1] + basis_y * tangent[..., 1:2]
    sinc = jnp.where(angle > EPS, jnp.sin(angle) / jnp.maximum(angle, EPS), 1.0)
    direction = jnp.cos(angle) * direction_center + sinc * tangent_3d
    return direction / _safe_norm(direction)


def _log_map(direction_center, direction):
    dot = jnp.sum(direction_center * direction, axis=-1, keepdims=True)
    angle = jnp.arccos(jnp.clip(dot, -1.0 + EPS, 1.0))
    angle = jnp.minimum(angle, MAX_ANGLE * (1.0 - EPS))
    orthogonal = direction - dot * direction_center
    orthogonal = orthogonal / _safe_norm(orthogonal)
    tangent_3d = angle * orthogonal
    basis_x, basis_y = _tangent_basis(direction_center)
    tangent = jnp.concatenate(
        [
            jnp.sum(tangent_3d * basis_x, axis=-1, keepdims=True),
            jnp.sum(tangent_3d * basis_y, axis=-1, keepdims=True),
        ],
        axis=-1,
    )
    return jnp.where(angle > EPS, tangent, jnp.zeros_like(tangent))


def _projected_map(direction_center, local_offset):
    """Normalize a Cartesian tangent proposal back onto the unit sphere."""
    basis_x, basis_y = _tangent_basis(direction_center)
    tangent_3d = (
        basis_x * local_offset[..., 0:1]
        + basis_y * local_offset[..., 1:2]
    )
    proposal = direction_center + tangent_3d
    return proposal / _safe_norm(proposal)


def _projected_inverse(direction_center, direction):
    """Inverse gnomonic coordinates on the center-facing hemisphere."""
    basis_x, basis_y = _tangent_basis(direction_center)
    center_dot = jnp.sum(direction_center * direction, axis=-1, keepdims=True)
    safe_dot = jnp.maximum(center_dot, EPS)
    tangent_3d = direction / safe_dot - direction_center
    return jnp.concatenate(
        [
            jnp.sum(tangent_3d * basis_x, axis=-1, keepdims=True),
            jnp.sum(tangent_3d * basis_y, axis=-1, keepdims=True),
        ],
        axis=-1,
    )


def _projected_direction_log_det(local_offset):
    """Area Jacobian of v -> normalize(center + Bv) on S^2."""
    squared_radius = jnp.sum(local_offset**2, axis=-1)
    return -1.5 * jnp.log1p(squared_radius)


class TangentPolicyBijector(tfb.Bijector):
    """Conditioned bijector from local raw coordinates to ``action_pi``."""

    def __init__(
        self,
        direction_centers,
        action_groups,
        projected_tangent=False,
        name="tangent_policy",
    ):
        super().__init__(forward_min_event_ndims=1, name=name)
        self._direction_centers = direction_centers
        self._action_groups = tuple(action_groups)
        self._projected_tangent = projected_tangent
        self._group_info = []
        raw_start = 0
        for group in self._action_groups:
            self._group_info.append(
                {
                    "kind": group.kind,
                    "raw_start": raw_start,
                    "raw_len": group.size,
                    "max_speed": group.max_speed,
                    "speed_bound": group.speed_bound,
                }
            )
            raw_start += group.size

    def _forward(self, action_raw):
        policy_actions = []
        offset = 0
        direction_index = 0
        for group in self._action_groups:
            raw = action_raw[..., offset : offset + group.size]
            if group.kind is ActionGroupKind.TANGENT:
                if self._projected_tangent:
                    direction = _projected_map(
                        self._direction_centers[..., direction_index, :],
                        raw[..., :2],
                    )
                else:
                    tangent, _, _ = _bound_local_offset(raw[..., :2])
                    direction = _exp_map(
                        self._direction_centers[..., direction_index, :], tangent
                    )
                speed = radial_bound(
                    direction, group.max_speed, group.speed_bound
                ) * bounded_sigmoid(raw[..., 2:3])
                policy_actions.append(direction * speed)
                direction_index += 1
            else:
                policy_actions.append(jnp.tanh(raw))
            offset += group.size
        return jnp.concatenate(policy_actions, axis=-1)

    def _inverse(self, action_pi):
        raw_actions = []
        offset = 0
        direction_index = 0
        for group in self._action_groups:
            action = action_pi[..., offset : offset + group.size]
            if group.kind is ActionGroupKind.TANGENT:
                speed = _safe_norm(action)
                direction = action / jnp.maximum(speed, EPS)
                if self._projected_tangent:
                    local_offset = _projected_inverse(
                        self._direction_centers[..., direction_index, :], direction
                    )
                else:
                    tangent = _log_map(
                        self._direction_centers[..., direction_index, :], direction
                    )
                    local_offset = _unbound_local_offset(tangent)
                bound = radial_bound(direction, group.max_speed, group.speed_bound)
                speed_raw = bounded_sigmoid_inverse(speed / bound)
                raw_actions.append(jnp.concatenate([local_offset, speed_raw], axis=-1))
                direction_index += 1
            else:
                clipped = jnp.clip(action, -1.0 + EPS, 1.0 - EPS)
                raw_actions.append(jnp.arctanh(clipped))
            offset += group.size
        return jnp.concatenate(raw_actions, axis=-1)

    def _forward_log_det_jacobian(self, action_raw):
        total = jnp.zeros(action_raw.shape[:-1], dtype=action_raw.dtype)
        offset = 0
        direction_index = 0
        for group in self._action_groups:
            raw = action_raw[..., offset : offset + group.size]
            if group.kind is ActionGroupKind.TANGENT:
                if self._projected_tangent:
                    direction = _projected_map(
                        self._direction_centers[..., direction_index, :],
                        raw[..., :2],
                    )
                    direction_log_det = _projected_direction_log_det(raw[..., :2])
                else:
                    tangent, raw_radius, angle = _bound_local_offset(raw[..., :2])
                    direction = _exp_map(
                        self._direction_centers[..., direction_index, :], tangent
                    )
                    direction_log_det = _direction_log_det(raw_radius, angle)
                bound = radial_bound(direction, group.max_speed, group.speed_bound)
                speed = bound * bounded_sigmoid(raw[..., 2:3])
                speed_grad = bound * bounded_sigmoid_grad(raw[..., 2:3])
                total += (
                    direction_log_det
                    + 2.0 * jnp.log(jnp.maximum(speed.squeeze(-1), EPS))
                    + jnp.log(jnp.maximum(speed_grad.squeeze(-1), EPS))
                )
                direction_index += 1
            else:
                squashed = jnp.tanh(raw)
                total += jnp.sum(jnp.log(jnp.maximum(1.0 - squashed**2, EPS)), axis=-1)
            offset += group.size
        return total

    def _inverse_log_det_jacobian(self, action_pi):
        action_raw = self._inverse(action_pi)
        return -self._forward_log_det_jacobian(action_raw)

    def semantic_log_det_components(self, action_raw):
        """Raw-coordinate Jacobians for direction, normalized speed, scalars."""
        direction = jnp.zeros(action_raw.shape[:-1], dtype=action_raw.dtype)
        speed = jnp.zeros(action_raw.shape[:-1], dtype=action_raw.dtype)
        scalar = jnp.zeros(action_raw.shape[:-1], dtype=action_raw.dtype)
        offset = 0
        for group in self._action_groups:
            raw = action_raw[..., offset : offset + group.size]
            if group.kind is ActionGroupKind.TANGENT:
                if self._projected_tangent:
                    direction += _projected_direction_log_det(raw[..., :2])
                else:
                    _, raw_radius, angle = _bound_local_offset(raw[..., :2])
                    direction += _direction_log_det(raw_radius, angle)
                speed += jnp.log(
                    jnp.maximum(bounded_sigmoid_grad(raw[..., 2]), EPS)
                )
            else:
                squashed = jnp.tanh(raw)
                scalar += jnp.sum(
                    jnp.log(jnp.maximum(1.0 - squashed**2, EPS)), axis=-1
                )
            offset += group.size
        return {"direction": direction, "speed": speed, "scalar": scalar}

    def semantic_log_volume_components(self):
        direction_groups = sum(
            group.kind is ActionGroupKind.TANGENT for group in self._action_groups
        )
        scalar_dims = sum(
            group.size
            for group in self._action_groups
            if group.kind is ActionGroupKind.SCALAR
        )
        return {
            "direction": direction_groups * math.log(4.0 * math.pi),
            "speed": 0.0,
            "scalar": scalar_dims * math.log(2.0),
        }

    def action_log_volume(self):
        total = 0.0
        for group in self._action_groups:
            if group.kind is ActionGroupKind.TANGENT:
                if group.speed_bound == "sphere":
                    total += math.log(4.0 * math.pi / 3.0) + 3.0 * math.log(
                        group.max_speed
                    )
                else:
                    total += math.log(8.0)
            else:
                total += group.size * math.log(2.0)
        return total


class TangentDirectionSpeedNormal(nn.Module):
    """Build tangent parameters, raw Gaussian sampler, and policy bijector."""

    base_cls: Type[nn.Module]
    action_dim: int
    action_groups: Sequence[ActionGroup]
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    angular_log_std_offset: float = DEFAULT_ANGULAR_LOG_STD_OFFSET
    angular_log_std_max: float = DEFAULT_ANGULAR_LOG_STD_MAX
    speed_log_std_offset: float = DEFAULT_SPEED_LOG_STD_OFFSET
    speed_log_std_max: float = DEFAULT_SPEED_LOG_STD_MAX
    speed_mean_offset: float = DEFAULT_SPEED_MEAN_OFFSET
    projected_tangent: bool = False
    moment_controlled: bool = False
    moment_angular_log_std_init: float = -1.7
    moment_speed_log_std_init: float = -2.1
    moment_speed_mean_init: float = 0.4
    moment_speed_residual_scale: float = 0.0
    moment_control_speed_mean: bool = True

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        features = self.base_cls()(inputs, *args, **kwargs)
        direction_count = sum(
            group.kind is ActionGroupKind.TANGENT for group in self.action_groups
        )
        scalar_count = sum(
            group.size
            for group in self.action_groups
            if group.kind is ActionGroupKind.SCALAR
        )
        total_size = sum(group.size for group in self.action_groups)
        if total_size != self.action_dim:
            raise ValueError(
                f"Action groups have {total_size} dims, expected {self.action_dim}"
            )
        if direction_count + sum(
            group.kind is ActionGroupKind.SCALAR for group in self.action_groups
        ) != len(self.action_groups):
            raise ValueError("Tangent policy received an unsupported action group")

        location_parameters, log_scale_parameters = DistributionParameterHeads(
            location_size=direction_count * 4 + scalar_count,
            log_scale_size=direction_count * 3 + scalar_count,
            log_scale_min=self.log_std_min,
            log_scale_max=self.log_std_max,
            name="ParameterHeads",
        )(features)

        direction_locations = location_parameters[..., : direction_count * 4].reshape(
            location_parameters.shape[:-1] + (direction_count, 4)
        )
        raw_centers = direction_locations[..., :3]
        speed_means = direction_locations[..., 3:4]
        scalar_means = location_parameters[..., direction_count * 4 :]

        direction_log_scales = log_scale_parameters[..., : direction_count * 3].reshape(
            log_scale_parameters.shape[:-1] + (direction_count, 3)
        )
        angular_log_stds = jnp.clip(
            direction_log_scales[..., :2] + self.angular_log_std_offset,
            self.log_std_min,
            self.angular_log_std_max,
        )
        speed_log_stds = jnp.clip(
            direction_log_scales[..., 2:3] + self.speed_log_std_offset,
            self.log_std_min,
            self.speed_log_std_max,
        )
        scalar_log_stds = log_scale_parameters[..., direction_count * 3 :]

        if self.moment_controlled:
            angular_control = self.param(
                "moment_angular_log_std",
                nn.initializers.constant(self.moment_angular_log_std_init),
                (direction_count, 1),
            )
            speed_control = self.param(
                "moment_speed_log_std",
                nn.initializers.constant(self.moment_speed_log_std_init),
                (direction_count, 1),
            )
            mean_control = self.param(
                "moment_speed_mean",
                nn.initializers.constant(self.moment_speed_mean_init),
                (direction_count, 1),
            )
            self.param(
                "moment_num_updates",
                nn.initializers.zeros_init(),
                (),
            )
            angular_log_stds = jnp.broadcast_to(
                jax.lax.stop_gradient(angular_control),
                direction_log_scales[..., :2].shape,
            )
            speed_log_stds = jnp.broadcast_to(
                jax.lax.stop_gradient(speed_control),
                direction_log_scales[..., 2:3].shape,
            )
            if self.moment_control_speed_mean:
                speed_means = jnp.broadcast_to(
                    jax.lax.stop_gradient(mean_control), speed_means.shape
                ) + self.moment_speed_residual_scale * jnp.tanh(speed_means)

        direction_centers = _normalize_direction_centers(raw_centers)

        raw_locations = []
        raw_log_scales = []
        direction_index = 0
        scalar_offset = 0
        for group in self.action_groups:
            if group.kind is ActionGroupKind.TANGENT:
                zeros = jnp.zeros_like(speed_means[..., direction_index, :])
                raw_locations.append(
                    jnp.concatenate(
                            [
                                zeros,
                                zeros,
                                speed_means[..., direction_index, :]
                                + self.speed_mean_offset,
                            ],
                            axis=-1,
                    )
                )
                raw_log_scales.append(
                    jnp.concatenate(
                        [
                            angular_log_stds[..., direction_index, :],
                            speed_log_stds[..., direction_index, :],
                        ],
                        axis=-1,
                    )
                )
                direction_index += 1
            else:
                scalar_slice = slice(scalar_offset, scalar_offset + group.size)
                raw_locations.append(scalar_means[..., scalar_slice])
                raw_log_scales.append(scalar_log_stds[..., scalar_slice])
                scalar_offset += group.size

        raw_distribution = tfd.MultivariateNormalDiag(
            loc=jnp.concatenate(raw_locations, axis=-1),
            scale_diag=jnp.exp(jnp.concatenate(raw_log_scales, axis=-1)),
        )
        policy_bijector = TangentPolicyBijector(
            direction_centers=direction_centers,
            action_groups=self.action_groups,
            projected_tangent=self.projected_tangent,
            name=(
                "projected_tangent_policy"
                if self.projected_tangent
                else "tangent_policy"
            ),
        )
        return ReferenceMeasureActionDistribution(
            distribution=raw_distribution,
            bijector=policy_bijector,
        )
