"""
Direction-Speed action distribution for structured action parameterization.

Decomposes continuous actions into:
- Direction head: unit vectors on the sphere (using spherical or stereographic coordinates for 3D groups)
- Speed head: magnitude scalar (sigmoid-gated)
- Scalar head: regular tanh-squashed Gaussian (for yaw, gripper, etc.)

Implements Distribution-integrated DS with a proper bijector (forward + inverse)
for correct log_prob computation in SAC.

The same forward policy can be regularized against two different reference
measures:

- ``action_volume``: Lebesgue volume in the executed environment action.
- ``semantic_product``: uniform surface measure on direction, uniform normalized
  radial fraction, and Lebesgue measure for the remaining scalar controls.

Keeping sampling fixed while changing only this reference is the matched-forward
counterfactual used by the DS-FER experiments.
"""

import math
from typing import Optional, Sequence

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

import flax.linen as nn
import jax.numpy as jnp

from direction_speed.geometry import bounded_sigmoid, bounded_sigmoid_grad, radial_bound
from direction_speed.specs import ActionGroup, ActionGroupKind
from rlpd_networks import default_init
from rlpd_distributions.parameter_heads import DistributionParameterHeads
from rlpd_distributions.pipeline import BijectedActionDistribution


def _safe_log(x, eps=1e-8):
    return jnp.log(jnp.clip(x, eps, 1e8))


def _safe_atanh(x, eps=1e-6):
    """Inverse of tanh: atanh(x) = 0.5 * log((1+x)/(1-x))"""
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * jnp.log((1.0 + x) / (1.0 - x))


class IntegratedDirectionSpeedBijector(tfb.Bijector):
    """
    Bijector from raw Gaussian params → structured action via direction-speed decomposition.

    Forward (raw → action):
        For each direction_speed_3d group:
            raw (theta_raw, phi_raw, speed_raw) → action (x, y, z)
        For each direction_speed_stereographic_3d group:
            raw (plane_x, plane_y, speed_raw) → action (x, y, z)
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

    def __init__(
        self, action_groups: Sequence[ActionGroup], name="integrated_direction_speed"
    ):
        super().__init__(forward_min_event_ndims=1, name=name)
        # Pre-compute group slicing info
        self._group_info = []
        raw_idx = 0
        act_idx = 0
        for group in action_groups:
            info = {"kind": group.kind}
            if group.kind in (ActionGroupKind.SPHERICAL, ActionGroupKind.STEREOGRAPHIC):
                info["raw_start"] = raw_idx
                info["raw_len"] = 3
                info["act_start"] = act_idx
                info["act_len"] = 3  # x, y, z
                info["max_speed"] = group.max_speed
                info["speed_bound"] = group.speed_bound
            elif group.kind is ActionGroupKind.SCALAR:
                n = group.size
                info["raw_start"] = raw_idx
                info["raw_len"] = n
                info["act_start"] = act_idx
                info["act_len"] = n
            else:
                raise ValueError(f"Unsupported integrated action group: {group.kind}")
            self._group_info.append(info)
            raw_idx += info["raw_len"]
            act_idx += info["act_len"]

        self._total_raw_dim = raw_idx
        self._total_act_dim = act_idx

    def _forward(self, x):
        """Transform raw → action."""
        actions = []
        for info in self._group_info:
            raw = x[..., info["raw_start"] : info["raw_start"] + info["raw_len"]]
            if info["kind"] is ActionGroupKind.SPHERICAL:
                a = self._forward_ds3d(raw, info["max_speed"], info["speed_bound"])
            elif info["kind"] is ActionGroupKind.STEREOGRAPHIC:
                a = self._forward_stereo3d(raw, info["max_speed"], info["speed_bound"])
            else:
                a = jnp.tanh(raw)
            actions.append(a)
        return jnp.concatenate(actions, axis=-1)

    def _forward_ds3d(self, raw, max_speed, speed_bound):
        """Transform (theta_raw, phi_raw, speed_raw) → (x, y, z)."""
        theta_raw = raw[..., 0:1]
        phi_raw = raw[..., 1:2]
        speed_raw = raw[..., 2:3]

        theta_frac = bounded_sigmoid(theta_raw)
        phi_frac = bounded_sigmoid(phi_raw)
        speed_frac = bounded_sigmoid(speed_raw)

        theta = jnp.pi * theta_frac
        phi = 2.0 * jnp.pi * phi_frac

        sin_theta = jnp.sin(theta)
        direction_x = sin_theta * jnp.cos(phi)
        direction_y = sin_theta * jnp.sin(phi)
        direction_z = jnp.cos(theta)
        direction = jnp.concatenate([direction_x, direction_y, direction_z], axis=-1)
        speed = radial_bound(direction, max_speed, speed_bound) * speed_frac
        x = speed * direction_x
        y = speed * direction_y
        z = speed * direction_z
        return jnp.concatenate([x, y, z], axis=-1)

    def _forward_stereo3d(self, raw, max_speed, speed_bound):
        """Transform (plane_x, plane_y, speed_raw) -> (x, y, z)."""
        plane_x = raw[..., 0:1]
        plane_y = raw[..., 1:2]
        speed_raw = raw[..., 2:3]

        rho2 = plane_x**2 + plane_y**2
        denom = 1.0 + rho2
        direction_x = 2.0 * plane_x / denom
        direction_y = 2.0 * plane_y / denom
        direction_z = (1.0 - rho2) / denom
        direction = jnp.concatenate([direction_x, direction_y, direction_z], axis=-1)
        speed = radial_bound(direction, max_speed, speed_bound) * bounded_sigmoid(
            speed_raw
        )

        return jnp.concatenate(
            [speed * direction_x, speed * direction_y, speed * direction_z],
            axis=-1,
        )

    def _forward_log_det_jacobian(self, x):
        """Log determinant of forward Jacobian."""
        total_log_det = jnp.zeros(x.shape[:-1])  # batch dims

        for info in self._group_info:
            raw = x[..., info["raw_start"] : info["raw_start"] + info["raw_len"]]
            if info["kind"] is ActionGroupKind.SPHERICAL:
                log_det = self._forward_log_det_ds3d(
                    raw, info["max_speed"], info["speed_bound"]
                )
            elif info["kind"] is ActionGroupKind.STEREOGRAPHIC:
                log_det = self._forward_log_det_stereo3d(
                    raw, info["max_speed"], info["speed_bound"]
                )
            else:
                # tanh: log det = sum log(1 - tanh(raw)²)
                a = jnp.tanh(raw)
                log_det = jnp.sum(_safe_log(1.0 - a**2), axis=-1)
            total_log_det = total_log_det + log_det

        return total_log_det

    def semantic_log_det_components(self, x):
        """Raw-coordinate Jacobians for the semantic product measure.

        Unlike the environment-volume Jacobian, this excludes the ``r^2`` volume
        factor and any direction-dependent radial bound.  The radial coordinate
        is the normalized fraction in ``[0, 1]``.
        """
        direction = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        speed = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        scalar = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        for info in self._group_info:
            raw = x[..., info["raw_start"] : info["raw_start"] + info["raw_len"]]
            if info["kind"] is ActionGroupKind.SPHERICAL:
                theta_raw = raw[..., 0]
                phi_raw = raw[..., 1]
                speed_raw = raw[..., 2]
                theta = jnp.pi * bounded_sigmoid(theta_raw)
                direction += (
                    _safe_log(jnp.abs(jnp.sin(theta)))
                    + _safe_log(jnp.pi * bounded_sigmoid_grad(theta_raw))
                    + _safe_log(2.0 * jnp.pi * bounded_sigmoid_grad(phi_raw))
                )
                speed += _safe_log(bounded_sigmoid_grad(speed_raw))
            elif info["kind"] is ActionGroupKind.STEREOGRAPHIC:
                rho2 = raw[..., 0] ** 2 + raw[..., 1] ** 2
                direction += jnp.log(4.0) - 2.0 * jnp.log1p(rho2)
                speed += _safe_log(bounded_sigmoid_grad(raw[..., 2]))
            else:
                squashed = jnp.tanh(raw)
                scalar += jnp.sum(_safe_log(1.0 - squashed**2), axis=-1)
        return {"direction": direction, "speed": speed, "scalar": scalar}

    def semantic_log_volume_components(self):
        """Log volumes of the normalized semantic reference factors."""
        direction_groups = sum(
            info["kind"] in (ActionGroupKind.SPHERICAL, ActionGroupKind.STEREOGRAPHIC)
            for info in self._group_info
        )
        scalar_dims = sum(
            info["raw_len"]
            for info in self._group_info
            if info["kind"] is ActionGroupKind.SCALAR
        )
        return {
            "direction": direction_groups * math.log(4.0 * math.pi),
            "speed": 0.0,
            "scalar": scalar_dims * math.log(2.0),
        }

    def action_log_volume(self):
        """Log volume of the executed action domain."""
        total = 0.0
        for info in self._group_info:
            if info["kind"] in (
                ActionGroupKind.SPHERICAL,
                ActionGroupKind.STEREOGRAPHIC,
                ActionGroupKind.TANGENT,
            ):
                if info["speed_bound"] == "sphere":
                    total += math.log(4.0 * math.pi / 3.0) + 3.0 * math.log(
                        info["max_speed"]
                    )
                else:
                    total += math.log(8.0)
            else:
                total += info["raw_len"] * math.log(2.0)
        return total

    def _forward_log_det_ds3d(self, raw, max_speed, speed_bound):
        """Log det for spherical coordinate transformation."""
        theta_raw = raw[..., 0]
        phi_raw = raw[..., 1]
        speed_raw = raw[..., 2]

        sig_theta = bounded_sigmoid(theta_raw)
        sig_phi = bounded_sigmoid(phi_raw)
        sig_speed = bounded_sigmoid(speed_raw)

        theta = jnp.pi * sig_theta
        phi = 2.0 * jnp.pi * sig_phi
        sin_theta = jnp.sin(theta)
        direction = jnp.stack(
            [
                sin_theta * jnp.cos(phi),
                sin_theta * jnp.sin(phi),
                jnp.cos(theta),
            ],
            axis=-1,
        )
        bound = radial_bound(direction, max_speed, speed_bound).squeeze(-1)
        speed = bound * sig_speed
        sigmoid_theta_grad = bounded_sigmoid_grad(theta_raw)
        sigmoid_phi_grad = bounded_sigmoid_grad(phi_raw)
        sigmoid_speed_grad = bounded_sigmoid_grad(speed_raw)

        # Spherical Jacobian: r² sin(θ)
        log_r2 = 2.0 * _safe_log(speed)
        log_sin_theta = _safe_log(jnp.abs(jnp.sin(theta)))

        # Sigmoid Jacobians
        log_dtheta = _safe_log(jnp.pi * sigmoid_theta_grad)
        log_dphi = _safe_log(2.0 * jnp.pi * sigmoid_phi_grad)
        log_dspeed = _safe_log(bound * sigmoid_speed_grad)

        return log_r2 + log_sin_theta + log_dtheta + log_dphi + log_dspeed

    def _forward_log_det_stereo3d(self, raw, max_speed, speed_bound):
        """Log det for stereographic direction + radial speed."""
        plane_x = raw[..., 0]
        plane_y = raw[..., 1]
        speed_raw = raw[..., 2]

        rho2 = plane_x**2 + plane_y**2
        denom = 1.0 + rho2
        direction = jnp.stack(
            [
                2.0 * plane_x / denom,
                2.0 * plane_y / denom,
                (1.0 - rho2) / denom,
            ],
            axis=-1,
        )
        bound = radial_bound(direction, max_speed, speed_bound).squeeze(-1)
        speed = bound * bounded_sigmoid(speed_raw)
        speed_grad = bound * bounded_sigmoid_grad(speed_raw)

        # Stereographic surface element on S^2 is 4 / (1 + rho^2)^2.
        log_surface = jnp.log(4.0) - 2.0 * jnp.log1p(rho2)
        return 2.0 * _safe_log(speed) + log_surface + _safe_log(speed_grad)

    def _inverse(self, y):
        """Transform action → raw."""
        raws = []
        for info in self._group_info:
            act = y[..., info["act_start"] : info["act_start"] + info["act_len"]]
            if info["kind"] is ActionGroupKind.SPHERICAL:
                raw = self._inverse_ds3d(act, info["max_speed"], info["speed_bound"])
            elif info["kind"] is ActionGroupKind.STEREOGRAPHIC:
                raw = self._inverse_stereo3d(
                    act, info["max_speed"], info["speed_bound"]
                )
            else:
                raw = _safe_atanh(act)
            raws.append(raw)
        return jnp.concatenate(raws, axis=-1)

    def _inverse_ds3d(self, action, max_speed, speed_bound):
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
        direction = action / jnp.clip(r[..., None], 1e-12, 1e8)
        bound = radial_bound(direction, max_speed, speed_bound).squeeze(-1)

        # Invert sigmoids
        theta_frac = jnp.clip(theta / jnp.pi, eps, 1.0 - eps)
        phi_frac = jnp.clip(phi / (2.0 * jnp.pi), eps, 1.0 - eps)
        speed_frac = jnp.clip(r / bound, eps, 1.0 - eps)

        theta_raw = jnp.log(theta_frac / (1.0 - theta_frac))
        phi_raw = jnp.log(phi_frac / (1.0 - phi_frac))
        speed_raw = jnp.log(speed_frac / (1.0 - speed_frac))

        return jnp.stack([theta_raw, phi_raw, speed_raw], axis=-1)

    def _inverse_stereo3d(self, action, max_speed, speed_bound):
        """Transform (x, y, z) -> (plane_x, plane_y, speed_raw)."""
        x = action[..., 0]
        y = action[..., 1]
        z = action[..., 2]

        speed = jnp.sqrt(x**2 + y**2 + z**2 + 1e-12)
        eps = 1e-6
        inv_speed = 1.0 / jnp.clip(speed, eps, 1e8)
        direction_x = x * inv_speed
        direction_y = y * inv_speed
        direction_z = z * inv_speed

        denom = jnp.clip(1.0 + direction_z, eps, 1e8)
        plane_x = direction_x / denom
        plane_y = direction_y / denom

        direction = jnp.stack([direction_x, direction_y, direction_z], axis=-1)
        bound = radial_bound(direction, max_speed, speed_bound).squeeze(-1)
        speed_frac = jnp.clip(speed / bound, eps, 1.0 - eps)
        speed_raw = jnp.log(speed_frac / (1.0 - speed_frac))

        return jnp.stack([plane_x, plane_y, speed_raw], axis=-1)

    def _inverse_log_det_jacobian(self, y):
        """Log determinant of inverse Jacobian = -forward_log_det(inverse(y))."""
        x = self._inverse(y)
        return -self._forward_log_det_jacobian(x)

    def _forward_event_shape(self, input_shape):
        return input_shape[:-1] + (self._total_act_dim,)

    def _inverse_event_shape(self, output_shape):
        return output_shape[:-1] + (self._total_raw_dim,)


class IntegratedDirectionSpeedNormal(nn.Module):
    """Gaussian actor transformed directly into environment action coordinates."""

    base_cls: nn.Module
    action_dim: int
    action_groups: Sequence[ActionGroup]
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    separate_component_trunks: bool = False

    def _component_parameters(self, inputs, component, size):
        """Build one independently parameterized semantic component."""
        features = self.base_cls(name=f"{component.title()}MLP")(inputs)
        if self.state_dependent_std:
            return DistributionParameterHeads(
                location_size=size,
                log_scale_size=size,
                log_scale_min=self.log_std_min,
                log_scale_max=self.log_std_max,
                name=f"{component.title()}ParameterHeads",
            )(features)
        means = nn.Dense(
            size,
            kernel_init=default_init(),
            name=f"{component.title()}LocationParameters",
        )(features)
        log_stds = self.param(
            f"{component.title()}OutputLogStd",
            nn.initializers.zeros,
            (size,),
            jnp.float32,
        )
        return means, jnp.clip(log_stds, self.log_std_min, self.log_std_max)

    def _separate_trunk_parameters(self, inputs):
        direction_size = sum(
            2 for group in self.action_groups if group.kind is not ActionGroupKind.SCALAR
        )
        speed_size = sum(
            1 for group in self.action_groups if group.kind is not ActionGroupKind.SCALAR
        )
        scalar_size = sum(
            group.size
            for group in self.action_groups
            if group.kind is ActionGroupKind.SCALAR
        )
        component_params = {
            "direction": self._component_parameters(
                inputs, "direction", direction_size
            ),
            "speed": self._component_parameters(inputs, "speed", speed_size),
            "scalar": self._component_parameters(inputs, "scalar", scalar_size),
        }
        offsets = {"direction": 0, "speed": 0, "scalar": 0}
        means, log_stds = [], []
        for group in self.action_groups:
            if group.kind is ActionGroupKind.SCALAR:
                component, width = "scalar", group.size
            else:
                component, width = "direction", 2
            start = offsets[component]
            means.append(component_params[component][0][..., start : start + width])
            log_stds.append(
                component_params[component][1][..., start : start + width]
            )
            offsets[component] += width
            if group.kind is not ActionGroupKind.SCALAR:
                start = offsets["speed"]
                means.append(component_params["speed"][0][..., start : start + 1])
                log_stds.append(
                    component_params["speed"][1][..., start : start + 1]
                )
                offsets["speed"] += 1
        return jnp.concatenate(means, axis=-1), jnp.concatenate(log_stds, axis=-1)

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        if self.separate_component_trunks:
            means, log_stds = self._separate_trunk_parameters(inputs)
        else:
            x = self.base_cls()(inputs, *args, **kwargs)
            if self.state_dependent_std:
                means, log_stds = DistributionParameterHeads(
                    location_size=self.action_dim,
                    log_scale_size=self.action_dim,
                    log_scale_min=self.log_std_min,
                    log_scale_max=self.log_std_max,
                    name="ParameterHeads",
                )(x)
            else:
                means = nn.Dense(
                    self.action_dim,
                    kernel_init=default_init(),
                    name="LocationParameters",
                )(x)
                log_stds = self.param(
                    "OutputLogStd",
                    nn.initializers.zeros,
                    (self.action_dim,),
                    jnp.float32,
                )
                log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))

        bijector = IntegratedDirectionSpeedBijector(self.action_groups)
        return ReferenceMeasureActionDistribution(
            distribution=base_dist, bijector=bijector
        )


class ReferenceMeasureActionDistribution(BijectedActionDistribution):
    """Integrated action distribution with matched-forward reference choices."""

    def _raw_coordinate_log_probs(self, action_raw):
        return tfd.Normal(
            loc=self.raw_distribution.mean(),
            scale=self.raw_distribution.stddev(),
        ).log_prob(action_raw)

    def semantic_component_log_ratios(self, action_pi):
        """Return log(pi/reference) for direction, speed, and scalar factors."""
        action_raw = self.policy_to_raw(action_pi)
        raw_log_probs = self._raw_coordinate_log_probs(action_raw)
        log_dets = self.policy_transform.semantic_log_det_components(action_raw)
        log_volumes = self.policy_transform.semantic_log_volume_components()
        raw_totals = {
            "direction": jnp.zeros(action_raw.shape[:-1], dtype=action_raw.dtype),
            "speed": jnp.zeros(action_raw.shape[:-1], dtype=action_raw.dtype),
            "scalar": jnp.zeros(action_raw.shape[:-1], dtype=action_raw.dtype),
        }
        for info in self.policy_transform._group_info:
            start = info["raw_start"]
            if info["kind"] in (
                ActionGroupKind.SPHERICAL,
                ActionGroupKind.STEREOGRAPHIC,
                ActionGroupKind.TANGENT,
            ):
                raw_totals["direction"] += jnp.sum(
                    raw_log_probs[..., start : start + 2], axis=-1
                )
                raw_totals["speed"] += raw_log_probs[..., start + 2]
            else:
                raw_totals["scalar"] += jnp.sum(
                    raw_log_probs[..., start : start + info["raw_len"]], axis=-1
                )
        return {
            name: raw_totals[name] - log_dets[name] + log_volumes[name]
            for name in raw_totals
        }

    def reference_log_prob(self, action_pi, reference):
        """Log density ratio relative to a normalized uniform reference."""
        if reference == "action_volume":
            return self.log_prob(action_pi) + self.policy_transform.action_log_volume()
        if reference == "semantic_product":
            components = self.semantic_component_log_ratios(action_pi)
            return sum(components.values())
        raise ValueError(f"Unknown action reference: {reference}")
