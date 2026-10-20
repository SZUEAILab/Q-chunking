"""Policy-space direction-speed codecs used by action-adapter DS.

The policy and critic operate on ``a_pi``.  Only environment interaction uses
``compose(a_pi)`` to produce ``a_env``.  Three policy representations are
supported for the leading XYZ displacement:

* cartesian: ``[u_x, u_y, u_z, speed, ...]`` (D+1 -> D)
* spherical: ``[theta_frac, phi_frac, speed, ...]`` (D -> D)
* stereographic: ``[p_x, p_y, speed, ...]`` (D -> D)

Actor speed squashing and adapter-side speed interpretation are independent.
The radial bound can be the unit sphere or the full normalized action cube.
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from direction_speed.postprocessors import L2BallProjectionPostprocessor
from direction_speed.specs import (
    DirectionRepresentation,
    RadialBound,
    SpeedInterpretation,
    SpeedSquash,
)
from direction_speed.speed import AdapterSpeedPipeline


EPS = 1e-6
DEFAULT_DIR_DIMS = 3
AUXILIARY_EXECUTION_MAPS = ("cartesian_aux", "cartesian_projection_aux")


@jax.custom_jvp
def _direction_norm(value):
    """Preserve the L2 norm forward pass and choose a finite zero subgradient."""
    return jnp.linalg.norm(value, axis=-1, keepdims=True)


@_direction_norm.defjvp
def _direction_norm_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    norm = _direction_norm(value)
    denominator = jnp.where(norm > 0, norm, 1.0)
    derivative = jnp.sum(value * tangent, axis=-1, keepdims=True) / denominator
    return norm, derivative


def _validate(representation, speed_squash, speed_interpretation, speed_bound):
    return (
        DirectionRepresentation(representation),
        SpeedSquash(speed_squash),
        SpeedInterpretation(speed_interpretation),
        RadialBound(speed_bound),
    )


def _policy_action_dim(env_action_dim: int, representation: str) -> int:
    if env_action_dim < DEFAULT_DIR_DIMS:
        raise ValueError(
            f"env_action_dim must be at least {DEFAULT_DIR_DIMS}, got {env_action_dim}"
        )
    representation = DirectionRepresentation(representation)
    return (
        env_action_dim + 1
        if representation is DirectionRepresentation.CARTESIAN
        else env_action_dim
    )


def _policy_squash_types(policy_dim: int, representation: str, speed_squash: str):
    """Per-dimension actor transforms for one (unchunked) policy action."""
    representation = DirectionRepresentation(representation)
    speed_squash = SpeedSquash(speed_squash).value
    if representation is DirectionRepresentation.CARTESIAN:
        prefix = ["tanh", "tanh", "tanh", speed_squash]
    elif representation is DirectionRepresentation.SPHERICAL:
        prefix = ["sigmoid", "sigmoid", speed_squash]
    else:
        prefix = ["identity", "identity", speed_squash]
    if policy_dim < len(prefix):
        raise ValueError(f"policy_dim={policy_dim} is too small for {representation}")
    return prefix + ["tanh"] * (policy_dim - len(prefix))


def _np_direction_from_action(actions, representation, eps):
    if representation is DirectionRepresentation.CARTESIAN:
        raw = actions[..., :3]
        norm = np.linalg.norm(raw, axis=-1, keepdims=True)
        return raw / np.maximum(norm, eps), 3
    if representation is DirectionRepresentation.SPHERICAL:
        theta = np.pi * np.clip(actions[..., 0:1], 0.0, 1.0)
        phi = 2.0 * np.pi * np.clip(actions[..., 1:2], 0.0, 1.0)
        sin_theta = np.sin(theta)
        return (
            np.concatenate(
                [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)],
                axis=-1,
            ),
            2,
        )
    plane_x, plane_y = actions[..., 0:1], actions[..., 1:2]
    rho2 = plane_x**2 + plane_y**2
    denom = 1.0 + rho2
    return (
        np.concatenate(
            [2.0 * plane_x / denom, 2.0 * plane_y / denom, (1.0 - rho2) / denom],
            axis=-1,
        ),
        2,
    )


def _jax_direction_from_action(actions, representation, eps):
    if representation is DirectionRepresentation.CARTESIAN:
        raw = actions[..., :3]
        norm = _direction_norm(raw)
        return raw / jnp.maximum(norm, eps), 3
    if representation is DirectionRepresentation.SPHERICAL:
        theta = jnp.pi * jnp.clip(actions[..., 0:1], 0.0, 1.0)
        phi = 2.0 * jnp.pi * jnp.clip(actions[..., 1:2], 0.0, 1.0)
        sin_theta = jnp.sin(theta)
        return (
            jnp.concatenate(
                [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), jnp.cos(theta)],
                axis=-1,
            ),
            2,
        )
    plane_x, plane_y = actions[..., 0:1], actions[..., 1:2]
    rho2 = plane_x**2 + plane_y**2
    denom = 1.0 + rho2
    return (
        jnp.concatenate(
            [2.0 * plane_x / denom, 2.0 * plane_y / denom, (1.0 - rho2) / denom],
            axis=-1,
        ),
        2,
    )


def _np_radial_bound(direction, speed_bound, eps):
    if speed_bound is RadialBound.SPHERE:
        return np.ones(direction.shape[:-1] + (1,), dtype=direction.dtype)
    return 1.0 / np.maximum(np.max(np.abs(direction), axis=-1, keepdims=True), eps)


def _jax_radial_bound(direction, speed_bound, eps):
    if speed_bound is RadialBound.SPHERE:
        return jnp.ones(direction.shape[:-1] + (1,), dtype=direction.dtype)
    return 1.0 / jnp.maximum(jnp.max(jnp.abs(direction), axis=-1, keepdims=True), eps)


def _np_encode_direction(direction, representation, eps):
    if representation is DirectionRepresentation.CARTESIAN:
        return direction
    if representation is DirectionRepresentation.SPHERICAL:
        theta_frac = np.arccos(np.clip(direction[..., 2:3], -1.0, 1.0)) / np.pi
        phi = np.mod(np.arctan2(direction[..., 1:2], direction[..., 0:1]), 2.0 * np.pi)
        phi_frac = phi / (2.0 * np.pi)
        return np.concatenate(
            [np.clip(theta_frac, eps, 1.0 - eps), np.clip(phi_frac, eps, 1.0 - eps)],
            axis=-1,
        )
    denom = np.maximum(1.0 + direction[..., 2:3], eps)
    return direction[..., :2] / denom


def _jax_encode_direction(direction, representation, eps):
    if representation is DirectionRepresentation.CARTESIAN:
        return direction
    if representation is DirectionRepresentation.SPHERICAL:
        theta_frac = jnp.arccos(jnp.clip(direction[..., 2:3], -1.0, 1.0)) / jnp.pi
        phi = jnp.mod(
            jnp.arctan2(direction[..., 1:2], direction[..., 0:1]), 2.0 * jnp.pi
        )
        phi_frac = phi / (2.0 * jnp.pi)
        return jnp.concatenate(
            [jnp.clip(theta_frac, eps, 1.0 - eps), jnp.clip(phi_frac, eps, 1.0 - eps)],
            axis=-1,
        )
    denom = jnp.maximum(1.0 + direction[..., 2:3], eps)
    return direction[..., :2] / denom


def _decompose_numpy(
    np_actions: np.ndarray,
    eps: float = EPS,
    representation: str = "cartesian",
    speed_squash: str = "tanh",
    speed_interpretation: str = "log",
    speed_bound: str = "sphere",
):
    """Convert environment actions to policy actions ``a_pi``."""
    representation, speed_squash, speed_interpretation, speed_bound = _validate(
        representation, speed_squash, speed_interpretation, speed_bound
    )
    speed_pipeline = AdapterSpeedPipeline(speed_squash, speed_interpretation)
    xyz = np_actions[..., :DEFAULT_DIR_DIMS]
    remainder = np_actions[..., DEFAULT_DIR_DIMS:]
    magnitude = np.linalg.norm(xyz, axis=-1, keepdims=True)
    direction = xyz / np.maximum(magnitude, eps)
    zero = magnitude < eps
    north = np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=xyz.dtype), direction.shape)
    direction = np.where(zero, north, direction)
    radial_frac = magnitude / _np_radial_bound(direction, speed_bound, eps)
    speed = speed_pipeline.interpreter.encode_numpy(radial_frac, eps)
    return np.concatenate(
        [_np_encode_direction(direction, representation, eps), speed, remainder],
        axis=-1,
    )


def _compose_numpy(
    np_policy_actions: np.ndarray,
    eps: float = EPS,
    representation: str = "cartesian",
    speed_squash: str = "tanh",
    speed_interpretation: str = "log",
    speed_bound: str = "sphere",
):
    """Convert policy actions ``a_pi`` to environment actions ``a_env``."""
    representation, speed_squash, speed_interpretation, speed_bound = _validate(
        representation, speed_squash, speed_interpretation, speed_bound
    )
    speed_pipeline = AdapterSpeedPipeline(speed_squash, speed_interpretation)
    direction, speed_idx = _np_direction_from_action(
        np_policy_actions, representation, eps
    )
    speed_value = np_policy_actions[..., speed_idx : speed_idx + 1]
    remainder = np_policy_actions[..., speed_idx + 1 :]
    radial_frac = speed_pipeline.interpreter.decode_numpy(speed_value)
    xyz = direction * _np_radial_bound(direction, speed_bound, eps) * radial_frac
    return np.clip(np.concatenate([xyz, remainder], axis=-1), -1.0, 1.0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "eps",
        "representation",
        "speed_squash",
        "speed_interpretation",
        "speed_bound",
    ),
)
def _decompose_jax(
    actions: jnp.ndarray,
    eps: float = EPS,
    representation: str = "cartesian",
    speed_squash: str = "tanh",
    speed_interpretation: str = "log",
    speed_bound: str = "sphere",
):
    representation, speed_squash, speed_interpretation, speed_bound = _validate(
        representation, speed_squash, speed_interpretation, speed_bound
    )
    speed_pipeline = AdapterSpeedPipeline(speed_squash, speed_interpretation)
    xyz = actions[..., :DEFAULT_DIR_DIMS]
    remainder = actions[..., DEFAULT_DIR_DIMS:]
    magnitude = _direction_norm(xyz)
    direction = xyz / jnp.maximum(magnitude, eps)
    north = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 1.0], dtype=xyz.dtype), direction.shape
    )
    direction = jnp.where(magnitude < eps, north, direction)
    radial_frac = magnitude / _jax_radial_bound(direction, speed_bound, eps)
    speed = speed_pipeline.interpreter.encode_jax(radial_frac, eps)
    return jnp.concatenate(
        [_jax_encode_direction(direction, representation, eps), speed, remainder],
        axis=-1,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "eps",
        "representation",
        "speed_squash",
        "speed_interpretation",
        "speed_bound",
    ),
)
def _compose_jax(
    policy_actions: jnp.ndarray,
    eps: float = EPS,
    representation: str = "cartesian",
    speed_squash: str = "tanh",
    speed_interpretation: str = "log",
    speed_bound: str = "sphere",
):
    representation, speed_squash, speed_interpretation, speed_bound = _validate(
        representation, speed_squash, speed_interpretation, speed_bound
    )
    speed_pipeline = AdapterSpeedPipeline(speed_squash, speed_interpretation)
    direction, speed_idx = _jax_direction_from_action(
        policy_actions, representation, eps
    )
    speed_value = policy_actions[..., speed_idx : speed_idx + 1]
    remainder = policy_actions[..., speed_idx + 1 :]
    radial_frac = speed_pipeline.interpreter.decode_jax(speed_value)
    xyz = direction * _jax_radial_bound(direction, speed_bound, eps) * radial_frac
    return jnp.clip(jnp.concatenate([xyz, remainder], axis=-1), -1.0, 1.0)


def _decompose_chunked(
    actions: np.ndarray,
    horizon_length: int,
    representation: str = "cartesian",
    speed_squash: str = "tanh",
    speed_interpretation: str = "log",
    speed_bound: str = "sphere",
):
    """Convert ``(N,H,D)`` env actions to flattened chunked policy actions."""
    if actions.ndim == 3:
        _, horizon, _ = actions.shape
    else:
        horizon = horizon_length
        actions = actions.reshape(actions.shape[0], horizon, -1)
    parts = [
        _decompose_numpy(
            actions[:, t, :],
            representation=representation,
            speed_squash=speed_squash,
            speed_interpretation=speed_interpretation,
            speed_bound=speed_bound,
        )
        for t in range(horizon)
    ]
    return np.concatenate(parts, axis=-1)


@dataclass(frozen=True)
class DirectionSpeedAdapter:
    """Policy/environment codec, including an explicit auxiliary-head control.

    The auxiliary controls drop coordinate 3 at execution; the projection
    variant also projects XYZ onto the unit ball. Their placeholder lift is
    only for dimensional bookkeeping, and is not an inverse outside the ball.
    Training is restricted to an env critic without BC: raw environment replay
    stays unchanged and the placeholder is not an observed BC label.
    """

    representation: DirectionRepresentation | str = DirectionRepresentation.CARTESIAN
    speed_squash: SpeedSquash | str = SpeedSquash.SIGMOID
    speed_interpretation: SpeedInterpretation | str = SpeedInterpretation.DIRECT
    speed_bound: RadialBound | str = RadialBound.CUBE
    eps: float = EPS
    execution_map: str = "direction_speed"

    def __post_init__(self):
        object.__setattr__(
            self, "representation", DirectionRepresentation(self.representation)
        )
        object.__setattr__(self, "speed_squash", SpeedSquash(self.speed_squash))
        object.__setattr__(
            self,
            "speed_interpretation",
            SpeedInterpretation(self.speed_interpretation),
        )
        object.__setattr__(self, "speed_bound", RadialBound(self.speed_bound))
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.execution_map not in ("direction_speed", *AUXILIARY_EXECUTION_MAPS):
            raise ValueError("Unknown adapter execution map")
        if self.execution_map in AUXILIARY_EXECUTION_MAPS and (
            self.representation is not DirectionRepresentation.CARTESIAN
            or self.speed_squash is not SpeedSquash.TANH
            or self.speed_interpretation is not SpeedInterpretation.LOG
            or self.speed_bound is not RadialBound.SPHERE
        ):
            raise ValueError("Cartesian auxiliary control requires cartesian/tanh/log/sphere")

    def policy_action_dim(self, env_action_dim: int) -> int:
        return _policy_action_dim(env_action_dim, self.representation.value)

    def squash_types(self, policy_dim: int, repetitions: int = 1) -> tuple[str, ...]:
        one_step = tuple(
            _policy_squash_types(
                policy_dim,
                self.representation.value,
                self.speed_squash.value,
            )
        )
        return one_step * repetitions

    def decompose(self, env_actions: np.ndarray) -> np.ndarray:
        if env_actions.ndim == 0 or env_actions.shape[-1] < DEFAULT_DIR_DIMS:
            raise ValueError("env_actions must have at least three action dimensions")
        if self.execution_map in AUXILIARY_EXECUTION_MAPS:
            # A placeholder lift, not an observed auxiliary action or BC label.
            return np.concatenate(
                [env_actions[..., :3], np.zeros_like(env_actions[..., :1]), env_actions[..., 3:]], axis=-1
            )
        return _decompose_numpy(
            env_actions,
            self.eps,
            self.representation.value,
            self.speed_squash.value,
            self.speed_interpretation.value,
            self.speed_bound.value,
        )

    def compose(self, policy_actions: np.ndarray) -> np.ndarray:
        minimum_dim = (
            4 if self.representation is DirectionRepresentation.CARTESIAN else 3
        )
        if policy_actions.ndim == 0 or policy_actions.shape[-1] < minimum_dim:
            raise ValueError(
                f"policy_actions requires at least {minimum_dim} dimensions for "
                f"{self.representation.value}"
            )
        if self.execution_map in AUXILIARY_EXECUTION_MAPS:
            actions = np.concatenate([policy_actions[..., :3], policy_actions[..., 4:]], axis=-1)
            if self.execution_map == "cartesian_projection_aux":
                actions = L2BallProjectionPostprocessor().compose(actions)
            return np.clip(actions, -1., 1.)
        return _compose_numpy(
            policy_actions,
            self.eps,
            self.representation.value,
            self.speed_squash.value,
            self.speed_interpretation.value,
            self.speed_bound.value,
        )

    def compose_policy_output(
        self, policy_output: np.ndarray, policy_action_dim: int
    ) -> np.ndarray:
        """Compose a flat or chunked actor output into per-step env actions."""
        policy_actions = np.asarray(policy_output).reshape(-1, policy_action_dim)
        return self.compose(policy_actions)

    def decompose_jax(self, env_actions: jnp.ndarray) -> jnp.ndarray:
        if self.execution_map in AUXILIARY_EXECUTION_MAPS:
            return jnp.concatenate(
                [env_actions[..., :3], jnp.zeros_like(env_actions[..., :1]), env_actions[..., 3:]], axis=-1
            )
        return _decompose_jax(
            env_actions,
            self.eps,
            self.representation.value,
            self.speed_squash.value,
            self.speed_interpretation.value,
            self.speed_bound.value,
        )

    def compose_jax(self, policy_actions: jnp.ndarray) -> jnp.ndarray:
        if self.execution_map in AUXILIARY_EXECUTION_MAPS:
            actions = jnp.concatenate([policy_actions[..., :3], policy_actions[..., 4:]], axis=-1)
            if self.execution_map == "cartesian_projection_aux":
                actions = L2BallProjectionPostprocessor().compose_jax(actions)
            return jnp.clip(actions, -1., 1.)
        return _compose_jax(
            policy_actions,
            self.eps,
            self.representation.value,
            self.speed_squash.value,
            self.speed_interpretation.value,
            self.speed_bound.value,
        )

    def decompose_chunked(
        self, env_actions: np.ndarray, horizon_length: int
    ) -> np.ndarray:
        if horizon_length < 1:
            raise ValueError("horizon_length must be at least 1")
        if self.execution_map in AUXILIARY_EXECUTION_MAPS:
            actions = env_actions.reshape(env_actions.shape[0], horizon_length, -1)
            return self.decompose(actions).reshape(actions.shape[0], -1)
        return _decompose_chunked(
            env_actions,
            horizon_length,
            self.representation.value,
            self.speed_squash.value,
            self.speed_interpretation.value,
            self.speed_bound.value,
        )


def validate_auxiliary_control(config):
    """Limit the new map intervention to its implemented H1 SAC/env interface."""
    if config.get("adapter_execution_map", "direction_speed") not in AUXILIARY_EXECUTION_MAPS:
        return
    expected = dict(agent_name="acrlpd", adapter_representation="cartesian", adapter_critic_space="env",
                    adapter_speed_squash="tanh", adapter_speed_interpretation="log", ds_speed_bound="sphere",
                    env_action_dim=5, policy_action_dim=6, critic_action_dim=5, horizon_length=1,
                    actor_update_mode="sac", bc_alpha=0., ds_regularizer="policy_coordinate",
                    ds_phase_regularization="none", ds_stage_speed_constraint="none",
                    use_integrated_ds=False, use_cartesian_l2_projection=False,
                    ds_separate_alpha=False, ds_separate_actor_trunks=False, ds_moment_controlled=False)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Cartesian auxiliary control requires {key}={value!r}")
