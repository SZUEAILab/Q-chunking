"""Reward-blind control of executed direction/speed exploration moments.

The controller only consumes action samples from a fixed observation panel.  It
does not receive rewards, successes, critic values, or replay priorities.  This
keeps the intervention a geometry manipulation rather than an RL algorithm.
"""

from dataclasses import dataclass
from typing import Mapping

import flax
import jax
import jax.numpy as jnp
import numpy as np


_METRIC_KEYS = frozenset(
    {
        "radius_mean",
        "radius_std_within_state",
        "angular_deviation_rad",
    }
)


@flax.struct.dataclass
class MomentControllerState:
    angular_log_std: float
    speed_log_std: float
    speed_mean: float
    num_updates: int


@dataclass(frozen=True)
class MomentTargets:
    radius_mean: float
    radius_std_within_state: float
    angular_deviation_rad: float


@dataclass(frozen=True)
class MomentControllerConfig:
    control_radius_mean: bool = True
    angular_gain: float = 0.35
    radial_std_gain: float = 0.35
    radius_mean_gain: float = 4.0
    angular_log_std_min: float = -5.0
    angular_log_std_max: float = 0.5
    speed_log_std_min: float = -5.0
    speed_log_std_max: float = 0.5
    speed_mean_min: float = -6.0
    speed_mean_max: float = 6.0
    ratio_epsilon: float = 1e-5


_PARAMETER_NAMES = {
    "angular_log_std": "moment_angular_log_std",
    "speed_log_std": "moment_speed_log_std",
    "speed_mean": "moment_speed_mean",
    "num_updates": "moment_num_updates",
}


def _actor_params(params):
    try:
        return params["modules_actor"]
    except KeyError as error:
        raise ValueError("parameter tree has no modules_actor") from error


def read_moment_controller_params(params) -> MomentControllerState:
    """Read the externally controlled leaves from an actor parameter tree."""
    actor = _actor_params(params)
    try:
        values = {
            field: np.asarray(actor[name])
            for field, name in _PARAMETER_NAMES.items()
        }
    except KeyError as error:
        raise ValueError(f"missing moment-controller parameter {error.args[0]}") from error
    for field in ("angular_log_std", "speed_log_std", "speed_mean"):
        if not np.allclose(values[field], values[field].flat[0]):
            raise ValueError(f"moment-controller parameter {field} is not shared")
    return MomentControllerState(
        angular_log_std=float(values["angular_log_std"].flat[0]),
        speed_log_std=float(values["speed_log_std"].flat[0]),
        speed_mean=float(values["speed_mean"].flat[0]),
        num_updates=int(values["num_updates"].item()),
    )


def replace_moment_controller_params(params, state: MomentControllerState):
    """Return a parameter tree with only the controller leaves replaced."""
    was_frozen = isinstance(params, flax.core.FrozenDict)
    mutable = flax.core.unfreeze(params) if was_frozen else dict(params)
    mutable["modules_actor"] = dict(_actor_params(mutable))
    actor = mutable["modules_actor"]
    replacements = {
        "moment_angular_log_std": state.angular_log_std,
        "moment_speed_log_std": state.speed_log_std,
        "moment_speed_mean": state.speed_mean,
        "moment_num_updates": float(state.num_updates),
    }
    for name, value in replacements.items():
        if name not in actor:
            raise ValueError(f"missing moment-controller parameter {name}")
        actor[name] = jnp.full_like(actor[name], value)
    return flax.core.freeze(mutable) if was_frozen else mutable


def measure_translation_geometry(actions) -> dict[str, float]:
    """Measure conditional geometry from ``[samples, states, action_dim]``.

    Direction spread and radial standard deviation are computed within each
    observation before averaging.  Consequently, variation of policy means
    across observations is not mistaken for exploration noise.
    """
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[-1] < 3:
        raise ValueError(
            "actions must have shape [samples, states, action_dim>=3]"
        )
    xyz = actions[..., :3]
    radii = np.linalg.norm(xyz, axis=-1)
    directions = xyz / np.maximum(radii[..., None], 1e-8)
    mean_direction = directions.mean(axis=0)
    resultant = np.linalg.norm(mean_direction, axis=-1)
    centers = mean_direction / np.maximum(resultant[..., None], 1e-8)
    cosines = np.sum(directions * centers[None, ...], axis=-1)
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    return {
        "radius_mean": float(radii.mean()),
        "radius_std_within_state": float(radii.std(axis=0).mean()),
        "angular_deviation_rad": float(angles.mean()),
    }


def sample_executed_translation_geometry(
    agent,
    observations,
    postprocessor,
    policy_action_dim: int,
    num_samples: int,
    rng,
) -> dict[str, float]:
    """Sample an actor on a fixed panel and measure executed XYZ geometry."""
    distribution = agent.network.select("actor")(jnp.asarray(observations))
    policy_actions = distribution.sample(num_samples, seed=rng)
    flat_actions = policy_actions.reshape((-1, policy_action_dim))
    env_actions = postprocessor.compose_jax(flat_actions).reshape(
        (num_samples, len(observations), -1)
    )
    return measure_translation_geometry(jax.device_get(env_actions))


def update_moment_controller(
    state: MomentControllerState,
    metrics: Mapping[str, float],
    targets: MomentTargets,
    config: MomentControllerConfig,
) -> MomentControllerState:
    """Return one bounded proportional update using action geometry only."""
    supplied = frozenset(metrics)
    if supplied != _METRIC_KEYS:
        unexpected = sorted(supplied - _METRIC_KEYS)
        missing = sorted(_METRIC_KEYS - supplied)
        raise ValueError(
            f"unexpected metrics={unexpected}; missing metrics={missing}"
        )
    eps = config.ratio_epsilon
    angular_delta = config.angular_gain * np.log(
        max(targets.angular_deviation_rad, eps)
        / max(float(metrics["angular_deviation_rad"]), eps)
    )
    radial_delta = config.radial_std_gain * np.log(
        max(targets.radius_std_within_state, eps)
        / max(float(metrics["radius_std_within_state"]), eps)
    )
    mean_delta = 0.0
    if config.control_radius_mean:
        mean_delta = config.radius_mean_gain * (
            targets.radius_mean - float(metrics["radius_mean"])
        )
    return MomentControllerState(
        angular_log_std=float(
            np.clip(
                state.angular_log_std + angular_delta,
                config.angular_log_std_min,
                config.angular_log_std_max,
            )
        ),
        speed_log_std=float(
            np.clip(
                state.speed_log_std + radial_delta,
                config.speed_log_std_min,
                config.speed_log_std_max,
            )
        ),
        speed_mean=float(
            np.clip(
                state.speed_mean + mean_delta,
                config.speed_mean_min,
                config.speed_mean_max,
            )
        ),
        num_updates=state.num_updates + 1,
    )
