"""Shared JAX geometry used by integrated direction-speed distributions."""

import jax
import jax.numpy as jnp

from direction_speed.specs import RadialBound


EPS = 1e-6


def bounded_sigmoid(value, eps: float = EPS):
    return eps + (1.0 - 2.0 * eps) * jax.nn.sigmoid(value)


def bounded_sigmoid_inverse(value, eps: float = EPS):
    unit = (jnp.clip(value, eps, 1.0 - eps) - eps) / (1.0 - 2.0 * eps)
    unit = jnp.clip(unit, eps, 1.0 - eps)
    return jnp.log(unit) - jnp.log1p(-unit)


def bounded_sigmoid_grad(value, eps: float = EPS):
    unit = jax.nn.sigmoid(value)
    return (1.0 - 2.0 * eps) * unit * (1.0 - unit)


def radial_bound(direction, max_speed: float, speed_bound: str | RadialBound):
    """Return the legal radial extent in a normalized sphere or cube."""
    speed_bound = RadialBound(speed_bound)
    if speed_bound is RadialBound.SPHERE:
        shape = direction.shape[:-1] + (1,)
        return jnp.ones(shape, dtype=direction.dtype) * max_speed
    max_abs = jnp.max(jnp.abs(direction), axis=-1, keepdims=True)
    return 1.0 / jnp.maximum(max_abs, EPS)
