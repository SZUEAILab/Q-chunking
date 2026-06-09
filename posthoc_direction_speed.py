"""
Post-hoc direction + speed decomposition helpers.

Decomposes a D-dimensional action a in [-1, 1]^D into:
  - direction: unit vector of the first dir_dims dimensions (on S^{dir_dims-1})
  - log_speed:  scalar = log |a[:dir_dims]|
  - other dims: passed through unchanged (e.g. yaw, gripper)

This module supports the non-invertible D+1 post-hoc DS ablation path. It is
separate from rlpd_distributions.direction_speed, which implements
Jacobian-corrected actor distributions for ACRLPD.

Default for OGBench (5D): dir_dims=3  →  [dx, dy, dz] decomposed.
For Robomimic (7D):      dir_dims=3  →  [dx, dy, dz] decomposed, orientation+grip pass through.
"""

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-6

# Default: OGBench 5D action = [dx, dy, dz, dyaw, grip]
DEFAULT_DIR_DIMS = 3


# =============================================================================
# NumPy helpers (dataset preprocessing)
# =============================================================================

def decompose(np_actions: np.ndarray, dir_dims: int = DEFAULT_DIR_DIMS, eps: float = EPS):
    """Raw actions -> [direction_unit | log_speed | other_dims].

    Args:
        np_actions: (..., D) raw actions in [-1, 1].
        dir_dims:   number of leading dims to treat as a direction vector (default 3).
    Returns:
        (..., D+1) where:
          [..., :dir_dims]     = direction unit vector
          [..., dir_dims]      = log_speed
          [..., dir_dims+1:]   = remaining dims unchanged
    """
    direction_raw = np_actions[..., :dir_dims]
    remainder = np_actions[..., dir_dims:]

    magnitude = np.linalg.norm(direction_raw, axis=-1, keepdims=True)
    direction_unit = direction_raw / np.maximum(magnitude, eps)
    log_speed = np.log(np.maximum(magnitude, eps))

    parts = [direction_unit, log_speed]
    if remainder.shape[-1] > 0:
        parts.append(remainder)
    return np.concatenate(parts, axis=-1)


def compose(np_decomposed: np.ndarray, dir_dims: int = DEFAULT_DIR_DIMS, eps: float = EPS):
    """[direction | log_speed | other] → raw actions.

    Args:
        np_decomposed: (..., D+1) with direction + log_speed + other dims.
        dir_dims:      number of direction dimensions.
    Returns:
        (..., D) raw actions in [-1, 1].
    """
    direction = np_decomposed[..., :dir_dims]
    log_speed = np_decomposed[..., dir_dims:dir_dims + 1]
    remainder = np_decomposed[..., dir_dims + 1:]

    direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction_unit = direction / np.maximum(direction_norm, eps)
    speed = np.exp(log_speed)

    xyz = direction_unit * speed
    parts = [xyz]
    if remainder.shape[-1] > 0:
        parts.append(remainder)
    return np.clip(np.concatenate(parts, axis=-1), -1.0, 1.0)


# =============================================================================
# JAX helpers (online interaction)
# =============================================================================

@jax.jit
def decompose_jax(actions: jnp.ndarray, dir_dims: int = DEFAULT_DIR_DIMS, eps: float = EPS):
    direction_raw = actions[..., :dir_dims]
    remainder = actions[..., dir_dims:]
    magnitude = jnp.linalg.norm(direction_raw, axis=-1, keepdims=True)
    direction_unit = direction_raw / jnp.maximum(magnitude, eps)
    log_speed = jnp.log(jnp.maximum(magnitude, eps))
    parts = [direction_unit, log_speed]
    if remainder.shape[-1] > 0:
        parts.append(remainder)
    return jnp.concatenate(parts, axis=-1)


@jax.jit
def compose_jax(decomposed: jnp.ndarray, dir_dims: int = DEFAULT_DIR_DIMS, eps: float = EPS):
    direction = decomposed[..., :dir_dims]
    log_speed = decomposed[..., dir_dims:dir_dims + 1]
    remainder = decomposed[..., dir_dims + 1:]

    direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    direction_unit = direction / jnp.maximum(direction_norm, eps)
    speed = jnp.exp(log_speed)
    xyz = direction_unit * speed
    parts = [xyz]
    if remainder.shape[-1] > 0:
        parts.append(remainder)
    return jnp.clip(jnp.concatenate(parts, axis=-1), -1.0, 1.0)


# =============================================================================
# Batch helpers for action chunking
# =============================================================================

def decompose_chunked(actions: np.ndarray, horizon_length: int,
                      dir_dims: int = DEFAULT_DIR_DIMS):
    """Decompose a chunked action sequence (N, H, D) -> (N, (D+1)*H)."""
    if actions.ndim == 3:
        N, H, D = actions.shape
    else:
        N, H = actions.shape[0], horizon_length, actions.shape[-1] // horizon_length
        actions = actions.reshape(N, H, D)

    parts = [decompose(actions[:, t, :], dir_dims=dir_dims) for t in range(H)]
    return np.concatenate(parts, axis=-1)
