"""Policy-action to environment-action boundary interfaces."""

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
import numpy as np


class ActionPostprocessor(Protocol):
    """Execution-only transform from ``action_pi`` to ``action_env``."""

    def compose_policy_output(
        self, policy_output: np.ndarray, policy_action_dim: int
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class IdentityActionPostprocessor:
    """Identity execution transform, including action-chunk reshaping."""

    def compose_policy_output(
        self, policy_output: np.ndarray, policy_action_dim: int
    ) -> np.ndarray:
        return np.asarray(policy_output).reshape(-1, policy_action_dim)

    def compose(self, policy_actions: np.ndarray) -> np.ndarray:
        return np.asarray(policy_actions)

    def decompose(self, env_actions: np.ndarray) -> np.ndarray:
        return np.asarray(env_actions)

    def compose_jax(self, policy_actions: jnp.ndarray) -> jnp.ndarray:
        return policy_actions

    def decompose_jax(self, env_actions: jnp.ndarray) -> jnp.ndarray:
        return env_actions


@dataclass(frozen=True)
class L2BallProjectionPostprocessor:
    """Project the first spatial coordinates onto an L2 ball at execution.

    This is the deliberately simple counterfactual for direction--speed:
    retain the ordinary Cartesian TanhNormal policy, but enforce the same
    unit-sphere translation constraint as the best DS adapter.
    """

    spatial_dims: int = 3
    max_norm: float = 1.0
    eps: float = 1e-8

    def __post_init__(self):
        if self.spatial_dims < 1:
            raise ValueError("spatial_dims must be positive")
        if self.max_norm <= 0.0:
            raise ValueError("max_norm must be positive")

    def compose_policy_output(
        self, policy_output: np.ndarray, policy_action_dim: int
    ) -> np.ndarray:
        actions = np.asarray(policy_output).reshape(-1, policy_action_dim)
        return self.compose(actions)

    def compose(self, policy_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(policy_actions)
        if actions.shape[-1] < self.spatial_dims:
            raise ValueError(
                f"Expected at least {self.spatial_dims} action dimensions, "
                f"got {actions.shape[-1]}"
            )
        spatial = actions[..., : self.spatial_dims]
        norm = np.linalg.norm(spatial, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.max_norm / np.maximum(norm, self.eps))
        return np.concatenate(
            (spatial * scale, actions[..., self.spatial_dims :]), axis=-1
        )

    def compose_jax(self, policy_actions: jnp.ndarray) -> jnp.ndarray:
        spatial = policy_actions[..., : self.spatial_dims]
        norm = jnp.linalg.norm(spatial, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, self.max_norm / jnp.maximum(norm, self.eps))
        return jnp.concatenate(
            (spatial * scale, policy_actions[..., self.spatial_dims :]), axis=-1
        )
