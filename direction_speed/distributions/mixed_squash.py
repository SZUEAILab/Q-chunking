"""Gaussian policy with a per-coordinate action-space transform."""

from typing import Optional, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from rlpd_distributions.parameter_heads import DistributionParameterHeads
from rlpd_distributions.pipeline import BijectedActionDistribution


tfd = tfp.distributions
tfb = tfp.bijectors


class MixedSquashBijector(tfb.Bijector):
    def __init__(self, squash_types: Sequence[str], eps: float = 1e-6):
        invalid = set(squash_types) - {"identity", "tanh", "sigmoid"}
        if invalid:
            raise ValueError(f"Unknown squash types: {sorted(invalid)}")
        super().__init__(forward_min_event_ndims=1, name="mixed_squash")
        self._types = tuple(squash_types)
        self._eps = eps

    def _forward(self, x):
        outputs = []
        for idx, kind in enumerate(self._types):
            value = x[..., idx : idx + 1]
            if kind == "tanh":
                value = jnp.tanh(value)
            elif kind == "sigmoid":
                value = self._eps + (1.0 - 2.0 * self._eps) * jax.nn.sigmoid(value)
            outputs.append(value)
        return jnp.concatenate(outputs, axis=-1)

    def _inverse(self, y):
        outputs = []
        for idx, kind in enumerate(self._types):
            value = y[..., idx : idx + 1]
            if kind == "tanh":
                value = jnp.clip(value, -1.0 + self._eps, 1.0 - self._eps)
                value = jnp.arctanh(value)
            elif kind == "sigmoid":
                value = jnp.clip(value, self._eps, 1.0 - self._eps)
                unit = (value - self._eps) / (1.0 - 2.0 * self._eps)
                unit = jnp.clip(unit, self._eps, 1.0 - self._eps)
                value = jnp.log(unit) - jnp.log1p(-unit)
            outputs.append(value)
        return jnp.concatenate(outputs, axis=-1)

    def _forward_log_det_jacobian(self, x):
        terms = []
        for idx, kind in enumerate(self._types):
            value = x[..., idx]
            if kind == "tanh":
                term = 2.0 * (jnp.log(2.0) - value - jax.nn.softplus(-2.0 * value))
            elif kind == "sigmoid":
                term = (
                    jnp.log(1.0 - 2.0 * self._eps)
                    + jax.nn.log_sigmoid(value)
                    + jax.nn.log_sigmoid(-value)
                )
            else:
                term = jnp.zeros_like(value)
            terms.append(term)
        return jnp.sum(jnp.stack(terms, axis=-1), axis=-1)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))


class MixedSquashNormal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    squash_types: Sequence[str]
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        if len(self.squash_types) != self.action_dim:
            raise ValueError("squash_types must match action_dim")
        x = self.base_cls()(inputs, *args, **kwargs)
        means, log_stds = DistributionParameterHeads(
            location_size=self.action_dim,
            log_scale_size=self.action_dim,
            log_scale_min=self.log_std_min,
            log_scale_max=self.log_std_max,
            name="ParameterHeads",
        )(x)
        base = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return BijectedActionDistribution(
            distribution=base,
            bijector=MixedSquashBijector(self.squash_types),
        )
