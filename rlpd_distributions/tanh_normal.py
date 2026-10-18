import functools
from typing import Optional, Type

import tensorflow_probability

from rlpd_distributions.tanh_transformed import TanhTransformedDistribution

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

import flax.linen as nn
import jax.numpy as jnp

from rlpd_networks import default_init
from rlpd_distributions.parameter_heads import DistributionParameterHeads


class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
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
                "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )
            log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)
