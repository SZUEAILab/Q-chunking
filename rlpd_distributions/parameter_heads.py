"""Actor parameter heads sized by a distribution schema rather than action dims."""

from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from rlpd_networks import default_init


class DistributionParameterHeads(nn.Module):
    """Produce independently sized location and log-scale parameter vectors."""

    location_size: int
    log_scale_size: int
    log_scale_min: Optional[float] = -20
    log_scale_max: Optional[float] = 2

    @nn.compact
    def __call__(self, features):
        location_parameters = nn.Dense(
            self.location_size,
            kernel_init=default_init(),
            name="LocationParameters",
        )(features)
        log_scale_parameters = nn.Dense(
            self.log_scale_size,
            kernel_init=default_init(),
            name="LogScaleParameters",
        )(features)
        log_scale_parameters = jnp.clip(
            log_scale_parameters, self.log_scale_min, self.log_scale_max
        )
        return location_parameters, log_scale_parameters
