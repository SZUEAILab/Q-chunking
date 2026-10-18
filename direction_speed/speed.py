"""Independent actor-squash and adapter-interpretation modules for speed."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from direction_speed.specs import SpeedInterpretation, SpeedSquash


class LogSpeedInterpreter:
    """Interpret the policy coordinate as log-speed."""

    @staticmethod
    def encode_numpy(radial_fraction, eps):
        return np.log(np.maximum(radial_fraction, eps))

    @staticmethod
    def decode_numpy(policy_value):
        return np.exp(policy_value)

    @staticmethod
    def encode_jax(radial_fraction, eps):
        return jnp.log(jnp.maximum(radial_fraction, eps))

    @staticmethod
    def decode_jax(policy_value):
        return jnp.exp(policy_value)


class DirectSpeedInterpreter:
    """Treat the actor output as speed without an adapter-side transform."""

    @staticmethod
    def encode_numpy(radial_fraction, eps):
        del eps
        return radial_fraction

    @staticmethod
    def decode_numpy(policy_value):
        return policy_value

    @staticmethod
    def encode_jax(radial_fraction, eps):
        del eps
        return radial_fraction

    @staticmethod
    def decode_jax(policy_value):
        return policy_value


SPEED_INTERPRETER_REGISTRY = {
    SpeedInterpretation.LOG: LogSpeedInterpreter(),
    SpeedInterpretation.DIRECT: DirectSpeedInterpreter(),
}


@dataclass(frozen=True)
class AdapterSpeedPipeline:
    """Select actor output squash and adapter-side speed semantics independently."""

    squash: SpeedSquash | str = SpeedSquash.SIGMOID
    interpretation: SpeedInterpretation | str = SpeedInterpretation.DIRECT

    def __post_init__(self):
        object.__setattr__(self, "squash", SpeedSquash(self.squash))
        object.__setattr__(
            self, "interpretation", SpeedInterpretation(self.interpretation)
        )

    @property
    def actor_squash_type(self):
        return self.squash.value

    @property
    def interpreter(self):
        return SPEED_INTERPRETER_REGISTRY[self.interpretation]
