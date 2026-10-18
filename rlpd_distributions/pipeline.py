"""Explicit raw-action sampling and policy-action transformation interfaces."""

import tensorflow_probability.substrates.jax as tfp


tfd = tfp.distributions


class BijectedActionDistribution(tfd.TransformedDistribution):
    """A transformed distribution exposing every raw -> policy action stage."""

    @property
    def raw_distribution(self):
        return self.distribution

    @property
    def policy_transform(self):
        return self.bijector

    def sample_raw(self, seed=None):
        return self.raw_distribution.sample(seed=seed)

    def raw_mode(self):
        return self.raw_distribution.mode()

    def raw_log_prob(self, action_raw):
        return self.raw_distribution.log_prob(action_raw)

    def raw_to_policy(self, action_raw):
        return self.policy_transform.forward(action_raw)

    def policy_to_raw(self, action_pi):
        return self.policy_transform.inverse(action_pi)

    def forward_log_det(self, action_raw):
        return self.policy_transform.forward_log_det_jacobian(
            action_raw, event_ndims=1
        )

    def sample(self, sample_shape=(), seed=None, name="sample", **kwargs):
        action_raw = self.raw_distribution.sample(
            sample_shape=sample_shape, seed=seed, name=name, **kwargs
        )
        return self.raw_to_policy(action_raw)

    def mode(self):
        return self.raw_to_policy(self.raw_mode())

    def log_prob(self, action_pi, name="log_prob", **kwargs):
        del name
        action_raw = self.policy_to_raw(action_pi)
        return self.raw_log_prob(action_raw) - self.forward_log_det(action_raw)
