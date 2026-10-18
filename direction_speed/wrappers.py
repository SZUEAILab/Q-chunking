"""Execution pipeline exposing environment actions from policy actions."""

from dataclasses import dataclass
from typing import Any

from direction_speed.postprocessors import ActionPostprocessor


@dataclass(frozen=True)
class ActionExecutionPipeline:
    """Keep actor sampling and environment postprocessing as separate stages."""

    agent: Any
    postprocessor: ActionPostprocessor
    policy_action_dim: int

    def sample_policy_actions(self, observations, rng=None):
        return self.agent.sample_actions(observations, rng=rng)

    def policy_to_env(self, action_pi):
        return self.postprocessor.compose_policy_output(
            action_pi,
            self.policy_action_dim,
        )

    def sample_actions(self, observations, rng=None):
        action_pi = self.sample_policy_actions(observations, rng=rng)
        return self.policy_to_env(action_pi)
