"""Original posthoc action adapter with the pinned upstream RLPD learner recipe.

The inherited learner operates on six-dimensional latent actions. The existing
driver decomposes prior/replay actions and composes environment/evaluation
actions. Entropy, including entropy backup, is measured in latent coordinates.
This is a local method adaptation, not an upstream implementation of posthoc.
"""

import numpy as np

from agents.rlpd_reference import RLPDReferenceAgent, get_config as reference_config


class RLPDPosthocReferenceAgent(RLPDReferenceAgent):
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        cls._validate_learner_config(config)
        required = dict(agent_name="rlpd_posthoc_reference", adapter_representation="cartesian", adapter_speed_squash="tanh",
                        adapter_speed_interpretation="log", ds_speed_bound="sphere",
                        adapter_critic_space="latent", ds_regularizer="policy_coordinate")
        if any(config.get(name) != value for name, value in required.items()):
            raise ValueError("Reference posthoc requires original Cartesian tanh/log/sphere latent recipe.")
        if any(config.get(name, False) for name in ("use_integrated_ds", "use_cartesian_l2_projection",
                                                   "ds_moment_controlled", "ds_separate_actor_trunks")):
            raise ValueError("Reference posthoc does not include other policy interventions.")
        if np.shape(ex_actions) != (6,):
            raise ValueError("Reference posthoc expects one six-dimensional encoded action.")
        for name, value in dict(env_action_dim=5, policy_action_dim=6, critic_action_dim=6, action_dim=6).items():
            if name in config and config[name] != value:
                raise ValueError("Reference posthoc action dimension mismatch: " + name)
            config[name] = value
        if config["target_entropy"] is None:
            config["target_entropy"] = -3.
        config["reference_action_adaptation"] = "original_posthoc_latent"
        return cls._create_networks(seed, ex_observations, ex_actions, config)


def get_config():
    config = reference_config()
    config.agent_name = "rlpd_posthoc_reference"
    config.adapter_representation = "cartesian"
    config.adapter_speed_squash = "tanh"
    config.adapter_speed_interpretation = "log"
    config.adapter_critic_space = "latent"
    config.ds_speed_bound = "sphere"
    config.ds_regularizer = "policy_coordinate"
    return config
