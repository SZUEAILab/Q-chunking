"""Original log/sphere execution map with the pinned FQL latent learner.

FQL retains its hard-clipped flow outputs and teacher/student objectives. This
is an explicit FQL adaptation, not a tanh-Gaussian actor or an entropy objective.
"""
from agents.fql_reference import FQLReferenceAgent, get_config as native_config


class FQLPosthocReferenceAgent(FQLReferenceAgent):
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        cls._validate_learner_config(config)
        cls._validate_method_config(config)
        policy_dim = ex_actions.shape[-1]
        env_dim = config.get('env_action_dim', policy_dim - 1)
        if env_dim < 3 or policy_dim != env_dim + 1:
            raise ValueError('Posthoc policy dimensions must equal environment dimensions plus one.')
        for name in ('policy_action_dim', 'critic_action_dim'):
            if name in config and config[name] != policy_dim:
                raise ValueError('Posthoc latent dimension mismatch: ' + name)
            config[name] = policy_dim
        config['env_action_dim'] = env_dim
        return cls._create_networks(seed, ex_observations, ex_actions, config)

    @staticmethod
    def _validate_method_config(config):
        if (config.get('adapter_representation') != 'cartesian'
                or config.get('adapter_speed_interpretation') != 'log'
                or config.get('ds_speed_bound') != 'sphere'
                or config.get('adapter_critic_space') != 'latent'
                or config.get('policy_output_transform') != 'clip'):
            raise ValueError('FQL posthoc requires Cartesian log/sphere latent coordinates and clipped flow outputs.')
        # speed_squash names the shared codec convention; FQL does not apply its actor squash.
        if config.get('adapter_speed_squash') != 'tanh':
            raise ValueError('Use the original codec convention; FQL policy outputs remain hard-clipped.')


def get_config():
    config = native_config()
    config.update(dict(agent_name='fql_posthoc_reference', adapter_representation='cartesian',
                       adapter_speed_squash='tanh', adapter_speed_interpretation='log',
                       ds_speed_bound='sphere', adapter_critic_space='latent', policy_output_transform='clip'))
    return config
