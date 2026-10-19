import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.fql_posthoc_reference import FQLPosthocReferenceAgent, get_config
from direction_speed.adapter import DirectionSpeedAdapter
from main_fql_reference import environment_action, prepare_batch, evaluate_policy, make_learner_config


def make_agent(**overrides):
    config = get_config().to_dict()
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), flow_steps=2)
    config.update(overrides)
    return FQLPosthocReferenceAgent.create(73, np.zeros(7, np.float32), np.zeros(6, np.float32), config)


def mapper():
    return DirectionSpeedAdapter(speed_squash='tanh', speed_interpretation='log', speed_bound='sphere')


def test_raw_replay_is_unchanged_and_zero_action_encoding_is_not_clamped():
    raw = dict(observations=np.ones((3, 7), np.float32), next_observations=np.ones((3, 7), np.float32),
               actions=np.asarray([[0, 0, 0, 0, 0], [.2, .1, 0, .5, -.5], [-1, 1, 1, -.7, .1]], np.float32),
               rewards=np.asarray([-1, 0, -1], np.float32), masks=np.asarray([1, 0, 1], np.float32))
    original = {name: value.copy() for name, value in raw.items()}
    encoded = prepare_batch(raw, mapper())
    assert encoded['actions'].shape == (3, 6) and encoded['actions'][0, 3] < -1
    np.testing.assert_array_equal(encoded['actions'], mapper().decompose(raw['actions']))
    for name in raw: np.testing.assert_array_equal(raw[name], original[name])
    agent, info = make_agent().update(encoded)
    assert int(agent.network.step) == 2
    assert all(np.isfinite(np.asarray(value)).all() for value in info.values())
    restored = flax.serialization.from_bytes(make_agent(), flax.serialization.to_bytes(agent))
    for x, y in zip(jax.tree_util.tree_leaves(agent.network.params), jax.tree_util.tree_leaves(restored.network.params), strict=True):
        np.testing.assert_array_equal(x, y)


def test_latent_critic_shape_and_execution_scalar_values_are_consistent():
    agent = make_agent(); key = jax.random.PRNGKey(54)
    latent, executed = environment_action(agent, np.zeros(7, np.float32), key, mapper())
    assert latent.shape == (6,) and executed.shape == (5,)
    assert agent.config['env_action_dim'] == 5
    assert agent.config['policy_action_dim'] == agent.config['critic_action_dim'] == 6
    assert agent.config['policy_output_transform'] == 'clip'
    np.testing.assert_array_equal(executed, mapper().compose(latent[None])[0])
    np.testing.assert_array_equal(executed[3:], latent[4:])


class TwoStepEnvironment:
    def __init__(self): self.seeds=[]; self.actions=[]
    def reset(self, seed=None):
        self.seeds.append(seed); self.steps=0
        return np.zeros(7, np.float32), {}
    def step(self, action):
        self.actions.append(action.copy()); self.steps+=1
        return np.ones(7, np.float32), -1., False, self.steps==2, {'success': float(self.seeds[-1] % 2)}


def test_evaluation_observes_episode_counts_uses_external_rng_and_does_not_update():
    agent=make_agent(); env=TwoStepEnvironment(); before=flax.serialization.to_bytes(agent)
    result=evaluate_policy(agent, env, mapper(), seed=40, episodes=3)
    assert result['episode_count']==3 and result['success_count']==1
    assert result['success']==1/3 and [row['length'] for row in result['episodes']]==[2]*3
    assert env.seeds==[40,41,42] and len(env.actions)==6
    assert flax.serialization.to_bytes(agent)==before


@pytest.mark.parametrize('overrides', [dict(horizon_length=5), dict(adapter_representation='spherical'),
    dict(adapter_speed_interpretation='direct'), dict(ds_speed_bound='cube'), dict(adapter_critic_space='env'),
    dict(adapter_critic_space='canonical'), dict(policy_output_transform='tanh'), dict(adapter_speed_squash='sigmoid'),
    dict(policy_action_dim=5), dict(env_action_dim=6), dict(critic_action_dim=5)])
def test_unsupported_posthoc_recipe_is_rejected(overrides):
    with pytest.raises(ValueError): make_agent(**overrides)


def test_driver_does_not_accept_unknown_learner_overrides():
    with pytest.raises(ValueError, match='Unknown learner'):
        make_learner_config(dict(method='posthoc', learner={'typo_alpha': 3}))
