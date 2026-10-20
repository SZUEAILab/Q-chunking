"""Independent native RLPD recipe ablations; frozen reference classes unchanged.

The target intervention averages the *same two sampled heads* instead of taking
their minimum. Head count, RNG consumption, actor loss and optimizer schedule
are inherited from the reference. This is not a ten-head-mean target variant.
"""

import jax
import jax.numpy as jnp
import optax

from agents.rlpd_reference import RLPDReferenceAgent, get_config as reference_config


class RLPDRecipeAblationAgent(RLPDReferenceAgent):
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        if config.get('recipe_target_reduction') not in ('min', 'mean'):
            raise ValueError('Recipe target reduction must be min or mean.')
        if config['num_qs'] != 10 or config['num_min_qs'] != 2:
            raise ValueError('Recipe ablation holds ten heads and a two-head draw fixed.')
        return super().create(seed, ex_observations, ex_actions, config)

    def update_critic(self, batch):
        dist = self.actor.apply_fn({'params': self.actor.params}, batch['next_observations'])
        key, rng = jax.random.split(self.rng)
        next_actions = dist.sample(seed=key)
        key, rng = jax.random.split(rng)
        params = self.target_critic.params
        indices = jax.random.choice(key, jnp.arange(self.config['num_qs']), shape=(2,), replace=False)
        if 'Ensemble_0' in params:
            params = params.copy(add_or_replace={'Ensemble_0': jax.tree_util.tree_map(
                lambda p: p[indices], params['Ensemble_0'])})
        else:
            params = jax.tree_util.tree_map(lambda p: p[indices], params)
        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {'params': params}, batch['next_observations'], next_actions, True,
            rngs={'dropout': key})
        next_q = (next_qs.min(axis=0) if self.config['recipe_target_reduction'] == 'min'
                  else next_qs.mean(axis=0))
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        if self.config['backup_entropy']:
            target_q -= (self.config['discount'] * batch['masks']
                         * self.temp.apply_fn({'params': self.temp.params})
                         * dist.log_prob(next_actions))
        key, rng = jax.random.split(rng)

        def loss(params):
            qs = self.critic.apply_fn({'params': params}, batch['observations'],
                                      batch['actions'], True, rngs={'dropout': key})
            value = jnp.square(qs - target_q).mean()
            return value, {'critic_loss': value, 'q': qs.mean()}

        grads, info = jax.grad(loss, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target = self.target_critic.replace(params=optax.incremental_update(
            critic.params, self.target_critic.params, self.config['tau']))
        return self.replace(critic=critic, target_critic=target, rng=rng), info


def get_config():
    config = reference_config()
    config.agent_name = 'rlpd_recipe_ablation'
    config.recipe_target_reduction = 'min'
    return config
