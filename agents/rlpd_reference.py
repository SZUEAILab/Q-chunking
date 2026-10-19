"""RLPD learner recipe port for the project's one-step OGBench driver.

Update equations, key splitting, network initialization and optimizer separation
follow ikostrikov/rlpd at c90fd4baf28c9c9ef40a81460a2e395092844f88.
The driver, datasets, evaluation and externally seeded action sampling are local
adaptations. This is not an upstream OGBench benchmark or a tuned Cube baseline.

Adapted from code under the MIT license, copyright (c) 2022 Ilya Kostrikov,
Philip J. Ball, Laura Smith. See docs/reference/rlpd-upstream-LICENSE.txt.
"""

from functools import partial
from typing import Any

import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from tensorflow_probability.substrates import jax as tfp

from agents.acrlpd import Temperature, get_config as project_config
from rlpd_networks import Ensemble, MLP, StateActionValue
from utils.flax_utils import nonpytree_field


UPSTREAM_COMMIT = "c90fd4baf28c9c9ef40a81460a2e395092844f88"


class _TanhDistribution(tfp.distributions.TransformedDistribution):
    """The upstream bijector, including its density and sampling behavior."""

    def __init__(self, distribution, validate_args=False):
        super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(),
                         validate_args=validate_args)

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype=None, num_classes=None):
        properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del properties["bijector"]
        return properties


class _UpstreamTanhNormal(nn.Module):
    base_cls: Any
    action_dim: int

    @nn.compact
    def __call__(self, observations):
        features = self.base_cls()(observations)
        means = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform(),
                         name="OutputDenseMean")(features)
        log_stds = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform(),
                            name="OutputDenseLogStd")(features)
        return _TanhDistribution(tfp.distributions.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(jnp.clip(log_stds, -20, 2))))


class RLPDReferenceAgent(flax.struct.PyTreeNode):
    rng: Any
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        cls._validate_learner_config(config)
        if (config.get("adapter_representation", "none") != "none"
                or config.get("use_integrated_ds", False)
                or config.get("use_cartesian_l2_projection", False)):
            raise ValueError("RLPD reference port currently supports native actions only.")
        action_dim = ex_actions.shape[-1]
        for name in ("action_dim", "env_action_dim", "policy_action_dim", "critic_action_dim"):
            if name in config and config[name] != action_dim:
                raise ValueError(f"Reference native action dimension mismatch: {name}")
            config[name] = action_dim
        if config["target_entropy"] is None:
            config["target_entropy"] = -action_dim / 2
        return cls._create_networks(seed, ex_observations, ex_actions, config)

    @staticmethod
    def _validate_learner_config(config):
        if config["horizon_length"] != 1:
            raise ValueError("RLPD reference port requires horizon_length=1.")
        if (config.get("actor_update_mode", "sac") != "sac"
                or config.get("ds_phase_regularization", "none") != "none"
                or config.get("ds_stage_speed_constraint", "none") != "none"
                or config.get("ds_separate_alpha", False)
                or config.get("bc_alpha", 0) != 0
                or config.get("fixed_alpha", -1) > 0):
            raise ValueError("Reference recipe does not support project actor interventions.")
        num_qs, subset = config["num_qs"], config["num_min_qs"]
        if not isinstance(num_qs, int) or num_qs < 1:
            raise ValueError("num_qs must be a positive integer.")
        if subset is not None and (not isinstance(subset, int) or not 1 <= subset <= num_qs):
            raise ValueError("num_min_qs must be None or an integer in [1, num_qs].")

    @classmethod
    def _create_networks(cls, seed, ex_observations, ex_actions, config):
        """Build validated shapes with the upstream initialization and key order."""
        action_dim = config["policy_action_dim"]
        num_qs, subset = config["num_qs"], config["num_min_qs"]
        config["reference_upstream_commit"] = UPSTREAM_COMMIT
        rng, actor_key, critic_key, temp_key = jax.random.split(jax.random.PRNGKey(seed), 4)
        actor_base = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
        actor_def = _UpstreamTanhNormal(actor_base, action_dim)
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=flax.core.freeze(actor_def.init(actor_key, ex_observations)["params"]),
            tx=optax.adam(config["actor_lr"]))
        critic_base = partial(MLP, hidden_dims=config["value_hidden_dims"], activate_final=True,
                              use_layer_norm=config["layer_norm"])
        critic_cls = partial(StateActionValue, base_cls=critic_base)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = flax.core.freeze(critic_def.init(critic_key, ex_observations, ex_actions)["params"])
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params,
                                   tx=optax.adam(config["critic_lr"]))
        target_def = Ensemble(critic_cls, num=subset or num_qs)
        target = TrainState.create(apply_fn=target_def.apply, params=critic_params,
                                   tx=optax.GradientTransformation(lambda _: None, lambda _: None))
        temp_def = Temperature(config["init_temp"])
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=flax.core.freeze(temp_def.init(temp_key)["params"]),
            tx=optax.adam(config["temp_lr"]))
        return cls(rng, actor, critic, target, temp, flax.core.FrozenDict(config))

    def update_critic(self, batch):
        dist = self.actor.apply_fn({"params": self.actor.params}, batch["next_observations"])
        key, rng = jax.random.split(self.rng)
        next_actions = dist.sample(seed=key)
        key, rng = jax.random.split(rng)
        params = self.target_critic.params
        subset = self.config["num_min_qs"]
        if subset is not None:
            indices = jax.random.choice(key, jnp.arange(self.config["num_qs"]),
                                        shape=(subset,), replace=False)
            if "Ensemble_0" in params:
                params = params.copy(add_or_replace={"Ensemble_0": jax.tree_util.tree_map(
                    lambda p: p[indices], params["Ensemble_0"])})
            else:
                params = jax.tree_util.tree_map(lambda p: p[indices], params)
        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": params}, batch["next_observations"], next_actions, True,
            rngs={"dropout": key})
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_qs.min(axis=0)
        if self.config["backup_entropy"]:
            target_q -= (self.config["discount"] * batch["masks"]
                         * self.temp.apply_fn({"params": self.temp.params})
                         * dist.log_prob(next_actions))
        key, rng = jax.random.split(rng)

        def loss(params):
            qs = self.critic.apply_fn({"params": params}, batch["observations"],
                                      batch["actions"], True, rngs={"dropout": key})
            value = jnp.square(qs - target_q).mean()
            return value, {"critic_loss": value, "q": qs.mean()}

        grads, info = jax.grad(loss, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target = self.target_critic.replace(params=optax.incremental_update(
            critic.params, self.target_critic.params, self.config["tau"]))
        return self.replace(critic=critic, target_critic=target, rng=rng), info

    def update_actor(self, batch):
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def loss(params):
            dist = self.actor.apply_fn({"params": params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            q = self.critic.apply_fn({"params": self.critic.params}, batch["observations"],
                                     actions, True, rngs={"dropout": key2}).mean(axis=0)
            value = (log_probs * self.temp.apply_fn({"params": self.temp.params}) - q).mean()
            return value, {"actor_loss": value, "entropy": -log_probs.mean()}

        grads, info = jax.grad(loss, has_aux=True)(self.actor.params)
        return self.replace(actor=self.actor.apply_gradients(grads=grads), rng=rng), info

    def update_temperature(self, entropy):
        def loss(params):
            temperature = self.temp.apply_fn({"params": params})
            value = temperature * (entropy - self.config["target_entropy"]).mean()
            return value, {"temperature": temperature, "temperature_loss": value}

        grads, info = jax.grad(loss, has_aux=True)(self.temp.params)
        return self.replace(temp=self.temp.apply_gradients(grads=grads)), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch, utd_ratio=1):
        if utd_ratio < 1:
            raise ValueError("utd_ratio must be positive.")
        agent = self
        for index in range(utd_ratio):
            def take(values):
                if values.shape[0] % utd_ratio:
                    raise ValueError("Batch size must be divisible by utd_ratio.")
                size = values.shape[0] // utd_ratio
                return values[index * size:(index + 1) * size]
            minibatch = jax.tree_util.tree_map(take, batch)
            agent, critic_info = agent.update_critic(minibatch)
        agent, actor_info = agent.update_actor(minibatch)
        agent, temp_info = agent.update_temperature(actor_info["entropy"])
        return agent, {**actor_info, **critic_info, **temp_info}

    @jax.jit
    def batch_update(self, batch):
        """Adapt the driver's (UTD, batch, H=1, ...) sequence layout."""
        utd, size = batch["observations"].shape[:2]
        flat = {"observations": batch["observations"].reshape((utd * size, -1))}
        for name in ("next_observations", "actions", "rewards", "masks"):
            values = batch[name]
            if values.shape[2] != 1:
                raise ValueError("RLPD reference requires one-step sequence batches.")
            flat[name] = values[:, :, 0].reshape((utd * size,) + values.shape[3:])
        return self.update(flat, utd_ratio=utd)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        # Local driver owns interaction/evaluation RNG; learner RNG is unchanged.
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        return jnp.clip(dist.sample(seed=rng), -1, 1)


def get_config():
    config = project_config()
    config.agent_name = "rlpd_reference"
    config.actor_hidden_dims = (256, 256)
    config.value_hidden_dims = (256, 256)
    config.actor_lr = config.critic_lr = config.temp_lr = 3e-4
    config.num_qs = 10
    config.num_min_qs = 2
    config.layer_norm = True
    config.backup_entropy = True
    config.reference_upstream_commit = UPSTREAM_COMMIT
    # Mark ignored project fields explicitly, preventing a false effective recipe.
    del config["lr"]
    del config["q_agg"]
    return config
