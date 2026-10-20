import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import linen as nn

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from rlpd_networks import Ensemble, StateActionValue, MLP
from direction_speed import DirectionSpeedAdapter
from direction_speed.adapter import AUXILIARY_EXECUTION_MAPS, validate_auxiliary_control
from direction_speed.postprocessors import L2BallProjectionPostprocessor
from direction_speed.distributions import assemble_action_pipeline


from functools import partial


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


def reward_stage_weight(
    rewards,
    reward_floor=-3.0,
    decay=0.5,
    max_progress=2.0,
):
    """Return a fixed translation-regularization weight for task progress.

    Cube-triple's relabeled reward is -3, -2, or -1 after zero, one, or two
    subtasks have been completed.  The weight therefore becomes 1, ``decay``,
    or ``decay**2``.  Clipping makes the helper well-defined for replay data
    outside that nominal range and stop-gradient makes the schedule an
    experimental intervention rather than another learned signal.
    """
    progress = jnp.clip(rewards - reward_floor, 0.0, max_progress)
    return jax.lax.stop_gradient(jnp.power(decay, progress))


def reward_stage_speed_cap(
    rewards,
    base_cap=1.0,
    decay=0.5,
    reward_floor=-3.0,
    max_progress=2.0,
):
    """Return the maximum translation norm allowed at each task stage.

    For cube-triple, rewards ``-3, -2, -1`` encode zero, one, and two
    completed subtasks.  With the default settings, their caps are therefore
    ``1, 0.5, 0.25``.  The label is used only while training; at execution the
    actor receives the ordinary observation and needs no reward-stage input.
    """
    progress = jnp.clip(rewards - reward_floor, 0.0, max_progress)
    caps = base_cap * jnp.power(decay, progress)
    return jax.lax.stop_gradient(caps)


def normalized_advantage_weights(advantages, temperature=1.0, max_weight=20.0):
    """Return stopped, mean-one exponential weights for replay AWR."""
    weights = jnp.exp(
        jnp.clip(advantages / temperature, -20.0, jnp.log(max_weight))
    )
    return jax.lax.stop_gradient(
        weights / jnp.maximum(weights.mean(), 1e-6)
    )


def translation_speed_cap_statistics(
    env_actions,
    stage_rewards,
    env_action_dim,
    repetitions=1,
    base_cap=1.0,
    decay=0.5,
    reward_floor=-3.0,
    max_progress=2.0,
):
    """Compute per-sample speed-cap violations in executed action space.

    The first three coordinates of each one-step environment action are its
    Cartesian translation.  For an action chunk, the squared hinge penalty is
    averaged across chunk elements so that its scale is comparable across
    different horizons.
    """
    actions = jnp.reshape(
        env_actions, env_actions.shape[:-1] + (repetitions, env_action_dim)
    )
    translation_norms = jnp.linalg.norm(actions[..., :3], axis=-1)
    caps = reward_stage_speed_cap(
        stage_rewards,
        base_cap=base_cap,
        decay=decay,
        reward_floor=reward_floor,
        max_progress=max_progress,
    )
    violations = jax.nn.relu(translation_norms - caps[..., None])
    return {
        "penalty": jnp.mean(jnp.square(violations), axis=-1),
        "translation_norm": jnp.mean(translation_norms, axis=-1),
        "cap": caps,
        "violation_fraction": jnp.mean(violations > 0.0, axis=-1),
    }


def _adapter_from_config(config):
    return DirectionSpeedAdapter(
        representation=config["adapter_representation"],
        speed_squash=config["adapter_speed_squash"],
        speed_interpretation=config["adapter_speed_interpretation"],
        speed_bound=config.get("ds_speed_bound", "cube"),
        execution_map=config.get("adapter_execution_map", "direction_speed"),
    )


def _clip_to_policy_support(actions, squash_types, eps=1e-5):
    clipped = []
    for idx, kind in enumerate(squash_types):
        value = actions[..., idx : idx + 1]
        if kind == "tanh":
            value = jnp.clip(value, -1.0 + eps, 1.0 - eps)
        elif kind == "sigmoid":
            value = jnp.clip(value, eps, 1.0 - eps)
        clipped.append(value)
    return jnp.concatenate(clipped, axis=-1)


def _adapter_policy_to_env(actions, config):
    """Compose flattened/chunked adapter actions into environment actions."""
    adapter = _adapter_from_config(config)
    policy_dim = config["action_dim"]
    env_dim = config["env_action_dim"]
    repetitions = config["horizon_length"] if config["action_chunking"] else 1
    leading_shape = actions.shape[:-1]
    one_step = jnp.reshape(actions, (-1, policy_dim))
    env_actions = adapter.compose_jax(one_step)
    return jnp.reshape(env_actions, leading_shape + (repetitions * env_dim,))


def _adapter_env_to_policy(actions, config):
    """Decompose flattened/chunked environment actions for BC log-prob."""
    adapter = _adapter_from_config(config)
    policy_dim = config["action_dim"]
    env_dim = config["env_action_dim"]
    repetitions = config["horizon_length"] if config["action_chunking"] else 1
    leading_shape = actions.shape[:-1]
    one_step = jnp.reshape(actions, (-1, env_dim))
    policy_actions = adapter.decompose_jax(one_step)
    return jnp.reshape(policy_actions, leading_shape + (repetitions * policy_dim,))


class ACRLPDAgent(flax.struct.PyTreeNode):
    """Soft actor-critic (SAC) agent with action chunking.

    This agent can also be used for reinforcement learning with prior data (RLPD).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _critic_action(self, action_pi):
        """Use the declared representation for actor and target Q queries."""
        if self.config.get("adapter_representation", "none") != "none":
            space = self.config.get("adapter_critic_space", "latent")
            if space in ("env", "canonical"):
                env_action = _adapter_policy_to_env(action_pi, self.config)
                if space == "canonical":
                    # Replay already uses decompose(env_action). Apply that
                    # same embedding to sampled actions, including clipping,
                    # while retaining the declared policy-space entropy.
                    return _adapter_env_to_policy(env_action, self.config)
                return env_action
        if not self.config.get("use_cartesian_l2_projection", False):
            return action_pi
        original_shape = action_pi.shape
        one_step_actions = jnp.reshape(action_pi, (-1, self.config["action_dim"]))
        projected = L2BallProjectionPostprocessor().compose_jax(one_step_actions)
        return jnp.reshape(projected, original_shape)

    def _regularizer_log_prob(self, dist, action_pi):
        reference = self.config.get("ds_regularizer", "policy_coordinate")
        if reference in ("action_volume", "semantic_product") and hasattr(
            dist, "reference_log_prob"
        ):
            return dist.reference_log_prob(action_pi, reference)
        return dist.log_prob(action_pi)

    def _batch_action_to_policy(self, batch_action_pi):
        """Map replay actions into the actor distribution's support."""
        if self.config.get("adapter_representation", "none") != "none":
            adapter = _adapter_from_config(self.config)
            if self.config.get("adapter_critic_space", "latent") == "env":
                batch_action_pi = _adapter_env_to_policy(
                    batch_action_pi, self.config
                )
            repetitions = (
                self.config["horizon_length"]
                if self.config["action_chunking"]
                else 1
            )
            squash_types = adapter.squash_types(
                self.config["action_dim"], repetitions
            )
            return _clip_to_policy_support(batch_action_pi, squash_types)
        return jnp.clip(batch_action_pi, -1 + 1e-5, 1 - 1e-5)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss."""

        if self.config["horizon_length"] == 1:
            batch_action_pi = batch["actions"][
                ..., 0, :
            ]  # slice: 2x faster, preserves XLA fusion
        elif self.config["action_chunking"]:
            batch_action_pi = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_action_pi = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)

        next_dist = self.network.select("actor")(batch["next_observations"][..., -1, :])
        next_action_pi = next_dist.sample(seed=sample_rng)
        next_action_pi = self._critic_action(next_action_pi)

        next_qs = self.network.select("target_critic")(
            batch["next_observations"][..., -1, :], next_action_pi
        )
        if self.config["q_agg"] == "min":
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** self.config["horizon_length"])
            * batch["masks"][..., -1]
            * next_q
        )

        q = self.network.select("critic")(
            batch["observations"], batch_action_pi, params=grad_params
        )
        critic_loss = (jnp.square(q - target_q) * batch["valid"][..., -1]).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss."""
        if self.config["horizon_length"] == 1:
            batch_action_pi = batch["actions"][
                ..., 0, :
            ]  # slice: 2x faster, preserves XLA fusion
        elif self.config["action_chunking"]:
            batch_action_pi = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_action_pi = batch["actions"][..., 0, :]

        dist = self.network.select("actor")(batch["observations"], params=grad_params)
        action_pi = dist.sample(seed=rng)
        critic_action_pi = self._critic_action(action_pi)

        # Actor loss.
        qs = self.network.select("critic")(batch["observations"], critic_action_pi)
        q = jnp.mean(qs, axis=0)
        separate_alpha = self.config.get("ds_separate_alpha", False)
        phase_mode = self.config.get("ds_phase_regularization", "none")
        if phase_mode == "reward_stage_translation":
            phase_weight = reward_stage_weight(
                batch["stage_rewards"],
                reward_floor=self.config["ds_phase_reward_floor"],
                decay=self.config["ds_phase_decay"],
                max_progress=self.config["ds_phase_max_progress"],
            )
        else:
            phase_weight = jnp.ones_like(q)
        component_info = {}
        if separate_alpha:
            component_log_probs = dist.semantic_component_log_ratios(action_pi)
            regularization = jnp.zeros_like(q)
            alpha_loss = jnp.zeros(())
            alpha_values = []
            entropies = []
            for name in ("direction", "speed", "scalar"):
                alpha_module = f"alpha_{name}"
                alpha_value = self.network.select(alpha_module)()
                component_weight = (
                    phase_weight if name in ("direction", "speed") else 1.0
                )
                regularization += (
                    component_weight * alpha_value * component_log_probs[name]
                )
                alpha = self.network.select(alpha_module)(params=grad_params)
                entropy = -jax.lax.stop_gradient(component_log_probs[name]).mean()
                component_alpha_loss = alpha * (
                    entropy - self.config[f"target_entropy_{name}"]
                )
                alpha_loss += component_alpha_loss
                alpha_values.append(alpha)
                entropies.append(entropy)
                component_info[f"alpha_{name}"] = alpha
                component_info[f"entropy_{name}"] = entropy
                component_info[f"alpha_loss_{name}"] = component_alpha_loss
            log_probs = sum(component_log_probs.values())
            translation_log_probs = (
                component_log_probs["direction"] + component_log_probs["speed"]
            )
            alpha = sum(alpha_values) / len(alpha_values)
            entropy = sum(entropies)
        else:
            if phase_mode == "reward_stage_translation":
                component_log_probs = dist.semantic_component_log_ratios(action_pi)
                translation_log_probs = (
                    component_log_probs["direction"]
                    + component_log_probs["speed"]
                )
                scalar_log_probs = component_log_probs["scalar"]
                log_probs = translation_log_probs + scalar_log_probs
            else:
                log_probs = self._regularizer_log_prob(dist, action_pi)
                translation_log_probs = jnp.zeros_like(log_probs)
            fixed_alpha = float(self.config.get("fixed_alpha", -1.0))
            if fixed_alpha > 0.0:
                alpha = jnp.asarray(fixed_alpha, dtype=log_probs.dtype)
                alpha_value = alpha
            else:
                alpha = self.network.select("alpha")(params=grad_params)
                alpha_value = self.network.select("alpha")()
            if phase_mode == "reward_stage_translation":
                regularization = alpha_value * (
                    phase_weight * translation_log_probs + scalar_log_probs
                )
            else:
                regularization = log_probs * alpha_value
            entropy = -jax.lax.stop_gradient(log_probs).mean()
            if fixed_alpha > 0.0:
                alpha_loss = jnp.zeros(())
            else:
                alpha_loss = (
                    alpha * (entropy - self.config["target_entropy"])
                ).mean()

        phase_info = {}
        if phase_mode == "reward_stage_translation":
            late_mask = (phase_weight < 1.0).astype(log_probs.dtype)
            early_mask = 1.0 - late_mask

            def masked_mean(values, mask):
                return (values * mask).sum() / jnp.maximum(mask.sum(), 1.0)

            translation_entropy = -jax.lax.stop_gradient(translation_log_probs)
            phase_info = {
                "phase_weight": phase_weight.mean(),
                "phase_late_fraction": late_mask.mean(),
                "translation_entropy_early": masked_mean(
                    translation_entropy, early_mask
                ),
                "translation_entropy_late": masked_mean(
                    translation_entropy, late_mask
                ),
            }

        speed_cap_mode = self.config.get("ds_stage_speed_constraint", "none")
        speed_cap_info = {}
        if speed_cap_mode == "reward_stage_cap":
            repetitions = (
                self.config["horizon_length"]
                if self.config["action_chunking"]
                else 1
            )
            cap_stats = translation_speed_cap_statistics(
                critic_action_pi,
                batch["stage_rewards"],
                env_action_dim=self.config["env_action_dim"],
                repetitions=repetitions,
                base_cap=self.config["ds_stage_speed_base_cap"],
                decay=self.config["ds_stage_speed_decay"],
                reward_floor=self.config["ds_phase_reward_floor"],
                max_progress=self.config["ds_phase_max_progress"],
            )
            speed_cap_penalty = (
                self.config["ds_stage_speed_coefficient"] * cap_stats["penalty"]
            )
            speed_cap_info = {
                "speed_cap": cap_stats["cap"].mean(),
                "translation_norm": cap_stats["translation_norm"].mean(),
                "speed_cap_raw_penalty": cap_stats["penalty"].mean(),
                "speed_cap_violation_fraction": cap_stats[
                    "violation_fraction"
                ].mean(),
                "speed_cap_penalty": speed_cap_penalty.mean(),
            }
            stage_progress = jnp.clip(
                batch["stage_rewards"] - self.config["ds_phase_reward_floor"],
                0.0,
                self.config["ds_phase_max_progress"],
            )

            def masked_mean(values, mask):
                return (values * mask).sum() / jnp.maximum(mask.sum(), 1.0)

            for stage in range(3):
                stage_mask = jnp.isclose(
                    stage_progress, float(stage)
                ).astype(cap_stats["translation_norm"].dtype)
                speed_cap_info[f"translation_norm_stage{stage}"] = masked_mean(
                    cap_stats["translation_norm"], stage_mask
                )
                speed_cap_info[f"speed_cap_violation_stage{stage}"] = masked_mean(
                    cap_stats["violation_fraction"], stage_mask
                )
                speed_cap_info[f"speed_cap_stage{stage}_fraction"] = (
                    stage_mask.mean()
                )
        else:
            speed_cap_penalty = jnp.zeros_like(q)

        actor_update_mode = self.config.get("actor_update_mode", "sac")
        awr_info = {}
        if actor_update_mode in ("replay_awr", "sac_direction_awr"):
            data_qs = self.network.select("critic")(
                batch["observations"], batch_action_pi
            )
            if self.config["q_agg"] == "min":
                data_q = data_qs.min(axis=0)
                policy_q = qs.min(axis=0)
            else:
                data_q = data_qs.mean(axis=0)
                policy_q = qs.mean(axis=0)
            advantage = jax.lax.stop_gradient(data_q - policy_q)
            weights = normalized_advantage_weights(
                advantage,
                temperature=self.config["awr_temperature"],
                max_weight=self.config["awr_max_weight"],
            )
            replay_action_pi = self._batch_action_to_policy(batch_action_pi)
            if actor_update_mode == "replay_awr":
                replay_log_probs = dist.log_prob(replay_action_pi)
                awr_loss = -(weights * replay_log_probs).mean()
                actor_loss = (
                    awr_loss
                    + self.config["awr_entropy_scale"] * regularization.mean()
                    + speed_cap_penalty.mean()
                )
            else:
                replay_components = dist.semantic_component_log_ratios(
                    replay_action_pi
                )
                awr_loss = -(
                    weights * replay_components["direction"]
                ).mean()
                actor_loss = (
                    regularization - q + speed_cap_penalty
                ).mean() + self.config["direction_awr_coefficient"] * awr_loss
            awr_info = {
                "awr_loss": awr_loss,
                "awr_advantage": advantage.mean(),
                "awr_weight_mean": weights.mean(),
                "awr_weight_max": weights.max(),
                "awr_data_q": data_q.mean(),
                "awr_policy_q": policy_q.mean(),
            }
        else:
            actor_loss = (regularization - q + speed_cap_penalty).mean()

        # BC loss.
        if self.config["bc_alpha"] == 0.0:
            bc_loss = jnp.zeros(())
        else:
            bc_action_pi = self._batch_action_to_policy(batch_action_pi)
            bc_loss = -dist.log_prob(bc_action_pi).mean() * self.config["bc_alpha"]

        total_loss = actor_loss + alpha_loss + bc_loss

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "bc_loss": bc_loss,
            "alpha": alpha,
            "entropy": -log_probs.mean(),
            "q": q.mean(),
            **component_info,
            **phase_info,
            **speed_cap_info,
            **awr_info,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        """Sample actions from the actor."""
        dist = self.network.select("actor")(observations)
        action_pi = dist.sample(seed=rng)
        if self.config.get("adapter_representation", "none") == "none":
            action_pi = jnp.clip(action_pi, -1, 1)
        return action_pi

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        validate_auxiliary_control(config)
        if config.get("adapter_execution_map") in AUXILIARY_EXECUTION_MAPS and ex_actions.shape[-1] != 5:
            raise ValueError("Cartesian auxiliary control requires five-dimensional replay actions")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        observed_action_dim = ex_actions.shape[-1]
        action_dim = config.get("policy_action_dim", observed_action_dim)
        critic_action_dim = config.get("critic_action_dim", observed_action_dim)
        config["action_dim"] = action_dim
        config["critic_action_dim"] = critic_action_dim
        if config["action_chunking"]:
            full_policy_dim = action_dim * config["horizon_length"]
            full_critic_actions = jnp.zeros(
                ex_actions.shape[:-1]
                + (critic_action_dim * config["horizon_length"],),
                dtype=ex_actions.dtype,
            )
        else:
            full_policy_dim = action_dim
            full_critic_actions = jnp.zeros(
                ex_actions.shape[:-1] + (critic_action_dim,), dtype=ex_actions.dtype
            )

        if config["target_entropy"] is None:
            config["target_entropy"] = (
                -config["target_entropy_multiplier"] * full_policy_dim
            )

        separate_alpha = config.get("ds_separate_alpha", False)
        if separate_alpha:
            env_action_dim = config["env_action_dim"]
            if env_action_dim == 5:
                direction_groups, scalar_dims = 1, 2
            elif env_action_dim == 7:
                direction_groups, scalar_dims = 2, 1
            else:
                raise ValueError(
                    "Separate DS temperatures require a 5D or 7D action layout."
                )
            repetitions = config["horizon_length"] if config["action_chunking"] else 1
            component_dims = {
                "direction": 2 * direction_groups * repetitions,
                "speed": direction_groups * repetitions,
                "scalar": scalar_dims * repetitions,
            }
            for name, component_dim in component_dims.items():
                key = f"target_entropy_{name}"
                if config.get(key) is None:
                    config[key] = -config["target_entropy_multiplier"] * component_dim

        # Define networks
        critic_base_cls = partial(
            MLP,
            hidden_dims=config["value_hidden_dims"],
            activate_final=True,
            use_layer_norm=config["layer_norm"],
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config["num_qs"])

        actor_base_cls = partial(
            MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True
        )
        action_pipeline = assemble_action_pipeline(
            base_cls=actor_base_cls,
            one_step_policy_dim=action_dim,
            full_policy_dim=full_policy_dim,
            config=config,
        )
        actor_def = action_pipeline.actor

        network_info = dict(
            critic=(critic_def, (ex_observations, full_critic_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_critic_actions)),
            actor=(actor_def, (ex_observations,)),
        )
        if separate_alpha:
            for name in ("direction", "speed", "scalar"):
                network_info[f"alpha_{name}"] = (
                    Temperature(config["init_temp"]),
                    (),
                )
        else:
            network_info["alpha"] = (Temperature(config["init_temp"]), ())
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="acrlpd",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            num_qs=10,
            target_entropy=ml_collections.config_dict.placeholder(
                float
            ),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            alpha=1.0,
            fixed_alpha=-1.0,
            bc_alpha=0.0,
            q_agg="mean",  # Aggregation function for target Q values.
            horizon_length=ml_collections.config_dict.placeholder(int),  # will be set
            action_chunking=True,
            # Spherical-coordinate integrated direction-speed distribution in raw action space.
            use_integrated_ds=False,  # Use spherical-coordinate integrated direction-speed distribution.
            ds_separate_actor_trunks=False,
            use_cartesian_l2_projection=False,
            ds_max_speed=1.0,
            ds_integrated_type="spherical",  # spherical, stereographic, tangent, or projected_tangent.
            ds_speed_bound="cube",  # "sphere" has unit radius; "cube" reaches the cube boundary.
            ds_tangent_angular_log_std_offset=-1.0,
            ds_tangent_angular_log_std_max=0.0,
            ds_tangent_speed_log_std_offset=0.0,
            ds_tangent_speed_log_std_max=2.0,
            ds_tangent_speed_mean_offset=0.0,
            ds_moment_controlled=False,
            ds_moment_angular_log_std_init=-1.7,
            ds_moment_speed_log_std_init=-2.1,
            ds_moment_speed_mean_init=0.4,
            ds_moment_speed_residual_scale=0.0,
            ds_moment_control_speed_mean=True,
            ds_moment_control_interval=5000,
            ds_moment_panel_size=512,
            ds_moment_samples_per_state=32,
            ds_moment_target_radius_mean=0.989,
            ds_moment_target_radius_std=0.022,
            ds_moment_target_angle=0.161,
            ds_moment_angular_gain=0.35,
            ds_moment_radial_std_gain=0.35,
            ds_moment_radius_mean_gain=4.0,
            adapter_representation="none",
            adapter_execution_map="direction_speed",
            adapter_speed_squash="tanh",
            adapter_speed_interpretation="log",
            adapter_critic_space="latent",
            ds_regularizer="auto",
            ds_separate_alpha=False,
            ds_phase_regularization="none",
            ds_phase_reward_floor=-3.0,
            ds_phase_decay=0.5,
            ds_phase_max_progress=2.0,
            ds_stage_speed_constraint="none",
            ds_stage_speed_base_cap=1.0,
            ds_stage_speed_decay=0.5,
            ds_stage_speed_coefficient=100.0,
            actor_update_mode="sac",
            awr_temperature=1.0,
            awr_max_weight=20.0,
            awr_entropy_scale=1.0,
            direction_awr_coefficient=1.0,
            target_entropy_direction=ml_collections.config_dict.placeholder(float),
            target_entropy_speed=ml_collections.config_dict.placeholder(float),
            target_entropy_scalar=ml_collections.config_dict.placeholder(float),
            init_temp=1.0,
        )
    )
    return config
