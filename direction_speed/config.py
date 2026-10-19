"""Validate direction-speed settings and apply them to an agent config.

Shared by ``main.py``, ``main_online.py``, and ``eval_video.py`` to avoid
duplicating the same validation + adapter-construction logic.
"""

from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from typing import Any

from direction_speed.adapter import DirectionSpeedAdapter
from direction_speed.distributions.factory import (
    ActionPipelineRecipe,
    resolve_pipeline_recipe,
)
from direction_speed.specs import (
    ActionReference,
    DirectionSpeedMode,
    IntegratedDirectionType,
    RadialBound,
)


@dataclass(frozen=True)
class ResolvedDirectionSpeed:
    """Validated runtime objects derived from CLI and agent configuration."""

    mode: DirectionSpeedMode
    speed_bound: RadialBound
    adapter: DirectionSpeedAdapter
    recipe: ActionPipelineRecipe
    adapter_critic_space: str
    action_reference: ActionReference
    separate_alpha: bool

    @property
    def postprocessor(self):
        return self.recipe.postprocessor

    @property
    def components(self):
        return self.recipe.components

    @property
    def use_adapter(self) -> bool:
        return self.mode is DirectionSpeedMode.ADAPTER

    @property
    def use_integrated(self) -> bool:
        return self.mode is DirectionSpeedMode.INTEGRATED

    @property
    def use_projection(self) -> bool:
        return self.mode is DirectionSpeedMode.PROJECTION

    @property
    def use_env_action_critic(self) -> bool:
        return self.use_adapter and self.adapter_critic_space == "env"


def configure_direction_speed(
    *,
    ds_mode: str = "none",
    ds_speed_bound: str = "config",
    ds_integrated_type: str = "spherical",
    ds_adapter_representation: str = "cartesian",
    ds_adapter_speed_squash: str = "sigmoid",
    ds_adapter_speed_interpretation: str = "direct",
    ds_adapter_critic_space: str = "config",
    ds_adapter_execution_map: str = "config",
    ds_regularizer: str = "config",
    ds_separate_alpha: bool = False,
    config: MutableMapping[str, Any],
    agent_name: str = "",
    ds_speed_bound_default: str = "cube",
) -> ResolvedDirectionSpeed:
    """Mutate ``config`` with validated DS settings and return runtime objects.

    Parameters
    ----------
    config : dict
        Mutable agent config dict.  The following keys are written in-place:
        ``ds_speed_bound``, ``use_integrated_ds``, ``ds_integrated_type``,
        ``use_cartesian_l2_projection``,
        ``adapter_representation``, ``adapter_speed_squash``,
        ``adapter_speed_interpretation``, ``adapter_critic_space``,
        ``ds_regularizer``, ``ds_separate_alpha``.
    ds_speed_bound_default : str
        Fallback used when *ds_speed_bound* is ``"config"`` and the config
        dict does not contain ``ds_speed_bound``.  Training scripts pass
        ``"cube"``; ``eval_video.py`` passes ``"sphere"`` for backward
        compatibility with older experiments that defaulted to sphere.
    """
    mode = DirectionSpeedMode(ds_mode)
    integrated_type = IntegratedDirectionType(ds_integrated_type)

    # --- resolve speed bound ---
    bound_value = (
        config.get("ds_speed_bound", ds_speed_bound_default)
        if ds_speed_bound == "config"
        else ds_speed_bound
    )
    speed_bound = RadialBound(bound_value)
    config["ds_speed_bound"] = speed_bound.value

    # --- integrated DS ---
    if (
        config.get("use_integrated_ds", False)
        and mode is not DirectionSpeedMode.INTEGRATED
    ):
        raise ValueError("Integrated DS agent settings require --ds_mode=integrated.")
    if mode is DirectionSpeedMode.INTEGRATED:
        config["use_integrated_ds"] = True
        config["ds_integrated_type"] = integrated_type.value
    else:
        config["use_integrated_ds"] = False

    if mode is DirectionSpeedMode.INTEGRATED and agent_name != "acrlpd":
        raise ValueError("--ds_mode=integrated is only implemented for ACRLPD.")

    config["use_cartesian_l2_projection"] = (
        mode is DirectionSpeedMode.PROJECTION
    )
    if mode is DirectionSpeedMode.PROJECTION and agent_name != "acrlpd":
        raise ValueError("--ds_mode=projection is only implemented for ACRLPD.")

    # --- adapter DS ---
    execution_map = (config.get("adapter_execution_map", "direction_speed")
                     if ds_adapter_execution_map == "config" else ds_adapter_execution_map)
    if execution_map == "cartesian_aux" and (mode is not DirectionSpeedMode.ADAPTER or agent_name != "acrlpd"):
        raise ValueError("Cartesian auxiliary control requires ACRLPD adapter mode")
    config["adapter_execution_map"] = execution_map
    adapter = DirectionSpeedAdapter(
        representation=ds_adapter_representation,
        speed_squash=ds_adapter_speed_squash,
        speed_interpretation=ds_adapter_speed_interpretation,
        speed_bound=speed_bound,
        execution_map=execution_map,
    )
    critic_space = "latent"
    if mode is DirectionSpeedMode.ADAPTER:
        if agent_name != "acrlpd" and (
            ds_adapter_representation != "cartesian"
            or ds_adapter_speed_squash != "tanh"
            or ds_adapter_speed_interpretation != "log"
        ):
            raise ValueError(
                "Non-ACRLPD adapters only support cartesian + tanh + log speed."
            )
        config["adapter_representation"] = ds_adapter_representation
        config["adapter_speed_squash"] = ds_adapter_speed_squash
        config["adapter_speed_interpretation"] = ds_adapter_speed_interpretation
        critic_space = (
            config.get("adapter_critic_space", "latent")
            if ds_adapter_critic_space == "config"
            else ds_adapter_critic_space
        )
        if critic_space not in ("latent", "env", "canonical"):
            raise ValueError("adapter critic space must be 'latent', 'env', or 'canonical'.")
        if critic_space == "canonical" and agent_name != "acrlpd":
            raise ValueError("Canonical adapter critic is only implemented for ACRLPD.")
        config["adapter_critic_space"] = critic_space
        if execution_map == "cartesian_aux" and critic_space != "env":
            raise ValueError("Cartesian auxiliary control requires the env critic")
    else:
        config["adapter_representation"] = "none"
        config["adapter_critic_space"] = "latent"

    configured_reference = (
        config.get("ds_regularizer", "auto")
        if ds_regularizer == "config"
        else ds_regularizer
    )
    if configured_reference == "auto":
        configured_reference = (
            "action_volume"
            if mode is DirectionSpeedMode.INTEGRATED
            else "policy_coordinate"
        )
    action_reference = ActionReference(configured_reference)
    if mode is DirectionSpeedMode.INTEGRATED:
        if action_reference is ActionReference.POLICY_COORDINATE:
            raise ValueError(
                "Integrated DS requires action_volume or semantic_product regularization."
            )
    elif action_reference is not ActionReference.POLICY_COORDINATE:
        raise ValueError(
            "action_volume and semantic_product regularizers require --ds_mode=integrated."
        )
    if ds_separate_alpha and not (
        mode is DirectionSpeedMode.INTEGRATED
        and action_reference is ActionReference.SEMANTIC_PRODUCT
    ):
        raise ValueError(
            "Separate component temperatures require integrated semantic_product DS."
        )
    config["ds_regularizer"] = action_reference.value
    config["ds_separate_alpha"] = bool(ds_separate_alpha)

    recipe = resolve_pipeline_recipe(config, adapter=adapter)
    config["action_pipeline_components"] = asdict(recipe.components)
    return ResolvedDirectionSpeed(
        mode=mode,
        speed_bound=speed_bound,
        adapter=adapter,
        recipe=recipe,
        adapter_critic_space=critic_space,
        action_reference=action_reference,
        separate_alpha=bool(ds_separate_alpha),
    )
