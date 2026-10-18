"""Per-method recipes that select and assemble complete action pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from direction_speed.adapter import DirectionSpeedAdapter
from direction_speed.distributions.integrated import IntegratedDirectionSpeedNormal
from direction_speed.distributions.mixed_squash import MixedSquashNormal
from direction_speed.distributions.tangent import TangentDirectionSpeedNormal
from direction_speed.postprocessors import (
    ActionPostprocessor,
    IdentityActionPostprocessor,
    L2BallProjectionPostprocessor,
)
from direction_speed.specs import (
    DirectionRepresentation,
    IntegratedDirectionType,
    build_action_groups,
)
from rlpd_distributions import TanhNormal


@dataclass(frozen=True)
class PipelineComponents:
    parameter_heads: str
    raw_sampler: str
    policy_bijector: str
    postprocessor: str


@dataclass(frozen=True)
class ActionPipelineAssembly:
    """Fully assembled training and execution components for one method."""

    actor: object
    postprocessor: ActionPostprocessor
    components: PipelineComponents


class ActionPipelineRecipe(ABC):
    """A method owns component selection, parameterization, and assembly."""

    @property
    @abstractmethod
    def components(self) -> PipelineComponents: ...

    @property
    @abstractmethod
    def postprocessor(self) -> ActionPostprocessor: ...

    @abstractmethod
    def build_actor(
        self,
        *,
        base_cls,
        one_step_policy_dim: int,
        full_policy_dim: int,
        horizon_length: int,
        action_chunking: bool,
    ): ...

    def assemble(
        self,
        *,
        base_cls,
        one_step_policy_dim: int,
        full_policy_dim: int,
        horizon_length: int,
        action_chunking: bool,
    ) -> ActionPipelineAssembly:
        actor = self.build_actor(
            base_cls=base_cls,
            one_step_policy_dim=one_step_policy_dim,
            full_policy_dim=full_policy_dim,
            horizon_length=horizon_length,
            action_chunking=action_chunking,
        )
        return ActionPipelineAssembly(actor, self.postprocessor, self.components)


@dataclass(frozen=True)
class BaselinePipelineRecipe(ActionPipelineRecipe):
    @property
    def components(self):
        return PipelineComponents(
            "DistributionParameterHeads",
            "MultivariateNormalDiag",
            "Tanh",
            "IdentityActionPostprocessor",
        )

    @property
    def postprocessor(self):
        return IdentityActionPostprocessor()

    def build_actor(
        self,
        *,
        base_cls,
        one_step_policy_dim,
        full_policy_dim,
        horizon_length,
        action_chunking,
    ):
        del one_step_policy_dim, horizon_length, action_chunking
        return TanhNormal(base_cls, full_policy_dim)


@dataclass(frozen=True)
class ProjectedCartesianPipelineRecipe(BaselinePipelineRecipe):
    """Ordinary Cartesian actor with an execution-time XYZ L2 projection."""

    @property
    def components(self):
        return PipelineComponents(
            "DistributionParameterHeads",
            "MultivariateNormalDiag",
            "Tanh",
            "L2BallProjectionPostprocessor[xyz,max_norm=1]",
        )

    @property
    def postprocessor(self):
        return L2BallProjectionPostprocessor()


@dataclass(frozen=True)
class AdapterPipelineRecipe(ActionPipelineRecipe):
    adapter: DirectionSpeedAdapter

    @property
    def components(self):
        return PipelineComponents(
            "DistributionParameterHeads",
            "MultivariateNormalDiag",
            f"MixedSquashBijector[speed={self.adapter.speed_squash.value}]",
            "DirectionSpeedAdapter["
            f"{self.adapter.representation.value},"
            f"speed={self.adapter.speed_interpretation.value}]",
        )

    @property
    def postprocessor(self):
        return self.adapter

    def build_actor(
        self,
        *,
        base_cls,
        one_step_policy_dim,
        full_policy_dim,
        horizon_length,
        action_chunking,
    ):
        repetitions = horizon_length if action_chunking else 1
        squash_types = self.adapter.squash_types(one_step_policy_dim, repetitions)
        return MixedSquashNormal(base_cls, full_policy_dim, squash_types)


class CartesianAdapterPipelineRecipe(AdapterPipelineRecipe):
    pass


class SphericalAdapterPipelineRecipe(AdapterPipelineRecipe):
    pass


class StereographicAdapterPipelineRecipe(AdapterPipelineRecipe):
    pass


@dataclass(frozen=True)
class IntegratedPipelineRecipe(ActionPipelineRecipe, ABC):
    max_speed: float
    speed_bound: str
    separate_component_trunks: bool = False

    @property
    @abstractmethod
    def direction_type(self) -> IntegratedDirectionType: ...

    @property
    @abstractmethod
    def actor_cls(self): ...

    @property
    @abstractmethod
    def bijector_name(self) -> str: ...

    @property
    def actor_kwargs(self):
        return {"separate_component_trunks": self.separate_component_trunks}

    @property
    def components(self):
        return PipelineComponents(
            "DistributionParameterHeads",
            "MultivariateNormalDiag",
            self.bijector_name,
            "IdentityActionPostprocessor",
        )

    @property
    def postprocessor(self):
        return IdentityActionPostprocessor()

    def build_actor(
        self,
        *,
        base_cls,
        one_step_policy_dim,
        full_policy_dim,
        horizon_length,
        action_chunking,
    ):
        action_groups = build_action_groups(
            one_step_policy_dim,
            horizon_length,
            action_chunking,
            self.max_speed,
            self.direction_type,
            self.speed_bound,
        )
        return self.actor_cls(
            base_cls,
            full_policy_dim,
            action_groups,
            **self.actor_kwargs,
        )


@dataclass(frozen=True)
class SphericalPipelineRecipe(IntegratedPipelineRecipe):
    @property
    def direction_type(self):
        return IntegratedDirectionType.SPHERICAL

    @property
    def actor_cls(self):
        return IntegratedDirectionSpeedNormal

    @property
    def bijector_name(self):
        return "IntegratedDirectionSpeedBijector[spherical]"


@dataclass(frozen=True)
class StereographicPipelineRecipe(IntegratedPipelineRecipe):
    @property
    def direction_type(self):
        return IntegratedDirectionType.STEREOGRAPHIC

    @property
    def actor_cls(self):
        return IntegratedDirectionSpeedNormal

    @property
    def bijector_name(self):
        return "IntegratedDirectionSpeedBijector[stereographic]"


@dataclass(frozen=True)
class TangentPipelineRecipe(IntegratedPipelineRecipe):
    angular_log_std_offset: float = -1.0
    angular_log_std_max: float = 0.0
    speed_log_std_offset: float = 0.0
    speed_log_std_max: float = 2.0
    speed_mean_offset: float = 0.0
    moment_controlled: bool = False
    moment_angular_log_std_init: float = -1.7
    moment_speed_log_std_init: float = -2.1
    moment_speed_mean_init: float = 0.4
    moment_speed_residual_scale: float = 0.0
    moment_control_speed_mean: bool = True

    @property
    def direction_type(self):
        return IntegratedDirectionType.TANGENT

    @property
    def actor_cls(self):
        return TangentDirectionSpeedNormal

    @property
    def actor_kwargs(self):
        return {
            "angular_log_std_offset": self.angular_log_std_offset,
            "angular_log_std_max": self.angular_log_std_max,
            "speed_log_std_offset": self.speed_log_std_offset,
            "speed_log_std_max": self.speed_log_std_max,
            "speed_mean_offset": self.speed_mean_offset,
            "moment_controlled": self.moment_controlled,
            "moment_angular_log_std_init": self.moment_angular_log_std_init,
            "moment_speed_log_std_init": self.moment_speed_log_std_init,
            "moment_speed_mean_init": self.moment_speed_mean_init,
            "moment_speed_residual_scale": self.moment_speed_residual_scale,
            "moment_control_speed_mean": self.moment_control_speed_mean,
        }

    @property
    def bijector_name(self):
        return (
            "TangentPolicyBijector[center=unit3,std=diag2,"
            f"offset={self.angular_log_std_offset:g},"
            f"max={self.angular_log_std_max:g},"
            f"speed_offset={self.speed_log_std_offset:g},"
            f"speed_max={self.speed_log_std_max:g},"
            f"speed_mean_offset={self.speed_mean_offset:g}]"
        )


@dataclass(frozen=True)
class ProjectedTangentPipelineRecipe(TangentPipelineRecipe):
    @property
    def actor_kwargs(self):
        values = super().actor_kwargs
        return {**values, "projected_tangent": True}

    @property
    def bijector_name(self):
        return (
            "ProjectedTangentPolicyBijector[center=unit3,std=diag2,"
            f"offset={self.angular_log_std_offset:g},"
            f"max={self.angular_log_std_max:g},"
            f"speed_offset={self.speed_log_std_offset:g},"
            f"speed_max={self.speed_log_std_max:g},"
            f"speed_mean_offset={self.speed_mean_offset:g}]"
        )


ADAPTER_RECIPE_REGISTRY = {
    DirectionRepresentation.CARTESIAN: CartesianAdapterPipelineRecipe,
    DirectionRepresentation.SPHERICAL: SphericalAdapterPipelineRecipe,
    DirectionRepresentation.STEREOGRAPHIC: StereographicAdapterPipelineRecipe,
}

INTEGRATED_RECIPE_REGISTRY = {
    IntegratedDirectionType.SPHERICAL: SphericalPipelineRecipe,
    IntegratedDirectionType.STEREOGRAPHIC: StereographicPipelineRecipe,
    IntegratedDirectionType.TANGENT: TangentPipelineRecipe,
    IntegratedDirectionType.PROJECTED_TANGENT: ProjectedTangentPipelineRecipe,
}


def resolve_pipeline_recipe(config, *, adapter=None) -> ActionPipelineRecipe:
    """Look up a method recipe; the recipe handles all later assembly."""
    representation = config.get("adapter_representation", "none")
    if representation != "none":
        representation = DirectionRepresentation(representation)
        adapter = adapter or DirectionSpeedAdapter(
            representation=representation,
            speed_squash=config["adapter_speed_squash"],
            speed_interpretation=config["adapter_speed_interpretation"],
            speed_bound=config.get("ds_speed_bound", "cube"),
        )
        return ADAPTER_RECIPE_REGISTRY[representation](adapter)

    if config.get("use_integrated_ds", False):
        direction_type = IntegratedDirectionType(config["ds_integrated_type"])
        recipe_cls = INTEGRATED_RECIPE_REGISTRY[direction_type]
        recipe_kwargs = dict(
            max_speed=config["ds_max_speed"],
            speed_bound=config.get("ds_speed_bound", "cube"),
            separate_component_trunks=config.get(
                "ds_separate_actor_trunks", False
            ),
        )
        if direction_type in (
            IntegratedDirectionType.TANGENT,
            IntegratedDirectionType.PROJECTED_TANGENT,
        ):
            if recipe_kwargs.pop("separate_component_trunks"):
                raise ValueError(
                    "Separate component trunks are not implemented for tangent DS."
                )
            recipe_kwargs.update(
                angular_log_std_offset=config.get(
                    "ds_tangent_angular_log_std_offset", -1.0
                ),
                angular_log_std_max=config.get("ds_tangent_angular_log_std_max", 0.0),
                speed_log_std_offset=config.get(
                    "ds_tangent_speed_log_std_offset", 0.0
                ),
                speed_log_std_max=config.get("ds_tangent_speed_log_std_max", 2.0),
                speed_mean_offset=config.get("ds_tangent_speed_mean_offset", 0.0),
                moment_controlled=config.get("ds_moment_controlled", False),
                moment_angular_log_std_init=config.get(
                    "ds_moment_angular_log_std_init", -1.7
                ),
                moment_speed_log_std_init=config.get(
                    "ds_moment_speed_log_std_init", -2.1
                ),
                moment_speed_mean_init=config.get(
                    "ds_moment_speed_mean_init", 0.4
                ),
                moment_speed_residual_scale=config.get(
                    "ds_moment_speed_residual_scale", 0.0
                ),
                moment_control_speed_mean=config.get(
                    "ds_moment_control_speed_mean", True
                ),
            )
        return recipe_cls(**recipe_kwargs)

    if config.get("use_cartesian_l2_projection", False):
        return ProjectedCartesianPipelineRecipe()

    return BaselinePipelineRecipe()


def assemble_action_pipeline(
    *,
    base_cls,
    one_step_policy_dim: int,
    full_policy_dim: int,
    config,
) -> ActionPipelineAssembly:
    recipe = resolve_pipeline_recipe(config)
    return recipe.assemble(
        base_cls=base_cls,
        one_step_policy_dim=one_step_policy_dim,
        full_policy_dim=full_policy_dim,
        horizon_length=config["horizon_length"],
        action_chunking=config["action_chunking"],
    )
