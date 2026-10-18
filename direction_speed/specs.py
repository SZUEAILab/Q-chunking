"""Typed configuration and action layouts for direction-speed policies."""

from dataclasses import dataclass
from enum import StrEnum


class DirectionSpeedMode(StrEnum):
    NONE = "none"
    PROJECTION = "projection"
    ADAPTER = "adapter"
    INTEGRATED = "integrated"


class DirectionRepresentation(StrEnum):
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    STEREOGRAPHIC = "stereographic"


class IntegratedDirectionType(StrEnum):
    SPHERICAL = "spherical"
    STEREOGRAPHIC = "stereographic"
    TANGENT = "tangent"
    PROJECTED_TANGENT = "projected_tangent"


class SpeedSquash(StrEnum):
    TANH = "tanh"
    SIGMOID = "sigmoid"


class SpeedInterpretation(StrEnum):
    LOG = "log"
    DIRECT = "direct"


class RadialBound(StrEnum):
    SPHERE = "sphere"
    CUBE = "cube"


class ActionReference(StrEnum):
    POLICY_COORDINATE = "policy_coordinate"
    ACTION_VOLUME = "action_volume"
    SEMANTIC_PRODUCT = "semantic_product"


class ActionGroupKind(StrEnum):
    SPHERICAL = "spherical"
    STEREOGRAPHIC = "stereographic"
    TANGENT = "tangent"
    SCALAR = "scalar"


@dataclass(frozen=True)
class ActionGroup:
    kind: ActionGroupKind
    size: int
    max_speed: float = 1.0
    speed_bound: RadialBound = RadialBound.CUBE

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError("ActionGroup size must be positive")
        if self.kind is not ActionGroupKind.SCALAR and self.size != 3:
            raise ValueError(f"{self.kind.value} direction groups must have size 3")


def build_action_groups(
    action_dim: int,
    horizon_length: int,
    action_chunking: bool,
    max_speed: float,
    direction_type: str | IntegratedDirectionType,
    speed_bound: str | RadialBound,
) -> tuple[ActionGroup, ...]:
    """Build immutable per-step groups, then repeat them for action chunks."""
    if horizon_length < 1:
        raise ValueError("horizon_length must be at least 1")
    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive")
    direction_type = IntegratedDirectionType(direction_type)
    speed_bound = RadialBound(speed_bound)
    direction_kind = ActionGroupKind(direction_type.value)

    if action_dim == 5:
        base_groups = (
            ActionGroup(direction_kind, 3, max_speed, speed_bound),
            ActionGroup(ActionGroupKind.SCALAR, 1),
            ActionGroup(ActionGroupKind.SCALAR, 1),
        )
    elif action_dim == 7:
        base_groups = (
            ActionGroup(direction_kind, 3, max_speed, speed_bound),
            ActionGroup(direction_kind, 3, max_speed, speed_bound),
            ActionGroup(ActionGroupKind.SCALAR, 1),
        )
    else:
        raise ValueError(f"No direction-speed layout for action_dim={action_dim}")

    horizon = horizon_length if action_chunking else 1
    return base_groups * horizon
