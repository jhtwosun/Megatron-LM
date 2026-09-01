# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure identity and process-group contracts for MDP Dynamic-CP.

This module does not create process groups or read global parallel state. The
decoder path injects the native Dynamic-CP group getter; the encoder bootstrap
materializes :class:`DynamicCpGroupSpec` records through the MDP group registry.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from megatron.core.mdp.errors import MdpConfigurationError


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_nonnegative_integer(name: str, value: Any) -> None:
    if not _is_integer(value) or value < 0:
        raise MdpConfigurationError(
            f"MDP: {name}={value!r} violates: {name} is a non-negative integer."
        )


@dataclass(frozen=True, order=True)
class GlobalSampleId:
    """Stable sample identity across decoder DP lanes.

    ``local_sample_order`` is the source lane's capture order, not a dataset
    implementation's optional sample identifier.
    """

    source_dp_lane: int
    local_sample_order: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer("source_dp_lane", self.source_dp_lane)
        _require_nonnegative_integer("local_sample_order", self.local_sample_order)

    def to_wire_tuple(self) -> tuple[int, int]:
        """Serialize as the fixed-width ``(source lane, local order)`` tuple."""
        return (self.source_dp_lane, self.local_sample_order)

    @classmethod
    def from_wire_tuple(cls, value: tuple[int, int]) -> "GlobalSampleId":
        """Deserialize a validated fixed-width identity tuple."""
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not all(_is_integer(component) for component in value)
        ):
            raise MdpConfigurationError(
                f"MDP: global sample wire value {value!r} violates: a 2-int tuple."
            )
        return cls(*value)


@dataclass(frozen=True, order=True)
class GlobalVisionItemId:
    """Stable vision-item identity across decoder DP lanes."""

    source_dp_lane: int
    local_item_id: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer("source_dp_lane", self.source_dp_lane)
        _require_nonnegative_integer("local_item_id", self.local_item_id)

    def to_wire_tuple(self) -> tuple[int, int]:
        """Serialize as the fixed-width ``(source lane, local item id)`` tuple."""
        return (self.source_dp_lane, self.local_item_id)

    @classmethod
    def from_wire_tuple(cls, value: tuple[int, int]) -> "GlobalVisionItemId":
        """Deserialize a validated fixed-width identity tuple."""
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not all(_is_integer(component) for component in value)
        ):
            raise MdpConfigurationError(
                f"MDP: global vision-item wire value {value!r} violates: a 2-int tuple."
            )
        return cls(*value)


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return _is_integer(value) and value > 0 and value & (value - 1) == 0


def dynamic_cp_group_sizes(minimum_size: int, maximum_size: int) -> tuple[int, ...]:
    """Return the inclusive power-of-two schedule from minimum to maximum."""
    if not is_power_of_two(minimum_size):
        raise MdpConfigurationError(
            f"MDP: minimum_size={minimum_size} violates: minimum_size is a power of two."
        )
    if not is_power_of_two(maximum_size):
        raise MdpConfigurationError(
            f"MDP: maximum_size={maximum_size} violates: maximum_size is a power of two."
        )
    if minimum_size > maximum_size:
        raise MdpConfigurationError(
            f"MDP: minimum_size={minimum_size} violates: minimum_size <= "
            f"maximum_size ({maximum_size})."
        )

    sizes = []
    size = minimum_size
    while size <= maximum_size:
        sizes.append(size)
        size *= 2
    return tuple(sizes)


@dataclass(frozen=True)
class DynamicCpGroupSpec:
    """One deterministic subgroup specification inside a maximum-size pool."""

    group_size: int
    group_index: int
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not is_power_of_two(self.group_size):
            raise MdpConfigurationError(
                f"MDP: group_size={self.group_size!r} violates: group_size is a "
                "positive power-of-two integer."
            )
        _require_nonnegative_integer("group_index", self.group_index)
        if not isinstance(self.ranks, tuple):
            raise MdpConfigurationError(
                f"MDP: ranks={self.ranks!r} violates: ranks is an immutable tuple."
            )
        for rank in self.ranks:
            _require_nonnegative_integer("rank", rank)
        if self.group_size != len(self.ranks):
            raise MdpConfigurationError(
                f"MDP: group_size={self.group_size} violates: group_size == "
                f"len(ranks) ({len(self.ranks)})."
            )
        if len(set(self.ranks)) != len(self.ranks):
            raise MdpConfigurationError(
                f"MDP: ranks={self.ranks} violates: subgroup ranks are unique."
            )


@dataclass(frozen=True)
class DynamicCpGroupMembership:
    """One rank's handle and ordered global ranks for a Dynamic-CP size."""

    group_size: int
    ranks: tuple[int, ...]
    group: Any = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not is_power_of_two(self.group_size):
            raise MdpConfigurationError(
                f"MDP: group_size={self.group_size!r} violates: group_size is a "
                "positive power-of-two integer."
            )
        if not isinstance(self.ranks, tuple):
            raise MdpConfigurationError(
                f"MDP: ranks={self.ranks!r} violates: ranks is an immutable tuple."
            )
        for rank in self.ranks:
            _require_nonnegative_integer("rank", rank)
        if self.group_size != len(self.ranks):
            raise MdpConfigurationError(
                f"MDP: group_size={self.group_size} violates: group_size == "
                f"len(ranks) ({len(self.ranks)})."
            )
        if len(set(self.ranks)) != len(self.ranks):
            raise MdpConfigurationError(
                f"MDP: ranks={self.ranks} violates: subgroup ranks are unique."
            )
        if self.group is None:
            raise MdpConfigurationError(
                "MDP: group=None violates: a Dynamic-CP membership has a process-group handle."
            )


def nested_dynamic_cp_group_specs(
    pool_ranks: Sequence[int], *, minimum_size: int
) -> tuple[DynamicCpGroupSpec, ...]:
    """Partition ``pool_ranks`` into deterministic nested power-of-two groups.

    Rank order is authoritative and may be physically non-contiguous. Groups
    are emitted by ascending size, then by their position in the pool.
    """
    ranks = tuple(pool_ranks)
    for rank in ranks:
        _require_nonnegative_integer("rank", rank)
    if len(set(ranks)) != len(ranks):
        raise MdpConfigurationError(
            f"MDP: pool_ranks={ranks} violates: maximum encoder pool ranks are unique."
        )
    sizes = dynamic_cp_group_sizes(minimum_size, len(ranks))
    specs = []
    for group_size in sizes:
        for group_index, start in enumerate(range(0, len(ranks), group_size)):
            specs.append(
                DynamicCpGroupSpec(
                    group_size=group_size,
                    group_index=group_index,
                    ranks=ranks[start : start + group_size],
                )
            )
    return tuple(specs)


def member_dynamic_cp_group_specs(
    specs: Sequence[DynamicCpGroupSpec], global_rank: int
) -> tuple[DynamicCpGroupSpec, ...]:
    """Return the single nested subgroup containing ``global_rank`` at each size."""
    _require_nonnegative_integer("global_rank", global_rank)
    memberships = tuple(spec for spec in specs if global_rank in spec.ranks)
    sizes = tuple(spec.group_size for spec in memberships)
    if len(sizes) != len(set(sizes)):
        raise MdpConfigurationError(
            f"MDP: rank {global_rank} violates: at most one dynamic subgroup per size."
        )
    if not memberships:
        raise MdpConfigurationError(
            f"MDP: rank {global_rank} violates: rank belongs to the maximum encoder pool."
        )
    return memberships


def select_dynamic_cp_group(
    memberships: Sequence[DynamicCpGroupMembership], group_size: int
) -> DynamicCpGroupMembership:
    """Resolve one rank-local Dynamic-CP membership by its exact group size."""
    if not is_power_of_two(group_size):
        raise MdpConfigurationError(
            f"MDP: group_size={group_size!r} violates: group_size is a "
            "positive power-of-two integer."
        )
    matches = tuple(membership for membership in memberships if membership.group_size == group_size)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise MdpConfigurationError(
            f"MDP: group_size={group_size} violates: one rank-local group per size."
        )
    available = tuple(membership.group_size for membership in memberships)
    raise MdpConfigurationError(
        f"MDP: group_size={group_size} violates: group_size is in available sizes {available}."
    )


def lookup_decoder_dynamic_cp_group(
    group_size: int, *, minimum_size: int, maximum_size: int, group_getter: Callable[..., Any]
) -> Any:
    """Resolve a prebuilt native decoder Dynamic-CP group through an injected getter.

    The caller supplies the native ``DP x CP`` group lookup. Keeping that read
    outside MDP core avoids another dependency on global parallel state.
    """
    scheduled_sizes = dynamic_cp_group_sizes(minimum_size, maximum_size)
    if maximum_size == 1:
        raise MdpConfigurationError(
            "MDP: decoder maximum_size=1 violates: decoder maximum_size > 1."
        )
    if group_size not in scheduled_sizes:
        raise MdpConfigurationError(
            f"MDP: decoder group_size={group_size} violates: group_size is a "
            f"scheduled group size in {scheduled_sizes}."
        )
    try:
        group = group_getter(group_size=group_size)
    except LookupError as error:
        raise MdpConfigurationError(
            f"MDP: decoder group_size={group_size} violates: native Dynamic-CP "
            "group is available from the injected lookup."
        ) from error
    if group is None:
        raise MdpConfigurationError(
            f"MDP: decoder group_size={group_size} violates: native Dynamic-CP "
            "group was created during model-parallel initialization."
        )
    return group
