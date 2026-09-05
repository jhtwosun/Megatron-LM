# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private native-group authority for repeated four-rank D4 domains."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.mdp.dynamic_cp_d4_collective_order import _make_repeated_d4_collective_runner
from megatron.core.mdp.dynamic_cp_d4_domain_status import _make_repeated_d4_domain_status_collector
from megatron.core.mdp.dynamic_cp_d4_iteration_nonce import acquire_repeated_d4_world_attempt_nonce
from megatron.core.mdp.dynamic_cp_d4_status import (
    _make_repeated_d4_world_pre_gate,
    _validate_world_ranks,
)
from megatron.core.mdp.dynamic_cp_execution import _validate_precollective_timeout
from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

__all__ = ()

_DOMAIN_WIDTH = 4
_BINDING_SEAL = object()
_AUTHORITY_SEAL = object()


def _actual_group_ranks(
    name: str, group: Any, group_ranks_getter: Callable[[Any], Any]
) -> tuple[int, ...]:
    if group is None:
        raise MdpConfigurationError(f"MDP: repeated-D4 {name} group is installed.")
    try:
        ranks = tuple(group_ranks_getter(group))
    except Exception as error:
        raise MdpConfigurationError(
            f"MDP: repeated-D4 {name} group exposes its ordered ranks."
        ) from error
    if any(type(rank) is not int for rank in ranks) or len(set(ranks)) != len(ranks):
        raise MdpConfigurationError(
            f"MDP: repeated-D4 {name} group has unique exact-integer ranks."
        )
    return ranks


@dataclass(frozen=True)
class _RepeatedD4GroupAuthority:
    world_ranks: tuple[int, ...]
    domain_ranks: tuple[int, ...]
    global_rank: int
    expert_parallel_size: int
    _world_group: Any = field(compare=False, repr=False)
    _domain_group: Any = field(compare=False, repr=False)
    _expert_group: Any = field(compare=False, repr=False)
    _device: torch.device = field(compare=False, repr=False)
    _timeout_seconds: float = field(compare=False, repr=False)
    _status_gather_factory: Callable[..., Any] = field(compare=False, repr=False)
    _group_ranks_getter: Callable[[Any], Any] = field(compare=False, repr=False)
    _world_pre_gate: Callable[..., None] = field(compare=False, repr=False)
    _domain_status: Callable[..., Any] = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _RepeatedD4GroupAuthority or self._seal is not _AUTHORITY_SEAL:
            raise MdpConfigurationError(
                "MDP: repeated-D4 group authority is captured after native-group validation."
            )


@dataclass(frozen=True)
class _RepeatedD4GroupBinding:
    """Immutable carrier for one captured native D4 group authority."""

    world_ranks: tuple[int, ...]
    domain_ranks: tuple[int, ...]
    global_rank: int
    expert_parallel_size: int
    _authority: _RepeatedD4GroupAuthority = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _RepeatedD4GroupBinding or self._seal is not _BINDING_SEAL:
            raise MdpConfigurationError(
                "MDP: repeated-D4 group binding is created from validated native groups."
            )

    @property
    def world_group(self) -> Any:
        return self._authority._world_group

    @property
    def domain_group(self) -> Any:
        return self._authority._domain_group

    @property
    def expert_group(self) -> Any:
        return self._authority._expert_group

    def begin_attempt(self, *, byte_generator: Callable[[int], Any] | None = None):
        """Acquire one WORLD-common nonce and bind all status stages to it."""
        authority = _validate_repeated_d4_group_binding(self)
        kwargs = {}
        if byte_generator is not None:
            kwargs["byte_generator"] = byte_generator
        attempt_nonce = acquire_repeated_d4_world_attempt_nonce(
            group=authority._world_group,
            world_ranks=authority.world_ranks,
            global_rank=authority.global_rank,
            device=authority._device,
            timeout_seconds=authority._timeout_seconds,
            status_gather_factory=authority._status_gather_factory,
            **kwargs,
        )
        return _make_repeated_d4_collective_runner(
            attempt_nonce=attempt_nonce,
            world_pre_gate=authority._world_pre_gate,
            domain_status_collector=authority._domain_status,
        )


def _validate_repeated_d4_group_binding(value: Any) -> _RepeatedD4GroupAuthority:
    """Revalidate the sealed carrier against its installed native groups."""
    if (
        type(value) is not _RepeatedD4GroupBinding
        or value._seal is not _BINDING_SEAL
        or type(value._authority) is not _RepeatedD4GroupAuthority
        or value._authority._seal is not _AUTHORITY_SEAL
    ):
        raise MdpStateError("MDP: repeated-D4 group binding retains its private seal.")
    authority = value._authority
    if (
        value.world_ranks != authority.world_ranks
        or value.domain_ranks != authority.domain_ranks
        or value.global_rank != authority.global_rank
        or value.expert_parallel_size != authority.expert_parallel_size
        or not callable(authority._group_ranks_getter)
        or not callable(authority._status_gather_factory)
        or not callable(authority._world_pre_gate)
        or not callable(authority._domain_status)
    ):
        raise MdpStateError("MDP: repeated-D4 group binding matches its captured authority.")
    world_ranks = _validate_world_ranks(
        _actual_group_ranks("WORLD", authority._world_group, authority._group_ranks_getter)
    )
    if type(authority.global_rank) is not int or authority.global_rank not in world_ranks:
        raise MdpStateError("MDP: repeated-D4 group binding retains its WORLD rank authority.")
    start = (world_ranks.index(authority.global_rank) // _DOMAIN_WIDTH) * _DOMAIN_WIDTH
    domain_ranks = world_ranks[start : start + _DOMAIN_WIDTH]
    actual_domain = _actual_group_ranks(
        "domain", authority._domain_group, authority._group_ranks_getter
    )
    expert_ranks = None
    if authority._expert_group is not None:
        expert_ranks = _actual_group_ranks(
            "expert", authority._expert_group, authority._group_ranks_getter
        )
    expected_expert = None if authority.expert_parallel_size == 1 else domain_ranks
    if (
        authority.world_ranks != world_ranks
        or authority.domain_ranks != domain_ranks
        or actual_domain != domain_ranks
        or authority.expert_parallel_size not in (1, 4)
        or expert_ranks != expected_expert
        or not isinstance(authority._device, torch.device)
        or authority._device.type != "cuda"
        or _validate_precollective_timeout(authority._timeout_seconds) != authority._timeout_seconds
    ):
        raise MdpStateError(
            "MDP: repeated-D4 group binding matches its sealed native group authority."
        )
    return authority


def _make_repeated_d4_group_binding(
    *,
    world_group: Any,
    domain_group: Any,
    expert_group: Any,
    global_rank: int,
    expert_parallel_size: int,
    device: torch.device,
    timeout_seconds: float,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    status_gather_factory: Callable[..., Any] = make_precollective_status_gather,
) -> _RepeatedD4GroupBinding:
    """Validate and bind one rank's existing WORLD, D4, and optional EP4 groups."""
    if not callable(group_ranks_getter):
        raise MdpConfigurationError("MDP: repeated-D4 group-ranks getter is callable.")
    if not callable(status_gather_factory):
        raise MdpConfigurationError("MDP: repeated-D4 status-gather factory is callable.")
    world_ranks = _validate_world_ranks(
        _actual_group_ranks("WORLD", world_group, group_ranks_getter)
    )
    if type(global_rank) is not int or global_rank not in world_ranks:
        raise MdpConfigurationError("MDP: repeated-D4 global rank belongs to WORLD.")
    domain_start = (world_ranks.index(global_rank) // _DOMAIN_WIDTH) * _DOMAIN_WIDTH
    domain_ranks = world_ranks[domain_start : domain_start + _DOMAIN_WIDTH]
    actual_domain_ranks = _actual_group_ranks("domain", domain_group, group_ranks_getter)
    if actual_domain_ranks != domain_ranks:
        raise MdpConfigurationError(
            "MDP: repeated-D4 native domain group matches the derived local D4 domain."
        )

    if type(expert_parallel_size) is not int or expert_parallel_size not in (1, 4):
        raise MdpConfigurationError("MDP: repeated-D4 supports EP1 or domain-local EP4 exactly.")
    if expert_parallel_size == 1:
        if expert_group is not None:
            raise MdpConfigurationError("MDP: repeated-D4 EP1 has no expert group.")
    else:
        expert_ranks = _actual_group_ranks("expert", expert_group, group_ranks_getter)
        if expert_ranks != domain_ranks:
            raise MdpConfigurationError(
                "MDP: repeated-D4 EP4 expert group matches the local D4 domain."
            )

    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: repeated-D4 group binding uses a CUDA device.")
    timeout = _validate_precollective_timeout(timeout_seconds)
    world_pre_gate = _make_repeated_d4_world_pre_gate(
        group=world_group,
        world_ranks=world_ranks,
        global_rank=global_rank,
        device=device,
        timeout_seconds=timeout,
        status_gather_factory=status_gather_factory,
    )
    domain_status = _make_repeated_d4_domain_status_collector(
        group=domain_group,
        domain_ranks=domain_ranks,
        global_rank=global_rank,
        device=device,
        timeout_seconds=timeout,
        status_gather_factory=status_gather_factory,
    )
    authority = _RepeatedD4GroupAuthority(
        world_ranks=world_ranks,
        domain_ranks=domain_ranks,
        global_rank=global_rank,
        expert_parallel_size=expert_parallel_size,
        _world_group=world_group,
        _domain_group=domain_group,
        _expert_group=expert_group,
        _device=device,
        _timeout_seconds=timeout,
        _status_gather_factory=status_gather_factory,
        _group_ranks_getter=group_ranks_getter,
        _world_pre_gate=world_pre_gate,
        _domain_status=domain_status,
        _seal=_AUTHORITY_SEAL,
    )
    return _RepeatedD4GroupBinding(
        world_ranks=world_ranks,
        domain_ranks=domain_ranks,
        global_rank=global_rank,
        expert_parallel_size=expert_parallel_size,
        _authority=authority,
        _seal=_BINDING_SEAL,
    )
