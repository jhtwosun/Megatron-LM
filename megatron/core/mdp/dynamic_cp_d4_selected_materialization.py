# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dormant selected-rank materialization owner for repeated D4."""

import weakref
from typing import Any

from megatron.core.mdp.dynamic_cp import DynamicCpGroupMembership
from megatron.core.mdp.dynamic_cp_d4_authority_collective import _snapshot_local_authority
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_plan import EncoderDynamicPlan, validate_encoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_runtime import (
    _dynamic_iteration_plan_digest,
    _DynamicIterationAuthority,
)
from megatron.core.mdp.dynamic_encoder_adapter_capability import (
    DynamicEncoderLocatorAdapterOperations,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError
from megatron.core.mdp.groups import MdpProcessGroups
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.runtime import MdpRuntime
from megatron.core.mdp.vision_locator import VisionLocatorCatalog, validate_vision_locator_catalog

__all__ = ()

_ACTIVE = object()
_RETIRED = object()
# Kept as an inert value so knowledge of an old/static seal cannot mint an owner.
_FACTORY_SEAL = object()
_FACTORY_SEALS: dict[object, tuple[Any, ...]] = {}
_RESERVED_RUNTIMES: dict[int, Any] = {}
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}


def _selected_ranks(authority: _DynamicIterationAuthority) -> tuple[int, ...]:
    plan = authority.encoder_plan
    if type(plan) is not EncoderDynamicPlan:
        raise MdpStateError("MDP: selected materialization requires a joint encoder plan.")
    validate_encoder_dynamic_plan(plan)
    if plan.pool_ranks != authority.participant_ranks:
        raise MdpStateError("MDP: selected materialization plan matches the exact D4 domain.")
    if plan.waves == ():
        if authority.global_manifest.items:
            raise MdpStateError(
                "MDP: selected materialization is empty exactly for text-only input."
            )
        return ()
    if len(plan.waves) != 1 or len(plan.waves[0].executions) != 1:
        raise MdpStateError("MDP: selected materialization supports one execution wave.")
    execution = plan.waves[0].executions[0]
    if (
        execution.group_size not in (1, 2, 4)
        or execution.group_index != 0
        or execution.rank_slots != tuple(range(execution.group_size))
        or execution.item_ids != tuple(item.item_id for item in authority.global_manifest.items)
    ):
        raise MdpStateError(
            "MDP: selected materialization uses the E1/E2/E4 manifest-order prefix."
        )
    selected = tuple(plan.pool_ranks[slot] for slot in execution.rank_slots)
    if selected != authority.participant_ranks[: execution.group_size]:
        raise MdpStateError("MDP: selected materialization uses the exact domain prefix.")
    return selected


def _validate_membership(
    runtime: MdpRuntime, selected_ranks: tuple[int, ...], global_rank: int
) -> DynamicCpGroupMembership | None:
    if not selected_ranks or global_rank not in selected_ranks:
        return None
    groups = runtime.process_groups
    if type(groups) is not MdpProcessGroups or type(groups.encoder_cp_groups) is not tuple:
        raise MdpStateError("MDP: selected materialization uses installed encoder memberships.")
    matches = tuple(
        membership
        for membership in groups.encoder_cp_groups
        if type(membership) is DynamicCpGroupMembership
        and membership.group_size == len(selected_ranks)
        and membership.ranks == selected_ranks
    )
    if len(matches) != 1:
        raise MdpStateError("MDP: selected materialization uses one exact encoder membership.")
    return matches[0]


def _validate_inputs(runtime: Any, authority: Any, catalog: Any) -> tuple[Any, ...]:
    if type(runtime) is not MdpRuntime:
        raise MdpConfigurationError("MDP: selected materialization uses an exact MdpRuntime.")
    if runtime.vision_capture_mode is not VisionCaptureMode.STABLE_LOCATOR_CATALOG:
        raise MdpConfigurationError("MDP: selected materialization requires locator capture mode.")
    operations = runtime.adapter
    if type(operations) is not DynamicEncoderLocatorAdapterOperations:
        raise MdpConfigurationError("MDP: selected materialization uses an exact locator escrow.")
    adapter = operations._adapter()
    capability = runtime.dynamic_adapter_capability
    try:
        capability_record = capability._record
    except AttributeError as error:
        raise MdpStateError(
            "MDP: selected materialization retains its exact runtime adapter capability."
        ) from error
    if (
        operations._record is not capability_record
        or capability_record.capability is not capability
        or capability_record.operations is not operations
        or runtime._dynamic_adapter_owner is not adapter
    ):
        raise MdpStateError(
            "MDP: selected materialization retains its exact runtime adapter capability."
        )
    binding = runtime.dynamic_group_binding
    if type(binding) is not _RepeatedD4GroupBinding:
        raise MdpStateError("MDP: selected materialization retains its exact runtime binding.")
    group = _validate_repeated_d4_group_binding(binding)
    runtime_rank = getattr(runtime.rank_view, "global_rank", None)
    if type(runtime_rank) is not int or runtime_rank != group.global_rank:
        raise MdpStateError("MDP: selected materialization runtime rank matches its exact binding.")
    if type(authority) is not _DynamicIterationAuthority:
        raise MdpConfigurationError("MDP: selected materialization uses exact iteration authority.")
    try:
        _snapshot_local_authority(binding, authority)
        _dynamic_iteration_plan_digest(authority)
    except MdpPlanError as error:
        raise MdpStateError(
            "MDP: selected materialization retains exact plan authority."
        ) from error
    catalog = validate_vision_locator_catalog(catalog)
    if (
        authority.locator_catalog_digest is None
        or authority.locator_catalog_digest != catalog.digest
    ):
        raise MdpStateError("MDP: selected materialization catalog matches locator authority.")
    items = authority.global_manifest.items
    if tuple(entry.item_id for entry in catalog.entries) != tuple(item.item_id for item in items):
        raise MdpStateError("MDP: selected materialization catalog follows manifest item order.")
    if any(
        entry.locator.grid_thw != item.grid_thw
        for entry, item in zip(catalog.entries, items, strict=True)
    ):
        raise MdpStateError("MDP: selected materialization locator grids match the manifest.")
    selected_ranks = _selected_ranks(authority)
    membership = _validate_membership(runtime, selected_ranks, group.global_rank)
    return operations, binding, selected_ranks, membership


class _D4SelectedMaterializationOwner:
    """One active pre-Gate0 payload reservation with no publication authority."""

    __slots__ = (
        "__weakref__",
        "_runtime",
        "authority",
        "catalog",
        "operations",
        "_binding",
        "selected_ranks",
        "_membership",
        "_payloads",
        "_state",
        "_seal",
    )

    def __init__(
        self,
        runtime: MdpRuntime,
        authority: _DynamicIterationAuthority,
        catalog: VisionLocatorCatalog,
        operations: DynamicEncoderLocatorAdapterOperations,
        selected_ranks: tuple[int, ...],
        payloads: tuple[bytes, ...],
        *,
        _factory_seal: object,
    ) -> None:
        trusted = _FACTORY_SEALS.pop(_factory_seal, None)
        if trusted is None or any(
            actual is not expected
            for actual, expected in zip(
                (runtime, authority, catalog, operations, selected_ranks, payloads),
                (*trusted[:4], trusted[5], trusted[7]),
                strict=True,
            )
        ):
            raise MdpConfigurationError("MDP: selected materialization owner is privately minted.")
        binding, membership = trusted[4], trusted[6]
        self._runtime = runtime
        self.authority = authority
        self.catalog = catalog
        self.operations = operations
        self._binding = binding
        self.selected_ranks = selected_ranks
        self._membership = membership
        self._payloads = payloads
        self._state = _ACTIVE
        self._seal = _factory_seal
        identity = id(self)

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            entry = _ACTIVE_OWNERS.get(identity)
            if entry is not None and entry[0] is reference:
                _ACTIVE_OWNERS.pop(identity, None)
                _RESERVED_RUNTIMES.pop(id(entry[1]), None)

        reference = weakref.ref(self, retire)
        _ACTIVE_OWNERS[identity] = (
            reference,
            runtime,
            authority,
            catalog,
            operations,
            binding,
            selected_ranks,
            membership,
            payloads,
        )
        _RESERVED_RUNTIMES[id(runtime)] = reference

    def require(self) -> "_D4SelectedMaterializationOwner":
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_OWNERS.get(id(self))
            if retired is not None and retired() is self:
                raise MdpStateError("MDP: selected materialization owner is retired.")
            raise MdpStateError("MDP: selected materialization owner is inactive or stale.")
        current = (
            self._runtime,
            self.authority,
            self.catalog,
            self.operations,
            self._binding,
            self.selected_ranks,
            self._membership,
            self._payloads,
        )
        if self._state is not _ACTIVE or any(
            actual is not expected for actual, expected in zip(current, entry[1:], strict=True)
        ):
            raise MdpStateError("MDP: selected materialization owner retains sealed fields.")
        try:
            operations, binding, selected_ranks, membership = _validate_inputs(
                self._runtime, self.authority, self.catalog
            )
        except MdpConfigurationError as error:
            raise MdpStateError(
                "MDP: selected materialization owner retains valid sealed inputs."
            ) from error
        if (
            operations is not self.operations
            or binding is not self._binding
            or selected_ranks != self.selected_ranks
            or membership is not self._membership
        ):
            raise MdpStateError(
                "MDP: selected materialization owner retains exact runtime carriers."
            )
        return self

    def _retire(self) -> tuple[bytes, ...]:
        entry = _ACTIVE_OWNERS.pop(id(self), None)
        if entry is None or entry[0]() is not self:
            self.require()
        _RESERVED_RUNTIMES.pop(id(entry[1]), None)
        identity = id(self)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            if _RETIRED_OWNERS.get(identity) is reference:
                _RETIRED_OWNERS.pop(identity, None)

        _RETIRED_OWNERS[identity] = weakref.ref(self, remove)
        payloads = entry[8]
        self._state = _RETIRED
        self._runtime = None
        self.authority = None
        self.catalog = None
        self.operations = None
        self._binding = None
        self.selected_ranks = None
        self._membership = None
        self._payloads = None
        self._seal = None
        return payloads

    def _claim_for_gate0(self) -> tuple[bytes, ...]:
        """Consume the exact prepared bytes for later Gate0 integration."""
        try:
            self.require()
        except BaseException:
            if id(self) in _ACTIVE_OWNERS:
                self._retire()
            raise
        return self._retire()

    def abort(self) -> None:
        """Retire the active reservation without transferring its payloads."""
        try:
            self.require()
        except BaseException:
            if id(self) in _ACTIVE_OWNERS:
                self._retire()
            raise
        self._retire()


def materialize_d4_selected_locator_catalog(
    runtime: MdpRuntime, authority: _DynamicIterationAuthority, catalog: VisionLocatorCatalog
) -> _D4SelectedMaterializationOwner:
    """Prepare one selected rank's exact catalog bytes without communication."""
    runtime_identity = id(runtime)
    if runtime_identity in _RESERVED_RUNTIMES:
        raise MdpStateError("MDP: selected materialization runtime already has an active owner.")
    reservation = object()
    _RESERVED_RUNTIMES[runtime_identity] = reservation
    try:
        operations, binding, selected_ranks, membership = _validate_inputs(
            runtime, authority, catalog
        )
        if binding.global_rank in selected_ranks:
            payloads = tuple(
                operations.materialize_vision_locator(entry.locator) for entry in catalog.entries
            )
            if any(type(payload) is not bytes for payload in payloads):
                raise MdpStateError("MDP: selected materialization produces exact image bytes.")
        else:
            payloads = ()
        factory_seal = object()
        _FACTORY_SEALS[factory_seal] = (
            runtime,
            authority,
            catalog,
            operations,
            binding,
            selected_ranks,
            membership,
            payloads,
        )
        return _D4SelectedMaterializationOwner(
            runtime,
            authority,
            catalog,
            operations,
            selected_ranks,
            payloads,
            _factory_seal=factory_seal,
        )
    except BaseException:
        if "factory_seal" in locals():
            _FACTORY_SEALS.pop(factory_seal, None)
        if _RESERVED_RUNTIMES.get(runtime_identity) is reservation:
            _RESERVED_RUNTIMES.pop(runtime_identity, None)
        raise
