# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure one-wave encoder execution claim for repeated D4."""

import weakref
from types import MappingProxyType
from typing import Any

from megatron.core.mdp.dynamic_cp import DynamicCpGroupMembership
from megatron.core.mdp.dynamic_cp_d4_authority_collective import _snapshot_local_authority
from megatron.core.mdp.dynamic_cp_d4_encoder_capture import _D4EncoderCaptureOwner
from megatron.core.mdp.dynamic_cp_plan import EncoderDynamicPlan, validate_encoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_runtime import (
    _dynamic_iteration_plan_digest,
    _DynamicIterationAuthority,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.groups import MdpProcessGroups
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.vision_locator import VisionLocatorCatalog

__all__ = ()

_ACTIVE = object()
_RETIRED = object()
_PENDING: dict[object, tuple[int, ...]] = {}
_ACTIVE_CLAIMS: dict[int, tuple[Any, ...]] = {}
_RETIRED_CLAIMS: dict[int, weakref.ReferenceType[Any]] = {}


class _D4EncoderExecutionClaim:
    """One exact pure execution selection and transferred capture ownership."""

    __slots__ = (
        "__weakref__",
        "authority",
        "binding",
        "selected_ranks",
        "membership",
        "layout",
        "is_selected",
        "is_leader",
        "text_only",
        "capture_mode",
        "locator_catalog",
        "source_window",
        "local_manifest",
        "sample_locations",
        "pixel_sidecar",
        "_runtime",
        "_pixels",
        "_state",
    )

    def __init__(
        self,
        *,
        authority: _DynamicIterationAuthority,
        binding: Any,
        selected_ranks: tuple[int, ...],
        membership: DynamicCpGroupMembership | None,
        layout: EncoderThdLayout | None,
        is_selected: bool,
        is_leader: bool,
        text_only: bool,
        capture_mode: VisionCaptureMode = VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
        locator_catalog: VisionLocatorCatalog | None = None,
        _factory_seal: object,
    ) -> None:
        values = (authority, binding, selected_ranks, membership, layout)
        if capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG:
            values += (locator_catalog,)
        if _PENDING.pop(_factory_seal, None) != tuple(id(value) for value in values):
            raise MdpConfigurationError(
                "MDP: D4 encoder execution claim is minted by its private factory."
            )
        self.authority = authority
        self.binding = binding
        self.selected_ranks = selected_ranks
        self.membership = membership
        self.layout = layout
        self.is_selected = is_selected
        self.is_leader = is_leader
        self.text_only = text_only
        self.capture_mode = capture_mode
        self.locator_catalog = locator_catalog
        self.source_window = None
        self.local_manifest = None
        self.sample_locations = MappingProxyType({})
        self.pixel_sidecar = MappingProxyType({})
        self._runtime = None
        self._pixels = None
        self._state = _ACTIVE

    def _activate(self, transfer: tuple[Any, ...]) -> None:
        if len(transfer) == 6:
            runtime, binding, source_window, manifest, locations, pixels = transfer
            capture_mode, locator_catalog = VisionCaptureMode.SOURCE_PIXEL_SIDECAR, None
        else:
            (
                runtime,
                binding,
                source_window,
                manifest,
                locations,
                pixels,
                capture_mode,
                locator_catalog,
            ) = transfer
        if binding is not self.binding or type(pixels) is not dict:
            raise MdpStateError("MDP: encoder execution receives its exact capture transfer.")
        if capture_mode is not self.capture_mode or locator_catalog is not self.locator_catalog:
            raise MdpStateError("MDP: encoder execution receives its exact capture mode authority.")
        self._runtime = runtime
        self.source_window = source_window
        self.local_manifest = manifest
        self.sample_locations = locations
        self._pixels = pixels
        self.pixel_sidecar = MappingProxyType(pixels)
        identity = id(self)

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            entry = _ACTIVE_CLAIMS.get(identity)
            if entry is not None and entry[0] is reference:
                del _ACTIVE_CLAIMS[identity]

        reference = weakref.ref(self, retire)
        _ACTIVE_CLAIMS[identity] = (
            reference,
            runtime,
            self.authority,
            self.binding,
            self.selected_ranks,
            self.membership,
            self.layout,
            self.source_window,
            self.local_manifest,
            self.sample_locations,
            pixels,
            self.pixel_sidecar,
            self.is_selected,
            self.is_leader,
            self.text_only,
            self.capture_mode,
            self.locator_catalog,
        )

    def require(self) -> "_D4EncoderExecutionClaim":
        """Validate this exact live claim without consuming it."""
        entry = _ACTIVE_CLAIMS.get(id(self))
        if entry is None or entry[0]() is not self:
            reference = _RETIRED_CLAIMS.get(id(self))
            if reference is not None and reference() is self:
                raise MdpStateError("MDP: D4 encoder execution claim is retired.")
            raise MdpStateError("MDP: D4 encoder execution claim is the exact active claim.")
        current = (
            self._runtime,
            self.authority,
            self.binding,
            self.selected_ranks,
            self.membership,
            self.layout,
            self.source_window,
            self.local_manifest,
            self.sample_locations,
            self._pixels,
            self.pixel_sidecar,
            self.is_selected,
            self.is_leader,
            self.text_only,
            self.capture_mode,
            self.locator_catalog,
        )
        if self._state is not _ACTIVE or any(
            actual is not expected for actual, expected in zip(current, entry[1:], strict=True)
        ):
            raise MdpStateError("MDP: D4 encoder execution claim retains sealed fields.")
        return self

    def abort(self) -> None:
        """Retire once and release only this claim's transferred pixels."""
        entry = _ACTIVE_CLAIMS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        pixels = entry[10]
        identity = id(self)
        _ACTIVE_CLAIMS.pop(identity)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            if _RETIRED_CLAIMS.get(identity) is reference:
                del _RETIRED_CLAIMS[identity]

        _RETIRED_CLAIMS[identity] = weakref.ref(self, remove)
        self._state = _RETIRED
        for name in (
            "authority",
            "binding",
            "selected_ranks",
            "membership",
            "layout",
            "source_window",
            "local_manifest",
            "sample_locations",
            "pixel_sidecar",
            "capture_mode",
            "locator_catalog",
            "_runtime",
            "_pixels",
        ):
            setattr(self, name, None)
        pixels.clear()


def _source_layout(
    authority: _DynamicIterationAuthority,
    owner: _D4EncoderCaptureOwner,
    capture_mode: VisionCaptureMode,
):
    items = authority.global_manifest.items
    locations = owner.sample_locations
    source_window = owner.source_window
    if tuple(item.item_id for item in source_window.items) != tuple(item.item_id for item in items):
        raise MdpStateError("MDP: encoder execution source items match manifest order.")
    pixels = owner.pixel_sidecar
    if capture_mode is VisionCaptureMode.SOURCE_PIXEL_SIDECAR and tuple(pixels) != tuple(
        item.item_id.local_item_id for item in items
    ):
        raise MdpStateError("MDP: encoder execution source pixels match manifest item order.")
    payload_start = 0
    output_start = 0
    segments = []
    for item in items:
        try:
            microbatch_id, local_sample_id = locations[item.sample_id]
        except (KeyError, TypeError, ValueError) as error:
            raise MdpStateError(
                "MDP: encoder execution source locations cover the manifest."
            ) from error
        t, h, w = item.grid_thw
        payload_rows = t * h * w
        segments.append(
            EncoderThdSegment(
                global_item_id=item.item_id.local_item_id,
                microbatch_id=microbatch_id,
                sample_id=local_sample_id,
                image_ordinal=item.image_ordinal,
                payload_row_start=payload_start,
                payload_rows=payload_rows,
                output_row_start=output_start,
                output_rows=item.output_rows,
                grid_thw=item.grid_thw,
            )
        )
        payload_start += payload_rows
        output_start += item.output_rows
    return EncoderThdLayout(producer_worker_id=0, segments=tuple(segments))


def _claim_d4_encoder_execution(owner, authority, *, capture_mode, locator_catalog):
    """Purely select one E1/E2/E4 execution and consume its capture owner."""
    if type(owner) is not _D4EncoderCaptureOwner:
        raise MdpConfigurationError("MDP: encoder execution uses an exact capture owner.")
    owner.require()
    if owner.capture_mode is not capture_mode:
        raise MdpStateError("MDP: encoder execution claim matches exact capture mode.")
    if type(authority) is not _DynamicIterationAuthority:
        raise MdpConfigurationError("MDP: encoder execution uses exact iteration authority.")
    binding = owner.binding
    _snapshot_local_authority(binding, authority)
    _dynamic_iteration_plan_digest(authority)
    plan = authority.encoder_plan
    if type(plan) is not EncoderDynamicPlan:
        raise MdpStateError("MDP: encoder execution requires a joint encoder plan.")
    validate_encoder_dynamic_plan(plan)
    if (
        plan.pool_ranks != binding.domain_ranks
        or authority.participant_ranks != binding.domain_ranks
    ):
        raise MdpStateError("MDP: encoder execution plan matches its exact D4 domain.")
    text_only = plan.waves == ()
    membership = None
    layout = None
    selected_ranks = ()
    is_selected = False
    is_leader = False
    if text_only:
        if authority.global_manifest.items:
            raise MdpStateError("MDP: encoder execution is empty exactly for text-only input.")
    else:
        if len(plan.waves) != 1 or len(plan.waves[0].executions) != 1:
            raise MdpStateError("MDP: encoder execution supports exactly one wave and execution.")
        execution = plan.waves[0].executions[0]
        if (
            execution.group_size not in (1, 2, 4)
            or execution.group_index != 0
            or execution.rank_slots != tuple(range(execution.group_size))
            or execution.item_ids != tuple(item.item_id for item in authority.global_manifest.items)
        ):
            raise MdpStateError("MDP: encoder execution is an E1/E2/E4 manifest-order prefix.")
        selected_ranks = tuple(plan.pool_ranks[slot] for slot in execution.rank_slots)
        if selected_ranks[0] != binding.domain_ranks[0]:
            raise MdpStateError("MDP: encoder execution leader is the exact metadata source.")
        is_selected = binding.global_rank in selected_ranks
        is_leader = binding.global_rank == selected_ranks[0]
        runtime = owner._trusted_runtime
        if is_selected:
            process_groups = runtime.process_groups
            if type(process_groups) is not MdpProcessGroups:
                raise MdpStateError("MDP: encoder execution uses exact installed process groups.")
            memberships = process_groups.encoder_cp_groups
            if type(memberships) is not tuple:
                raise MdpStateError("MDP: encoder execution memberships are immutable.")
            matches = tuple(
                candidate
                for candidate in memberships
                if type(candidate) is DynamicCpGroupMembership
                and candidate.group_size == execution.group_size
            )
            if len(matches) != 1 or matches[0].ranks != selected_ranks:
                raise MdpStateError("MDP: encoder execution selects its exact E1b membership.")
            membership = matches[0]
        if is_leader:
            layout = _source_layout(authority, owner, capture_mode)
    values = (authority, binding, selected_ranks, membership, layout)
    if capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG:
        values += (locator_catalog,)
    token = object()
    _PENDING[token] = tuple(id(value) for value in values)
    claim = _D4EncoderExecutionClaim(
        authority=authority,
        binding=binding,
        selected_ranks=selected_ranks,
        membership=membership,
        layout=layout,
        is_selected=is_selected,
        is_leader=is_leader,
        text_only=text_only,
        capture_mode=capture_mode,
        locator_catalog=locator_catalog,
        _factory_seal=token,
    )
    transfer = owner._claim_for_execution(authority, locator_catalog)
    claim._activate(transfer)
    return claim


def claim_d4_encoder_execution(
    owner: _D4EncoderCaptureOwner, authority: _DynamicIterationAuthority
) -> _D4EncoderExecutionClaim:
    """Select SOURCE_PIXEL_SIDECAR execution with the legacy two-argument API."""
    return _claim_d4_encoder_execution(
        owner, authority, capture_mode=VisionCaptureMode.SOURCE_PIXEL_SIDECAR, locator_catalog=None
    )


def claim_d4_locator_encoder_execution(
    owner: _D4EncoderCaptureOwner,
    authority: _DynamicIterationAuthority,
    locator_catalog: VisionLocatorCatalog,
) -> _D4EncoderExecutionClaim:
    """Select locator execution bound to the exact projected domain catalog."""
    return _claim_d4_encoder_execution(
        owner,
        authority,
        capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        locator_catalog=locator_catalog,
    )
