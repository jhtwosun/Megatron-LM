# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private Gate5 selected-subgroup encoder backward."""

import weakref
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as _gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as _gradient
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as _replay
from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _snapshot_local_authority,
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import (
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment
from megatron.core.mdp.protocols import DynamicEncoderCpBinding

__all__ = ()

_ACTIVE = object()
_RETIRED = object()
_PREPARED_SEAL = object()
_COMPLETE_SEAL = object()
_OWNER_SEAL = object()
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_COMPLETIONS: dict[int, tuple[Any, ...]] = {}
_ACTIVE_PREPARED: dict[int, tuple[Any, ...]] = {}


def _add_cleanup_note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


@dataclass(frozen=True, slots=True)
class _PreparedD4EncoderBackward:
    owner: Any = field(compare=False, repr=False)
    gradients: tuple[Tensor, ...] = field(compare=False, repr=False)
    selected: bool
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _PreparedD4EncoderBackward or self._seal is not _PREPARED_SEAL:
            raise MdpConfigurationError("MDP: selected encoder backward is privately prepared.")


@dataclass(frozen=True, slots=True)
class _D4EncoderBackwardComplete:
    authority: _DynamicIterationAuthority = field(compare=False, repr=False)
    completion: _replay._D4FixedDecoderCompletion = field(compare=False, repr=False)
    gate4_carrier: Any = field(compare=False, repr=False)
    selected: bool
    text_only: bool
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D4EncoderBackwardComplete or self._seal is not _COMPLETE_SEAL:
            raise MdpConfigurationError("MDP: encoder backward completion is privately minted.")


def _validated_handle(
    carrier: _gate4._D4EncoderBackwardMember, operations: Any
) -> tuple[EncoderForwardHandle, tuple[tuple[Tensor, EncoderThdLayout], ...]]:
    if (
        type(operations) is not _replay._forward._D4EncoderForwardOperations
        or operations._seal is not _replay._forward._OPERATIONS_SEAL
        or type(operations.device) is not torch.device
    ):
        raise MdpStateError("MDP: Gate5 retains exact encoder forward operations.")
    handle = carrier.forward_handle
    if type(handle) is not EncoderForwardHandle:
        raise MdpStateError("MDP: Gate5 retains an exact encoder forward handle.")
    if (
        type(handle.chunk_outputs) is not tuple
        or type(handle.chunk_layouts) is not tuple
        or not handle.chunk_outputs
        or len(handle.chunk_outputs) != len(handle.chunk_layouts)
    ):
        raise MdpStateError("MDP: Gate5 retains one layout per encoder output chunk.")
    pairs = []
    for output, layout in zip(handle.chunk_outputs, handle.chunk_layouts, strict=True):
        if (
            type(output) is not Tensor
            or output.dim() != 2
            or not output.requires_grad
            or output.grad_fn is None
            or type(layout) is not EncoderThdLayout
            or type(layout.segments) is not tuple
            or not layout.segments
            or type(carrier.authority.bridge_width) is not int
            or tuple(output.shape) != (layout.total_output_rows, carrier.authority.bridge_width)
            or output.dtype != carrier.authority.bridge_dtype
            or output.device != operations.device
            or not output.is_contiguous()
        ):
            raise MdpStateError("MDP: Gate5 retains exact graph-connected chunk outputs.")
        cursor = 0
        for segment in layout.segments:
            if (
                type(segment) is not EncoderThdSegment
                or type(segment.global_item_id) is not int
                or type(segment.output_row_start) is not int
                or type(segment.output_rows) is not int
                or segment.output_row_start != cursor
                or segment.output_rows <= 0
            ):
                raise MdpPlanError("MDP: Gate5 encoder layouts densely cover chunk outputs.")
            cursor += segment.output_rows
        if cursor != output.shape[0]:
            raise MdpPlanError("MDP: Gate5 encoder layouts cover every output row.")
        pairs.append((output, layout))
    return handle, tuple(pairs)


def _leader_gradients(
    carrier: _gate4._D4EncoderBackwardMember, operations: Any
) -> tuple[Tensor, ...]:
    _handle, pairs = _validated_handle(carrier, operations)
    item_ids = tuple(item.item_id for item in carrier.authority.global_manifest.items)
    by_local_id = {}
    for item_id in item_ids:
        if type(item_id) is not GlobalVisionItemId:
            raise MdpPlanError("MDP: Gate5 manifest uses exact global vision item ids.")
        if item_id.local_item_id in by_local_id:
            raise MdpPlanError("MDP: Gate5 manifest has unique local encoder item ids.")
        by_local_id[item_id.local_item_id] = item_id
    routed = {}
    for key, gradient in carrier.routed_gradients.items():
        if type(key) is not DynamicBridgeKey or key.item_id in routed:
            raise MdpPlanError("MDP: Gate5 receives one routed gradient per encoder item.")
        routed[key.item_id] = gradient
    gradients = []
    visited = []
    for output, layout in pairs:
        full = torch.zeros_like(output)
        cursor = 0
        for segment in layout.segments:
            try:
                item_id = by_local_id[segment.global_item_id]
                item_gradient = routed.pop(item_id)
            except KeyError as error:
                raise MdpPlanError(
                    "MDP: Gate5 gradients cover every encoder layout item."
                ) from error
            if (
                type(item_gradient) is not Tensor
                or tuple(item_gradient.shape) != (segment.output_rows, output.shape[1])
                or item_gradient.dtype != output.dtype
                or item_gradient.device != output.device
                or not item_gradient.is_contiguous()
                or item_gradient.requires_grad
                or item_gradient.grad_fn is not None
            ):
                raise MdpStateError("MDP: Gate5 item gradients match encoder output geometry.")
            full.narrow(0, cursor, segment.output_rows).copy_(item_gradient)
            visited.append(item_id)
            cursor += segment.output_rows
        gradients.append(full)
    if tuple(visited) != item_ids or routed:
        raise MdpPlanError("MDP: Gate5 gradients follow exact manifest order without extras.")
    return tuple(gradients)


def _prepare_gradients(carrier: Any, operations: Any) -> tuple[Tensor, ...]:
    if type(carrier) is _gate4._D4EncoderBackwardEmpty:
        return ()
    if type(carrier) is not _gate4._D4EncoderBackwardMember:
        raise MdpStateError("MDP: Gate5 uses an exact armed carrier.")
    if carrier.is_leader:
        return _leader_gradients(carrier, operations)
    _handle, pairs = _validated_handle(carrier, operations)
    return tuple(torch.zeros_like(output) for output, _layout in pairs)


class _D4EncoderSelectedBackwardOwner:
    """Sole owner after selected Gate5 backward completes."""

    __slots__ = (
        "__weakref__",
        "binding",
        "authority",
        "completion",
        "gate4_carrier",
        "backward_completion",
        "_runtime",
        "_trusted",
        "_state",
        "_backward_done",
    )

    def __init__(self, trusted: tuple[Any, ...], seal: object) -> None:
        if seal is not _OWNER_SEAL:
            raise MdpConfigurationError("MDP: Gate5 owner is privately minted.")
        self._runtime, self.binding, self.authority, self.completion = trusted[:4]
        self.gate4_carrier, self.backward_completion = trusted[4:6]
        self._trusted = trusted
        self._state = _ACTIVE
        self._backward_done = False

    def require(self) -> "_D4EncoderSelectedBackwardOwner":
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            if id(self) in _RETIRED_OWNERS:
                raise MdpStateError("MDP: Gate5 selected backward owner is retired.")
            raise MdpStateError("MDP: Gate5 selected backward owner is exact and active.")
        current = (
            self._runtime,
            self.binding,
            self.authority,
            self.completion,
            self.gate4_carrier,
            self.backward_completion,
        )
        if self._state is not _ACTIVE or any(
            actual is not expected for actual, expected in zip(current, entry[1:7], strict=True)
        ):
            raise MdpStateError("MDP: Gate5 selected backward owner retains sealed resources.")
        _snapshot_local_authority(self.binding, self.authority)
        completion_entry = _ACTIVE_COMPLETIONS.get(id(self.backward_completion))
        if (
            completion_entry is None
            or completion_entry[0]() is not self
            or completion_entry[1] is not self.backward_completion
            or completion_entry[2] is not self.backward_completion.selected
            or completion_entry[3] is not self.backward_completion.text_only
            or self.backward_completion.authority is not self.authority
            or self.backward_completion.completion is not self.completion
            or self.backward_completion.gate4_carrier is not self.gate4_carrier
            or self.backward_completion._seal is not _COMPLETE_SEAL
            or self._backward_done is not True
        ):
            raise MdpStateError("MDP: Gate5 retains exact completed backward authority.")
        resources = entry[7]
        binding_owner, _buffers, _operations, handle = resources
        if type(self.gate4_carrier) is _gate4._D4EncoderBackwardMember:
            if (
                handle is not self.gate4_carrier.forward_handle
                or handle._backward_done is not True
                or handle._released is not False
                or type(binding_owner) is not DynamicEncoderCpBinding
                or binding_owner.active is not True
            ):
                raise MdpStateError("MDP: Gate5 retains exact backward graph and binding.")
        elif handle is not None or binding_owner is not None:
            raise MdpStateError("MDP: Gate5 empty owner retains no selected resources.")
        return self

    def abort(self, primary_error: BaseException | None = None) -> None:
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: Gate5 abort primary is an exception.")
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        trusted = entry[1:]
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: Gate5 owner aborted.")
        )
        _ACTIVE_OWNERS.pop(id(self))
        _RETIRED_OWNERS[id(self)] = weakref.ref(self)
        runtime = trusted[0]
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            del _replay._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)]
        _ACTIVE_COMPLETIONS.pop(id(trusted[5]), None)
        for prepared_id, prepared_entry in tuple(_ACTIVE_PREPARED.items()):
            if prepared_entry[1] is self:
                _ACTIVE_PREPARED.pop(prepared_id)
        _gate4._ACTIVE_CARRIERS.pop(id(trusted[4]), None)
        _gradient._ACTIVE_RECEIPTS.pop(id(trusted[7]), None)
        _replay._ACTIVE_COMPLETIONS.pop(id(trusted[3]), None)
        self._state = _RETIRED
        self._backward_done = True
        self.binding = self.authority = self.completion = self.gate4_carrier = None
        self.backward_completion = self._runtime = None
        self._trusted = ()
        resources, leaf_bases, transport_buffers, operations = (
            trusted[6],
            trusted[8],
            trusted[9],
            trusted[10],
        )
        binding_owner, predecessor_buffers, _old_operations, handle = resources
        if handle is not None and not handle.consumed:
            try:
                handle.release_forward_only()
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed Gate5 graph release error: {error!r}")
        if binding_owner is not None:
            try:
                binding_owner.restore(primary)
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed Gate5 binding restore error: {error!r}")
        for buffer in (*transport_buffers, *leaf_bases, *predecessor_buffers):
            try:
                operations.release(buffer)
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed Gate5 buffer release error: {error!r}")


def run_repeated_d4_encoder_selected_backward(
    predecessor: _gate4._D4EncoderBackwardAuthorizationOwner,
    authority: _DynamicIterationAuthority,
    completion: _replay._D4FixedDecoderCompletion,
    *,
    byte_generator=None,
) -> _D4EncoderSelectedBackwardOwner:
    """Authorize Gate5, then execute backward on every selected member."""
    if type(predecessor) is not _gate4._D4EncoderBackwardAuthorizationOwner:
        raise MdpConfigurationError("MDP: selected Gate5 uses an exact Gate4 owner.")
    predecessor.require()
    if authority is not predecessor.authority or completion is not predecessor.completion:
        raise MdpStateError("MDP: selected Gate5 uses exact authority and completion.")
    binding = predecessor.binding
    prepared = None
    successor = None
    successor_entry = None
    started = False

    def prepare() -> _PreparedD4EncoderBackward:
        nonlocal prepared, successor, successor_entry, started
        if started:
            raise MdpStateError("MDP: selected Gate5 preparation is one-shot.")
        started = True
        predecessor.require()
        gradients = _prepare_gradients(predecessor.carrier, predecessor._trusted[8])
        predecessor.require()
        selected = type(predecessor.carrier) is _gate4._D4EncoderBackwardMember
        text_only = (
            type(predecessor.carrier) is _gate4._D4EncoderBackwardEmpty
            and predecessor.carrier.text_only
        )
        complete = _D4EncoderBackwardComplete(
            authority, completion, predecessor.carrier, selected, text_only, _COMPLETE_SEAL
        )
        prior_entry = _gate4._ACTIVE_OWNERS.get(id(predecessor))
        carrier_entry = _gate4._ACTIVE_CARRIERS.get(id(predecessor.carrier))
        receipt_entry = _gradient._ACTIVE_RECEIPTS.get(id(predecessor.receipt))
        completion_entry = _replay._ACTIVE_COMPLETIONS.get(id(completion))
        runtime_entry = (
            _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(prior_entry[1]))
            if prior_entry is not None
            else None
        )
        if (
            prior_entry is None
            or prior_entry[0]() is not predecessor
            or runtime_entry is None
            or runtime_entry[0] is not prior_entry[1]
            or runtime_entry[1]() is not predecessor
            or carrier_entry is None
            or carrier_entry[0]() is not predecessor
            or receipt_entry is None
            or receipt_entry[0]() is not predecessor
            or completion_entry is None
            or completion_entry[0]() is not predecessor
        ):
            raise MdpStateError("MDP: Gate5 claims exact predecessor registries.")
        prior = prior_entry[1:]
        trusted = (
            prior[0],
            binding,
            authority,
            completion,
            prior[4],
            complete,
            prior[5],
            prior[3],
            prior[6],
            prior[7],
            prior[8],
        )
        successor = _D4EncoderSelectedBackwardOwner(trusted, _OWNER_SEAL)
        reference = weakref.ref(successor)
        successor_entry = (reference, *trusted)
        runtime_entry = (prior[0], reference)
        migrated_carrier = (reference, *carrier_entry[1:])
        migrated_receipt = (reference, *receipt_entry[1:])
        migrated_completion = (reference, *completion_entry[1:])
        prepared = _PreparedD4EncoderBackward(successor, gradients, selected, _PREPARED_SEAL)
        predecessor._claim_for_selected_backward(
            successor,
            _ACTIVE_OWNERS,
            successor_entry,
            runtime_entry,
            migrated_carrier,
            migrated_receipt,
            migrated_completion,
        )
        _ACTIVE_COMPLETIONS[id(complete)] = (reference, complete, selected, text_only)
        _ACTIVE_PREPARED[id(prepared)] = (prepared, successor, gradients, selected)
        return prepared

    def authorize(value):
        entry = _ACTIVE_PREPARED.get(id(value))
        if (
            type(value) is not _PreparedD4EncoderBackward
            or value is not prepared
            or entry is None
            or entry[0] is not value
            or value.owner is not entry[1]
            or value.gradients is not entry[2]
            or type(value.selected) is not bool
            or value.selected is not entry[3]
            or value._seal is not _PREPARED_SEAL
        ):
            raise MdpTaskFatalError("MDP: Gate5 retains exact prepared backward.")
        return value

    try:
        result = run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=5,
            prepare=prepare,
            domain_collective=authorize,
            byte_generator=byte_generator,
        )
        try:
            if authorize(result) is not prepared:
                raise MdpStateError("MDP: Gate5 returns exact post-WORLD authorization.")
            owner_entry = _ACTIVE_OWNERS.get(id(successor))
            if owner_entry is None or owner_entry[0]() is not successor:
                raise MdpStateError("MDP: Gate5 retains exact post-WORLD owner.")
            if prepared.selected:
                owner_entry[5].forward_handle.backward(prepared.gradients)
            successor._backward_done = True
            _ACTIVE_PREPARED.pop(id(prepared))
            return successor.require()
        except BaseException as error:
            if type(error) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: selected encoder backward failed after Gate5 final WORLD."
            ) from error
    except BaseException as error:
        target = (
            successor
            if successor is not None and _ACTIVE_OWNERS.get(id(successor)) is successor_entry
            else predecessor
        )
        try:
            target.abort(error)
        except BaseException as cleanup_error:
            _add_cleanup_note(error, f"suppressed Gate5 cleanup error: {cleanup_error!r}")
        raise
