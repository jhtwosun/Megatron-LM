# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private Gate5 selected-subgroup encoder backward."""

import weakref
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
from torch import Tensor

from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as _gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_forward as _forward
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as _gradient
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
_NORMALIZED_COMPLETION_SEAL = object()
_OWNER_ESCROW_SEAL = object()
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_COMPLETIONS: dict[int, tuple[Any, ...]] = {}
_ACTIVE_PREPARED: dict[int, tuple[Any, ...]] = {}
_OWNER_ESCROWS: dict[int, Any] = {}
_TRUSTED_OWNER_ESCROWS: dict[int, Any] = {}
_CANONICAL_OWNER_ESCROWS: dict[int, Any] = {}


def _add_cleanup_note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


def _same_entry(actual: Any, expected: tuple[Any, ...]) -> bool:
    return (
        type(actual) is tuple
        and len(actual) == len(expected)
        and all(value is item for value, item in zip(actual, expected, strict=True))
    )


def _mark_owner_retired(owner: Any) -> None:
    identity = id(owner)

    def remove(reference, registry=_RETIRED_OWNERS, identity=identity):
        if registry.get(identity) is reference:
            del registry[identity]

    _RETIRED_OWNERS[identity] = weakref.ref(owner, remove)


class _NormalizedDecoderCompletionToken(NamedTuple):
    registry: dict[int, tuple[Any, ...]]
    release: Any
    provenance: _gradient._D4ReplayGradientProvenance
    seal: object


class _OwnerEscrow(NamedTuple):
    reference: weakref.ReferenceType[Any]
    owner_entry: tuple[Any, ...]
    runtime_entry: tuple[Any, ...]
    carrier_entry: tuple[Any, ...]
    receipt_entry: tuple[Any, ...]
    decoder_completion_entry: tuple[Any, ...]
    backward_completion_entry: tuple[Any, ...]
    prepared_identity: int
    normalized: _NormalizedDecoderCompletionToken
    seal: object


def _validate_normalized_completion(
    owner: Any,
    authority: _DynamicIterationAuthority,
    completion: Any,
    normalized: Any,
    completion_entry: Any,
) -> None:
    if (
        type(normalized) is not _NormalizedDecoderCompletionToken
        or normalized.seal is not _NORMALIZED_COMPLETION_SEAL
        or type(normalized.registry) is not dict
        or normalized.registry.get(id(completion)) is not completion_entry
        or type(completion_entry) is not tuple
        or len(completion_entry) < 2
        or type(completion_entry[0]) is not weakref.ReferenceType
        or completion_entry[0]() is not owner
        or completion_entry[1] is not completion
        or type(normalized.provenance) is not _gradient._D4ReplayGradientProvenance
        or normalized.provenance._seal is not _gradient._PROVENANCE_SEAL
        or normalized.release is not normalized.provenance.release
        or completion._owner is not normalized.provenance.predecessor
        or completion.authority is not authority
        or completion.globally_reduced_num_tokens is not normalized.provenance.token
        or _gate4._tensor_descriptor(normalized.provenance.token)
        != normalized.provenance.token_descriptor
    ):
        raise MdpStateError("MDP: Gate5 retains exact normalized decoder completion.")


def _owner_escrow_integrity_unchecked(owner: Any, escrow: Any) -> bool:
    if (
        type(escrow) is not _OwnerEscrow
        or escrow.seal is not _OWNER_ESCROW_SEAL
        or type(escrow.reference) is not weakref.ReferenceType
        or escrow.reference() is not owner
        or type(escrow.owner_entry) is not tuple
        or len(escrow.owner_entry) != 12
        or type(escrow.runtime_entry) is not tuple
        or len(escrow.runtime_entry) != 2
        or type(escrow.carrier_entry) is not tuple
        or len(escrow.carrier_entry) < 2
        or type(escrow.receipt_entry) is not tuple
        or len(escrow.receipt_entry) < 2
        or type(escrow.decoder_completion_entry) is not tuple
        or len(escrow.decoder_completion_entry) < 2
        or type(escrow.backward_completion_entry) is not tuple
        or len(escrow.backward_completion_entry) != 5
        or escrow.owner_entry[0] is not escrow.reference
        or not _same_entry(escrow.runtime_entry, (escrow.owner_entry[1], escrow.reference))
        or escrow.carrier_entry[0] is not escrow.reference
        or escrow.carrier_entry[1] is not escrow.owner_entry[5]
        or escrow.receipt_entry[0] is not escrow.reference
        or escrow.receipt_entry[1] is not escrow.owner_entry[8]
        or escrow.decoder_completion_entry[0] is not escrow.reference
        or escrow.decoder_completion_entry[1] is not escrow.owner_entry[4]
        or type(escrow.owner_entry[6]) is not _D4EncoderBackwardComplete
        or not _same_entry(
            escrow.backward_completion_entry,
            (
                escrow.reference,
                escrow.owner_entry[6],
                escrow.owner_entry[6].selected,
                escrow.owner_entry[6].text_only,
                escrow.normalized,
            ),
        )
        or escrow.owner_entry[6]._seal is not _COMPLETE_SEAL
        or escrow.owner_entry[6].normalized_completion_escrow is not escrow.normalized
        or type(escrow.normalized) is not _NormalizedDecoderCompletionToken
        or escrow.normalized.seal is not _NORMALIZED_COMPLETION_SEAL
        or type(escrow.normalized.provenance) is not _gradient._D4ReplayGradientProvenance
        or escrow.normalized.provenance._seal is not _gradient._PROVENANCE_SEAL
        or escrow.normalized.release is not escrow.normalized.provenance.release
        or escrow.owner_entry[4]._owner is not escrow.normalized.provenance.predecessor
        or escrow.owner_entry[4].authority is not escrow.owner_entry[3]
        or escrow.owner_entry[4].globally_reduced_num_tokens
        is not escrow.normalized.provenance.token
        or _gate4._tensor_descriptor(escrow.normalized.provenance.token)
        != escrow.normalized.provenance.token_descriptor
        or type(escrow.owner_entry[7]) is not tuple
        or len(escrow.owner_entry[7]) != 4
        or escrow.owner_entry[7][2] is not escrow.owner_entry[11]
        or type(escrow.owner_entry[11]) is not _forward._D4EncoderForwardOperations
        or escrow.owner_entry[11]._seal is not _forward._OPERATIONS_SEAL
        or escrow.normalized.release is not escrow.owner_entry[11].release
    ):
        return False
    return True


def _owner_escrow_integrity_exact(owner: Any, escrow: Any) -> bool:
    try:
        return _owner_escrow_integrity_unchecked(owner, escrow)
    except BaseException:
        return False


def _is_canonical_owner_escrow(owner: Any, escrow: Any) -> bool:
    return (
        type(escrow) is _OwnerEscrow
        and escrow.seal is _OWNER_ESCROW_SEAL
        and type(escrow.reference) is weakref.ReferenceType
        and escrow.reference() is owner
        and _CANONICAL_OWNER_ESCROWS.get(id(owner)) is escrow
    )


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
    completion: Any = field(compare=False, repr=False)
    gate4_carrier: Any = field(compare=False, repr=False)
    selected: bool
    text_only: bool
    _seal: object = field(compare=False, repr=False)
    normalized_completion_escrow: _NormalizedDecoderCompletionToken = field(
        compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if (
            type(self) is not _D4EncoderBackwardComplete
            or self._seal is not _COMPLETE_SEAL
            or type(self.normalized_completion_escrow) is not _NormalizedDecoderCompletionToken
            or self.normalized_completion_escrow.seal is not _NORMALIZED_COMPLETION_SEAL
        ):
            raise MdpConfigurationError("MDP: encoder backward completion is privately minted.")


def _validated_handle(
    carrier: _gate4._D4EncoderBackwardMember, operations: Any
) -> tuple[EncoderForwardHandle, tuple[tuple[Tensor, EncoderThdLayout], ...]]:
    if (
        type(operations) is not _forward._D4EncoderForwardOperations
        or operations._seal is not _forward._OPERATIONS_SEAL
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
    expected_keys = tuple(
        entry.key
        for entry in carrier.authority.gradient_ledger.entries
        if entry.dst_global_rank == carrier.selected_ranks[0]
    )
    if tuple(carrier.routed_gradients) != expected_keys:
        raise MdpPlanError("MDP: Gate5 receives exact canonical encoder gradient routes.")
    routed = {}
    for key, gradient in carrier.routed_gradients.items():
        if type(key) is not DynamicBridgeKey or key.item_id not in item_ids:
            raise MdpPlanError("MDP: Gate5 receives only manifest encoder gradients.")
        routed.setdefault(key.item_id, []).append(gradient)
    gradients = []
    visited = []
    for output, layout in pairs:
        full = torch.zeros_like(output)
        cursor = 0
        for segment in layout.segments:
            try:
                item_id = by_local_id[segment.global_item_id]
                item_gradients = routed.pop(item_id)
            except KeyError as error:
                raise MdpPlanError(
                    "MDP: Gate5 gradients cover every encoder layout item."
                ) from error
            for item_gradient in item_gradients:
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
            destination = full.narrow(0, cursor, segment.output_rows)
            for item_gradient in item_gradients:
                destination.add_(item_gradient)
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
        return self._require_backward_state(True)

    def _require_backward_state(self, backward_done: bool) -> "_D4EncoderSelectedBackwardOwner":
        retired = _RETIRED_OWNERS.get(id(self))
        if retired is not None and retired() is self:
            raise MdpStateError("MDP: Gate5 selected backward owner is retired.")
        entry = _ACTIVE_OWNERS.get(id(self))
        if (
            type(entry) is not tuple
            or len(entry) != 12
            or type(entry[0]) is not weakref.ReferenceType
            or entry[0]() is not self
        ):
            raise MdpStateError("MDP: Gate5 selected backward owner is exact and active.")
        escrow = _CANONICAL_OWNER_ESCROWS.get(id(self))
        current = (
            self._runtime,
            self.binding,
            self.authority,
            self.completion,
            self.gate4_carrier,
            self.backward_completion,
        )
        if (
            type(escrow) is not _OwnerEscrow
            or escrow.seal is not _OWNER_ESCROW_SEAL
            or _OWNER_ESCROWS.get(id(self)) is not escrow
            or _TRUSTED_OWNER_ESCROWS.get(id(self)) is not escrow
            or _CANONICAL_OWNER_ESCROWS.get(id(self)) is not escrow
            or entry is not escrow.owner_entry
            or not _same_entry(entry, (escrow.reference, *self._trusted))
            or self._state is not _ACTIVE
            or any(
                actual is not expected for actual, expected in zip(current, entry[1:7], strict=True)
            )
        ):
            raise MdpStateError("MDP: Gate5 selected backward owner retains sealed resources.")
        _snapshot_local_authority(self.binding, self.authority)
        completion_entry = _ACTIVE_COMPLETIONS.get(id(self.backward_completion))
        if (
            completion_entry is not escrow.backward_completion_entry
            or completion_entry[0]() is not self
            or completion_entry[1] is not self.backward_completion
            or completion_entry[2] is not self.backward_completion.selected
            or completion_entry[3] is not self.backward_completion.text_only
            or completion_entry[4] is not self.backward_completion.normalized_completion_escrow
            or self.backward_completion.authority is not self.authority
            or self.backward_completion.completion is not self.completion
            or self.backward_completion.gate4_carrier is not self.gate4_carrier
            or self.backward_completion._seal is not _COMPLETE_SEAL
            or self._backward_done is not backward_done
        ):
            raise MdpStateError("MDP: Gate5 retains exact completed backward authority.")
        _validate_normalized_completion(
            self,
            self.authority,
            self.completion,
            self.backward_completion.normalized_completion_escrow,
            escrow.decoder_completion_entry,
        )
        resources = entry[7]
        binding_owner, _buffers, _operations, handle = resources
        if type(self.gate4_carrier) is _gate4._D4EncoderBackwardMember:
            if (
                handle is not self.gate4_carrier.forward_handle
                or handle._backward_done is not backward_done
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
        escrow = _CANONICAL_OWNER_ESCROWS.get(id(self))
        if not _is_canonical_owner_escrow(self, escrow):
            self.require()
        self._abort_from_escrow(escrow, primary_error)

    def _abort_from_escrow(
        self, escrow: _OwnerEscrow, primary_error: BaseException | None = None
    ) -> None:
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: Gate5 abort primary is an exception.")
        retired = _RETIRED_OWNERS.get(id(self))
        if retired is not None and retired() is self:
            raise MdpStateError("MDP: Gate5 selected backward owner is retired.")
        if (
            type(escrow) is not _OwnerEscrow
            or escrow.seal is not _OWNER_ESCROW_SEAL
            or escrow.reference() is not self
        ):
            raise MdpStateError("MDP: Gate5 cleanup retains exact immutable escrow.")
        trusted = escrow.owner_entry[1:]
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: Gate5 owner aborted.")
        )
        live = tuple(
            getattr(self, name, None)
            for name in (
                "_runtime",
                "binding",
                "authority",
                "completion",
                "gate4_carrier",
                "backward_completion",
            )
        )
        if (
            not _owner_escrow_integrity_exact(self, escrow)
            or _ACTIVE_OWNERS.get(id(self)) is not escrow.owner_entry
            or _OWNER_ESCROWS.get(id(self)) is not escrow
            or _TRUSTED_OWNER_ESCROWS.get(id(self)) is not escrow
            or _CANONICAL_OWNER_ESCROWS.get(id(self)) is not escrow
            or not _same_entry(
                escrow.owner_entry, (escrow.reference, *getattr(self, "_trusted", ()))
            )
            or any(
                value is not expected
                for value, expected in zip(live, escrow.owner_entry[1:7], strict=True)
            )
        ):
            _add_cleanup_note(primary, "Gate5 cleanup recovered mutated owner integrity.")
        if _ACTIVE_OWNERS.get(id(self)) is escrow.owner_entry:
            del _ACTIVE_OWNERS[id(self)]
        if _OWNER_ESCROWS.get(id(self)) is escrow:
            del _OWNER_ESCROWS[id(self)]
        if _TRUSTED_OWNER_ESCROWS.get(id(self)) is escrow:
            del _TRUSTED_OWNER_ESCROWS[id(self)]
        if _CANONICAL_OWNER_ESCROWS.get(id(self)) is escrow:
            del _CANONICAL_OWNER_ESCROWS[id(self)]
        _mark_owner_retired(self)
        runtime = trusted[0]
        runtime_entry = _forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is escrow.runtime_entry:
            del _forward._ACTIVE_RUNTIME_OWNERS[id(runtime)]
        if _ACTIVE_COMPLETIONS.get(id(trusted[5])) is escrow.backward_completion_entry:
            del _ACTIVE_COMPLETIONS[id(trusted[5])]
        prepared_entry = _ACTIVE_PREPARED.get(escrow.prepared_identity)
        if type(prepared_entry) is tuple and len(prepared_entry) == 4 and prepared_entry[1] is self:
            del _ACTIVE_PREPARED[escrow.prepared_identity]
        if _gate4._ACTIVE_CARRIERS.get(id(trusted[4])) is escrow.carrier_entry:
            del _gate4._ACTIVE_CARRIERS[id(trusted[4])]
        if _gradient._ACTIVE_RECEIPTS.get(id(trusted[7])) is escrow.receipt_entry:
            del _gradient._ACTIVE_RECEIPTS[id(trusted[7])]
        normalized = escrow.normalized
        if normalized.registry.get(id(trusted[3])) is escrow.decoder_completion_entry:
            del normalized.registry[id(trusted[3])]
        self._state = _RETIRED
        self._backward_done = True
        self.binding = self.authority = self.completion = self.gate4_carrier = None
        self.backward_completion = self._runtime = None
        self._trusted = ()
        resources, leaf_bases, transport_buffers, _operations = (
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
                normalized.release(buffer)
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed Gate5 buffer release error: {error!r}")

    def _claim_for_gradient_finalize(
        self,
        successor: Any,
        successor_registry: dict[int, tuple[Any, ...]],
        successor_entry: tuple[Any, ...],
        runtime_entry: tuple[Any, ...],
        carrier_entry: tuple[Any, ...],
        receipt_entry: tuple[Any, ...],
        decoder_completion_entry: tuple[Any, ...],
        backward_completion_entry: tuple[Any, ...],
    ) -> None:
        """Transfer Gate5 ownership to a prepared Gate6 successor without release."""
        self.require()
        escrow = _CANONICAL_OWNER_ESCROWS.get(id(self))
        reference = successor_entry[0] if type(successor_entry) is tuple else None
        if (
            type(escrow) is not _OwnerEscrow
            or escrow.seal is not _OWNER_ESCROW_SEAL
            or _OWNER_ESCROWS.get(id(self)) is not escrow
            or _TRUSTED_OWNER_ESCROWS.get(id(self)) is not escrow
            or _CANONICAL_OWNER_ESCROWS.get(id(self)) is not escrow
            or _ACTIVE_OWNERS.get(id(self)) is not escrow.owner_entry
            or type(successor_registry) is not dict
            or successor_registry.get(id(successor)) is not None
            or type(successor_entry) is not tuple
            or len(successor_entry) != 21
            or type(reference) is not weakref.ReferenceType
            or reference() is not successor
            or any(
                actual is not expected
                for actual, expected in zip(
                    successor_entry[1:12], escrow.owner_entry[1:12], strict=True
                )
            )
            or not _same_entry(runtime_entry, (self._runtime, reference))
            or not _same_entry(carrier_entry, (reference, *escrow.carrier_entry[1:]))
            or not _same_entry(receipt_entry, (reference, *escrow.receipt_entry[1:]))
            or not _same_entry(
                decoder_completion_entry, (reference, *escrow.decoder_completion_entry[1:])
            )
            or not _same_entry(
                backward_completion_entry, (reference, *escrow.backward_completion_entry[1:])
            )
            or _forward._ACTIVE_RUNTIME_OWNERS.get(id(self._runtime)) is not escrow.runtime_entry
            or _gate4._ACTIVE_CARRIERS.get(id(self.gate4_carrier)) is not escrow.carrier_entry
            or _gradient._ACTIVE_RECEIPTS.get(id(escrow.receipt_entry[1]))
            is not escrow.receipt_entry
            or escrow.normalized.registry.get(id(self.completion))
            is not escrow.decoder_completion_entry
            or _ACTIVE_COMPLETIONS.get(id(self.backward_completion))
            is not escrow.backward_completion_entry
        ):
            raise MdpStateError("MDP: Gate6 claims exact normalized Gate5 ownership.")
        _forward._ACTIVE_RUNTIME_OWNERS[id(self._runtime)] = runtime_entry
        _gate4._ACTIVE_CARRIERS[id(self.gate4_carrier)] = carrier_entry
        _gradient._ACTIVE_RECEIPTS[id(escrow.receipt_entry[1])] = receipt_entry
        escrow.normalized.registry[id(self.completion)] = decoder_completion_entry
        _ACTIVE_COMPLETIONS[id(self.backward_completion)] = backward_completion_entry
        successor_registry[id(successor)] = successor_entry
        del _ACTIVE_OWNERS[id(self)]
        del _OWNER_ESCROWS[id(self)]
        del _TRUSTED_OWNER_ESCROWS[id(self)]
        del _CANONICAL_OWNER_ESCROWS[id(self)]
        _mark_owner_retired(self)
        self._state = _RETIRED
        self._backward_done = True
        self.binding = self.authority = self.completion = self.gate4_carrier = None
        self.backward_completion = self._runtime = None
        self._trusted = ()


def run_repeated_d4_encoder_selected_backward(
    predecessor: _gate4._D4EncoderBackwardAuthorizationOwner,
    authority: _DynamicIterationAuthority,
    completion: Any,
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
    successor_escrow = None
    started = False

    def prepare() -> _PreparedD4EncoderBackward:
        nonlocal prepared, successor, successor_entry, successor_escrow, started
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
        prior_entry = _gate4._ACTIVE_OWNERS.get(id(predecessor))
        carrier_entry = _gate4._ACTIVE_CARRIERS.get(id(predecessor.carrier))
        receipt_entry = _gradient._ACTIVE_RECEIPTS.get(id(predecessor.receipt))
        completion_escrow = prior_entry[11] if prior_entry is not None else None
        completion_entry = (
            completion_escrow[1].get(id(completion))
            if type(completion_escrow) is tuple and len(completion_escrow) == 6
            else None
        )
        runtime_entry = (
            _forward._ACTIVE_RUNTIME_OWNERS.get(id(prior_entry[1]))
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
        normalized = _NormalizedDecoderCompletionToken(
            completion_escrow[1],
            completion_escrow[3],
            completion_escrow[4],
            _NORMALIZED_COMPLETION_SEAL,
        )
        complete = _D4EncoderBackwardComplete(
            authority,
            completion,
            predecessor.carrier,
            selected,
            text_only,
            _COMPLETE_SEAL,
            normalized,
        )
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
        escrow_holder = []
        successor_identity = id(successor)

        def retire(
            retired_reference,
            holder=escrow_holder,
            identity=successor_identity,
            registries=(_OWNER_ESCROWS, _TRUSTED_OWNER_ESCROWS, _CANONICAL_OWNER_ESCROWS),
        ):
            if not holder:
                return
            exact_escrow = holder[0]
            if exact_escrow.reference is not retired_reference:
                return
            for registry in registries:
                if registry.get(identity) is exact_escrow:
                    del registry[identity]

        reference = weakref.ref(successor, retire)
        successor_entry = (reference, *trusted)
        successor_runtime_entry = (prior[0], reference)
        migrated_carrier = (reference, *carrier_entry[1:])
        migrated_receipt = (reference, *receipt_entry[1:])
        migrated_completion = (reference, *completion_entry[1:])
        prepared = _PreparedD4EncoderBackward(successor, gradients, selected, _PREPARED_SEAL)
        backward_completion_entry = (reference, complete, selected, text_only, normalized)
        prepared_entry = (prepared, successor, gradients, selected)
        owner_escrow = _OwnerEscrow(
            reference,
            successor_entry,
            successor_runtime_entry,
            migrated_carrier,
            migrated_receipt,
            migrated_completion,
            backward_completion_entry,
            id(prepared),
            normalized,
            _OWNER_ESCROW_SEAL,
        )
        escrow_holder.append(owner_escrow)
        successor_escrow = owner_escrow
        _OWNER_ESCROWS[id(successor)] = owner_escrow
        _TRUSTED_OWNER_ESCROWS[id(successor)] = owner_escrow
        _CANONICAL_OWNER_ESCROWS[id(successor)] = owner_escrow
        try:
            predecessor._claim_for_selected_backward(
                successor,
                _ACTIVE_OWNERS,
                successor_entry,
                successor_runtime_entry,
                migrated_carrier,
                migrated_receipt,
                migrated_completion,
            )
        except BaseException:
            if _ACTIVE_OWNERS.get(id(successor)) is not successor_entry:
                if _OWNER_ESCROWS.get(id(successor)) is owner_escrow:
                    del _OWNER_ESCROWS[id(successor)]
                if _TRUSTED_OWNER_ESCROWS.get(id(successor)) is owner_escrow:
                    del _TRUSTED_OWNER_ESCROWS[id(successor)]
                if _CANONICAL_OWNER_ESCROWS.get(id(successor)) is owner_escrow:
                    del _CANONICAL_OWNER_ESCROWS[id(successor)]
            raise
        _ACTIVE_COMPLETIONS[id(complete)] = backward_completion_entry
        _ACTIVE_PREPARED[id(prepared)] = prepared_entry
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
        value.owner._require_backward_state(False)
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
        try:
            escrow = successor_escrow
            if (
                type(escrow) is _OwnerEscrow
                and escrow.seal is _OWNER_ESCROW_SEAL
                and (
                    _ACTIVE_OWNERS.get(id(successor)) is escrow.owner_entry
                    or (
                        (retired := _gate4._RETIRED_OWNERS.get(id(predecessor))) is not None
                        and retired() is predecessor
                    )
                )
            ):
                successor._abort_from_escrow(escrow, error)
            else:
                predecessor.abort(error)
        except BaseException as cleanup_error:
            _add_cleanup_note(error, f"suppressed Gate5 cleanup error: {cleanup_error!r}")
        raise
