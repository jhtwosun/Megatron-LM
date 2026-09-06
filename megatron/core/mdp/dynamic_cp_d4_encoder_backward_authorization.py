# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private Gate4 authorization for encoder-only selected backward."""

import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as _gradient
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as _replay
from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp_bridge_transport import validate_prepared_dynamic_bridge_exchange
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _snapshot_local_authority,
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_plan import EncoderDynamicPlan, validate_encoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import (
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)
from megatron.core.mdp.protocols import DynamicEncoderCpBinding

__all__ = ()

_ACTIVE = object()
_RETIRED = object()
_MEMBER_SEAL = object()
_EMPTY_SEAL = object()
_OWNER_SEAL = object()
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_CARRIERS: dict[int, tuple[Any, ...]] = {}


def _add_cleanup_note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


@dataclass(frozen=True, slots=True)
class _D4EncoderBackwardMember:
    """Sealed Gate4 authorization for one selected encoder rank."""

    authority: _DynamicIterationAuthority = field(compare=False, repr=False)
    completion: _replay._D4FixedDecoderCompletion = field(compare=False, repr=False)
    receipt: _gradient._D4EncoderOnlyGradientReceipt = field(compare=False, repr=False)
    selected_ranks: tuple[int, ...]
    member_index: int
    is_leader: bool
    forward_handle: Any = field(compare=False, repr=False)
    routed_gradients: MappingProxyType = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D4EncoderBackwardMember or self._seal is not _MEMBER_SEAL:
            raise MdpConfigurationError("MDP: encoder backward member is privately minted.")


@dataclass(frozen=True, slots=True)
class _D4EncoderBackwardEmpty:
    """Sealed Gate4 authorization for a nonmember or text-only rank."""

    authority: _DynamicIterationAuthority = field(compare=False, repr=False)
    completion: _replay._D4FixedDecoderCompletion = field(compare=False, repr=False)
    receipt: _gradient._D4EncoderOnlyGradientReceipt = field(compare=False, repr=False)
    selected_ranks: tuple[int, ...]
    text_only: bool
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D4EncoderBackwardEmpty or self._seal is not _EMPTY_SEAL:
            raise MdpConfigurationError("MDP: encoder backward empty carrier is privately minted.")


def _selected_ranks(authority: _DynamicIterationAuthority) -> tuple[int, ...]:
    plan = authority.encoder_plan
    if type(plan) is not EncoderDynamicPlan:
        raise MdpStateError("MDP: encoder backward authorization requires a joint encoder plan.")
    validate_encoder_dynamic_plan(plan)
    if plan.pool_ranks != authority.participant_ranks:
        raise MdpStateError("MDP: encoder backward plan matches the exact D4 domain.")
    if plan.waves == ():
        if authority.global_manifest.items:
            raise MdpPlanError("MDP: encoder backward is empty exactly for text-only input.")
        return ()
    if len(plan.waves) != 1 or len(plan.waves[0].executions) != 1:
        raise MdpPlanError("MDP: encoder backward supports one execution wave.")
    execution = plan.waves[0].executions[0]
    if (
        execution.group_size not in (1, 2, 4)
        or execution.group_index != 0
        or execution.rank_slots != tuple(range(execution.group_size))
        or execution.item_ids != tuple(item.item_id for item in authority.global_manifest.items)
    ):
        raise MdpPlanError("MDP: encoder backward uses the E1/E2/E4 manifest-order prefix.")
    selected = tuple(plan.pool_ranks[slot] for slot in execution.rank_slots)
    if selected[0] != authority.participant_ranks[0]:
        raise MdpPlanError("MDP: encoder backward leader is the exact domain source.")
    return selected


class _D4EncoderBackwardAuthorizationOwner:
    """Sole runtime owner after Gate4 authorization."""

    __slots__ = (
        "__weakref__",
        "authority",
        "completion",
        "receipt",
        "carrier",
        "_runtime",
        "_trusted",
        "_state",
        "_prepared_reference",
        "_prepared_predecessor_reference",
    )

    def __init__(self, trusted: tuple[Any, ...], seal: object) -> None:
        if seal is not _OWNER_SEAL:
            raise MdpConfigurationError("MDP: encoder backward owner is privately minted.")
        self._runtime, self.authority, self.completion, self.receipt, self.carrier = trusted[:5]
        self._trusted = trusted
        self._state = _ACTIVE
        self._prepared_reference = None
        self._prepared_predecessor_reference = None

    def _prepare_from(self, predecessor: _gradient._D4EncoderGradientRouteOwner) -> None:
        entry = _gradient._ACTIVE_OWNERS.get(id(predecessor))
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(self._runtime))
        if (
            entry is None
            or entry[0]() is not predecessor
            or runtime_entry is None
            or runtime_entry[0] is not self._runtime
            or runtime_entry[1]() is not predecessor
        ):
            raise MdpStateError("MDP: encoder backward replaces the exact gradient owner.")
        identity = id(self)
        runtime_identity = id(self._runtime)

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            current = _ACTIVE_OWNERS.get(identity)
            if current is not None and current[0] is reference:
                del _ACTIVE_OWNERS[identity]
            current_runtime = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
            if current_runtime is not None and current_runtime[1] is reference:
                del _replay._forward._ACTIVE_RUNTIME_OWNERS[runtime_identity]

        self._prepared_reference = weakref.ref(self, retire)
        self._prepared_predecessor_reference = weakref.ref(predecessor)

    def _claim_prepared(self, predecessor: _gradient._D4EncoderGradientRouteOwner) -> None:
        reference = self._prepared_reference
        _ACTIVE_OWNERS[id(self)] = (reference, *self._trusted)
        if type(self.carrier) is _D4EncoderBackwardMember:
            role = (
                self.carrier.selected_ranks,
                self.carrier.member_index,
                self.carrier.is_leader,
                self.carrier.forward_handle,
                self.carrier.routed_gradients,
            )
        else:
            role = (self.carrier.selected_ranks, self.carrier.text_only)
        _ACTIVE_CARRIERS[id(self.carrier)] = (
            reference,
            self.carrier,
            self.authority,
            self.completion,
            self.receipt,
            *role,
        )
        _replay._forward._ACTIVE_RUNTIME_OWNERS[id(self._runtime)] = (self._runtime, reference)
        receipt_entry = _gradient._ACTIVE_RECEIPTS[id(self.receipt)]
        _gradient._ACTIVE_RECEIPTS[id(self.receipt)] = (reference, *receipt_entry[1:])
        completion_entry = _replay._ACTIVE_COMPLETIONS[id(self.completion)]
        _replay._ACTIVE_COMPLETIONS[id(self.completion)] = (reference, *completion_entry[1:])
        _gradient._ACTIVE_OWNERS.pop(id(predecessor))
        _gradient._RETIRED_OWNERS[id(predecessor)] = self._prepared_predecessor_reference
        predecessor._state = _gradient._RETIRED
        predecessor.authority = None
        predecessor.binding = None
        predecessor.receipt = None
        predecessor.completion = None
        predecessor.records = ()
        predecessor.embedding_leaves = MappingProxyType({})
        predecessor.text_only = None
        predecessor.is_selected = None
        predecessor._runtime = None
        predecessor._trusted = ()
        predecessor._prepared_reference = None
        predecessor._prepared_handoff_reference = None
        self._prepared_reference = None
        self._prepared_predecessor_reference = None

    def require(self) -> "_D4EncoderBackwardAuthorizationOwner":
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_OWNERS.get(id(self))
            if retired is not None and retired() is self:
                raise MdpStateError("MDP: encoder backward owner is retired.")
            raise MdpStateError("MDP: encoder backward owner is the exact active owner.")
        current = (self._runtime, self.authority, self.completion, self.receipt, self.carrier)
        if self._state is not _ACTIVE or any(
            actual is not expected for actual, expected in zip(current, entry[1:6], strict=True)
        ):
            raise MdpStateError("MDP: encoder backward owner retains sealed resources.")
        carrier_entry = _ACTIVE_CARRIERS.get(id(self.carrier))
        receipt_entry = _gradient._ACTIVE_RECEIPTS.get(id(self.receipt))
        completion_entry = _replay._ACTIVE_COMPLETIONS.get(id(self.completion))
        if (
            carrier_entry is None
            or carrier_entry[0]() is not self
            or carrier_entry[1] is not self.carrier
            or carrier_entry[2] is not self.authority
            or carrier_entry[3] is not self.completion
            or carrier_entry[4] is not self.receipt
            or receipt_entry is None
            or receipt_entry[0]() is not self
            or receipt_entry[1] is not self.receipt
            or self.receipt.authority is not self.authority
            or self.receipt.completion is not self.completion
            or self.receipt.exchange is not receipt_entry[4]
            or self.receipt.received_tensors is not receipt_entry[5]
            or self.receipt.received_tensors is not self.receipt.exchange.received_tensors
            or self.receipt._seal is not _gradient._RECEIPT_SEAL
            or completion_entry is None
            or completion_entry[0]() is not self
            or completion_entry[1] is not self.completion
            or completion_entry[2] is not self.authority
            or self.completion.globally_reduced_num_tokens is not completion_entry[3]
            or self.completion._seal is not _replay._COMPLETION_SEAL
            or _replay._tensor_descriptor(completion_entry[3]) != completion_entry[4]
        ):
            raise MdpStateError("MDP: encoder backward owner retains exact Gate3 capabilities.")
        carrier = self.carrier
        binding_owner, _buffers, _operations, forward_handle = entry[6]
        if type(carrier) is _D4EncoderBackwardMember:
            if (
                carrier.authority is not self.authority
                or carrier.completion is not self.completion
                or carrier.receipt is not self.receipt
                or carrier._seal is not _MEMBER_SEAL
                or type(carrier.forward_handle) is not EncoderForwardHandle
                or carrier.forward_handle._released is not False
                or carrier.forward_handle._backward_done is not False
                or carrier.forward_handle is not forward_handle
                or type(binding_owner) is not DynamicEncoderCpBinding
                or binding_owner.active is not True
                or binding_owner.membership.ranks != carrier.selected_ranks
                or carrier.routed_gradients is not self.receipt.received_tensors
                or carrier.selected_ranks is not carrier_entry[5]
                or type(carrier.member_index) is not int
                or carrier.member_index != carrier_entry[6]
                or carrier.is_leader is not carrier_entry[7]
                or carrier.forward_handle is not carrier_entry[8]
                or carrier.routed_gradients is not carrier_entry[9]
            ):
                raise MdpStateError("MDP: encoder backward member retains sealed authorization.")
        elif type(carrier) is _D4EncoderBackwardEmpty:
            if (
                carrier.authority is not self.authority
                or carrier.completion is not self.completion
                or carrier.receipt is not self.receipt
                or carrier._seal is not _EMPTY_SEAL
                or binding_owner is not None
                or forward_handle is not None
                or carrier.selected_ranks is not carrier_entry[5]
                or carrier.text_only is not carrier_entry[6]
            ):
                raise MdpStateError("MDP: encoder backward empty retains sealed authorization.")
        else:
            raise MdpStateError("MDP: encoder backward owner retains an exact carrier type.")
        return self

    def require_carrier(self, carrier: Any) -> Any:
        self.require()
        if carrier is not self.carrier:
            raise MdpStateError("MDP: encoder backward requires its exact armed carrier.")
        return carrier

    def abort(self, primary_error: BaseException | None = None) -> None:
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: encoder backward abort error is an exception.")
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        trusted = entry[1:]
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: encoder backward authorization was aborted.")
        )
        integrity_error = None
        try:
            current = tuple(
                object.__getattribute__(self, name)
                for name in ("_runtime", "authority", "completion", "receipt", "carrier")
            )
            if object.__getattribute__(self, "_state") is not _ACTIVE or any(
                actual is not expected
                for actual, expected in zip(current, trusted[:5], strict=True)
            ):
                integrity_error = MdpStateError(
                    "MDP: encoder backward owner retains sealed resources."
                )
        except BaseException as error:
            integrity_error = error
        _ACTIVE_OWNERS.pop(id(self))
        _RETIRED_OWNERS[id(self)] = weakref.ref(self)
        runtime = trusted[0]
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            del _replay._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)]
        _ACTIVE_CARRIERS.pop(id(trusted[4]), None)
        _gradient._ACTIVE_RECEIPTS.pop(id(trusted[3]), None)
        _replay._ACTIVE_COMPLETIONS.pop(id(trusted[2]), None)
        self._state = _RETIRED
        self.authority = None
        self.completion = None
        self.receipt = None
        self.carrier = None
        self._runtime = None
        self._trusted = ()
        self._prepared_reference = None
        self._prepared_predecessor_reference = None
        if integrity_error is not None:
            _add_cleanup_note(primary, "encoder backward authorization integrity failed.")
        handoff_resources, leaf_bases, transport_buffers, operations = trusted[5:9]
        binding_owner, predecessor_buffers, _predecessor_operations, forward_handle = (
            handoff_resources
        )
        if forward_handle is not None:
            try:
                forward_handle.release_forward_only()
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed encoder graph release error: {error!r}")
        if binding_owner is not None:
            try:
                binding_owner.restore(primary)
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed encoder CP restore error: {error!r}")
        for buffer in (*transport_buffers, *leaf_bases, *predecessor_buffers):
            try:
                operations.release(buffer)
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed encoder backward buffer release error: {error!r}"
                )


def run_repeated_d4_encoder_backward_authorization(
    predecessor: _gradient._D4EncoderGradientRouteOwner,
    authority: _DynamicIterationAuthority,
    completion: _replay._D4FixedDecoderCompletion,
    *,
    byte_generator=None,
) -> _D4EncoderBackwardAuthorizationOwner:
    """Authorize Gate4 and arm exactly one member or empty carrier."""
    if type(predecessor) is not _gradient._D4EncoderGradientRouteOwner:
        raise MdpConfigurationError("MDP: encoder backward uses an exact Gate3 owner.")
    entry = _gradient._ACTIVE_OWNERS.get(id(predecessor))
    if entry is None or entry[0]() is not predecessor:
        predecessor.require()
    if authority is not entry[2] or completion is not entry[5]:
        raise MdpStateError("MDP: encoder backward uses exact authority and completion.")
    candidate = None
    prepared_owner = None
    prepare_started = False

    def prepare():
        nonlocal candidate, prepared_owner, prepare_started
        if prepare_started:
            raise MdpStateError("MDP: encoder backward authorization prepares once.")
        prepare_started = True
        predecessor.require()
        trusted = _gradient._ACTIVE_OWNERS[id(predecessor)][1:]
        binding = trusted[2]
        _snapshot_local_authority(binding, authority)
        selected_ranks = _selected_ranks(authority)
        global_rank = binding.global_rank
        is_selected = global_rank in selected_ranks
        text_only = selected_ranks == ()
        handoff_resources = trusted[7]
        binding_owner, _buffers, _operations, forward_handle = handoff_resources
        if is_selected != trusted[13] or text_only != trusted[12]:
            raise MdpStateError("MDP: encoder backward role matches retained Gate3 ownership.")
        expected_keys = tuple(
            entry.key
            for entry in authority.gradient_ledger.entries
            if entry.dst_global_rank == global_rank
        )
        received = trusted[3].received_tensors
        validate_prepared_dynamic_bridge_exchange(trusted[3].exchange)
        if received is not trusted[3].exchange.received_tensors:
            raise MdpStateError("MDP: encoder backward retains exact received gradient views.")
        if tuple(received) != expected_keys:
            raise MdpPlanError("MDP: encoder backward retains exact incoming gradient routes.")
        if is_selected:
            if (
                type(forward_handle) is not EncoderForwardHandle
                or forward_handle._released is not False
                or forward_handle._backward_done is not False
                or type(binding_owner) is not DynamicEncoderCpBinding
                or binding_owner.active is not True
            ):
                raise MdpStateError("MDP: selected encoder backward retains graph and binding.")
            membership = binding_owner.membership
            if membership.ranks != selected_ranks:
                raise MdpStateError("MDP: encoder backward retains exact selected membership.")
            member_index = selected_ranks.index(global_rank)
            if member_index != 0 and received:
                raise MdpPlanError("MDP: encoder backward followers retain no routed gradients.")
            candidate = _D4EncoderBackwardMember(
                authority,
                completion,
                trusted[3],
                selected_ranks,
                member_index,
                member_index == 0,
                forward_handle,
                received,
                _MEMBER_SEAL,
            )
        else:
            if forward_handle is not None or binding_owner is not None or received:
                raise MdpStateError("MDP: empty encoder backward retains no selected resources.")
            candidate = _D4EncoderBackwardEmpty(
                authority, completion, trusted[3], selected_ranks, text_only, _EMPTY_SEAL
            )
        owner_trusted = (
            trusted[0],
            authority,
            completion,
            trusted[3],
            candidate,
            trusted[7],
            trusted[8],
            trusted[9],
            trusted[10],
        )
        prepared_owner = _D4EncoderBackwardAuthorizationOwner(owner_trusted, _OWNER_SEAL)
        prepared_owner._prepare_from(predecessor)
        prepared_owner._claim_prepared(predecessor)
        return candidate

    def authorize(value):
        if value is not candidate:
            raise MdpStateError("MDP: encoder backward Gate4 retains exact prepared carrier.")
        return value

    try:
        result = run_repeated_d4_authority_collective(
            entry[3],
            authority,
            gate_id=4,
            prepare=prepare,
            domain_collective=authorize,
            byte_generator=byte_generator,
        )
        try:
            if result is not candidate:
                raise MdpStateError("MDP: encoder backward Gate4 returns exact carrier.")
            prepared_owner.require()
            validate_prepared_dynamic_bridge_exchange(prepared_owner.receipt.exchange)
        except BaseException as error:
            raise MdpTaskFatalError(
                "MDP: encoder backward Gate4 post-WORLD result is exact."
            ) from error
        return prepared_owner
    except BaseException as error:
        try:
            if prepared_owner is not None and id(prepared_owner) in _ACTIVE_OWNERS:
                prepared_owner.abort(error)
            else:
                predecessor.abort(error)
        except BaseException as cleanup_error:
            _add_cleanup_note(
                error, f"suppressed Gate4 predecessor cleanup error: {cleanup_error!r}"
            )
        raise
