# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private Gate6 restore and encoder-gradient finalization for repeated D4."""

import weakref
from dataclasses import dataclass, field
from typing import Any

import torch

from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as _gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as _gradient
from megatron.core.mdp import dynamic_cp_d4_encoder_selected_backward as _gate5
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as _replay
from megatron.core.mdp import encoder as _encoder
from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp_bridge_transport import validate_prepared_dynamic_bridge_exchange
from megatron.core.mdp.dynamic_cp_d3_iteration_commit import _token_authority
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _snapshot_local_authority,
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.protocols import DynamicEncoderCpBinding
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState

__all__ = ()

_ACTIVE = object()
_RETIRED = object()
_OWNER_SEAL = object()
_READY_SEAL = object()
_PREPARED_SEAL = object()
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_READY: dict[int, tuple[Any, ...]] = {}
_ACTIVE_PREPARED: dict[int, tuple[Any, ...]] = {}
_ACTIVE_COMMIT_HANDOFFS: dict[int, tuple[Any, ...]] = {}
_RETIRED_COMMIT_HANDOFFS: dict[int, weakref.ReferenceType[Any]] = {}
_COMMIT_HANDOFF_SEAL = object()


def _mark_owner_retired(owner: Any) -> None:
    identity = id(owner)

    def remove(reference: weakref.ReferenceType[Any]) -> None:
        if _RETIRED_OWNERS.get(identity) is reference:
            del _RETIRED_OWNERS[identity]

    _RETIRED_OWNERS[identity] = weakref.ref(owner, remove)


def _add_cleanup_note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


def _validate_runtime_authority(runtime, token, iteration, token_descriptor) -> None:
    if (
        type(runtime) is not MdpRuntime
        or type(iteration) is not int
        or iteration < 0
        or runtime.state is not MdpRuntimeState.EMPTY
        or runtime._iteration != iteration
        or runtime._captured_num_tokens is not None
        or runtime._token_capture_count != 0
        or runtime._token_consumed is not False
        or not isinstance(token, torch.Tensor)
        or _token_authority(token) != token_descriptor
    ):
        raise MdpStateError("MDP: Gate6 retains exact runtime and decoder-token authority.")


@dataclass(frozen=True, slots=True)
class _D4EncoderOnlyCommitReady:
    """Registered post-Gate6 capability retained for Gate7 cleanup."""

    owner: Any = field(compare=False, repr=False)
    runtime: MdpRuntime = field(compare=False, repr=False)
    authority: _DynamicIterationAuthority = field(compare=False, repr=False)
    token: torch.Tensor = field(compare=False, repr=False)
    iteration: int
    token_authority: tuple = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D4EncoderOnlyCommitReady or self._seal is not _READY_SEAL:
            raise MdpConfigurationError("MDP: encoder-only commit authority is privately minted.")


@dataclass(frozen=True, slots=True)
class _PreparedD4EncoderFinalize:
    owner: Any = field(compare=False, repr=False)
    local_error: BaseException | None = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _PreparedD4EncoderFinalize or self._seal is not _PREPARED_SEAL:
            raise MdpConfigurationError("MDP: Gate6 preparation is privately minted.")


class _D4EncoderFinalizeOwner:
    """Sole resource owner spanning restored Gate6 and Gate7 cleanup."""

    __slots__ = (
        "__weakref__",
        "binding",
        "authority",
        "completion",
        "backward_completion",
        "commit_ready",
        "_runtime",
        "_trusted",
        "_state",
        "_restore_started",
        "_finalized",
    )

    def __init__(self, trusted: tuple[Any, ...], seal: object) -> None:
        if seal is not _OWNER_SEAL:
            raise MdpConfigurationError("MDP: Gate6 owner is privately minted.")
        self._runtime, self.binding, self.authority, self.completion = trusted[:4]
        self.backward_completion = trusted[5]
        self.commit_ready = None
        self._trusted = trusted
        self._state = _ACTIVE
        self._restore_started = False
        self._finalized = False

    def require(self) -> "_D4EncoderFinalizeOwner":
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_OWNERS.get(id(self))
            if retired is not None and retired() is self:
                raise MdpStateError("MDP: Gate6 encoder-finalize owner is retired.")
            raise MdpStateError("MDP: Gate6 encoder-finalize owner is exact and active.")
        trusted = entry[1:]
        if (
            self._state is not _ACTIVE
            or self._runtime is not trusted[0]
            or self.binding is not trusted[1]
            or self.authority is not trusted[2]
            or self.completion is not trusted[3]
            or self.backward_completion is not trusted[5]
            or self._restore_started is not entry[-2]
            or self._finalized is not entry[-1]
        ):
            raise MdpStateError("MDP: Gate6 encoder-finalize owner retains sealed resources.")
        _snapshot_local_authority(self.binding, self.authority)
        runtime, token, iteration, token_descriptor, operations, finalize = (
            trusted[0],
            trusted[11],
            trusted[12],
            trusted[13],
            trusted[10],
            trusted[15],
        )
        if self._finalized is False:
            _validate_runtime_authority(runtime, token, iteration, token_descriptor)
        elif not (
            runtime.state is MdpRuntimeState.EMPTY
            and runtime._iteration == iteration
            and runtime._captured_num_tokens is token
            and runtime._token_capture_count == 1
            and runtime._token_consumed is True
            and _token_authority(token) == token_descriptor
        ):
            raise MdpStateError("MDP: Gate6 owner retains consumed runtime-token authority.")
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        complete_entry = _gate5._ACTIVE_COMPLETIONS.get(id(trusted[5]))
        carrier_entry = _gate4._ACTIVE_CARRIERS.get(id(trusted[4]))
        receipt_entry = _gradient._ACTIVE_RECEIPTS.get(id(trusted[7]))
        replay_entry = _replay._ACTIVE_COMPLETIONS.get(id(trusted[3]))
        if (
            runtime_entry is None
            or runtime_entry[0] is not runtime
            or runtime_entry[1]() is not self
            or complete_entry is None
            or complete_entry[0]() is not self
            or carrier_entry is None
            or carrier_entry[0]() is not self
            or receipt_entry is None
            or receipt_entry[0]() is not self
            or replay_entry is None
            or replay_entry[0]() is not self
            or type(operations) is not _replay._forward._D4EncoderForwardOperations
            or operations._seal is not _replay._forward._OPERATIONS_SEAL
            or operations.encoder_ddp is not trusted[14]
            or operations.release is not trusted[16]
            or finalize is not _encoder.finalize_encoder_grads
        ):
            raise MdpStateError("MDP: Gate6 retains every exact migrated capability.")
        backward_completion = trusted[5]
        carrier = trusted[4]
        receipt = trusted[7]
        replay_completion = trusted[3]
        if (
            complete_entry[1] is not backward_completion
            or complete_entry[2] is not backward_completion.selected
            or complete_entry[3] is not backward_completion.text_only
            or backward_completion.authority is not trusted[2]
            or backward_completion.completion is not replay_completion
            or backward_completion.gate4_carrier is not carrier
            or backward_completion._seal is not _gate5._COMPLETE_SEAL
            or carrier_entry[1] is not carrier
            or carrier_entry[2] is not trusted[2]
            or carrier_entry[3] is not replay_completion
            or carrier_entry[4] is not receipt
            or carrier.authority is not trusted[2]
            or carrier.completion is not replay_completion
            or carrier.receipt is not receipt
            or receipt_entry[1] is not receipt
            or receipt.authority is not trusted[2]
            or receipt.completion is not replay_completion
            or receipt.exchange is not receipt_entry[4]
            or receipt.received_tensors is not receipt_entry[5]
            or receipt.received_tensors is not receipt.exchange.received_tensors
            or receipt._seal is not _gradient._RECEIPT_SEAL
            or replay_entry[1] is not replay_completion
            or replay_entry[2] is not trusted[2]
            or replay_completion.authority is not trusted[2]
            or replay_completion.globally_reduced_num_tokens is not replay_entry[3]
            or replay_completion._seal is not _replay._COMPLETION_SEAL
            or replay_completion._owner is not trusted[17]
            or _replay._tensor_descriptor(replay_entry[3]) != replay_entry[4]
        ):
            raise MdpStateError("MDP: Gate6 retains exact nested capability authority.")
        validate_prepared_dynamic_bridge_exchange(receipt.exchange)
        if type(carrier) is _gate4._D4EncoderBackwardMember:
            if (
                carrier.selected_ranks is not carrier_entry[5]
                or type(carrier.member_index) is not int
                or carrier.member_index != carrier_entry[6]
                or carrier.is_leader is not carrier_entry[7]
                or carrier.forward_handle is not carrier_entry[8]
                or carrier.routed_gradients is not carrier_entry[9]
                or carrier._seal is not _gate4._MEMBER_SEAL
            ):
                raise MdpStateError("MDP: Gate6 retains exact selected encoder role.")
            binding_owner, _buffers, resource_operations, handle = trusted[6]
            if (
                type(binding_owner) is not DynamicEncoderCpBinding
                or binding_owner.active is not False
                or resource_operations is not operations
                or handle is not carrier.forward_handle
                or type(handle) is not EncoderForwardHandle
                or handle._backward_done is not True
                or handle._released is not False
            ):
                raise MdpStateError("MDP: Gate6 retains restored selected encoder resources.")
        elif type(carrier) is _gate4._D4EncoderBackwardEmpty:
            if (
                carrier.selected_ranks is not carrier_entry[5]
                or carrier.text_only is not carrier_entry[6]
                or carrier._seal is not _gate4._EMPTY_SEAL
            ):
                raise MdpStateError("MDP: Gate6 retains exact empty encoder role.")
            binding_owner, _buffers, resource_operations, handle = trusted[6]
            if (
                binding_owner is not None
                or handle is not None
                or resource_operations is not operations
            ):
                raise MdpStateError("MDP: Gate6 empty role retains exact encoder resources.")
        else:
            raise MdpStateError("MDP: Gate6 retains exact encoder role carrier.")
        return self

    def require_commit_ready(self, ready: _D4EncoderOnlyCommitReady):
        self.require()
        entry = _ACTIVE_READY.get(id(ready))
        trusted = _ACTIVE_OWNERS[id(self)][1:]
        if (
            self._finalized is not True
            or self.commit_ready is not ready
            or entry is None
            or entry[0]() is not self
            or entry[1] is not ready
            or ready.owner is not self
            or ready.runtime is not trusted[0]
            or ready.authority is not trusted[2]
            or ready.token is not trusted[11]
            or type(ready.iteration) is not int
            or ready.iteration != trusted[12]
            or ready.token_authority != trusted[13]
            or ready._seal is not _READY_SEAL
            or ready.token_authority != _token_authority(ready.token)
        ):
            raise MdpStateError("MDP: Gate6 retains exact encoder-only commit authority.")
        if (
            ready.runtime.state is not MdpRuntimeState.EMPTY
            or ready.runtime._iteration != ready.iteration
            or ready.runtime._captured_num_tokens is not ready.token
            or ready.runtime._token_capture_count != 1
            or ready.runtime._token_consumed is not True
        ):
            raise MdpStateError("MDP: Gate6 commit authority retains consumed runtime token.")
        return ready

    def abort(self, primary_error: BaseException | None = None) -> None:
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: Gate6 abort primary is an exception.")
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        trusted = entry[1:]
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: Gate6 owner aborted.")
        )
        _ACTIVE_OWNERS.pop(id(self))
        _mark_owner_retired(self)
        for prepared_id, prepared_entry in tuple(_ACTIVE_PREPARED.items()):
            if prepared_entry[1] is self:
                _ACTIVE_PREPARED.pop(prepared_id)
        for ready_id, ready_entry in tuple(_ACTIVE_READY.items()):
            if ready_entry[0]() is self:
                _ACTIVE_READY.pop(ready_id)
                exact_ready = ready_entry[1]
                object.__setattr__(exact_ready, "owner", None)
                object.__setattr__(exact_ready, "runtime", None)
                object.__setattr__(exact_ready, "authority", None)
                object.__setattr__(exact_ready, "token", None)
                object.__setattr__(exact_ready, "token_authority", ())
        runtime = trusted[0]
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            _replay._forward._ACTIVE_RUNTIME_OWNERS.pop(id(runtime))
        if (
            runtime._captured_num_tokens is trusted[11]
            and runtime._token_capture_count == 1
            and runtime._token_consumed is True
        ):
            runtime._captured_num_tokens = None
            runtime._token_capture_count = 0
            runtime._token_consumed = False
        for registry, value in (
            (_gate5._ACTIVE_COMPLETIONS, trusted[5]),
            (_gate4._ACTIVE_CARRIERS, trusted[4]),
            (_gradient._ACTIVE_RECEIPTS, trusted[7]),
            (_replay._ACTIVE_COMPLETIONS, trusted[3]),
        ):
            capability_entry = registry.get(id(value))
            if capability_entry is not None and capability_entry[0]() is self:
                registry.pop(id(value))
        self._state = _RETIRED
        self._restore_started = True
        self._finalized = True
        self.binding = self.authority = self.completion = self.backward_completion = None
        self.commit_ready = self._runtime = None
        self._trusted = ()
        resources, leaf_bases, transport_buffers, operations = (
            trusted[6],
            trusted[8],
            trusted[9],
            trusted[10],
        )
        _binding_owner, predecessor_buffers, _old_operations, handle = resources
        if handle is not None:
            try:
                handle.release()
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed Gate6 graph release error: {error!r}")
        release = trusted[16]
        for buffer in (*transport_buffers, *leaf_bases, *predecessor_buffers):
            try:
                release(buffer)
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed Gate6 buffer release error: {error!r}")


class _D4EncoderCommitHandoff:
    """Cleaned one-shot ownership retained for a later Gate7 commit."""

    __slots__ = (
        "__weakref__",
        "runtime",
        "authority",
        "ready",
        "token",
        "iteration",
        "token_authority",
        "_state",
        "_seal",
    )

    def __init__(self, trusted: tuple[Any, ...], seal: object) -> None:
        if seal is not _COMMIT_HANDOFF_SEAL:
            raise MdpConfigurationError("MDP: Gate7 cleanup handoff is privately minted.")
        (
            self.runtime,
            self.authority,
            self.ready,
            self.token,
            self.iteration,
            self.token_authority,
        ) = trusted[:6]
        self._state = _ACTIVE
        self._seal = seal

    def require(self) -> "_D4EncoderCommitHandoff":
        entry = _ACTIVE_COMMIT_HANDOFFS.get(id(self))
        if entry is None or entry[0]() is not self:
            if id(self) in _RETIRED_COMMIT_HANDOFFS:
                raise MdpStateError("MDP: Gate7 cleanup handoff is retired.")
            raise MdpStateError("MDP: Gate7 cleanup handoff is exact and active.")
        trusted = entry[1:]
        runtime, authority, ready, token, iteration, token_authority, idle_slots = trusted
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        ready_entry = _ACTIVE_READY.get(id(ready))
        if (
            self._state is not _ACTIVE
            or self._seal is not _COMMIT_HANDOFF_SEAL
            or self.runtime is not runtime
            or self.authority is not authority
            or self.ready is not ready
            or self.token is not token
            or type(self.iteration) is not int
            or self.iteration != iteration
            or self.token_authority != token_authority
            or type(ready) is not _D4EncoderOnlyCommitReady
            or ready.owner is not self
            or ready.runtime is not runtime
            or ready.authority is not authority
            or ready.token is not token
            or ready.iteration != iteration
            or ready.token_authority != token_authority
            or ready._seal is not _READY_SEAL
            or runtime_entry is None
            or runtime_entry[0] is not runtime
            or runtime_entry[1]() is not self
            or ready_entry is None
            or ready_entry[0]() is not self
            or ready_entry[1] is not ready
            or runtime.state is not MdpRuntimeState.EMPTY
            or runtime._iteration != iteration
            or runtime._captured_num_tokens is not token
            or runtime._token_capture_count != 1
            or runtime._token_consumed is not True
            or _token_authority(token) != token_authority
            or any(registry.get(identity) is not None for registry, identity in idle_slots)
        ):
            raise MdpStateError("MDP: Gate7 cleanup handoff retains exact commit authority.")
        return self

    def abort(self, primary_error: BaseException | None = None) -> None:
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: Gate7 cleanup abort primary is an exception.")
        entry = _ACTIVE_COMMIT_HANDOFFS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        trusted = entry[1:]
        runtime, _authority, ready, token = trusted[:4]
        _ACTIVE_COMMIT_HANDOFFS.pop(id(self))
        _RETIRED_COMMIT_HANDOFFS[id(self)] = weakref.ref(self)
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            _replay._forward._ACTIVE_RUNTIME_OWNERS.pop(id(runtime))
        ready_entry = _ACTIVE_READY.get(id(ready))
        if ready_entry is not None and ready_entry[0]() is self:
            _ACTIVE_READY.pop(id(ready))
        if (
            runtime._captured_num_tokens is token
            and runtime._token_capture_count == 1
            and runtime._token_consumed is True
        ):
            runtime._captured_num_tokens = None
            runtime._token_capture_count = 0
            runtime._token_consumed = False
        self._state = _RETIRED
        self.runtime = self.authority = self.ready = self.token = None
        self.iteration = -1
        self.token_authority = ()
        self._seal = None
        object.__setattr__(ready, "owner", None)
        object.__setattr__(ready, "runtime", None)
        object.__setattr__(ready, "authority", None)
        object.__setattr__(ready, "token", None)
        object.__setattr__(ready, "token_authority", ())


def _claim_for_commit(
    owner: _D4EncoderFinalizeOwner,
    authority: _DynamicIterationAuthority,
    ready: _D4EncoderOnlyCommitReady,
) -> _D4EncoderCommitHandoff:
    """Retire Gate6 resources and preserve only exact consumed-token authority."""
    if type(owner) is not _D4EncoderFinalizeOwner:
        raise MdpConfigurationError("MDP: Gate7 cleanup requires its exact Gate6 owner.")
    owner.require_commit_ready(ready)
    if authority is not owner.authority:
        raise MdpStateError("MDP: Gate7 cleanup uses the exact Gate6 authority.")
    owner_entry = _ACTIVE_OWNERS.get(id(owner))
    ready_entry = _ACTIVE_READY.get(id(ready))
    if (
        owner_entry is None
        or owner_entry[0]() is not owner
        or ready_entry is None
        or ready_entry[0]() is not owner
    ):
        raise MdpStateError("MDP: Gate7 cleanup claims exact Gate6 registries.")
    prior = owner_entry[1:-2]
    runtime, token, iteration, token_authority = prior[0], prior[11], prior[12], prior[13]
    idle_slots = (
        (_ACTIVE_OWNERS, id(owner)),
        (_gate5._ACTIVE_COMPLETIONS, id(prior[5])),
        (_gate4._ACTIVE_CARRIERS, id(prior[4])),
        (_gradient._ACTIVE_RECEIPTS, id(prior[7])),
        (_replay._ACTIVE_COMPLETIONS, id(prior[3])),
    )
    trusted = (runtime, authority, ready, token, iteration, token_authority, idle_slots)
    handoff = _D4EncoderCommitHandoff(trusted, _COMMIT_HANDOFF_SEAL)
    reference = weakref.ref(handoff)
    runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
    capability_entries = tuple(
        (registry, value, registry.get(id(value)))
        for registry, value in (
            (_gate5._ACTIVE_COMPLETIONS, prior[5]),
            (_gate4._ACTIVE_CARRIERS, prior[4]),
            (_gradient._ACTIVE_RECEIPTS, prior[7]),
            (_replay._ACTIVE_COMPLETIONS, prior[3]),
        )
    )
    if (
        runtime_entry is None
        or runtime_entry[0] is not runtime
        or runtime_entry[1]() is not owner
        or any(
            entry is None or entry[0]() is not owner
            for _registry, _value, entry in capability_entries
        )
    ):
        raise MdpStateError("MDP: Gate7 cleanup retains every exact Gate6 capability.")
    handoff_entry = (reference, *trusted)
    _ACTIVE_COMMIT_HANDOFFS[id(handoff)] = handoff_entry
    _replay._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    _ACTIVE_READY[id(ready)] = (reference, ready)
    object.__setattr__(ready, "owner", handoff)
    for registry, value, _entry in capability_entries:
        registry.pop(id(value))
    _ACTIVE_OWNERS.pop(id(owner))
    _mark_owner_retired(owner)
    for prepared_id, prepared_entry in tuple(_ACTIVE_PREPARED.items()):
        if prepared_entry[1] is owner:
            _ACTIVE_PREPARED.pop(prepared_id)
    owner._state = _RETIRED
    owner.binding = owner.authority = owner.completion = owner.backward_completion = None
    owner.commit_ready = owner._runtime = None
    owner._trusted = ()
    resources, leaf_bases, transport_buffers = prior[6], prior[8], prior[9]
    _binding_owner, predecessor_buffers, _operations, handle = resources
    actions = []
    if handle is not None:
        actions.append(("graph", handle.release))
    actions.extend(
        ("buffer", lambda buffer=buffer: prior[16](buffer))
        for buffer in (*transport_buffers, *leaf_bases, *predecessor_buffers)
    )
    primary = None
    secondary_errors = []
    try:
        for label, callback in actions:
            try:
                callback()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    secondary_errors.append((label, error))
        if primary is not None:
            raise primary
        return handoff.require()
    except BaseException as error:
        current = _ACTIVE_COMMIT_HANDOFFS.get(id(handoff))
        if current is not handoff_entry:
            _ACTIVE_COMMIT_HANDOFFS[id(handoff)] = handoff_entry
        try:
            handoff.abort(error)
        except BaseException as cleanup_error:
            secondary_errors.append(("handoff", cleanup_error))
        for label, secondary in secondary_errors:
            _add_cleanup_note(error, f"suppressed Gate7 {label} cleanup error: {secondary!r}")
        raise


def run_repeated_d4_encoder_gradient_finalize(
    predecessor: _gate5._D4EncoderSelectedBackwardOwner,
    authority: _DynamicIterationAuthority,
    completion: _replay._D4FixedDecoderCompletion,
    *,
    byte_generator=None,
) -> _D4EncoderOnlyCommitReady:
    """Restore encoder CP at Gate6, then physically finalize encoder gradients."""
    if type(predecessor) is not _gate5._D4EncoderSelectedBackwardOwner:
        raise MdpConfigurationError("MDP: Gate6 uses an exact Gate5 owner.")
    predecessor.require()
    if authority is not predecessor.authority or completion is not predecessor.completion:
        raise MdpStateError("MDP: Gate6 uses exact authority and decoder completion.")
    binding = predecessor.binding
    successor = None
    successor_entry = None
    prepared = None
    started = False

    def prepare() -> _PreparedD4EncoderFinalize:
        nonlocal successor, successor_entry, prepared, started
        if started:
            raise MdpStateError("MDP: Gate6 preparation is one-shot.")
        started = True
        predecessor.require()
        prior_entry = _gate5._ACTIVE_OWNERS.get(id(predecessor))
        if prior_entry is None or prior_entry[0]() is not predecessor:
            raise MdpStateError("MDP: Gate6 claims its exact Gate5 registry.")
        prior = prior_entry[1:]
        runtime, token, operations = prior[0], completion.globally_reduced_num_tokens, prior[10]
        token_descriptor = _token_authority(token) if isinstance(token, torch.Tensor) else ()
        if (
            type(operations) is not _replay._forward._D4EncoderForwardOperations
            or operations._seal is not _replay._forward._OPERATIONS_SEAL
        ):
            raise MdpStateError("MDP: Gate6 retains exact runtime, token, and encoder operations.")
        iteration = runtime._iteration
        _validate_runtime_authority(runtime, token, iteration, token_descriptor)
        finalize = _encoder.finalize_encoder_grads
        trusted = (
            *prior,
            token,
            iteration,
            token_descriptor,
            operations.encoder_ddp,
            finalize,
            operations.release,
            completion._owner,
        )
        successor = _D4EncoderFinalizeOwner(trusted, _OWNER_SEAL)
        successor._restore_started = True
        reference = weakref.ref(successor)
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        complete_entry = _gate5._ACTIVE_COMPLETIONS.get(id(prior[5]))
        carrier_entry = _gate4._ACTIVE_CARRIERS.get(id(prior[4]))
        receipt_entry = _gradient._ACTIVE_RECEIPTS.get(id(prior[7]))
        replay_entry = _replay._ACTIVE_COMPLETIONS.get(id(prior[3]))
        if (
            runtime_entry is None
            or runtime_entry[1]() is not predecessor
            or complete_entry is None
            or complete_entry[0]() is not predecessor
            or carrier_entry is None
            or carrier_entry[0]() is not predecessor
            or receipt_entry is None
            or receipt_entry[0]() is not predecessor
            or replay_entry is None
            or replay_entry[0]() is not predecessor
        ):
            raise MdpStateError("MDP: Gate6 claims every exact predecessor capability.")
        successor_entry = (reference, *trusted, True, False)
        _ACTIVE_OWNERS[id(successor)] = successor_entry
        _replay._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
        _gate5._ACTIVE_COMPLETIONS[id(prior[5])] = (reference, *complete_entry[1:])
        _gate4._ACTIVE_CARRIERS[id(prior[4])] = (reference, *carrier_entry[1:])
        _gradient._ACTIVE_RECEIPTS[id(prior[7])] = (reference, *receipt_entry[1:])
        _replay._ACTIVE_COMPLETIONS[id(prior[3])] = (reference, *replay_entry[1:])
        _gate5._ACTIVE_OWNERS.pop(id(predecessor))
        _gate5._RETIRED_OWNERS[id(predecessor)] = weakref.ref(predecessor)
        predecessor._state = _gate5._RETIRED
        predecessor.binding = predecessor.authority = predecessor.completion = None
        predecessor.gate4_carrier = predecessor.backward_completion = predecessor._runtime = None
        predecessor._trusted = ()
        predecessor._backward_done = True
        local_error = None
        binding_owner = prior[6][0]
        if binding_owner is not None:
            if (
                type(binding_owner) is not DynamicEncoderCpBinding
                or binding_owner.active is not True
            ):
                local_error = MdpStateError("MDP: Gate6 restores its exact active encoder binding.")
            else:
                try:
                    binding_owner.restore()
                except BaseException as error:
                    local_error = error
        prepared = _PreparedD4EncoderFinalize(successor, local_error, _PREPARED_SEAL)
        _ACTIVE_PREPARED[id(prepared)] = (prepared, successor, local_error)
        if local_error is not None:
            raise local_error
        return prepared

    def identity(value):
        entry = _ACTIVE_PREPARED.get(id(value))
        if (
            type(value) is not _PreparedD4EncoderFinalize
            or value is not prepared
            or entry is None
            or entry[0] is not value
            or value.owner is not entry[1]
            or value.local_error is not entry[2]
            or value._seal is not _PREPARED_SEAL
        ):
            raise MdpTaskFatalError("MDP: Gate6 retains exact restored preparation.")
        value.owner.require()
        return value

    try:
        result = run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=6,
            prepare=prepare,
            domain_collective=identity,
            byte_generator=byte_generator,
        )
        try:
            identity(result)
            entry = _ACTIVE_OWNERS.get(id(successor))
            if entry is None or entry[0]() is not successor:
                raise MdpStateError("MDP: Gate6 retains exact post-WORLD owner.")
            trusted = entry[1:-2]
            token, iteration, token_descriptor, encoder_ddp, finalize = trusted[11:16]
            successor.require()
            _validate_runtime_authority(trusted[0], token, iteration, token_descriptor)
            finalize(encoder_ddp, globally_reduced_num_tokens=token)
            successor.require()
            if _token_authority(token) != token_descriptor:
                raise MdpStateError("MDP: Gate6 finalizer preserves decoder token authority.")
            runtime = trusted[0]
            runtime._captured_num_tokens = token
            runtime._token_capture_count = 1
            runtime._token_consumed = True
            ready = _D4EncoderOnlyCommitReady(
                successor, runtime, authority, token, iteration, token_descriptor, _READY_SEAL
            )
            successor.commit_ready = ready
            successor._finalized = True
            _ACTIVE_OWNERS[id(successor)] = (*entry[:-2], True, True)
            _ACTIVE_READY[id(ready)] = (weakref.ref(successor), ready)
            _ACTIVE_PREPARED.pop(id(prepared))
            return successor.require_commit_ready(ready)
        except BaseException as error:
            if type(error) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: encoder finalization failed after Gate6 final WORLD."
            ) from error
    except BaseException as error:
        target = None
        if successor is None:
            target = predecessor
        elif _ACTIVE_OWNERS.get(id(successor)) is successor_entry:
            target = successor
        elif (
            (retired := _RETIRED_OWNERS.get(id(successor))) is None or retired() is not successor
        ) and successor_entry is not None:
            _ACTIVE_OWNERS[id(successor)] = successor_entry
            target = successor
        if target is not None:
            try:
                target.abort(error)
            except BaseException as cleanup_error:
                _add_cleanup_note(error, f"suppressed Gate6 cleanup error: {cleanup_error!r}")
        raise
