# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Passive capability-transition owner for one repeated-D4 transaction."""

import weakref
from dataclasses import dataclass, field
from typing import Any

from megatron.core.mdp import dynamic_cp_d4_dynamic_decoder_replay as _dynamic_replay
from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as _gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_capture as _capture
from megatron.core.mdp import dynamic_cp_d4_encoder_execution as _execution
from megatron.core.mdp import dynamic_cp_d4_encoder_forward as _forward
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as _gradient
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient_finalize as _gate6
from megatron.core.mdp import dynamic_cp_d4_encoder_selected_backward as _gate5
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as _replay
from megatron.core.mdp import dynamic_cp_d4_native_schedule as _native
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_d4_authority_collective import _snapshot_local_authority
from megatron.core.mdp.dynamic_cp_d4_group_binding import _validate_repeated_d4_group_binding
from megatron.core.mdp.dynamic_cp_d4_source_catalog import (
    _D4SourceCatalogProjection,
    _validate_d4_source_catalog,
)
from megatron.core.mdp.dynamic_cp_execution import (
    build_decoder_global_manifest,
    validate_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

__all__ = ()

_ACTIVE, _RETIRED = object(), object()
_CAPTURE, _EXECUTION, _FORWARD, _PUBLICATION = object(), object(), object(), object()
_REPLAY, _NATIVE, _GRADIENT, _AUTHORIZED = object(), object(), object(), object()
_BACKWARD, _FINALIZED, _PENDING = object(), object(), object()
_LEASE_SEAL = object()
_FACTORY_SEALS: dict[object, tuple[int, int]] = {}


@dataclass(slots=True)
class _TransactionEscrow:
    reference: Any
    binding: Any
    projection: Any
    projection_snapshot: Any
    authority: Any
    stage: object
    capability: tuple[Any, ...]
    completion: Any
    ready: Any
    lease: Any


@dataclass(slots=True)
class _LeaseEscrow:
    identity: int
    reference: Any
    owner: Any
    owner_entry: _TransactionEscrow
    prior: Any
    expected_stage: object
    successor_stage: object


_ACTIVE_TRANSACTIONS: dict[int, _TransactionEscrow] = {}
_TRUSTED_TRANSACTIONS: dict[int, _TransactionEscrow] = {}
_RETIRED_TRANSACTIONS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_LEASES: dict[int, _LeaseEscrow] = {}


def _note(primary: BaseException, error: BaseException) -> None:
    try:
        primary.add_note(f"suppressed D4 transaction cleanup error: {error!r}")
    except BaseException:
        pass


def _abort(value: Any, primary: BaseException) -> None:
    try:
        if type(value) is _execution._D4EncoderExecutionClaim:
            value.abort()
        else:
            value.abort(primary)
    except MdpStateError as error:
        message = _retired_message(type(value))
        if type(error) is not MdpStateError or message is None or error.args != (message,):
            _note(primary, error)
    except BaseException as error:
        _note(primary, error)


def _retired_message(value_type: type) -> str | None:
    return {
        _execution._D4EncoderExecutionClaim: "MDP: D4 encoder execution claim is retired.",
        _forward._D4EncoderForwardOwner: "MDP: D4 encoder forward owner is retired.",
        _forward._D4EncoderPublicationOwner: "MDP: D4 encoder publication owner is retired.",
        _replay._D4FixedDecoderReplayOwner: "MDP: fixed decoder replay owner is retired.",
        _dynamic_replay._D4DynamicDecoderReplayOwner: (
            "MDP: dynamic decoder replay owner is retired."
        ),
        _gradient._D4EncoderGradientRouteOwner: "MDP: encoder-only gradient route owner is retired.",
        _gate4._D4EncoderBackwardAuthorizationOwner: "MDP: encoder backward owner is retired.",
        _gate5._D4EncoderSelectedBackwardOwner: "MDP: Gate5 selected backward owner is retired.",
        _gate6._D4EncoderFinalizeOwner: "MDP: Gate6 encoder-finalize owner is retired.",
    }.get(value_type)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _D4TransactionTransitionLease:
    owner: Any = field(compare=False, repr=False)
    prior: Any = field(compare=False, repr=False)
    expected_stage: object = field(compare=False, repr=False)
    successor_stage: object = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D4TransactionTransitionLease or self._seal is not _LEASE_SEAL:
            raise MdpConfigurationError("MDP: D4 transaction lease is privately minted.")

    def adopt(self, successor: Any, /) -> Any:
        """Adopt the exact result of the immediately preceding trusted adapter call."""
        entry = _require_lease(self)
        owner = entry.owner
        successor_stage = entry.successor_stage
        try:
            capability, completion, ready = _identify(successor_stage, successor, entry.prior)
        except BaseException as error:
            prior = _retire(self, entry)
            _abort(prior, error)
            raise
        try:
            _validate_capability(
                successor_stage, capability, owner.binding, owner.authority, completion, ready
            )
            owner_entry = _ACTIVE_TRANSACTIONS.get(id(owner))
            if owner_entry is not entry.owner_entry or owner_entry.reference() is not owner:
                raise MdpStateError("MDP: D4 transaction remains active through its lease.")
            if owner_entry.stage is not _PENDING or owner_entry.lease is not entry:
                raise MdpStateError("MDP: D4 transaction remains active through its lease.")
            if successor_stage is _NATIVE and capability[0] is not entry.prior:
                raise MdpStateError(
                    "MDP: native schedule completion belongs to the exact replay owner."
                )
            if successor_stage in (_GRADIENT, _AUTHORIZED, _BACKWARD, _FINALIZED):
                if completion is not owner_entry.completion:
                    raise MdpStateError(
                        "MDP: D4 transaction retains the exact decoder completion lineage."
                    )
            if _ACTIVE_LEASES.get(id(self)) is not entry:
                raise MdpStateError("MDP: D4 transaction lease remains exact and active.")
            _ACTIVE_LEASES.pop(id(self))
            owner.stage, owner.capability = successor_stage, capability
            owner.completion, owner.ready = completion, ready
            owner_entry.stage, owner_entry.capability = owner.stage, capability
            owner_entry.completion, owner_entry.ready, owner_entry.lease = completion, ready, None
            owner.require()
            _scrub_lease(entry)
            return successor
        except BaseException as error:
            reconcile_prior = successor_stage is _NATIVE
            prior = _retire(self, entry)
            _abort(prior if reconcile_prior else capability[0], error)
            raise

    def fail(self, primary: BaseException, /) -> None:
        if not isinstance(primary, BaseException):
            raise MdpConfigurationError("MDP: D4 transaction lease failure is an exception.")
        entry = _trusted_lease_entry(self)
        integrity_error = None
        try:
            _require_lease(self)
        except BaseException as error:
            integrity_error = error
        prior = _retire(self, entry)
        if integrity_error is not None:
            _note(primary, integrity_error)
        _abort(prior, primary)

    def arguments(self) -> tuple[Any, ...]:
        entry = _require_lease(self)
        if self.expected_stage is not _FINALIZED or self.successor_stage is not _RETIRED:
            raise MdpStateError("MDP: only the terminal lease retires for Gate7 commit.")
        owner = entry.owner
        ready = owner.ready
        _validate_capability(
            _FINALIZED,
            (entry.prior, ready),
            owner.binding,
            owner.authority,
            owner.completion,
            ready,
        )
        return (owner.binding, owner.authority, entry.prior, ready)

    def complete(self, result: Any, /) -> None:
        entry = _require_lease(self)
        if self.expected_stage is not _FINALIZED or self.successor_stage is not _RETIRED:
            raise MdpStateError("MDP: only the terminal lease completes Gate7 commit.")
        if result is not None:
            error = MdpStateError("MDP: Gate7 commit returns None.")
            prior = _retire(self, entry)
            _abort(prior, error)
            raise error
        try:
            entry.prior.require()
        except MdpStateError as error:
            message = _retired_message(type(entry.prior))
            if type(error) is not MdpStateError or message is None or error.args != (message,):
                prior = _retire(self, entry)
                _abort(prior, error)
                raise
        else:
            error = MdpStateError("MDP: Gate7 commit consumes the exact Gate6 owner.")
            prior = _retire(self, entry)
            _abort(prior, error)
            raise error
        _retire(self, entry)


class _D4TransactionOwner:
    __slots__ = (
        "__weakref__",
        "binding",
        "projection",
        "authority",
        "stage",
        "capability",
        "completion",
        "ready",
        "_state",
    )

    def __init__(self, capture_owner, binding, seal) -> None:
        if _FACTORY_SEALS.pop(seal, None) != (id(capture_owner), id(binding)):
            raise MdpConfigurationError("MDP: D4 transaction owner is privately minted.")
        self.binding, self.projection, self.authority = binding, None, None
        self.stage, self.capability = _CAPTURE, (capture_owner,)
        self.completion, self.ready, self._state = None, None, _ACTIVE

    def require(self):
        entry = _ACTIVE_TRANSACTIONS.get(id(self))
        if type(entry) is not _TransactionEscrow or entry.reference() is not self:
            if id(self) in _RETIRED_TRANSACTIONS:
                raise MdpStateError("MDP: D4 transaction owner is retired.")
            raise MdpStateError("MDP: D4 transaction owner is exact and active.")
        current = (
            self.binding,
            self.projection,
            self.authority,
            self.stage,
            self.capability,
            self.completion,
            self.ready,
        )
        if self._state is not _ACTIVE or any(
            actual is not expected
            for actual, expected in zip(
                current,
                (
                    entry.binding,
                    entry.projection,
                    entry.authority,
                    entry.stage,
                    entry.capability,
                    entry.completion,
                    entry.ready,
                ),
                strict=True,
            )
        ):
            raise MdpStateError("MDP: D4 transaction retains exact escrowed state.")
        if self.stage is _PENDING:
            raise MdpStateError("MDP: D4 transaction permits one active lease.")
        if self.projection is not None:
            current_projection = _projection_snapshot(self.projection, self.binding)
            if any(
                actual is not expected
                for actual, expected in zip(
                    current_projection, entry.projection_snapshot, strict=True
                )
            ):
                raise MdpStateError("MDP: D4 transaction retains exact projected source metadata.")
        if self.authority is not None:
            _validate_authority(self.binding, self.projection, self.authority)
        _validate_capability(
            self.stage, self.capability, self.binding, self.authority, self.completion, self.ready
        )
        return self

    def attach_source_catalog(self, projection, /):
        self.require()
        entry = _ACTIVE_TRANSACTIONS[id(self)]
        if self.stage is not _CAPTURE or self.projection is not None:
            raise MdpStateError("MDP: D4 transaction attaches source metadata exactly once.")
        _validate_projection(projection, self.binding)
        self.projection = projection
        entry.projection = projection
        entry.projection_snapshot = _projection_snapshot(projection, self.binding)
        return self

    def attach_authority(self, authority, /):
        self.require()
        entry = _ACTIVE_TRANSACTIONS[id(self)]
        if self.stage is not _CAPTURE or self.projection is None or self.authority is not None:
            raise MdpStateError("MDP: D4 transaction attaches joint authority exactly once.")
        _validate_authority(self.binding, self.projection, authority)
        self.authority = authority
        entry.authority = authority
        return self

    def begin_execution(self):
        return self._begin(_CAPTURE, _EXECUTION)

    def begin_forward(self):
        return self._begin(_EXECUTION, _FORWARD)

    def begin_publication(self):
        return self._begin(_FORWARD, _PUBLICATION)

    def begin_replay(self):
        return self._begin(_PUBLICATION, _REPLAY)

    def begin_native_schedule(self):
        return self._begin(_REPLAY, _NATIVE)

    def begin_gradient(self):
        return self._begin(_NATIVE, _GRADIENT)

    def begin_backward_authorization(self):
        return self._begin(_GRADIENT, _AUTHORIZED)

    def begin_backward(self):
        return self._begin(_AUTHORIZED, _BACKWARD)

    def begin_finalize(self):
        return self._begin(_BACKWARD, _FINALIZED)

    def begin_commit(self):
        return self._begin(_FINALIZED, _RETIRED)

    def _begin(self, expected, successor):
        self.require()
        entry = _ACTIVE_TRANSACTIONS[id(self)]
        if self.authority is None or self.stage is not expected:
            raise MdpStateError("MDP: D4 transaction follows exact phase order and authority.")
        prior = self.capability[0]
        lease = _D4TransactionTransitionLease(self, prior, expected, successor, _LEASE_SEAL)
        reference = weakref.ref(lease)
        lease_entry = _LeaseEscrow(id(lease), reference, self, entry, prior, expected, successor)
        _ACTIVE_LEASES[id(lease)] = lease_entry
        self.stage, self.capability = _PENDING, ()
        entry.stage, entry.capability, entry.lease = _PENDING, self.capability, lease_entry
        return lease

    def abort(self, primary: BaseException | None = None, /) -> None:
        if primary is not None and not isinstance(primary, BaseException):
            raise MdpConfigurationError("MDP: D4 transaction abort error is an exception.")
        entry = _ACTIVE_TRANSACTIONS.get(id(self))
        if type(entry) is not _TransactionEscrow or entry.reference() is not self:
            entry = _TRUSTED_TRANSACTIONS.get(id(self))
            if entry is None:
                self.require()
        lease_entry = entry.lease if type(entry.lease) is _LeaseEscrow else None
        lease = lease_entry.reference() if lease_entry is not None else None
        value = entry.capability[0] if entry.capability else lease_entry.prior
        error = (
            primary if primary is not None else MdpStateError("MDP: D4 transaction was aborted.")
        )
        if lease_entry is not None:
            if _ACTIVE_LEASES.get(lease_entry.identity) is lease_entry:
                _ACTIVE_LEASES.pop(lease_entry.identity)
            _scrub_lease(lease_entry)
        _retire_owner(self, entry)
        _abort(value, error)


def _require_lease(lease):
    entry = _ACTIVE_LEASES.get(id(lease))
    if (
        type(entry) is not _LeaseEscrow
        or entry.reference() is not lease
        or lease.owner is not entry.owner
        or lease.prior is not entry.prior
        or lease.expected_stage is not entry.expected_stage
        or lease.successor_stage is not entry.successor_stage
        or lease._seal is not _LEASE_SEAL
    ):
        raise MdpStateError("MDP: D4 transaction lease retains exact transition authority.")
    return entry


def _trusted_lease_entry(lease):
    entry = _ACTIVE_LEASES.get(id(lease))
    if type(entry) is _LeaseEscrow and entry.reference() is lease:
        return entry
    for owner_entry in _ACTIVE_TRANSACTIONS.values():
        if type(owner_entry) is not _TransactionEscrow:
            continue
        candidate = owner_entry.lease
        if type(candidate) is _LeaseEscrow and candidate.reference() is lease:
            return candidate
    raise MdpStateError("MDP: D4 transaction lease retains exact transition authority.")


def _retire_owner(owner, entry):
    if _ACTIVE_TRANSACTIONS.get(id(owner)) is entry:
        _ACTIVE_TRANSACTIONS.pop(id(owner))
    if _TRUSTED_TRANSACTIONS.get(id(owner)) is entry:
        _TRUSTED_TRANSACTIONS.pop(id(owner))
    _RETIRED_TRANSACTIONS[id(owner)] = weakref.ref(owner)
    owner._state = _RETIRED
    owner.binding = owner.projection = owner.authority = None
    owner.stage, owner.capability = None, ()
    owner.completion = owner.ready = None
    entry.binding = entry.projection = entry.projection_snapshot = entry.authority = None
    entry.stage, entry.capability = _RETIRED, ()
    entry.completion = entry.ready = entry.lease = None


def _retire(lease, entry):
    prior = entry.prior
    owner = entry.owner
    owner_entry = entry.owner_entry
    if _ACTIVE_LEASES.get(id(lease)) is entry:
        _ACTIVE_LEASES.pop(id(lease))
    _scrub_lease(entry)
    _retire_owner(owner, owner_entry)
    return prior


def _scrub_lease(entry):
    lease = entry.reference() if entry.reference is not None else None
    if type(lease) is _D4TransactionTransitionLease:
        for name in ("owner", "prior", "expected_stage", "successor_stage", "_seal"):
            object.__setattr__(lease, name, None)
    entry.identity = 0
    entry.reference = entry.owner = entry.owner_entry = entry.prior = None
    entry.expected_stage = entry.successor_stage = _RETIRED


def _identify(stage, value, prior):
    expected = {
        _EXECUTION: _execution._D4EncoderExecutionClaim,
        _FORWARD: _forward._D4EncoderForwardOwner,
        _PUBLICATION: _forward._D4EncoderPublicationOwner,
        _GRADIENT: _gradient._D4EncoderGradientRouteOwner,
        _AUTHORIZED: _gate4._D4EncoderBackwardAuthorizationOwner,
        _BACKWARD: _gate5._D4EncoderSelectedBackwardOwner,
    }
    if stage is _REPLAY:
        if type(value) is _replay._D4FixedDecoderReplayOwner:
            return (value,), None, None
        if type(value) is _dynamic_replay._D4DynamicDecoderReplayOwner:
            return (value,), None, None
        raise MdpStateError("MDP: D4 transaction receives the exact next phase owner.")
    if stage is _NATIVE:
        expected = {
            _replay._D4FixedDecoderReplayOwner: (
                _native._D4NativeScheduleSuccess,
                _replay._D4FixedDecoderCompletion,
            ),
            _dynamic_replay._D4DynamicDecoderReplayOwner: (
                _native._D4DynamicNativeScheduleSuccess,
                _dynamic_replay._D4DynamicDecoderCompletion,
            ),
        }.get(type(prior))
        if (
            expected is None
            or type(value) is not expected[0]
            or type(value.completion) is not expected[1]
        ):
            raise MdpStateError("MDP: D4 transaction receives exact native schedule success.")
        return (value.completion._owner,), value.completion, None
    if stage is _FINALIZED:
        if type(value) is not _gate6._D4EncoderOnlyCommitReady:
            raise MdpStateError("MDP: D4 transaction receives exact Gate6 commit authority.")
        return (value.owner, value), value.owner.completion, value
    if type(value) is not expected.get(stage):
        raise MdpStateError("MDP: D4 transaction receives the exact next phase owner.")
    completion = (
        getattr(value, "completion", None) if stage in (_GRADIENT, _AUTHORIZED, _BACKWARD) else None
    )
    return (value,), completion, None


def _validate_projection(value, binding):
    if type(value) is not _D4SourceCatalogProjection:
        raise MdpStateError("MDP: D4 transaction retains exact projected source metadata.")
    catalog = _validate_d4_source_catalog(value.catalog)
    group = _validate_repeated_d4_group_binding(binding)
    lane = group.world_ranks.index(group.domain_ranks[0]) // len(group.domain_ranks)
    if (
        lane >= len(catalog.entries)
        or value.local_source_manifest is not catalog.entries[lane].manifest
    ):
        raise MdpStateError("MDP: D4 transaction retains exact projected source metadata.")
    expected = build_decoder_global_manifest((value.local_source_manifest,))
    metadata = value.metadata
    if (
        type(metadata) is not DecoderMetadataGatherResult
        or metadata.global_manifest.samples != expected.samples
        or metadata.global_manifest.items != expected.items
        or metadata.global_manifest.payloads != expected.payloads
        or metadata.global_manifest.digest != expected.digest
        or dict(metadata.source_rank_by_lane) != {lane: group.domain_ranks[0]}
    ):
        raise MdpStateError("MDP: D4 transaction retains exact projected source metadata.")
    validate_decoder_global_manifest(metadata.global_manifest)
    return value


def _projection_snapshot(value, binding):
    _validate_projection(value, binding)
    catalog = value.catalog
    metadata = value.metadata
    return (
        value,
        value.catalog,
        catalog.entries,
        catalog.digest,
        value.local_source_manifest,
        metadata,
        metadata.global_manifest,
        metadata.source_rank_by_lane,
    )


def _validate_authority(binding, projection, authority):
    if type(authority) is not _DynamicIterationAuthority or projection is None:
        raise MdpStateError("MDP: D4 transaction retains exact joint authority.")
    _snapshot_local_authority(binding, authority)
    if (
        authority.global_manifest is not projection.metadata.global_manifest
        or dict(authority.source_rank_by_lane) != dict(projection.metadata.source_rank_by_lane)
        or authority.encoder_plan is None
        or authority.joint_plan_digest is None
    ):
        raise MdpStateError("MDP: D4 transaction retains exact joint authority.")
    return authority


def _validate_capability(stage, capability, binding, authority, completion, ready):
    value = capability[0]
    if stage is _CAPTURE:
        value.require()
        return
    if stage is _NATIVE:
        value.require_completion(completion)
        return
    if stage is _FINALIZED:
        value.require_commit_ready(ready)
    else:
        value.require()
    if value.authority is not authority or value.binding is not binding:
        raise MdpStateError("MDP: D4 transaction capability matches exact authority and binding.")


def _begin_d4_transaction(capture_owner, /):
    if type(capture_owner) is not _capture._D4EncoderCaptureOwner:
        raise MdpConfigurationError("MDP: D4 transaction begins with an exact capture owner.")
    capture_owner.require()
    binding = capture_owner.binding
    seal = object()
    _FACTORY_SEALS[seal] = (id(capture_owner), id(binding))
    owner = _D4TransactionOwner(capture_owner, binding, seal)
    reference = weakref.ref(owner)
    entry = _TransactionEscrow(
        reference, binding, None, None, None, _CAPTURE, owner.capability, None, None, None
    )
    _ACTIVE_TRANSACTIONS[id(owner)] = entry
    _TRUSTED_TRANSACTIONS[id(owner)] = entry
    return owner.require()
