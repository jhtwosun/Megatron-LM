# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private exact Gate-4 claim and local D3 encoder backward."""

from dataclasses import dataclass, field
from typing import Any

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding import (
    _D3EncoderCompletionGateBinding,
)
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation import (
    _capture_prepared_authority,
    _PreparedD3EncoderCompletion,
)
from megatron.core.mdp.dynamic_cp_d3_producer_owner import (
    _completion_authority,
    _D3ProducerOwner,
    _PreparedNativeEncoderCompletion,
    _validate_completed_native_encoder_completion,
    _validate_prepared_native_encoder_completion,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.runtime import MdpRuntime

__all__ = ()

_PENDING_READY_SEALS: dict[object, tuple[int, ...]] = {}
_PENDING_CLAIM_SEALS: dict[object, tuple[int, ...]] = {}
_ACTIVE_CLAIMS: dict[int, tuple["_D3EncoderBackwardClaim", _D3ProducerOwner, tuple, Any]] = {}


def _outer_authority(prepared: _PreparedD3EncoderCompletion):
    authority = _capture_prepared_authority(prepared)
    if prepared._authority != authority:
        raise MdpStateError("MDP: encoder backward requires the sealed outer preparation.")
    return authority


@dataclass(frozen=True, slots=True)
class _D3EncoderBackwardClaim:
    """Sealed single-use proof of an exact successful Gate-4 claim."""

    gate_binding: _D3EncoderCompletionGateBinding = field(compare=False, repr=False)
    prepared: _PreparedD3EncoderCompletion = field(compare=False, repr=False)
    native_completion: _PreparedNativeEncoderCompletion = field(compare=False, repr=False)
    owner: _D3ProducerOwner = field(compare=False, repr=False)
    outer_authority: Any = field(compare=False, repr=False)
    authority: Any = field(compare=False, repr=False)
    receipt: Any = field(compare=False, repr=False)
    ready: Any = field(compare=False, repr=False)
    gate_tombstone: tuple = field(compare=False, repr=False)
    _authority: tuple | None = field(default=None, init=False, compare=False, repr=False)
    _state: str = field(default="fresh", init=False, compare=False, repr=False)
    _factory_seal: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        values = (
            self.gate_binding,
            self.prepared,
            self.native_completion,
            self.owner,
            self.outer_authority,
            self.authority,
            self.receipt,
            self.ready,
            self.gate_tombstone,
        )
        fingerprint = _PENDING_CLAIM_SEALS.pop(self._factory_seal, None)
        if type(self) is not _D3EncoderBackwardClaim or fingerprint != tuple(
            id(value) for value in values
        ):
            raise MdpStateError("MDP: D3 encoder backward claim is factory-minted.")


def _claim_authority(claim: _D3EncoderBackwardClaim) -> tuple:
    return (
        id(claim),
        id(claim.gate_binding),
        id(claim.prepared),
        id(claim.outer_authority),
        claim.outer_authority,
        _outer_authority(claim.prepared),
        id(claim.native_completion),
        claim.native_completion._authority,
        _completion_authority(claim.native_completion),
        id(claim.owner),
        id(claim.authority),
        id(claim.receipt),
        id(claim.ready),
        id(claim.gate_tombstone),
        tuple(reference() for reference in claim.gate_tombstone),
        claim._state,
    )


def _validate_d3_encoder_backward_claim(claim: Any) -> _D3EncoderBackwardClaim:
    """Validate an exact unconsumed Gate-4 claim without entering backward."""
    if type(claim) is not _D3EncoderBackwardClaim:
        raise MdpConfigurationError("MDP: encoder backward claim has its exact private type.")
    try:
        active = _ACTIVE_CLAIMS.get(id(claim))
        valid = (
            active is not None
            and active[0] is claim
            and active[1] is claim.owner
            and active[2] == claim._authority
            and active[3] is claim.outer_authority
            and claim._state == "fresh"
            and claim._authority == _claim_authority(claim)
        )
    except BaseException as error:
        raise MdpStateError("MDP: encoder backward claim retains valid resources.") from error
    if not valid:
        raise MdpStateError("MDP: encoder backward claim retains its exact unconsumed fresh seal.")
    _validate_prepared_native_encoder_completion(claim.native_completion, owner=claim.owner)
    _require_consumed_gate4(
        claim.gate_binding,
        authority=claim.authority,
        ready=claim.ready,
        receipt=claim.receipt,
        expected_tombstone=claim.gate_tombstone,
    )
    return claim


def _mint_backward_claim(*values: Any) -> _D3EncoderBackwardClaim:
    token = object()
    _PENDING_CLAIM_SEALS[token] = tuple(id(value) for value in values)
    try:
        claim = _D3EncoderBackwardClaim(*values, _factory_seal=token)
    except BaseException:
        _PENDING_CLAIM_SEALS.pop(token, None)
        raise
    authority = _claim_authority(claim)
    object.__setattr__(claim, "_authority", authority)
    if id(claim) in _ACTIVE_CLAIMS:
        raise MdpStateError("MDP: D3 encoder backward claim identity is already active.")
    _ACTIVE_CLAIMS[id(claim)] = (claim, claim.owner, authority, claim.outer_authority)
    return claim


@dataclass(frozen=True, slots=True)
class _D3EncoderFinalizeReady:
    """Sealed local proof that exact retained encoder backward returned."""

    prepared: _PreparedD3EncoderCompletion = field(compare=False, repr=False)
    native_completion: _PreparedNativeEncoderCompletion = field(compare=False, repr=False)
    owner: _D3ProducerOwner = field(compare=False, repr=False)
    runtime: MdpRuntime = field(compare=False, repr=False)
    handle: EncoderForwardHandle | None = field(compare=False, repr=False)
    encoder_domain: Any = field(compare=False, repr=False)
    encoder_ddp: Any = field(compare=False, repr=False)
    globally_reduced_num_tokens: Any = field(compare=False, repr=False)
    _authority: tuple | None = field(default=None, init=False, compare=False, repr=False)
    _factory_seal: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        fingerprint = _PENDING_READY_SEALS.pop(self._factory_seal, None)
        values = (
            self.prepared,
            self.native_completion,
            self.owner,
            self.runtime,
            self.handle,
            self.encoder_domain,
            self.encoder_ddp,
            self.globally_reduced_num_tokens,
        )
        if type(self) is not _D3EncoderFinalizeReady or fingerprint != tuple(
            id(value) for value in values
        ):
            raise MdpStateError("MDP: D3 encoder finalize-ready capability is factory-minted.")


def _ready_authority(ready: _D3EncoderFinalizeReady) -> tuple:
    completion = ready.native_completion
    handle = ready.handle
    return (
        id(ready),
        _outer_authority(ready.prepared),
        id(completion),
        completion._authority,
        _completion_authority(completion),
        id(ready.owner),
        id(ready.runtime),
        None if handle is None else id(handle),
        None if handle is None else handle._backward_done,
        None if handle is None else handle._released,
        id(ready.encoder_domain),
        id(ready.encoder_ddp),
        id(ready.globally_reduced_num_tokens),
    )


def _mint_finalize_ready(
    prepared: _PreparedD3EncoderCompletion,
    completion: _PreparedNativeEncoderCompletion,
    owner: _D3ProducerOwner,
    claimed_outer_authority: Any,
) -> _D3EncoderFinalizeReady:
    try:
        outer_matches = _outer_authority(prepared) == claimed_outer_authority
    except BaseException as error:
        raise MdpTaskFatalError(
            "MDP: encoder backward retained a valid claimed outer preparation."
        ) from error
    if not outer_matches:
        raise MdpTaskFatalError(
            "MDP: encoder backward retained its exact claimed outer preparation."
        )
    values = (
        prepared,
        completion,
        owner,
        completion.runtime,
        completion.handle,
        completion.encoder_domain,
        completion.encoder_ddp,
        completion.globally_reduced_num_tokens,
    )
    token = object()
    _PENDING_READY_SEALS[token] = tuple(id(value) for value in values)
    try:
        ready = _D3EncoderFinalizeReady(*values, _factory_seal=token)
    except BaseException:
        _PENDING_READY_SEALS.pop(token, None)
        raise
    object.__setattr__(ready, "_authority", _ready_authority(ready))
    return ready


def _validate_d3_encoder_finalize_ready(ready: Any) -> _D3EncoderFinalizeReady:
    """Revalidate exact local post-backward state without entering finalization."""
    if type(ready) is not _D3EncoderFinalizeReady:
        raise MdpConfigurationError("MDP: encoder finalize-ready capability has its exact type.")
    prepared, completion, owner, runtime, handle = (
        ready.prepared,
        ready.native_completion,
        ready.owner,
        ready.runtime,
        ready.handle,
    )
    if (
        type(prepared) is not _PreparedD3EncoderCompletion
        or type(completion) is not _PreparedNativeEncoderCompletion
        or type(owner) is not _D3ProducerOwner
        or type(runtime) is not MdpRuntime
    ):
        raise MdpConfigurationError("MDP: encoder finalize-ready inputs have exact private types.")
    _validate_completed_native_encoder_completion(completion, owner=owner)
    domain, ddp, token = (
        completion.encoder_domain,
        completion.encoder_ddp,
        completion.globally_reduced_num_tokens,
    )
    try:
        pre_authority = owner.producer
        completion_authority = _completion_authority(completion)
    except BaseException as error:
        raise MdpStateError(
            "MDP: finalize-ready owner retains exact post-backward resources."
        ) from error
    if (
        owner._state != "backward-complete"
        or owner._runtime is not runtime
        or owner._owned_runtime is not runtime
        or owner._prepared is not completion
        or owner._prepared_authority != completion._authority
        or runtime._handle is not handle
        or prepared.native_completion is not completion
        or getattr(prepared.producer, "owner", None) is not owner
        or getattr(prepared.producer, "pre_authority", None) is not pre_authority
        or getattr(pre_authority, "owner", None) is not owner
        or completion.owner is not owner
        or completion.runtime is not runtime
        or completion.handle is not handle
        or completion.encoder_domain is not domain
        or completion.encoder_ddp is not ddp
        or completion.globally_reduced_num_tokens is not token
        or ready.encoder_domain is not domain
        or ready.encoder_ddp is not ddp
        or ready.globally_reduced_num_tokens is not token
        or completion._authority != completion_authority
    ):
        raise MdpStateError("MDP: encoder finalize-ready state retains exact Gate-4 ownership.")
    if handle is None:
        if (
            completion.encoder_cp_follower
            or completion.gradient_views
            or completion.allocation_bases
        ):
            raise MdpStateError("MDP: empty encoder completion has no backward resources.")
    elif (
        type(handle) is not EncoderForwardHandle
        or handle._backward_done is not True
        or handle._released is not False
    ):
        raise MdpStateError("MDP: encoder finalize-ready handle completed backward exactly once.")
    try:
        authority_matches = ready._authority is not None and ready._authority == _ready_authority(
            ready
        )
    except BaseException as error:
        raise MdpStateError(
            "MDP: encoder finalize-ready capability has valid resources."
        ) from error
    if not authority_matches:
        raise MdpStateError("MDP: encoder finalize-ready capability retains its exact seal.")
    return ready


def _abort_after_claim(owner: Any, error: BaseException) -> None:
    if type(owner) is not _D3ProducerOwner:
        return
    try:
        owner.abort(error)
    except BaseException as cleanup_error:
        try:
            error.add_note(f"suppressed D3 post-claim cleanup error: {cleanup_error!r}")
        except BaseException:
            pass


def _require_consumed_gate4(gate_binding, *, authority, ready, receipt, expected_tombstone=None):
    tombstone = getattr(gate_binding, "_tombstone", None)
    try:
        valid = (
            gate_binding._state == "idle"
            and gate_binding._armed is None
            and type(tombstone) is tuple
            and len(tombstone) == 3
            and tombstone[0]() is authority
            and tombstone[1]() is ready
            and tombstone[2]() is receipt
            and (expected_tombstone is None or tombstone is expected_tombstone)
        )
    except BaseException:
        valid = False
    if not valid:
        raise MdpTaskFatalError(
            "MDP: encoder backward requires its exact idle successful Gate-4 claim."
        )
    return tombstone


def _prepare_d3_encoder_backward_claim(
    gate_binding: _D3EncoderCompletionGateBinding, prepared: _PreparedD3EncoderCompletion, /
) -> _D3EncoderBackwardClaim:
    """Consume Gate 4; the caller must resolve the retained claim once by execute or abort."""
    if (
        type(gate_binding) is not _D3EncoderCompletionGateBinding
        or type(prepared) is not _PreparedD3EncoderCompletion
    ):
        raise MdpConfigurationError("MDP: encoder backward requires exact Gate-4 inputs.")
    expected_completion = prepared.native_completion
    cleanup_owner = prepared.producer.owner
    _validate_prepared_native_encoder_completion(expected_completion, owner=cleanup_owner)
    claim = None
    claimed = False
    try:
        completion = gate_binding.claim_for_backward(prepared)
        claimed = True
        claimed_outer_authority = _outer_authority(prepared)
        claimed_authority = prepared.authority
        claimed_receipt = prepared.receipt
        claimed_ready = claimed_receipt.prepared.ready
        gate_tombstone = _require_consumed_gate4(
            gate_binding, authority=claimed_authority, ready=claimed_ready, receipt=claimed_receipt
        )
        if (
            completion is not expected_completion
            or prepared.native_completion is not expected_completion
            or type(cleanup_owner) is not _D3ProducerOwner
            or completion.owner is not cleanup_owner
            or prepared.producer.owner is not cleanup_owner
        ):
            raise MdpTaskFatalError(
                "MDP: encoder backward requires the exact claimed native completion."
            )
        _validate_prepared_native_encoder_completion(completion, owner=cleanup_owner)
        claim = _mint_backward_claim(
            gate_binding,
            prepared,
            completion,
            cleanup_owner,
            claimed_outer_authority,
            claimed_authority,
            claimed_receipt,
            claimed_ready,
            gate_tombstone,
        )
        return _validate_d3_encoder_backward_claim(claim)
    except BaseException as error:
        if not claimed:
            raise
        if claim is not None and _ACTIVE_CLAIMS.get(id(claim), (None,))[0] is claim:
            _scrub_backward_claim(claim, "failed")
        _abort_after_claim(cleanup_owner, error)
        raise MdpTaskFatalError(
            "MDP: post-claim encoder backward failure is task-fatal."
        ) from error


def _scrub_backward_claim(claim: _D3EncoderBackwardClaim, state: str) -> None:
    _ACTIVE_CLAIMS.pop(id(claim), None)
    for name in (
        "gate_binding",
        "prepared",
        "native_completion",
        "owner",
        "outer_authority",
        "authority",
        "receipt",
        "ready",
        "gate_tombstone",
        "_authority",
    ):
        object.__setattr__(claim, name, None)
    object.__setattr__(claim, "_state", state)


def _abort_d3_encoder_backward_claim(
    claim: _D3EncoderBackwardClaim, primary_error: BaseException, /
) -> None:
    """Abort one valid prepared claim while preserving its caller's error."""
    if not isinstance(primary_error, BaseException):
        raise MdpConfigurationError("MDP: encoder backward claim abort uses an exception.")
    active = _ACTIVE_CLAIMS.get(id(claim)) if type(claim) is _D3EncoderBackwardClaim else None
    if active is None or active[0] is not claim:
        _validate_d3_encoder_backward_claim(claim)
    owner = active[1]
    try:
        _validate_d3_encoder_backward_claim(claim)
    except BaseException as validation_error:
        try:
            primary_error.add_note(
                f"suppressed D3 prepared-claim validation error: {validation_error!r}"
            )
        except BaseException:
            pass
    finally:
        _scrub_backward_claim(claim, "aborted")
        _abort_after_claim(owner, primary_error)


def _execute_d3_encoder_backward_claim(
    claim: _D3EncoderBackwardClaim, /
) -> _D3EncoderFinalizeReady:
    """Execute physical encoder backward from one exact unconsumed claim."""
    active = _ACTIVE_CLAIMS.get(id(claim)) if type(claim) is _D3EncoderBackwardClaim else None
    if active is None or active[0] is not claim:
        return _validate_d3_encoder_backward_claim(claim)
    trusted_owner = active[1]
    try:
        claim = _validate_d3_encoder_backward_claim(claim)
    except BaseException as error:
        _scrub_backward_claim(claim, "failed")
        _abort_after_claim(trusted_owner, error)
        raise MdpTaskFatalError(
            "MDP: post-claim encoder backward failure is task-fatal."
        ) from error
    gate_binding, prepared, completion, owner = (
        claim.gate_binding,
        claim.prepared,
        claim.native_completion,
        claim.owner,
    )
    claimed_outer_authority, authority, receipt, ready, tombstone = (
        claim.outer_authority,
        claim.authority,
        claim.receipt,
        claim.ready,
        claim.gate_tombstone,
    )
    _ACTIVE_CLAIMS.pop(id(claim), None)
    object.__setattr__(claim, "_state", "executing")
    try:
        owner._enter_native_encoder_backward(completion)
        handle = completion.handle
        if handle is not None:
            if type(handle) is not EncoderForwardHandle:
                raise MdpTaskFatalError("MDP: encoder backward requires the exact forward handle.")
            EncoderForwardHandle.backward(handle, completion.gradient_views)
        _require_consumed_gate4(
            gate_binding,
            authority=authority,
            ready=ready,
            receipt=receipt,
            expected_tombstone=tombstone,
        )
        owner._complete_native_encoder_backward(completion)
        finalize_ready = _mint_finalize_ready(prepared, completion, owner, claimed_outer_authority)
        finalize_ready = _validate_d3_encoder_finalize_ready(finalize_ready)
        _require_consumed_gate4(
            gate_binding,
            authority=authority,
            ready=ready,
            receipt=receipt,
            expected_tombstone=tombstone,
        )
    except BaseException as error:
        _scrub_backward_claim(claim, "failed")
        _abort_after_claim(owner, error)
        raise MdpTaskFatalError(
            "MDP: post-claim encoder backward failure is task-fatal."
        ) from error
    _scrub_backward_claim(claim, "consumed")
    return finalize_ready


def _execute_d3_encoder_backward(
    gate_binding: _D3EncoderCompletionGateBinding, prepared: _PreparedD3EncoderCompletion, /
) -> _D3EncoderFinalizeReady:
    """Consume exact Gate 4, run local encoder backward, and retain all resources."""
    claim = _prepare_d3_encoder_backward_claim(gate_binding, prepared)
    return _execute_d3_encoder_backward_claim(claim)
