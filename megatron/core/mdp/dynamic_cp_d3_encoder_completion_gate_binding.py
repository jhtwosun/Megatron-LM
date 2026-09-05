# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private physical gate-4 authorization for one D3 encoder completion."""

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    _dynamic_bridge_gate_authority_digest,
    build_dynamic_bridge_route_authority_digest,
)
from megatron.core.mdp.dynamic_cp_d3_coordinator import _D3GateStatusContext
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation import (
    _PreparedD3EncoderCompletion,
    _validate_prepared_d3_encoder_completion,
)
from megatron.core.mdp.dynamic_cp_d3_gradient_gate_binding import _validate_native_group_context
from megatron.core.mdp.dynamic_cp_d3_producer_owner import (
    _validate_prepared_native_encoder_completion,
)
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_execution import (
    _PrecollectiveStatus,
    _run_precollective_consensus,
    _validate_precollective_timeout,
)
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderGradientReceipt,
    DecoderReadyIteration,
    _decoder_gradient_wave_authority_digest,
    _dynamic_iteration_plan_digest,
    _DynamicIterationAuthority,
    _validate_retained_decoder_ready_iteration,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)

__all__ = ()

_INT64_MAX = 2**63 - 1
_ZERO_DIGEST = bytes(16)
_PENDING_OWNER_SEALS: dict[object, tuple[int, ...]] = {}
_PENDING_ATTEMPT_SEALS: dict[object, tuple[int, ...]] = {}


@dataclass(frozen=True, slots=True)
class _ArmedEncoderCompletion:
    authority: _DynamicIterationAuthority
    ready: DecoderReadyIteration
    prepared: _PreparedD3EncoderCompletion
    workspace: Any
    receipt: DecoderGradientReceipt
    pre_authority: Any
    native_completion: Any
    native_owner: Any
    gate_digest: bytes


@dataclass(frozen=True, slots=True)
class _D3EncoderCompletionGateAttempt:
    """Opaque result of rank-local gate-4 validation awaiting consensus."""

    _binding: Any
    _status: _PrecollectiveStatus
    _error: BaseException | None
    _authority: _DynamicIterationAuthority | None
    _ready: DecoderReadyIteration | None
    _armed: _ArmedEncoderCompletion | None
    _factory_seal: object

    def __post_init__(self) -> None:
        values = (
            self._binding,
            self._status,
            self._error,
            self._authority,
            self._ready,
            self._armed,
        )
        fingerprint = _PENDING_ATTEMPT_SEALS.pop(self._factory_seal, None)
        if type(self) is not _D3EncoderCompletionGateAttempt or fingerprint != tuple(
            id(value) for value in values
        ):
            raise MdpStateError("MDP: D3 encoder completion gate attempt is minted by its binding.")

    @property
    def status(self) -> _PrecollectiveStatus:
        """Return the immutable status row for an external consensus."""
        return self._status

    @property
    def error(self) -> BaseException | None:
        """Return the local validation error represented by the status row."""
        return self._error


class _D3EncoderCompletionGateBinding:
    """Own one reusable idle/armed/poisoned gate-4 authorization."""

    __slots__ = (
        "_workspace_owner",
        "_cp_partition_mode",
        "_group",
        "_group_ranks",
        "_global_rank",
        "_device",
        "_timeout_seconds",
        "_fallback_status_gate",
        "_all_gather_status",
        "_group_ranks_getter",
        "_state",
        "_armed",
        "_attempt",
        "_attempt_fingerprint",
        "_tombstone",
    )

    def __init__(
        self,
        *,
        workspace_owner: _D3WorkspaceBindingOwner,
        cp_partition_mode: str,
        group: Any,
        group_ranks: tuple[int, ...],
        global_rank: int,
        device: torch.device,
        timeout_seconds: float,
        fallback_status_gate: Callable[..., Any],
        all_gather_status: Callable[..., Any],
        group_ranks_getter: Callable[[Any], Any],
        _factory_seal: object | None = None,
    ) -> None:
        values = (
            workspace_owner,
            cp_partition_mode,
            group,
            group_ranks,
            global_rank,
            device,
            timeout_seconds,
            fallback_status_gate,
            all_gather_status,
            group_ranks_getter,
        )
        fingerprint = _PENDING_OWNER_SEALS.pop(_factory_seal, None)
        if type(self) is not _D3EncoderCompletionGateBinding or fingerprint != tuple(
            id(value) for value in values
        ):
            raise MdpStateError("MDP: D3 encoder completion gate binding is minted by its factory.")
        normalized_timeout = _validate_static_dependencies(*values)
        (
            self._workspace_owner,
            self._cp_partition_mode,
            self._group,
            self._group_ranks,
            self._global_rank,
            self._device,
            self._timeout_seconds,
            self._fallback_status_gate,
            self._all_gather_status,
            self._group_ranks_getter,
        ) = values
        self._timeout_seconds = normalized_timeout
        self._state = "idle"
        self._armed: _ArmedEncoderCompletion | None = None
        self._attempt: _D3EncoderCompletionGateAttempt | None = None
        self._attempt_fingerprint: tuple[Any, ...] | None = None
        self._tombstone: tuple[weakref.ReferenceType[Any], ...] | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == "idle"

    @property
    def is_armed(self) -> bool:
        return self._state == "armed"

    @property
    def is_poisoned(self) -> bool:
        return self._state == "poisoned"

    def _reject_replay(self, context: Any) -> None:
        tombstone = self._tombstone
        if tombstone is None or type(context) is not _D3GateStatusContext:
            return
        retired_authority, retired_ready, retired_receipt = (reference() for reference in tombstone)
        if retired_authority is retired_ready is retired_receipt is None:
            self._tombstone = None
            return
        phase_value = context.phase_value
        receipt = phase_value.receipt if type(phase_value) is _PreparedD3EncoderCompletion else None
        if (
            (retired_authority is not None and context.authority is retired_authority)
            or (retired_ready is not None and context.ready is retired_ready)
            or (retired_receipt is not None and receipt is retired_receipt)
        ):
            raise MdpTaskFatalError(
                "MDP: retired D3 encoder completion authority, ready, or receipt cannot be replayed."
            )

    @staticmethod
    def _record_secondary(primary: BaseException, secondary: BaseException) -> None:
        try:
            primary.add_note(
                f"suppressed D3 encoder completion gate validation error: {secondary!r}"
            )
        except BaseException:
            pass

    def _route_digest(self, authority: _DynamicIterationAuthority) -> bytes:
        return build_dynamic_bridge_route_authority_digest(
            authority.gradient_ledger,
            authority.embedding_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            width=authority.bridge_width,
            dtype=authority.bridge_dtype,
            participant_ranks=self._group_ranks,
        )

    def _validate_gate4(
        self, context: Any, local_error: BaseException | None
    ) -> tuple[
        _ArmedEncoderCompletion | None,
        bytes,
        bytes,
        BaseException | None,
        _DynamicIterationAuthority | None,
        DecoderReadyIteration | None,
    ]:
        error = local_error
        if error is not None and not isinstance(error, BaseException):
            error = MdpConfigurationError(
                "MDP: D3 encoder completion gate local error is a BaseException or None."
            )
        manifest_digest = _ZERO_DIGEST
        gate_digest = _ZERO_DIGEST
        armed = None
        authority = None
        ready = None
        try:
            if type(context) is not _D3GateStatusContext or context.gate_id != 4:
                raise MdpConfigurationError(
                    "MDP: D3 encoder completion gate requires exact gate-4 context."
                )
            authority = context.authority
            if type(authority) is not _DynamicIterationAuthority:
                raise MdpConfigurationError(
                    "MDP: D3 encoder completion gate requires exact iteration authority."
                )
            ready = context.ready
            if type(ready) is not DecoderReadyIteration:
                raise MdpConfigurationError(
                    "MDP: D3 encoder completion gate requires exact ready handoff."
                )
            ready = _validate_retained_decoder_ready_iteration(
                ready,
                global_manifest=authority.global_manifest,
                plan=authority.plan,
                global_rank=self._global_rank,
                participant_ranks=self._group_ranks,
                embedding_width=authority.bridge_width,
                embedding_dtype=authority.bridge_dtype,
                cp_partition_mode=self._cp_partition_mode,
                plan_digest=_dynamic_iteration_plan_digest(authority),
            )
            workspace = self._workspace_owner.require_workspace(authority)
            if (
                workspace.authority is not authority
                or workspace._released
                or workspace.rank != self._global_rank
                or workspace.device != self._device
                or authority.participant_ranks != self._group_ranks
            ):
                raise MdpStateError(
                    "MDP: D3 encoder completion gate retains its active workspace and rank authority."
                )
            _validate_native_group_context(
                self._group, self._group_ranks, self._global_rank, self._group_ranks_getter
            )
            manifest_digest = authority.global_manifest.digest
            if error is not None:
                if context.phase_value is not None:
                    raise MdpStateError(
                        "MDP: failed D3 encoder completion preparation carries no phase value."
                    )
                return None, manifest_digest, _ZERO_DIGEST, error, authority, ready
            prepared = context.phase_value
            if type(prepared) is not _PreparedD3EncoderCompletion:
                raise MdpConfigurationError(
                    "MDP: D3 encoder completion gate requires its exact prepared carrier."
                )
            if prepared.receipt.prepared.ready is not ready:
                raise MdpBridgeError(
                    "MDP: D3 encoder completion gate retains the exact predecessor ready handoff."
                )
            prepared = _validate_prepared_d3_encoder_completion(
                prepared,
                workspace_owner=self._workspace_owner,
                authority=authority,
                producer=prepared.producer,
                cp_partition_mode=self._cp_partition_mode,
            )
            pre_authority = prepared.producer.pre_authority
            native_owner = prepared.producer.owner
            if (
                native_owner is not pre_authority.owner
                or native_owner.producer is not pre_authority
            ):
                raise MdpStateError(
                    "MDP: D3 encoder completion gate retains exact pre-authority ownership."
                )
            native_candidate = prepared.native_completion
            native_completion = _validate_prepared_native_encoder_completion(
                native_candidate, owner=native_owner
            )
            if native_completion is not native_candidate:
                raise MdpStateError(
                    "MDP: D3 encoder completion gate validates the exact native completion."
                )
            route_digest = self._route_digest(authority)
            if prepared.receipt.prepared.exchange.route_authority_digest != route_digest:
                raise MdpBridgeError(
                    "MDP: D3 encoder completion gate receipt matches rebuilt route authority."
                )
            wave_digest = _decoder_gradient_wave_authority_digest(
                ready, prepared.receipt.iteration_nonce
            )
            gate_digest = _dynamic_bridge_gate_authority_digest(
                BridgePhase.GRADIENT, route_digest, wave_digest
            )
            armed = _ArmedEncoderCompletion(
                authority=authority,
                ready=ready,
                prepared=prepared,
                workspace=workspace,
                receipt=prepared.receipt,
                pre_authority=pre_authority,
                native_completion=native_completion,
                native_owner=native_owner,
                gate_digest=gate_digest,
            )
        except BaseException as caught:
            gate_digest = _ZERO_DIGEST
            if error is None:
                error = caught
            elif caught is not error:
                self._record_secondary(error, caught)
        return armed, manifest_digest, gate_digest, error, authority, ready

    def _retire_attempt(
        self,
        authority: _DynamicIterationAuthority | None,
        ready: DecoderReadyIteration | None,
        armed: _ArmedEncoderCompletion | None,
    ) -> None:
        if authority is None or ready is None:
            return
        receipt = None if armed is None else armed.receipt
        values = (authority, ready, receipt)
        self._tombstone = tuple(
            weakref.ref(value) if value is not None else lambda: None for value in values
        )

    @staticmethod
    def _fingerprint_attempt(attempt: _D3EncoderCompletionGateAttempt) -> tuple[Any, ...]:
        return (
            id(attempt._binding),
            id(attempt._status),
            attempt._status.to_wire_tuple(),
            id(attempt._error),
            id(attempt._authority),
            id(attempt._ready),
            id(attempt._armed),
            id(attempt._factory_seal),
        )

    def _require_active_attempt(self, attempt: Any) -> _D3EncoderCompletionGateAttempt:
        active = self._attempt
        if (
            self._state != "claimed"
            or active is None
            or attempt is not active
            or type(attempt) is not _D3EncoderCompletionGateAttempt
            or attempt._binding is not self
        ):
            raise MdpStateError(
                "MDP: D3 encoder completion gate resolution requires its active attempt."
            )
        try:
            fingerprint = self._fingerprint_attempt(attempt)
        except BaseException:
            fingerprint = None
        if fingerprint != self._attempt_fingerprint:
            self._clear_attempt()
            self._state = "poisoned"
            raise MdpTaskFatalError(
                "MDP: D3 encoder completion gate attempt retains its sealed fields."
            )
        return attempt

    def _clear_attempt(self) -> None:
        self._attempt = None
        self._attempt_fingerprint = None

    def prepare_status_attempt(
        self, context: _D3GateStatusContext, local_error: BaseException | None, /
    ) -> _D3EncoderCompletionGateAttempt:
        """Validate gate 4 locally for a caller-guarded external consensus.

        State and replay failures deliberately raise before an attempt is
        returned.  Callers must invoke this method inside their guarded local
        preparation, raise an error-bearing attempt's ``error`` from that same
        guarded scope, and resolve every returned attempt exactly once.
        """
        if self._state == "poisoned":
            raise MdpTaskFatalError(
                "MDP: poisoned D3 encoder completion gate binding cannot be reused."
            )
        if self._state == "armed":
            raise MdpStateError("MDP: D3 encoder completion gate binding is already armed.")
        if self._state != "idle":
            raise MdpStateError("MDP: D3 encoder completion gate binding is already claimed.")
        try:
            self._reject_replay(context)
        except MdpTaskFatalError:
            self._state = "poisoned"
            raise
        self._state = "claimed"
        armed, manifest_digest, gate_digest, error, authority, ready = self._validate_gate4(
            context, local_error
        )
        status = _PrecollectiveStatus(
            global_rank=self._global_rank,
            global_manifest_digest=manifest_digest,
            plan_digest=gate_digest,
            error_code=int(error is not None),
            gate_id=4,
        )
        values = (self, status, error, authority, ready, armed)
        factory_seal = object()
        _PENDING_ATTEMPT_SEALS[factory_seal] = tuple(id(value) for value in values)
        try:
            try:
                attempt = _D3EncoderCompletionGateAttempt(*values, factory_seal)
            finally:
                _PENDING_ATTEMPT_SEALS.pop(factory_seal, None)
            self._attempt = attempt
            self._attempt_fingerprint = self._fingerprint_attempt(attempt)
        except BaseException:
            self._clear_attempt()
            self._state = "poisoned"
            raise
        return attempt

    def accept_status_attempt(self, attempt: _D3EncoderCompletionGateAttempt, /) -> None:
        """Install one exact locally prepared attempt after external consensus."""
        active = self._require_active_attempt(attempt)
        try:
            if active._error is not None:
                raise MdpStateError(
                    "MDP: D3 encoder completion gate status accepted a local error."
                ) from active._error
            if active._armed is None:
                raise MdpTaskFatalError(
                    "MDP: D3 encoder completion gate status accepted no prepared carrier."
                )
            self._armed = active._armed
            self._clear_attempt()
            self._state = "armed"
        except BaseException:
            self._clear_attempt()
            self._state = "poisoned"
            raise

    def abort_status_attempt(
        self, attempt: _D3EncoderCompletionGateAttempt, error: BaseException, /
    ) -> None:
        """Resolve one failed external consensus using its actual exception."""
        active = self._require_active_attempt(attempt)
        if not isinstance(error, BaseException):
            raise MdpConfigurationError(
                "MDP: D3 encoder completion gate abort requires the caught exception."
            )
        self._clear_attempt()
        if isinstance(error, MdpPlanError):
            self._retire_attempt(active._authority, active._ready, active._armed)
            self._state = "idle"
        else:
            self._state = "poisoned"

    def status_gate(
        self, context: _D3GateStatusContext, local_error: BaseException | None, /
    ) -> None:
        """Authorize the exact local gate-4 completion through status consensus."""
        if self._state == "poisoned":
            raise MdpTaskFatalError(
                "MDP: poisoned D3 encoder completion gate binding cannot be reused."
            )
        if type(context) is _D3GateStatusContext and context.gate_id != 4:
            if self._state != "idle":
                raise MdpStateError(
                    "MDP: non-completion status requires an idle encoder completion binding."
                )
            return self._fallback_status_gate(context, local_error)
        attempt = self.prepare_status_attempt(context, local_error)
        try:
            _run_precollective_consensus(
                attempt.status,
                group_ranks=self._group_ranks,
                all_gather_status=self._all_gather_status,
                timeout_seconds=self._timeout_seconds,
            )
        except MdpBridgeError as caught:
            self.abort_status_attempt(attempt, caught)
            if attempt.error is not None and caught.__cause__ is None:
                raise caught from attempt.error
            raise
        except MdpPlanError as caught:
            self.abort_status_attempt(attempt, caught)
            if attempt.error is not None and caught.__cause__ is None:
                raise caught from attempt.error
            raise
        except BaseException as caught:
            self.abort_status_attempt(attempt, caught)
            raise
        self.accept_status_attempt(attempt)

    def claim_for_backward(self, prepared: _PreparedD3EncoderCompletion, /) -> Any:
        """Consume one authorized carrier and return its opaque native preparation."""
        if self._state == "poisoned":
            raise MdpTaskFatalError(
                "MDP: poisoned D3 encoder completion gate binding cannot be reused."
            )
        armed = self._armed
        if self._state != "armed" or armed is None:
            retired_receipt = None
            if self._tombstone is not None:
                retired_receipt = self._tombstone[2]()
            if (
                type(prepared) is _PreparedD3EncoderCompletion
                and retired_receipt is not None
                and prepared.receipt is retired_receipt
            ):
                self._state = "poisoned"
                raise MdpTaskFatalError(
                    "MDP: claimed D3 encoder completion preparation cannot be replayed."
                )
            raise MdpStateError("MDP: D3 encoder backward claim requires one armed preparation.")
        self._armed = None
        self._state = "poisoned"
        try:
            if prepared is not armed.prepared:
                raise MdpTaskFatalError(
                    "MDP: D3 encoder backward claim requires the exact armed preparation."
                )
            if (
                prepared.native_completion is not armed.native_completion
                or prepared.producer.owner is not armed.native_owner
                or prepared.producer.pre_authority is not armed.pre_authority
            ):
                raise MdpTaskFatalError(
                    "MDP: D3 encoder backward claim requires the exact armed native completion."
                )
            workspace = self._workspace_owner.require_workspace(armed.authority)
            if (
                workspace is not armed.workspace
                or workspace._released
                or workspace.authority is not armed.authority
                or type(workspace.rank) is not int
                or workspace.rank != self._global_rank
                or not isinstance(workspace.device, torch.device)
                or workspace.device != self._device
                or armed.authority.participant_ranks != self._group_ranks
            ):
                raise MdpStateError(
                    "MDP: D3 encoder backward claim retains its exact active workspace."
                )
            _validate_native_group_context(
                self._group, self._group_ranks, self._global_rank, self._group_ranks_getter
            )
            route_digest = self._route_digest(armed.authority)
            if prepared.receipt.prepared.exchange.route_authority_digest != route_digest:
                raise MdpBridgeError(
                    "MDP: D3 encoder backward claim retains rebuilt route authority."
                )
            wave_digest = _decoder_gradient_wave_authority_digest(
                armed.ready, armed.receipt.iteration_nonce
            )
            gate_digest = _dynamic_bridge_gate_authority_digest(
                BridgePhase.GRADIENT, route_digest, wave_digest
            )
            if gate_digest != armed.gate_digest:
                raise MdpBridgeError(
                    "MDP: D3 encoder backward claim retains its accepted gate authority."
                )
            if prepared.receipt.prepared.ready is not armed.ready:
                raise MdpBridgeError(
                    "MDP: D3 encoder backward claim retains its predecessor ready handoff."
                )
            carrier = _validate_prepared_d3_encoder_completion(
                prepared,
                workspace_owner=self._workspace_owner,
                authority=armed.authority,
                producer=prepared.producer,
                cp_partition_mode=self._cp_partition_mode,
            )
            if (
                carrier.producer.pre_authority is not armed.pre_authority
                or carrier.producer.owner is not armed.native_owner
                or armed.pre_authority.owner is not armed.native_owner
                or armed.native_owner.producer is not armed.pre_authority
            ):
                raise MdpBridgeError(
                    "MDP: D3 encoder backward claim retains exact pre-authority ownership."
                )
            if carrier.receipt is not armed.receipt:
                raise MdpBridgeError(
                    "MDP: D3 encoder backward claim retains its exact gradient receipt."
                )
            native_completion = _validate_prepared_native_encoder_completion(
                carrier.native_completion, owner=armed.native_owner
            )
            if (
                native_completion is not armed.native_completion
                or carrier.producer.owner is not armed.native_owner
            ):
                raise MdpBridgeError(
                    "MDP: D3 encoder backward claim retains its exact native completion owner."
                )
            retirement = tuple(
                weakref.ref(value) for value in (armed.authority, armed.ready, armed.receipt)
            )
        except BaseException as caught:
            if type(caught) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: post-status D3 encoder completion failure is task-fatal."
            ) from caught
        self._tombstone = retirement
        self._state = "idle"
        return native_completion


def _validate_static_dependencies(*values: Any) -> float:
    (
        workspace_owner,
        cp_partition_mode,
        _group,
        group_ranks,
        global_rank,
        device,
        timeout_seconds,
        *callbacks,
    ) = values
    if type(workspace_owner) is not _D3WorkspaceBindingOwner:
        raise MdpConfigurationError(
            "MDP: D3 encoder completion gate uses its exact workspace owner."
        )
    if type(cp_partition_mode) is not str or cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError(
            "MDP: D3 encoder completion gate CP partition mode is supported."
        )
    if (
        type(group_ranks) is not tuple
        or not group_ranks
        or any(type(rank) is not int or rank < 0 or rank > _INT64_MAX for rank in group_ranks)
        or len(set(group_ranks)) != len(group_ranks)
    ):
        raise MdpConfigurationError(
            "MDP: D3 encoder completion gate ranks form a unique immutable tuple."
        )
    if type(global_rank) is not int or global_rank not in group_ranks:
        raise MdpConfigurationError("MDP: D3 encoder completion gate global rank is a participant.")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: D3 encoder completion gate uses an explicit CUDA device.")
    normalized_timeout = _validate_precollective_timeout(timeout_seconds)
    if not all(callable(callback) for callback in callbacks):
        raise MdpConfigurationError("MDP: D3 encoder completion gate dependencies are callable.")
    return normalized_timeout


def _make_d3_encoder_completion_gate_binding(
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    cp_partition_mode: str,
    group: Any,
    group_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    timeout_seconds: float,
    fallback_status_gate: Callable[..., Any],
    all_gather_status: Callable[..., Any],
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
) -> _D3EncoderCompletionGateBinding:
    """Mint one reusable physical gate-4 owner for coordinator callbacks."""
    values = (
        workspace_owner,
        cp_partition_mode,
        group,
        group_ranks,
        global_rank,
        device,
        timeout_seconds,
        fallback_status_gate,
        all_gather_status,
        group_ranks_getter,
    )
    _validate_static_dependencies(*values)
    token = object()
    _PENDING_OWNER_SEALS[token] = tuple(id(value) for value in values)
    try:
        return _D3EncoderCompletionGateBinding(
            workspace_owner=workspace_owner,
            cp_partition_mode=cp_partition_mode,
            group=group,
            group_ranks=group_ranks,
            global_rank=global_rank,
            device=device,
            timeout_seconds=timeout_seconds,
            fallback_status_gate=fallback_status_gate,
            all_gather_status=all_gather_status,
            group_ranks_getter=group_ranks_getter,
            _factory_seal=token,
        )
    except BaseException:
        _PENDING_OWNER_SEALS.pop(token, None)
        raise
