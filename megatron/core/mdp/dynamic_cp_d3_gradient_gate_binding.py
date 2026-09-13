# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private nonce-bound physical gate-3 binding for D3."""

import secrets
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    _dynamic_bridge_gate_authority_digest,
    _execute_validated_dynamic_bridge_exchange,
    _validate_group_binding,
    build_dynamic_bridge_route_authority_digest,
    validate_prepared_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_d3_coordinator import _D3GateStatusContext
from megatron.core.mdp.dynamic_cp_d3_iteration_nonce import acquire_d3_iteration_nonce
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_execution import (
    _PrecollectiveStatus,
    _run_precollective_consensus,
    _validate_precollective_timeout,
)
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderGradientReceipt,
    DecoderReadyIteration,
    PreparedDecoderGradientExchange,
    _decoder_gradient_wave_authority_digest,
    _dynamic_iteration_plan_digest,
    _DynamicIterationAuthority,
    _make_decoder_gradient_receipt,
    _validate_retained_decoder_ready_iteration,
    validate_prepared_decoder_gradient_exchange,
)
from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
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


@dataclass(frozen=True, slots=True)
class _ArmedGradientGate:
    authority: _DynamicIterationAuthority
    ready: DecoderReadyIteration
    prepared: PreparedDecoderGradientExchange
    workspace: Any
    exchange: Any
    nonce: bytes


def _storage_pointer(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _views_match(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    if tuple(actual) != tuple(expected):
        return False
    for key in actual:
        left, right = actual[key], expected[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        if (
            tuple(left.shape) != tuple(right.shape)
            or left.dtype != right.dtype
            or left.device != right.device
            or _storage_pointer(left) != _storage_pointer(right)
            or left.storage_offset() != right.storage_offset()
        ):
            return False
    return True


def _validate_native_group_context(
    group: Any, group_ranks: tuple[int, ...], global_rank: int, getter: Callable[[Any], Any]
) -> None:
    """Validate native geometry when local preparation produced no exchange."""
    try:
        actual_ranks, size, local_rank = tuple(getter(group)), group.size(), group.rank()
    except Exception as error:
        raise MdpConfigurationError("MDP: D3 gradient native group query succeeds.") from error
    if (
        actual_ranks != group_ranks
        or any(type(rank) is not int for rank in actual_ranks)
        or type(size) is not int
        or size != len(group_ranks)
        or type(local_rank) is not int
        or local_rank != group_ranks.index(global_rank)
    ):
        raise MdpConfigurationError(
            "MDP: D3 gradient native group matches exact participant geometry."
        )


class _D3GradientGateBinding:
    """Own the one reusable idle/armed/poisoned gate-3 callback pair."""

    __slots__ = (
        "_workspace_owner",
        "_cp_partition_mode",
        "_group",
        "_group_ranks",
        "_global_rank",
        "_device",
        "_timeout_seconds",
        "_fallback_status_gate",
        "_nonce_status_gather_factory",
        "_nonce_byte_generator",
        "_all_gather_status",
        "_group_ranks_getter",
        "_all_to_all_single",
        "_state",
        "_armed",
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
        nonce_status_gather_factory: Callable[..., Any],
        nonce_byte_generator: Callable[[int], Any],
        all_gather_status: Callable[..., Any],
        group_ranks_getter: Callable[[Any], Any],
        all_to_all_single: Callable[..., Any],
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
            nonce_status_gather_factory,
            nonce_byte_generator,
            all_gather_status,
            group_ranks_getter,
            all_to_all_single,
        )
        fingerprint = _PENDING_OWNER_SEALS.pop(_factory_seal, None)
        if type(self) is not _D3GradientGateBinding or fingerprint != tuple(
            id(value) for value in values
        ):
            raise MdpStateError("MDP: D3 gradient gate binding is minted by its factory.")
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
            self._nonce_status_gather_factory,
            self._nonce_byte_generator,
            self._all_gather_status,
            self._group_ranks_getter,
            self._all_to_all_single,
        ) = values
        self._timeout_seconds = normalized_timeout
        self._state = "idle"
        self._armed: _ArmedGradientGate | None = None
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
        if tombstone is None:
            return
        retired = tuple(reference() for reference in tombstone)
        if all(value is None for value in retired):
            self._tombstone = None
            return
        if type(context) is not _D3GateStatusContext:
            return
        retired_authority, retired_ready, retired_prepared = retired[0], retired[2], retired[3]
        if (
            (retired_authority is not None and context.authority is retired_authority)
            or (retired_ready is not None and context.ready is retired_ready)
            or (retired_prepared is not None and context.phase_value is retired_prepared)
        ):
            raise MdpTaskFatalError(
                "MDP: consumed D3 gradient authority or ready/prepared pair cannot be replayed "
                "or re-armed."
            )

    @staticmethod
    def _record_secondary(primary: BaseException, secondary: BaseException) -> None:
        try:
            primary.add_note(f"suppressed D3 gradient gate validation error: {secondary!r}")
        except BaseException:
            pass

    def _validate_gate3(
        self, context: Any, local_error: BaseException | None, nonce: bytes
    ) -> tuple[_ArmedGradientGate | None, bytes, bytes, BaseException | None]:
        error = local_error
        if error is not None and not isinstance(error, BaseException):
            error = MdpConfigurationError(
                "MDP: D3 gradient gate local error is a BaseException or None."
            )
        manifest_digest = _ZERO_DIGEST
        gate_digest = _ZERO_DIGEST
        armed = None
        try:
            if type(context) is not _D3GateStatusContext or context.gate_id != 3:
                raise MdpConfigurationError("MDP: D3 gradient gate requires exact gate-3 context.")
            authority = context.authority
            if type(authority) is not _DynamicIterationAuthority:
                raise MdpConfigurationError(
                    "MDP: D3 gradient gate requires exact iteration authority."
                )
            ready = context.ready
            if type(ready) is not DecoderReadyIteration:
                raise MdpConfigurationError("MDP: D3 gradient gate requires exact ready handoff.")
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
                    "MDP: D3 gradient gate retains its exact active workspace and rank authority."
                )
            buffers = workspace.gradient_transport_buffers
            if type(buffers) is not tuple or len(buffers) != 2:
                raise MdpConfigurationError(
                    "MDP: D3 gradient gate workspace owns one gradient transport pair."
                )
            manifest_digest = authority.global_manifest.digest
            route_digest = self._route_digest(authority)
            wave_digest = _decoder_gradient_wave_authority_digest(ready, nonce)
            gate_digest = _dynamic_bridge_gate_authority_digest(
                BridgePhase.GRADIENT, route_digest, wave_digest
            )
            if error is not None:
                _validate_native_group_context(
                    self._group, self._group_ranks, self._global_rank, self._group_ranks_getter
                )
                if context.phase_value is not None:
                    raise MdpStateError(
                        "MDP: failed D3 gradient preparation carries no phase value."
                    )
                return None, manifest_digest, gate_digest, error
            prepared, exchange = self._validate_prepared(
                context.phase_value,
                authority=authority,
                ready=ready,
                workspace=workspace,
                route_digest=route_digest,
            )
            armed = _ArmedGradientGate(
                authority=authority,
                ready=ready,
                prepared=prepared,
                workspace=workspace,
                exchange=exchange,
                nonce=nonce,
            )
        except BaseException as caught:
            if error is None:
                error = caught
            elif caught is not error:
                self._record_secondary(error, caught)
        return armed, manifest_digest, gate_digest, error

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

    def _validate_prepared(
        self,
        value: Any,
        *,
        authority: _DynamicIterationAuthority,
        ready: DecoderReadyIteration,
        workspace: Any,
        route_digest: bytes,
    ) -> tuple[PreparedDecoderGradientExchange, Any]:
        prepared = validate_prepared_decoder_gradient_exchange(
            value,
            global_manifest=authority.global_manifest,
            plan=authority.plan,
            global_rank=self._global_rank,
            participant_ranks=self._group_ranks,
            embedding_width=authority.bridge_width,
            embedding_dtype=authority.bridge_dtype,
            cp_partition_mode=self._cp_partition_mode,
            plan_digest=_dynamic_iteration_plan_digest(authority),
        )
        if prepared.ready is not ready:
            raise MdpBridgeError(
                "MDP: D3 gradient gate preparation retains the exact ready handoff."
            )
        exchange = validate_prepared_dynamic_bridge_exchange(prepared.exchange)
        _validate_group_binding(
            exchange, group=self._group, group_ranks_getter=self._group_ranks_getter
        )
        self._validate_exchange_workspace(exchange, workspace=workspace, route_digest=route_digest)
        return prepared, exchange

    @staticmethod
    def _validate_exchange_workspace(exchange: Any, *, workspace: Any, route_digest: bytes) -> None:
        buffers = workspace.gradient_transport_buffers
        if (
            type(buffers) is not tuple
            or len(buffers) != 2
            or exchange.send_buffer is not buffers[0]
            or exchange.receive_buffer is not buffers[1]
        ):
            raise MdpBridgeError("MDP: D3 gradient gate retains exact workspace transport buffers.")
        if exchange.route_authority_digest != route_digest:
            raise MdpBridgeError("MDP: D3 gradient gate exchange matches rebuilt route authority.")
        if not _views_match(exchange.received_tensors, workspace.gradient_views):
            raise MdpBridgeError(
                "MDP: D3 gradient gate received views match exact workspace geometry."
            )

    def status_gate(
        self, context: _D3GateStatusContext, local_error: BaseException | None, /
    ) -> None:
        """Run the physical gate-3 nonce and ordinary status rendezvous."""
        if self._state == "poisoned":
            raise MdpTaskFatalError("MDP: poisoned D3 gradient gate binding cannot be reused.")
        if type(context) is _D3GateStatusContext and context.gate_id != 3:
            if self._state != "idle":
                raise MdpStateError("MDP: non-gradient status requires an idle gradient binding.")
            return self._fallback_status_gate(context, local_error)
        if self._state == "armed":
            raise MdpStateError("MDP: D3 gradient gate binding is already armed.")
        try:
            self._reject_replay(context)
        except MdpTaskFatalError:
            self._state = "poisoned"
            raise
        try:
            nonce = acquire_d3_iteration_nonce(
                group=self._group,
                group_ranks=self._group_ranks,
                global_rank=self._global_rank,
                device=self._device,
                timeout_seconds=self._timeout_seconds,
                status_gather_factory=self._nonce_status_gather_factory,
                byte_generator=self._nonce_byte_generator,
            )
        except MdpPlanError:
            self._state = "idle"
            raise
        except BaseException:
            self._state = "poisoned"
            raise
        armed, manifest_digest, gate_digest, error = self._validate_gate3(
            context, local_error, nonce
        )
        status = _PrecollectiveStatus(
            global_rank=self._global_rank,
            global_manifest_digest=manifest_digest,
            plan_digest=gate_digest,
            error_code=int(error is not None),
            gate_id=3,
        )
        try:
            _run_precollective_consensus(
                status,
                group_ranks=self._group_ranks,
                all_gather_status=self._all_gather_status,
                timeout_seconds=self._timeout_seconds,
            )
        except MdpBridgeError as caught:
            self._state = "poisoned"
            if error is not None and caught.__cause__ is None:
                raise caught from error
            raise
        except MdpPlanError as caught:
            self._state = "idle"
            if error is not None and caught.__cause__ is None:
                raise caught from error
            raise
        except BaseException:
            self._state = "poisoned"
            raise
        if error is not None:
            self._state = "poisoned"
            raise MdpStateError("MDP: D3 gradient gate status accepted a local error.") from error
        if armed is None:
            self._state = "poisoned"
            raise MdpTaskFatalError("MDP: D3 gradient gate status accepted no prepared carrier.")
        self._armed = armed
        self._state = "armed"

    def execute_gradient(
        self, prepared: PreparedDecoderGradientExchange, /
    ) -> DecoderGradientReceipt:
        """Consume the exact armed carrier and execute one reverse A2A."""
        if self._state == "poisoned":
            raise MdpTaskFatalError("MDP: poisoned D3 gradient gate binding cannot be reused.")
        armed = self._armed
        if self._state != "armed" or armed is None:
            retired_prepared = None if self._tombstone is None else self._tombstone[3]()
            if prepared is retired_prepared:
                self._state = "poisoned"
                raise MdpTaskFatalError(
                    "MDP: consumed D3 gradient gate preparation cannot be replayed."
                )
            raise MdpStateError("MDP: D3 gradient execution requires one armed preparation.")
        self._armed = None
        self._state = "poisoned"
        try:
            if prepared is not armed.prepared:
                raise MdpTaskFatalError(
                    "MDP: D3 gradient execution requires the exact armed preparation."
                )
            workspace = self._workspace_owner.require_workspace(armed.authority)
            if (
                workspace is not armed.workspace
                or workspace._released
                or workspace.authority is not armed.authority
            ):
                raise MdpStateError("MDP: D3 gradient execution retains its active workspace.")
            carrier, exchange = self._validate_prepared(
                prepared,
                authority=armed.authority,
                ready=armed.ready,
                workspace=workspace,
                route_digest=self._route_digest(armed.authority),
            )
            if carrier.ready is not armed.ready or carrier.exchange is not armed.exchange:
                raise MdpBridgeError(
                    "MDP: D3 gradient execution retains exact ready and exchange identities."
                )
            received = _execute_validated_dynamic_bridge_exchange(
                exchange, group=self._group, all_to_all_single=self._all_to_all_single
            )
            if received is not exchange.received_tensors:
                raise MdpBridgeError(
                    "MDP: D3 gradient execution returns the exact received mapping."
                )
            receipt = _make_decoder_gradient_receipt(
                prepared, received, iteration_nonce=armed.nonce
            )
            if (
                type(receipt) is not DecoderGradientReceipt
                or receipt.prepared is not prepared
                or receipt.received_tensors is not received
                or receipt.iteration_nonce != armed.nonce
            ):
                raise MdpBridgeError("MDP: D3 gradient execution seals its exact receipt.")
            retirement = tuple(
                weakref.ref(value) for value in (armed.authority, workspace, armed.ready, prepared)
            )
        except BaseException as caught:
            if type(caught) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: post-status D3 gradient failure is task-fatal."
            ) from caught
        self._tombstone = retirement
        self._state = "idle"
        return receipt


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
        raise MdpConfigurationError("MDP: D3 gradient gate uses its exact workspace owner.")
    if type(cp_partition_mode) is not str or cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError("MDP: D3 gradient gate CP partition mode is supported.")
    if (
        type(group_ranks) is not tuple
        or not group_ranks
        or any(type(rank) is not int or rank < 0 or rank > _INT64_MAX for rank in group_ranks)
        or len(set(group_ranks)) != len(group_ranks)
    ):
        raise MdpConfigurationError("MDP: D3 gradient gate ranks form a unique immutable tuple.")
    if type(global_rank) is not int or global_rank not in group_ranks:
        raise MdpConfigurationError("MDP: D3 gradient gate global rank is a participant.")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: D3 gradient gate uses an explicit CUDA device.")
    normalized_timeout = _validate_precollective_timeout(timeout_seconds)
    if not all(callable(callback) for callback in callbacks):
        raise MdpConfigurationError("MDP: D3 gradient gate dependencies are callable.")
    return normalized_timeout


def _make_d3_gradient_gate_binding(
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    cp_partition_mode: str,
    group: Any,
    group_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    timeout_seconds: float,
    fallback_status_gate: Callable[..., Any],
    nonce_status_gather_factory: Callable[..., Any] = make_precollective_status_gather,
    nonce_byte_generator: Callable[[int], Any] = secrets.token_bytes,
    all_gather_status: Callable[..., Any] = None,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> _D3GradientGateBinding:
    """Mint one reusable physical gate-3 owner for coordinator callbacks."""
    values = (
        workspace_owner,
        cp_partition_mode,
        group,
        group_ranks,
        global_rank,
        device,
        timeout_seconds,
        fallback_status_gate,
        nonce_status_gather_factory,
        nonce_byte_generator,
        all_gather_status,
        group_ranks_getter,
        all_to_all_single,
    )
    _validate_static_dependencies(*values)
    token = object()
    _PENDING_OWNER_SEALS[token] = tuple(id(value) for value in values)
    try:
        return _D3GradientGateBinding(
            workspace_owner=workspace_owner,
            cp_partition_mode=cp_partition_mode,
            group=group,
            group_ranks=group_ranks,
            global_rank=global_rank,
            device=device,
            timeout_seconds=timeout_seconds,
            fallback_status_gate=fallback_status_gate,
            nonce_status_gather_factory=nonce_status_gather_factory,
            nonce_byte_generator=nonce_byte_generator,
            all_gather_status=all_gather_status,
            group_ranks_getter=group_ranks_getter,
            all_to_all_single=all_to_all_single,
            _factory_seal=token,
        )
    except BaseException:
        _PENDING_OWNER_SEALS.pop(token, None)
        raise
