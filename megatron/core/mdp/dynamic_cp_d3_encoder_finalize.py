# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private physical Gate 5 and exact D3 encoder finalization."""

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.mdp.dynamic_cp_d3_coordinator import _D3GateStatusContext
from megatron.core.mdp.dynamic_cp_d3_encoder_backward import (
    _D3EncoderFinalizeReady,
    _ready_authority,
    _validate_d3_encoder_finalize_ready,
)
from megatron.core.mdp.dynamic_cp_d3_gradient_gate_binding import _validate_native_group_context
from megatron.core.mdp.dynamic_cp_d3_producer_owner import _D3ProducerOwner
from megatron.core.mdp.dynamic_cp_execution import (
    _PrecollectiveStatus,
    _run_precollective_consensus,
    _validate_precollective_timeout,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)

__all__ = ()

_SCHEMA = b"megatron.mdp.d3.encoder-finalize.v1"
_ZERO_DIGEST = bytes(16)
_INT64_MAX = 2**63 - 1
_PENDING_SEALS: dict[object, tuple[int, ...]] = {}
_PENDING_BINDING_SEALS: dict[object, tuple[int, ...]] = {}


def _digest(label: bytes, iteration: int, ranks: tuple[int, ...]) -> bytes:
    value = hashlib.blake2b(digest_size=16)
    value.update(_SCHEMA)
    value.update(label)
    value.update(iteration.to_bytes(8, "little", signed=False))
    value.update(len(ranks).to_bytes(8, "little", signed=False))
    for rank in ranks:
        value.update(rank.to_bytes(8, "little", signed=False))
    return value.digest()


@dataclass(frozen=True, slots=True)
class _PreparedD3EncoderFinalization:
    ready: Any = field(compare=False, repr=False)
    owner: Any = field(compare=False, repr=False)
    runtime: Any = field(compare=False, repr=False)
    encoder_ddp: Any = field(compare=False, repr=False)
    token: Any = field(compare=False, repr=False)
    iteration: int
    local_authority: tuple = field(compare=False, repr=False)
    _authority: tuple | None = field(default=None, init=False, compare=False, repr=False)
    _factory_seal: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        values = (self.ready, self.owner, self.runtime, self.encoder_ddp, self.token)
        if type(self) is not _PreparedD3EncoderFinalization or _PENDING_SEALS.pop(
            self._factory_seal, None
        ) != tuple(id(value) for value in values):
            raise MdpStateError("MDP: encoder finalization capability is factory-minted.")


def _prepared_authority(prepared: _PreparedD3EncoderFinalization) -> tuple:
    return (
        id(prepared.ready),
        id(prepared.owner),
        id(prepared.runtime),
        id(prepared.encoder_ddp),
        id(prepared.token),
        prepared.iteration,
        prepared.local_authority,
    )


def _scrub_ready(ready: Any) -> None:
    if type(ready) is not _D3EncoderFinalizeReady:
        return
    for name in (
        "prepared",
        "native_completion",
        "owner",
        "runtime",
        "handle",
        "encoder_domain",
        "encoder_ddp",
        "globally_reduced_num_tokens",
        "_authority",
    ):
        object.__setattr__(ready, name, None)


def _prepare(
    context: _D3GateStatusContext,
    group: Any,
    ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
) -> _PreparedD3EncoderFinalization:
    ready = context.phase_value
    ready = _validate_d3_encoder_finalize_ready(ready)
    owner = ready.owner
    if type(owner) is not _D3ProducerOwner:
        raise MdpConfigurationError("MDP: encoder finalization requires its exact producer owner.")
    local_authority = _ready_authority(ready)
    if (
        ready.prepared.authority is not context.authority
        or ready.prepared.receipt.prepared.ready is not context.ready
    ):
        raise MdpStateError("MDP: Gate 5 retains exact local Gate-4 lineage.")
    runtime = ready.runtime
    process_groups = getattr(runtime, "process_groups", None)
    if (
        getattr(process_groups, "encoder_reduction_group", None) is not group
        or getattr(process_groups, "world_group", None) is not group
        or runtime.rank_view.global_rank != global_rank
        or runtime.device != device
    ):
        raise MdpConfigurationError(
            "MDP: encoder finalization uses the runtime's exact WORLD reduction group."
        )
    runtime, ddp, token, iteration = owner._prepare_native_encoder_finalization(
        ready.native_completion
    )
    process_groups = getattr(runtime, "process_groups", None)
    if (
        getattr(process_groups, "encoder_reduction_group", None) is not group
        or getattr(process_groups, "world_group", None) is not group
        or runtime.rank_view.global_rank != global_rank
        or runtime.device != device
    ):
        raise MdpStateError("MDP: local cleanup preserves exact WORLD rank authority.")
    values = (ready, owner, runtime, ddp, token)
    seal = object()
    _PENDING_SEALS[seal] = tuple(id(value) for value in values)
    try:
        prepared = _PreparedD3EncoderFinalization(
            *values, iteration, local_authority, _factory_seal=seal
        )
        object.__setattr__(prepared, "_authority", _prepared_authority(prepared))
        _scrub_ready(ready)
        return prepared
    except BaseException:
        _PENDING_SEALS.pop(seal, None)
        raise


def _validate_prepared(
    prepared: Any, *, group: Any, global_rank: int, device: torch.device
) -> _PreparedD3EncoderFinalization:
    if type(prepared) is not _PreparedD3EncoderFinalization:
        raise MdpConfigurationError("MDP: encoder finalization has its exact prepared type.")
    owner, runtime = prepared.owner, prepared.runtime
    if (
        type(owner) is not _D3ProducerOwner
        or prepared._authority is None
        or prepared._authority != _prepared_authority(prepared)
        or runtime.rank_view.global_rank != global_rank
        or runtime.device != device
    ):
        raise MdpStateError("MDP: encoder finalization retains its exact local owner and rank.")
    process_groups = getattr(runtime, "process_groups", None)
    if (
        getattr(process_groups, "encoder_reduction_group", None) is not group
        or getattr(process_groups, "world_group", None) is not group
    ):
        raise MdpStateError("MDP: encoder finalization retains its exact WORLD group.")
    owner._validate_native_encoder_finalization(
        runtime, prepared.encoder_ddp, prepared.token, prepared.iteration
    )
    return prepared


@dataclass(frozen=True, slots=True)
class _ArmedFinalization:
    ready: _D3EncoderFinalizeReady
    prepared: _PreparedD3EncoderFinalization
    owner: _D3ProducerOwner
    digest: bytes


class _D3EncoderFinalizeBinding:
    """Own one reusable physical Gate-5 status and exact finalizer claim."""

    __slots__ = (
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
        "_tombstone",
    )

    def __init__(
        self,
        *,
        group,
        group_ranks,
        global_rank,
        device,
        timeout_seconds,
        fallback_status_gate,
        all_gather_status,
        group_ranks_getter,
        _factory_seal=None,
    ) -> None:
        values = (
            group,
            group_ranks,
            global_rank,
            device,
            timeout_seconds,
            fallback_status_gate,
            all_gather_status,
            group_ranks_getter,
        )
        if type(self) is not _D3EncoderFinalizeBinding or _PENDING_BINDING_SEALS.pop(
            _factory_seal, None
        ) != tuple(id(value) for value in values):
            raise MdpStateError("MDP: encoder finalize binding is factory-minted.")
        _validate_dependencies(
            group_ranks,
            global_rank,
            device,
            timeout_seconds,
            fallback_status_gate,
            all_gather_status,
            group_ranks_getter,
        )
        self._group, self._group_ranks = group, group_ranks
        self._global_rank, self._device = global_rank, device
        self._timeout_seconds = _validate_precollective_timeout(timeout_seconds)
        self._fallback_status_gate = fallback_status_gate
        self._all_gather_status, self._group_ranks_getter = all_gather_status, group_ranks_getter
        self._state, self._armed, self._tombstone = "idle", None, None

    @property
    def is_idle(self):
        return self._state == "idle"

    @property
    def is_armed(self):
        return self._state == "armed"

    @property
    def is_poisoned(self):
        return self._state == "poisoned"

    def status_gate(self, context: _D3GateStatusContext, local_error: BaseException | None, /):
        if type(context) is _D3GateStatusContext and context.gate_id != 5:
            if self._state != "idle":
                raise MdpStateError("MDP: non-Gate-5 status requires an idle binding.")
            return self._fallback_status_gate(context, local_error)
        if self._state != "idle":
            raise MdpStateError("MDP: encoder finalization status requires an idle binding.")
        ready = context.phase_value if type(context) is _D3GateStatusContext else None
        if self._tombstone is not None and self._tombstone is ready:
            self._state = "poisoned"
            raise MdpTaskFatalError("MDP: encoder finalization ready cannot be replayed.")
        self._state = "claimed"
        prepared = None
        error = local_error
        iteration = None
        try:
            if error is not None and not isinstance(error, BaseException):
                raise MdpConfigurationError("MDP: Gate-5 local error is an exception or None.")
            _validate_native_group_context(
                self._group, self._group_ranks, self._global_rank, self._group_ranks_getter
            )
            if type(context) is not _D3GateStatusContext or context.gate_id != 5:
                raise MdpConfigurationError(
                    "MDP: encoder finalization requires exact Gate-5 context."
                )
            if error is not None:
                if ready is not None:
                    raise MdpStateError("MDP: failed Gate-5 preparation carries no phase value.")
            elif type(ready) is _D3EncoderFinalizeReady:
                candidate_iteration = ready.owner._iteration
                if (
                    type(candidate_iteration) is not int
                    or not 0 <= candidate_iteration <= _INT64_MAX
                ):
                    raise MdpStateError(
                        "MDP: Gate-5 iteration is a nonnegative signed-int64 integer."
                    )
                iteration = candidate_iteration
                prepared = _prepare(
                    context, self._group, self._group_ranks, self._global_rank, self._device
                )
            else:
                raise MdpConfigurationError("MDP: Gate 5 requires exact finalize-ready input.")
        except BaseException as caught:
            candidate_owner = getattr(ready, "owner", None)
            if type(candidate_owner) is _D3ProducerOwner and candidate_owner._runtime is not None:
                try:
                    candidate_owner.abort(caught)
                except BaseException as cleanup_error:
                    caught.add_note(
                        f"suppressed Gate-5 preparation cleanup error: {cleanup_error!r}"
                    )
            _scrub_ready(ready)
            if error is None:
                error = caught
            elif caught is not error:
                error.add_note(f"suppressed Gate-5 local preparation error: {caught!r}")
        manifest = (
            _ZERO_DIGEST
            if iteration is None
            else _digest(b"topology", iteration, self._group_ranks)
        )
        gate = (
            _ZERO_DIGEST if prepared is None else _digest(b"gate-5", iteration, self._group_ranks)
        )
        status = _PrecollectiveStatus(self._global_rank, manifest, gate, int(error is not None), 5)
        try:
            _run_precollective_consensus(
                status,
                group_ranks=self._group_ranks,
                all_gather_status=self._all_gather_status,
                timeout_seconds=self._timeout_seconds,
            )
        except MdpBridgeError:
            self._state = "poisoned"
            self._abort(prepared)
            raise
        except MdpPlanError as caught:
            self._state = "idle"
            self._tombstone = ready if type(ready) is _D3EncoderFinalizeReady else None
            self._abort(prepared, caught)
            if error is not None and caught.__cause__ is None:
                raise caught from error
            raise
        except BaseException:
            self._state = "poisoned"
            self._abort(prepared)
            raise
        if error is not None or prepared is None:
            self._state = "poisoned"
            self._abort(prepared, error)
            raise MdpTaskFatalError("MDP: Gate-5 accepted an invalid local preparation.") from error
        self._armed = _ArmedFinalization(ready, prepared, prepared.owner, gate)
        self._state = "armed"

    @staticmethod
    def _abort(prepared, error=None, owner=None):
        candidate = owner
        if candidate is None and type(prepared) is _PreparedD3EncoderFinalization:
            candidate = prepared.owner
        if type(candidate) is _D3ProducerOwner and candidate._runtime is not None:
            try:
                candidate.abort(error)
            except BaseException as cleanup_error:
                if error is not None:
                    error.add_note(f"suppressed Gate-5 owner cleanup error: {cleanup_error!r}")

    def finalize(self, ready: _D3EncoderFinalizeReady, /) -> None:
        if self._state == "finalizing":
            self._state = "poisoned"
            raise MdpTaskFatalError("MDP: encoder finalization claim cannot be reentered.")
        armed = self._armed
        if self._state != "armed" or armed is None:
            if self._tombstone is not None and self._tombstone is ready:
                self._state = "poisoned"
                raise MdpTaskFatalError("MDP: encoder finalization claim cannot be replayed.")
            raise MdpStateError("MDP: encoder finalization requires one armed claim.")
        self._armed, self._state = None, "finalizing"
        try:
            if ready is not armed.ready:
                raise MdpTaskFatalError("MDP: encoder finalization requires its exact armed ready.")
            _validate_native_group_context(
                self._group, self._group_ranks, self._global_rank, self._group_ranks_getter
            )
            prepared = _validate_prepared(
                armed.prepared,
                group=self._group,
                global_rank=self._global_rank,
                device=self._device,
            )
            if _digest(b"gate-5", prepared.iteration, self._group_ranks) != armed.digest:
                raise MdpTaskFatalError(
                    "MDP: encoder finalization retains accepted WORLD authority."
                )
            from megatron.core.mdp import encoder

            encoder.finalize_encoder_grads(
                prepared.encoder_ddp, globally_reduced_num_tokens=prepared.token
            )
            if self._state != "finalizing" or self._armed is not None:
                raise MdpTaskFatalError(
                    "MDP: encoder finalization retained its exact one-shot claim."
                )
            prepared.owner._complete_native_encoder_finalization(
                prepared.runtime, prepared.encoder_ddp, prepared.token, prepared.iteration
            )
        except BaseException as caught:
            self._state = "poisoned"
            self._abort(armed.prepared, caught, armed.owner)
            if type(caught) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: post-Gate-5 finalization failure is task-fatal."
            ) from caught
        self._tombstone = ready
        object.__setattr__(prepared, "ready", None)
        object.__setattr__(prepared, "owner", None)
        object.__setattr__(prepared, "runtime", None)
        object.__setattr__(prepared, "encoder_ddp", None)
        object.__setattr__(prepared, "token", None)
        object.__setattr__(prepared, "local_authority", ())
        object.__setattr__(prepared, "_authority", None)
        _scrub_ready(ready)
        self._state = "idle"


def _validate_dependencies(ranks, global_rank, device, timeout, *callbacks):
    if (
        type(ranks) is not tuple
        or ranks != tuple(range(len(ranks)))
        or not ranks
        or type(global_rank) is not int
        or global_rank not in ranks
    ):
        raise MdpConfigurationError("MDP: Gate-5 ranks are the exact ordered WORLD topology.")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: Gate 5 uses an explicit CUDA device.")
    _validate_precollective_timeout(timeout)
    if not all(callable(value) for value in callbacks):
        raise MdpConfigurationError("MDP: Gate-5 dependencies are callable.")


def _make_d3_encoder_finalize_binding(
    *,
    group,
    group_ranks,
    global_rank,
    device,
    timeout_seconds,
    fallback_status_gate,
    all_gather_status,
    group_ranks_getter=dist.get_process_group_ranks,
):
    """Mint one physical Gate-5 binding over the encoder WORLD group."""
    _validate_dependencies(
        group_ranks,
        global_rank,
        device,
        timeout_seconds,
        fallback_status_gate,
        all_gather_status,
        group_ranks_getter,
    )
    values = (
        group,
        group_ranks,
        global_rank,
        device,
        timeout_seconds,
        fallback_status_gate,
        all_gather_status,
        group_ranks_getter,
    )
    seal = object()
    _PENDING_BINDING_SEALS[seal] = tuple(id(value) for value in values)
    try:
        return _D3EncoderFinalizeBinding(
            group=group,
            group_ranks=group_ranks,
            global_rank=global_rank,
            device=device,
            timeout_seconds=timeout_seconds,
            fallback_status_gate=fallback_status_gate,
            all_gather_status=all_gather_status,
            group_ranks_getter=group_ranks_getter,
            _factory_seal=seal,
        )
    except BaseException:
        _PENDING_BINDING_SEALS.pop(seal, None)
        raise
