# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Caller-buffer preparation and execution for Dynamic-CP tensor bridges.

This module validates, packs, and synchronously exchanges one embedding or
gradient phase, including its precollective gate consensus. It does not allocate
buffers, create process groups, own runtime ordering, retry, or recovery policy.
"""

import hashlib
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import (
    DynamicBridgeKey,
    DynamicBridgeLedger,
    dynamic_bridge_split_sizes,
    validate_dynamic_bridge_ledger_pair,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    _PrecollectiveStatus,
    _run_precollective_consensus,
    _validate_precollective_timeout,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderDynamicPlan
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)

_INT64_MAX = 2**63 - 1
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_AUTHORITY_DOMAIN = b"megatron.mdp.dynamic-cp.dynamic-bridge-transport"
_AUTHORITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _PreparedDynamicBridgeAuthority:
    carrier_identity: int
    route_authority_digest: bytes
    phase: BridgePhase
    dtype: torch.dtype
    global_rank: int
    participant_ranks: tuple[int, ...]
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    device: torch.device
    send_buffer_geometry: tuple[int, int, int, int]
    receive_buffer_geometry: tuple[int, int, int, int]
    receive_descriptors: tuple[tuple[DynamicBridgeKey, int, tuple[int, int], int], ...]


@dataclass(frozen=True)
class PreparedDynamicBridgeExchange:
    """One rank's sealed caller buffers and views for one bridge phase."""

    phase: BridgePhase
    dtype: torch.dtype
    global_rank: int
    participant_ranks: tuple[int, ...]
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    route_authority_digest: bytes
    send_buffer: Tensor = field(compare=False, repr=False)
    receive_buffer: Tensor = field(compare=False, repr=False)
    received_tensors: Mapping[DynamicBridgeKey, Tensor] = field(compare=False, repr=False)
    _authority: _PreparedDynamicBridgeAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


def _require_integer(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > _INT64_MAX:
        raise MdpConfigurationError(f"MDP: {name} is a non-negative signed-int64 integer.")
    return value


def _digest_integers(hasher: Any, *values: int) -> None:
    checked = tuple(_require_integer("dynamic bridge authority field", value) for value in values)
    hasher.update(struct.pack(f"<{len(checked)}q", *checked))


def _digest_bytes(hasher: Any, value: bytes) -> None:
    if type(value) is not bytes:
        raise MdpBridgeError("MDP: dynamic bridge authority digest input is bytes.")
    _digest_integers(hasher, len(value))
    hasher.update(value)


def _digest_text(hasher: Any, value: str) -> None:
    if not isinstance(value, str):
        raise MdpBridgeError("MDP: dynamic bridge authority text is a string.")
    encoded = value.encode("utf-8")
    _digest_integers(hasher, len(encoded))
    hasher.update(encoded)


def _digest_ledger(hasher: Any, ledger: DynamicBridgeLedger) -> None:
    _digest_text(hasher, ledger.phase.value)
    _digest_integers(
        hasher,
        len(ledger.participant_ranks),
        *ledger.participant_ranks,
        ledger.total_bytes,
        ledger.remote_bytes,
        len(ledger.entries),
    )
    for entry in ledger.entries:
        _digest_text(hasher, entry.phase.value)
        _digest_integers(
            hasher,
            entry.src_global_rank,
            entry.dst_global_rank,
            *entry.key.item_id.to_wire_tuple(),
            entry.key.endpoint_rank,
            entry.element_count,
            entry.plan_offset,
        )
        _digest_text(hasher, str(entry.dtype))


def _bridge_authority_digest(
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    *,
    phase: BridgePhase,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    width: int,
    dtype: torch.dtype,
    participant_ranks: tuple[int, ...],
) -> bytes:
    hasher = hashlib.blake2b(digest_size=16)
    _digest_integers(hasher, len(_AUTHORITY_DOMAIN), _AUTHORITY_SCHEMA_VERSION)
    hasher.update(_AUTHORITY_DOMAIN)
    _digest_bytes(hasher, global_manifest.digest)
    _digest_bytes(hasher, plan.digest)
    _digest_text(hasher, phase.value)
    _digest_text(hasher, str(dtype))
    _digest_integers(hasher, width, len(participant_ranks), *participant_ranks)
    _digest_integers(hasher, len(global_manifest.items))
    for item in global_manifest.items:
        item_id = item.item_id
        _digest_integers(
            hasher,
            *item_id.to_wire_tuple(),
            producer_rank_by_item[item_id],
            output_rows_by_item[item_id],
        )
    _digest_ledger(hasher, embedding_ledger)
    _digest_ledger(hasher, gradient_ledger)
    return hasher.digest()


def build_dynamic_bridge_route_authority_digest(
    ledger: DynamicBridgeLedger,
    reverse_ledger: DynamicBridgeLedger,
    *,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    width: int,
    dtype: torch.dtype,
    participant_ranks: tuple[int, ...],
) -> bytes:
    """Revalidate route authority and derive its rank-common 16-byte digest."""
    if not isinstance(ledger, DynamicBridgeLedger):
        raise MdpBridgeError("MDP: dynamic bridge authority uses a typed selected ledger.")
    if ledger.phase is BridgePhase.EMBEDDING:
        embedding_ledger, gradient_ledger = ledger, reverse_ledger
        expected_reverse_phase = BridgePhase.GRADIENT
    elif ledger.phase is BridgePhase.GRADIENT:
        embedding_ledger, gradient_ledger = reverse_ledger, ledger
        expected_reverse_phase = BridgePhase.EMBEDDING
    else:
        raise MdpBridgeError("MDP: dynamic bridge authority phase is embedding or gradient.")
    if (
        not isinstance(reverse_ledger, DynamicBridgeLedger)
        or reverse_ledger.phase is not expected_reverse_phase
    ):
        raise MdpBridgeError("MDP: dynamic bridge authority uses the exact reverse phase.")
    embedding, gradient = validate_dynamic_bridge_ledger_pair(
        embedding_ledger,
        gradient_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=width,
        dtype=dtype,
        participant_ranks=participant_ranks,
    )
    return _bridge_authority_digest(
        embedding,
        gradient,
        phase=ledger.phase,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=width,
        dtype=dtype,
        participant_ranks=participant_ranks,
    )


def _storage_pointer(tensor: Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _buffer_geometry(tensor: Tensor) -> tuple[int, int, int, int]:
    return (id(tensor), _storage_pointer(tensor), tensor.storage_offset(), tensor.numel())


def _validate_buffer(name: str, value: Any, *, dtype: torch.dtype, elements: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise MdpConfigurationError(f"MDP: dynamic bridge {name} is a tensor.")
    if value.dtype != dtype:
        raise MdpConfigurationError(
            f"MDP: dynamic bridge {name} dtype matches the selected transport dtype."
        )
    if value.dim() != 1 or not value.is_contiguous():
        raise MdpConfigurationError(
            f"MDP: dynamic bridge {name} is a contiguous one-dimensional tensor."
        )
    if value.numel() != elements:
        raise MdpConfigurationError(
            f"MDP: dynamic bridge {name} holds exactly {elements} elements."
        )
    return value


def _participant_bases(split_sizes: tuple[int, ...]) -> tuple[int, ...]:
    bases = []
    cursor = 0
    for size in split_sizes:
        bases.append(cursor)
        cursor += size
    return tuple(bases)


def _capture_authority(prepared: PreparedDynamicBridgeExchange) -> _PreparedDynamicBridgeAuthority:
    receive_offset = prepared.receive_buffer.storage_offset()
    return _PreparedDynamicBridgeAuthority(
        carrier_identity=id(prepared),
        route_authority_digest=prepared.route_authority_digest,
        phase=prepared.phase,
        dtype=prepared.dtype,
        global_rank=prepared.global_rank,
        participant_ranks=prepared.participant_ranks,
        input_split_sizes=prepared.input_split_sizes,
        output_split_sizes=prepared.output_split_sizes,
        device=prepared.send_buffer.device,
        send_buffer_geometry=_buffer_geometry(prepared.send_buffer),
        receive_buffer_geometry=_buffer_geometry(prepared.receive_buffer),
        receive_descriptors=tuple(
            (key, id(tensor), tuple(tensor.shape), tensor.storage_offset() - receive_offset)
            for key, tensor in prepared.received_tensors.items()
        ),
    )


def _validate_split_sizes(name: str, value: Any, participant_count: int) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != participant_count:
        raise MdpBridgeError(f"MDP: prepared dynamic bridge {name} covers every participant.")
    total = 0
    for size in value:
        size = _require_integer(f"prepared dynamic bridge {name} entry", size)
        if size > _INT64_MAX - total:
            raise MdpBridgeError(f"MDP: prepared dynamic bridge {name} total fits signed int64.")
        total += size
    return value


def validate_prepared_dynamic_bridge_exchange(prepared: Any) -> PreparedDynamicBridgeExchange:
    """Validate an exact sealed preparation carrier without external authority."""
    if type(prepared) is not PreparedDynamicBridgeExchange:
        raise MdpBridgeError("MDP: dynamic bridge requires its exact prepared carrier type.")
    if type(prepared._authority) is not _PreparedDynamicBridgeAuthority:
        raise MdpBridgeError("MDP: dynamic bridge requires its sealed authority snapshot.")
    if prepared.phase not in (BridgePhase.EMBEDDING, BridgePhase.GRADIENT):
        raise MdpBridgeError("MDP: prepared dynamic bridge phase is embedding or gradient.")
    if not isinstance(prepared.dtype, torch.dtype):
        raise MdpConfigurationError("MDP: prepared dynamic bridge dtype is a torch dtype.")
    if (
        type(prepared.route_authority_digest) is not bytes
        or len(prepared.route_authority_digest) != 16
    ):
        raise MdpBridgeError("MDP: prepared dynamic bridge authority digest is exactly 16 bytes.")
    participants = prepared.participant_ranks
    if not isinstance(participants, tuple) or not participants:
        raise MdpConfigurationError("MDP: prepared dynamic bridge participants are immutable.")
    for rank in participants:
        _require_integer("prepared dynamic bridge participant", rank)
    if len(set(participants)) != len(participants):
        raise MdpConfigurationError("MDP: prepared dynamic bridge participants are unique.")
    rank = _require_integer("prepared dynamic bridge global rank", prepared.global_rank)
    if rank not in participants:
        raise MdpConfigurationError("MDP: prepared dynamic bridge rank is a participant.")
    inputs = _validate_split_sizes(
        "input split sizes", prepared.input_split_sizes, len(participants)
    )
    outputs = _validate_split_sizes(
        "output split sizes", prepared.output_split_sizes, len(participants)
    )
    send = _validate_buffer(
        "send buffer", prepared.send_buffer, dtype=prepared.dtype, elements=sum(inputs)
    )
    receive = _validate_buffer(
        "receive buffer", prepared.receive_buffer, dtype=prepared.dtype, elements=sum(outputs)
    )
    if send.device != receive.device:
        raise MdpConfigurationError("MDP: dynamic bridge buffers use the same device.")
    if send.numel() and receive.numel() and _storage_pointer(send) == _storage_pointer(receive):
        raise MdpConfigurationError("MDP: dynamic bridge buffers use disjoint storage.")
    if type(prepared.received_tensors) is not _MAPPING_PROXY_TYPE:
        raise MdpBridgeError("MDP: dynamic bridge receive views form an immutable mapping.")

    receive_start = receive.storage_offset()
    receive_end = receive_start + receive.numel()
    intervals = []
    for key, tensor in prepared.received_tensors.items():
        if not isinstance(key, DynamicBridgeKey) or not isinstance(tensor, Tensor):
            raise MdpBridgeError("MDP: dynamic bridge receive views use typed keys and tensors.")
        if (
            tensor.dtype != prepared.dtype
            or tensor.device != receive.device
            or not tensor.is_contiguous()
            or tensor.dim() != 2
            or tensor.numel() <= 0
            or _storage_pointer(tensor) != _storage_pointer(receive)
        ):
            raise MdpBridgeError(
                "MDP: dynamic bridge receive views match buffer dtype, device, and storage."
            )
        start = tensor.storage_offset()
        end = start + tensor.numel()
        if start < receive_start or end > receive_end:
            raise MdpBridgeError("MDP: dynamic bridge receive views stay within their buffer.")
        intervals.append((start, end))
    cursor = receive_start
    for start, end in sorted(intervals):
        if start != cursor:
            raise MdpBridgeError("MDP: dynamic bridge receive views partition their buffer.")
        cursor = end
    if cursor != receive_end:
        raise MdpBridgeError("MDP: dynamic bridge receive views partition their buffer.")
    if _capture_authority(prepared) != prepared._authority:
        raise MdpBridgeError("MDP: dynamic bridge public geometry matches its authority snapshot.")
    return prepared


def route_authority_digest(prepared: Any) -> bytes:
    """Return the validated common route digest; tensor contents are not sealed."""
    return validate_prepared_dynamic_bridge_exchange(prepared).route_authority_digest


def _validate_group_binding(
    prepared: PreparedDynamicBridgeExchange, *, group: Any, group_ranks_getter: Callable[[Any], Any]
) -> None:
    try:
        actual_ranks = tuple(group_ranks_getter(group))
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: dynamic bridge native group rank query succeeded."
        ) from error
    if actual_ranks != prepared.participant_ranks or any(
        type(rank) is not int for rank in actual_ranks
    ):
        raise MdpConfigurationError(
            "MDP: dynamic bridge native group uses the exact participant rank order."
        )
    try:
        size_query = getattr(group, "size")
        local_rank_query = getattr(group, "rank")
        if not callable(size_query) or not callable(local_rank_query):
            raise TypeError("native group size and rank queries are callable")
        group_size = size_query()
        local_rank = local_rank_query()
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: dynamic bridge native group geometry query succeeded."
        ) from error
    if type(group_size) is not int or group_size != len(prepared.participant_ranks):
        raise MdpConfigurationError(
            "MDP: dynamic bridge native group size matches participant count."
        )
    expected_local_rank = prepared.participant_ranks.index(prepared.global_rank)
    if type(local_rank) is not int or local_rank != expected_local_rank:
        raise MdpConfigurationError(
            "MDP: dynamic bridge native group local rank matches the global rank."
        )


def execute_dynamic_bridge_exchange(
    prepared: PreparedDynamicBridgeExchange,
    *,
    group: Any,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> Mapping[DynamicBridgeKey, Tensor]:
    """Execute one sealed synchronous embedding or gradient all-to-all.

    The prepared participant order must exactly match the native order returned
    by ``group_ranks_getter``. Every participant calls the collective once,
    including ranks with all-zero splits. This function provides no failure
    consensus, retry, or recovery.

    Args:
        prepared: Sealed caller-owned buffers and immutable receive views.
        group: Native process group whose ordered ranks match the carrier.
        group_ranks_getter: Injectable native ordered-rank query.
        all_to_all_single: Injectable synchronous collective implementation.

    Returns:
        The exact immutable receive-view mapping stored in ``prepared``.
    """
    carrier = validate_prepared_dynamic_bridge_exchange(prepared)
    if not callable(group_ranks_getter):
        raise MdpConfigurationError("MDP: dynamic bridge group_ranks_getter is callable.")
    if not callable(all_to_all_single):
        raise MdpConfigurationError("MDP: dynamic bridge all_to_all_single is callable.")
    _validate_group_binding(carrier, group=group, group_ranks_getter=group_ranks_getter)
    return _execute_validated_dynamic_bridge_exchange(
        carrier, group=group, all_to_all_single=all_to_all_single
    )


def _execute_validated_dynamic_bridge_exchange(
    carrier: PreparedDynamicBridgeExchange, *, group: Any, all_to_all_single: Callable[..., Any]
) -> Mapping[DynamicBridgeKey, Tensor]:
    try:
        all_to_all_single(
            carrier.receive_buffer,
            carrier.send_buffer,
            output_split_sizes=list(carrier.output_split_sizes),
            input_split_sizes=list(carrier.input_split_sizes),
            group=group,
            async_op=False,
        )
    except Exception as error:
        raise MdpBridgeError(
            "MDP: synchronous dynamic bridge all-to-all completed successfully."
        ) from error
    return carrier.received_tensors


def _validate_bridge_gate_context(
    *, global_rank: Any, group_ranks: Any, all_gather_status: Any, timeout_seconds: Any
) -> tuple[int, tuple[int, ...], Callable[..., Any], float]:
    """Validate rank-symmetric rendezvous inputs before local preparation.

    Because this check precedes consensus, asymmetric invalid rank, group,
    gather, or timeout inputs cannot be converged by this helper.
    """
    rank = _require_integer("dynamic bridge gate global rank", global_rank)
    if not isinstance(group_ranks, tuple) or not group_ranks:
        raise MdpConfigurationError(
            "MDP: dynamic bridge gate group ranks form a non-empty immutable tuple."
        )
    for participant in group_ranks:
        _require_integer("dynamic bridge gate participant rank", participant)
    if len(set(group_ranks)) != len(group_ranks):
        raise MdpConfigurationError("MDP: dynamic bridge gate group ranks are unique.")
    if rank not in group_ranks:
        raise MdpConfigurationError("MDP: dynamic bridge gate global rank belongs to its group.")
    if not callable(all_gather_status):
        raise MdpConfigurationError("MDP: dynamic bridge gate status gather is callable.")
    timeout = _validate_precollective_timeout(timeout_seconds)
    return rank, group_ranks, all_gather_status, timeout


def _snapshot_bridge_gate_digest(value: Any) -> bytes:
    try:
        digest = value.digest
    except Exception:
        return bytes(16)
    if type(digest) is not bytes or len(digest) != 16:
        return bytes(16)
    return digest


def _run_dynamic_bridge_gate(
    *,
    phase: BridgePhase,
    ledger: DynamicBridgeLedger,
    reverse_ledger: DynamicBridgeLedger,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    width: int,
    dtype: torch.dtype,
    global_rank: int,
    group_ranks: tuple[int, ...],
    all_gather_status: Any,
    local_prepare: Any,
    timeout_seconds: float,
    group: Any,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> Mapping[DynamicBridgeKey, Tensor]:
    """Gate one embedding or gradient exchange with deterministic consensus.

    Only rank-symmetric rendezvous context may reject before the status gather.
    All metadata, route, preparation, carrier, callback, and native-group
    failures become one local error status. ``local_prepare`` must remain
    rank-local and noncollective. The injected gather must not mutate its status
    input or sealed carrier buffers. This helper owns no retry or recovery.
    """
    gate_id = 3 if phase is BridgePhase.GRADIENT else 1
    rank, ranks, gather, timeout = _validate_bridge_gate_context(
        global_rank=global_rank,
        group_ranks=group_ranks,
        all_gather_status=all_gather_status,
        timeout_seconds=timeout_seconds,
    )
    manifest_digest = _snapshot_bridge_gate_digest(global_manifest)
    route_digest = bytes(16)
    carrier = None
    local_error = None
    try:
        if phase not in (BridgePhase.EMBEDDING, BridgePhase.GRADIENT):
            raise MdpPlanError("MDP: dynamic bridge gate phase is embedding or gradient.")
        if type(ledger) is not DynamicBridgeLedger or ledger.phase is not phase:
            raise MdpPlanError("MDP: dynamic bridge gate ledger matches its exact phase.")
        if type(global_manifest) is not DecoderGlobalManifest:
            raise MdpPlanError("MDP: dynamic bridge gate has an exact global manifest.")
        if type(plan) is not DecoderDynamicPlan:
            raise MdpPlanError("MDP: dynamic bridge gate has an exact decoder plan.")
        if not isinstance(dtype, torch.dtype):
            raise MdpConfigurationError("MDP: dynamic bridge gate dtype is a torch dtype.")
        route_digest = build_dynamic_bridge_route_authority_digest(
            ledger,
            reverse_ledger,
            plan=plan,
            global_manifest=global_manifest,
            producer_rank_by_item=producer_rank_by_item,
            output_rows_by_item=output_rows_by_item,
            width=width,
            dtype=dtype,
            participant_ranks=ranks,
        )
        if not callable(local_prepare):
            raise MdpConfigurationError("MDP: dynamic bridge gate local_prepare is callable.")
        if not callable(group_ranks_getter):
            raise MdpConfigurationError("MDP: dynamic bridge gate group_ranks_getter is callable.")
        if not callable(all_to_all_single):
            raise MdpConfigurationError("MDP: dynamic bridge gate all_to_all_single is callable.")
        carrier = validate_prepared_dynamic_bridge_exchange(local_prepare())
        if carrier.phase is not phase:
            raise MdpBridgeError("MDP: prepared dynamic bridge phase matches the gate phase.")
        if carrier.dtype != dtype:
            raise MdpBridgeError(
                "MDP: prepared dynamic bridge dtype matches the gate route authority."
            )
        if carrier.route_authority_digest != route_digest:
            raise MdpBridgeError(
                "MDP: prepared dynamic bridge matches the gate route authority digest."
            )
        if carrier.global_rank != rank:
            raise MdpBridgeError(
                "MDP: prepared dynamic bridge global rank matches the gate context."
            )
        if carrier.participant_ranks != ranks:
            raise MdpBridgeError(
                "MDP: prepared dynamic bridge participants match the gate context."
            )
        _validate_group_binding(carrier, group=group, group_ranks_getter=group_ranks_getter)
    except Exception as error:
        local_error = error

    status = _PrecollectiveStatus(
        global_rank=rank,
        global_manifest_digest=manifest_digest,
        plan_digest=route_digest,
        error_code=int(local_error is not None),
        gate_id=gate_id,
    )
    try:
        _run_precollective_consensus(
            status, group_ranks=ranks, all_gather_status=gather, timeout_seconds=timeout
        )
    except (MdpBridgeError, MdpPlanError) as error:
        if local_error is not None and error.__cause__ is None:
            raise error from local_error
        raise
    if local_error is not None:
        raise MdpStateError(
            "MDP: dynamic bridge gate consensus succeeded despite a local error."
        ) from local_error
    return _execute_validated_dynamic_bridge_exchange(
        carrier, group=group, all_to_all_single=all_to_all_single
    )


def prepare_dynamic_bridge_exchange(
    ledger: DynamicBridgeLedger,
    reverse_ledger: DynamicBridgeLedger,
    *,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    width: int,
    dtype: torch.dtype,
    participant_ranks: tuple[int, ...],
    global_rank: int,
    local_tensors: Mapping[DynamicBridgeKey, Tensor],
    send_buffer: Tensor,
    receive_buffer: Tensor,
) -> PreparedDynamicBridgeExchange:
    """Validate and pack one bridge phase into caller-owned one-dimensional buffers."""
    digest = build_dynamic_bridge_route_authority_digest(
        ledger,
        reverse_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=width,
        dtype=dtype,
        participant_ranks=participant_ranks,
    )
    phase = ledger.phase
    selected, reverse = ledger, reverse_ledger
    rank = _require_integer("dynamic bridge preparation global rank", global_rank)
    input_splits, output_splits = dynamic_bridge_split_sizes(
        selected,
        reverse_ledger=reverse,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=width,
        dtype=dtype,
        participant_ranks=participant_ranks,
        global_rank=rank,
    )
    send = _validate_buffer("send buffer", send_buffer, dtype=dtype, elements=sum(input_splits))
    receive = _validate_buffer(
        "receive buffer", receive_buffer, dtype=dtype, elements=sum(output_splits)
    )
    if send.device != receive.device:
        raise MdpConfigurationError("MDP: dynamic bridge buffers use the same device.")
    if send.numel() and receive.numel() and _storage_pointer(send) == _storage_pointer(receive):
        raise MdpConfigurationError("MDP: dynamic bridge buffers use disjoint storage.")
    if not isinstance(local_tensors, Mapping):
        raise MdpConfigurationError("MDP: dynamic bridge local tensors form a mapping.")
    source_entries = tuple(entry for entry in selected.entries if entry.src_global_rank == rank)
    expected_keys = tuple(entry.key for entry in source_entries)
    try:
        local_keys = tuple(local_tensors)
        exact_keys = set(local_keys) == set(expected_keys)
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: dynamic bridge local tensor keys are readable."
        ) from error
    if (
        any(not isinstance(key, DynamicBridgeKey) for key in local_keys)
        or not exact_keys
        or len(local_keys) != len(expected_keys)
    ):
        raise MdpConfigurationError("MDP: dynamic bridge local tensors match exact source keys.")

    participant_index = {value: index for index, value in enumerate(participant_ranks)}
    send_bases = _participant_bases(input_splits)
    receive_bases = _participant_bases(output_splits)
    copies = []
    for entry in source_entries:
        tensor = local_tensors[entry.key]
        expected_shape = (output_rows_by_item[entry.key.item_id], width)
        if not isinstance(tensor, Tensor):
            raise MdpConfigurationError("MDP: dynamic bridge local values are tensors.")
        if tuple(tensor.shape) != expected_shape or tensor.numel() != entry.element_count:
            raise MdpConfigurationError("MDP: dynamic bridge local tensor shape matches authority.")
        if tensor.dtype != dtype or tensor.device != send.device:
            raise MdpConfigurationError(
                "MDP: dynamic bridge local tensor dtype and device match its send buffer."
            )
        if tensor.requires_grad:
            raise MdpConfigurationError("MDP: dynamic bridge local transport tensors are detached.")
        if _storage_pointer(tensor) in {_storage_pointer(send), _storage_pointer(receive)}:
            raise MdpConfigurationError(
                "MDP: dynamic bridge local tensors do not alias transport buffers."
            )
        offset = send_bases[participant_index[entry.dst_global_rank]] + entry.plan_offset
        copies.append((send.narrow(0, offset, entry.element_count), tensor))

    receive_entries = tuple(entry for entry in selected.entries if entry.dst_global_rank == rank)
    views = {}
    for entry in sorted(
        receive_entries,
        key=lambda value: (participant_index[value.src_global_rank], value.plan_offset),
    ):
        shape = (output_rows_by_item[entry.key.item_id], width)
        offset = receive_bases[participant_index[entry.src_global_rank]] + entry.plan_offset
        views[entry.key] = receive.narrow(0, offset, entry.element_count).view(shape)

    prepared = PreparedDynamicBridgeExchange(
        phase=phase,
        dtype=dtype,
        global_rank=rank,
        participant_ranks=participant_ranks,
        input_split_sizes=input_splits,
        output_split_sizes=output_splits,
        route_authority_digest=digest,
        send_buffer=send,
        receive_buffer=receive,
        received_tensors=MappingProxyType(views),
    )
    object.__setattr__(prepared, "_authority", _capture_authority(prepared))
    validate_prepared_dynamic_bridge_exchange(prepared)
    for destination, tensor in copies:
        destination.view(tuple(tensor.shape)).copy_(tensor.detach())
    return prepared
