# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Caller-buffer decoder payload preparation and execution for MDP Dynamic-CP.

The caller owns tensor allocation, native process-group construction, broader
gate ordering, and recovery. This module owns the decoder-payload gate-0
consensus. ``participant_ranks`` is the canonical native group order, which may
differ from the decoder endpoint order in the plan.
"""

import hashlib
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from math import isfinite
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    DecoderTensorFieldSpec,
    _PrecollectiveStatus,
    _run_precollective_consensus,
    validate_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderDynamicPlan, validate_decoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_routing import (
    DecoderPayloadRouteKey,
    DecoderPayloadRouteLedger,
    build_decoder_payload_route_ledger,
    decoder_payload_split_sizes,
    validate_decoder_payload_route_ledger,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_PAYLOAD_ROUTE_AUTHORITY_DOMAIN = b"megatron.mdp.dynamic-cp.decoder-payload-route"
_PAYLOAD_ROUTE_AUTHORITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _PreparedDecoderPayloadAuthority:
    route_authority_digest: bytes
    dtype: torch.dtype
    global_rank: int
    participant_ranks: tuple[int, ...]
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    device: torch.device
    send_buffer_geometry: tuple[int, int, int]
    receive_buffer_geometry: tuple[int, int, int]
    receive_descriptors: tuple[tuple[DecoderPayloadRouteKey, tuple[int, ...], int], ...]


@dataclass(frozen=True)
class PreparedDecoderPayloadExchange:
    """One rank's validated buffers and split vectors for one payload dtype."""

    dtype: torch.dtype
    global_rank: int
    participant_ranks: tuple[int, ...]
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    send_buffer: Tensor = field(compare=False, repr=False)
    receive_buffer: Tensor = field(compare=False, repr=False)
    received_tensors: Mapping[DecoderPayloadRouteKey, Tensor] = field(compare=False, repr=False)
    _authority: _PreparedDecoderPayloadAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


def _validate_buffer(name: str, buffer: Any, *, dtype: torch.dtype, elements: int) -> Tensor:
    if not isinstance(buffer, Tensor):
        raise MdpConfigurationError(f"MDP: decoder payload {name} is a tensor.")
    if buffer.dtype != dtype:
        raise MdpConfigurationError(
            f"MDP: decoder payload {name} dtype matches the selected transport dtype."
        )
    if buffer.dim() != 1 or not buffer.is_contiguous():
        raise MdpConfigurationError(
            f"MDP: decoder payload {name} is a contiguous one-dimensional tensor."
        )
    if buffer.numel() != elements:
        raise MdpConfigurationError(
            f"MDP: decoder payload {name} holds exactly {elements} elements."
        )
    return buffer


def _field_specs(
    global_manifest: DecoderGlobalManifest,
) -> dict[tuple[Any, str], DecoderTensorFieldSpec]:
    return {
        (payload.sample_id, spec.name): spec
        for payload in global_manifest.payloads
        for spec in payload.field_specs
    }


def _route_digest_integers(hasher: Any, *values: int) -> None:
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > _INT64_MAX
        for value in values
    ):
        raise MdpBridgeError("MDP: decoder payload route digest fields fit signed int64.")
    hasher.update(struct.pack(f"<{len(values)}q", *values))


def _route_digest_bytes(hasher: Any, value: bytes) -> None:
    if type(value) is not bytes:
        raise MdpPlanError("MDP: decoder payload route digest input is bytes.")
    _route_digest_integers(hasher, len(value))
    hasher.update(value)


def _route_digest_text(hasher: Any, value: str) -> None:
    if not isinstance(value, str):
        raise MdpBridgeError("MDP: decoder payload route digest text is a string.")
    encoded = value.encode("utf-8")
    _route_digest_integers(hasher, len(encoded))
    hasher.update(encoded)


def _payload_route_authority_digest(
    ledger: DecoderPayloadRouteLedger,
    *,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    source_rank_by_lane: Mapping[int, int],
    participant_ranks: tuple[int, ...],
    dtype: torch.dtype,
) -> bytes:
    """Digest one validated decoder payload plan and route authority."""
    if not isinstance(dtype, torch.dtype):
        raise MdpConfigurationError("MDP: decoder payload route authority dtype is a torch dtype.")
    hasher = hashlib.blake2b(digest_size=16)
    _route_digest_integers(
        hasher, len(_PAYLOAD_ROUTE_AUTHORITY_DOMAIN), _PAYLOAD_ROUTE_AUTHORITY_SCHEMA_VERSION
    )
    hasher.update(_PAYLOAD_ROUTE_AUTHORITY_DOMAIN)
    _route_digest_bytes(hasher, global_manifest.digest)
    _route_digest_bytes(hasher, plan.digest)
    _route_digest_text(hasher, str(dtype))
    _route_digest_integers(hasher, len(participant_ranks), *participant_ranks)
    source_ranks = tuple(sorted(source_rank_by_lane.items()))
    _route_digest_integers(hasher, len(source_ranks))
    for lane, rank in source_ranks:
        _route_digest_integers(hasher, lane, rank)
    _route_digest_integers(hasher, len(ledger.entries))
    for entry in ledger.entries:
        _route_digest_integers(
            hasher,
            entry.src_global_rank,
            entry.dst_global_rank,
            *entry.key.sample_id.to_wire_tuple(),
            entry.key.endpoint_rank,
            entry.element_count,
            entry.plan_offset,
        )
        _route_digest_text(hasher, entry.key.field_name)
        _route_digest_text(hasher, str(entry.dtype))
    return hasher.digest()


def _storage_pointer(tensor: Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _capture_prepared_authority(
    prepared: PreparedDecoderPayloadExchange, *, route_authority_digest: bytes
) -> _PreparedDecoderPayloadAuthority:
    receive_offset = prepared.receive_buffer.storage_offset()
    return _PreparedDecoderPayloadAuthority(
        route_authority_digest=route_authority_digest,
        dtype=prepared.dtype,
        global_rank=prepared.global_rank,
        participant_ranks=prepared.participant_ranks,
        input_split_sizes=prepared.input_split_sizes,
        output_split_sizes=prepared.output_split_sizes,
        device=prepared.send_buffer.device,
        send_buffer_geometry=(
            id(prepared.send_buffer),
            _storage_pointer(prepared.send_buffer),
            prepared.send_buffer.storage_offset(),
        ),
        receive_buffer_geometry=(
            id(prepared.receive_buffer),
            _storage_pointer(prepared.receive_buffer),
            receive_offset,
        ),
        receive_descriptors=tuple(
            (key, tuple(tensor.shape), tensor.storage_offset() - receive_offset)
            for key, tensor in prepared.received_tensors.items()
        ),
    )


def _validate_split_sizes(name: str, value: Any, *, participant_count: int) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != participant_count:
        raise MdpConfigurationError(f"MDP: decoder payload {name} has one entry per participant.")
    total = 0
    for split in value:
        if (
            not isinstance(split, int)
            or isinstance(split, bool)
            or split < 0
            or split > _INT64_MAX - total
        ):
            raise MdpConfigurationError(
                f"MDP: decoder payload {name} contains non-negative signed-int64 sizes."
            )
        total += split
    return value


def _validate_prepared_exchange(prepared: Any) -> PreparedDecoderPayloadExchange:
    if type(prepared) is not PreparedDecoderPayloadExchange:
        raise MdpBridgeError(
            "MDP: decoder payload execution requires its exact prepared carrier type."
        )
    if type(prepared._authority) is not _PreparedDecoderPayloadAuthority:
        raise MdpBridgeError(
            "MDP: decoder payload execution requires its sealed preparation authority snapshot."
        )
    route_authority_digest = prepared._authority.route_authority_digest
    if type(route_authority_digest) is not bytes or len(route_authority_digest) != 16:
        raise MdpBridgeError(
            "MDP: decoder payload execution has a fixed 16-byte route authority digest."
        )
    if not isinstance(prepared.dtype, torch.dtype):
        raise MdpConfigurationError("MDP: prepared decoder payload dtype is a torch dtype.")
    participants = prepared.participant_ranks
    if not isinstance(participants, tuple) or not participants:
        raise MdpConfigurationError(
            "MDP: prepared decoder payload participants form a non-empty rank tuple."
        )
    if any(
        not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 or rank > _INT64_MAX
        for rank in participants
    ) or len(set(participants)) != len(participants):
        raise MdpConfigurationError(
            "MDP: prepared decoder payload participants are unique signed-int64 ranks."
        )
    if (
        not isinstance(prepared.global_rank, int)
        or isinstance(prepared.global_rank, bool)
        or prepared.global_rank not in participants
    ):
        raise MdpConfigurationError(
            "MDP: prepared decoder payload global rank is an exact participant."
        )
    input_splits = _validate_split_sizes(
        "input split sizes", prepared.input_split_sizes, participant_count=len(participants)
    )
    output_splits = _validate_split_sizes(
        "output split sizes", prepared.output_split_sizes, participant_count=len(participants)
    )
    send = _validate_buffer(
        "send buffer", prepared.send_buffer, dtype=prepared.dtype, elements=sum(input_splits)
    )
    receive = _validate_buffer(
        "receive buffer", prepared.receive_buffer, dtype=prepared.dtype, elements=sum(output_splits)
    )
    if send.device != receive.device:
        raise MdpConfigurationError("MDP: prepared decoder payload buffers use the same device.")
    if send.numel() and receive.numel() and _storage_pointer(send) == _storage_pointer(receive):
        raise MdpConfigurationError("MDP: prepared decoder payload buffers use disjoint storage.")
    if type(prepared.received_tensors) is not _MAPPING_PROXY_TYPE:
        raise MdpBridgeError(
            "MDP: prepared decoder payload receive views form an immutable mapping."
        )

    receive_start = receive.storage_offset()
    receive_end = receive_start + receive.numel()
    intervals = []
    for key, tensor in prepared.received_tensors.items():
        if not isinstance(key, DecoderPayloadRouteKey) or not isinstance(tensor, Tensor):
            raise MdpBridgeError(
                "MDP: prepared decoder payload receive views use typed keys and tensors."
            )
        if (
            tensor.dtype != prepared.dtype
            or tensor.device != receive.device
            or not tensor.is_contiguous()
            or tensor.numel() <= 0
            or _storage_pointer(tensor) != _storage_pointer(receive)
        ):
            raise MdpBridgeError(
                "MDP: prepared decoder payload receive views match buffer dtype, "
                "device, and storage."
            )
        start = tensor.storage_offset()
        end = start + tensor.numel()
        if start < receive_start or end > receive_end:
            raise MdpBridgeError(
                "MDP: prepared decoder payload receive views stay within the receive buffer."
            )
        intervals.append((start, end))
    cursor = receive_start
    for start, end in sorted(intervals):
        if start != cursor:
            raise MdpBridgeError(
                "MDP: prepared decoder payload receive views exactly partition the receive buffer."
            )
        cursor = end
    if cursor != receive_end:
        raise MdpBridgeError(
            "MDP: prepared decoder payload receive views exactly partition the receive buffer."
        )
    if (
        _capture_prepared_authority(prepared, route_authority_digest=route_authority_digest)
        != prepared._authority
    ):
        raise MdpBridgeError(
            "MDP: prepared decoder payload public geometry matches its authority snapshot."
        )
    return prepared


def _validate_group_binding(
    prepared: PreparedDecoderPayloadExchange,
    *,
    group: Any,
    group_ranks_getter: Callable[[Any], Any],
) -> None:
    if not callable(group_ranks_getter):
        raise MdpConfigurationError("MDP: decoder payload group_ranks_getter is callable.")
    try:
        actual_ranks = tuple(group_ranks_getter(group))
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: decoder payload native group rank query succeeded."
        ) from error
    if actual_ranks != prepared.participant_ranks or any(
        not isinstance(rank, int) or isinstance(rank, bool) for rank in actual_ranks
    ):
        raise MdpConfigurationError(
            "MDP: decoder payload native group uses the exact participant rank order."
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
            "MDP: decoder payload native group geometry query succeeded."
        ) from error
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size != len(prepared.participant_ranks)
    ):
        raise MdpConfigurationError(
            "MDP: decoder payload native group size matches participant count."
        )
    expected_local_rank = prepared.participant_ranks.index(prepared.global_rank)
    if (
        not isinstance(local_rank, int)
        or isinstance(local_rank, bool)
        or local_rank != expected_local_rank
    ):
        raise MdpConfigurationError(
            "MDP: decoder payload native group local rank matches the global rank."
        )


def execute_decoder_payload_exchange(
    prepared: PreparedDecoderPayloadExchange,
    *,
    group: Any,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> Mapping[DecoderPayloadRouteKey, Tensor]:
    """Execute one validated synchronous decoder-payload all-to-all.

    The prepared participant order must exactly match the native order returned
    by ``group_ranks_getter``. This function performs no cross-rank failure
    consensus, retry, or recovery; those remain runtime responsibilities.

    Args:
        prepared: Caller-owned buffers and immutable receive views from
            :func:`prepare_decoder_payload_exchange`.
        group: Native process group whose ordered ranks match the carrier.
        group_ranks_getter: Injectable native ordered-rank query.
        all_to_all_single: Injectable synchronous collective implementation.

    Returns:
        The exact immutable receive-view mapping stored in ``prepared``.
    """
    carrier = _validate_prepared_exchange(prepared)
    if not callable(all_to_all_single):
        raise MdpConfigurationError("MDP: decoder payload all_to_all_single is callable.")
    _validate_group_binding(carrier, group=group, group_ranks_getter=group_ranks_getter)
    return _execute_validated_decoder_payload_exchange(
        carrier, group=group, all_to_all_single=all_to_all_single
    )


def _execute_validated_decoder_payload_exchange(
    carrier: PreparedDecoderPayloadExchange, *, group: Any, all_to_all_single: Callable[..., Any]
) -> Mapping[DecoderPayloadRouteKey, Tensor]:
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
            "MDP: synchronous decoder payload all-to-all completed successfully."
        ) from error
    return carrier.received_tensors


def _validate_payload_gate_context(
    *, global_rank: Any, group_ranks: Any, all_gather_status: Any, timeout_seconds: Any
) -> tuple[int, tuple[int, ...], Callable[..., Any], float]:
    """Validate rank-symmetric inputs before any status gather.

    Because this check precedes consensus, asymmetric invalid rank, group,
    gather, or timeout inputs cannot be converged by this helper.
    """
    if (
        not isinstance(global_rank, int)
        or isinstance(global_rank, bool)
        or global_rank < 0
        or global_rank > _INT64_MAX
    ):
        raise MdpConfigurationError("MDP: decoder payload gate global rank is signed-int64.")
    if not isinstance(group_ranks, tuple):
        raise MdpConfigurationError("MDP: decoder payload gate group ranks use an immutable tuple.")
    if not group_ranks:
        raise MdpConfigurationError("MDP: decoder payload gate group ranks form a non-empty tuple.")
    if any(
        not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 or rank > _INT64_MAX
        for rank in group_ranks
    ):
        raise MdpConfigurationError(
            "MDP: decoder payload gate group ranks are signed-int64 integers."
        )
    if len(set(group_ranks)) != len(group_ranks):
        raise MdpConfigurationError("MDP: decoder payload gate group ranks are unique.")
    if global_rank not in group_ranks:
        raise MdpConfigurationError("MDP: decoder payload gate global rank belongs to its group.")
    if not callable(all_gather_status):
        raise MdpConfigurationError("MDP: decoder payload gate status gather is callable.")
    timeout = _validate_status_timeout(timeout_seconds)
    return global_rank, group_ranks, all_gather_status, timeout.total_seconds()


def _snapshot_payload_gate_digest(value: Any) -> bytes:
    try:
        digest = value.digest
    except Exception:
        return bytes(16)
    if type(digest) is not bytes or len(digest) != 16:
        return bytes(16)
    return digest


def _run_decoder_payload_gate(
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    ledger: DecoderPayloadRouteLedger,
    source_rank_by_lane: Mapping[int, int],
    dtype: torch.dtype,
    global_rank: int,
    group_ranks: tuple[int, ...],
    all_gather_status: Any,
    local_prepare: Any,
    timeout_seconds: float,
    group: Any,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> Mapping[DecoderPayloadRouteKey, Tensor]:
    """Gate one decoder payload exchange with deterministic rank consensus.

    Local validation and preparation precede the fixed gate-0 consensus. The
    synchronous payload collective runs only after every rank reports success.
    This helper provides no retry or recovery. ``local_prepare`` must remain
    rank-local and must not post or mutate collectives. Injected status gathers
    must not mutate submitted status metadata; malicious callback behavior is
    outside the contract.
    """
    rank, ranks, gather, timeout = _validate_payload_gate_context(
        global_rank=global_rank,
        group_ranks=group_ranks,
        all_gather_status=all_gather_status,
        timeout_seconds=timeout_seconds,
    )
    manifest_digest = _snapshot_payload_gate_digest(global_manifest)
    route_authority_digest = bytes(16)
    carrier = None
    local_error = None
    try:
        if type(global_manifest) is not DecoderGlobalManifest:
            raise MdpPlanError("MDP: decoder payload gate has an exact global manifest.")
        if type(plan) is not DecoderDynamicPlan:
            raise MdpPlanError("MDP: decoder payload gate has an exact decoder plan.")
        validate_decoder_global_manifest(global_manifest)
        validate_decoder_dynamic_plan(plan)
        if global_manifest.samples != plan.samples:
            raise MdpPlanError("MDP: decoder payload gate manifest and plan catalogs agree.")
        if not isinstance(dtype, torch.dtype):
            raise MdpConfigurationError("MDP: decoder payload gate dtype is a torch dtype.")
        expected_ledger = build_decoder_payload_route_ledger(
            plan,
            global_manifest=global_manifest,
            source_rank_by_lane=source_rank_by_lane,
            participant_ranks=ranks,
        )
        route_authority_digest = _payload_route_authority_digest(
            expected_ledger,
            plan=plan,
            global_manifest=global_manifest,
            source_rank_by_lane=source_rank_by_lane,
            participant_ranks=ranks,
            dtype=dtype,
        )
        validate_decoder_payload_route_ledger(
            ledger,
            plan=plan,
            global_manifest=global_manifest,
            source_rank_by_lane=source_rank_by_lane,
            participant_ranks=ranks,
        )
        if not callable(local_prepare):
            raise MdpConfigurationError("MDP: decoder payload gate local_prepare is callable.")
        if not callable(all_to_all_single):
            raise MdpConfigurationError("MDP: decoder payload gate all_to_all_single is callable.")
        carrier = _validate_prepared_exchange(local_prepare())
        if carrier.dtype != dtype:
            raise MdpBridgeError(
                "MDP: prepared decoder payload dtype matches the gate route authority."
            )
        if carrier._authority.route_authority_digest != route_authority_digest:
            raise MdpBridgeError(
                "MDP: prepared decoder payload matches the gate route authority digest."
            )
        if carrier.global_rank != rank:
            raise MdpBridgeError(
                "MDP: prepared decoder payload global rank matches the gate context."
            )
        if carrier.participant_ranks != ranks:
            raise MdpBridgeError(
                "MDP: prepared decoder payload participants match the gate context."
            )
        _validate_group_binding(carrier, group=group, group_ranks_getter=group_ranks_getter)
    except Exception as error:
        local_error = error

    # Gate 0 carries the composite route authority in the historical plan-digest slot.
    status = _PrecollectiveStatus(
        global_rank=rank,
        global_manifest_digest=manifest_digest,
        plan_digest=route_authority_digest,
        error_code=int(local_error is not None),
        gate_id=0,
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
            "MDP: decoder payload gate consensus succeeded despite a local error."
        ) from local_error
    return _execute_validated_decoder_payload_exchange(
        carrier, group=group, all_to_all_single=all_to_all_single
    )


def prepare_decoder_payload_exchange(
    ledger: DecoderPayloadRouteLedger,
    *,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    source_rank_by_lane: Mapping[int, int],
    participant_ranks: tuple[int, ...],
    dtype: torch.dtype,
    global_rank: int,
    local_tensors: Mapping[DecoderPayloadRouteKey, Tensor],
    send_buffer: Tensor,
    receive_buffer: Tensor,
) -> PreparedDecoderPayloadExchange:
    """Pack one rank and dtype without allocating or executing a collective.

    ``participant_ranks`` must use the future native process group's canonical
    rank order; decoder endpoint order remains independently plan-owned.
    """
    expected = validate_decoder_payload_route_ledger(
        ledger,
        plan=plan,
        global_manifest=global_manifest,
        source_rank_by_lane=source_rank_by_lane,
        participant_ranks=participant_ranks,
    )
    route_authority_digest = _payload_route_authority_digest(
        expected,
        plan=plan,
        global_manifest=global_manifest,
        source_rank_by_lane=source_rank_by_lane,
        participant_ranks=participant_ranks,
        dtype=dtype,
    )
    input_splits, output_splits = decoder_payload_split_sizes(
        expected,
        plan=plan,
        global_manifest=global_manifest,
        source_rank_by_lane=source_rank_by_lane,
        participant_ranks=participant_ranks,
        dtype=dtype,
        global_rank=global_rank,
    )
    send = _validate_buffer("send buffer", send_buffer, dtype=dtype, elements=sum(input_splits))
    receive = _validate_buffer(
        "receive buffer", receive_buffer, dtype=dtype, elements=sum(output_splits)
    )
    if send.device != receive.device:
        raise MdpConfigurationError(
            "MDP: decoder payload send and receive buffers use the same device."
        )
    if send.numel() and receive.numel() and _storage_pointer(send) == _storage_pointer(receive):
        raise MdpConfigurationError(
            "MDP: decoder payload send and receive buffers use disjoint storage."
        )
    if not isinstance(local_tensors, Mapping):
        raise MdpConfigurationError("MDP: decoder payload local tensors form a mapping.")

    typed_entries = tuple(entry for entry in expected.entries if entry.dtype == dtype)
    source_entries = tuple(entry for entry in typed_entries if entry.src_global_rank == global_rank)
    expected_source_keys = {entry.key for entry in source_entries}
    if set(local_tensors) != expected_source_keys:
        raise MdpBridgeError(
            "MDP: decoder payload local tensors exactly cover this rank's selected-dtype routes."
        )

    specs = _field_specs(global_manifest)
    device_types = {
        specs[(entry.key.sample_id, entry.key.field_name)].device_type for entry in typed_entries
    }
    if len(device_types) > 1:
        raise MdpConfigurationError(
            "MDP: one decoder payload dtype uses one manifest device type per exchange."
        )
    if device_types and send.device.type != next(iter(device_types)):
        raise MdpConfigurationError(
            "MDP: decoder payload transport buffers match the manifest device type."
        )
    participant_positions = {rank: index for index, rank in enumerate(expected.participant_ranks)}
    send_bases = []
    offset = 0
    for split in input_splits:
        send_bases.append(offset)
        offset += split

    pack_operations = []
    for entry in source_entries:
        tensor = local_tensors[entry.key]
        spec = specs[(entry.key.sample_id, entry.key.field_name)]
        shape = spec.shape
        if (
            not isinstance(tensor, Tensor)
            or tensor.dtype != dtype
            or tensor.device != send.device
            or tensor.device.type != spec.device_type
            or tuple(tensor.shape) != shape
            or tensor.numel() != entry.element_count
        ):
            raise MdpBridgeError(
                "MDP: decoder payload source tensor matches route dtype, device, shape, and extent."
            )
        if tensor.numel() and _storage_pointer(tensor) in {
            _storage_pointer(send),
            _storage_pointer(receive),
        }:
            raise MdpBridgeError(
                "MDP: decoder payload source tensors do not alias transport buffers."
            )
        destination = participant_positions[entry.dst_global_rank]
        start = send_bases[destination] + entry.plan_offset
        pack_operations.append((start, entry.element_count, shape, tensor))

    for start, element_count, shape, tensor in pack_operations:
        send[start : start + element_count].view(shape).copy_(tensor)

    receive_bases = []
    offset = 0
    for split in output_splits:
        receive_bases.append(offset)
        offset += split
    received = {}
    for entry in typed_entries:
        if entry.dst_global_rank != global_rank:
            continue
        source = participant_positions[entry.src_global_rank]
        start = receive_bases[source] + entry.plan_offset
        shape = specs[(entry.key.sample_id, entry.key.field_name)].shape
        received[entry.key] = receive[start : start + entry.element_count].view(shape)

    prepared = PreparedDecoderPayloadExchange(
        dtype=dtype,
        global_rank=global_rank,
        participant_ranks=expected.participant_ranks,
        input_split_sizes=input_splits,
        output_split_sizes=output_splits,
        send_buffer=send,
        receive_buffer=receive,
        received_tensors=MappingProxyType(received),
    )
    object.__setattr__(
        prepared,
        "_authority",
        _capture_prepared_authority(prepared, route_authority_digest=route_authority_digest),
    )
    return prepared


def _validate_status_wire(value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != 7:
        raise MdpPlanError("MDP: precollective status has an exact seven-word tuple wire.")
    if any(
        not isinstance(component, int)
        or isinstance(component, bool)
        or component < _INT64_MIN
        or component > _INT64_MAX
        for component in value
    ):
        raise MdpPlanError("MDP: every precollective status word is a signed-int64 integer.")
    return value


def _validate_status_timeout(value: Any) -> timedelta:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MdpConfigurationError(
            "MDP: precollective status timeout is a positive finite number."
        )
    try:
        seconds = float(value)
    except (OverflowError, ValueError) as error:
        raise MdpConfigurationError(
            "MDP: precollective status timeout is a positive finite number."
        ) from error
    if not isfinite(seconds) or seconds <= 0:
        raise MdpConfigurationError(
            "MDP: precollective status timeout is a positive finite number."
        )
    if seconds < 0.001:
        raise MdpConfigurationError(
            "MDP: precollective status timeout is at least one millisecond."
        )
    try:
        timeout = timedelta(seconds=seconds)
    except (OverflowError, ValueError) as error:
        raise MdpConfigurationError(
            "MDP: precollective status timeout fits a bounded wait duration."
        ) from error
    if timeout < timedelta(milliseconds=1):
        raise MdpConfigurationError(
            "MDP: precollective status timeout is at least one millisecond."
        )
    return timeout


def _validate_status_group_context(
    *,
    group: Any,
    group_ranks: Any,
    global_rank: Any,
    device: Any,
    group_ranks_getter: Any,
    all_gather_into_tensor: Any,
) -> tuple[int, ...]:
    if not isinstance(group_ranks, tuple):
        raise MdpConfigurationError("MDP: precollective status group ranks use an immutable tuple.")
    if not group_ranks:
        raise MdpConfigurationError("MDP: precollective status group ranks form a non-empty tuple.")
    if any(
        not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 or rank > _INT64_MAX
        for rank in group_ranks
    ):
        raise MdpConfigurationError(
            "MDP: precollective status group ranks are signed-int64 integers."
        )
    if len(set(group_ranks)) != len(group_ranks):
        raise MdpConfigurationError("MDP: precollective status group ranks are unique.")
    if (
        not isinstance(global_rank, int)
        or isinstance(global_rank, bool)
        or global_rank not in group_ranks
    ):
        raise MdpConfigurationError(
            "MDP: precollective status global rank is an exact group participant."
        )
    if not isinstance(device, torch.device):
        raise MdpConfigurationError("MDP: precollective status device is a torch.device.")
    if not callable(group_ranks_getter):
        raise MdpConfigurationError("MDP: precollective status group_ranks_getter is callable.")
    if not callable(all_gather_into_tensor):
        raise MdpConfigurationError("MDP: precollective status all_gather_into_tensor is callable.")
    try:
        actual_ranks = tuple(group_ranks_getter(group))
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: precollective status native group rank query succeeded."
        ) from error
    if actual_ranks != group_ranks or any(
        not isinstance(rank, int) or isinstance(rank, bool) for rank in actual_ranks
    ):
        raise MdpConfigurationError(
            "MDP: precollective status native group uses the exact rank order."
        )
    try:
        size_query = getattr(group, "size")
        rank_query = getattr(group, "rank")
        if not callable(size_query) or not callable(rank_query):
            raise TypeError("native group size and rank queries are callable")
        group_size = size_query()
        local_rank = rank_query()
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: precollective status native group geometry query succeeded."
        ) from error
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size != len(group_ranks)
    ):
        raise MdpConfigurationError(
            "MDP: precollective status native group size matches participant count."
        )
    if (
        not isinstance(local_rank, int)
        or isinstance(local_rank, bool)
        or local_rank != group_ranks.index(global_rank)
    ):
        raise MdpConfigurationError(
            "MDP: precollective status native group local rank matches the global rank."
        )
    return group_ranks


def make_precollective_status_gather(
    *,
    group: Any,
    group_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_gather_into_tensor: Callable[..., Any] = dist.all_gather_into_tensor,
) -> Callable[..., tuple[tuple[int, ...], ...]]:
    """Bind a fixed-width bounded status gather to one exact native group.

    All participants must call the returned closure in the same phase order.
    Wire and timeout validation is rank-local before NCCL, so this helper cannot
    recover from asymmetric misuse; caller-side ordering and consensus remain
    runtime responsibilities.
    """
    ranks = _validate_status_group_context(
        group=group,
        group_ranks=group_ranks,
        global_rank=global_rank,
        device=device,
        group_ranks_getter=group_ranks_getter,
        all_gather_into_tensor=all_gather_into_tensor,
    )

    def gather(value: tuple[int, ...], *, timeout_seconds: float) -> tuple[tuple[int, ...], ...]:
        wire = _validate_status_wire(value)
        timeout = _validate_status_timeout(timeout_seconds)
        _validate_status_group_context(
            group=group,
            group_ranks=ranks,
            global_rank=global_rank,
            device=device,
            group_ranks_getter=group_ranks_getter,
            all_gather_into_tensor=all_gather_into_tensor,
        )
        source = torch.tensor(wire, dtype=torch.int64, device=device)
        destination = torch.empty(len(ranks) * len(wire), dtype=torch.int64, device=device)
        try:
            work = all_gather_into_tensor(destination, source, group=group, async_op=True)
        except Exception as error:
            raise MdpBridgeError("MDP: precollective status gather failed.") from error
        try:
            wait = getattr(work, "wait")
        except Exception as error:
            raise MdpBridgeError(
                "MDP: precollective status work exposes a callable wait."
            ) from error
        if not callable(wait):
            raise MdpBridgeError("MDP: precollective status work exposes a callable wait.")
        try:
            completed = wait(timeout=timeout)
        except Exception as error:
            raise MdpBridgeError("MDP: precollective status wait failed.") from error
        if completed is False:
            raise MdpBridgeError("MDP: precollective status gather timed out.")
        if completed is not True:
            raise MdpBridgeError("MDP: precollective status wait returns a boolean completion.")
        try:
            rows = destination.view(len(ranks), len(wire)).tolist()
            return tuple(tuple(int(component) for component in row) for row in rows)
        except Exception as error:
            raise MdpBridgeError(
                "MDP: precollective status result materialization failed."
            ) from error

    return gather
