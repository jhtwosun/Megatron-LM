# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Caller-buffer preparation for Dynamic-CP embedding and gradient bridges.

This module validates and packs one bridge phase.  It does not allocate buffers,
perform collective communication, create process groups, or own retry policy.
"""

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import (
    DynamicBridgeKey,
    DynamicBridgeLedger,
    dynamic_bridge_split_sizes,
    validate_dynamic_bridge_ledger_pair,
)
from megatron.core.mdp.dynamic_cp_execution import DecoderGlobalManifest
from megatron.core.mdp.dynamic_cp_plan import DecoderDynamicPlan
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError

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
