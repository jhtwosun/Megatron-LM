# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pre-collective decoder payload packing for MDP Dynamic-CP.

The caller owns allocation and collective execution. This module only packs
one dtype into caller-provided buffers and exposes immutable receive views.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from megatron.core.mdp.dynamic_cp_execution import DecoderGlobalManifest, DecoderTensorFieldSpec
from megatron.core.mdp.dynamic_cp_plan import DecoderDynamicPlan
from megatron.core.mdp.dynamic_cp_routing import (
    DecoderPayloadRouteKey,
    DecoderPayloadRouteLedger,
    decoder_payload_split_sizes,
    validate_decoder_payload_route_ledger,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError


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


def _storage_pointer(tensor: Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


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
    """Pack one rank and dtype without allocating or executing a collective."""
    expected = validate_decoder_payload_route_ledger(
        ledger,
        plan=plan,
        global_manifest=global_manifest,
        source_rank_by_lane=source_rank_by_lane,
        participant_ranks=participant_ranks,
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

    return PreparedDecoderPayloadExchange(
        dtype=dtype,
        global_rank=global_rank,
        participant_ranks=expected.participant_ranks,
        input_split_sizes=input_splits,
        output_split_sizes=output_splits,
        send_buffer=send,
        receive_buffer=receive,
        received_tensors=MappingProxyType(received),
    )
