# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused contracts for pre-collective Dynamic-CP decoder payload packing."""

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import megatron.core.mdp.dynamic_cp_routing as routing
import megatron.core.mdp.dynamic_cp_transport as transport
from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    build_decoder_global_manifest,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata, build_decoder_dynamic_plan
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError

_DECODER_RANKS = (30, 10, 40, 20)
_PARTICIPANTS = (80, 30, 99, 70, 20, 10, 40)
_SOURCE_RANKS = {3: 70, 7: 80}
_FIELDS = ("tokens", "loss_mask")


def _strided_values(seed, *, dtype):
    base = torch.arange(seed, seed + 16, dtype=dtype).reshape(2, 8)
    return base[1:2, 1::2]


def _packet(lane, order):
    tensors = MappingProxyType(
        {
            "tokens": _strided_values(lane * 100 + order * 10, dtype=torch.int64),
            "loss_mask": _strided_values(lane * 100 + order * 10, dtype=torch.float32),
        }
    )
    specs = tuple(
        DecoderTensorFieldSpec(
            name=name,
            dtype=tensors[name].dtype,
            shape=tuple(tensors[name].shape),
            device_type=tensors[name].device.type,
        )
        for name in _FIELDS
    )
    header = DecoderPayloadHeaderV1(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        source_dp_lane=lane,
        local_sample_order=order,
        valid_seqlen=3,
        padded_seqlen=4,
        tensor_field_count=2,
        none_field_count=1,
        position_components_or_minus_one=-1,
    )
    return DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=GlobalSampleId(lane, order),
        valid_seqlen=3,
        padded_seqlen=4,
        header=header.to_wire_tuple(),
        field_specs=specs,
        tensor_fields=tensors,
        none_fields=("position_ids",),
    )


def _window(lane):
    packets = tuple(_packet(lane, order) for order in range(2))
    samples = tuple(
        DecoderSampleMetadata(
            sample_id=packet.sample_id,
            valid_seqlen=packet.valid_seqlen,
            padded_seqlen=packet.padded_seqlen,
            vision_items=(),
        )
        for packet in packets
    )
    return finalize_decoder_source_window(
        source_dp_lane=lane, samples=samples, items=(), packets=packets
    )


def _state(*, source_ranks=_SOURCE_RANKS):
    lane3 = _window(3)
    lane7 = _window(7)
    manifest = build_decoder_global_manifest((lane7.metadata_manifest(), lane3.metadata_manifest()))

    def solver(sample_seqlens, total_gpus, **kwargs):
        del kwargs
        assert sample_seqlens == [(0, 4), (1, 4), (2, 4), (3, 4)]
        assert total_gpus == 4
        return ([[4], [4], [4], [4]], [], None, [[0], [1], [2], [3]])

    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=_DECODER_RANKS,
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=solver,
    )
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane=source_ranks,
        participant_ranks=_PARTICIPANTS,
    )
    ledger = routing.build_decoder_payload_route_ledger(**authority)
    return SimpleNamespace(
        lane3=lane3, lane7=lane7, manifest=manifest, plan=plan, authority=authority, ledger=ledger
    )


def _local_tensors(state, rank, dtype):
    lane_by_rank = {
        state.authority["source_rank_by_lane"][3]: state.lane3,
        state.authority["source_rank_by_lane"][7]: state.lane7,
    }
    window = lane_by_rank.get(rank)
    if window is None:
        return MappingProxyType({})
    attached = routing.attach_local_decoder_payload_tensors(
        state.ledger, **state.authority, source_window=window, global_rank=rank
    )
    return MappingProxyType(
        {key: tensor for key, tensor in attached.items() if tensor.dtype == dtype}
    )


def _prepare(state, rank, dtype, *, local_tensors=None, send=None, receive=None):
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=dtype, global_rank=rank
    )
    if send is None:
        send = torch.empty(sum(input_splits), dtype=dtype)
    if receive is None:
        receive = torch.empty(sum(output_splits), dtype=dtype)
    return transport.prepare_decoder_payload_exchange(
        state.ledger,
        **state.authority,
        dtype=dtype,
        global_rank=rank,
        local_tensors=(
            _local_tensors(state, rank, dtype) if local_tensors is None else local_tensors
        ),
        send_buffer=send,
        receive_buffer=receive,
    )


def _expected_send(state, rank, dtype, local_tensors):
    chunks = []
    for destination in _PARTICIPANTS:
        chunks.extend(
            local_tensors[entry.key].reshape(-1)
            for entry in state.ledger.entries
            if entry.dtype == dtype
            and entry.src_global_rank == rank
            and entry.dst_global_rank == destination
        )
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_pack_uses_participant_order_and_preserves_noncontiguous_sources(dtype):
    state = _state()
    local = _local_tensors(state, 70, dtype)
    assert local and all(not tensor.is_contiguous() for tensor in local.values())

    prepared = _prepare(state, 70, dtype, local_tensors=local)

    assert prepared.participant_ranks == _PARTICIPANTS
    assert torch.equal(prepared.send_buffer, _expected_send(state, 70, dtype, local))
    assert sum(prepared.output_split_sizes) == 0


def test_receive_views_follow_source_blocks_offsets_and_manifest_shapes():
    state = _state()
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=30
    )
    receive = torch.arange(sum(output_splits), dtype=torch.int64)

    prepared = _prepare(
        state,
        30,
        torch.int64,
        local_tensors=MappingProxyType({}),
        send=torch.empty(sum(input_splits), dtype=torch.int64),
        receive=receive,
    )

    positions = {rank: index for index, rank in enumerate(_PARTICIPANTS)}
    receive_bases = []
    offset = 0
    for split in output_splits:
        receive_bases.append(offset)
        offset += split
    entries = tuple(
        entry
        for entry in state.ledger.entries
        if entry.dtype == torch.int64 and entry.dst_global_rank == 30
    )
    assert set(prepared.received_tensors) == {entry.key for entry in entries}
    intervals = []
    for entry in entries:
        start = receive_bases[positions[entry.src_global_rank]] + entry.plan_offset
        view = prepared.received_tensors[entry.key]
        assert tuple(view.shape) == (1, 4)
        assert view.storage_offset() == receive.storage_offset() + start
        assert torch.equal(view, receive[start : start + entry.element_count].view(view.shape))
        intervals.append((start, start + entry.element_count))
    assert len(set(intervals)) == len(intervals)


def test_self_route_uses_independent_send_and_receive_blocks():
    state = _state(source_ranks={3: 30, 7: 80})
    local = _local_tensors(state, 30, torch.int64)

    prepared = _prepare(state, 30, torch.int64, local_tensors=local)

    own_index = _PARTICIPANTS.index(30)
    assert prepared.input_split_sizes[own_index] > 0
    assert prepared.output_split_sizes[own_index] > 0
    assert (
        prepared.send_buffer.untyped_storage().data_ptr()
        != prepared.receive_buffer.untyped_storage().data_ptr()
    )
    assert torch.equal(prepared.send_buffer, _expected_send(state, 30, torch.int64, local))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_idle_participant_has_zero_buffers_and_no_received_views(dtype):
    prepared = _prepare(_state(), 99, dtype)

    assert prepared.input_split_sizes == (0,) * len(_PARTICIPANTS)
    assert prepared.output_split_sizes == (0,) * len(_PARTICIPANTS)
    assert prepared.send_buffer.numel() == prepared.receive_buffer.numel() == 0
    assert dict(prepared.received_tensors) == {}


def test_selected_dtype_does_not_pack_other_fields():
    state = _state()
    prepared = _prepare(state, 70, torch.float32)

    assert set(prepared.received_tensors) == set()
    assert all(
        entry.key.field_name == "loss_mask"
        for entry in state.ledger.entries
        if entry.dtype == prepared.dtype
    )


def test_carrier_is_frozen_and_hides_mutable_tensor_payloads():
    prepared = _prepare(_state(), 70, torch.int64)

    with pytest.raises(FrozenInstanceError):
        prepared.global_rank = 1
    with pytest.raises(TypeError):
        prepared.received_tensors[object()] = torch.empty(0)
    assert "send_buffer" not in repr(prepared)
    assert "receive_buffer" not in repr(prepared)
    assert "received_tensors" not in repr(prepared)


def test_forged_ledger_is_rejected_before_send_buffer_mutation():
    state = _state()
    local = _local_tensors(state, 70, torch.int64)
    send = torch.full(
        (
            sum(
                entry.element_count
                for entry in state.ledger.entries
                if entry.dtype == torch.int64 and entry.src_global_rank == 70
            ),
        ),
        -7,
        dtype=torch.int64,
    )
    forged = replace(state.ledger, entries=tuple(reversed(state.ledger.entries)))

    with pytest.raises(MdpBridgeError):
        transport.prepare_decoder_payload_exchange(
            forged,
            **state.authority,
            dtype=torch.int64,
            global_rank=70,
            local_tensors=local,
            send_buffer=send,
            receive_buffer=torch.empty(0, dtype=torch.int64),
        )
    assert torch.equal(send, torch.full_like(send, -7))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_local_tensor_keys_must_exactly_match_selected_source_routes(mutation):
    state = _state()
    local = dict(_local_tensors(state, 70, torch.int64))
    if mutation == "missing":
        local.pop(next(iter(local)))
    else:
        local[object()] = torch.empty(1, dtype=torch.int64)

    with pytest.raises(MdpBridgeError, match="exactly cover"):
        _prepare(state, 70, torch.int64, local_tensors=local)


def test_late_source_failure_does_not_partially_mutate_send_buffer():
    state = _state()
    local = dict(_local_tensors(state, 70, torch.int64))
    last_key = next(reversed(local))
    local[last_key] = local[last_key].float()
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=70
    )
    send = torch.full((sum(input_splits),), -13, dtype=torch.int64)

    with pytest.raises(MdpBridgeError):
        _prepare(
            state,
            70,
            torch.int64,
            local_tensors=local,
            send=send,
            receive=torch.empty(sum(output_splits), dtype=torch.int64),
        )
    assert torch.equal(send, torch.full_like(send, -13))


def test_manifest_device_type_is_authoritative():
    state = _state()
    local = {
        key: torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
        for key, tensor in _local_tensors(state, 70, torch.int64).items()
    }
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=70
    )

    with pytest.raises(MdpConfigurationError, match="manifest device type"):
        _prepare(
            state,
            70,
            torch.int64,
            local_tensors=local,
            send=torch.empty(sum(input_splits), dtype=torch.int64, device="meta"),
            receive=torch.empty(sum(output_splits), dtype=torch.int64, device="meta"),
        )


@pytest.mark.parametrize("mutation", ["dtype", "shape", "send-alias", "receive-alias"])
def test_source_tensor_must_match_route_and_not_alias_buffers(mutation):
    state = _state(source_ranks={3: 30, 7: 80})
    local = dict(_local_tensors(state, 30, torch.int64))
    key = next(iter(local))
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=30
    )
    send = torch.empty(sum(input_splits), dtype=torch.int64)
    receive = torch.empty(sum(output_splits), dtype=torch.int64)
    if mutation == "dtype":
        local[key] = local[key].float()
    elif mutation == "shape":
        local[key] = local[key].reshape(2, 2)
    elif mutation == "send-alias":
        local[key] = send[: local[key].numel()].view(local[key].shape)
    else:
        receive = torch.empty(max(1, local[key].numel()), dtype=torch.int64)
        local[key] = receive[: local[key].numel()].view(local[key].shape)

    with pytest.raises(MdpBridgeError):
        _prepare(state, 30, torch.int64, local_tensors=local, send=send, receive=receive)


@pytest.mark.parametrize("mutation", ["dtype", "rank", "size", "device-pair", "alias"])
def test_transport_buffers_must_match_exact_contract(mutation):
    state = _state(source_ranks={3: 30, 7: 80})
    local = _local_tensors(state, 30, torch.int64)
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=30
    )
    send = torch.empty(sum(input_splits), dtype=torch.int64)
    receive = torch.empty(sum(output_splits), dtype=torch.int64)
    if mutation == "dtype":
        send = send.float()
    elif mutation == "rank":
        send = send.reshape(2, -1)
    elif mutation == "size":
        send = torch.empty(send.numel() + 1, dtype=torch.int64)
    elif mutation == "device-pair":
        receive = torch.empty(receive.numel(), dtype=torch.int64, device="meta")
    else:
        shared = torch.empty(send.numel() + max(1, receive.numel()), dtype=torch.int64)
        send = shared[: send.numel()]
        receive = shared[send.numel() : send.numel() + receive.numel()]

    with pytest.raises(MdpConfigurationError):
        _prepare(state, 30, torch.int64, local_tensors=local, send=send, receive=receive)


def test_rank_and_dtype_are_validated_by_route_authority():
    state = _state()
    with pytest.raises(MdpConfigurationError):
        _prepare(state, 1234, torch.int64)
    with pytest.raises(MdpConfigurationError):
        transport.prepare_decoder_payload_exchange(
            state.ledger,
            **state.authority,
            dtype="int64",
            global_rank=70,
            local_tensors={},
            send_buffer=torch.empty(0, dtype=torch.int64),
            receive_buffer=torch.empty(0, dtype=torch.int64),
        )
