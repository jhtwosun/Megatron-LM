# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for preparing and synchronously exchanging Dynamic-CP decoder payloads."""

import os
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


def _strided_values(seed, *, dtype, device="cpu"):
    base = torch.arange(seed, seed + 16, dtype=dtype, device=device).reshape(2, 8)
    return base[1:2, 1::2]


def _packet(lane, order, *, device="cpu"):
    tensors = MappingProxyType(
        {
            "tokens": _strided_values(lane * 100 + order * 10, dtype=torch.int64, device=device),
            "loss_mask": _strided_values(
                lane * 100 + order * 10, dtype=torch.float32, device=device
            ),
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


def _window(lane, *, device="cpu", sample_count=2):
    packets = tuple(_packet(lane, order, device=device) for order in range(sample_count))
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


def _state(
    *,
    source_ranks=_SOURCE_RANKS,
    decoder_ranks=_DECODER_RANKS,
    participant_ranks=_PARTICIPANTS,
    device="cpu",
):
    lane3 = _window(3, device=device)
    lane7 = _window(7, device=device)
    manifest = build_decoder_global_manifest((lane7.metadata_manifest(), lane3.metadata_manifest()))

    def solver(sample_seqlens, total_gpus, **kwargs):
        del kwargs
        assert sample_seqlens == [(0, 4), (1, 4), (2, 4), (3, 4)]
        assert total_gpus == len(decoder_ranks)
        return ([[4], [4], [4], [4]], [], None, [[0], [1], [2], [3]])

    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=decoder_ranks,
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=solver,
    )
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane=source_ranks,
        participant_ranks=participant_ranks,
    )
    ledger = routing.build_decoder_payload_route_ledger(**authority)
    return SimpleNamespace(
        lane3=lane3,
        lane7=lane7,
        source_windows=MappingProxyType({3: lane3, 7: lane7}),
        manifest=manifest,
        plan=plan,
        authority=authority,
        ledger=ledger,
        device=torch.device(device),
    )


def _idle_state(*, device):
    source_window = _window(3, device=device, sample_count=4)
    manifest = build_decoder_global_manifest((source_window.metadata_manifest(),))

    def solver(sample_seqlens, total_gpus, **kwargs):
        del kwargs
        assert total_gpus == 2
        selected = sample_seqlens[:total_gpus]
        assert len(selected) == total_gpus
        return (
            [[length] for _, length in selected],
            sample_seqlens[total_gpus:],
            None,
            [[sample_id] for sample_id, _ in selected],
        )

    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=(1, 2),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=solver,
    )
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane={3: 0},
        participant_ranks=(0, 1, 2, 3),
    )
    return SimpleNamespace(
        lane3=source_window,
        lane7=None,
        source_windows=MappingProxyType({3: source_window}),
        manifest=manifest,
        plan=plan,
        authority=authority,
        ledger=routing.build_decoder_payload_route_ledger(**authority),
        device=torch.device(device),
    )


def _local_tensors(state, rank, dtype):
    lane_by_rank = {
        source_rank: state.source_windows[lane]
        for lane, source_rank in state.authority["source_rank_by_lane"].items()
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
        send = torch.empty(sum(input_splits), dtype=dtype, device=state.device)
    if receive is None:
        receive = torch.empty(sum(output_splits), dtype=dtype, device=state.device)
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
    for destination in state.authority["participant_ranks"]:
        chunks.extend(
            local_tensors[entry.key].reshape(-1)
            for entry in state.ledger.entries
            if entry.dtype == dtype
            and entry.src_global_rank == rank
            and entry.dst_global_rank == destination
        )
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=dtype, device=state.device)


class _FakeGroup:
    def __init__(self, participant_ranks, global_rank):
        self.participant_ranks = participant_ranks
        self.global_rank = global_rank

    def size(self):
        return len(self.participant_ranks)

    def rank(self):
        return self.participant_ranks.index(self.global_rank)


def _execute(prepared, *, group=None, group_ranks_getter=None, all_to_all_single=None):
    group = group or _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    kwargs = {"group": group}
    if group_ranks_getter is not None:
        kwargs["group_ranks_getter"] = group_ranks_getter
    else:
        kwargs["group_ranks_getter"] = lambda selected: list(selected.participant_ranks)
    if all_to_all_single is not None:
        kwargs["all_to_all_single"] = all_to_all_single
    else:
        kwargs["all_to_all_single"] = lambda *_args, **_kwargs: None
    return transport.execute_decoder_payload_exchange(prepared, **kwargs)


def _replace_preserving_authority(prepared, **changes):
    authority = prepared._authority
    forged = replace(prepared, **changes)
    object.__setattr__(forged, "_authority", authority)
    return forged


def test_execute_calls_one_sync_collective_with_exact_buffers_splits_and_group():
    prepared = _prepare(_state(), 30, torch.int64)
    group = _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    calls = []

    def all_to_all_single(output, input, **kwargs):
        calls.append((output, input, kwargs))
        output.copy_(torch.arange(output.numel(), dtype=output.dtype))

    received = _execute(prepared, group=group, all_to_all_single=all_to_all_single)

    assert received is prepared.received_tensors
    assert calls == [
        (
            prepared.receive_buffer,
            prepared.send_buffer,
            {
                "output_split_sizes": list(prepared.output_split_sizes),
                "input_split_sizes": list(prepared.input_split_sizes),
                "group": group,
                "async_op": False,
            },
        )
    ]
    assert any(torch.count_nonzero(tensor).item() for tensor in received.values())


@pytest.mark.parametrize(
    "mutation",
    (
        "carrier",
        "participant-order",
        "global-rank",
        "input-splits",
        "output-splits",
        "send-size",
        "receive-view",
    ),
)
def test_execute_revalidates_exact_carrier_geometry_before_collective(mutation):
    prepared = _prepare(_state(), 30, torch.int64)
    original_participants = prepared.participant_ranks
    if mutation == "carrier":
        prepared = object()
    elif mutation == "participant-order":
        prepared = replace(prepared, participant_ranks=tuple(reversed(prepared.participant_ranks)))
    elif mutation == "global-rank":
        prepared = replace(prepared, global_rank=123)
    elif mutation == "input-splits":
        prepared = replace(prepared, input_split_sizes=(*prepared.input_split_sizes[:-1], 1))
    elif mutation == "output-splits":
        prepared = replace(prepared, output_split_sizes=(*prepared.output_split_sizes[:-1], 1))
    elif mutation == "send-size":
        prepared = replace(prepared, send_buffer=torch.empty(1, dtype=prepared.dtype))
    else:
        key = next(iter(prepared.received_tensors))
        prepared = replace(
            prepared,
            received_tensors=MappingProxyType(
                {**dict(prepared.received_tensors), key: prepared.receive_buffer[:1]}
            ),
        )
    calls = []
    group = _FakeGroup(original_participants, 30)

    with pytest.raises((MdpBridgeError, MdpConfigurationError)):
        _execute(
            prepared,
            group=group,
            group_ranks_getter=lambda _group: list(original_participants),
            all_to_all_single=lambda *_args, **_kwargs: calls.append(True),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("split_name", "rank"), (("input_split_sizes", 70), ("output_split_sizes", 30))
)
def test_execute_rejects_total_preserving_peer_split_relocation(split_name, rank):
    prepared = _prepare(_state(), rank, torch.int64)
    splits = list(getattr(prepared, split_name))
    source = next(index for index, split in enumerate(splits) if split > 0)
    idle = next(index for index, split in enumerate(splits) if split == 0)
    splits[source] -= 1
    splits[idle] += 1
    forged = _replace_preserving_authority(prepared, **{split_name: tuple(splits)})
    calls = []

    with pytest.raises(MdpBridgeError, match="authority snapshot"):
        _execute(forged, all_to_all_single=lambda *_args, **_kwargs: calls.append(True))
    assert calls == []


@pytest.mark.parametrize("mutation", ("typed-key", "same-interval-reshape"))
def test_execute_rejects_receive_descriptor_mutation_with_exact_partition(mutation):
    prepared = _prepare(_state(), 30, torch.int64)
    views = list(prepared.received_tensors.items())
    key, tensor = views[0]
    if mutation == "typed-key":
        views[0] = (replace(key, sample_id=GlobalSampleId(99, 99)), tensor)
    else:
        views[0] = (key, tensor.reshape(-1))
    forged = _replace_preserving_authority(prepared, received_tensors=MappingProxyType(dict(views)))
    calls = []

    with pytest.raises(MdpBridgeError, match="authority snapshot"):
        _execute(forged, all_to_all_single=lambda *_args, **_kwargs: calls.append(True))
    assert calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("ranks-reordered", "rank order"),
        ("ranks-bool", "rank order"),
        ("size", "group size"),
        ("local-rank", "local rank"),
    ),
)
def test_execute_rejects_native_group_geometry_mismatch(mutation, message):
    prepared = _prepare(_state(), 30, torch.int64)
    group = _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    ranks = list(prepared.participant_ranks)
    if mutation == "ranks-reordered":
        ranks.reverse()
    elif mutation == "ranks-bool":
        ranks[0] = True
    elif mutation == "size":
        group.size = lambda: len(ranks) - 1
    else:
        group.rank = lambda: 0

    with pytest.raises(MdpConfigurationError, match=message):
        _execute(prepared, group=group, group_ranks_getter=lambda _group: ranks)


@pytest.mark.parametrize("phase", ("group-ranks", "group-size", "group-rank", "collective"))
def test_execute_normalizes_ordinary_query_and_collective_errors_with_cause(phase):
    prepared = _prepare(_state(), 30, torch.int64)
    group = _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    error = RuntimeError(phase)

    def fail():
        raise error

    kwargs = {}
    if phase == "group-ranks":
        kwargs["group_ranks_getter"] = lambda _group: fail()
    elif phase == "group-size":
        group.size = fail
    elif phase == "group-rank":
        group.rank = fail
    else:
        kwargs["all_to_all_single"] = lambda *_args, **_kwargs: fail()

    expected = MdpBridgeError if phase == "collective" else MdpConfigurationError
    with pytest.raises(expected) as caught:
        _execute(prepared, group=group, **kwargs)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("phase", ("group-ranks", "collective"))
def test_execute_does_not_catch_base_exception(phase):
    prepared = _prepare(_state(), 30, torch.int64)

    def fail(*_args, **_kwargs):
        raise KeyboardInterrupt

    kwargs = {"group_ranks_getter": fail} if phase == "group-ranks" else {"all_to_all_single": fail}
    with pytest.raises(KeyboardInterrupt):
        _execute(prepared, **kwargs)


@pytest.mark.parametrize("dependency", ("group_ranks_getter", "all_to_all_single"))
def test_execute_requires_callable_injected_dependencies(dependency):
    prepared = _prepare(_state(), 30, torch.int64)
    kwargs = {dependency: object()}

    with pytest.raises(MdpConfigurationError, match="callable"):
        transport.execute_decoder_payload_exchange(
            prepared, group=_FakeGroup(prepared.participant_ranks, prepared.global_rank), **kwargs
        )


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
    receive = torch.arange(sum(output_splits), dtype=torch.int64, device=state.device)

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


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
_WORLD4_PARTICIPANTS = (0, 1, 2, 3)
_WORLD4_DECODER_RANKS = (2, 0, 3, 1)

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def payload_group():
        Utils.initialize_model_parallel()
        group = torch.distributed.new_group(ranks=list(_WORLD4_PARTICIPANTS), backend="nccl")
        yield group
        torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
@pytest.mark.parametrize("dtype", (torch.int64, torch.float32))
def test_world4_nccl_exchange_preserves_self_and_remote_payloads(dtype, payload_group):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(
        source_ranks={3: 0, 7: 2},
        decoder_ranks=_WORLD4_DECODER_RANKS,
        participant_ranks=_WORLD4_PARTICIPANTS,
        device=device,
    )
    prepared = _prepare(state, rank, dtype)

    assert tuple(torch.distributed.get_process_group_ranks(payload_group)) == (_WORLD4_PARTICIPANTS)
    received = transport.execute_decoder_payload_exchange(prepared, group=payload_group)

    assert received is prepared.received_tensors
    packets = {
        packet.sample_id: packet
        for window in (state.lane3, state.lane7)
        for packet in window.packets
    }
    destination_entries = tuple(
        entry
        for entry in state.ledger.entries
        if entry.dtype == dtype and entry.dst_global_rank == rank
    )
    assert set(received) == {entry.key for entry in destination_entries}
    for entry in destination_entries:
        expected = packets[entry.key.sample_id].tensor_fields[entry.key.field_name]
        torch.testing.assert_close(received[entry.key], expected, rtol=0, atol=0)
    if rank == 0:
        assert any(entry.src_global_rank == entry.dst_global_rank for entry in destination_entries)
    else:
        assert all(entry.src_global_rank != entry.dst_global_rank for entry in destination_entries)
    assert (prepared.send_buffer.numel() > 0) == (rank in (0, 2))
    assert prepared.receive_buffer.numel() > 0


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_exchange_includes_source_only_destination_only_and_idle_ranks(payload_group):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _idle_state(device=device)
    prepared = _prepare(state, rank, torch.int64)
    calls = 0

    def tracked_all_to_all_single(*args, **kwargs):
        nonlocal calls
        calls += 1
        return torch.distributed.all_to_all_single(*args, **kwargs)

    received = transport.execute_decoder_payload_exchange(
        prepared, group=payload_group, all_to_all_single=tracked_all_to_all_single
    )

    assert calls == 1
    if rank == 0:
        assert prepared.send_buffer.numel() > 0
        assert prepared.receive_buffer.numel() == 0
    elif rank in (1, 2):
        assert prepared.send_buffer.numel() == 0
        assert prepared.receive_buffer.numel() > 0
    else:
        assert prepared.send_buffer.numel() == prepared.receive_buffer.numel() == 0
    packets = {
        packet.sample_id: packet
        for window in state.source_windows.values()
        for packet in window.packets
    }
    destination_entries = tuple(
        entry
        for entry in state.ledger.entries
        if entry.dtype == torch.int64 and entry.dst_global_rank == rank
    )
    assert set(received) == {entry.key for entry in destination_entries}
    for entry in destination_entries:
        expected = packets[entry.key.sample_id].tensor_fields[entry.key.field_name]
        torch.testing.assert_close(received[entry.key], expected, rtol=0, atol=0)
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=payload_group)
    assert completion.item() == 4
