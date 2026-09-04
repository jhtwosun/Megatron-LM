# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused caller-buffer contracts for Dynamic-CP bridge transport."""

import os
from dataclasses import FrozenInstanceError
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import megatron.core.mdp.dynamic_cp_bridge as bridge
import megatron.core.mdp.dynamic_cp_bridge_transport as transport
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
    build_decoder_global_manifest,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderSampleMetadata,
    EncoderVisionItemMetadata,
    build_decoder_dynamic_plan,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError

_DECODER_RANKS = (30, 10, 40, 20)
_PARTICIPANTS = (80, 30, 70, 20, 10, 40, 99)


def _packet(lane, order, valid, padded):
    tensor = torch.arange(padded, dtype=torch.int64).reshape(1, padded)
    spec = DecoderTensorFieldSpec(
        name="input_ids", dtype=tensor.dtype, shape=tuple(tensor.shape), device_type="cpu"
    )
    header = DecoderPayloadHeaderV1(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        source_dp_lane=lane,
        local_sample_order=order,
        valid_seqlen=valid,
        padded_seqlen=padded,
        tensor_field_count=1,
        none_field_count=2,
        position_components_or_minus_one=-1,
    )
    return DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=GlobalSampleId(lane, order),
        valid_seqlen=valid,
        padded_seqlen=padded,
        header=header.to_wire_tuple(),
        field_specs=(spec,),
        tensor_fields=MappingProxyType({"input_ids": tensor}),
        none_fields=("position_ids", "attention_mask"),
    )


def _source_window(lane, *, text_only=False):
    samples = []
    items = []
    packets = []
    for order, (valid, padded) in enumerate(((4, 6), (5, 7))):
        sample_id = GlobalSampleId(lane, order)
        sample_items = []
        if not text_only and order == 0:
            rows = (2, 1) if lane == 3 else (3,)
            decoder_start = 0
            for ordinal, output_rows in enumerate(rows):
                item_id = GlobalVisionItemId(lane, ordinal)
                sample_items.append(EncoderVisionItemMetadata(item_id, sample_id, ordinal))
                items.append(
                    DecoderVisionItemMetadata(
                        item_id=item_id,
                        sample_id=sample_id,
                        image_ordinal=ordinal,
                        grid_thw=(1, 1, output_rows),
                        output_rows=output_rows,
                        decoder_offsets=tuple(range(decoder_start, decoder_start + output_rows)),
                    )
                )
                decoder_start += output_rows
        samples.append(DecoderSampleMetadata(sample_id, valid, padded, tuple(sample_items)))
        packets.append(_packet(lane, order, valid, padded))
    return finalize_decoder_source_window(
        source_dp_lane=lane,
        samples=tuple(reversed(samples)),
        items=tuple(reversed(items)),
        packets=tuple(reversed(packets)),
    )


class _TwoWaveCp2Solver:
    def __init__(self):
        self.calls = 0

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        assert (total_gpus, max_seq_len_per_rank, min_cp_size) == (4, 4, 1)
        self.calls += 1
        if self.calls == 1:
            return ([[6], [6], [7], [7]], [(2, 6), (3, 7)], None, [[0], [0], [1], [1]])
        return ([[6], [6], [7], [7]], [], None, [[2], [2], [3], [3]])


class _World4Cp2Solver:
    def __init__(self):
        self.calls = 0

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        assert (total_gpus, max_seq_len_per_rank, min_cp_size) == (2, 7, 1)
        self.calls += 1
        if self.calls == 1:
            assert sample_seqlens == [(0, 6), (1, 7), (2, 6), (3, 7)]
            return ([[6, 7], [6, 7]], [(2, 6), (3, 7)], None, [[0, 1], [0, 1]])
        assert sample_seqlens == [(2, 6), (3, 7)]
        return ([[6, 7], [6, 7]], [], None, [[2, 3], [2, 3]])


def _state(
    *,
    dtype=torch.float32,
    width=4,
    producers=None,
    participants=_PARTICIPANTS,
    text_only=False,
    decoder_ranks=_DECODER_RANKS,
    max_seqlen_per_rank=4,
    solver=None,
    device=None,
):
    device = torch.device("cpu") if device is None else torch.device(device)
    lane3 = _source_window(3, text_only=text_only)
    lane7 = _source_window(7, text_only=text_only)
    manifest = build_decoder_global_manifest((lane7.metadata_manifest(), lane3.metadata_manifest()))
    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=decoder_ranks,
        max_seqlen_per_rank=max_seqlen_per_rank,
        minimum_cp_size=1,
        solver=_TwoWaveCp2Solver() if solver is None else solver,
    )
    if text_only:
        producers = {}
    elif producers is None:
        producers = {
            GlobalVisionItemId(3, 0): 30,
            GlobalVisionItemId(3, 1): 30,
            GlobalVisionItemId(7, 0): 80,
        }
    rows = {item.item_id: item.output_rows for item in manifest.items}
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        producer_rank_by_item=producers,
        output_rows_by_item=rows,
        width=width,
        dtype=dtype,
        participant_ranks=participants,
    )
    embedding, gradient = bridge.build_dynamic_bridge_ledgers(**authority)
    return SimpleNamespace(
        plan=plan,
        manifest=manifest,
        authority=authority,
        embedding=embedding,
        gradient=gradient,
        device=device,
    )


def _selected(state, phase):
    return state.embedding if phase is BridgePhase.EMBEDDING else state.gradient


def _splits(state, phase, rank):
    selected = _selected(state, phase)
    reverse = state.gradient if phase is BridgePhase.EMBEDDING else state.embedding
    return bridge.dynamic_bridge_split_sizes(
        selected, reverse_ledger=reverse, global_rank=rank, **state.authority
    )


def _local_tensors(state, phase, rank):
    result = {}
    embedding_values = {}
    for index, entry in enumerate(
        item for item in _selected(state, phase).entries if item.src_global_rank == rank
    ):
        rows = state.authority["output_rows_by_item"][entry.key.item_id]
        if phase is BridgePhase.EMBEDDING and entry.key.item_id in embedding_values:
            value = embedding_values[entry.key.item_id]
        else:
            base = torch.arange(rows * state.authority["width"] * 2, dtype=state.authority["dtype"])
            base = base.to(state.device)
            value = base.view(rows, state.authority["width"] * 2)[:, ::2]
            source_seed = 0
            if phase is BridgePhase.GRADIENT:
                source_seed = 16 * (state.authority["participant_ranks"].index(rank) + 1)
            value.add_(100 * (index + 1) + source_seed)
            if phase is BridgePhase.EMBEDDING:
                embedding_values[entry.key.item_id] = value
        result[entry.key] = value
    return result


def _prepare(state, phase, rank, *, local_tensors=None, send=None, receive=None):
    inputs, outputs = _splits(state, phase, rank)
    if local_tensors is None:
        local_tensors = _local_tensors(state, phase, rank)
    if send is None:
        send = torch.full((sum(inputs),), -1, dtype=state.authority["dtype"], device=state.device)
    if receive is None:
        receive = torch.arange(sum(outputs), dtype=state.authority["dtype"], device=state.device)
    prepared = transport.prepare_dynamic_bridge_exchange(
        _selected(state, phase),
        state.gradient if phase is BridgePhase.EMBEDDING else state.embedding,
        global_rank=rank,
        local_tensors=local_tensors,
        send_buffer=send,
        receive_buffer=receive,
        **state.authority,
    )
    return prepared, local_tensors


def _expected_digest(state, phase):
    return transport.build_dynamic_bridge_route_authority_digest(
        _selected(state, phase),
        state.gradient if phase is BridgePhase.EMBEDDING else state.embedding,
        **state.authority,
    )


def _bases(splits):
    result = []
    cursor = 0
    for split in splits:
        result.append(cursor)
        cursor += split
    return result


@pytest.mark.parametrize("phase", [BridgePhase.EMBEDDING, BridgePhase.GRADIENT])
def test_prepare_packs_exact_participant_blocks_for_both_phases(phase):
    state = _state()
    prepared, local = _prepare(state, phase, 30)
    participant_index = {rank: index for index, rank in enumerate(_PARTICIPANTS)}
    bases = _bases(prepared.input_split_sizes)

    assert prepared.phase is phase
    assert prepared.participant_ranks == _PARTICIPANTS
    assert all(not tensor.is_contiguous() for tensor in local.values())
    if phase is BridgePhase.EMBEDDING:
        by_item = {}
        for key, tensor in local.items():
            by_item.setdefault(key.item_id, []).append(tensor)
        repeated = next(values for values in by_item.values() if len(values) > 1)
        assert all(value is repeated[0] for value in repeated)
    torch.testing.assert_close(
        prepared.receive_buffer, torch.arange(prepared.receive_buffer.numel(), dtype=prepared.dtype)
    )
    for entry in _selected(state, phase).entries:
        if entry.src_global_rank != 30:
            continue
        start = bases[participant_index[entry.dst_global_rank]] + entry.plan_offset
        actual = prepared.send_buffer.narrow(0, start, entry.element_count)
        torch.testing.assert_close(actual, local[entry.key].reshape(-1))
    assert transport.validate_prepared_dynamic_bridge_exchange(prepared) is prepared


def test_embedding_and_gradient_reverse_source_endpoint_roles_and_idle_rank():
    state = _state()
    embedding = {rank: _splits(state, BridgePhase.EMBEDDING, rank) for rank in (80, 30, 10, 99)}
    gradient = {rank: _splits(state, BridgePhase.GRADIENT, rank) for rank in (80, 30, 10, 99)}

    assert sum(embedding[80][0]) > 0 and sum(embedding[80][1]) == 0
    assert sum(embedding[10][0]) == 0 and sum(embedding[10][1]) > 0
    assert all(sum(parts) > 0 for parts in embedding[30])
    rank30_index = _PARTICIPANTS.index(30)
    assert embedding[30][0][rank30_index] > 0
    assert embedding[30][1][rank30_index] > 0
    assert gradient[80] == tuple(reversed(embedding[80]))
    assert gradient[30] == tuple(reversed(embedding[30]))
    assert gradient[10] == tuple(reversed(embedding[10]))
    assert embedding[99] == gradient[99] == ((0,) * 7, (0,) * 7)

    idle, local = _prepare(state, BridgePhase.EMBEDDING, 99)
    assert local == {}
    assert idle.send_buffer.numel() == idle.receive_buffer.numel() == 0
    assert dict(idle.received_tensors) == {}


@pytest.mark.parametrize("phase", [BridgePhase.EMBEDDING, BridgePhase.GRADIENT])
def test_receive_views_have_exact_keys_shapes_offsets_and_storage(phase):
    state = _state()
    prepared, _ = _prepare(state, phase, 30)
    selected = _selected(state, phase)
    participant_index = {rank: index for index, rank in enumerate(_PARTICIPANTS)}
    bases = _bases(prepared.output_split_sizes)
    expected = [entry for entry in selected.entries if entry.dst_global_rank == 30]
    expected_order = tuple(
        entry.key
        for entry in sorted(
            expected,
            key=lambda entry: (participant_index[entry.src_global_rank], entry.plan_offset),
        )
    )

    assert type(prepared.received_tensors) is type(MappingProxyType({}))
    assert tuple(prepared.received_tensors) == expected_order
    assert set(prepared.received_tensors) == {entry.key for entry in expected}
    intervals = []
    for entry in expected:
        view = prepared.received_tensors[entry.key]
        offset = bases[participant_index[entry.src_global_rank]] + entry.plan_offset
        shape = (state.authority["output_rows_by_item"][entry.key.item_id], 4)
        assert tuple(view.shape) == shape
        assert (
            view.untyped_storage().data_ptr()
            == prepared.receive_buffer.untyped_storage().data_ptr()
        )
        assert view.storage_offset() - prepared.receive_buffer.storage_offset() == offset
        torch.testing.assert_close(
            view.reshape(-1), prepared.receive_buffer[offset : offset + entry.element_count]
        )
        intervals.append((offset, offset + entry.element_count))
    assert sorted(intervals)[0][0] == 0
    assert sorted(intervals)[-1][1] == prepared.receive_buffer.numel()


def test_gradient_receive_keys_keep_multi_endpoint_fanout_distinct():
    state = _state()
    prepared, _ = _prepare(state, BridgePhase.GRADIENT, 30)
    keys_by_item = {}
    for key in prepared.received_tensors:
        keys_by_item.setdefault(key.item_id, []).append(key)

    repeated = next(keys for keys in keys_by_item.values() if len(keys) > 1)
    assert len({key.endpoint_rank for key in repeated}) == len(repeated)


def test_carrier_is_frozen_opaque_and_has_fixed_common_authority_digest():
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)

    assert type(prepared.route_authority_digest) is bytes
    assert len(prepared.route_authority_digest) == 16
    assert transport.route_authority_digest(prepared) == prepared.route_authority_digest
    assert "send_buffer" not in repr(prepared)
    assert "received_tensors" not in repr(prepared)
    with pytest.raises(FrozenInstanceError):
        prepared.global_rank = 80
    with pytest.raises(TypeError):
        prepared.received_tensors[next(iter(prepared.received_tensors))] = torch.empty(1)


@pytest.mark.parametrize(
    "phase,rank",
    [
        (BridgePhase.EMBEDDING, 80),
        (BridgePhase.EMBEDDING, 30),
        (BridgePhase.EMBEDDING, 99),
        (BridgePhase.GRADIENT, 80),
        (BridgePhase.GRADIENT, 30),
        (BridgePhase.GRADIENT, 99),
    ],
)
def test_public_authority_rederivation_matches_every_rank(phase, rank):
    state = _state()
    prepared, _ = _prepare(state, phase, rank)

    assert prepared.route_authority_digest == _expected_digest(state, phase)


@pytest.mark.parametrize("mutation", ["splits", "key", "shape", "buffer", "digest"])
def test_forged_public_geometry_is_rejected_by_private_seal(mutation):
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)
    if mutation == "splits":
        splits = list(prepared.input_split_sizes)
        source = next(index for index, value in enumerate(splits) if value)
        target = next(index for index, value in enumerate(splits) if not value)
        splits[source] -= 1
        splits[target] += 1
        object.__setattr__(prepared, "input_split_sizes", tuple(splits))
    elif mutation == "key":
        views = dict(prepared.received_tensors)
        key, value = next(iter(views.items()))
        del views[key]
        views[bridge.DynamicBridgeKey(key.item_id, 20)] = value
        object.__setattr__(prepared, "received_tensors", MappingProxyType(views))
    elif mutation == "shape":
        views = dict(prepared.received_tensors)
        key, value = next(iter(views.items()))
        views[key] = value.view(-1, 1)
        object.__setattr__(prepared, "received_tensors", MappingProxyType(views))
    elif mutation == "buffer":
        object.__setattr__(prepared, "send_buffer", prepared.send_buffer.clone())
    else:
        object.__setattr__(prepared, "route_authority_digest", b"x" * 16)

    with pytest.raises((MdpBridgeError, MdpConfigurationError)):
        transport.validate_prepared_dynamic_bridge_exchange(prepared)


@pytest.mark.parametrize(
    "failure", ["shape", "dtype", "requires_grad", "send_alias", "receive_alias"]
)
def test_late_local_tensor_validation_never_mutates_send_buffer(failure):
    state = _state()
    phase = BridgePhase.EMBEDDING
    inputs, outputs = _splits(state, phase, 30)
    send = torch.full((sum(inputs),), -123, dtype=torch.float32)
    receive = torch.empty(sum(outputs), dtype=torch.float32)
    local = _local_tensors(state, phase, 30)
    key = tuple(local)[-1]
    if failure == "shape":
        local[key] = local[key].reshape(-1)
    elif failure == "dtype":
        local[key] = local[key].to(torch.float64)
    elif failure == "requires_grad":
        local[key] = local[key].requires_grad_()
    elif failure == "send_alias":
        rows = state.authority["output_rows_by_item"][key.item_id]
        local[key] = send[: rows * 4].view(rows, 4)
    else:
        rows = state.authority["output_rows_by_item"][key.item_id]
        local[key] = receive[: rows * 4].view(rows, 4)
    before = send.clone()

    with pytest.raises(MdpConfigurationError):
        _prepare(state, phase, 30, local_tensors=local, send=send, receive=receive)
    torch.testing.assert_close(send, before)


@pytest.mark.parametrize("failure", ["missing", "extra", "not_mapping", "wrong_send_size", "alias"])
def test_malformed_local_mapping_and_buffers_fail_closed(failure):
    state = _state()
    phase = BridgePhase.EMBEDDING
    inputs, outputs = _splits(state, phase, 30)
    local = _local_tensors(state, phase, 30)
    send = torch.empty(sum(inputs), dtype=torch.float32)
    receive = torch.empty(sum(outputs), dtype=torch.float32)
    if failure == "missing":
        del local[next(iter(local))]
    elif failure == "extra":
        local[bridge.DynamicBridgeKey(GlobalVisionItemId(7, 0), 99)] = torch.empty(3, 4)
    elif failure == "not_mapping":
        local = tuple(local.items())
    elif failure == "wrong_send_size":
        send = torch.empty(sum(inputs) + 1, dtype=torch.float32)
    else:
        backing = torch.empty(sum(inputs) + sum(outputs), dtype=torch.float32)
        send = backing[: sum(inputs)]
        receive = backing[: sum(outputs)]

    with pytest.raises(MdpConfigurationError):
        _prepare(state, phase, 30, local_tensors=local, send=send, receive=receive)


def test_authority_digest_binds_phase_dtype_width_participants_and_producer_mapping():
    base = _state()
    embedding, _ = _prepare(base, BridgePhase.EMBEDDING, 30)
    gradient, _ = _prepare(base, BridgePhase.GRADIENT, 30)
    bf16_state = _state(dtype=torch.bfloat16)
    bf16, _ = _prepare(bf16_state, BridgePhase.EMBEDDING, 30)
    wide_state = _state(width=5)
    wide, _ = _prepare(wide_state, BridgePhase.EMBEDDING, 30)
    reordered_state = _state(participants=tuple(reversed(_PARTICIPANTS)))
    reordered, _ = _prepare(reordered_state, BridgePhase.EMBEDDING, 30)
    moved = _state(
        producers={
            GlobalVisionItemId(3, 0): 70,
            GlobalVisionItemId(3, 1): 70,
            GlobalVisionItemId(7, 0): 80,
        }
    )
    moved_embedding, _ = _prepare(moved, BridgePhase.EMBEDDING, 30)

    digests = {
        embedding.route_authority_digest,
        gradient.route_authority_digest,
        bf16.route_authority_digest,
        wide.route_authority_digest,
        reordered.route_authority_digest,
        moved_embedding.route_authority_digest,
    }
    assert len(digests) == 6
    assert base.plan.digest not in digests
    assert base.manifest.digest not in digests
    assert embedding.route_authority_digest == _expected_digest(base, BridgePhase.EMBEDDING)
    assert gradient.route_authority_digest == _expected_digest(base, BridgePhase.GRADIENT)
    assert bf16.route_authority_digest == _expected_digest(bf16_state, BridgePhase.EMBEDDING)
    assert moved_embedding.route_authority_digest == _expected_digest(moved, BridgePhase.EMBEDDING)


def test_empty_embedding_and_gradient_authority_digests_remain_distinct():
    state = _state(text_only=True)
    embedding, _ = _prepare(state, BridgePhase.EMBEDDING, 99)
    gradient, _ = _prepare(state, BridgePhase.GRADIENT, 99)

    assert embedding.input_split_sizes == embedding.output_split_sizes == (0,) * 7
    assert gradient.input_split_sizes == gradient.output_split_sizes == (0,) * 7
    assert embedding.route_authority_digest != gradient.route_authority_digest


def test_foreign_ledger_authority_is_rejected_before_send_mutation():
    state = _state()
    foreign = _state(
        producers={
            GlobalVisionItemId(3, 0): 70,
            GlobalVisionItemId(3, 1): 70,
            GlobalVisionItemId(7, 0): 80,
        }
    )
    inputs, outputs = _splits(state, BridgePhase.EMBEDDING, 30)
    send = torch.full((sum(inputs),), -91, dtype=torch.float32)
    receive = torch.empty(sum(outputs), dtype=torch.float32)

    with pytest.raises(MdpBridgeError, match="exactly match plan authority"):
        transport.prepare_dynamic_bridge_exchange(
            foreign.embedding,
            foreign.gradient,
            global_rank=30,
            local_tensors=_local_tensors(state, BridgePhase.EMBEDDING, 30),
            send_buffer=send,
            receive_buffer=receive,
            **state.authority,
        )
    assert torch.equal(send, torch.full_like(send, -91))


def test_same_phase_reverse_is_rejected_before_buffer_mutation():
    state = _state()
    inputs, outputs = _splits(state, BridgePhase.EMBEDDING, 30)
    send = torch.full((sum(inputs),), -17, dtype=torch.float32)

    with pytest.raises(MdpBridgeError, match="reverse phase"):
        transport.prepare_dynamic_bridge_exchange(
            state.embedding,
            state.embedding,
            global_rank=30,
            local_tensors=_local_tensors(state, BridgePhase.EMBEDDING, 30),
            send_buffer=send,
            receive_buffer=torch.empty(sum(outputs), dtype=torch.float32),
            **state.authority,
        )
    assert torch.equal(send, torch.full_like(send, -17))


def test_overflowing_authority_is_rejected_before_buffer_mutation():
    state = _state()
    inputs, outputs = _splits(state, BridgePhase.EMBEDDING, 30)
    send = torch.full((sum(inputs),), -29, dtype=torch.float32)
    authority = dict(state.authority)
    authority["width"] = 2**63 - 1

    with pytest.raises(MdpConfigurationError, match="signed int64"):
        transport.prepare_dynamic_bridge_exchange(
            state.embedding,
            state.gradient,
            global_rank=30,
            local_tensors=_local_tensors(state, BridgePhase.EMBEDDING, 30),
            send_buffer=send,
            receive_buffer=torch.empty(sum(outputs), dtype=torch.float32),
            **authority,
        )
    assert torch.equal(send, torch.full_like(send, -29))


class _FakeGroup:
    def __init__(self, participant_ranks, global_rank):
        self.participant_ranks = participant_ranks
        self.global_rank = global_rank

    def size(self):
        return len(self.participant_ranks)

    def rank(self):
        return self.participant_ranks.index(self.global_rank)


def _execute(prepared, *, group=None, group_ranks_getter=None, all_to_all_single=None):
    if group is None:
        group = _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    if group_ranks_getter is None:
        group_ranks_getter = lambda _selected: list(prepared.participant_ranks)
    if all_to_all_single is None:
        all_to_all_single = lambda *_args, **_kwargs: None
    return getattr(transport, "execute_dynamic_bridge_exchange")(
        prepared,
        group=group,
        group_ranks_getter=group_ranks_getter,
        all_to_all_single=all_to_all_single,
    )


@pytest.mark.parametrize("phase", [BridgePhase.EMBEDDING, BridgePhase.GRADIENT])
def test_execute_calls_exact_collective_once_and_returns_live_views(phase):
    prepared, _ = _prepare(_state(), phase, 30)
    group = _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    calls = []

    def all_to_all_single(output, input_, **kwargs):
        calls.append((output, input_, kwargs))
        output.copy_(torch.arange(output.numel(), dtype=output.dtype) + 700)

    received = _execute(prepared, group=group, all_to_all_single=all_to_all_single)

    assert received is prepared.received_tensors
    assert len(calls) == 1
    output, input_, kwargs = calls[0]
    assert output is prepared.receive_buffer
    assert input_ is prepared.send_buffer
    assert kwargs == {
        "output_split_sizes": list(prepared.output_split_sizes),
        "input_split_sizes": list(prepared.input_split_sizes),
        "group": group,
        "async_op": False,
    }
    for view in received.values():
        offset = view.storage_offset() - prepared.receive_buffer.storage_offset()
        torch.testing.assert_close(
            view.reshape(-1),
            torch.arange(offset + 700, offset + 700 + view.numel(), dtype=view.dtype),
        )


@pytest.mark.parametrize(
    "phase,rank,role",
    [
        (BridgePhase.EMBEDDING, 30, "self"),
        (BridgePhase.GRADIENT, 30, "self"),
        (BridgePhase.EMBEDDING, 99, "idle"),
        (BridgePhase.GRADIENT, 99, "idle"),
    ],
)
def test_execute_self_and_all_zero_idle_ranks_still_call_once(phase, rank, role):
    prepared, _ = _prepare(_state(), phase, rank)
    calls = []

    received = _execute(prepared, all_to_all_single=lambda *_args, **_kwargs: calls.append(True))

    assert received is prepared.received_tensors
    assert calls == [True]
    local_index = prepared.participant_ranks.index(rank)
    if role == "self":
        assert prepared.input_split_sizes[local_index] > 0
        assert prepared.output_split_sizes[local_index] > 0
    else:
        assert prepared.input_split_sizes == prepared.output_split_sizes == (0,) * 7


@pytest.mark.parametrize("failure", ["seal", "group-getter", "collective"])
def test_execute_validates_seal_and_both_dependencies_before_side_effects(failure):
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)
    events = []
    group_ranks_getter = lambda _group: (events.append("group") or list(_PARTICIPANTS))
    all_to_all_single = lambda *_args, **_kwargs: events.append("collective")
    expected_error = MdpBridgeError if failure == "seal" else MdpConfigurationError
    if failure == "seal":
        object.__setattr__(prepared, "route_authority_digest", b"x" * 16)
    elif failure == "group-getter":
        group_ranks_getter = object()
    else:
        all_to_all_single = object()

    with pytest.raises(expected_error):
        _execute(
            prepared, group_ranks_getter=group_ranks_getter, all_to_all_single=all_to_all_single
        )
    assert events == []


@pytest.mark.parametrize(
    "actual_ranks",
    [
        tuple(reversed(_PARTICIPANTS)),
        (80, 30, 70, 20, 10, 40, True),
        (80, 30, 70, 20, 10, 40, 99.0),
        (80, 30, 70, 20, 10, 40, 40),
        _PARTICIPANTS[:-1],
        None,
    ],
)
def test_execute_rejects_non_authoritative_native_rank_order(actual_ranks):
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)
    calls = []

    with pytest.raises(MdpConfigurationError, match="rank"):
        _execute(
            prepared,
            group_ranks_getter=lambda _group: actual_ranks,
            all_to_all_single=lambda *_args, **_kwargs: calls.append(True),
        )
    assert calls == []


@pytest.mark.parametrize("size", [True, 7.0, 6, -1])
def test_execute_rejects_malformed_native_group_size(size):
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)
    group = SimpleNamespace(size=lambda: size, rank=lambda: 1)
    calls = []

    with pytest.raises(MdpConfigurationError, match="size"):
        _execute(
            prepared, group=group, all_to_all_single=lambda *_args, **_kwargs: calls.append(True)
        )
    assert calls == []


@pytest.mark.parametrize("local_rank", [True, 1.0, 0, -1, 7])
def test_execute_rejects_malformed_native_local_rank(local_rank):
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)
    group = SimpleNamespace(size=lambda: 7, rank=lambda: local_rank)
    calls = []

    with pytest.raises(MdpConfigurationError, match="local rank"):
        _execute(
            prepared, group=group, all_to_all_single=lambda *_args, **_kwargs: calls.append(True)
        )
    assert calls == []


@pytest.mark.parametrize(
    "query", ["ranks", "size", "local-rank", "noncallable-size", "noncallable-local-rank"]
)
def test_execute_normalizes_ordinary_group_query_errors_with_cause(query):
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)
    error = RuntimeError(query)
    calls = []

    def fail():
        raise error

    group = SimpleNamespace(size=lambda: 7, rank=lambda: 1)
    getter = lambda _group: list(_PARTICIPANTS)
    if query == "ranks":
        getter = lambda _group: fail()
    elif query == "size":
        group.size = fail
    elif query == "local-rank":
        group.rank = fail
    elif query == "noncallable-size":
        group.size = object()
    else:
        group.rank = object()

    with pytest.raises(MdpConfigurationError) as caught:
        _execute(
            prepared,
            group=group,
            group_ranks_getter=getter,
            all_to_all_single=lambda *_args, **_kwargs: calls.append(True),
        )
    if query.startswith("noncallable"):
        assert isinstance(caught.value.__cause__, TypeError)
    else:
        assert caught.value.__cause__ is error
    assert calls == []


def test_execute_normalizes_collective_error_with_cause():
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)
    error = RuntimeError("all-to-all")

    def fail(*_args, **_kwargs):
        raise error

    with pytest.raises(MdpBridgeError, match="all-to-all") as caught:
        _execute(prepared, all_to_all_single=fail)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("stage", ["ranks", "size", "local-rank", "collective"])
def test_execute_does_not_catch_base_exception(stage):
    prepared, _ = _prepare(_state(), BridgePhase.EMBEDDING, 30)

    def fail(*_args, **_kwargs):
        raise KeyboardInterrupt

    group = SimpleNamespace(size=lambda: 7, rank=lambda: 1)
    kwargs = {}
    if stage == "ranks":
        kwargs["group_ranks_getter"] = fail
    elif stage == "size":
        group.size = fail
    elif stage == "local-rank":
        group.rank = fail
    else:
        kwargs["all_to_all_single"] = fail

    with pytest.raises(KeyboardInterrupt):
        _execute(prepared, group=group, **kwargs)


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
_WORLD4_PARTICIPANTS = (0, 1, 2, 3)
_WORLD4_DECODER_RANKS = (2, 0)


def _world4_state(dtype, device):
    return _state(
        dtype=dtype,
        producers={
            GlobalVisionItemId(3, 0): 0,
            GlobalVisionItemId(3, 1): 0,
            GlobalVisionItemId(7, 0): 0,
        },
        participants=_WORLD4_PARTICIPANTS,
        decoder_ranks=_WORLD4_DECODER_RANKS,
        max_seqlen_per_rank=7,
        solver=_World4Cp2Solver(),
        device=device,
    )


if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def bridge_group():
        Utils.initialize_model_parallel()
        group = torch.distributed.new_group(ranks=list(_WORLD4_PARTICIPANTS), backend="nccl")
        yield group
        torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
@pytest.mark.parametrize("phase", [BridgePhase.EMBEDDING, BridgePhase.GRADIENT])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_world4_nccl_execute_preserves_self_remote_and_idle_ranks(phase, dtype, bridge_group):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _world4_state(dtype, device)
    prepared, _ = _prepare(state, phase, rank)
    calls = 0

    def tracked_all_to_all_single(*args, **kwargs):
        nonlocal calls
        calls += 1
        return torch.distributed.all_to_all_single(*args, **kwargs)

    received = getattr(transport, "execute_dynamic_bridge_exchange")(
        prepared, group=bridge_group, all_to_all_single=tracked_all_to_all_single
    )
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=bridge_group)

    assert completion.item() == 4
    assert calls == 1
    assert received is prepared.received_tensors
    assert tuple(torch.distributed.get_process_group_ranks(bridge_group)) == _WORLD4_PARTICIPANTS
    assert state.plan.decoder_ranks == _WORLD4_DECODER_RANKS
    assert any(
        entry.src_global_rank == entry.dst_global_rank == 0
        for entry in _selected(state, phase).entries
    )
    assert any(
        entry.src_global_rank != entry.dst_global_rank for entry in _selected(state, phase).entries
    )
    if rank in (1, 3):
        assert prepared.input_split_sizes == prepared.output_split_sizes == (0,) * 4
    elif rank == 0:
        assert sum(prepared.input_split_sizes) > 0
        assert sum(prepared.output_split_sizes) > 0
    elif phase is BridgePhase.EMBEDDING:
        assert sum(prepared.input_split_sizes) == 0
        assert sum(prepared.output_split_sizes) > 0
    else:
        assert sum(prepared.input_split_sizes) > 0
        assert sum(prepared.output_split_sizes) == 0

    oracle = {}
    for source_rank in _WORLD4_PARTICIPANTS:
        oracle.update(_local_tensors(state, phase, source_rank))
    if phase is BridgePhase.GRADIENT:
        values_by_item = {}
        for key, value in oracle.items():
            values_by_item.setdefault(key.item_id, []).append(value)
        endpoint_values = next(values for values in values_by_item.values() if len(values) > 1)
        assert not torch.equal(endpoint_values[0], endpoint_values[1])
    destination_entries = tuple(
        entry for entry in _selected(state, phase).entries if entry.dst_global_rank == rank
    )
    assert set(received) == {entry.key for entry in destination_entries}
    for entry in destination_entries:
        torch.testing.assert_close(received[entry.key], oracle[entry.key], rtol=0, atol=0)
