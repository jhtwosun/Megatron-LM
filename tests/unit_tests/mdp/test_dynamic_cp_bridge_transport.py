# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused caller-buffer contracts for Dynamic-CP bridge transport."""

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


def _state(
    *, dtype=torch.float32, width=4, producers=None, participants=_PARTICIPANTS, text_only=False
):
    lane3 = _source_window(3, text_only=text_only)
    lane7 = _source_window(7, text_only=text_only)
    manifest = build_decoder_global_manifest((lane7.metadata_manifest(), lane3.metadata_manifest()))
    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=_DECODER_RANKS,
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=_TwoWaveCp2Solver(),
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
        plan=plan, manifest=manifest, authority=authority, embedding=embedding, gradient=gradient
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
            value = base.view(rows, state.authority["width"] * 2)[:, ::2]
            value.add_(100 * (index + 1))
            if phase is BridgePhase.EMBEDDING:
                embedding_values[entry.key.item_id] = value
        result[entry.key] = value
    return result


def _prepare(state, phase, rank, *, local_tensors=None, send=None, receive=None):
    inputs, outputs = _splits(state, phase, rank)
    if local_tensors is None:
        local_tensors = _local_tensors(state, phase, rank)
    if send is None:
        send = torch.full((sum(inputs),), -1, dtype=state.authority["dtype"])
    if receive is None:
        receive = torch.arange(sum(outputs), dtype=state.authority["dtype"])
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
