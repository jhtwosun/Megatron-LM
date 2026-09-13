# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused deterministic contracts for Dynamic-CP bridge ledgers."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import megatron.core.mdp.dynamic_cp_bridge as bridge
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
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

_DECODER_RANKS = (30, 10, 40, 20)
_PARTICIPANTS = (80, 30, 70, 20, 10, 40, 99)
_INT64_MAX = 2**63 - 1


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
            assert sample_seqlens == [(0, 6), (1, 7), (2, 6), (3, 7)]
            return ([[6], [6], [7], [7]], [(2, 6), (3, 7)], None, [[0], [0], [1], [1]])
        assert sample_seqlens == [(2, 6), (3, 7)]
        return ([[6], [6], [7], [7]], [], None, [[2], [2], [3], [3]])


def _state(*, text_only=False, dtype=torch.bfloat16, width=4):
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
    producers = (
        {}
        if text_only
        else {
            GlobalVisionItemId(3, 0): 30,
            GlobalVisionItemId(3, 1): 30,
            GlobalVisionItemId(7, 0): 80,
        }
    )
    rows = {item.item_id: item.output_rows for item in manifest.items}
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        producer_rank_by_item=producers,
        output_rows_by_item=rows,
        width=width,
        dtype=dtype,
        participant_ranks=_PARTICIPANTS,
    )
    embedding, gradient = bridge.build_dynamic_bridge_ledgers(**authority)
    return SimpleNamespace(
        plan=plan, manifest=manifest, authority=authority, embedding=embedding, gradient=gradient
    )


def _canonical_ledger(ledger, entries):
    entries = sorted(
        entries,
        key=lambda entry: (
            entry.src_global_rank,
            entry.dst_global_rank,
            entry.key.item_id,
            entry.key.endpoint_rank,
        ),
    )
    offsets = {}
    canonical = []
    total = 0
    remote = 0
    for entry in entries:
        edge = (entry.src_global_rank, entry.dst_global_rank, entry.dtype)
        offset = offsets.get(edge, 0)
        offsets[edge] = offset + entry.element_count
        canonical.append(replace(entry, plan_offset=offset))
        size = entry.element_count * entry.dtype.itemsize
        total += size
        if entry.src_global_rank != entry.dst_global_rank:
            remote += size
    return replace(ledger, entries=tuple(canonical), total_bytes=total, remote_bytes=remote)


@pytest.mark.parametrize("dtype,itemsize", [(torch.bfloat16, 2), (torch.float32, 4)])
def test_canonical_fanout_reverse_bytes_and_participant_order(dtype, itemsize):
    state = _state(dtype=dtype)

    assert bridge.validate_dynamic_bridge_ledger_pair(
        state.embedding, state.gradient, **state.authority
    ) == (state.embedding, state.gradient)
    assert state.embedding.participant_ranks == _PARTICIPANTS
    assert len(state.embedding.entries) == len(state.gradient.entries) == 6
    assert [entry.element_count for entry in state.embedding.entries] == [8, 4, 8, 4, 12, 12]
    assert [entry.plan_offset for entry in state.embedding.entries] == [0, 8, 0, 8, 0, 0]
    assert state.embedding.total_bytes == state.gradient.total_bytes == 48 * itemsize
    assert state.embedding.remote_bytes == state.gradient.remote_bytes == 36 * itemsize
    embedding_edges = {
        (entry.src_global_rank, entry.dst_global_rank, entry.key, entry.element_count)
        for entry in state.embedding.entries
    }
    gradient_edges = {
        (entry.dst_global_rank, entry.src_global_rank, entry.key, entry.element_count)
        for entry in state.gradient.entries
    }
    assert embedding_edges == gradient_edges
    assert bridge.dynamic_bridge_split_sizes(
        state.embedding, reverse_ledger=state.gradient, global_rank=30, **state.authority
    ) == ((0, 12, 0, 0, 12, 0, 0), (12, 12, 0, 0, 0, 0, 0))
    assert bridge.dynamic_bridge_split_sizes(
        state.gradient, reverse_ledger=state.embedding, global_rank=30, **state.authority
    ) == ((12, 12, 0, 0, 0, 0, 0), (0, 12, 0, 0, 12, 0, 0))
    assert bridge.dynamic_bridge_split_sizes(
        state.embedding, reverse_ledger=state.gradient, global_rank=99, **state.authority
    ) == ((0,) * 7, (0,) * 7)


def test_text_only_ledgers_remain_empty_with_authoritative_participants():
    state = _state(text_only=True)

    assert state.embedding.entries == state.gradient.entries == ()
    assert state.embedding.total_bytes == state.embedding.remote_bytes == 0
    assert state.gradient.total_bytes == state.gradient.remote_bytes == 0
    assert state.embedding.participant_ranks == state.gradient.participant_ranks == _PARTICIPANTS
    assert bridge.dynamic_bridge_split_sizes(
        state.embedding, reverse_ledger=state.gradient, global_rank=70, **state.authority
    ) == ((0,) * 7, (0,) * 7)


@pytest.mark.parametrize("mutation", ["missing", "endpoint", "producer"])
def test_structurally_valid_authority_mutations_fail_pair_and_split(mutation):
    state = _state()
    selected = state.embedding.entries[0]
    if mutation == "missing":
        entries = state.embedding.entries[1:]
    elif mutation == "endpoint":
        entries = tuple(
            (
                replace(entry, dst_global_rank=20, key=replace(entry.key, endpoint_rank=20))
                if entry == selected
                else entry
            )
            for entry in state.embedding.entries
        )
    else:
        entries = tuple(
            replace(entry, src_global_rank=70) if entry == selected else entry
            for entry in state.embedding.entries
        )
    mutated = _canonical_ledger(state.embedding, entries)

    with pytest.raises(MdpBridgeError, match="exactly match plan authority"):
        bridge.validate_dynamic_bridge_ledger_pair(mutated, state.gradient, **state.authority)
    with pytest.raises(MdpBridgeError, match="exactly match plan authority"):
        bridge.dynamic_bridge_split_sizes(
            mutated, reverse_ledger=state.gradient, global_rank=30, **state.authority
        )


def test_structurally_valid_gradient_reverse_mutation_is_rejected():
    state = _state()
    entry = state.gradient.entries[0]
    mutated = _canonical_ledger(
        state.gradient, (replace(entry, dst_global_rank=70), *state.gradient.entries[1:])
    )

    with pytest.raises(MdpBridgeError, match="exactly match plan authority"):
        bridge.validate_dynamic_bridge_ledger_pair(state.embedding, mutated, **state.authority)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda ledger: replace(ledger, entries=list(ledger.entries)), "typed immutable"),
        (
            lambda ledger: replace(
                ledger,
                entries=(replace(ledger.entries[0], key=bridge.DynamicBridgeKey([], 10)),)
                + ledger.entries[1:],
            ),
            "typed vision-item ID",
        ),
        (
            lambda ledger: replace(
                ledger,
                entries=(
                    replace(
                        ledger.entries[0],
                        key=replace(
                            ledger.entries[0].key, item_id=GlobalVisionItemId(_INT64_MAX + 1, 0)
                        ),
                    ),
                )
                + ledger.entries[1:],
            ),
            "signed-int64",
        ),
        (
            lambda ledger: replace(
                ledger, entries=(replace(ledger.entries[0], plan_offset=1),) + ledger.entries[1:]
            ),
            "offsets are contiguous",
        ),
        (lambda ledger: replace(ledger, total_bytes=ledger.total_bytes + 1), "total bytes match"),
    ],
)
def test_malformed_nested_ledger_carriers_raise_typed_bridge_errors(mutation, match):
    state = _state()
    with pytest.raises(MdpBridgeError, match=match):
        bridge.validate_dynamic_bridge_ledger_pair(
            mutation(state.embedding), state.gradient, **state.authority
        )


@pytest.mark.parametrize(
    "override,error,match",
    [
        ({"producer_rank_by_item": {}}, MdpConfigurationError, "exact vision-item catalog"),
        ({"output_rows_by_item": {}}, MdpConfigurationError, "exact vision-item catalog"),
        ({"participant_ranks": (30, 10)}, MdpConfigurationError, "every decoder endpoint"),
        ({"width": 0}, MdpConfigurationError, "positive signed-int64"),
        ({"dtype": "bf16"}, MdpConfigurationError, "torch dtype"),
    ],
)
def test_builder_rejects_malformed_authority_arguments(override, error, match):
    state = _state()
    authority = {**state.authority, **override}
    with pytest.raises(error, match=match):
        bridge.build_dynamic_bridge_ledgers(**authority)


def test_plan_manifest_catalog_mismatch_is_a_plan_error():
    state = _state()
    text = _state(text_only=True)
    with pytest.raises(MdpPlanError, match="exact decoder plan catalog"):
        bridge.build_dynamic_bridge_ledgers(**{**state.authority, "global_manifest": text.manifest})


def test_builder_rejects_output_rows_value_mismatch():
    state = _state()
    rows = dict(state.authority["output_rows_by_item"])
    rows[next(iter(rows))] += 1

    with pytest.raises(MdpConfigurationError, match="output rows match"):
        bridge.build_dynamic_bridge_ledgers(**{**state.authority, "output_rows_by_item": rows})


def test_structure_rejects_typed_edge_extent_overflow():
    state = _state()
    extent = _INT64_MAX // 2 + 1
    first, second, *remaining = state.embedding.entries
    mutated = replace(
        state.embedding,
        entries=(
            replace(first, dtype=torch.uint8, element_count=extent, plan_offset=0),
            replace(second, dtype=torch.uint8, element_count=extent, plan_offset=extent),
            *remaining,
        ),
        total_bytes=0,
        remote_bytes=0,
    )

    with pytest.raises(MdpBridgeError, match="typed-edge extent fits signed int64"):
        bridge.validate_dynamic_bridge_ledger_pair(mutated, state.gradient, **state.authority)


@pytest.mark.parametrize(
    "width,dtype,match",
    [
        (_INT64_MAX, torch.uint8, "element count fits signed int64"),
        (_INT64_MAX // 3, torch.float32, "entry bytes fits signed int64"),
        (_INT64_MAX // 12 + 1, torch.uint8, "total bytes fits signed int64"),
    ],
)
def test_builder_rejects_product_add_and_byte_overflow(width, dtype, match):
    state = _state()
    with pytest.raises(MdpConfigurationError, match=match):
        bridge.build_dynamic_bridge_ledgers(**{**state.authority, "width": width, "dtype": dtype})
