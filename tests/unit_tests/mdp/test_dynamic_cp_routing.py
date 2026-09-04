# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused contracts for decoder payload Dynamic-CP routing."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import megatron.core.mdp.dynamic_cp_execution as execution
import megatron.core.mdp.dynamic_cp_routing as routing
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
_PARTICIPANTS = (80, 30, 99, 70, 20, 10, 40)
_SOURCE_RANKS = {3: 70, 7: 80}
_FIELD_ORDER = ("position_ids", "loss_mask", "input_ids")
_INT64_MAX = 2**63 - 1


def _strided_tensor(seed, padded_seqlen, *, dtype=torch.int64):
    base = torch.arange(seed, seed + 4 * padded_seqlen, dtype=dtype).reshape(2, 2 * padded_seqlen)
    return base[1:2, 1::2]


def _packet(lane, order, valid_seqlen, padded_seqlen):
    tensors = {
        "position_ids": _strided_tensor(lane * 1000 + order * 100, padded_seqlen),
        "loss_mask": _strided_tensor(lane * 1000 + order * 100, padded_seqlen, dtype=torch.float32),
        "input_ids": _strided_tensor(lane * 1000 + order * 100 + 10_000, padded_seqlen),
    }
    tensor_fields = MappingProxyType({name: tensors[name] for name in _FIELD_ORDER})
    field_specs = tuple(
        DecoderTensorFieldSpec(
            name=name, dtype=tensor.dtype, shape=tuple(tensor.shape), device_type=tensor.device.type
        )
        for name, tensor in tensor_fields.items()
    )
    header = DecoderPayloadHeaderV1(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        source_dp_lane=lane,
        local_sample_order=order,
        valid_seqlen=valid_seqlen,
        padded_seqlen=padded_seqlen,
        tensor_field_count=len(field_specs),
        none_field_count=1,
        position_components_or_minus_one=1,
    )
    return DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=GlobalSampleId(lane, order),
        valid_seqlen=valid_seqlen,
        padded_seqlen=padded_seqlen,
        header=header.to_wire_tuple(),
        field_specs=field_specs,
        tensor_fields=tensor_fields,
        none_fields=("attention_mask",),
    )


def _source_window(lane, *, text_only=False):
    lengths = ((4, 6), (5, 7))
    samples = []
    items = []
    packets = []
    for order, (valid_seqlen, padded_seqlen) in enumerate(lengths):
        sample_id = GlobalSampleId(lane, order)
        sample_items = ()
        if not text_only:
            item_id = GlobalVisionItemId(lane, order)
            sample_items = (
                EncoderVisionItemMetadata(item_id=item_id, sample_id=sample_id, image_ordinal=0),
            )
            items.append(
                DecoderVisionItemMetadata(
                    item_id=item_id,
                    sample_id=sample_id,
                    image_ordinal=0,
                    grid_thw=(1, 1, 1),
                    output_rows=1,
                    decoder_offsets=(order,),
                )
            )
        samples.append(
            DecoderSampleMetadata(
                sample_id=sample_id,
                valid_seqlen=valid_seqlen,
                padded_seqlen=padded_seqlen,
                vision_items=sample_items,
            )
        )
        packets.append(_packet(lane, order, valid_seqlen, padded_seqlen))
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
        assert total_gpus == 4
        assert max_seq_len_per_rank == 4
        assert min_cp_size == 1
        self.calls += 1
        if self.calls == 1:
            assert sample_seqlens == [(0, 6), (1, 7), (2, 6), (3, 7)]
            return ([[6], [6], [7], [7]], [(2, 6), (3, 7)], None, [[0], [0], [1], [1]])
        assert sample_seqlens == [(2, 6), (3, 7)]
        return ([[6], [6], [7], [7]], [], None, [[2], [2], [3], [3]])


def _state(*, text_only=False):
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
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane=_SOURCE_RANKS,
        participant_ranks=_PARTICIPANTS,
    )
    ledger = routing.build_decoder_payload_route_ledger(**authority)
    return SimpleNamespace(
        lane3=lane3, lane7=lane7, manifest=manifest, plan=plan, authority=authority, ledger=ledger
    )


def _recanonicalize(ledger, entries):
    field_order = {name: index for index, name in enumerate(_FIELD_ORDER)}
    ordered = sorted(
        entries,
        key=lambda entry: (
            entry.src_global_rank,
            entry.dst_global_rank,
            entry.key.sample_id,
            field_order[entry.key.field_name],
        ),
    )
    offsets = {}
    canonical = []
    for entry in ordered:
        edge = (entry.src_global_rank, entry.dst_global_rank, entry.dtype)
        offset = offsets.get(edge, 0)
        offsets[edge] = offset + entry.element_count
        canonical.append(replace(entry, plan_offset=offset))
    return replace(ledger, entries=tuple(canonical))


def _authority_mutation(state, mutation):
    selected = tuple(
        entry
        for entry in state.ledger.entries
        if entry.key.sample_id == GlobalSampleId(3, 0) and entry.dst_global_rank == 30
    )
    assert tuple(entry.key.field_name for entry in selected) == _FIELD_ORDER
    entries = list(state.ledger.entries)
    if mutation == "missing-endpoint":
        entries = [entry for entry in entries if entry not in selected]
    elif mutation == "unauthorized-endpoint":
        entries = [
            (
                replace(entry, dst_global_rank=20, key=replace(entry.key, endpoint_rank=20))
                if entry in selected
                else entry
            )
            for entry in entries
        ]
    else:
        entries = [
            replace(entry, src_global_rank=80) if entry in selected else entry for entry in entries
        ]
    return _recanonicalize(state.ledger, entries)


def _projection_mutation(window, mutation):
    packet = window.packets[0]
    samples = list(window.samples)
    items = list(window.items)
    header = list(packet.header)
    fields = dict(packet.tensor_fields)
    specs = list(packet.field_specs)
    none_fields = packet.none_fields
    valid_seqlen = packet.valid_seqlen
    padded_seqlen = packet.padded_seqlen

    if mutation == "shape":
        fields["position_ids"] = fields["position_ids"].unsqueeze(0)
        index = next(i for i, spec in enumerate(specs) if spec.name == "position_ids")
        specs[index] = replace(specs[index], shape=tuple(fields["position_ids"].shape))
    elif mutation == "valid":
        valid_seqlen -= 1
        header[3] = valid_seqlen
        samples[0] = replace(samples[0], valid_seqlen=valid_seqlen)
    elif mutation == "padded":
        padded_seqlen -= 1
        header[4] = padded_seqlen
        samples[0] = replace(samples[0], padded_seqlen=padded_seqlen)
        for index, spec in enumerate(specs):
            fields[spec.name] = fields[spec.name][..., :padded_seqlen]
            specs[index] = replace(spec, shape=tuple(fields[spec.name].shape))
    elif mutation == "none-schema":
        none_fields = (*none_fields, "future_none")
        header[6] = len(none_fields)
    else:
        items[0] = replace(items[0], grid_thw=(1, 1, 2))

    packets = list(window.packets)
    packets[0] = replace(
        packet,
        valid_seqlen=valid_seqlen,
        padded_seqlen=padded_seqlen,
        header=tuple(header),
        field_specs=tuple(specs),
        tensor_fields=MappingProxyType(fields),
        none_fields=none_fields,
    )
    return finalize_decoder_source_window(
        source_dp_lane=window.source_dp_lane,
        samples=tuple(samples),
        items=tuple(items),
        packets=tuple(packets),
    )


def test_builder_uses_plan_endpoints_manifest_field_order_and_typed_edge_offsets():
    state = _state()

    assert (
        routing.validate_decoder_payload_route_ledger(state.ledger, **state.authority)
        is state.ledger
    )
    assert state.ledger.participant_ranks == _PARTICIPANTS
    assert len(state.ledger.entries) == 24
    endpoint_by_sample = {
        GlobalSampleId(3, 0): (30, 10),
        GlobalSampleId(3, 1): (40, 20),
        GlobalSampleId(7, 0): (30, 10),
        GlobalSampleId(7, 1): (40, 20),
    }
    for sample_id, endpoints in endpoint_by_sample.items():
        for endpoint in endpoints:
            selected = tuple(
                entry
                for entry in state.ledger.entries
                if entry.key.sample_id == sample_id and entry.dst_global_rank == endpoint
            )
            assert tuple(entry.key.field_name for entry in selected) == _FIELD_ORDER
            padded = state.manifest.payloads[
                state.manifest.samples.index(
                    next(
                        sample for sample in state.manifest.samples if sample.sample_id == sample_id
                    )
                )
            ].padded_seqlen
            assert tuple(entry.plan_offset for entry in selected) == (0, 0, padded)
            assert tuple(entry.element_count for entry in selected) == (padded, padded, padded)


def test_split_sizes_follow_noncontiguous_participant_order_and_dtype():
    state = _state()

    assert routing.decoder_payload_split_sizes(
        state.ledger, dtype=torch.int64, global_rank=70, **state.authority
    ) == ((0, 12, 0, 0, 14, 12, 14), (0, 0, 0, 0, 0, 0, 0))
    assert routing.decoder_payload_split_sizes(
        state.ledger, dtype=torch.int64, global_rank=30, **state.authority
    ) == ((0, 0, 0, 0, 0, 0, 0), (12, 0, 0, 12, 0, 0, 0))
    assert routing.decoder_payload_split_sizes(
        state.ledger, dtype=torch.float32, global_rank=30, **state.authority
    ) == ((0, 0, 0, 0, 0, 0, 0), (6, 0, 0, 6, 0, 0, 0))
    assert routing.decoder_payload_split_sizes(
        state.ledger, dtype=torch.float64, global_rank=99, **state.authority
    ) == ((0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0))


def test_split_sizes_include_self_edge_in_authoritative_participant_position():
    state = _state()
    authority = dict(state.authority, source_rank_by_lane={3: 30, 7: 80})
    ledger = routing.build_decoder_payload_route_ledger(**authority)

    assert any(entry.src_global_rank == entry.dst_global_rank == 30 for entry in ledger.entries)
    assert routing.decoder_payload_split_sizes(
        ledger, dtype=torch.int64, global_rank=30, **authority
    ) == ((0, 12, 0, 0, 14, 12, 14), (12, 12, 0, 0, 0, 0, 0))
    assert routing.decoder_payload_split_sizes(
        ledger, dtype=torch.float32, global_rank=30, **authority
    ) == ((0, 6, 0, 0, 7, 6, 7), (6, 6, 0, 0, 0, 0, 0))


@pytest.mark.parametrize(("window_name", "rank"), (("lane3", 70), ("lane7", 80)))
def test_attachment_preserves_exact_tensor_view_identity_and_is_immutable(window_name, rank):
    state = _state()
    window = getattr(state, window_name)
    packets = {packet.sample_id: packet for packet in window.packets}

    attached = routing.attach_local_decoder_payload_tensors(
        state.ledger, source_window=window, global_rank=rank, **state.authority
    )

    assert tuple(attached) == tuple(
        entry.key for entry in state.ledger.entries if entry.src_global_rank == rank
    )
    for key, tensor in attached.items():
        source = packets[key.sample_id].tensor_fields[key.field_name]
        assert tensor is source
        assert tensor.shape == source.shape
        assert tensor.stride() == source.stride()
        assert tensor.storage_offset() == source.storage_offset()
        assert tensor.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    with pytest.raises(TypeError):
        attached[next(iter(attached))] = torch.empty(0)


def test_text_only_windows_route_decoder_tensors_but_not_none_fields():
    state = _state(text_only=True)

    assert state.manifest.items == ()
    assert state.ledger.entries
    assert {entry.key.field_name for entry in state.ledger.entries} == set(_FIELD_ORDER)
    assert all(entry.key.field_name != "attention_mask" for entry in state.ledger.entries)
    attached = routing.attach_local_decoder_payload_tensors(
        state.ledger, source_window=state.lane3, global_rank=70, **state.authority
    )
    assert len(attached) == 12


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("entry-carrier", "typed route entries"),
        ("key-carrier", "typed key"),
        ("sample-carrier", "typed sample ID"),
        ("field-carrier", "field name"),
        ("endpoint", "matches its destination"),
        ("duplicate", "keys are unique"),
        ("reorder", "canonical order"),
        ("offset", "offsets are contiguous"),
        ("overflow", "extent fits signed int64"),
    ),
)
def test_structural_validator_rejects_malformed_or_unhashable_entries(mutation, message):
    state = _state()
    entries = list(state.ledger.entries)
    if mutation == "entry-carrier":
        entries[0] = []
    elif mutation == "key-carrier":
        entries[0] = replace(entries[0], key=[])
    elif mutation == "sample-carrier":
        entries[0] = replace(entries[0], key=replace(entries[0].key, sample_id=[]))
    elif mutation == "field-carrier":
        entries[0] = replace(entries[0], key=replace(entries[0].key, field_name=[]))
    elif mutation == "endpoint":
        entries[0] = replace(entries[0], key=replace(entries[0].key, endpoint_rank=40))
    elif mutation == "duplicate":
        entries.insert(1, entries[0])
    elif mutation == "reorder":
        group = tuple(
            index
            for index, entry in enumerate(entries)
            if entry.src_global_rank == entries[0].src_global_rank
            and entry.dst_global_rank != entries[0].dst_global_rank
        )
        entries[0], entries[group[-1]] = entries[group[-1]], entries[0]
        offsets = {}
        for index, entry in enumerate(entries):
            edge = (entry.src_global_rank, entry.dst_global_rank, entry.dtype)
            offset = offsets.get(edge, 0)
            entries[index] = replace(entry, plan_offset=offset)
            offsets[edge] = offset + entry.element_count
    elif mutation == "offset":
        entries[0] = replace(entries[0], plan_offset=1)
    else:
        entries[0] = replace(entries[0], element_count=_INT64_MAX)
        entries[2] = replace(entries[2], plan_offset=_INT64_MAX)
    corrupted = replace(state.ledger, entries=tuple(entries))

    with pytest.raises(MdpBridgeError, match=message):
        routing._validate_decoder_payload_route_structure(corrupted)


@pytest.mark.parametrize(
    ("carrier", "message"),
    (
        (object(), "typed immutable route ledger"),
        (routing.DecoderPayloadRouteLedger([], _PARTICIPANTS), "typed immutable route ledger"),
        (routing.DecoderPayloadRouteLedger((), []), "participants form"),
    ),
)
def test_structural_validator_rejects_malformed_ledger_carriers(carrier, message):
    with pytest.raises(MdpBridgeError, match=message):
        routing._validate_decoder_payload_route_structure(carrier)


@pytest.mark.parametrize("mutation", ("missing-endpoint", "unauthorized-endpoint", "wrong-source"))
@pytest.mark.parametrize("consumer", ("split", "attachment"))
def test_consumers_rederive_authority_before_using_structurally_valid_mutations(mutation, consumer):
    state = _state()
    corrupted = _authority_mutation(state, mutation)
    assert routing._validate_decoder_payload_route_structure(corrupted) is corrupted

    with pytest.raises(MdpBridgeError, match="plan and manifest authority"):
        if consumer == "split":
            routing.decoder_payload_split_sizes(
                corrupted, dtype=torch.int64, global_rank=70, **state.authority
            )
        else:
            routing.attach_local_decoder_payload_tensors(
                corrupted, source_window=state.lane3, global_rank=70, **state.authority
            )


@pytest.mark.parametrize("mutation", ("shape", "valid", "padded", "none-schema", "items"))
def test_attachment_rejects_valid_source_window_metadata_outside_global_projection(mutation):
    state = _state()
    corrupted = _projection_mutation(state.lane3, mutation)

    with pytest.raises(MdpBridgeError, match="exact global-manifest lane projection"):
        routing.attach_local_decoder_payload_tensors(
            state.ledger, source_window=corrupted, global_rank=70, **state.authority
        )


def test_attachment_rejects_wrong_source_window_lane_before_tensor_lookup():
    state = _state()

    with pytest.raises(MdpBridgeError, match="belongs to the attaching source rank"):
        routing.attach_local_decoder_payload_tensors(
            state.ledger, source_window=state.lane7, global_rank=70, **state.authority
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"source_rank_by_lane": object()}, "lane-to-rank mapping"),
        ({"source_rank_by_lane": {3: 70}}, "exactly to unique source ranks"),
        ({"source_rank_by_lane": {3: 70, 7: 70}}, "exactly to unique source ranks"),
        ({"source_rank_by_lane": {3: 70, 7: _INT64_MAX + 1}}, "signed-int64"),
        ({"participant_ranks": (80, 30, 70, 20, 10)}, "decoder endpoint"),
        ({"participant_ranks": (80, 30, 70, 20, 10, 10)}, "unique ranks"),
    ),
)
def test_builder_rejects_malformed_authority_arguments(overrides, message):
    state = _state()
    authority = dict(state.authority)
    authority.update(overrides)

    with pytest.raises(MdpConfigurationError, match=message):
        routing.build_decoder_payload_route_ledger(**authority)


def test_builder_rejects_field_shape_product_overflow():
    state = _state()
    payloads = []
    for payload in state.manifest.payloads:
        specs = tuple(
            replace(spec, shape=(2**62, spec.shape[-1])) if spec.name == "input_ids" else spec
            for spec in payload.field_specs
        )
        payloads.append(replace(payload, field_specs=specs))
    payloads = tuple(payloads)
    manifest = replace(
        state.manifest,
        payloads=payloads,
        digest=execution._manifest_digest(
            execution._GLOBAL_MANIFEST_DOMAIN,
            state.manifest.samples,
            state.manifest.items,
            payloads,
        ),
    )

    with pytest.raises(MdpConfigurationError, match="route element count.*signed-int64"):
        routing.build_decoder_payload_route_ledger(
            state.plan,
            global_manifest=manifest,
            source_rank_by_lane=_SOURCE_RANKS,
            participant_ranks=_PARTICIPANTS,
        )


@pytest.mark.parametrize(
    ("dtype", "rank", "message"),
    (("int64", 70, "dtype is a torch dtype"), (torch.int64, 999, "rank is a participant")),
)
def test_split_rejects_malformed_dtype_and_rank(dtype, rank, message):
    state = _state()

    with pytest.raises(MdpConfigurationError, match=message):
        routing.decoder_payload_split_sizes(
            state.ledger, dtype=dtype, global_rank=rank, **state.authority
        )
