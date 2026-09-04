# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused source-codec tests for the multimodal MDP adapter."""

from dataclasses import replace
from types import MappingProxyType

import pytest
import torch

from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec, Qwen35VLMdpAdapter
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import (
    build_decoder_global_manifest,
    build_decoder_source_window,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams

_MISSING = object()
_ROUTED_TENSOR_FIELDS = ("input_ids", "labels", "loss_mask", "padding_mask")


def _cumulative(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32)


def _packed(
    valid_lengths, padded_lengths, *, qkv_format="thd", kv_valid_lengths=None, total_tokens=None
):
    valid_q = _cumulative(valid_lengths)
    valid_kv = _cumulative(valid_lengths if kv_valid_lengths is None else kv_valid_lengths)
    padded = _cumulative(padded_lengths)
    return PackedSeqParams(
        qkv_format=qkv_format,
        cu_seqlens_q=valid_q,
        cu_seqlens_kv=valid_kv,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded.clone(),
        max_seqlen_q=max(padded_lengths),
        max_seqlen_kv=max(padded_lengths),
        total_tokens=int(padded[-1]) if total_tokens is None else total_tokens,
    )


def _payload(base, padded_lengths, *, grids, position_components=1, attention_mask=_MISSING):
    total_tokens = sum(padded_lengths)
    payload = {
        "input_ids": torch.arange(base, base + total_tokens, dtype=torch.int64).view(
            1, total_tokens
        ),
        "labels": torch.arange(base + 100, base + 100 + total_tokens, dtype=torch.int64).view(
            1, total_tokens
        ),
        "loss_mask": torch.arange(total_tokens, dtype=torch.float32).view(1, total_tokens),
        "padding_mask": torch.zeros(1, total_tokens, dtype=torch.bool),
        "image_grid_thw": (
            torch.tensor(grids, dtype=torch.int64)
            if grids
            else torch.empty((0, 3), dtype=torch.int64)
        ),
    }
    if position_components is not _MISSING:
        if position_components is None:
            payload["position_ids"] = None
        else:
            positions = torch.arange(base + 200, base + 200 + total_tokens, dtype=torch.int64)
            payload["position_ids"] = (
                positions.view(1, total_tokens)
                if position_components == 1
                else positions.repeat(position_components, 1, 1)
            )
    if attention_mask is not _MISSING:
        payload["attention_mask"] = attention_mask
    return MappingProxyType(payload)


def _records(*, position_components=1, attention_mask=_MISSING):
    record_zero = MdpMicrobatchRecord(
        microbatch_id=0,
        text_only=False,
        vision_items=(
            MdpMicrobatchVisionRecord(
                global_item_id=0,
                sample_id=0,
                image_ordinal=0,
                grid_thw=(1, 2, 2),
                output_rows=1,
                decoder_positions=(1,),
            ),
            MdpMicrobatchVisionRecord(
                global_item_id=1,
                sample_id=1,
                image_ordinal=0,
                grid_thw=(1, 2, 2),
                output_rows=1,
                decoder_positions=(5,),
            ),
        ),
        decoder_packed_seq_params=_packed((3, 4), (4, 6)),
        model_payload=_payload(
            0,
            (4, 6),
            grids=((1, 2, 2), (1, 2, 2)),
            position_components=position_components,
            attention_mask=attention_mask,
        ),
    )
    record_one = MdpMicrobatchRecord(
        microbatch_id=1,
        text_only=True,
        vision_items=(),
        decoder_packed_seq_params=_packed((2,), (3,)),
        model_payload=_payload(
            1000,
            (3,),
            grids=(),
            position_components=position_components,
            attention_mask=attention_mask,
        ),
    )
    return (record_zero, record_one)


def _replace_payload(record, mutate):
    payload = dict(record.model_payload)
    mutate(payload)
    return replace(record, model_payload=MappingProxyType(payload))


def _replace_all_payloads(records, mutate):
    return tuple(_replace_payload(record, mutate) for record in records)


@pytest.mark.parametrize("position_components", (1, 3), ids=("rope", "mrope"))
def test_codec_builds_canonical_zero_copy_source_window(position_components):
    records = _records(position_components=position_components)

    window = build_decoder_source_window(
        records, source_dp_lane=7, codec=MultimodalDecoderPayloadCodec()
    )

    assert window.sample_ids == tuple(GlobalSampleId(7, index) for index in range(3))
    assert tuple(item.item_id for item in window.items) == (
        GlobalVisionItemId(7, 0),
        GlobalVisionItemId(7, 1),
    )
    assert tuple(item.sample_id for item in window.items) == (
        GlobalSampleId(7, 0),
        GlobalSampleId(7, 1),
    )
    assert tuple(item.decoder_offsets for item in window.items) == ((1,), (1,))
    assert tuple((sample.valid_seqlen, sample.padded_seqlen) for sample in window.samples) == (
        (3, 4),
        (4, 6),
        (2, 3),
    )
    assert window.samples[-1].vision_items == ()
    assert tuple(packet.sample_id for packet in window.packets) == window.sample_ids
    assert records[0].vision_items[0].global_item_id == 0

    starts = (0, 4)
    for packet, start in zip(window.packets[:2], starts):
        assert tuple(packet.tensor_fields) == (*_ROUTED_TENSOR_FIELDS, "position_ids")
        for name, view in packet.tensor_fields.items():
            source = records[0].model_payload[name]
            assert view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
            torch.testing.assert_close(view, source[..., start : start + packet.padded_seqlen])
        assert packet.tensor_fields["position_ids"].shape[0] == position_components

    text_packet = window.packets[-1]
    for name, view in text_packet.tensor_fields.items():
        source = records[1].model_payload[name]
        assert view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        torch.testing.assert_close(view, source)


@pytest.mark.parametrize(
    ("position_components", "attention_mask", "expected_none"),
    (
        (_MISSING, _MISSING, ("position_ids", "attention_mask")),
        (None, _MISSING, ("position_ids", "attention_mask")),
        (1, None, ("attention_mask",)),
    ),
    ids=("position-missing", "position-none", "attention-none"),
)
def test_codec_preserves_supported_optional_none_fields(
    position_components, attention_mask, expected_none
):
    window = MultimodalDecoderPayloadCodec().build_source_window(
        _records(position_components=position_components, attention_mask=attention_mask),
        source_dp_lane=0,
    )

    assert all(packet.none_fields == expected_none for packet in window.packets)
    assert all(
        all(name not in packet.tensor_fields for name in expected_none) for packet in window.packets
    )


def test_codec_canonicalizes_attention_missing_and_none_within_and_across_lanes():
    codec = MultimodalDecoderPayloadCodec()
    mixed_records = list(_records())
    mixed_records[1] = _replace_payload(
        mixed_records[1], lambda payload: payload.__setitem__("attention_mask", None)
    )

    mixed = codec.build_source_window(tuple(mixed_records), source_dp_lane=0)
    assert all(packet.none_fields == ("attention_mask",) for packet in mixed.packets)

    missing = codec.build_source_window(_records(), source_dp_lane=7).metadata_manifest()
    explicit_none = codec.build_source_window(
        _records(attention_mask=None), source_dp_lane=8
    ).metadata_manifest()
    global_manifest = build_decoder_global_manifest((explicit_none, missing))

    assert all(payload.none_fields == ("attention_mask",) for payload in global_manifest.payloads)
    schemas = {
        tuple(
            (spec.name, spec.dtype, spec.device_type, spec.shape[:-1])
            for spec in payload.field_specs
        )
        for payload in global_manifest.payloads
    }
    assert len(schemas) == 1
    assert len(global_manifest.digest) == 16


def test_qwen_adapter_exposes_iteration_independent_source_codec():
    adapter = Qwen35VLMdpAdapter(out_hidden_size=16)

    first = adapter.build_dynamic_decoder_payload_codec()
    second = adapter.build_dynamic_decoder_payload_codec()

    assert isinstance(first, MultimodalDecoderPayloadCodec)
    assert isinstance(second, MultimodalDecoderPayloadCodec)
    assert first is not second


@pytest.mark.parametrize(
    ("records", "source_dp_lane"),
    (((), 0), ((object(),), 0), (_records(), -1), (_records(), True)),
    ids=("empty", "record-type", "negative-lane", "bool-lane"),
)
def test_codec_rejects_malformed_record_and_lane_states(records, source_dp_lane):
    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        MultimodalDecoderPayloadCodec().build_source_window(records, source_dp_lane=source_dp_lane)


@pytest.mark.parametrize("mutation", ("carrier", "format", "kv", "total"))
def test_codec_rejects_malformed_thd_metadata(mutation):
    records = list(_records())
    if mutation == "carrier":
        packed = object()
    elif mutation == "format":
        packed = _packed((3, 4), (4, 6), qkv_format="bshd")
    elif mutation == "kv":
        packed = _packed((3, 4), (4, 6), kv_valid_lengths=(2, 5))
    else:
        packed = _packed((3, 4), (4, 6), total_tokens=9)
    records[0] = replace(records[0], decoder_packed_seq_params=packed)

    with pytest.raises(MdpConfigurationError):
        MultimodalDecoderPayloadCodec().build_source_window(tuple(records), source_dp_lane=0)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.__setitem__("pixel_values", torch.ones(1)),
        lambda payload: payload.pop("input_ids"),
        lambda payload: payload.__setitem__("position_ids", [b"nested-jpg"]),
        lambda payload: payload.__setitem__("attention_mask", torch.ones(1, 10)),
        lambda payload: payload.__setitem__("position_ids", torch.zeros(2, 1, 10)),
        lambda payload: payload.__setitem__("labels", payload["labels"][..., :-1]),
    ),
    ids=("pixel", "missing", "nested", "attention-tensor", "position-shape", "extent"),
)
def test_codec_rejects_unauthorized_or_malformed_payload_fields(mutation):
    records = list(_records())
    records[0] = _replace_payload(records[0], mutation)

    with pytest.raises(MdpConfigurationError):
        MultimodalDecoderPayloadCodec().build_source_window(tuple(records), source_dp_lane=0)


def test_codec_rejects_cross_record_position_presence_mismatch():
    records = list(_records())
    records[0] = _replace_payload(
        records[0], lambda payload: payload.__setitem__("position_ids", None)
    )

    with pytest.raises(MdpConfigurationError, match="consistent|agree"):
        MultimodalDecoderPayloadCodec().build_source_window(tuple(records), source_dp_lane=0)


@pytest.mark.parametrize(
    ("field", "shape"),
    (
        ("input_ids", "batch-two"),
        ("labels", "rank-three"),
        ("loss_mask", "rank-one"),
        ("padding_mask", "batch-two"),
    ),
)
def test_codec_rejects_uniform_non_qwen_required_tensor_shapes(field, shape):
    def mutate(payload):
        value = payload[field]
        if shape == "batch-two":
            value = value.repeat(2, 1)
        elif shape == "rank-three":
            value = value.unsqueeze(0)
        else:
            value = value.squeeze(0)
        payload[field] = value

    records = _replace_all_payloads(_records(), mutate)
    with pytest.raises(MdpConfigurationError, match=r"shape \[1, T\]"):
        MultimodalDecoderPayloadCodec().build_source_window(records, source_dp_lane=0)


@pytest.mark.parametrize("shape", ("three-flat", "one-rank-three", "two-component"))
def test_codec_rejects_uniform_non_qwen_position_shapes(shape):
    def mutate(payload):
        value = payload["position_ids"]
        if shape == "three-flat":
            value = value.repeat(3, 1)
        elif shape == "one-rank-three":
            value = value.unsqueeze(0)
        else:
            value = value.repeat(2, 1, 1)
        payload["position_ids"] = value

    records = _replace_all_payloads(_records(), mutate)
    with pytest.raises(MdpConfigurationError, match=r"\[1, T\].*\[3, 1, T\]"):
        MultimodalDecoderPayloadCodec().build_source_window(records, source_dp_lane=0)


@pytest.mark.parametrize("mutation", ("grid-count", "grid-value", "slot", "text-only"))
def test_codec_rejects_malformed_vision_record_state(mutation):
    records = list(_records())
    record = records[0]
    if mutation == "grid-count":
        record = _replace_payload(
            record,
            lambda payload: payload.__setitem__("image_grid_thw", payload["image_grid_thw"][:1]),
        )
    elif mutation == "grid-value":
        item = replace(record.vision_items[0], grid_thw=(2, 2, 2))
        record = replace(record, vision_items=(item, *record.vision_items[1:]))
    elif mutation == "slot":
        item = replace(record.vision_items[1], decoder_positions=(9,))
        record = replace(record, vision_items=(record.vision_items[0], item))
    else:
        record = replace(record, text_only=True)
    records[0] = record

    with pytest.raises(MdpConfigurationError):
        MultimodalDecoderPayloadCodec().build_source_window(tuple(records), source_dp_lane=0)


def test_codec_rejects_overlapping_decoder_slots_within_one_sample():
    records = list(_records())
    record = records[0]
    second = replace(record.vision_items[1], sample_id=0, image_ordinal=1, decoder_positions=(1,))
    records[0] = replace(record, vision_items=(record.vision_items[0], second))

    with pytest.raises(MdpConfigurationError, match="unique"):
        MultimodalDecoderPayloadCodec().build_source_window(tuple(records), source_dp_lane=0)
