# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 metadata transport wire and collective contracts."""

import os
from datetime import timedelta
from importlib import import_module
from types import MappingProxyType

import pytest
import torch

from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata, EncoderVisionItemMetadata
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError


def _transport_api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_metadata_transport")


def _source_manifest(lane=0):
    sample_id = GlobalSampleId(lane, 0)
    item_id = GlobalVisionItemId(lane, 0)
    sample = DecoderSampleMetadata(
        sample_id=sample_id,
        valid_seqlen=4,
        padded_seqlen=4,
        vision_items=(EncoderVisionItemMetadata(item_id, sample_id, 0),),
    )
    item = DecoderVisionItemMetadata(
        item_id=item_id,
        sample_id=sample_id,
        image_ordinal=0,
        grid_thw=(1, 1, 1),
        output_rows=1,
        decoder_offsets=(1,),
    )
    tensors = {
        "input_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
        "position_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
    }
    fields = tuple(
        DecoderTensorFieldSpec(name, tensor.dtype, tuple(tensor.shape), tensor.device.type)
        for name, tensor in tensors.items()
    )
    header = DecoderPayloadHeaderV1(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        source_dp_lane=lane,
        local_sample_order=0,
        valid_seqlen=4,
        padded_seqlen=4,
        tensor_field_count=len(fields),
        none_field_count=1,
        position_components_or_minus_one=1,
    ).to_wire_tuple()
    packet = DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=sample_id,
        valid_seqlen=4,
        padded_seqlen=4,
        header=header,
        field_specs=fields,
        tensor_fields=MappingProxyType(tensors),
        none_fields=("attention_mask",),
    )
    return finalize_decoder_source_window(
        source_dp_lane=lane, samples=(sample,), items=(item,), packets=(packet,)
    ).metadata_manifest()


def _two_sample_manifest():
    first = _source_manifest()
    first_sample, first_item = first.samples[0], first.items[0]
    first_payload = first.payloads[0]
    second_id = GlobalSampleId(0, 1)
    second_item_id = GlobalVisionItemId(0, 1)
    second_sample = DecoderSampleMetadata(
        sample_id=second_id,
        valid_seqlen=4,
        padded_seqlen=4,
        vision_items=(EncoderVisionItemMetadata(second_item_id, second_id, 0),),
    )
    second_item = DecoderVisionItemMetadata(
        item_id=second_item_id,
        sample_id=second_id,
        image_ordinal=0,
        grid_thw=first_item.grid_thw,
        output_rows=first_item.output_rows,
        decoder_offsets=first_item.decoder_offsets,
    )
    header = list(first_payload.header)
    header[2] = 1
    packet = DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=second_id,
        valid_seqlen=4,
        padded_seqlen=4,
        header=tuple(header),
        field_specs=first_payload.field_specs,
        tensor_fields=MappingProxyType(
            {
                "input_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
                "position_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
            }
        ),
        none_fields=first_payload.none_fields,
    )
    return finalize_decoder_source_window(
        source_dp_lane=0,
        samples=(first_sample, second_sample),
        items=(first_item, second_item),
        packets=(
            DecoderPayloadPacket(
                schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
                sample_id=first_payload.sample_id,
                valid_seqlen=first_payload.valid_seqlen,
                padded_seqlen=first_payload.padded_seqlen,
                header=first_payload.header,
                field_specs=first_payload.field_specs,
                tensor_fields=MappingProxyType(
                    {
                        "input_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
                        "position_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
                    }
                ),
                none_fields=first_payload.none_fields,
            ),
            packet,
        ),
    ).metadata_manifest()


def test_source_manifest_codec_round_trip_is_exact_and_payload_free():
    api = _transport_api()
    manifest = _source_manifest(2)

    wire = api.encode_decoder_source_manifest(manifest)

    assert api.decode_decoder_source_manifest(wire) == manifest
    assert wire == api.encode_decoder_source_manifest(manifest)
    assert isinstance(wire, tuple)
    assert all(type(word) is int for word in wire)
    assert all(not isinstance(word, torch.Tensor) for word in wire)


@pytest.mark.parametrize(
    "corrupt",
    (
        lambda wire: wire[:-1],
        lambda wire: (*wire, 0),
        lambda wire: (True, *wire[1:]),
        lambda wire: (*wire[:-1], wire[-1] ^ 1),
    ),
)
def test_source_manifest_codec_rejects_truncation_trailing_nonint_and_digest(corrupt):
    api = _transport_api()

    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        api.decode_decoder_source_manifest(
            corrupt(api.encode_decoder_source_manifest(_source_manifest()))
        )


def test_source_manifest_codec_reuses_carrier_validation_for_noncanonical_input():
    api = _transport_api()
    manifest = _source_manifest()

    with pytest.raises(MdpPlanError):
        api.encode_decoder_source_manifest(
            type(manifest)(
                source_dp_lane=manifest.source_dp_lane,
                samples=manifest.samples,
                items=manifest.items,
                payloads=manifest.payloads,
                digest=bytes(reversed(manifest.digest)),
            )
        )


def test_source_manifest_codec_rejects_decodable_noncanonical_sample_order():
    api = _transport_api()
    wire = api.encode_decoder_source_manifest(_two_sample_manifest())
    # Each fixture sample is its two-id, two-length, one-item descriptor block.
    first, second = wire[3:13], wire[13:23]
    noncanonical = (*wire[:3], *second, *first, *wire[23:])

    with pytest.raises(MdpPlanError):
        api.decode_decoder_source_manifest(noncanonical)


def _replace_word(wire, index, value):
    return (*wire[:index], value, *wire[index + 1 :])


_WIRE = {
    "version": 0,
    "source_lane": 1,
    "sample_count": 2,
    "sample_id_lane": 3,
    "sample_valid_length": 5,
    "encoder_item_id_lane": 8,
    "encoder_item_ordinal": 12,
    "decoder_item_id_lane": 14,
    "decoder_item_grid": 19,
    "decoder_item_rows": 22,
    "decoder_item_offset_count": 23,
    "payload_sample_id_lane": 26,
    "payload_valid_length": 28,
    "payload_header_width": 30,
    "payload_header_schema": 31,
    "payload_header_tensor_count": 36,
    "field_name_first_byte": 41,
    "field_dtype_last_byte": 61,
    "field_shape_dimension": 63,
    "none_field_first_byte": 101,
}


@pytest.mark.parametrize(
    "corrupt",
    (
        lambda wire: _replace_word(wire, _WIRE["version"], 2),
        lambda wire: _replace_word(wire, _WIRE["source_lane"], 1),
        lambda wire: _replace_word(wire, _WIRE["sample_count"], -1),
        lambda wire: _replace_word(wire, _WIRE["sample_id_lane"], 1),
        lambda wire: _replace_word(wire, _WIRE["sample_valid_length"], 0),
        lambda wire: _replace_word(wire, _WIRE["encoder_item_id_lane"], 1),
        lambda wire: _replace_word(wire, _WIRE["encoder_item_ordinal"], 1),
        lambda wire: _replace_word(wire, _WIRE["decoder_item_id_lane"], 1),
        lambda wire: _replace_word(wire, _WIRE["decoder_item_grid"], 0),
        lambda wire: _replace_word(wire, _WIRE["decoder_item_rows"], 0),
        lambda wire: _replace_word(wire, _WIRE["decoder_item_offset_count"], 0),
        lambda wire: _replace_word(wire, _WIRE["payload_sample_id_lane"], 1),
        lambda wire: _replace_word(wire, _WIRE["payload_valid_length"], 0),
        lambda wire: _replace_word(wire, _WIRE["payload_header_width"], 7),
        lambda wire: _replace_word(wire, _WIRE["payload_header_schema"], 2),
        lambda wire: _replace_word(wire, _WIRE["payload_header_tensor_count"], 1),
        lambda wire: _replace_word(wire, _WIRE["field_name_first_byte"], 255),
        lambda wire: _replace_word(wire, _WIRE["field_dtype_last_byte"], ord("x")),
        lambda wire: _replace_word(wire, _WIRE["field_shape_dimension"], 0),
        lambda wire: _replace_word(wire, _WIRE["none_field_first_byte"], 255),
        lambda wire: _replace_word(wire, _WIRE["version"], 2**63),
    ),
)
def test_source_manifest_codec_rejects_field_family_malformed_wires(corrupt):
    api = _transport_api()

    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        api.decode_decoder_source_manifest(
            corrupt(api.encode_decoder_source_manifest(_source_manifest()))
        )


@pytest.mark.parametrize("timeout", (0, -1, 0.000999, float("inf"), float("nan")))
def test_metadata_timeout_rejects_sub_millisecond_and_nonfinite_values(timeout):
    api = _transport_api()

    with pytest.raises(MdpConfigurationError):
        api._prepare_timeout(timeout)
    assert api._prepare_timeout(0.001)[1] == timedelta(milliseconds=1)


def test_metadata_status_rejects_unknown_error_before_body(monkeypatch):
    api = _transport_api()
    body_called = False

    def status_gather(value, *, timeout_seconds):
        return ((value[0], value[1], 2, value[3], value[4], value[5], value[6]),)

    def body(*args, **kwargs):
        nonlocal body_called
        body_called = True
        raise AssertionError("must not gather a body after rejected status")

    monkeypatch.setattr(api, "make_precollective_status_gather", lambda **_: status_gather)
    monkeypatch.setattr(api, "_gather_body", body)
    with pytest.raises(MdpPlanError):
        api.gather_decoder_source_manifests(
            _source_manifest(),
            expected_source_lanes=(0,),
            group=object(),
            group_ranks=(0,),
            global_rank=0,
            device=torch.device("cuda", 0),
            timeout_seconds=0.001,
        )
    assert not body_called


def test_local_prepare_error_is_the_source_rank_status_rejection_cause(monkeypatch):
    api = _transport_api()
    original = RuntimeError("rank-local prepare failure")
    body_called = False

    def status_gather(value, *, timeout_seconds):
        return (value,)

    def body(*args, **kwargs):
        nonlocal body_called
        body_called = True
        raise AssertionError("must not gather a body after rejected status")

    monkeypatch.setattr(api, "make_precollective_status_gather", lambda **_: status_gather)
    monkeypatch.setattr(api, "_gather_body", body)
    with pytest.raises(MdpPlanError, match="preparation failed before body") as caught:
        api.gather_decoder_source_manifests(
            _source_manifest(),
            expected_source_lanes=(0,),
            group=object(),
            group_ranks=(0,),
            global_rank=0,
            device=torch.device("cuda", 0),
            timeout_seconds=0.001,
            local_prepare_error=original,
        )

    assert caught.value.__cause__ is original
    assert not body_called


def test_metadata_transport_rejects_cpu_before_status_factory(monkeypatch):
    api = _transport_api()
    factory_called = False

    def status_factory(**kwargs):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("CPU device must reject before status factory binding")

    monkeypatch.setattr(api, "make_precollective_status_gather", status_factory)
    with pytest.raises(MdpConfigurationError):
        api.gather_decoder_source_manifests(
            _source_manifest(),
            expected_source_lanes=(0,),
            group=object(),
            group_ranks=(0,),
            global_rank=0,
            device=torch.device("cpu"),
            timeout_seconds=0.001,
        )
    assert not factory_called


def test_metadata_transport_converges_malformed_body_before_return(monkeypatch):
    api = _transport_api()
    body = api.encode_decoder_source_manifest(_source_manifest())
    statuses = []

    def status_gather(value, *, timeout_seconds):
        statuses.append(value)
        return (value,)

    monkeypatch.setattr(api, "make_precollective_status_gather", lambda **_: status_gather)
    monkeypatch.setattr(api, "_gather_body", lambda *_, **__: (body[:-1],))

    with pytest.raises(MdpPlanError):
        api.gather_decoder_source_manifests(
            _source_manifest(),
            expected_source_lanes=(0,),
            group=object(),
            group_ranks=(0,),
            global_rank=0,
            device=torch.device("cuda", 0),
            timeout_seconds=1.0,
        )
    assert len(statuses) == 2
    assert statuses[1][2] == 1


@pytest.mark.skipif(
    os.environ.get("MDP_D3_METADATA_WORLD1") != "1",
    reason="requires the explicit one-rank NCCL srun",
)
def test_world1_nccl_metadata_gather():
    """The real single-rank NCCL path returns immutable lane authority."""
    api = _transport_api()
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl")
    try:
        result = api.gather_decoder_source_manifests(
            _source_manifest(),
            expected_source_lanes=(0,),
            group=torch.distributed.group.WORLD,
            group_ranks=(0,),
            global_rank=0,
            device=torch.device("cuda", 0),
            timeout_seconds=10.0,
        )
        assert result.global_manifest.sample_ids == (GlobalSampleId(0, 0),)
        assert dict(result.source_rank_by_lane) == {0: 0}
        with pytest.raises(TypeError):
            result.source_rank_by_lane[1] = 1
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("MDP_D3_METADATA_WORLD4") != "1",
    reason="requires the explicit four-rank NCCL srun",
)
def test_world4_nccl_metadata_gather_rejects_before_body_and_retries(monkeypatch):
    """Both pre-body and post-body asymmetric failures converge and retry safely."""
    api = _transport_api()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    ranks = tuple(range(world_size))
    common = dict(
        expected_source_lanes=ranks,
        group=torch.distributed.group.WORLD,
        group_ranks=ranks,
        global_rank=rank,
        device=device,
        timeout_seconds=10.0,
    )
    try:
        original_gather_body = api._gather_body
        body_calls = 0

        def count_body(*args, **kwargs):
            nonlocal body_calls
            body_calls += 1
            return original_gather_body(*args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(api, "_gather_body", count_body)
            local_prepare_error = RuntimeError("rank-local prepare failure") if rank == 2 else None
            with pytest.raises(MdpPlanError) as caught:
                api.gather_decoder_source_manifests(
                    _source_manifest(rank), local_prepare_error=local_prepare_error, **common
                )
            assert body_calls == 0
            if rank == 2:
                assert caught.value.__cause__ is local_prepare_error
                assert str(caught.value.__cause__) == "rank-local prepare failure"
            else:
                assert caught.value.__cause__ is None
        torch.distributed.barrier()

        body_calls = 0
        with monkeypatch.context() as patched:
            patched.setattr(api, "_gather_body", count_body)
            with pytest.raises(MdpPlanError):
                api.gather_decoder_source_manifests(
                    _source_manifest(3 if rank == 2 else rank), **common
                )
            assert body_calls == 0
        torch.distributed.barrier()

        def truncate_one_rank(*args, **kwargs):
            rows = original_gather_body(*args, **kwargs)
            if rank != 2:
                return rows
            return (rows[0][:-1], *rows[1:])

        with monkeypatch.context() as patched:
            patched.setattr(api, "_gather_body", truncate_one_rank)
            with pytest.raises(MdpPlanError):
                api.gather_decoder_source_manifests(_source_manifest(rank), **common)
        torch.distributed.barrier()

        result = api.gather_decoder_source_manifests(_source_manifest(rank), **common)
        assert result.global_manifest.sample_ids == tuple(GlobalSampleId(lane, 0) for lane in ranks)
        assert dict(result.source_rank_by_lane) == {lane: lane for lane in ranks}
    finally:
        torch.distributed.destroy_process_group()
