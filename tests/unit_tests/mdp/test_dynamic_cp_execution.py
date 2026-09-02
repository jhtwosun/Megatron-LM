# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Core-only contracts for MDP decoder Dynamic-CP execution metadata."""

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass, replace
from types import MappingProxyType

import pytest
import torch

from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderGlobalManifest,
    DecoderMicrobatchKey,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderSourceWindow,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
    build_decoder_global_manifest,
    build_decoder_source_window,
    finalize_decoder_source_window,
    validate_decoder_global_manifest,
    validate_decoder_payload_packet,
    validate_decoder_source_manifest,
    validate_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata, EncoderVisionItemMetadata
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankView

_DEFAULT_RECORDS = ((4, 6, ((0,),)), (5, 7, ((1,), (2, 3))))


def _packet(
    *,
    source_dp_lane: int,
    local_sample_order: int,
    valid_seqlen: int,
    padded_seqlen: int,
    position_components: int | None = 1,
) -> DecoderPayloadPacket:
    sample_id = GlobalSampleId(source_dp_lane, local_sample_order)
    input_ids = torch.arange(padded_seqlen, dtype=torch.int64).reshape(1, padded_seqlen)
    tensor_fields = {"input_ids": input_ids}
    field_specs = [
        DecoderTensorFieldSpec(
            name="input_ids",
            dtype=input_ids.dtype,
            shape=tuple(input_ids.shape),
            device_type=input_ids.device.type,
        )
    ]
    none_fields = ()
    if position_components is None:
        none_fields = ("position_ids",)
        position_sentinel = -1
    else:
        position_ids = torch.arange(padded_seqlen, dtype=torch.int64).repeat(position_components, 1)
        tensor_fields["position_ids"] = position_ids
        field_specs.append(
            DecoderTensorFieldSpec(
                name="position_ids",
                dtype=position_ids.dtype,
                shape=tuple(position_ids.shape),
                device_type=position_ids.device.type,
            )
        )
        position_sentinel = position_components
    header = DecoderPayloadHeaderV1(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        source_dp_lane=source_dp_lane,
        local_sample_order=local_sample_order,
        valid_seqlen=valid_seqlen,
        padded_seqlen=padded_seqlen,
        tensor_field_count=len(field_specs),
        none_field_count=len(none_fields),
        position_components_or_minus_one=position_sentinel,
    )
    return DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=sample_id,
        valid_seqlen=valid_seqlen,
        padded_seqlen=padded_seqlen,
        header=header.to_wire_tuple(),
        field_specs=tuple(field_specs),
        tensor_fields=MappingProxyType(tensor_fields),
        none_fields=none_fields,
    )


def _source_window(
    source_dp_lane: int, records=_DEFAULT_RECORDS, *, position_components: int | None = 1
) -> DecoderSourceWindow:
    samples = []
    items = []
    packets = []
    local_item_id = 0
    for local_sample_order, (valid_seqlen, padded_seqlen, item_offsets) in enumerate(records):
        sample_id = GlobalSampleId(source_dp_lane, local_sample_order)
        sample_items = []
        for image_ordinal, decoder_offsets in enumerate(item_offsets):
            item_id = GlobalVisionItemId(source_dp_lane, local_item_id)
            sample_items.append(
                EncoderVisionItemMetadata(
                    item_id=item_id, sample_id=sample_id, image_ordinal=image_ordinal
                )
            )
            items.append(
                DecoderVisionItemMetadata(
                    item_id=item_id,
                    sample_id=sample_id,
                    image_ordinal=image_ordinal,
                    grid_thw=(1, 1, max(1, len(decoder_offsets))),
                    output_rows=len(decoder_offsets),
                    decoder_offsets=decoder_offsets,
                )
            )
            local_item_id += 1
        samples.append(
            DecoderSampleMetadata(
                sample_id=sample_id,
                valid_seqlen=valid_seqlen,
                padded_seqlen=padded_seqlen,
                vision_items=tuple(sample_items),
            )
        )
        packets.append(
            _packet(
                source_dp_lane=source_dp_lane,
                local_sample_order=local_sample_order,
                valid_seqlen=valid_seqlen,
                padded_seqlen=padded_seqlen,
                position_components=position_components,
            )
        )
    return finalize_decoder_source_window(
        source_dp_lane=source_dp_lane,
        samples=tuple(reversed(samples)),
        items=tuple(reversed(items)),
        packets=tuple(reversed(packets)),
    )


class _InjectedCodec:
    def __init__(self, *, position_components: int | None = 1):
        self.position_components = position_components
        self.calls = []

    def build_source_window(self, records, *, source_dp_lane):
        self.calls.append((records, source_dp_lane))
        return _source_window(source_dp_lane, records, position_components=self.position_components)


def _assert_tensor_free(value):
    assert not isinstance(value, torch.Tensor)
    if is_dataclass(value):
        for field in dataclass_fields(value):
            _assert_tensor_free(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_tensor_free(key)
            _assert_tensor_free(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_tensor_free(item)


def _singleton_rank_view(**changes):
    values = {
        "global_rank": 70,
        "outer_dp_rank": 7,
        "lane_id": 7,
        "my_worker_id": 0,
        "endpoint_rank": 70,
        "planning_group_ranks": (70,),
        "worker_ids": (0,),
    }
    values.update(changes)
    return MdpRankView(**values)


def _singleton_plan(manifest, *, rank_view=None, sample_locations=None):
    if sample_locations is None:
        sample_locations = _singleton_sample_locations(manifest)
    descriptors = tuple(
        VisionDescriptor(
            global_item_id=item.item_id.local_item_id,
            sample_id=sample_locations[item.sample_id][1],
            image_ordinal=item.image_ordinal,
            owner_dp_lane=7,
            microbatch_id=sample_locations[item.sample_id][0],
            estimated_cost_units=item.grid_thw[0] * item.grid_thw[1] * item.grid_thw[2],
            payload_rows=item.grid_thw[0] * item.grid_thw[1] * item.grid_thw[2],
            output_rows=item.output_rows,
            grid_thw=item.grid_thw,
            owner_worker_id=0,
        )
        for item in manifest.items
    )
    return MdpPlanner(
        _singleton_rank_view() if rank_view is None else rank_view,
        locality_slack_permille=0,
        capacity_policy=RowCapacityPolicy(),
    ).build_plan(
        0, descriptors, tuple(sorted({location[0] for location in sample_locations.values()}))
    )


def _singleton_sample_locations(manifest):
    return {
        sample.sample_id: (sample.sample_id.local_sample_order, 0) for sample in manifest.samples
    }


def _validate_singleton_proof(**kwargs):
    from megatron.core.mdp import dynamic_cp_execution

    validator = getattr(dynamic_cp_execution, "_validate_local_singleton_producer_proof", None)
    assert callable(validator), "production must provide the singleton producer proof validator"
    return validator(**kwargs)


def test_singleton_producer_proof_accepts_exact_contributor_and_empty_noncontributor():
    window = _source_window(7)
    manifest = window.metadata_manifest()
    assert (
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=_singleton_plan(manifest),
            sample_location_by_id=_singleton_sample_locations(manifest),
        )
        is None
    )
    assert (
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(
                global_rank=80,
                outer_dp_rank=8,
                lane_id=None,
                endpoint_rank=80,
                planning_group_ranks=(80,),
            ),
            source_rank_by_lane={7: 70},
            local_manifest=None,
            source_window=None,
            static_plan=None,
            sample_location_by_id={},
        )
        is None
    )


def test_singleton_producer_proof_accepts_packed_microbatch_sample_locations():
    window = _source_window(7, records=((4, 6, ((0,),)), (5, 7, ((1,),))))
    manifest = window.metadata_manifest()
    locations = {manifest.samples[0].sample_id: (0, 0), manifest.samples[1].sample_id: (0, 1)}

    assert (
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=_singleton_plan(manifest, sample_locations=locations),
            sample_location_by_id=locations,
        )
        is None
    )


def test_singleton_producer_proof_rejects_duplicate_sample_locations():
    window = _source_window(7, records=((4, 6, ()), (5, 7, ())))
    manifest = window.metadata_manifest()
    locations = {sample.sample_id: (0, 0) for sample in manifest.samples}

    with pytest.raises(MdpPlanError):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=_singleton_plan(manifest),
            sample_location_by_id=locations,
        )


def test_singleton_producer_proof_rejects_sparse_microbatch_sample_locations():
    window = _source_window(7, records=((4, 6, ()), (5, 7, ())))
    manifest = window.metadata_manifest()
    locations = {manifest.samples[0].sample_id: (0, 0), manifest.samples[1].sample_id: (0, 2)}

    with pytest.raises(MdpPlanError):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=_singleton_plan(manifest),
            sample_location_by_id=locations,
        )


def test_singleton_producer_proof_rejects_reordered_packed_endpoint_segments():
    window = _source_window(7, records=((4, 6, ((0,),)), (5, 7, ((1,),))))
    manifest = window.metadata_manifest()
    locations = {manifest.samples[0].sample_id: (0, 0), manifest.samples[1].sample_id: (0, 1)}
    plan = _singleton_plan(manifest, sample_locations=locations)
    layout = plan.layouts[0]
    first, second = layout.segments
    reordered = replace(
        layout,
        segments=(
            replace(second, leaf_row_start=0),
            replace(first, leaf_row_start=second.output_rows),
        ),
    )

    with pytest.raises(MdpPlanError):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=replace(plan, layouts=(reordered,)),
            sample_location_by_id=locations,
        )


@pytest.mark.parametrize("residue", ("manifest", "window", "plan"))
def test_singleton_noncontributor_rejects_stale_source_residue(residue):
    window = _source_window(7)
    manifest = window.metadata_manifest()
    state = {"local_manifest": None, "source_window": None, "static_plan": None}
    state[
        {"manifest": "local_manifest", "window": "source_window", "plan": "static_plan"}[residue]
    ] = {"manifest": manifest, "window": window, "plan": _singleton_plan(manifest)}[residue]

    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(
                global_rank=80,
                outer_dp_rank=8,
                lane_id=None,
                endpoint_rank=80,
                planning_group_ranks=(80,),
            ),
            source_rank_by_lane={7: 70},
            sample_location_by_id={},
            **state,
        )


def test_singleton_contributor_rejects_mismatched_manifest_window_and_plan_digest():
    window = _source_window(7)
    manifest = window.metadata_manifest()
    other_window = _source_window(7, records=((4, 6, ()),))
    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=other_window,
            static_plan=_singleton_plan(manifest),
            sample_location_by_id=_singleton_sample_locations(manifest),
        )

    plan = _singleton_plan(manifest)
    object.__setattr__(plan, "digest", b"x" * 16)
    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=plan,
            sample_location_by_id=_singleton_sample_locations(manifest),
        )


@pytest.mark.parametrize("field", ("microbatch_id", "sample_id"))
def test_singleton_proof_rejects_encoder_segment_location_mutation(field):
    window = _source_window(7)
    manifest = window.metadata_manifest()
    plan = _singleton_plan(manifest)
    encoder_layout = plan.encoder_layouts[0]
    segment = encoder_layout.segments[0]
    mutated_segment = replace(segment, **{field: getattr(segment, field) + 1})
    mutated_layout = replace(
        encoder_layout, segments=(mutated_segment, *encoder_layout.segments[1:])
    )
    plan = replace(plan, encoder_layouts=(mutated_layout,))

    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=plan,
            sample_location_by_id=_singleton_sample_locations(manifest),
        )


def test_singleton_proof_rejects_layout_relocation_and_missing_text_only_layout():
    records = ((4, 6, ((0,),)), (5, 7, ()))
    window = _source_window(7, records=records)
    manifest = window.metadata_manifest()
    plan = _singleton_plan(manifest)
    locations = _singleton_sample_locations(manifest)

    relocated = replace(
        plan, layouts=(replace(plan.layouts[0], microbatch_id=9), *plan.layouts[1:])
    )
    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=relocated,
            sample_location_by_id=locations,
        )

    without_text_only = replace(
        plan, layouts=tuple(layout for layout in plan.layouts if not layout.text_only)
    )
    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        _validate_singleton_proof(
            rank_view=_singleton_rank_view(),
            source_rank_by_lane={7: 70},
            local_manifest=manifest,
            source_window=window,
            static_plan=without_text_only,
            sample_location_by_id=locations,
        )


def _mutate_digest_field(window, carrier):
    if carrier == "sample":
        sample = window.samples[0]
        packet = window.packets[0]
        valid_seqlen = sample.valid_seqlen + 1
        header = list(packet.header)
        header[3] = valid_seqlen
        return replace(
            window,
            samples=(replace(sample, valid_seqlen=valid_seqlen), *window.samples[1:]),
            packets=(
                replace(packet, valid_seqlen=valid_seqlen, header=tuple(header)),
                *window.packets[1:],
            ),
        )
    if carrier == "item":
        item = window.items[0]
        grid_thw = (*item.grid_thw[:-1], item.grid_thw[-1] + 1)
        return replace(window, items=(replace(item, grid_thw=grid_thw), *window.items[1:]))

    packet = window.packets[0]
    field_spec = replace(packet.field_specs[0], dtype=torch.int32)
    tensors = dict(packet.tensor_fields)
    tensors[field_spec.name] = tensors[field_spec.name].to(torch.int32)
    return replace(
        window,
        packets=(
            replace(
                packet,
                field_specs=(field_spec, *packet.field_specs[1:]),
                tensor_fields=MappingProxyType(tensors),
            ),
            *window.packets[1:],
        ),
    )


def test_payload_header_round_trips_fixed_schema_order():
    header = DecoderPayloadHeaderV1(
        schema_version=1,
        source_dp_lane=7,
        local_sample_order=2,
        valid_seqlen=5,
        padded_seqlen=8,
        tensor_field_count=2,
        none_field_count=0,
        position_components_or_minus_one=3,
    )

    assert header.to_wire_tuple() == (1, 7, 2, 5, 8, 2, 0, 3)
    assert DecoderPayloadHeaderV1.from_wire_tuple(header.to_wire_tuple()) == header


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 2),
        ("valid_seqlen", 0),
        ("valid_seqlen", 9),
        ("position_components_or_minus_one", 2),
    ),
)
def test_payload_header_rejects_invalid_schema_values(field, value):
    values = dict(
        schema_version=1,
        source_dp_lane=0,
        local_sample_order=0,
        valid_seqlen=4,
        padded_seqlen=8,
        tensor_field_count=2,
        none_field_count=0,
        position_components_or_minus_one=1,
    )
    values[field] = value

    with pytest.raises(MdpConfigurationError):
        DecoderPayloadHeaderV1(**values)


def test_payload_header_rejects_noncanonical_wire_carrier():
    with pytest.raises(MdpConfigurationError):
        DecoderPayloadHeaderV1.from_wire_tuple([1, 0, 0, 4, 4, 2, 0, 1])


def test_microbatch_key_and_field_spec_validate_scalar_contracts():
    assert DecoderMicrobatchKey(3).microbatch_index == 3
    assert DecoderTensorFieldSpec("x", torch.float32, (2, 3), "cpu").element_count == 6
    with pytest.raises(MdpConfigurationError):
        DecoderMicrobatchKey(True)


def test_packet_metadata_excludes_tensor_payload():
    packet = _packet(source_dp_lane=3, local_sample_order=0, valid_seqlen=4, padded_seqlen=6)

    validate_decoder_payload_packet(packet)
    metadata = packet.metadata()
    assert metadata.sample_id == packet.sample_id
    assert metadata.field_specs == packet.field_specs
    assert not hasattr(metadata, "tensor_fields")


@pytest.mark.parametrize(
    "malformation", ("field_order", "tensor_dtype", "header_count", "position_sentinel")
)
def test_packet_validator_rejects_malformed_carriers(malformation):
    packet = _packet(source_dp_lane=0, local_sample_order=0, valid_seqlen=4, padded_seqlen=6)
    if malformation == "field_order":
        packet = replace(
            packet,
            tensor_fields=MappingProxyType(
                {
                    "position_ids": packet.tensor_fields["position_ids"],
                    "input_ids": packet.tensor_fields["input_ids"],
                }
            ),
        )
    elif malformation == "tensor_dtype":
        tensors = dict(packet.tensor_fields)
        tensors["input_ids"] = tensors["input_ids"].to(torch.float32)
        packet = replace(packet, tensor_fields=MappingProxyType(tensors))
    else:
        header = list(packet.header)
        header[5 if malformation == "header_count" else 7] = (
            99 if malformation == "header_count" else 3
        )
        packet = replace(packet, header=tuple(header))

    with pytest.raises(MdpConfigurationError):
        validate_decoder_payload_packet(packet)


def test_packet_validator_rejects_tensor_extent_that_only_matches_its_spec():
    packet = _packet(source_dp_lane=0, local_sample_order=0, valid_seqlen=4, padded_seqlen=6)
    tensors = dict(packet.tensor_fields)
    tensors["input_ids"] = tensors["input_ids"][..., :-1]
    input_spec = replace(packet.field_specs[0], shape=tuple(tensors["input_ids"].shape))
    packet = replace(
        packet,
        field_specs=(input_spec, *packet.field_specs[1:]),
        tensor_fields=MappingProxyType(tensors),
    )

    with pytest.raises(MdpConfigurationError, match="extent matches padded_seqlen"):
        validate_decoder_payload_packet(packet)


def test_source_finalizer_canonicalizes_and_freezes_codec_output():
    window = _source_window(5)

    validate_decoder_source_window(window)
    assert window.sample_ids == (GlobalSampleId(5, 0), GlobalSampleId(5, 1))
    assert tuple(item.item_id.local_item_id for item in window.items) == (0, 1, 2)
    assert tuple(packet.sample_id for packet in window.packets) == window.sample_ids
    with pytest.raises(TypeError):
        window.packets[0].tensor_fields["new_field"] = torch.ones(1)


def test_public_source_builder_uses_injected_codec_boundary():
    codec = _InjectedCodec()

    window = build_decoder_source_window(_DEFAULT_RECORDS, source_dp_lane=9, codec=codec)

    assert codec.calls == [(_DEFAULT_RECORDS, 9)]
    assert window.source_dp_lane == 9
    assert window.sample_ids == (GlobalSampleId(9, 0), GlobalSampleId(9, 1))


def test_public_source_builder_rejects_missing_codec_method():
    with pytest.raises(MdpConfigurationError, match="provides build_source_window"):
        build_decoder_source_window(_DEFAULT_RECORDS, source_dp_lane=0, codec=object())


def test_public_source_builder_normalizes_untyped_codec_failure():
    class _BrokenCodec:
        def build_source_window(self, records, *, source_dp_lane):
            raise RuntimeError("codec bug")

    with pytest.raises(MdpConfigurationError, match="failed to build") as exc_info:
        build_decoder_source_window(_DEFAULT_RECORDS, source_dp_lane=0, codec=_BrokenCodec())
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_public_source_builder_rejects_codec_lane_remap():
    class _WrongLaneCodec:
        def build_source_window(self, records, *, source_dp_lane):
            return _source_window(source_dp_lane + 1, records)

    with pytest.raises(MdpConfigurationError, match="lane mapping mismatch"):
        build_decoder_source_window(_DEFAULT_RECORDS, source_dp_lane=4, codec=_WrongLaneCodec())


def test_public_source_builder_rejects_stale_codec_window():
    class _StaleCodec(_InjectedCodec):
        def build_source_window(self, records, *, source_dp_lane):
            window = super().build_source_window(records, source_dp_lane=source_dp_lane)
            return replace(window, digest=b"\x00" * 16)

    with pytest.raises(MdpPlanError, match="digest matches"):
        build_decoder_source_window(_DEFAULT_RECORDS, source_dp_lane=4, codec=_StaleCodec())


def test_source_manifest_is_payload_free_and_keeps_source_digest():
    window = _source_window(2)

    manifest = window.metadata_manifest()

    validate_decoder_source_manifest(manifest)
    assert manifest.digest == window.digest
    assert tuple(payload.sample_id for payload in manifest.payloads) == window.sample_ids
    assert all(not hasattr(payload, "tensor_fields") for payload in manifest.payloads)


def test_source_manifest_rejects_payload_extent_that_does_not_match_header():
    manifest = _source_window(2).metadata_manifest()
    payload = manifest.payloads[0]
    input_spec = replace(payload.field_specs[0], shape=(1, payload.padded_seqlen - 1))
    payload = replace(payload, field_specs=(input_spec, *payload.field_specs[1:]))
    manifest = replace(manifest, payloads=(payload, *manifest.payloads[1:]))

    with pytest.raises(MdpConfigurationError, match="extent matches padded_seqlen"):
        validate_decoder_source_manifest(manifest)


def test_source_validator_rejects_stale_digest():
    window = _source_window(2)

    with pytest.raises(MdpPlanError, match="digest matches"):
        validate_decoder_source_window(replace(window, digest=b"\x00" * 16))


@pytest.mark.parametrize("carrier", ("sample", "item", "payload"))
def test_source_digest_binds_representative_metadata_fields(carrier):
    window = _source_window(0)

    with pytest.raises(MdpPlanError, match="digest matches"):
        validate_decoder_source_window(_mutate_digest_field(window, carrier))


@pytest.mark.parametrize("malformation", ("missing", "remapped", "overlapping"))
def test_source_validator_rejects_invalid_vision_catalog(malformation):
    window = _source_window(0)
    if malformation == "missing":
        malformed = replace(window, items=window.items[:-1])
        expected_error = MdpPlanError
    elif malformation == "remapped":
        item = replace(window.items[-1], sample_id=window.samples[0].sample_id)
        malformed = replace(window, items=(*window.items[:-1], item))
        expected_error = MdpPlanError
    else:
        item = replace(
            window.items[-1],
            output_rows=window.items[-2].output_rows,
            decoder_offsets=window.items[-2].decoder_offsets,
        )
        malformed = replace(window, items=(*window.items[:-1], item))
        expected_error = MdpConfigurationError

    with pytest.raises(expected_error):
        validate_decoder_source_window(malformed)


def test_global_manifest_joins_lanes_canonically_without_payload_tensors():
    lane_7 = _source_window(7, records=((4, 5, ((0,),)),)).metadata_manifest()
    lane_3 = _source_window(3, records=((5, 8, ((1,),)),)).metadata_manifest()

    manifest = build_decoder_global_manifest((lane_7, lane_3))

    validate_decoder_global_manifest(manifest)
    assert manifest.sample_ids == (GlobalSampleId(3, 0), GlobalSampleId(7, 0))
    assert tuple(item.item_id.source_dp_lane for item in manifest.items) == (3, 7)
    assert all(not hasattr(payload, "tensor_fields") for payload in manifest.payloads)
    assert len(manifest.digest) == 16
    _assert_tensor_free(manifest)


def test_global_manifest_is_identical_for_reversed_source_input():
    sources = (_source_window(0).metadata_manifest(), _source_window(1).metadata_manifest())

    forward = build_decoder_global_manifest(sources)
    reversed_input = build_decoder_global_manifest(tuple(reversed(sources)))

    assert reversed_input == forward
    assert reversed_input.digest == forward.digest


def test_source_and_global_manifest_digests_are_domain_separated():
    source = _source_window(0).metadata_manifest()

    global_manifest = build_decoder_global_manifest((source,))

    assert global_manifest.samples == source.samples
    assert global_manifest.items == source.items
    assert global_manifest.payloads == source.payloads
    assert global_manifest.digest != source.digest


def test_global_builder_rejects_stale_source_manifest():
    source = _source_window(0).metadata_manifest()

    with pytest.raises(MdpPlanError, match="digest matches"):
        build_decoder_global_manifest((replace(source, digest=b"\x00" * 16),))


def test_global_manifest_rejects_duplicate_source_lane():
    manifest = _source_window(0).metadata_manifest()

    with pytest.raises(MdpPlanError, match="source lanes are unique"):
        build_decoder_global_manifest((manifest, manifest))


@pytest.mark.parametrize("position_components", (None, 3))
def test_global_manifest_rejects_incompatible_payload_schema(position_components):
    lane_0 = _source_window(0, records=((4, 5, ((0,),)),)).metadata_manifest()
    lane_1 = _source_window(
        1, records=((4, 7, ((0,),)),), position_components=position_components
    ).metadata_manifest()

    with pytest.raises(MdpConfigurationError, match="globally consistent|globally compatible"):
        build_decoder_global_manifest((lane_0, lane_1))


def test_global_manifest_validator_rejects_stale_digest():
    manifest = build_decoder_global_manifest(
        (_source_window(0).metadata_manifest(), _source_window(1).metadata_manifest())
    )
    stale = DecoderGlobalManifest(
        samples=manifest.samples,
        items=manifest.items,
        payloads=manifest.payloads,
        digest=b"\xff" * 16,
    )

    with pytest.raises(MdpPlanError, match="digest matches"):
        validate_decoder_global_manifest(stale)
