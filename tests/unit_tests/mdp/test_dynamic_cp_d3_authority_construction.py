# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 producer/output authority construction contracts."""

from importlib import import_module
from types import MappingProxyType

import pytest
import torch

from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
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
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError


def _authority_api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_authority_construction")


def _source_manifest(lane):
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
        output_rows=lane + 1,
        decoder_offsets=tuple(range(lane + 1)),
    )
    tensors = {
        "input_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
        "position_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
    }
    fields = tuple(
        DecoderTensorFieldSpec(name, tensor.dtype, tuple(tensor.shape), tensor.device.type)
        for name, tensor in tensors.items()
    )
    packet = DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=sample_id,
        valid_seqlen=4,
        padded_seqlen=4,
        header=DecoderPayloadHeaderV1(
            schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
            source_dp_lane=lane,
            local_sample_order=0,
            valid_seqlen=4,
            padded_seqlen=4,
            tensor_field_count=len(fields),
            none_field_count=1,
            position_components_or_minus_one=1,
        ).to_wire_tuple(),
        field_specs=fields,
        tensor_fields=MappingProxyType(tensors),
        none_fields=("attention_mask",),
    )
    return finalize_decoder_source_window(
        source_dp_lane=lane, samples=(sample,), items=(item,), packets=(packet,)
    ).metadata_manifest()


def _metadata():
    manifests = (_source_manifest(0), _source_manifest(1))
    return DecoderMetadataGatherResult(
        global_manifest=build_decoder_global_manifest(manifests), source_rank_by_lane={0: 3, 1: 5}
    )


def test_derives_immutable_producer_and_output_authority_in_manifest_item_order():
    api = _authority_api()
    metadata = _metadata()

    authority = api.derive_decoder_item_authority(
        metadata, participant_ranks=(3, 5, 7), decoder_ranks=(5, 7)
    )

    item_ids = tuple(item.item_id for item in metadata.global_manifest.items)
    assert tuple(authority.producer_rank_by_item) == item_ids
    assert tuple(authority.output_rows_by_item) == item_ids
    assert dict(authority.producer_rank_by_item) == {item_ids[0]: 3, item_ids[1]: 5}
    assert dict(authority.output_rows_by_item) == {item_ids[0]: 1, item_ids[1]: 2}
    with pytest.raises(TypeError):
        authority.producer_rank_by_item[item_ids[0]] = 99


def test_preserves_authoritative_nonnumeric_rank_order():
    api = _authority_api()

    authority = api.derive_decoder_item_authority(
        _metadata(), participant_ranks=(7, 3, 5), decoder_ranks=(5, 7)
    )

    assert authority.participant_ranks == (7, 3, 5)
    assert authority.decoder_ranks == (5, 7)


@pytest.mark.parametrize(
    ("participant_ranks", "decoder_ranks"),
    (
        ((3, 3, 7), (3, 7)),
        ((3, True, 7), (3, 7)),
        ((3, -1, 7), (3, 7)),
        ((3, 2**63, 7), (3, 7)),
        ((3, 5, 7), (5, 9)),
    ),
)
def test_rejects_invalid_or_nonparticipant_rank_tuples(participant_ranks, decoder_ranks):
    api = _authority_api()

    with pytest.raises(MdpConfigurationError):
        api.derive_decoder_item_authority(
            _metadata(), participant_ranks=participant_ranks, decoder_ranks=decoder_ranks
        )


@pytest.mark.parametrize(
    "source_authority", ({0: 3}, {0: 3, 1: 5, 2: 7}, {0: 3, 1: 9}, {0: 3, 1: 3}, {0: 3, True: 5})
)
def test_rejects_mutated_metadata_authority_before_deriving_maps(source_authority):
    api = _authority_api()
    metadata = _metadata()
    object.__setattr__(metadata, "source_rank_by_lane", source_authority)

    with pytest.raises(MdpPlanError):
        api.derive_decoder_item_authority(
            metadata, participant_ranks=(3, 5, 7), decoder_ranks=(5, 7)
        )


@pytest.mark.parametrize(
    "producer_or_rows",
    (
        "producer_bool",
        "producer_overflow",
        "producer_bool_equal_one",
        "rows_bool",
        "rows_zero",
        "wrong_key",
        "source_extra_lane",
        "source_duplicate_rank",
    ),
)
def test_direct_authority_rejects_nonexact_item_keys_and_values(producer_or_rows):
    api = _authority_api()
    authority = api.derive_decoder_item_authority(
        _metadata(), participant_ranks=(3, 5, 7), decoder_ranks=(5, 7)
    )
    producers = dict(authority.producer_rank_by_item)
    rows = dict(authority.output_rows_by_item)
    source_authority = dict(authority.source_rank_by_lane)
    participants = authority.participant_ranks
    first_item = next(iter(producers))
    if producer_or_rows == "producer_bool":
        producers[first_item] = True
    elif producer_or_rows == "producer_overflow":
        producers[first_item] = 2**63
    elif producer_or_rows == "producer_bool_equal_one":
        source_authority[0] = 1
        producers[first_item] = True
        participants = (1, 5, 7)
    elif producer_or_rows == "rows_bool":
        rows[first_item] = True
    elif producer_or_rows == "rows_zero":
        rows[first_item] = 0
    else:
        if producer_or_rows == "source_extra_lane":
            source_authority[2] = 7
        elif producer_or_rows == "source_duplicate_rank":
            source_authority[1] = 3
        else:
            value = producers.pop(first_item)
            producers[object()] = value

    with pytest.raises(MdpPlanError):
        api.DecoderItemAuthority(
            global_manifest=authority.global_manifest,
            source_rank_by_lane=source_authority,
            producer_rank_by_item=producers,
            output_rows_by_item=rows,
            participant_ranks=participants,
            decoder_ranks=authority.decoder_ranks,
        )


class _FullGroupSolver:
    def __init__(self):
        self.calls = []

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size):
        self.calls.append((tuple(sample_seqlens), total_gpus, max_seq_len_per_rank, min_cp_size))
        sample_ids = [sample_id for sample_id, _ in sample_seqlens]
        lengths = [length for _, length in sample_seqlens]
        return ([lengths] * total_gpus, [], None, [sample_ids] * total_gpus)


def _item_authority(api):
    return api.derive_decoder_item_authority(
        _metadata(), participant_ranks=(3, 5, 7), decoder_ranks=(5, 7)
    )


def test_builds_exact_typed_iteration_authority_deterministically():
    api = _authority_api()
    solver = _FullGroupSolver()
    item_authority = _item_authority(api)

    first = api.build_d3_iteration_authority(
        item_authority,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=solver,
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    second = api.build_d3_iteration_authority(
        item_authority,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )

    assert type(first) is api._DynamicIterationAuthority
    assert first == second
    assert solver.calls == [(((0, 4), (1, 4)), 2, 8, 1)]
    assert first.participant_ranks == (3, 5, 7)
    assert first.bridge_width == 16
    assert first.bridge_dtype is torch.bfloat16
    with pytest.raises(TypeError):
        first.producer_rank_by_item[next(iter(first.producer_rank_by_item))] = 9


def test_invalid_item_authority_rejects_before_solver():
    api = _authority_api()
    valid = _item_authority(api)
    invalid = object.__new__(api.DecoderItemAuthority)
    object.__setattr__(invalid, "global_manifest", valid.global_manifest)
    object.__setattr__(invalid, "source_rank_by_lane", valid.source_rank_by_lane)
    object.__setattr__(
        invalid, "producer_rank_by_item", {next(iter(valid.producer_rank_by_item)): 99}
    )
    object.__setattr__(invalid, "output_rows_by_item", valid.output_rows_by_item)
    object.__setattr__(invalid, "participant_ranks", valid.participant_ranks)
    object.__setattr__(invalid, "decoder_ranks", valid.decoder_ranks)
    solver = _FullGroupSolver()
    snapshots = (
        dict(invalid.source_rank_by_lane),
        dict(invalid.producer_rank_by_item),
        dict(invalid.output_rows_by_item),
        invalid.participant_ranks,
        invalid.decoder_ranks,
    )

    with pytest.raises(MdpPlanError):
        api.build_d3_iteration_authority(
            invalid,
            max_seqlen_per_rank=8,
            minimum_cp_size=1,
            solver=solver,
            bridge_width=16,
            bridge_dtype=torch.bfloat16,
        )
    assert not solver.calls
    assert snapshots == (
        dict(invalid.source_rank_by_lane),
        dict(invalid.producer_rank_by_item),
        dict(invalid.output_rows_by_item),
        invalid.participant_ranks,
        invalid.decoder_ranks,
    )


@pytest.mark.parametrize("solver", (lambda *args: None, lambda *args: ([], [], [], [])))
def test_rejects_malformed_solver_output(solver):
    api = _authority_api()

    with pytest.raises(MdpPlanError):
        api.build_d3_iteration_authority(
            _item_authority(api),
            max_seqlen_per_rank=8,
            minimum_cp_size=1,
            solver=solver,
            bridge_width=16,
            bridge_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("bridge_width, bridge_dtype", ((0, torch.bfloat16), (16, object())))
def test_rejects_width_or_dtype_after_planning(bridge_width, bridge_dtype):
    api = _authority_api()
    solver = _FullGroupSolver()

    with pytest.raises(MdpConfigurationError):
        api.build_d3_iteration_authority(
            _item_authority(api),
            max_seqlen_per_rank=8,
            minimum_cp_size=1,
            solver=solver,
            bridge_width=bridge_width,
            bridge_dtype=bridge_dtype,
        )
    assert solver.calls == [(((0, 4), (1, 4)), 2, 8, 1)]


def test_calls_current_builders_in_plan_payload_bridge_order(monkeypatch):
    api = _authority_api()
    item_authority = _item_authority(api)
    calls = []
    planner = api.build_decoder_dynamic_plan
    payload_builder = api.build_decoder_payload_route_ledger
    bridge_builder = api.build_dynamic_bridge_ledgers

    def plan(*args, **kwargs):
        calls.append(("plan", args, kwargs))
        return planner(*args, **kwargs)

    def payload(*args, **kwargs):
        calls.append(("payload", args, kwargs))
        return payload_builder(*args, **kwargs)

    def bridge(*args, **kwargs):
        calls.append(("bridge", args, kwargs))
        return bridge_builder(*args, **kwargs)

    monkeypatch.setattr(api, "build_decoder_dynamic_plan", plan)
    monkeypatch.setattr(api, "build_decoder_payload_route_ledger", payload)
    monkeypatch.setattr(api, "build_dynamic_bridge_ledgers", bridge)
    result = api.build_d3_iteration_authority(
        item_authority,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )

    assert [name for name, _, _ in calls] == ["plan", "payload", "bridge"]
    _, plan_args, plan_kwargs = calls[0]
    assert plan_args == (item_authority.global_manifest.samples,)
    assert plan_kwargs["decoder_ranks"] == (5, 7)
    _, payload_args, payload_kwargs = calls[1]
    assert payload_args == (result.plan,)
    assert payload_kwargs["global_manifest"] is result.global_manifest
    assert payload_kwargs["participant_ranks"] == (3, 5, 7)
    assert dict(payload_kwargs["source_rank_by_lane"]) == dict(result.source_rank_by_lane)
    _, bridge_args, bridge_kwargs = calls[2]
    assert bridge_args == (result.plan,)
    assert bridge_kwargs["global_manifest"] is result.global_manifest
    assert bridge_kwargs["participant_ranks"] == (3, 5, 7)
    assert bridge_kwargs["width"] == 16
    assert bridge_kwargs["dtype"] is torch.bfloat16
    assert dict(bridge_kwargs["producer_rank_by_item"]) == dict(result.producer_rank_by_item)
    assert dict(bridge_kwargs["output_rows_by_item"]) == dict(result.output_rows_by_item)


def test_rejects_foreign_plan_before_ledger_construction(monkeypatch):
    api = _authority_api()
    item_authority = _item_authority(api)
    foreign_plan = build_decoder_dynamic_plan(
        (item_authority.global_manifest.samples[0],),
        decoder_ranks=(5, 7),
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
    )
    bridge_called = False

    def bridge(*args, **kwargs):
        nonlocal bridge_called
        bridge_called = True
        raise AssertionError("foreign plan must reject before bridge construction")

    monkeypatch.setattr(api, "build_decoder_dynamic_plan", lambda *args, **kwargs: foreign_plan)
    monkeypatch.setattr(api, "build_dynamic_bridge_ledgers", bridge)
    with pytest.raises(MdpPlanError):
        api.build_d3_iteration_authority(
            item_authority,
            max_seqlen_per_rank=8,
            minimum_cp_size=1,
            solver=_FullGroupSolver(),
            bridge_width=16,
            bridge_dtype=torch.bfloat16,
        )
    assert not bridge_called


def test_final_authority_boundary_rejects_malformed_builder_outputs(monkeypatch):
    api = _authority_api()
    item_authority = _item_authority(api)
    payload_builder = api.build_decoder_payload_route_ledger
    bridge_builder = api.build_dynamic_bridge_ledgers

    monkeypatch.setattr(api, "build_decoder_payload_route_ledger", lambda *args, **kwargs: object())
    with pytest.raises(MdpConfigurationError):
        api.build_d3_iteration_authority(
            item_authority,
            max_seqlen_per_rank=8,
            minimum_cp_size=1,
            solver=_FullGroupSolver(),
            bridge_width=16,
            bridge_dtype=torch.bfloat16,
        )
    monkeypatch.setattr(api, "build_decoder_payload_route_ledger", payload_builder)
    monkeypatch.setattr(
        api,
        "build_dynamic_bridge_ledgers",
        lambda *args, **kwargs: tuple(reversed(bridge_builder(*args, **kwargs))),
    )
    with pytest.raises(MdpConfigurationError):
        api.build_d3_iteration_authority(
            item_authority,
            max_seqlen_per_rank=8,
            minimum_cp_size=1,
            solver=_FullGroupSolver(),
            bridge_width=16,
            bridge_dtype=torch.bfloat16,
        )
