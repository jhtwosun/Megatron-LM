# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 decoder-ready artifact materialization contracts."""

from dataclasses import replace
from importlib import import_module
from types import MappingProxyType

import pytest
import torch

from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderMicrobatchKey,
    LocalDecoderAssignment,
    build_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_routing import DecoderPayloadRouteKey
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.mdp.test_dynamic_cp_d3_authority_construction import _authority_api
from tests.unit_tests.mdp.test_dynamic_cp_d3_local_placement import _prepared, _producer, _workspace
from tests.unit_tests.mdp.test_multimodal_mdp_adapter import _records


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_ready_artifacts")


def _placement(*, participant_ranks=(3, 5, 7)):
    placement_api = import_module("megatron.core.mdp.dynamic_cp_d3_local_placement")
    workspace = _workspace(participant_ranks=participant_ranks)
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)
    return workspace, placement_api._place_d3_local_decoder_inputs(
        workspace=workspace, producer=producer, payload_bundle=payload, embedding_exchange=embedding
    )


class _TwoWaveSolver:
    """Put the two vision samples before the text-only sample."""

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size):
        del max_seq_len_per_rank, min_cp_size
        if len(sample_seqlens) == 3:
            return ([[4, 6]] * total_gpus, [sample_seqlens[2]], None, [[0, 1]] * total_gpus)
        return ([[3]] * total_gpus, [], None, [[2]] * total_gpus)


def _codec_placement():
    codec = MultimodalDecoderPayloadCodec()
    records = []
    for record in _records():
        payload = {
            name: (
                value.to(device="cuda")
                if isinstance(value, torch.Tensor) and name != "image_grid_thw"
                else value
            )
            for name, value in record.model_payload.items()
        }
        records.append(replace(record, model_payload=MappingProxyType(payload)))
    source = codec.build_source_window(tuple(records), source_dp_lane=0)
    metadata = DecoderMetadataGatherResult(
        global_manifest=build_decoder_global_manifest((source.metadata_manifest(),)),
        source_rank_by_lane={0: 3},
    )
    authority_api = _authority_api()
    item_authority = authority_api.derive_decoder_item_authority(
        metadata, participant_ranks=(3, 5, 7), decoder_ranks=(5, 7)
    )
    authority = authority_api.build_d3_iteration_authority(
        item_authority,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_TwoWaveSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    workspace_api = import_module("megatron.core.mdp.dynamic_cp_d3_workspace")
    allocator = DirectBufferAllocator()
    workspace = workspace_api._DynamicIterationWorkspace(
        authority=authority,
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)
    placement_api = import_module("megatron.core.mdp.dynamic_cp_d3_local_placement")
    return (
        workspace,
        placement_api._place_d3_local_decoder_inputs(
            workspace=workspace,
            producer=producer,
            payload_bundle=payload,
            embedding_exchange=embedding,
        ),
        codec,
    )


class _Group:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _assignments(placement):
    assignments = []
    for microbatch in placement.workspace.authority.plan.microbatches:
        candidates = tuple(
            assignment
            for assignment in microbatch.assignments
            if placement.workspace.rank in assignment.endpoint_ranks
        )
        assert len(candidates) == 1
        assignment = candidates[0]
        assignments.append(
            LocalDecoderAssignment(
                key=DecoderMicrobatchKey(microbatch.microbatch_index),
                assignment=assignment,
                cp_group=_Group(assignment.local_cp_size),
            )
        )
    return tuple(assignments)


def _rebuild(calls):
    def rebuild(global_manifest, assignment, *, packets, key, cp_group, cp_partition_mode):
        calls.append((global_manifest, assignment, packets, key, cp_group, cp_partition_mode))
        samples = {sample.sample_id: sample for sample in global_manifest.samples}
        items = {item.item_id: item for item in global_manifest.items}
        vision_items = []
        padded_start = 0
        for local_sample_id, sample_id in enumerate(assignment.sample_ids):
            sample = samples[sample_id]
            for encoder_item in sample.vision_items:
                item = items[encoder_item.item_id]
                vision_items.append(
                    MdpMicrobatchVisionRecord(
                        global_item_id=item.item_id,
                        sample_id=local_sample_id,
                        image_ordinal=item.image_ordinal,
                        grid_thw=item.grid_thw,
                        output_rows=item.output_rows,
                        decoder_positions=tuple(
                            padded_start + offset for offset in item.decoder_offsets
                        ),
                    )
                )
            padded_start += sample.padded_seqlen
        payload = MappingProxyType({"packets": packets})
        offsets = [0]
        for sample_id in assignment.sample_ids:
            offsets.append(offsets[-1] + samples[sample_id].padded_seqlen)
        cu = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=packets[0].tensor_fields[next(iter(packets[0].tensor_fields))].device,
        )
        return MdpMicrobatchRecord(
            microbatch_id=key.microbatch_index,
            text_only=not vision_items,
            vision_items=tuple(vision_items),
            decoder_packed_seq_params=PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu.clone(),
                cu_seqlens_q_padded=cu.clone(),
                cu_seqlens_kv_padded=cu.clone(),
                max_seqlen_q=max(sample.padded_seqlen for sample in samples.values()),
                max_seqlen_kv=max(sample.padded_seqlen for sample in samples.values()),
                local_cp_size=assignment.local_cp_size,
                cp_group=cp_group,
                total_tokens=offsets[-1],
                cp_partition_mode=cp_partition_mode,
            ),
            model_payload=payload,
        )

    return rebuild


def test_materializes_manifest_packets_as_exact_payload_views_and_codec_records():
    workspace, placement = _placement(participant_ranks=(7, 3, 5))
    assignments = _assignments(placement)
    calls = []

    try:
        artifacts = _api()._materialize_d3_decoder_ready_artifacts(
            placement=placement,
            assignments=assignments,
            cp_partition_mode="contiguous",
            rebuild_microbatch=_rebuild(calls),
        )

        runtime = import_module("megatron.core.mdp.dynamic_cp_runtime")
        assert type(artifacts) is runtime._LocalDecoderReadyArtifacts
        assert artifacts.embedding_leaves is placement.embedding_leaves
        assert tuple(artifacts.embedding_leaves) == tuple(placement.embedding_leaves)
        assert len(calls) == len(assignments) == len(artifacts.records)
        payload_by_sample = {
            payload.sample_id: payload
            for payload in placement.workspace.authority.global_manifest.payloads
        }
        for assignment, record, call in zip(assignments, artifacts.records, calls):
            manifest, callback_assignment, packets, key, group, mode = call
            assert manifest is placement.workspace.authority.global_manifest
            assert callback_assignment is assignment.assignment
            expected_key = next(iter(placement.embedding_leaves))
            assert assignment.key is not expected_key
            assert key is expected_key
            assert group is assignment.cp_group
            assert mode == "contiguous"
            assert record.microbatch_id == assignment.key.microbatch_index
            assert tuple(packet.sample_id for packet in packets) == assignment.assignment.sample_ids
            for packet in packets:
                payload = payload_by_sample[packet.sample_id]
                assert packet.metadata() is not payload
                assert packet.metadata() == payload
                for spec in packet.field_specs:
                    route_key = DecoderPayloadRouteKey(
                        packet.sample_id, placement.workspace.rank, spec.name
                    )
                    assert (
                        packet.tensor_fields[spec.name]
                        is placement.payload_destination_views[route_key]
                    )
    finally:
        workspace.release()


def test_materializes_actual_qwen_codec_records_and_text_only_microbatch():
    workspace, placement, codec = _codec_placement()
    assignments = _assignments(placement)
    calls = []

    def rebuild(*args, **kwargs):
        calls.append((args, kwargs))
        return codec.rebuild_microbatch(*args, **kwargs)

    try:
        artifacts = _api()._materialize_d3_decoder_ready_artifacts(
            placement=placement,
            assignments=assignments,
            cp_partition_mode="contiguous",
            rebuild_microbatch=rebuild,
        )
        assert len(calls) == len(assignments) == 2
        assert tuple(record.text_only for record in artifacts.records) == (False, True)
        assert tuple(artifacts.embedding_leaves) == (assignments[0].key,)
        assert (
            calls[0][1]["packets"][0].tensor_fields["input_ids"]
            is placement.payload_destination_views[
                DecoderPayloadRouteKey(
                    calls[0][1]["packets"][0].sample_id, workspace.rank, "input_ids"
                )
            ]
        )
    finally:
        workspace.release()


@pytest.mark.parametrize("mutation", ("foreign", "missing-payload", "extra-payload"))
def test_rejects_noncanonical_assignments_or_payload_views_before_callback(mutation):
    workspace, placement = _placement()
    assignments = _assignments(placement)
    calls = []
    if mutation == "foreign":
        assignments = (
            replace(
                assignments[0], key=DecoderMicrobatchKey(assignments[0].key.microbatch_index + 99)
            ),
        )
    else:
        views = dict(placement.payload_destination_views)
        if mutation == "missing-payload":
            views.pop(next(iter(views)))
        else:
            views[object()] = next(iter(views.values()))
        object.__setattr__(placement, "payload_destination_views", MappingProxyType(views))

    try:
        with pytest.raises((MdpBridgeError, MdpConfigurationError, MdpPlanError)):
            _api()._materialize_d3_decoder_ready_artifacts(
                placement=placement,
                assignments=assignments,
                cp_partition_mode="contiguous",
                rebuild_microbatch=_rebuild(calls),
            )
        assert calls == []
    finally:
        workspace.release()


def test_rejects_malformed_callback_record_without_losing_exact_placement_leaves():
    workspace, placement = _placement()
    assignments = _assignments(placement)

    try:
        with pytest.raises(MdpConfigurationError):
            _api()._materialize_d3_decoder_ready_artifacts(
                placement=placement,
                assignments=assignments,
                cp_partition_mode="contiguous",
                rebuild_microbatch=lambda *_args, **_kwargs: object(),
            )
        assert all(
            placement.embedding_leaves[key] is workspace.storage.get_leaf(key.microbatch_index)
            for key in placement.embedding_leaves
        )
    finally:
        workspace.release()
