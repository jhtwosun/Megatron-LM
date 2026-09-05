# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Domain-local iteration-authority contracts for repeated D4."""

import os
from importlib import import_module
from types import MappingProxyType

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_group_binding as binding_api
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import (
    DecoderMetadataGatherResult,
    gather_decoder_source_manifests,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
    build_decoder_global_manifest,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata, EncoderVisionItemMetadata
from megatron.core.mdp.errors import MdpPlanError, MdpStateError

_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8


def _authority_api():
    return import_module("megatron.core.mdp.dynamic_cp_d4_authority_construction")


class _Group:
    def __init__(self, ranks):
        self.ranks = tuple(ranks)


class _FullGroupSolver:
    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size):
        sample_ids = [sample_id for sample_id, _ in sample_seqlens]
        lengths = [length for _, length in sample_seqlens]
        return ([lengths] * total_gpus, [], None, [sample_ids] * total_gpus)


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


def _binding(rank=2, ep=1):
    domain_start = rank // 4 * 4
    domain_ranks = tuple(range(domain_start, domain_start + 4))
    return binding_api._make_repeated_d4_group_binding(
        world_group=_Group(range(8)),
        domain_group=_Group(domain_ranks),
        expert_group=None if ep == 1 else _Group(domain_ranks),
        global_rank=rank,
        expert_parallel_size=ep,
        device=torch.device("cuda", 0),
        timeout_seconds=5.0,
        group_ranks_getter=lambda group: group.ranks,
        status_gather_factory=lambda **_: lambda *_args, **_kwargs: None,
    )


def _metadata(lane, producer_rank):
    manifest = _source_manifest(lane)
    return DecoderMetadataGatherResult(
        global_manifest=build_decoder_global_manifest((manifest,)),
        source_rank_by_lane={lane: producer_rank},
    )


@pytest.mark.parametrize(("rank", "ep"), ((2, 1), (6, 4)))
def test_builds_authority_only_within_the_bound_local_domain(rank, ep):
    api = _authority_api()
    binding = _binding(rank, ep)
    lane = rank // 4

    authority = api.build_repeated_d4_iteration_authority(
        binding,
        _metadata(lane, binding.domain_ranks[0]),
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )

    domain = set(binding.domain_ranks)
    assert authority.participant_ranks == binding.domain_ranks
    assert authority.plan.decoder_ranks == binding.domain_ranks
    assert dict(authority.source_rank_by_lane) == {lane: binding.domain_ranks[0]}
    for ledger in (authority.payload_ledger, authority.embedding_ledger, authority.gradient_ledger):
        assert ledger.participant_ranks == binding.domain_ranks
        assert all(
            entry.src_global_rank in domain and entry.dst_global_rank in domain
            for entry in ledger.entries
        )


@pytest.mark.parametrize(
    ("rank", "metadata"),
    ((2, _metadata(1, 0)), (2, _metadata(0, 1)), (6, _metadata(0, 4)), (6, _metadata(1, 5))),
)
def test_rejects_foreign_lane_or_nonleader_source_authority(rank, metadata):
    api = _authority_api()

    with pytest.raises(MdpPlanError, match="source lane and domain leader"):
        api.build_repeated_d4_iteration_authority(
            _binding(rank),
            metadata,
            max_seqlen_per_rank=8,
            minimum_cp_size=1,
            solver=_FullGroupSolver(),
            bridge_width=16,
            bridge_dtype=torch.bfloat16,
        )


def test_revalidates_group_binding_before_planning():
    api = _authority_api()
    binding = _binding()
    object.__setattr__(binding, "domain_ranks", (4, 5, 6, 7))

    with pytest.raises(MdpStateError, match="captured authority"):
        api.build_repeated_d4_iteration_authority(
            binding,
            _metadata(0, 0),
            max_seqlen_per_rank=8,
            minimum_cp_size=1,
            solver=lambda *_args: pytest.fail("planner entered after invalid binding"),
            bridge_width=16,
            bridge_dtype=torch.bfloat16,
        )


if _WORLD8:

    @pytest.fixture(scope="module")
    def groups():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl")
        domain_groups = (
            torch.distributed.new_group(ranks=(0, 1, 2, 3)),
            torch.distributed.new_group(ranks=(4, 5, 6, 7)),
        )
        rank = torch.distributed.get_rank()
        domain = domain_groups[rank // 4]
        yield torch.distributed.group.WORLD, domain
        torch.distributed.destroy_process_group(domain)
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_gathers_and_builds_two_independent_domain_authorities(groups):
    api = _authority_api()
    world, domain_group = groups
    rank = torch.distributed.get_rank()
    lane = rank // 4
    domain_ranks = tuple(range(lane * 4, lane * 4 + 4))
    device = torch.device("cuda", torch.cuda.current_device())
    binding = binding_api._make_repeated_d4_group_binding(
        world_group=world,
        domain_group=domain_group,
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=device,
        timeout_seconds=30.0,
    )
    metadata = gather_decoder_source_manifests(
        _source_manifest(lane) if rank == domain_ranks[0] else None,
        expected_source_lanes=(lane,),
        group=domain_group,
        group_ranks=domain_ranks,
        global_rank=rank,
        device=device,
        timeout_seconds=30.0,
    )

    authority = api.build_repeated_d4_iteration_authority(
        binding,
        metadata,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )

    assert authority.participant_ranks == domain_ranks
    assert dict(authority.source_rank_by_lane) == {lane: domain_ranks[0]}
    edges = tuple(
        (entry.src_global_rank, entry.dst_global_rank)
        for ledger in (authority.payload_ledger, authority.embedding_ledger)
        for entry in ledger.entries
    )
    assert edges
    assert all(src in domain_ranks and dst in domain_ranks for src, dst in edges)
