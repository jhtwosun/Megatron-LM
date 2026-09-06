# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure E1/E2/E4 execution-claim and leader-layout contracts."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
from megatron.core.mdp import dynamic_cp_d4_encoder_capture as capture_api
from megatron.core.mdp import dynamic_cp_d4_encoder_execution as execution_api
from megatron.core.mdp.dynamic_cp import DynamicCpGroupMembership
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_d4_authority_construction import (
    build_repeated_d4_joint_iteration_authority,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
from megatron.core.mdp.dynamic_cp_execution import build_decoder_global_manifest
from megatron.core.mdp.dynamic_cp_plan import (
    EncoderDynamicPlan,
    EncoderExecution,
    EncoderExecutionWave,
    EncoderWorkEstimate,
    _encoder_digest,
)
from megatron.core.mdp.dynamic_cp_runtime import _joint_dynamic_plan_digest
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpStateError
from megatron.core.mdp.groups import MdpProcessGroups
from megatron.core.mdp.runtime import MdpRuntimeState
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams


class _Group:
    def __init__(self, ranks):
        self.ranks = tuple(ranks)


class _Solver:
    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size):
        del max_seq_len_per_rank, min_cp_size
        sample_ids = [sample_id for sample_id, _ in sample_seqlens]
        lengths = [length for _, length in sample_seqlens]
        return ([lengths] * total_gpus, [], None, [sample_ids] * total_gpus)


def _binding(rank):
    domain_start = rank // 4 * 4
    domain = tuple(range(domain_start, domain_start + 4))
    return _make_repeated_d4_group_binding(
        world_group=_Group(range(8)),
        domain_group=_Group(domain),
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=torch.device("cuda", 0),
        timeout_seconds=5.0,
        group_ranks_getter=lambda group: group.ranks,
        status_gather_factory=lambda **_kwargs: lambda *_args, **_kwargs: None,
    )


def _memberships(rank):
    domain_start = rank // 4 * 4
    return (
        DynamicCpGroupMembership(1, (rank,), _Group((rank,))),
        DynamicCpGroupMembership(2, (rank // 2 * 2, rank // 2 * 2 + 1), _Group((rank,))),
        DynamicCpGroupMembership(
            4,
            tuple(range(domain_start, domain_start + 4)),
            _Group(range(domain_start, domain_start + 4)),
        ),
    )


def _runtime(rank):
    domain_start = rank // 4 * 4
    domain = tuple(range(domain_start, domain_start + 4))
    runtime = object.__new__(capture_api.MdpRuntime)
    runtime._state = MdpRuntimeState.EMPTY
    runtime._window = None
    runtime._plan = None
    runtime.num_vpp_chunks = 1
    runtime.device = torch.device("cuda", 0)
    runtime.rank_view = SimpleNamespace(
        global_rank=rank,
        outer_dp_rank=rank // 4,
        lane_id=rank // 4 if rank == domain_start else None,
        my_worker_id=0,
        endpoint_rank=domain_start,
        planning_group_ranks=domain,
        worker_ids=(0,),
    )
    runtime.rank_map = SimpleNamespace(
        spec=SimpleNamespace(world_size=8, tp=1, pp=1, cp=4, encoder_cp=4)
    )
    group = _Group((rank,))
    domain = tuple(range(rank // 4 * 4, rank // 4 * 4 + 4))
    runtime.process_groups = MdpProcessGroups(
        planning_group=_Group(domain),
        encoder_cp_group=_Group(domain),
        encoder_cp_group_ranks=domain,
        encoder_cp_leader_rank=domain[0],
        singleton_group=group,
        encoder_reduction_group=_Group(range(8)),
        world_group=_Group(range(8)),
        encoder_cp_groups=_memberships(rank),
    )
    runtime._d4_encoder_capture_owner = None
    runtime._d4_encoder_capture_trusted_owner = None
    runtime._retired_d4_encoder_capture_owners = {}
    return runtime


def _authority(binding, source_window, selected_size):
    metadata = DecoderMetadataGatherResult(
        global_manifest=build_decoder_global_manifest((source_window.metadata_manifest(),)),
        source_rank_by_lane={source_window.source_dp_lane: binding.domain_ranks[0]},
    )

    def workload(_items, *, group_size):
        return EncoderWorkEstimate(9 if group_size < selected_size else 1, 1)

    return build_repeated_d4_joint_iteration_authority(
        binding,
        metadata,
        decoder_max_seqlen_per_rank=8,
        decoder_minimum_cp_size=1,
        decoder_solver=_Solver(),
        encoder_max_seqlen_per_rank=8,
        encoder_minimum_cp_size=1,
        encoder_workload_query=workload,
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )


def _source_window(lane, *, num_items=1, microbatch_ids=(3,)):
    total_tokens = max(1, num_items)
    boundaries = torch.tensor((0, total_tokens), dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries.clone(),
        cu_seqlens_q_padded=boundaries.clone(),
        cu_seqlens_kv_padded=boundaries.clone(),
        max_seqlen_q=total_tokens,
        max_seqlen_kv=total_tokens,
        total_tokens=total_tokens,
    )
    payload = MappingProxyType(
        {
            "input_ids": torch.zeros((1, total_tokens), dtype=torch.int64),
            "labels": torch.zeros((1, total_tokens), dtype=torch.int64),
            "loss_mask": torch.ones((1, total_tokens)),
            "padding_mask": torch.zeros((1, total_tokens), dtype=torch.bool),
            "position_ids": torch.zeros((1, total_tokens), dtype=torch.int64),
            "attention_mask": None,
            "image_grid_thw": torch.tensor(((1, 2, 2),) * num_items, dtype=torch.int64).reshape(
                num_items, 3
            ),
        }
    )
    source_window, locations = MultimodalDecoderPayloadCodec().build_source_window_with_locations(
        tuple(
            MdpMicrobatchRecord(
                microbatch_id=microbatch_id,
                text_only=num_items == 0,
                vision_items=tuple(
                    MdpMicrobatchVisionRecord(
                        global_item_id=item_id,
                        sample_id=0,
                        image_ordinal=item_id,
                        grid_thw=(1, 2, 2),
                        output_rows=1,
                        decoder_positions=(item_id,),
                    )
                    for item_id in range(num_items)
                ),
                decoder_packed_seq_params=packed,
                model_payload=payload,
            )
            for microbatch_id in microbatch_ids
        ),
        source_dp_lane=lane,
    )
    return source_window, source_window.metadata_manifest(), locations


def _owner(monkeypatch, rank, source_window, locations, *, pixels=None):
    runtime = _runtime(rank)
    binding = _binding(rank)
    pixels = {0: torch.ones(4, 4)} if pixels is None else pixels
    window = SimpleNamespace(
        records=lambda: (object(),),
        payload_sidecar=lambda: dict(pixels),
        release_pixels=lambda: pixels.clear(),
    )
    monkeypatch.setattr(capture_api.MdpIterationWindow, "capture", lambda *_a, **_k: window)

    class _Adapter:
        spatial_merge_size = 2

        @staticmethod
        def get_batch(_iterator):
            return object()

        @staticmethod
        def estimate_cost(_item):
            return 1

    class _Codec:
        @staticmethod
        def build_source_window_with_locations(_records, *, source_dp_lane):
            assert source_dp_lane == source_window.source_dp_lane
            return source_window, locations

    operations = capture_api._snapshot_d4_encoder_capture_operations(_Adapter(), _Codec())
    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=binding,
        data_iterators=iter((object(),)),
        num_microbatches=len(source_window.packets),
        operations=operations,
    )
    return runtime, binding, owner


@pytest.mark.parametrize("selected_size", (1, 2, 4))
@pytest.mark.parametrize("rank", range(8))
def test_claim_derives_prefix_membership_and_source_only_exact_layout(
    monkeypatch, rank, selected_size
):
    lane = rank // 4
    source_window, _, locations = _source_window(lane)
    locations = MappingProxyType({next(iter(locations)): (3, 0)})
    runtime, binding, owner = _owner(monkeypatch, rank, source_window, locations)
    authority = _authority(binding, source_window, selected_size)

    claim = execution_api.claim_d4_encoder_execution(owner, authority)

    assert claim.require() is claim
    domain_start = lane * 4
    assert claim.selected_ranks == tuple(range(domain_start, domain_start + selected_size))
    assert claim.is_selected is (rank < domain_start + selected_size)
    assert claim.is_leader is (rank == domain_start)
    assert claim.text_only is False
    assert runtime._d4_encoder_capture_owner is None
    with pytest.raises(MdpStateError, match="retired"):
        owner.require()
    if claim.is_selected:
        assert (
            claim.membership
            is runtime.process_groups.encoder_cp_groups[{1: 0, 2: 1, 4: 2}[selected_size]]
        )
    else:
        assert claim.membership is None
    if claim.is_leader:
        assert claim.layout.producer_worker_id == 0
        assert tuple(segment.global_item_id for segment in claim.layout.segments) == (
            authority.global_manifest.items[0].item_id.local_item_id,
        )
        segment = claim.layout.segments[0]
        assert type(segment.global_item_id) is int
        assert type(segment.microbatch_id) is int
        assert type(segment.sample_id) is int
        assert (segment.microbatch_id, segment.sample_id) == (3, 0)
        assert (segment.payload_row_start, segment.payload_rows) == (0, 4)
        assert (segment.output_row_start, segment.output_rows) == (0, 1)
    else:
        assert claim.layout is None
    claim.abort()
    with pytest.raises(MdpStateError, match="retired"):
        claim.require()


def _text_only_source_window(lane):
    source_window, _, locations = _source_window(lane, num_items=0)
    return source_window, locations


def _two_item_source_window():
    source_window, _, locations = _source_window(0, num_items=2)
    return source_window, locations


@pytest.mark.parametrize("rank", range(8))
def test_text_only_claim_is_typed_empty_and_consumes_capture(monkeypatch, rank):
    source_window, locations = _text_only_source_window(rank // 4)
    runtime, binding, owner = _owner(monkeypatch, rank, source_window, locations, pixels={})
    authority = _authority(binding, source_window, 4)

    claim = execution_api.claim_d4_encoder_execution(owner, authority)

    assert claim.require() is claim
    assert claim.text_only is True
    assert claim.selected_ranks == ()
    assert claim.membership is None
    assert claim.layout is None
    assert claim.is_selected is False
    assert claim.is_leader is False
    assert runtime._d4_encoder_capture_owner is None
    with pytest.raises(MdpStateError, match="retired"):
        owner.require()
    claim.abort()


def test_source_layout_preserves_manifest_order_and_cumulative_rows(monkeypatch):
    source_window, locations = _two_item_source_window()
    _, binding, owner = _owner(
        monkeypatch, 0, source_window, locations, pixels={0: torch.ones(4, 4), 1: torch.ones(4, 4)}
    )
    authority = _authority(binding, source_window, 2)

    claim = execution_api.claim_d4_encoder_execution(owner, authority)

    assert tuple(segment.global_item_id for segment in claim.layout.segments) == (
        source_window.items[0].item_id.local_item_id,
        source_window.items[1].item_id.local_item_id,
    )
    assert tuple(
        (segment.payload_row_start, segment.payload_rows) for segment in claim.layout.segments
    ) == ((0, 4), (4, 4))
    assert tuple(
        (segment.output_row_start, segment.output_rows) for segment in claim.layout.segments
    ) == ((0, 1), (1, 1))
    claim.abort()


def test_valid_multiple_executions_reject_without_consuming_capture(monkeypatch):
    source_window, locations = _two_item_source_window()
    _, binding, owner = _owner(
        monkeypatch, 0, source_window, locations, pixels={0: torch.ones(4, 4), 1: torch.ones(4, 4)}
    )
    authority = _authority(binding, source_window, 1)
    original = authority.encoder_plan
    template = original.waves[0].executions[0]
    executions = tuple(
        EncoderExecution(
            group_size=1,
            group_index=index,
            rank_slots=(index,),
            item_ids=(item.item_id,),
            effective_rows_per_rank=template.effective_rows_per_rank,
            cost_units=template.cost_units,
        )
        for index, item in enumerate(source_window.items)
    )
    waves = (EncoderExecutionWave(0, executions),)
    plan = EncoderDynamicPlan(
        original.source_samples,
        original.pool_ranks,
        original.max_seqlen_per_rank,
        waves,
        _encoder_digest(
            original.source_samples, original.pool_ranks, original.max_seqlen_per_rank, waves
        ),
    )
    authority = replace(
        authority,
        encoder_plan=plan,
        joint_plan_digest=_joint_dynamic_plan_digest(authority.plan, plan),
    )

    with pytest.raises(MdpStateError, match="exactly one wave and execution"):
        execution_api.claim_d4_encoder_execution(owner, authority)

    assert owner.require() is owner
    owner.abort()


@pytest.mark.parametrize(
    ("location", "message"),
    (
        (None, "MDP: encoder execution source locations cover the manifest."),
        ((3, 1), "MDP: encoder execution source retains exact authority metadata."),
    ),
)
def test_incomplete_or_nondense_source_locations_reject_without_consuming(
    monkeypatch, location, message
):
    source_window, _, locations = _source_window(0)
    sample_id = next(iter(locations))
    invalid_locations = MappingProxyType({} if location is None else {sample_id: location})
    _, binding, owner = _owner(monkeypatch, 0, source_window, invalid_locations)
    authority = _authority(binding, source_window, 1)

    with pytest.raises(MdpStateError) as raised:
        execution_api.claim_d4_encoder_execution(owner, authority)
    assert str(raised.value) == message

    assert owner.require() is owner
    owner.abort()


def test_duplicate_source_locations_reject_without_consuming(monkeypatch):
    source_window, _, locations = _source_window(0, num_items=0, microbatch_ids=(3, 7))
    invalid_locations = MappingProxyType({sample_id: (3, 0) for sample_id in locations})
    _, binding, owner = _owner(monkeypatch, 0, source_window, invalid_locations, pixels={})
    authority = _authority(binding, source_window, 1)

    with pytest.raises(MdpStateError) as raised:
        execution_api.claim_d4_encoder_execution(owner, authority)
    assert str(raised.value) == "MDP: encoder execution source retains exact authority metadata."

    assert owner.require() is owner
    owner.abort()


@pytest.mark.parametrize(
    ("field", "error_type", "message"),
    (
        (
            "source_rank_by_lane",
            MdpBridgeError,
            "MDP: decoder payload routes match plan and manifest authority.",
        ),
        (
            "payload_ledger",
            MdpConfigurationError,
            "MDP: dynamic iteration authority payload_ledger has its exact typed carrier.",
        ),
    ),
)
def test_mutated_authority_rejects_without_consuming_capture(
    monkeypatch, field, error_type, message
):
    source_window, _, locations = _source_window(0)
    _, binding, owner = _owner(monkeypatch, 0, source_window, locations)
    authority = _authority(binding, source_window, 1)
    value = MappingProxyType({0: 1}) if field == "source_rank_by_lane" else object()
    object.__setattr__(authority, field, value)

    with pytest.raises(error_type) as raised:
        execution_api.claim_d4_encoder_execution(owner, authority)
    assert str(raised.value) == message

    assert owner.require() is owner
    owner.abort()


def test_nonprefix_execution_rejects_without_consuming_capture(monkeypatch):
    source_window, _, locations = _source_window(0)
    _, binding, owner = _owner(monkeypatch, 0, source_window, locations)
    authority = _authority(binding, source_window, 1)
    original = authority.encoder_plan
    original_execution = original.waves[0].executions[0]
    execution = EncoderExecution(
        group_size=1,
        group_index=1,
        rank_slots=(1,),
        item_ids=original_execution.item_ids,
        effective_rows_per_rank=original_execution.effective_rows_per_rank,
        cost_units=original_execution.cost_units,
    )
    waves = (EncoderExecutionWave(0, (execution,)),)
    plan = EncoderDynamicPlan(
        original.source_samples,
        original.pool_ranks,
        original.max_seqlen_per_rank,
        waves,
        _encoder_digest(
            original.source_samples, original.pool_ranks, original.max_seqlen_per_rank, waves
        ),
    )
    authority = replace(
        authority,
        encoder_plan=plan,
        joint_plan_digest=_joint_dynamic_plan_digest(authority.plan, plan),
    )

    with pytest.raises(MdpStateError) as raised:
        execution_api.claim_d4_encoder_execution(owner, authority)
    assert str(raised.value) == "MDP: encoder execution is an E1/E2/E4 manifest-order prefix."

    assert owner.require() is owner
    owner.abort()


def test_nonselected_rank_never_reads_membership_registry(monkeypatch):
    source_window, _, locations = _source_window(0)
    runtime, binding, owner = _owner(monkeypatch, 3, source_window, locations)
    authority = _authority(binding, source_window, 2)

    class _BombGroups:
        def __getattribute__(self, _name):
            raise AssertionError("nonselected membership registry was read")

    runtime.process_groups = _BombGroups()
    claim = execution_api.claim_d4_encoder_execution(owner, authority)

    assert claim.is_selected is False
    assert claim.membership is None
    assert claim.layout is None
    claim.abort()


def test_foreign_membership_rejects_without_consuming_capture(monkeypatch):
    source_window, _, locations = _source_window(0)
    runtime, binding, owner = _owner(monkeypatch, 1, source_window, locations)
    authority = _authority(binding, source_window, 2)
    object.__setattr__(
        runtime.process_groups,
        "encoder_cp_groups",
        (DynamicCpGroupMembership(2, (2, 3), _Group((2, 3))),),
    )

    with pytest.raises(MdpStateError, match="exact E1b membership"):
        execution_api.claim_d4_encoder_execution(owner, authority)

    assert owner.require() is owner
    owner.abort()


def test_claim_mutation_rejects_and_abort_releases_trusted_pixels(monkeypatch):
    source_window, _, locations = _source_window(0)
    _, binding, owner = _owner(monkeypatch, 0, source_window, locations)
    authority = _authority(binding, source_window, 1)
    claim = execution_api.claim_d4_encoder_execution(owner, authority)
    pixels = claim._pixels
    object.__setattr__(claim, "layout", None)

    with pytest.raises(MdpStateError, match="sealed fields"):
        claim.require()
    claim.abort()

    assert pixels == {}
    with pytest.raises(MdpStateError, match="retired"):
        claim.abort()
