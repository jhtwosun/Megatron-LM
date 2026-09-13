# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual NCCL runtime gates for repeated-D4 selected encoder publication.

The encoder is intentionally a tiny CUDA Linear module.  These tests cover the
MDP protocol, ownership, geometry, and collective order; full-model numerical
parity remains a separate model integration gate.
"""

import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
from megatron.core.mdp import dynamic_cp_d4_encoder_capture as capture_api
from megatron.core.mdp import dynamic_cp_d4_encoder_execution as execution_api
from megatron.core.mdp import dynamic_cp_d4_encoder_forward as forward_api
from megatron.core.mdp.dynamic_cp import DynamicCpGroupMembership
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_d4_authority_construction import (
    build_repeated_d4_joint_iteration_authority,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
from megatron.core.mdp.dynamic_cp_execution import build_decoder_global_manifest
from megatron.core.mdp.dynamic_cp_plan import EncoderWorkEstimate
from megatron.core.mdp.errors import MdpPlanError, MdpStateError
from megatron.core.mdp.groups import MdpProcessGroups
from megatron.core.mdp.protocols import DynamicEncoderCpBinding, VisionCaptureMode
from megatron.core.mdp.runtime import MdpRuntimeState
from megatron.core.mdp.vision_locator import build_vision_locator_catalog
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_DISTRIBUTED = _WORLD_SIZE in (4, 8)
_REPEATED_D4 = _WORLD_SIZE == 8


class _Solver:
    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size):
        del max_seq_len_per_rank, min_cp_size
        sample_ids = [sample_id for sample_id, _ in sample_seqlens]
        lengths = [length for _, length in sample_seqlens]
        return ([lengths] * total_gpus, [], None, [sample_ids] * total_gpus)


class _Allocator:
    def __init__(self):
        self.live = []

    def acquire(self, *, rows, width, dtype, device, tag):
        del tag
        value = torch.empty((rows,) if width == 0 else (rows, width), dtype=dtype, device=device)
        self.live.append(value)
        return value

    def release(self, value):
        for index, candidate in enumerate(self.live):
            if candidate is value:
                del self.live[index]
                return
        raise AssertionError("released an unknown allocation")


class _EncoderDdp:
    def __init__(self, module):
        self.module = module

    def zero_grad_buffer(self):
        self.module.zero_grad(set_to_none=True)


class _Adapter:
    payload_width = 4

    def __init__(self, ddp, *, invalid=False):
        self.ddp = ddp
        self.invalid = invalid
        self.encode_calls = 0
        self.restore_calls = 0

    def bind_dynamic_encoder_cp(self, encoder, *, membership, global_rank):
        assert encoder is self.ddp.module
        assert global_rank in membership.ranks

        def restore(_primary):
            self.restore_calls += 1

        return DynamicEncoderCpBinding(membership, is_current=lambda: True, restore=restore)

    def encode(self, encoder, pixels, layout):
        assert encoder is self.ddp
        self.encode_calls += 1
        if self.invalid:
            return object()
        rows = sum(segment.output_rows for segment in layout.segments)
        pooled = pixels.mean(dim=0, keepdim=True).expand(rows, -1)
        return encoder.module(pooled).contiguous()


def _source_window(lane, *, geometry, device):
    grid, output_rows = ((1, 2, 2), 1) if geometry == "qwen" else ((1, 2, 2), 5)
    boundaries = torch.tensor((0, 8), dtype=torch.int32, device=device)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries.clone(),
        cu_seqlens_q_padded=boundaries.clone(),
        cu_seqlens_kv_padded=boundaries.clone(),
        max_seqlen_q=8,
        max_seqlen_kv=8,
        total_tokens=8,
    )
    payload = MappingProxyType(
        {
            "input_ids": torch.arange(8, dtype=torch.int64, device=device).view(1, 8),
            "labels": torch.arange(8, dtype=torch.int64, device=device).view(1, 8),
            "loss_mask": torch.ones((1, 8), device=device),
            "padding_mask": torch.zeros((1, 8), dtype=torch.bool, device=device),
            "position_ids": torch.arange(8, dtype=torch.int64, device=device).view(1, 8),
            "attention_mask": None,
            "image_grid_thw": torch.tensor((grid,), dtype=torch.int64, device=device),
        }
    )
    record = MdpMicrobatchRecord(
        microbatch_id=3,
        text_only=False,
        vision_items=(
            MdpMicrobatchVisionRecord(
                global_item_id=0,
                sample_id=0,
                image_ordinal=0,
                grid_thw=grid,
                output_rows=output_rows,
                decoder_positions=tuple(range(output_rows)),
            ),
        ),
        decoder_packed_seq_params=packed,
        model_payload=payload,
    )
    return MultimodalDecoderPayloadCodec().build_source_window_with_locations(
        (record,), source_dp_lane=lane
    )


@pytest.fixture(scope="module")
def groups():
    if not _DISTRIBUTED:
        yield None
        return
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    starts = tuple(range(0, _WORLD_SIZE, 4))
    domains = tuple(dist.new_group(tuple(range(start, start + 4))) for start in starts)
    pairs = tuple(dist.new_group((start, start + 1)) for start in range(0, _WORLD_SIZE, 2))
    singles = tuple(dist.new_group((rank,)) for rank in range(_WORLD_SIZE))
    yield domains, pairs, singles
    for group in (*singles, *pairs, *domains):
        dist.destroy_process_group(group)
    dist.destroy_process_group()


def _parts(groups, *, selected_size, geometry, invalid=False, runtime=None):
    domains, pairs, singles = groups
    rank = dist.get_rank()
    lane = rank // 4
    domain_ranks = tuple(range(lane * 4, lane * 4 + 4))
    domain_group = domains[lane]
    binding = _make_repeated_d4_group_binding(
        world_group=dist.group.WORLD,
        domain_group=domain_group,
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=torch.device("cuda", torch.cuda.current_device()),
        timeout_seconds=30.0,
    )
    events = []
    authority_slot = binding._authority
    world_gate = authority_slot._world_pre_gate
    domain_status = authority_slot._domain_status

    def tracked_world(**kwargs):
        events.append((kwargs["gate_id"], "world"))
        return world_gate(**kwargs)

    def tracked_domain(**kwargs):
        events.append((kwargs["gate_id"], "domain"))
        return domain_status(**kwargs)

    object.__setattr__(authority_slot, "_world_pre_gate", tracked_world)
    object.__setattr__(authority_slot, "_domain_status", tracked_domain)
    device = torch.device("cuda", torch.cuda.current_device())
    source_window, locations = _source_window(lane, geometry=geometry, device=device)
    metadata = DecoderMetadataGatherResult(
        global_manifest=build_decoder_global_manifest((source_window.metadata_manifest(),)),
        source_rank_by_lane={lane: domain_ranks[0]},
    )

    def workload(_items, *, group_size):
        return EncoderWorkEstimate(9 if group_size < selected_size else 1, 1)

    authority = build_repeated_d4_joint_iteration_authority(
        binding,
        metadata,
        decoder_max_seqlen_per_rank=8,
        decoder_minimum_cp_size=1,
        decoder_solver=_Solver(),
        encoder_max_seqlen_per_rank=8,
        encoder_minimum_cp_size=1,
        encoder_workload_query=workload,
        bridge_width=2,
        bridge_dtype=torch.float32,
    )
    selected = domain_ranks[:selected_size]
    is_leader = rank == selected[0]
    pixels = (
        {0: torch.arange(16, dtype=torch.float32, device=device).view(4, 4)} if is_leader else {}
    )
    if runtime is None:
        module = torch.nn.Linear(4, 2, bias=False, device=device)
        dist.broadcast(module.weight, src=domain_ranks[0], group=domain_group)
        ddp = _EncoderDdp(module)
        runtime = object.__new__(capture_api.MdpRuntime)
        runtime._state = MdpRuntimeState.EMPTY
        runtime._window = None
        runtime._plan = None
        runtime.num_vpp_chunks = 1
        runtime.device = device
        runtime.params_dtype = torch.float32
        runtime._iteration = 0
        runtime.rank_view = SimpleNamespace(
            global_rank=rank,
            outer_dp_rank=lane,
            lane_id=lane if is_leader else None,
            my_worker_id=0,
            endpoint_rank=domain_ranks[0],
            planning_group_ranks=domain_ranks,
            worker_ids=(0,),
        )
        runtime.rank_map = SimpleNamespace(
            spec=SimpleNamespace(world_size=8, tp=1, pp=1, cp=4, encoder_cp=4)
        )
        runtime._d4_encoder_capture_owner = None
        runtime._d4_encoder_capture_trusted_owner = None
        runtime._retired_d4_encoder_capture_owners = {}
        runtime.allocator = _Allocator()
        runtime.encoder_domain = SimpleNamespace(encoder_ddp=ddp)
        runtime.process_groups = MdpProcessGroups(
            planning_group=domain_group,
            encoder_cp_group=domain_group,
            encoder_cp_group_ranks=domain_ranks,
            encoder_cp_leader_rank=domain_ranks[0],
            singleton_group=singles[rank],
            encoder_reduction_group=dist.group.WORLD,
            world_group=dist.group.WORLD,
            encoder_cp_groups=(
                DynamicCpGroupMembership(1, (rank,), singles[rank]),
                DynamicCpGroupMembership(2, (rank // 2 * 2, rank // 2 * 2 + 1), pairs[rank // 2]),
                DynamicCpGroupMembership(4, domain_ranks, domain_group),
            ),
        )
    else:
        ddp = runtime.encoder_domain.encoder_ddp
    adapter = _Adapter(ddp, invalid=invalid and rank == 1)
    runtime.adapter = adapter
    token = object()
    capture_api._PENDING_OWNER_SEALS[token] = (
        id(runtime),
        id(binding),
        VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
    )
    capture_owner = capture_api._D4EncoderCaptureOwner(
        runtime, binding, VisionCaptureMode.SOURCE_PIXEL_SIDECAR, _factory_seal=token
    )
    runtime._register_d4_encoder_capture_owner(capture_owner)
    if is_leader:
        capture_owner._install_source(
            source_window=source_window,
            local_manifest=source_window.metadata_manifest(),
            sample_locations=locations,
            pixels=pixels,
            locator_catalog=build_vision_locator_catalog((), ()),
        )
    claim = execution_api.claim_d4_encoder_execution(capture_owner, authority)
    execution = authority.encoder_plan.waves[0].executions[0]
    assert execution.group_size == selected_size
    assert execution.rank_slots == tuple(range(selected_size))
    return runtime, authority, claim, events


def _run(runtime, authority, claim, events):
    nonce_calls = []
    a2a_calls = []

    def nonce(size):
        nonce_calls.append(size)
        return bytes((len(nonce_calls),)) * size

    def a2a(*args, **kwargs):
        a2a_calls.append("a2a")
        return dist.all_to_all_single(*args, **kwargs)

    forward = forward_api.run_repeated_d4_encoder_forward(
        runtime,
        claim,
        authority,
        all_to_all_single=a2a,
        broadcast=dist.broadcast,
        byte_generator=nonce,
    )
    before_gate1 = len(a2a_calls)
    publication = forward_api.run_repeated_d4_encoder_publication(
        forward, all_to_all_single=a2a, byte_generator=nonce
    )
    return publication, nonce_calls, a2a_calls, before_gate1, events


@pytest.mark.skipif(not _REPEATED_D4, reason="repeated D4 requires torchrun world8")
@pytest.mark.parametrize("selected_size", (1, 2, 4))
@pytest.mark.parametrize("geometry", ("qwen", "nemotron"))
def test_real_gate0_gate1_roles_geometry_and_order(groups, selected_size, geometry):
    runtime, authority, claim, events = _parts(
        groups, selected_size=selected_size, geometry=geometry
    )
    publication, nonce_calls, a2a_calls, before_gate1, events = _run(
        runtime, authority, claim, events
    )
    rank = dist.get_rank()
    selected = publication.selected_ranks
    assert events == [
        (0, "world"),
        (0, "domain"),
        (0, "world"),
        (1, "world"),
        (1, "domain"),
        (1, "world"),
    ]
    nonce_counts = [None] * _WORLD_SIZE
    dist.all_gather_object(nonce_counts, tuple(nonce_calls))
    assert nonce_counts == [(16, 16)] * _WORLD_SIZE
    assert len(a2a_calls) > before_gate1
    assert runtime.adapter.encode_calls == (1 if rank in selected else 0)
    assert (publication.forward_handle is not None) == (rank in selected)
    assert bool(publication.item_outputs) == (rank == selected[0])
    rows = authority.global_manifest.items[0].output_rows
    assert rows == (1 if geometry == "qwen" else 5)
    publication.abort()
    assert runtime.allocator.live == []
    assert runtime.adapter.restore_calls == (1 if rank in selected else 0)


@pytest.mark.skipif(not _REPEATED_D4, reason="repeated D4 requires torchrun world8")
def test_one_rank_post_forward_error_converges_before_embedding_then_fresh_retry(groups):
    runtime, authority, claim, events = _parts(
        groups, selected_size=4, geometry="qwen", invalid=True
    )
    a2a_calls = []

    def a2a(*args, **kwargs):
        a2a_calls.append("a2a")
        return dist.all_to_all_single(*args, **kwargs)

    forward = forward_api.run_repeated_d4_encoder_forward(
        runtime, claim, authority, all_to_all_single=a2a, broadcast=dist.broadcast
    )
    retained = forward.local_forward_error
    before_gate1 = len(a2a_calls)
    with pytest.raises(MdpPlanError) as raised:
        forward_api.run_repeated_d4_encoder_publication(forward, all_to_all_single=a2a)
    assert str(raised.value) == "MDP: repeated-D4 WORLD rejected rank 1 with error code 1."
    if dist.get_rank() == 1:
        assert raised.value.__cause__ is retained
        assert type(retained) is MdpStateError
    else:
        assert retained is None
        assert raised.value.__cause__ is None
    observations = [None] * _WORLD_SIZE
    dist.all_gather_object(
        observations,
        (
            type(raised.value).__name__,
            str(raised.value),
            retained is not None and raised.value.__cause__ is retained,
        ),
    )
    assert observations == [
        ("MdpPlanError", "MDP: repeated-D4 WORLD rejected rank 1 with error code 1.", rank == 1)
        for rank in range(_WORLD_SIZE)
    ]
    assert len(a2a_calls) == before_gate1
    assert events == [(0, "world"), (0, "domain"), (0, "world"), (1, "world")]
    assert runtime.allocator.live == []
    assert runtime.adapter.restore_calls == 1

    fresh_runtime, fresh_authority, fresh_claim, fresh_events = _parts(
        groups, selected_size=4, geometry="qwen", invalid=False, runtime=runtime
    )
    assert fresh_runtime is runtime
    publication, _, _, _, _ = _run(fresh_runtime, fresh_authority, fresh_claim, fresh_events)
    publication.abort()
    assert fresh_runtime.allocator.live == []


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4 or world8")
@pytest.mark.parametrize("selected_size", (1, 2, 4))
def test_actual_world4_or_world8_e1_e2_e4_prefix_memberships(groups, selected_size):
    domains, pairs, singles = groups
    rank = dist.get_rank()
    domain_start = rank // 4 * 4
    selected = tuple(range(domain_start, domain_start + selected_size))
    if rank in selected:
        group = (
            singles[rank]
            if selected_size == 1
            else pairs[domain_start // 2] if selected_size == 2 else domains[domain_start // 4]
        )
        membership = DynamicCpGroupMembership(selected_size, selected, group)
        assert membership.ranks == selected
        assert dist.get_world_size(membership.group) == selected_size
    dist.barrier()
