# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 gate-2 ready-handoff composition contracts."""

from dataclasses import replace
from importlib import import_module
from types import MappingProxyType

import pytest
import torch

from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.dynamic_cp_bridge_transport import prepare_dynamic_bridge_exchange
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_execution import build_decoder_global_manifest
from megatron.core.mdp.dynamic_cp_transport import prepare_decoder_payload_bundle
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpStateError
from megatron.core.mdp.rank_mapping import MdpRankView
from megatron.core.mdp.storage import MdpEmbeddingStorage
from tests.unit_tests.mdp.test_dynamic_cp_d3_authority_construction import _authority_api
from tests.unit_tests.mdp.test_dynamic_cp_d3_local_placement import (
    _bridge_sources,
    _payload_sources,
)
from tests.unit_tests.mdp.test_multimodal_mdp_adapter import _records


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_ready_handoff")


class _TwoWaveSolver:
    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size):
        del max_seq_len_per_rank, min_cp_size
        if len(sample_seqlens) == 3:
            return ([[4, 6]] * total_gpus, [sample_seqlens[2]], None, [[0, 1]] * total_gpus)
        return ([[3]] * total_gpus, [], None, [[2]] * total_gpus)


class _Group:
    def __init__(self, ranks, rank):
        self._ranks = ranks
        self._rank = rank

    def size(self):
        return len(self._ranks)

    def rank(self):
        return self._ranks.index(self._rank)


class _ProducerRuntime:
    def __init__(self):
        self.active = None
        self.retired = set()

    def _validate_pre_authority_dynamic_producer(self, owner, producer):
        assert producer.owner is owner and producer is self.active

    def _consume_pre_authority_dynamic_producer(self, owner, producer):
        self._validate_pre_authority_dynamic_producer(owner, producer)
        self.active = None
        self.retired.add(id(producer))


class _ProducerOwner:
    def __init__(self):
        self._runtime = _ProducerRuntime()
        self.aborts = 0

    def prepare_dynamic_completion(self, gradients, *, transport_dtype=None):
        self.transport_dtype = transport_dtype
        return gradients

    def abort(self):
        self.aborts += 1


def _authority(*, participant_ranks=(7, 3, 5), decoder_ranks=(5, 7), source_rank=3):
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
        source_rank_by_lane={0: source_rank},
    )
    authority_api = _authority_api()
    item_authority = authority_api.derive_decoder_item_authority(
        metadata, participant_ranks=participant_ranks, decoder_ranks=decoder_ranks
    )
    return codec, authority_api.build_d3_iteration_authority(
        item_authority,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_TwoWaveSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )


def _registered_producer(authority, rank):
    runtime = import_module("megatron.core.mdp.dynamic_cp_runtime")
    owner = _ProducerOwner()
    empty = MappingProxyType({})
    producer = runtime._PreAuthorityDynamicProducer(
        rank_view=MdpRankView(
            global_rank=rank,
            outer_dp_rank=0,
            lane_id=None,
            my_worker_id=0,
            endpoint_rank=rank,
            planning_group_ranks=(rank,),
            worker_ids=(0,),
        ),
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs=empty,
        sample_location_by_id=empty,
        owner=owner,
        local_prepare_error=None,
        forward_only=False,
    )
    owner._runtime.active = producer
    return owner, producer


def _context(*, rank=5, participant_ranks=(7, 3, 5), decoder_ranks=(5, 7), source_rank=3):
    binding_api = import_module("megatron.core.mdp.dynamic_cp_d3_workspace_binding")
    codec, authority = _authority(
        participant_ranks=participant_ranks, decoder_ranks=decoder_ranks, source_rank=source_rank
    )
    allocator = DirectBufferAllocator()
    owner = binding_api._D3WorkspaceBindingOwner(
        rank=rank,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    producer_owner, producer = _registered_producer(authority, rank)
    bound = owner.bind(authority=authority, producer=producer)
    workspace = owner.require_workspace(authority)
    payload = prepare_decoder_payload_bundle(
        authority.payload_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        source_rank_by_lane=authority.source_rank_by_lane,
        participant_ranks=authority.participant_ranks,
        global_rank=rank,
        local_tensors=_payload_sources(authority, rank, workspace.device),
        buffers_by_dtype=workspace.payload_transport_buffers,
    )
    embedding = prepare_dynamic_bridge_exchange(
        authority.embedding_ledger,
        authority.gradient_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        width=authority.bridge_width,
        dtype=authority.bridge_dtype,
        participant_ranks=authority.participant_ranks,
        global_rank=rank,
        local_tensors=_bridge_sources(
            authority, authority.embedding_ledger, rank, workspace.device
        ),
        send_buffer=workspace.embedding_transport_buffers[0],
        receive_buffer=workspace.embedding_transport_buffers[1],
    )
    return codec, authority, owner, producer_owner, bound, payload, embedding


def _group_getter(rank):
    def get_group(*, group_size):
        return _Group((rank,) if group_size == 1 else (5, 7), rank)

    return get_group


def _group_ranks(group):
    return group._ranks


def _compose(
    context,
    *,
    authority=None,
    rebuild=None,
    payload_result=None,
    embedding_exchange=None,
    embedding_result=None,
):
    codec, current_authority, owner, _, bound, payload, embedding = context
    authority = current_authority if authority is None else authority
    embedding_exchange = embedding if embedding_exchange is None else embedding_exchange
    return _api()._compose_d3_decoder_ready_handoff(
        workspace_owner=owner,
        authority=authority,
        producer=bound,
        payload_bundle=payload,
        payload_result=payload.received_tensors if payload_result is None else payload_result,
        embedding_exchange=embedding_exchange,
        embedding_result=(
            embedding_exchange.received_tensors if embedding_result is None else embedding_result
        ),
        cp_partition_mode="contiguous",
        decoder_group_getter=_group_getter(owner.require_workspace(current_authority).rank),
        decoder_group_ranks_getter=_group_ranks,
        rebuild_microbatch=codec.rebuild_microbatch if rebuild is None else rebuild,
    )


def test_composes_qwen_vision_and_text_ready_handoff_with_one_canonical_assignment_tuple(
    monkeypatch,
):
    context = _context()
    _, authority, owner, _, bound, payload, embedding = context
    api = _api()
    materialize = api._materialize_d3_decoder_ready_artifacts
    captured = {}

    def capture_materialization(**kwargs):
        captured["assignments"] = kwargs["assignments"]
        return materialize(**kwargs)

    monkeypatch.setattr(api, "_materialize_d3_decoder_ready_artifacts", capture_materialization)
    try:
        ready = _compose(context)
        runtime = import_module("megatron.core.mdp.dynamic_cp_runtime")

        assert type(ready) is runtime.DecoderReadyIteration
        assert tuple(record.text_only for record in ready.records) == (False, True)
        assert tuple(ready.embedding_leaves) == (ready.assignments[0].key,)
        assert ready.assignments[0].key is next(iter(ready.embedding_leaves))
        assert captured["assignments"] is ready.assignments
        runtime.validate_decoder_ready_iteration(
            ready,
            global_manifest=authority.global_manifest,
            plan=authority.plan,
            payload_bundle=payload,
            payload_tensors=payload.received_tensors,
            embedding_exchange=embedding,
            embedding_tensors=embedding.received_tensors,
            expected_assignments=ready.assignments,
            authority_digest=ready.authority_digest,
            embedding_width=authority.bridge_width,
            embedding_dtype=authority.bridge_dtype,
            cp_partition_mode="contiguous",
        )
        with pytest.raises(MdpStateError, match="fresh"):
            _compose(context)
    finally:
        bound.cleanup()
        assert owner.is_idle


def test_nondecode_rank_returns_empty_without_placement_or_rebuild():
    context = _context(rank=9, participant_ranks=(7, 3, 5, 9))
    _, _, owner, _, bound, _, _ = context
    calls = []
    try:
        ready = _compose(context, rebuild=lambda *args, **kwargs: calls.append((args, kwargs)))
        assert ready.role == "non-decoder"
        assert ready.assignments == ready.records == ()
        assert dict(ready.embedding_leaves) == {}
        assert calls == []
    finally:
        bound.cleanup()
        assert owner.is_idle


def test_d4_ready_handoff_reuses_exact_prepared_carriers_inside_gate_2(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_ready_handoff")
    binding_api = import_module("megatron.core.mdp.dynamic_cp_d4_group_binding")
    context = _context(
        rank=2, participant_ranks=(0, 1, 2, 3), decoder_ranks=(0, 1, 2, 3), source_rank=0
    )
    codec, authority, owner, _, bound, payload, embedding = context
    binding = binding_api._make_repeated_d4_group_binding(
        world_group=_Group(tuple(range(8)), 2),
        domain_group=_Group((0, 1, 2, 3), 2),
        expert_group=None,
        global_rank=2,
        expert_parallel_size=1,
        device=torch.device("cuda", 0),
        timeout_seconds=5.0,
        group_ranks_getter=lambda group: group._ranks,
        status_gather_factory=lambda **_: lambda *_args, **_kwargs: None,
    )
    events = []

    class _Runner:
        def run(self, **kwargs):
            assert kwargs["gate_id"] == 2
            events.append("run")
            ready = kwargs["prepare"]()
            events.append("prepared")
            return kwargs["domain_collective"](ready)

    monkeypatch.setattr(type(binding), "begin_attempt", lambda *_args, **_kwargs: _Runner())

    def decoder_group(*, group_size):
        endpoint_ranks = {
            assignment.endpoint_ranks
            for microbatch in authority.plan.microbatches
            for assignment in microbatch.assignments
            if 2 in assignment.endpoint_ranks and len(assignment.endpoint_ranks) == group_size
        }
        assert len(endpoint_ranks) == 1
        return _Group(endpoint_ranks.pop(), 2)

    try:
        ready = api.run_repeated_d4_decoder_ready(
            binding,
            authority,
            workspace_owner=owner,
            producer=bound,
            payload_bundle=payload,
            embedding_exchange=embedding,
            cp_partition_mode="contiguous",
            decoder_group_getter=decoder_group,
            decoder_group_ranks_getter=lambda group: group._ranks,
            rebuild_microbatch=codec.rebuild_microbatch,
        )

        assert ready.payload_bundle_authority_digest == payload.bundle_authority_digest
        assert ready.embedding_route_authority_digest == embedding.route_authority_digest
        assert events == ["run", "prepared"]
    finally:
        bound.cleanup()
        assert owner.is_idle


@pytest.mark.parametrize(
    "field", ("payload", "embedding", "swapped-carrier", "foreign-authority", "released")
)
def test_rejects_invalid_handoff_inputs_before_callback(field):
    context = _context()
    _, authority, owner, _, bound, payload, embedding = context
    calls = []
    foreign = None
    try:
        if field == "payload":
            kwargs = {"payload_result": MappingProxyType(dict(payload.received_tensors))}
        elif field == "embedding":
            kwargs = {"embedding_result": MappingProxyType(dict(embedding.received_tensors))}
        elif field == "swapped-carrier":
            foreign = _context()
            kwargs = {
                "embedding_exchange": foreign[-1],
                "embedding_result": foreign[-1].received_tensors,
            }
        elif field == "foreign-authority":
            foreign = _context()
            kwargs = {"authority": foreign[1]}
        else:
            owner.require_workspace(authority).release()
            kwargs = {}
        with pytest.raises((MdpBridgeError, MdpConfigurationError, MdpStateError)):
            _compose(
                context, rebuild=lambda *args, **kwargs: calls.append((args, kwargs)), **kwargs
            )
        assert calls == []
    finally:
        bound.cleanup()
        if foreign is not None:
            foreign[4].cleanup()


def test_callback_failure_leaves_cleanup_to_owner_and_fresh_bind_is_isolated():
    context = _context()
    _, authority, owner, _, bound, _, _ = context
    try:
        with pytest.raises(MdpConfigurationError):
            _compose(context, rebuild=lambda *args, **kwargs: object())
        assert not owner.is_idle
        bound.cleanup()
        assert owner.is_idle
        owner2, producer2 = _registered_producer(authority, 5)
        retry = owner.bind(authority=authority, producer=producer2)
        try:
            assert owner.require_workspace(authority).authority is authority
        finally:
            retry.cleanup()
        assert owner2.aborts == 1
    finally:
        if not owner.is_idle:
            bound.cleanup()
