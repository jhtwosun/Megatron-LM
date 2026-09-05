# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Domain-local iteration-authority contracts for repeated D4."""

import os
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

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


def _source_window(lane, *, device="cpu"):
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
        "input_ids": (lane * 100 + torch.arange(4, dtype=torch.int64, device=device)).view(1, 4),
        "position_ids": torch.arange(4, dtype=torch.int64, device=device).view(1, 4),
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
    )


def _source_manifest(lane):
    return _source_window(lane).metadata_manifest()


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


def _iteration_authority(rank=2, ep=1):
    binding = _binding(rank, ep)
    lane = rank // 4
    authority = _authority_api().build_repeated_d4_iteration_authority(
        binding,
        _metadata(lane, binding.domain_ranks[0]),
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    return binding, authority


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


def test_authority_collective_binds_exact_digests_and_callbacks(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_authority_collective")
    binding, authority = _iteration_authority()
    events = []
    digest_authorities = []

    def plan_digest(actual):
        digest_authorities.append(actual)
        return actual.plan.digest

    monkeypatch.setattr(api, "_dynamic_iteration_plan_digest", plan_digest)

    class _Runner:
        def run(self, **kwargs):
            events.append(("run", kwargs))
            return kwargs["domain_collective"](kwargs["prepare"]())

    def begin_attempt(self, *, byte_generator=None):
        assert self is binding
        events.append(("begin", byte_generator))
        return _Runner()

    monkeypatch.setattr(type(binding), "begin_attempt", begin_attempt)
    prepare = lambda: events.append(("prepare",)) or "prepared"
    collective = lambda value: events.append(("collective", value)) or "result"

    result = api.run_repeated_d4_authority_collective(
        binding,
        authority,
        gate_id=2,
        prepare=prepare,
        domain_collective=collective,
        byte_generator=bytes,
    )

    assert result == "result"
    assert events[0] == ("begin", bytes)
    assert events[1][0] == "run"
    call = events[1][1]
    assert call["global_manifest_digest"] == authority.global_manifest.digest
    assert call["plan_digest"] == authority.plan.digest
    assert digest_authorities == [authority]
    assert call["gate_id"] == 2
    assert call["prepare"] is not prepare and callable(call["prepare"])
    assert call["domain_collective"] is collective
    assert events[2:] == [("prepare",), ("collective", "prepared")]


def test_malformed_plan_digest_helper_still_enters_gate0_world(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_authority_collective")
    order = import_module("megatron.core.mdp.dynamic_cp_d4_collective_order")
    binding, authority = _iteration_authority()
    world_calls = []
    runner = order._make_repeated_d4_collective_runner(
        attempt_nonce=b"n" * 16,
        world_pre_gate=lambda **kwargs: world_calls.append(kwargs),
        domain_status_collector=lambda **_kwargs: pytest.fail("domain ran after malformed Gate 0"),
    )
    monkeypatch.setattr(type(binding), "begin_attempt", lambda *_args, **_kwargs: runner)
    monkeypatch.setattr(
        api,
        "_dynamic_iteration_plan_digest",
        lambda _authority: (_ for _ in ()).throw(KeyboardInterrupt("bad plan digest")),
    )

    with pytest.raises(MdpStateError, match="accepted a local error"):
        api.run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=0,
            prepare=lambda: pytest.fail("prepare ran after malformed plan digest"),
            domain_collective=lambda _value: pytest.fail("collective ran"),
        )

    assert len(world_calls) == 1
    assert world_calls[0]["gate_id"] == 0
    assert isinstance(world_calls[0]["local_error"], MdpPlanError)


def test_nested_d3_authority_status_uses_exact_legacy_iteration_plan_digest(monkeypatch):
    composition = import_module("megatron.core.mdp.dynamic_cp_d3_composition")
    group = SimpleNamespace(ranks=(0, 1), size=lambda: 2, rank=lambda: 0)
    runtime = SimpleNamespace(
        rank_map=SimpleNamespace(spec=SimpleNamespace(tp=1, ep=1, pp=1, cp=1, encoder_cp=1)),
        num_vpp_chunks=1,
        config=SimpleNamespace(dynamic_encoder_cp=False, overlap_window_capture=False),
        device=torch.device("cuda", 0),
        params_dtype=torch.bfloat16,
        hidden_size=8,
        allocator=object(),
        storage=object(),
        _prepare_dynamic_encoder_producer=lambda *_args, **_kwargs: None,
    )
    codec = SimpleNamespace(rebuild_microbatch=lambda *_args, **_kwargs: None)
    statuses = []
    digest_authorities = []

    def plan_digest(actual):
        digest_authorities.append(actual)
        return actual.plan.digest

    monkeypatch.setattr(composition, "_dynamic_iteration_plan_digest", plan_digest)
    monkeypatch.setattr(
        composition,
        "_run_precollective_consensus",
        lambda status, **_kwargs: statuses.append(status),
    )
    facade = composition._build_d3_runtime_facade(
        producer_runtime=runtime,
        codec=codec,
        group=group,
        participant_ranks=(0, 1),
        global_rank=0,
        device=runtime.device,
        expected_source_lanes=(0,),
        decoder_solver=_FullGroupSolver(),
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        decoder_group_getter=lambda _size: group,
        decoder_group_ranks_getter=lambda _group: (0, 1),
        timeout_seconds=5.0,
        group_ranks_getter=lambda _group: (0, 1),
        all_to_all_single=lambda *_args, **_kwargs: None,
    )
    config = facade._config_factory(None, num_microbatches=1, forward_only=False)
    coordinator = facade._coordinator_factory(None, num_microbatches=1, forward_only=False)
    authority = coordinator._bindings.build_authority(_metadata(0, 0), object(), config)
    coordinator._bindings.authority_status_gate(None)

    assert digest_authorities == [authority]
    assert statuses[0].plan_digest == authority.plan.digest


def test_d3_gate2_status_uses_exact_legacy_iteration_plan_digest(monkeypatch):
    composition = import_module("megatron.core.mdp.dynamic_cp_d3_composition")
    coordinator = import_module("megatron.core.mdp.dynamic_cp_d3_coordinator")
    binding, authority = _iteration_authority()
    statuses = []
    digest_authorities = []

    def plan_digest(actual):
        digest_authorities.append(actual)
        return actual.plan.digest

    monkeypatch.setattr(composition, "_dynamic_iteration_plan_digest", plan_digest)
    monkeypatch.setattr(
        composition,
        "_run_precollective_consensus",
        lambda status, **_kwargs: statuses.append(status),
    )
    composition._run_status(
        coordinator._make_d3_gate_status_context(
            gate_id=2, authority=authority, phase_value=None, ready=None
        ),
        None,
        global_rank=binding.global_rank,
        group_ranks=binding.domain_ranks,
        all_gather_status=lambda _status: None,
        timeout_seconds=5.0,
    )

    assert digest_authorities == [authority]
    assert statuses[0].plan_digest == authority.plan.digest
    assert statuses[0].gate_id == 2


def test_authority_collective_rejects_foreign_domain_inside_preparation(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_authority_collective")
    binding, _ = _iteration_authority(rank=2)
    _, foreign = _iteration_authority(rank=6)
    events = []

    class _Runner:
        def run(self, **kwargs):
            events.append("runner")
            return kwargs["prepare"]()

    monkeypatch.setattr(type(binding), "begin_attempt", lambda *_args, **_kwargs: _Runner())

    with pytest.raises(MdpStateError, match="matches its local domain"):
        api.run_repeated_d4_authority_collective(
            binding, foreign, gate_id=2, prepare=lambda: None, domain_collective=lambda _value: None
        )
    assert events == ["runner"]


def test_payload_transport_prepares_everything_before_domain_collective(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_payload_transport")
    binding, authority = _iteration_authority(rank=0)
    source_window = _source_window(0)
    buffers = {torch.int64: (object(), object())}
    prepared = object()
    received = object()
    events = []

    def run(binding_arg, authority_arg, **kwargs):
        assert binding_arg is binding
        assert authority_arg is authority
        assert kwargs["gate_id"] == 0
        events.append("run")
        value = kwargs["prepare"]()
        events.append("prepared")
        return kwargs["domain_collective"](value)

    def attach(ledger, **kwargs):
        assert ledger is authority.payload_ledger
        assert kwargs["source_window"] is source_window
        events.append("attach")
        return {"local": "tensors"}

    def prepare(ledger, **kwargs):
        assert ledger is authority.payload_ledger
        assert kwargs["local_tensors"] == {"local": "tensors"}
        assert kwargs["buffers_by_dtype"] is buffers
        events.append("bundle")
        return prepared

    def execute(value, **kwargs):
        assert value is prepared
        assert kwargs["group"] is binding.domain_group
        events.append("collective")
        return received

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(api, "attach_local_decoder_payload_tensors", attach)
    monkeypatch.setattr(api, "prepare_decoder_payload_bundle", prepare)
    monkeypatch.setattr(api, "_execute_validated_decoder_payload_bundle", execute)

    result = api.run_repeated_d4_decoder_payload(
        binding,
        authority,
        source_window=source_window,
        buffers_by_dtype=buffers,
        all_to_all_single=lambda *_args, **_kwargs: None,
    )

    assert result is prepared
    assert events == ["run", "attach", "bundle", "prepared", "collective"]


def test_embedding_transport_prepares_everything_before_domain_collective(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_embedding_transport")
    binding, authority = _iteration_authority(rank=0)
    item_id = authority.global_manifest.items[0].item_id
    output = torch.zeros((1, authority.bridge_width), dtype=authority.bridge_dtype)
    item_outputs = {item_id: output}
    send_buffer, receive_buffer = object(), object()
    prepared = object()
    received = object()
    events = []

    def run(binding_arg, authority_arg, **kwargs):
        assert binding_arg is binding
        assert authority_arg is authority
        assert kwargs["gate_id"] == 1
        events.append("run")
        value = kwargs["prepare"]()
        events.append("prepared")
        return kwargs["domain_collective"](value)

    def prepare(ledger, reverse_ledger, **kwargs):
        assert ledger is authority.embedding_ledger
        assert reverse_ledger is authority.gradient_ledger
        expected_keys = {
            entry.key
            for entry in authority.embedding_ledger.entries
            if entry.src_global_rank == binding.global_rank
        }
        assert set(kwargs["local_tensors"]) == expected_keys
        assert all(value is output for value in kwargs["local_tensors"].values())
        assert kwargs["send_buffer"] is send_buffer
        assert kwargs["receive_buffer"] is receive_buffer
        events.append("prepare")
        return prepared

    def execute(value, **kwargs):
        assert value is prepared
        assert kwargs["group"] is binding.domain_group
        events.append("collective")
        return received

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(api, "prepare_dynamic_bridge_exchange", prepare)
    monkeypatch.setattr(api, "_execute_validated_dynamic_bridge_exchange", execute)

    result = api.run_repeated_d4_embedding(
        binding,
        authority,
        item_outputs=item_outputs,
        send_buffer=send_buffer,
        receive_buffer=receive_buffer,
        all_to_all_single=lambda *_args, **_kwargs: None,
    )

    assert result is prepared
    assert events == ["run", "prepare", "prepared", "collective"]


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


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_authority_collective_converges_rank6_failure_and_retries(groups):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_authority_collective")
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
    authority = _authority_api().build_repeated_d4_iteration_authority(
        binding,
        metadata,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )

    def prepare(*, fail=False):
        if fail:
            raise RuntimeError("rank 6 preparation")
        return torch.tensor(rank, dtype=torch.int64, device=device)

    with pytest.raises(MdpPlanError, match="gate mismatch at rank 6"):
        api.run_repeated_d4_authority_collective(
            binding,
            object() if rank == 6 else authority,
            gate_id=2,
            prepare=prepare,
            domain_collective=lambda _value: pytest.fail("data entered after rejection"),
        )

    with pytest.raises(MdpPlanError, match="rejected rank 6"):
        api.run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=2,
            prepare=lambda: prepare(fail=rank == 6),
            domain_collective=lambda _value: pytest.fail("data entered after rejection"),
        )

    def collect(value):
        torch.distributed.all_reduce(value, group=domain_group)
        return value.item()

    result = api.run_repeated_d4_authority_collective(
        binding, authority, gate_id=2, prepare=prepare, domain_collective=collect
    )
    assert result == (6 if lane == 0 else 22)


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_payload_rejects_rank6_then_retries_without_cross_domain_routes(groups):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_payload_transport")
    routing = import_module("megatron.core.mdp.dynamic_cp_routing")
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
    source_window = _source_window(lane, device=device)
    metadata = gather_decoder_source_manifests(
        source_window.metadata_manifest() if rank == domain_ranks[0] else None,
        expected_source_lanes=(lane,),
        group=domain_group,
        group_ranks=domain_ranks,
        global_rank=rank,
        device=device,
        timeout_seconds=30.0,
    )
    authority = _authority_api().build_repeated_d4_iteration_authority(
        binding,
        metadata,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    dtypes = tuple(
        dict.fromkeys(
            spec.dtype
            for payload in authority.global_manifest.payloads
            for spec in payload.field_specs
        )
    )
    buffers = {}
    for dtype in dtypes:
        inputs, outputs = routing.decoder_payload_split_sizes(
            authority.payload_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            source_rank_by_lane=authority.source_rank_by_lane,
            participant_ranks=authority.participant_ranks,
            dtype=dtype,
            global_rank=rank,
        )
        buffers[dtype] = (
            torch.empty(sum(inputs), dtype=dtype, device=device),
            torch.empty(sum(outputs), dtype=dtype, device=device),
        )

    collective_calls = 0

    def tracked_all_to_all(*args, **kwargs):
        nonlocal collective_calls
        collective_calls += 1
        return torch.distributed.all_to_all_single(*args, **kwargs)

    with pytest.raises(MdpPlanError, match="rejected rank 6"):
        api.run_repeated_d4_decoder_payload(
            binding,
            authority,
            source_window=source_window if rank in (domain_ranks[0], 6) else None,
            buffers_by_dtype=buffers,
            all_to_all_single=tracked_all_to_all,
        )
    assert collective_calls == 0

    prepared = api.run_repeated_d4_decoder_payload(
        binding,
        authority,
        source_window=source_window if rank == domain_ranks[0] else None,
        buffers_by_dtype=buffers,
        all_to_all_single=tracked_all_to_all,
    )
    received = prepared.received_tensors

    assert collective_calls == len(dtypes)
    expected_packet = source_window.packets[0]
    assert received
    assert all(key.sample_id.source_dp_lane == lane for key in received)
    for key, tensor in received.items():
        torch.testing.assert_close(tensor, expected_packet.tensor_fields[key.field_name])


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_embedding_rejects_rank6_then_retries_without_cross_domain_routes(groups):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_embedding_transport")
    bridge = import_module("megatron.core.mdp.dynamic_cp_bridge")
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
    source_window = _source_window(lane, device=device)
    metadata = gather_decoder_source_manifests(
        source_window.metadata_manifest() if rank == domain_ranks[0] else None,
        expected_source_lanes=(lane,),
        group=domain_group,
        group_ranks=domain_ranks,
        global_rank=rank,
        device=device,
        timeout_seconds=30.0,
    )
    authority = _authority_api().build_repeated_d4_iteration_authority(
        binding,
        metadata,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    item_id = authority.global_manifest.items[0].item_id
    output = torch.full(
        (authority.output_rows_by_item[item_id], authority.bridge_width),
        lane + 1,
        dtype=authority.bridge_dtype,
        device=device,
    )
    inputs, outputs = bridge.dynamic_bridge_split_sizes(
        authority.embedding_ledger,
        reverse_ledger=authority.gradient_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        width=authority.bridge_width,
        dtype=authority.bridge_dtype,
        participant_ranks=authority.participant_ranks,
        global_rank=rank,
    )
    send_buffer = torch.empty(sum(inputs), dtype=authority.bridge_dtype, device=device)
    receive_buffer = torch.empty(sum(outputs), dtype=authority.bridge_dtype, device=device)
    collective_calls = 0

    def tracked_all_to_all(*args, **kwargs):
        nonlocal collective_calls
        collective_calls += 1
        return torch.distributed.all_to_all_single(*args, **kwargs)

    invalid_outputs = {item_id: output} if rank in (domain_ranks[0], 6) else {}
    with pytest.raises(MdpPlanError, match="rejected rank 6"):
        api.run_repeated_d4_embedding(
            binding,
            authority,
            item_outputs=invalid_outputs,
            send_buffer=send_buffer,
            receive_buffer=receive_buffer,
            all_to_all_single=tracked_all_to_all,
        )
    assert collective_calls == 0

    prepared = api.run_repeated_d4_embedding(
        binding,
        authority,
        item_outputs={item_id: output} if rank == domain_ranks[0] else {},
        send_buffer=send_buffer,
        receive_buffer=receive_buffer,
        all_to_all_single=tracked_all_to_all,
    )
    received = prepared.received_tensors

    assert collective_calls == 1
    assert received
    assert all(key.item_id.source_dp_lane == lane for key in received)
    for tensor in received.values():
        torch.testing.assert_close(tensor, output)
