# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Encoder-only Gate3 gradient routing and ownership tests."""

import copy
import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_dynamic_decoder_replay as dynamic_replay_api
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as api
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay_api
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_bridge_transport import PreparedDynamicBridgeExchange
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_d4_authority_construction import (
    build_repeated_d4_joint_iteration_authority,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _AUTHORITY_SEAL,
    _BINDING_SEAL,
    _RepeatedD4GroupAuthority,
    _RepeatedD4GroupBinding,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderMicrobatchKey,
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
    EncoderWorkEstimate,
)
from megatron.core.mdp.errors import MdpPlanError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord


class _Group:
    def __init__(self, rank=0):
        self._rank = rank

    def size(self):
        return 4

    def rank(self):
        return self._rank


class _Solver:
    def __init__(self, cp_size):
        self.cp_size = cp_size

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        del max_seq_len_per_rank, min_cp_size
        count = total_gpus // self.cp_size
        selected = sample_seqlens[:count]
        lengths = []
        sample_ids = []
        for sample_id, length in selected:
            lengths.extend([[length]] * self.cp_size)
            sample_ids.extend([[sample_id]] * self.cp_size)
        return lengths, sample_seqlens[count:], None, sample_ids


def _dynamic_authority(binding, *, cp_size, vision, sample_count):
    samples = []
    items = []
    packets = []
    for index in range(sample_count):
        sample_id = GlobalSampleId(0, index)
        item_id = GlobalVisionItemId(0, index)
        sample_items = () if not vision else (EncoderVisionItemMetadata(item_id, sample_id, 0),)
        samples.append(DecoderSampleMetadata(sample_id, 4, 4, sample_items))
        if vision:
            items.append(DecoderVisionItemMetadata(item_id, sample_id, 0, (1, 1, 2), 2, (0, 1)))
        tensor = torch.arange(index * 4, index * 4 + 4).view(1, 4)
        packets.append(
            DecoderPayloadPacket(
                DECODER_EXECUTION_SCHEMA_VERSION,
                sample_id,
                4,
                4,
                DecoderPayloadHeaderV1(
                    DECODER_EXECUTION_SCHEMA_VERSION, 0, index, 4, 4, 1, 1, -1
                ).to_wire_tuple(),
                (DecoderTensorFieldSpec("input_ids", tensor.dtype, tuple(tensor.shape), "cpu"),),
                MappingProxyType({"input_ids": tensor}),
                ("position_ids",),
            )
        )
    source = finalize_decoder_source_window(
        source_dp_lane=0, samples=tuple(samples), items=tuple(items), packets=tuple(packets)
    )
    metadata = DecoderMetadataGatherResult(
        build_decoder_global_manifest((source.metadata_manifest(),)), {0: 0}
    )
    return build_repeated_d4_joint_iteration_authority(
        binding,
        metadata,
        decoder_max_seqlen_per_rank=4,
        decoder_minimum_cp_size=1,
        decoder_solver=_Solver(cp_size),
        encoder_max_seqlen_per_rank=max(4, sample_count),
        encoder_minimum_cp_size=1,
        encoder_workload_query=lambda values, group_size: EncoderWorkEstimate(
            max(1, len(values)), group_size
        ),
        bridge_width=3,
        bridge_dtype=torch.float32,
    )


class _Allocator:
    def __init__(self):
        self.acquired = []
        self.released = []
        self.fail_release = False
        self.fail_acquire_at = None
        self.acquire_error = None
        self.release_callback = None

    def acquire(self, *, rows, width, dtype, device, tag):
        if self.fail_acquire_at == len(self.acquired):
            raise self.acquire_error
        tensor = torch.empty(rows if width == 0 else (rows, width), dtype=dtype, device=device)
        self.acquired.append((tag, tensor))
        return tensor

    def release(self, tensor):
        self.released.append(tensor)
        if self.release_callback is not None:
            self.release_callback()
        if self.fail_release:
            raise RuntimeError("release failed")


class _BindingOwner:
    def __init__(self):
        self.active = True
        self.calls = 0

    def restore(self, _primary=None):
        self.calls += 1
        self.active = False


class _Handle:
    def __init__(self):
        self.consumed = False
        self.calls = 0

    def release_forward_only(self):
        self.calls += 1
        self.consumed = True


def _binding(group, rank=0):
    world_group = object()
    authority = _RepeatedD4GroupAuthority(
        world_ranks=tuple(range(8)),
        domain_ranks=(0, 1, 2, 3),
        global_rank=rank,
        expert_parallel_size=1,
        _world_group=world_group,
        _domain_group=group,
        _expert_group=None,
        _device=torch.device("cuda"),
        _timeout_seconds=1.0,
        _status_gather_factory=lambda: (lambda *_args, **_kwargs: None),
        _group_ranks_getter=lambda candidate: (
            tuple(range(8)) if candidate is world_group else (0, 1, 2, 3)
        ),
        _world_pre_gate=lambda *_args, **_kwargs: None,
        _domain_status=lambda *_args, **_kwargs: None,
        _seal=_AUTHORITY_SEAL,
    )
    return _RepeatedD4GroupBinding(
        world_ranks=tuple(range(8)),
        domain_ranks=(0, 1, 2, 3),
        global_rank=rank,
        expert_parallel_size=1,
        _authority=authority,
        _seal=_BINDING_SEAL,
    )


def _parts(
    monkeypatch,
    *,
    vision=True,
    selected=True,
    rank=0,
    runtime=None,
    missing_grad=False,
    leaf_corruption=None,
    replay_owner=False,
    dynamic=False,
    decoder_cp_size=1,
):
    group = _Group(rank)
    binding = _binding(group, rank)
    record_count = 4 if dynamic else 1
    item_ids = tuple(GlobalVisionItemId(0, index) for index in range(record_count))
    item_id = item_ids[0]
    route_key = DynamicBridgeKey(item_id, rank)
    received_key = DynamicBridgeKey(item_id, (rank - 1) % 4)
    items = (
        tuple(SimpleNamespace(item_id=value, output_rows=2) for value in item_ids) if vision else ()
    )
    entries = (
        tuple(
            SimpleNamespace(
                key=DynamicBridgeKey(value, source),
                src_global_rank=source,
                dst_global_rank=(source + 1) % 4,
            )
            for value in item_ids
            for source in range(4)
        )
        if vision
        else ()
    )
    authority = SimpleNamespace(
        plan=SimpleNamespace(decoder_cp_size=decoder_cp_size),
        global_manifest=SimpleNamespace(items=items),
        producer_rank_by_item=(
            MappingProxyType({value: 1 for value in item_ids}) if vision else MappingProxyType({})
        ),
        output_rows_by_item=(
            MappingProxyType({value: 2 for value in item_ids}) if vision else MappingProxyType({})
        ),
        embedding_ledger=SimpleNamespace(entries=entries),
        gradient_ledger=SimpleNamespace(entries=entries),
        participant_ranks=(0, 1, 2, 3),
        bridge_width=3,
        bridge_dtype=torch.float32,
    )
    if dynamic:
        authority = _dynamic_authority(
            binding, cp_size=decoder_cp_size, vision=vision, sample_count=record_count
        )
    if dynamic:
        samples = {sample.sample_id: sample for sample in authority.global_manifest.samples}
        manifest_items = {item.item_id: item for item in authority.global_manifest.items}
        local_records = []
        for microbatch in authority.plan.microbatches:
            assignment = next(
                value for value in microbatch.assignments if rank in value.endpoint_ranks
            )
            vision_records = []
            padded_start = 0
            for local_sample_id, sample_id in enumerate(assignment.sample_ids):
                sample = samples[sample_id]
                for encoder_item in sample.vision_items:
                    item = manifest_items[encoder_item.item_id]
                    vision_records.append(
                        MdpMicrobatchVisionRecord(
                            item.item_id,
                            local_sample_id,
                            item.image_ordinal,
                            item.grid_thw,
                            item.output_rows,
                            tuple(padded_start + offset for offset in item.decoder_offsets),
                        )
                    )
                padded_start += sample.padded_seqlen
            local_records.append(
                MdpMicrobatchRecord(
                    microbatch.microbatch_index,
                    not vision,
                    tuple(vision_records),
                    object(),
                    MappingProxyType({}),
                )
            )
        records = tuple(local_records)
        record_count = len(records)
        assert record_count == decoder_cp_size
        assert all(
            next(
                assignment
                for assignment in authority.plan.microbatches[record.microbatch_id].assignments
                if rank in assignment.endpoint_ranks
            ).local_cp_size
            == decoder_cp_size
            for record in records
        )
    else:
        records = tuple(
            MdpMicrobatchRecord(
                index,
                not vision,
                (
                    (MdpMicrobatchVisionRecord(value, 0, index, (1, 1, 2), 2, (0, 1)),)
                    if vision
                    else ()
                ),
                object(),
                MappingProxyType({}),
            )
            for index, value in enumerate(item_ids)
        )
    record = records[0]
    leaf_values = {
        DecoderMicrobatchKey(record.microbatch_id): torch.ones(
            (sum(item.output_rows for item in record.vision_items), 3), requires_grad=True
        )
        for record in records
        if vision
    }
    leaf = leaf_values.get(DecoderMicrobatchKey(0))
    if leaf_corruption == "geometry":
        leaf = torch.ones((1, 3), requires_grad=True)
        leaf_values[DecoderMicrobatchKey(0)] = leaf
    if not missing_grad:
        for value in leaf_values.values():
            (value * 2.0).sum().backward()
    if leaf_corruption == "missing":
        leaf_values.clear()
    elif leaf_corruption == "extra":
        leaf_values[DecoderMicrobatchKey(1)] = leaf
    leaves = MappingProxyType(leaf_values)
    allocator = _Allocator() if runtime is None else runtime.allocator
    runtime = runtime or SimpleNamespace(device=torch.device("cpu"), allocator=allocator)
    operations = SimpleNamespace(acquire=allocator.acquire, release=allocator.release)
    binding_owner = _BindingOwner() if selected else None
    handle = _Handle() if selected else None
    predecessor_buffers = (torch.empty(0), torch.empty(0))
    handoff_resources = (binding_owner, predecessor_buffers, operations, handle)
    leaf_bases = tuple(leaf_values.values())
    token = torch.tensor(4.0)
    replay_trusted = (runtime, authority, binding, records, leaves, handoff_resources, leaf_bases)
    monkeypatch.setattr(replay_api, "_snapshot_local_authority", lambda b, a: a)
    if replay_owner and not dynamic:
        replay = replay_api._D4FixedDecoderReplayOwner(
            trusted=replay_trusted, seal=replay_api._OWNER_SEAL
        )
        reference = weakref.ref(replay)
        replay_api._ACTIVE_OWNERS[id(replay)] = (
            reference,
            *replay_trusted,
            replay_api._OwnerEscrow(),
        )
        replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
        cursor = replay.replay_cursor()
        assert next(cursor) is record
        replay.capture_global_num_tokens(token)
        returned = replay.mark_schedule_returned(cursor)
        completion = replay.prepare_completion(cursor, returned)
        handoff = None
    elif not dynamic:
        replay = None
        predecessor = SimpleNamespace()
        completion = replay_api._D4FixedDecoderCompletion(
            authority, token, predecessor, replay_api._COMPLETION_SEAL
        )
        trusted = (
            *replay_trusted,
            completion,
            predecessor,
            replay_api._tensor_descriptor(token),
            not vision,
            selected,
        )
        handoff = replay_api._D4FixedDecoderGradientHandoff(trusted)
        reference = weakref.ref(handoff)
        completion_entry = (reference, completion, authority, token, trusted[9])
        trusted = (*trusted, completion_entry)
        handoff._trusted = trusted
        replay_api._ACTIVE_GRADIENT_HANDOFFS[id(handoff)] = (reference, *trusted, False)
        replay_api._ACTIVE_COMPLETIONS[id(completion)] = completion_entry
        replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    else:
        handoff_trusted = (
            runtime,
            authority,
            binding,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            handle,
            not vision,
            selected,
            selected and rank == 0,
            binding_owner,
            predecessor_buffers,
            operations,
        )
        owner_trusted = (
            runtime,
            authority,
            binding,
            handoff_trusted,
            operations.acquire,
            operations.release,
            runtime.device,
        )
        replay = dynamic_replay_api._D4DynamicDecoderReplayOwner(
            owner_trusted, seal=dynamic_replay_api._OWNER_SEAL
        )
        reference = weakref.ref(replay)
        lifecycle = dynamic_replay_api._OwnerEscrow(
            reference, None, seal=dynamic_replay_api._OWNER_ESCROW_SEAL
        )
        cleanup = dynamic_replay_api._CleanupEscrow(seal=dynamic_replay_api._CLEANUP_ESCROW_SEAL)
        ready = SimpleNamespace(records=records, embedding_leaves=leaves)
        owner_entry = (reference, *owner_trusted, cleanup, lifecycle, leaf_bases, ready)
        lifecycle.entry = owner_entry
        replay.ready = ready
        replay.records = ready.records
        replay.embedding_leaves = ready.embedding_leaves
        replay._state = dynamic_replay_api._ACTIVE
        replay._trusted = owner_entry
        dynamic_replay_api._ACTIVE_OWNERS[id(replay)] = owner_entry
        dynamic_replay_api._TRUSTED_OWNERS[id(replay)] = owner_entry
        dynamic_replay_api._OWNER_ESCROWS[id(replay)] = lifecycle
        replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
        cursor = replay.replay_cursor()
        for expected_record in records:
            assert next(cursor) is expected_record
        replay.capture_global_num_tokens(token)
        returned = replay.mark_schedule_returned(cursor)
        completion = replay.prepare_completion(cursor, returned)
        monkeypatch.setattr(dynamic_replay_api, "_snapshot_local_authority", lambda b, a: a)
        handoff = None if replay_owner else replay._claim_for_gradient(authority, completion)
    events = []
    prepared_calls = []
    physical_calls = []

    monkeypatch.setattr(api, "_validate_repeated_d4_group_binding", lambda value: value._authority)
    monkeypatch.setattr(
        api,
        "dynamic_bridge_split_sizes",
        lambda *_args, **_kwargs: (
            (
                tuple(6 * record_count if index == (rank + 1) % 4 else 0 for index in range(4)),
                tuple(6 * record_count if index == (rank - 1) % 4 else 0 for index in range(4)),
            )
            if vision
            else ((0, 0, 0, 0), (0, 0, 0, 0))
        ),
    )

    def prepare_exchange(*args, **kwargs):
        prepared_calls.append((args, kwargs))
        received = (
            MappingProxyType(
                {DynamicBridgeKey(value, (rank - 1) % 4): torch.empty((2, 3)) for value in item_ids}
            )
            if vision
            else MappingProxyType({})
        )
        input_splits, output_splits = api.dynamic_bridge_split_sizes()
        return PreparedDynamicBridgeExchange(
            BridgePhase.GRADIENT,
            torch.float32,
            rank,
            (0, 1, 2, 3),
            input_splits,
            output_splits,
            b"g" * 16,
            kwargs["send_buffer"],
            kwargs["receive_buffer"],
            received,
        )

    monkeypatch.setattr(api, "prepare_dynamic_bridge_exchange", prepare_exchange)
    monkeypatch.setattr(api, "validate_prepared_dynamic_bridge_exchange", lambda value: value)

    def physical(exchange, *, group, all_to_all_single):
        events.append("physical")
        physical_calls.append((exchange, group, all_to_all_single))
        return exchange.received_tensors

    monkeypatch.setattr(api, "_execute_validated_dynamic_bridge_exchange", physical)

    def runner(_binding, _authority, **kwargs):
        events.append(("gate", kwargs["gate_id"], kwargs["byte_generator"]))
        candidate = kwargs["prepare"]()
        events.append("world0")
        candidate = kwargs["domain_collective"](candidate)
        events.extend(("domain", "world1"))
        return candidate

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    return SimpleNamespace(
        runtime=runtime,
        allocator=allocator,
        authority=authority,
        binding=binding,
        handoff=handoff,
        replay=replay,
        records=records,
        completion=completion,
        leaf=leaf,
        leaves=leaves,
        route_key=route_key,
        received_key=received_key,
        binding_owner=binding_owner,
        handle=handle,
        predecessor_buffers=predecessor_buffers,
        operations=operations,
        events=events,
        prepared_calls=prepared_calls,
        physical_calls=physical_calls,
    )


def _run(parts, *, byte_generator=None):
    return api.run_repeated_d4_encoder_gradient(
        parts.handoff,
        parts.authority,
        parts.completion,
        all_to_all_single=lambda *_args, **_kwargs: None,
        byte_generator=byte_generator,
    )


def _run_dynamic(parts, *, byte_generator=None):
    return api.run_repeated_d4_dynamic_encoder_gradient(
        parts.handoff,
        parts.authority,
        parts.completion,
        all_to_all_single=lambda *_args, **_kwargs: None,
        byte_generator=byte_generator,
    )


def _run_from_replay(parts, *, byte_generator=None):
    return api._run_repeated_d4_encoder_gradient_from_replay(
        parts.replay,
        parts.authority,
        parts.completion,
        all_to_all_single=lambda *_args, **_kwargs: None,
        byte_generator=byte_generator,
    )


def _run_dynamic_from_replay(parts, *, byte_generator=None):
    return api._run_repeated_d4_dynamic_encoder_gradient_from_replay(
        parts.replay,
        parts.authority,
        parts.completion,
        all_to_all_single=lambda *_args, **_kwargs: None,
        byte_generator=byte_generator,
    )


def test_replay_gradient_claim_and_successor_transfer_are_atomic(monkeypatch):
    parts = _parts(monkeypatch, replay_owner=True)
    generator = object()

    owner = _run_from_replay(parts, byte_generator=generator)

    assert owner.require() is owner
    assert parts.events == [("gate", 3, generator), "world0", "domain", "world1", "physical"]
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is owner
    with pytest.raises(MdpStateError, match="replay owner is retired"):
        parts.replay.require()
    owner.abort()
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1


@pytest.mark.parametrize("mutated", (False, True))
def test_replay_gradient_preclaim_failure_retires_once_and_retries(monkeypatch, mutated):
    parts = _parts(monkeypatch, replay_owner=True)
    if mutated:
        parts.replay.records = ()
        authority = parts.authority
        message = "retains sealed resources"
    else:
        authority = copy.copy(parts.authority)
        message = "exact iteration authority"

    with pytest.raises(MdpStateError, match=message):
        api._run_repeated_d4_encoder_gradient_from_replay(parts.replay, authority, parts.completion)
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    assert tuple(parts.allocator.released) == (parts.leaf, *parts.predecessor_buffers)
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    fresh = _parts(monkeypatch, runtime=parts.runtime, replay_owner=True)
    owner = _run_from_replay(fresh)
    owner.abort()


def test_dynamic_replay_gradient_claim_and_successor_transfer_are_atomic(monkeypatch):
    parts = _parts(monkeypatch, replay_owner=True, dynamic=True, decoder_cp_size=4)
    generator = object()

    owner = _run_dynamic_from_replay(parts, byte_generator=generator)

    assert owner.require() is owner
    assert parts.events == [("gate", 3, generator), "world0", "domain", "world1", "physical"]
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is owner
    with pytest.raises(MdpStateError, match="dynamic decoder replay owner is retired"):
        parts.replay.require()
    owner.abort()
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1


@pytest.mark.parametrize("mutated", (False, True))
def test_dynamic_replay_gradient_preclaim_failure_retires_once_and_retries(monkeypatch, mutated):
    parts = _parts(monkeypatch, replay_owner=True, dynamic=True, decoder_cp_size=4)
    if mutated:
        parts.replay.records = ()
        authority = parts.authority
        message = "retains sealed fields"
    else:
        authority = copy.copy(parts.authority)
        message = "exact iteration authority"

    with pytest.raises(MdpStateError, match=message):
        api._run_repeated_d4_dynamic_encoder_gradient_from_replay(
            parts.replay, authority, parts.completion
        )
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    expected_releases = (*parts.leaves.values(), *parts.predecessor_buffers)
    assert len(parts.allocator.released) == len(expected_releases)
    assert all(
        sum(released is expected for released in parts.allocator.released) == 1
        for expected in expected_releases
    )
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    fresh = _parts(
        monkeypatch, runtime=parts.runtime, replay_owner=True, dynamic=True, decoder_cp_size=4
    )
    owner = _run_dynamic_from_replay(fresh)
    owner.abort()


@pytest.mark.parametrize("stage", ("world", "physical"))
def test_dynamic_replay_gradient_postclaim_failure_cleans_successor_and_retries(monkeypatch, stage):
    parts = _parts(monkeypatch, replay_owner=True, dynamic=True, decoder_cp_size=4)
    if stage == "world":
        primary = MdpPlanError("first WORLD rejected")
        monkeypatch.setattr(
            api,
            "run_repeated_d4_authority_collective",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )
        expected = MdpPlanError
    else:
        monkeypatch.setattr(
            api,
            "_execute_validated_dynamic_bridge_exchange",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("physical failed")),
        )
        expected = MdpTaskFatalError

    with pytest.raises(expected):
        _run_dynamic_from_replay(parts)
    with pytest.raises(MdpStateError, match="dynamic decoder replay owner is retired"):
        parts.replay.require()
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    assert dynamic_replay_api._ACTIVE_GRADIENT_HANDOFFS == {}
    assert dynamic_replay_api._ACTIVE_COMPLETIONS == {}
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    fresh = _parts(
        monkeypatch, runtime=parts.runtime, replay_owner=True, dynamic=True, decoder_cp_size=4
    )
    owner = _run_dynamic_from_replay(fresh)
    owner.abort()


@pytest.mark.parametrize(
    "mutation", ("raise", "delete", "substitute", "runtime", "receipt", "completion", "trusted")
)
def test_replay_gradient_activation_failure_cleans_exact_successor(monkeypatch, mutation):
    parts = _parts(monkeypatch, replay_owner=True)
    original = api._D4EncoderGradientRouteOwner._activate_prepared
    primary = RuntimeError("activation failed after registry installation")
    foreign_owner = _Handle()
    foreign_runtime = (parts.runtime, weakref.ref(foreign_owner))
    foreign_nested = (weakref.ref(foreign_owner), object())
    observed_foreign = []

    def fail(owner, handoff, owner_entry, handoff_entry):
        original(owner, handoff, owner_entry, handoff_entry)
        if mutation == "delete":
            del api._ACTIVE_OWNERS[id(owner)]
        elif mutation == "substitute":
            api._ACTIVE_OWNERS[id(owner)] = foreign_nested
        elif mutation == "runtime":
            replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign_runtime
            parts.allocator.release_callback = lambda: observed_foreign.append(
                replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is foreign_runtime
            )
        elif mutation == "receipt":
            receipt_identity = id(owner.receipt)
            api._ACTIVE_RECEIPTS[receipt_identity] = foreign_nested
            parts.allocator.release_callback = lambda: observed_foreign.append(
                api._ACTIVE_RECEIPTS.get(receipt_identity) is foreign_nested
            )
        elif mutation == "completion":
            completion_identity = id(owner.completion)
            replay_api._ACTIVE_COMPLETIONS[completion_identity] = foreign_nested
            parts.allocator.release_callback = lambda: observed_foreign.append(
                replay_api._ACTIVE_COMPLETIONS.get(completion_identity) is foreign_nested
            )
        elif mutation == "trusted":
            owner._trusted = ()
        raise primary

    monkeypatch.setattr(api._D4EncoderGradientRouteOwner, "_activate_prepared", fail)
    with pytest.raises(MdpTaskFatalError, match="physical encoder-gradient") as raised:
        _run_from_replay(parts)
    assert raised.value.__cause__ is primary
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    acquired = tuple(tensor for _tag, tensor in parts.allocator.acquired)
    expected_releases = (*acquired, parts.leaf, *parts.predecessor_buffers)
    assert len(parts.allocator.released) == len(expected_releases)
    assert all(
        actual is expected
        for actual, expected in zip(parts.allocator.released, expected_releases, strict=True)
    )
    if mutation in ("runtime", "receipt", "completion"):
        assert observed_foreign and all(observed_foreign)
        parts.allocator.release_callback = None
    if mutation == "runtime":
        assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
        del replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]
    else:
        assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    if mutation == "receipt":
        assert next(iter(api._ACTIVE_RECEIPTS.values())) is foreign_nested
        api._ACTIVE_RECEIPTS.clear()
    else:
        assert api._ACTIVE_RECEIPTS == {}
    if mutation == "completion":
        assert replay_api._ACTIVE_COMPLETIONS[id(parts.completion)] is foreign_nested
        del replay_api._ACTIVE_COMPLETIONS[id(parts.completion)]
    else:
        assert replay_api._ACTIVE_COMPLETIONS.get(id(parts.completion)) is None
    if mutation == "substitute":
        assert api._ACTIVE_OWNERS.pop(next(iter(api._ACTIVE_OWNERS))) is foreign_nested
    else:
        assert api._ACTIVE_OWNERS == {}

    monkeypatch.setattr(api._D4EncoderGradientRouteOwner, "_activate_prepared", original)
    fresh = _parts(monkeypatch, runtime=parts.runtime, replay_owner=True)
    owner = _run_from_replay(fresh)
    owner.abort()


@pytest.mark.parametrize("mutation", ("receipt", "completion", "runtime", "prepared"))
def test_replay_gradient_preactivation_field_mutation_never_installs_wrong_keys(
    monkeypatch, mutation
):
    parts = _parts(monkeypatch, replay_owner=True)
    original = api._D4EncoderGradientRouteOwner._activate_prepared
    foreign = object()

    def mutate(owner, handoff, owner_entry, handoff_entry):
        if mutation == "receipt":
            owner.receipt = foreign
        elif mutation == "completion":
            owner.completion = foreign
        elif mutation == "runtime":
            owner._runtime = foreign
        else:
            owner._prepared_entry = None
        original(owner, handoff, owner_entry, handoff_entry)

    monkeypatch.setattr(api._D4EncoderGradientRouteOwner, "_activate_prepared", mutate)
    with pytest.raises(MdpTaskFatalError, match="physical encoder-gradient"):
        _run_from_replay(parts)

    assert id(foreign) not in api._ACTIVE_RECEIPTS
    assert id(foreign) not in replay_api._ACTIVE_COMPLETIONS
    assert id(foreign) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert api._ACTIVE_OWNERS == {}
    assert api._ACTIVE_RECEIPTS == {}
    assert replay_api._ACTIVE_COMPLETIONS.get(id(parts.completion)) is None
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1

    monkeypatch.setattr(api._D4EncoderGradientRouteOwner, "_activate_prepared", original)
    fresh = _parts(monkeypatch, runtime=parts.runtime, replay_owner=True)
    owner = _run_from_replay(fresh)
    owner.abort()


@pytest.mark.parametrize("mutation", ("delete", "substitute", "runtime"))
def test_replay_gradient_preactivation_uses_handoff_escrow(monkeypatch, mutation):
    parts = _parts(monkeypatch, replay_owner=True)
    primary = RuntimeError("gradient handoff callback failed")
    foreign_owner = _Handle()
    foreign_handoff = (weakref.ref(foreign_owner), object())
    foreign_runtime = (parts.runtime, weakref.ref(foreign_owner))
    observed_foreign = []
    seen = []

    def fail(_binding, _authority, **_kwargs):
        handoff = replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]()
        seen.append(handoff)
        if mutation == "delete":
            del replay_api._ACTIVE_GRADIENT_HANDOFFS[id(handoff)]
        elif mutation == "substitute":
            replay_api._ACTIVE_GRADIENT_HANDOFFS[id(handoff)] = foreign_handoff
        else:
            replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign_runtime
            parts.allocator.release_callback = lambda: observed_foreign.append(
                replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is foreign_runtime
            )
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", fail)
    with pytest.raises(RuntimeError) as raised:
        _run_from_replay(parts)
    assert raised.value is primary
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    assert tuple(parts.allocator.released) == (parts.leaf, *parts.predecessor_buffers)
    if mutation == "substitute":
        assert replay_api._ACTIVE_GRADIENT_HANDOFFS[id(seen[0])] is foreign_handoff
        del replay_api._ACTIVE_GRADIENT_HANDOFFS[id(seen[0])]
    if mutation == "runtime":
        assert observed_foreign and all(observed_foreign)
        parts.allocator.release_callback = None
        assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
        del replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]
    else:
        assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


@pytest.mark.parametrize("dynamic", (False, True))
@pytest.mark.parametrize("timing", ("before", "after"))
def test_prepare_from_failure_without_local_owner_entry_cleans_predecessor_once(
    monkeypatch, dynamic, timing
):
    parts = _parts(monkeypatch, dynamic=dynamic)
    primary = RuntimeError(f"prepare-from {timing} failed")
    original = api._D4EncoderGradientRouteOwner._prepare_from

    def fail(owner, lifecycle):
        if timing == "after":
            original(owner, lifecycle)
        raise primary

    monkeypatch.setattr(api._D4EncoderGradientRouteOwner, "_prepare_from", fail)
    with pytest.raises(RuntimeError) as caught:
        (_run_dynamic if dynamic else _run)(parts)
    assert caught.value is primary
    assert getattr(primary, "__notes__", ()) == ()
    expected = (
        *(tensor for _tag, tensor in parts.allocator.acquired),
        parts.leaf,
        *parts.predecessor_buffers,
    )
    assert all(
        sum(released is tensor for released in parts.allocator.released) == 1 for tensor in expected
    )
    handoff_registry = (
        dynamic_replay_api._ACTIVE_GRADIENT_HANDOFFS
        if dynamic
        else replay_api._ACTIVE_GRADIENT_HANDOFFS
    )
    completion_registry = (
        dynamic_replay_api._ACTIVE_COMPLETIONS if dynamic else replay_api._ACTIVE_COMPLETIONS
    )
    assert api._ACTIVE_OWNERS == {}
    assert api._ACTIVE_RECEIPTS == {}
    assert handoff_registry.get(id(parts.handoff)) is None
    assert completion_registry.get(id(parts.completion)) is None
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    monkeypatch.setattr(api._D4EncoderGradientRouteOwner, "_prepare_from", original)
    fresh = _parts(monkeypatch, dynamic=dynamic, runtime=parts.runtime)
    owner = (_run_dynamic if dynamic else _run)(fresh)
    owner.abort()


@pytest.mark.parametrize(
    ("vision", "selected", "rank"), ((False, False, 0), (True, False, 1), (True, True, 0))
)
def test_gate3_routes_manifest_order_gradients_after_full_authorization(
    monkeypatch, vision, selected, rank
):
    parts = _parts(monkeypatch, vision=vision, selected=selected, rank=rank)
    generator = object()
    releases = tuple(parts.allocator.released)

    owner = _run(parts, byte_generator=generator)

    assert owner.require() is owner
    assert parts.events == [("gate", 3, generator), "world0", "domain", "world1", "physical"]
    assert len(parts.physical_calls) == 1
    assert parts.physical_calls[0][0] is owner.receipt.exchange
    assert owner.receipt.received_tensors is owner.receipt.exchange.received_tensors
    if vision:
        assert tuple(owner.receipt.received_tensors) == (parts.received_key,)
    assert owner.completion is parts.completion
    completion_entry = replay_api._ACTIVE_COMPLETIONS[id(parts.completion)]
    assert completion_entry is owner._trusted[15]
    assert len(completion_entry) == 5
    assert completion_entry[0]() is owner
    assert owner.text_only is (not vision)
    assert owner.is_selected is selected
    assert tuple(parts.allocator.released) == releases
    if vision:
        local = parts.prepared_calls[0][1]["local_tensors"]
        assert tuple(local) == (parts.route_key,)
        assert local[parts.route_key].data_ptr() == parts.leaf.grad.data_ptr()
        assert tuple(local[parts.route_key].shape) == (2, 3)
    else:
        assert not parts.prepared_calls[0][1]["local_tensors"]
    with pytest.raises(MdpStateError, match="handoff is retired"):
        parts.handoff.require()
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is owner
    owner.abort()
    assert parts.handle is None or parts.handle.consumed
    assert parts.binding_owner is None or not parts.binding_owner.active


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing_grad", "exact detached leaf gradients"),
        ("missing", "covers every vision replay leaf"),
        ("extra", "exact replay leaf coverage"),
        ("geometry", "exact detached leaf gradients"),
    ),
)
def test_local_leaf_error_enters_first_world_and_allows_same_runtime_retry(
    monkeypatch, corruption, message
):
    parts = _parts(
        monkeypatch,
        missing_grad=corruption == "missing_grad",
        leaf_corruption=None if corruption == "missing_grad" else corruption,
    )
    common = MdpPlanError("rejected rank 0 with error code 1")

    def converge(_binding, _authority, **kwargs):
        parts.events.append(("gate", kwargs["gate_id"], kwargs["byte_generator"]))
        try:
            kwargs["prepare"]()
        except MdpStateError as local:
            assert message in str(local)
            parts.events.append(("world0", local))
            raise common from local
        raise AssertionError("missing gradient unexpectedly passed preparation")

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", converge)
    with pytest.raises(MdpPlanError) as raised:
        _run(parts)
    assert raised.value is common
    assert isinstance(raised.value.__cause__, MdpStateError)
    assert parts.events[0][0:2] == ("gate", 3)
    assert parts.events[1][0] == "world0"
    assert parts.physical_calls == []
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    fresh = _parts(monkeypatch, runtime=parts.runtime)
    owner = _run(fresh)
    owner.abort()


def test_projection_slices_multiple_items_and_leaves_in_manifest_order():
    item_ids = tuple(GlobalVisionItemId(0, index) for index in range(3))
    records = (
        MdpMicrobatchRecord(
            0,
            False,
            (
                MdpMicrobatchVisionRecord(item_ids[0], 0, 0, (1, 1, 1), 1, (0,)),
                MdpMicrobatchVisionRecord(item_ids[1], 0, 1, (1, 1, 2), 2, (1, 2)),
            ),
            object(),
            MappingProxyType({}),
        ),
        MdpMicrobatchRecord(
            1,
            False,
            (MdpMicrobatchVisionRecord(item_ids[2], 0, 2, (1, 1, 1), 1, (0,)),),
            object(),
            MappingProxyType({}),
        ),
    )
    first = torch.ones((3, 2), requires_grad=True)
    second = torch.ones((1, 2), requires_grad=True)
    (first * torch.tensor([[1.0], [2.0], [3.0]])).sum().backward()
    (second * 4.0).sum().backward()
    keys = tuple(DynamicBridgeKey(item_id, 1) for item_id in item_ids)
    authority = SimpleNamespace(
        bridge_dtype=torch.float32,
        bridge_width=2,
        global_manifest=SimpleNamespace(
            items=tuple(SimpleNamespace(item_id=item_id) for item_id in item_ids)
        ),
        gradient_ledger=SimpleNamespace(
            entries=tuple(SimpleNamespace(key=key, src_global_rank=1) for key in keys)
        ),
    )

    projected = api._project_leaf_gradients(
        records=records,
        embedding_leaves=MappingProxyType(
            {DecoderMicrobatchKey(0): first, DecoderMicrobatchKey(1): second}
        ),
        authority=authority,
        global_rank=1,
    )

    assert tuple(projected) == keys
    assert projected[keys[0]].data_ptr() == first.grad.data_ptr()
    assert projected[keys[1]].storage_offset() == 2
    assert torch.equal(projected[keys[0]], torch.ones((1, 2)))
    assert torch.equal(projected[keys[1]], torch.tensor([[2.0, 2.0], [3.0, 3.0]]))
    assert torch.equal(projected[keys[2]], torch.full((1, 2), 4.0))


@pytest.mark.parametrize("stage", ("first", "final", "substitute", "reenter"))
def test_gate3_rejection_and_runner_substitution_never_execute_a2a(monkeypatch, stage):
    parts = _parts(monkeypatch)
    primary = MdpPlanError(f"{stage} WORLD rejected")

    def runner(_binding, _authority, **kwargs):
        candidate = kwargs["prepare"]()
        parts.events.append("world0")
        if stage == "first":
            raise primary
        if stage == "reenter":
            kwargs["prepare"]()
        candidate = kwargs["domain_collective"](candidate)
        parts.events.append("domain")
        if stage == "final":
            parts.events.append("world1")
            raise primary
        return object()

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    expected = (
        MdpStateError
        if stage == "reenter"
        else (MdpTaskFatalError if stage == "substitute" else MdpPlanError)
    )
    with pytest.raises(expected) as raised:
        _run(parts)
    if stage in ("first", "final"):
        assert raised.value is primary
    assert parts.physical_calls == []
    assert parts.handle.consumed
    assert id(parts.handoff) not in replay_api._ACTIVE_GRADIENT_HANDOFFS
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    owner = _run(fresh)
    owner.abort()


def test_physical_failure_is_task_fatal_and_cleanup_uses_escrow(monkeypatch):
    parts = _parts(monkeypatch)

    def physical(*_args, **_kwargs):
        parts.handoff._trusted = ()
        raise RuntimeError("A2A failed")

    monkeypatch.setattr(api, "_execute_validated_dynamic_bridge_exchange", physical)
    with pytest.raises(MdpTaskFatalError, match="physical encoder-gradient route"):
        _run(parts)
    assert all(
        any(released is tensor for released in parts.allocator.released)
        for _tag, tensor in parts.allocator.acquired
    )
    assert parts.handle.consumed
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


def test_partial_preparation_cleanup_preserves_primary_notes_and_fresh_runtime(monkeypatch):
    parts = _parts(monkeypatch)
    primary = MdpStateError("second allocation failed")
    parts.allocator.fail_acquire_at = 1
    parts.allocator.acquire_error = primary
    parts.allocator.fail_release = True
    events = []

    def converge(_binding, _authority, **kwargs):
        events.append("begin")
        try:
            kwargs["prepare"]()
        except BaseException as error:
            events.append(("world0", error))
            raise

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", converge)
    with pytest.raises(MdpStateError) as raised:
        _run(parts)
    assert raised.value is primary
    assert events == ["begin", ("world0", primary)]
    assert any("preparation cleanup error" in note for note in primary.__notes__)
    assert len(parts.allocator.released) == 4
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    parts.allocator.fail_acquire_at = None
    parts.allocator.fail_release = False
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    owner = _run(fresh)
    owner.abort()


def test_owner_abort_clears_before_hostile_note_reentry(monkeypatch):
    parts = _parts(monkeypatch)
    owner = _run(parts)
    reentry = []

    class HostilePrimary(BaseException):
        def add_note(self, _message):
            try:
                owner.abort(self)
            except MdpStateError as error:
                reentry.append(error)
            raise RuntimeError("hostile note")

    del owner.authority
    parts.allocator.fail_release = True
    primary = HostilePrimary("later failure")
    owner.abort(primary)
    assert reentry and all("retired" in str(error) for error in reentry)
    assert parts.handle.consumed
    assert not parts.binding_owner.active
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


@pytest.mark.parametrize("decoder_cp_size", (1, 2, 4))
@pytest.mark.parametrize(
    ("vision", "selected", "rank"), ((True, True, 0), (True, False, 1), (False, False, 0))
)
def test_dynamic_gate3_uses_shared_lifecycle_and_exact_routes(
    monkeypatch, decoder_cp_size, vision, selected, rank
):
    parts = _parts(
        monkeypatch,
        dynamic=True,
        decoder_cp_size=decoder_cp_size,
        vision=vision,
        selected=selected,
        rank=rank,
    )
    owner = _run_dynamic(parts)
    assert owner.require() is owner
    assert parts.events == [("gate", 3, None), "world0", "domain", "world1", "physical"]
    assert owner.completion is parts.completion
    assert owner.text_only is (not vision)
    assert owner.is_selected is selected
    if vision:
        local = parts.prepared_calls[0][1]["local_tensors"]
        expected_keys = tuple(
            entry.key
            for entry in parts.authority.gradient_ledger.entries
            if entry.src_global_rank == rank
        )
        assert tuple(local) == expected_keys
        assert len(parts.records) == decoder_cp_size
        assert tuple(
            item.global_item_id for record in parts.records for item in record.vision_items
        ) == tuple(key.item_id for key in expected_keys)
        for record in parts.records:
            gradient = parts.leaves[DecoderMicrobatchKey(record.microbatch_id)].grad
            offset = 0
            for item in record.vision_items:
                key = DynamicBridgeKey(item.global_item_id, rank)
                view = local[key]
                assert view.data_ptr() == gradient.narrow(0, offset, item.output_rows).data_ptr()
                assert (
                    view.storage_offset() == gradient.storage_offset() + offset * gradient.stride(0)
                )
                offset += item.output_rows
    else:
        assert not parts.prepared_calls[0][1]["local_tensors"]
    with pytest.raises(MdpStateError, match="dynamic decoder gradient handoff is retired"):
        parts.handoff.require()
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is owner
    assert dynamic_replay_api._ACTIVE_COMPLETIONS[id(parts.completion)] is owner._trusted[15]
    owner.abort()
    assert dynamic_replay_api._ACTIVE_COMPLETIONS.get(id(parts.completion)) is None


@pytest.mark.parametrize("stage", ("first", "final"))
def test_dynamic_gate3_rejection_is_prephysical_and_retryable(monkeypatch, stage):
    parts = _parts(monkeypatch, dynamic=True)
    primary = MdpPlanError(f"{stage} WORLD rejected")

    def runner(_binding, _authority, **kwargs):
        candidate = kwargs["prepare"]()
        parts.events.append("world0")
        if stage == "first":
            raise primary
        kwargs["domain_collective"](candidate)
        parts.events.append("domain")
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpPlanError) as raised:
        _run_dynamic(parts)
    assert raised.value is primary
    assert parts.physical_calls == []
    assert id(parts.handoff) not in dynamic_replay_api._ACTIVE_GRADIENT_HANDOFFS
    assert dynamic_replay_api._ACTIVE_COMPLETIONS.get(id(parts.completion)) is None
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    fresh = _parts(monkeypatch, dynamic=True, runtime=parts.runtime)
    owner = _run_dynamic(fresh)
    owner.abort()


@pytest.mark.parametrize("registry", ("handoff", "completion", "runtime"))
def test_dynamic_gate3_postfinal_failure_preserves_foreign_registries(monkeypatch, registry):
    parts = _parts(monkeypatch, dynamic=True)
    foreign_owner = _Handle()
    foreign = object()
    observed = []

    def physical(*_args, **_kwargs):
        if registry == "handoff":
            dynamic_replay_api._ACTIVE_GRADIENT_HANDOFFS[id(parts.handoff)] = foreign
        elif registry == "completion":
            dynamic_replay_api._ACTIVE_COMPLETIONS[id(parts.completion)] = foreign
        else:
            foreign_runtime = (parts.runtime, weakref.ref(foreign_owner))
            replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign_runtime
            parts.allocator.release_callback = lambda: observed.append(
                replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
            )
        raise RuntimeError("physical failed")

    monkeypatch.setattr(api, "_execute_validated_dynamic_bridge_exchange", physical)
    with pytest.raises(MdpTaskFatalError, match="physical encoder-gradient route"):
        _run_dynamic(parts)
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    if registry == "handoff":
        assert dynamic_replay_api._ACTIVE_GRADIENT_HANDOFFS[id(parts.handoff)] is foreign
        del dynamic_replay_api._ACTIVE_GRADIENT_HANDOFFS[id(parts.handoff)]
    elif registry == "completion":
        assert dynamic_replay_api._ACTIVE_COMPLETIONS[id(parts.completion)] is foreign
        del dynamic_replay_api._ACTIVE_COMPLETIONS[id(parts.completion)]
    else:
        assert observed and all(observed)
        del replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]


@pytest.mark.parametrize(
    ("callback", "message"),
    (
        ("acquire", "dynamic encoder gradient route retains exact replay provenance"),
        ("release", "encoder gradient replay lifecycle retains exact provenance"),
    ),
)
def test_dynamic_gate3_rejects_mutated_operations_and_uses_trusted_cleanup(
    monkeypatch, callback, message
):
    parts = _parts(monkeypatch, dynamic=True)
    calls = []

    def bomb(*_args, **_kwargs):
        calls.append(callback)
        raise AssertionError("mutable operation callback ran")

    setattr(parts.operations, callback, bomb)
    with pytest.raises(MdpStateError, match=message) as caught:
        _run_dynamic(parts)
    assert calls == []
    assert getattr(caught.value, "__notes__", ()) == ()
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    assert all(
        sum(released is expected for released in parts.allocator.released) == 1
        for expected in (parts.leaf, *parts.predecessor_buffers)
    )
    assert api._ACTIVE_OWNERS == {}
    assert api._ACTIVE_RECEIPTS == {}
    assert dynamic_replay_api._ACTIVE_GRADIENT_HANDOFFS.get(id(parts.handoff)) is None
    assert dynamic_replay_api._ACTIVE_COMPLETIONS.get(id(parts.completion)) is None
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    fresh = _parts(monkeypatch, dynamic=True, runtime=parts.runtime)
    owner = _run_dynamic(fresh)
    owner.abort()


@pytest.mark.parametrize("field", ("_owner", "_owner_entry", "_lifecycle", "authority"))
def test_dynamic_gate3_rejects_completion_provenance_mutation(monkeypatch, field):
    parts = _parts(monkeypatch, dynamic=True)
    object.__setattr__(parts.completion, field, object())
    with pytest.raises(
        MdpStateError, match="dynamic decoder gradient handoff retains sealed resources"
    ):
        _run_dynamic(parts)
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    assert all(
        any(released is expected for released in parts.allocator.released)
        for expected in (parts.leaf, *parts.predecessor_buffers)
    )
    assert dynamic_replay_api._ACTIVE_GRADIENT_HANDOFFS.get(id(parts.handoff)) is None
    assert dynamic_replay_api._ACTIVE_COMPLETIONS.get(id(parts.completion)) is None
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    fresh = _parts(monkeypatch, dynamic=True, runtime=parts.runtime)
    owner = _run_dynamic(fresh)
    owner.abort()


def test_dynamic_gate3_foreign_completion_argument_is_nonconsuming(monkeypatch):
    parts = _parts(monkeypatch, dynamic=True)
    foreign = copy.copy(parts.completion)
    with pytest.raises(MdpStateError, match="exact authority and completion"):
        api.run_repeated_d4_dynamic_encoder_gradient(parts.handoff, parts.authority, foreign)
    assert parts.handoff.require() is parts.handoff
    owner = _run_dynamic(parts)
    owner.abort()


def test_dynamic_gate3_final_world_mutation_never_launches_physical(monkeypatch):
    parts = _parts(monkeypatch, dynamic=True)
    foreign = object()

    def runner(_binding, _authority, **kwargs):
        candidate = kwargs["prepare"]()
        parts.events.append("world0")
        kwargs["domain_collective"](candidate)
        parts.events.extend(("domain", "world1"))
        dynamic_replay_api._ACTIVE_COMPLETIONS[id(parts.completion)] = foreign
        return candidate

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpTaskFatalError, match="physical encoder-gradient route"):
        _run_dynamic(parts)
    assert parts.physical_calls == []
    assert dynamic_replay_api._ACTIVE_COMPLETIONS[id(parts.completion)] is foreign
    assert parts.handle.calls == 1
    assert parts.binding_owner.calls == 1
    del dynamic_replay_api._ACTIVE_COMPLETIONS[id(parts.completion)]


def test_dynamic_owner_cleanup_uses_immutable_mode_and_release_anchor(monkeypatch):
    parts = _parts(monkeypatch, dynamic=True)
    owner = _run_dynamic(parts)
    provenance = owner._trusted[16][4]
    calls = []

    def bomb(_tensor):
        calls.append("bomb")
        raise AssertionError("mutable release callback ran")

    object.__setattr__(provenance, "mode", replay_api)
    parts.operations.release = bomb
    primary = RuntimeError("later failure")
    owner.abort(primary)
    assert calls == []
    assert any("integrity validation failed" in note for note in primary.__notes__)
    assert all(
        any(released is expected for released in parts.allocator.released)
        for expected in (parts.leaf, *parts.predecessor_buffers)
    )


def test_foreign_inputs_are_nonconsuming_and_owner_abort_preserves_primary(monkeypatch):
    parts = _parts(monkeypatch)
    for authority, completion in (
        (copy.copy(parts.authority), parts.completion),
        (parts.authority, copy.copy(parts.completion)),
    ):
        with pytest.raises(MdpStateError, match="exact authority and completion"):
            api.run_repeated_d4_encoder_gradient(parts.handoff, authority, completion)
        assert parts.handoff.require() is parts.handoff
    owner = _run(parts)
    primary = MdpStateError("later failure")
    parts.allocator.fail_release = True
    owner.abort(primary)
    assert any("buffer release error" in note for note in primary.__notes__)
    with pytest.raises(MdpStateError, match="owner is retired"):
        owner.abort(primary)
