# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Publication-consuming dynamic Gate2 replay ownership contracts."""

import gc
import weakref
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_encoder_forward as forward_api
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import dynamic_bridge_split_sizes
from megatron.core.mdp.dynamic_cp_bridge_transport import prepare_dynamic_bridge_exchange
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_d4_authority_construction import (
    build_repeated_d4_joint_iteration_authority,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
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
    EncoderWorkEstimate,
)
from megatron.core.mdp.dynamic_cp_routing import decoder_payload_split_sizes
from megatron.core.mdp.dynamic_cp_transport import prepare_decoder_payload_bundle
from megatron.core.mdp.errors import MdpStateError, MdpTaskFatalError
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d4_dynamic_decoder_replay")


class _Group:
    def __init__(self, ranks, rank):
        self._ranks = ranks
        self._rank = rank

    def size(self):
        return len(self._ranks)

    def rank(self):
        return self._ranks.index(self._rank)


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


def _source_window(*, text_only=False, sample_count=4):
    samples = []
    items = []
    packets = []
    for index in range(sample_count):
        sample_id = GlobalSampleId(0, index)
        item_id = GlobalVisionItemId(0, index)
        vision = () if text_only else (EncoderVisionItemMetadata(item_id, sample_id, 0),)
        samples.append(DecoderSampleMetadata(sample_id, 4, 4, vision))
        if vision:
            items.append(DecoderVisionItemMetadata(item_id, sample_id, 0, (1, 1, 1), 1, (1,)))
        tensor = torch.arange(index * 4, index * 4 + 4, device="cuda").view(1, 4)
        spec = DecoderTensorFieldSpec("input_ids", tensor.dtype, tuple(tensor.shape), "cuda")
        packets.append(
            DecoderPayloadPacket(
                DECODER_EXECUTION_SCHEMA_VERSION,
                sample_id,
                4,
                4,
                DecoderPayloadHeaderV1(
                    DECODER_EXECUTION_SCHEMA_VERSION, 0, index, 4, 4, 1, 1, -1
                ).to_wire_tuple(),
                (spec,),
                MappingProxyType({"input_ids": tensor}),
                ("position_ids",),
            )
        )
    return finalize_decoder_source_window(
        source_dp_lane=0, samples=tuple(samples), items=tuple(items), packets=tuple(packets)
    )


def _authority_and_binding(*, cp_size=2, rank=0, text_only=False, sample_count=4):
    binding = _make_repeated_d4_group_binding(
        world_group=_Group(tuple(range(8)), rank),
        domain_group=_Group((0, 1, 2, 3), rank),
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=torch.device("cuda", 0),
        timeout_seconds=1.0,
        group_ranks_getter=lambda group: group._ranks,
        status_gather_factory=lambda **_: lambda *_args, **_kwargs: None,
    )
    source = _source_window(text_only=text_only, sample_count=sample_count)
    metadata = DecoderMetadataGatherResult(
        build_decoder_global_manifest((source.metadata_manifest(),)), {0: 0}
    )
    authority = build_repeated_d4_joint_iteration_authority(
        binding,
        metadata,
        decoder_max_seqlen_per_rank=4,
        decoder_minimum_cp_size=1,
        decoder_solver=_Solver(cp_size),
        encoder_max_seqlen_per_rank=max(4, sample_count),
        encoder_minimum_cp_size=1,
        encoder_workload_query=lambda items, group_size: EncoderWorkEstimate(
            max(1, len(items)), group_size
        ),
        bridge_width=2,
        bridge_dtype=torch.float32,
    )
    return source, binding, authority


def _transport(source, authority, *, rank):
    payload_sources = {
        entry.key: next(
            packet for packet in source.packets if packet.sample_id == entry.key.sample_id
        ).tensor_fields[entry.key.field_name]
        for entry in authority.payload_ledger.entries
        if entry.src_global_rank == rank
    }
    buffers = {}
    for dtype in tuple(dict.fromkeys(entry.dtype for entry in authority.payload_ledger.entries)):
        input_splits, output_splits = decoder_payload_split_sizes(
            authority.payload_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            source_rank_by_lane=authority.source_rank_by_lane,
            participant_ranks=authority.participant_ranks,
            dtype=dtype,
            global_rank=rank,
        )
        buffers[dtype] = (
            torch.empty(sum(input_splits), dtype=dtype, device="cuda"),
            torch.empty(sum(output_splits), dtype=dtype, device="cuda"),
        )
    payload = prepare_decoder_payload_bundle(
        authority.payload_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        source_rank_by_lane=authority.source_rank_by_lane,
        participant_ranks=authority.participant_ranks,
        global_rank=rank,
        local_tensors=payload_sources,
        buffers_by_dtype=buffers,
    )
    input_splits, output_splits = dynamic_bridge_split_sizes(
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
    local = {
        entry.key: torch.full(
            (authority.output_rows_by_item[entry.key.item_id], authority.bridge_width),
            float(entry.key.item_id.local_item_id + 1),
            device="cuda",
        )
        for entry in authority.embedding_ledger.entries
        if entry.src_global_rank == rank
    }
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
        local_tensors=local,
        send_buffer=torch.empty(sum(input_splits), device="cuda"),
        receive_buffer=torch.empty(sum(output_splits), device="cuda"),
    )
    for index, tensor in enumerate(embedding.received_tensors.values(), start=1):
        tensor.fill_(index)
    return payload, embedding


def _rebuild(calls):
    def rebuild(manifest, assignment, *, packets, key, cp_group, cp_partition_mode):
        calls.append((manifest, assignment, packets, key, cp_group, cp_partition_mode))
        samples = {sample.sample_id: sample for sample in manifest.samples}
        items = {item.item_id: item for item in manifest.items}
        vision = []
        offset = 0
        for local_sample, sample_id in enumerate(assignment.sample_ids):
            sample = samples[sample_id]
            for planned in sample.vision_items:
                item = items[planned.item_id]
                vision.append(
                    MdpMicrobatchVisionRecord(
                        item.item_id,
                        local_sample,
                        item.image_ordinal,
                        item.grid_thw,
                        item.output_rows,
                        tuple(offset + value for value in item.decoder_offsets),
                    )
                )
            offset += sample.padded_seqlen
        cu = torch.tensor((0, offset), dtype=torch.int32, device="cuda")
        return MdpMicrobatchRecord(
            key.microbatch_index,
            not vision,
            tuple(vision),
            PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu.clone(),
                cu_seqlens_q_padded=cu.clone(),
                cu_seqlens_kv_padded=cu.clone(),
                max_seqlen_q=offset,
                max_seqlen_kv=offset,
                total_tokens=offset,
                local_cp_size=assignment.local_cp_size,
                cp_group=cp_group,
                cp_partition_mode=cp_partition_mode,
            ),
            MappingProxyType({"packets": packets}),
        )

    return rebuild


class _Allocator:
    def __init__(self):
        self.acquired = []
        self.released = []
        self.fail_at = None
        self.callback = None
        self.acquire_callback = None
        self.alias = None
        self.fail_release = False

    def acquire(self, *, rows, width, dtype, device, tag):
        if self.fail_at == len(self.acquired):
            raise RuntimeError("leaf acquire failed")
        value = (
            self.alias
            if self.alias is not None
            else torch.empty((rows, width), dtype=dtype, device=device)
        )
        self.acquired.append((tag, value))
        if self.acquire_callback is not None:
            self.acquire_callback()
        return value

    def release(self, value):
        self.released.append(value)
        if self.callback is not None:
            self.callback()
        if self.fail_release:
            raise RuntimeError("release failed")


class _Handle:
    def __init__(self):
        self.calls = 0

    def release_forward_only(self):
        self.calls += 1


class _BindingOwner:
    def __init__(self):
        self.calls = []

    def restore(self, primary=None):
        self.calls.append(primary)


def _parts(monkeypatch, *, runtime=None, cp_size=2, text_only=False, selected=True, sample_count=4):
    source, binding, authority = _authority_and_binding(
        cp_size=cp_size, text_only=text_only, sample_count=sample_count
    )
    rank = binding.global_rank
    payload, embedding = _transport(source, authority, rank=rank)
    allocator = _Allocator() if runtime is None else runtime.allocator
    runtime = runtime or SimpleNamespace(device=torch.device("cuda", 0), allocator=allocator)
    handle = _Handle()
    binding_owner = _BindingOwner()
    old_buffers = tuple(
        buffer
        for exchange in payload.exchanges
        for buffer in (exchange.send_buffer, exchange.receive_buffer)
    ) + (embedding.send_buffer, embedding.receive_buffer)
    encoder_ddp = SimpleNamespace(module=object())
    operations = forward_api._D4EncoderForwardOperations(
        allocator=allocator,
        acquire=allocator.acquire,
        release=allocator.release,
        device=runtime.device,
        params_dtype=torch.float32,
        encoder_domain=SimpleNamespace(encoder_ddp=encoder_ddp),
        encoder_ddp=encoder_ddp,
        raw_encoder=encoder_ddp.module,
        zero_grad=lambda: None,
        bind=lambda *_args, **_kwargs: None,
        encode=lambda *_args, **_kwargs: None,
        payload_width=1,
        bridge_width=authority.bridge_width,
        bridge_dtype=authority.bridge_dtype,
        _seal=forward_api._OPERATIONS_SEAL,
    )
    trusted = (
        runtime,
        authority,
        binding,
        (rank,) if selected else (),
        object() if selected else None,
        object() if selected else None,
        payload,
        embedding,
        object() if selected else None,
        MappingProxyType({}),
        handle if selected else None,
        text_only,
        selected,
        selected,
        binding_owner if selected else None,
        old_buffers,
        operations,
    )
    publication = forward_api._D4EncoderPublicationOwner(trusted)
    reference = weakref.ref(publication)
    forward_api._ACTIVE_PUBLICATIONS[id(publication)] = (reference, *trusted)
    forward_api._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    api = _api()
    events = []

    def runner(_binding, _authority, **kwargs):
        assert kwargs["gate_id"] == 2
        try:
            prepared = kwargs["prepare"]()
        except BaseException as error:
            events.append("world0")
            raise MdpStateError(
                "MDP: repeated-D4 WORLD preparation gate accepted a local error."
            ) from error
        events.append("world0")
        events.append("domain-status")
        events.append("world1")
        events.append("domain-callback")
        return kwargs["domain_collective"](prepared)

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    groups = {}

    def group_getter(*, group_size):
        candidates = {
            assignment.endpoint_ranks
            for microbatch in authority.plan.microbatches
            for assignment in microbatch.assignments
            if rank in assignment.endpoint_ranks and assignment.local_cp_size == group_size
        }
        assert len(candidates) == 1
        ranks = candidates.pop()
        return groups.setdefault(ranks, _Group(ranks, rank))

    return SimpleNamespace(
        api=api,
        source=source,
        authority=authority,
        binding=binding,
        publication=publication,
        runtime=runtime,
        allocator=allocator,
        handle=handle,
        binding_owner=binding_owner,
        old_buffers=old_buffers,
        events=events,
        group_getter=group_getter,
    )


def _run(parts, *, rebuild=None):
    return parts.api.run_repeated_d4_dynamic_decoder_replay(
        parts.publication,
        parts.authority,
        decoder_group_getter=parts.group_getter,
        decoder_group_ranks_getter=lambda group: group._ranks,
        rebuild_microbatch=_rebuild([]) if rebuild is None else rebuild,
        cp_partition_mode="contiguous",
    )


@pytest.mark.parametrize("mutation", ("active_entry", "authority"))
def test_dynamic_gate2_rejects_publication_substitution_before_runner_attempt(
    monkeypatch, mutation
):
    parts = _parts(monkeypatch)
    runner_calls = []
    attempt_calls = []
    original_entry = forward_api._ACTIVE_PUBLICATIONS[id(parts.publication)]

    def runner(*_args, **_kwargs):
        runner_calls.append(True)

    def begin_attempt(*_args, **_kwargs):
        attempt_calls.append(True)

    monkeypatch.setattr(parts.api, "run_repeated_d4_authority_collective", runner)
    monkeypatch.setattr(type(parts.binding), "begin_attempt", begin_attempt)
    if mutation == "active_entry":
        forged = list(original_entry)
        forged[2] = object()
        forward_api._ACTIVE_PUBLICATIONS[id(parts.publication)] = tuple(forged)
        expected = "publication owner retains sealed fields"
        supplied_authority = parts.authority
    else:
        expected = "exact iteration authority"
        supplied_authority = object()

    with pytest.raises(MdpStateError, match=expected):
        parts.api.run_repeated_d4_dynamic_decoder_replay(
            parts.publication,
            supplied_authority,
            decoder_group_getter=parts.group_getter,
            decoder_group_ranks_getter=lambda group: group._ranks,
            rebuild_microbatch=_rebuild([]),
            cp_partition_mode="contiguous",
        )

    assert runner_calls == []
    assert attempt_calls == []
    forward_api._ACTIVE_PUBLICATIONS[id(parts.publication)] = original_entry
    parts.publication.abort()


@pytest.mark.parametrize("cp_size", (1, 2, 4))
def test_dynamic_gate2_claims_publication_and_owns_exact_ready_leaves(monkeypatch, cp_size):
    parts = _parts(monkeypatch, cp_size=cp_size)
    owner = _run(parts)

    assert owner.require() is owner
    assert owner.authority is parts.authority
    assert owner.binding is parts.binding
    assert owner.ready.records is owner.records
    assert owner.ready.embedding_leaves is owner.embedding_leaves
    assert all(value.assignment.local_cp_size == cp_size for value in owner.ready.assignments)
    assert parts.events == ["world0", "domain-status", "world1", "domain-callback"]
    with pytest.raises(MdpStateError, match="publication owner is retired"):
        parts.publication.require()
    assert forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is owner
    acquired = tuple(value for _, value in parts.allocator.acquired)

    owner.abort()

    assert parts.allocator.released == [*acquired, *parts.old_buffers]
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS


def test_dynamic_gate2_reuses_one_captured_group_across_microbatches(monkeypatch):
    parts = _parts(monkeypatch, cp_size=1, sample_count=8)
    owner = _run(parts)

    assert len(owner.ready.assignments) > 1
    assert len({id(local.cp_group) for local in owner.ready.assignments}) == 1
    owner.abort()


@pytest.mark.parametrize(
    ("text_only", "selected"), ((True, False), (False, False)), ids=("text", "nonselected")
)
def test_dynamic_gate2_preserves_text_and_nonselected_publication_roles(
    monkeypatch, text_only, selected
):
    parts = _parts(monkeypatch, text_only=text_only, selected=selected)
    owner = _run(parts)
    assert all(record.text_only is text_only for record in owner.records)
    assert (not owner.embedding_leaves) is text_only
    owner.abort()
    assert parts.handle.calls == int(selected)
    assert len(parts.binding_owner.calls) == int(selected)


def test_dynamic_gate2_materialization_failure_cleans_pending_owner_and_retries(monkeypatch):
    parts = _parts(monkeypatch)
    primary = RuntimeError("codec failed")

    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted") as caught:
        _run(parts, rebuild=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary))

    assert caught.value.__cause__ is primary
    assert parts.events == ["world0"]
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert parts.allocator.released == [
        *(value for _, value in parts.allocator.acquired),
        *parts.old_buffers,
    ]
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    owner = _run(fresh)
    owner.abort()


def test_dynamic_gate2_rejection_cleans_before_any_later_stage(monkeypatch):
    parts = _parts(monkeypatch)
    primary = MdpStateError("first WORLD rejected")

    def reject(_binding, _authority, **kwargs):
        parts.events.append("world0")
        kwargs["prepare"]()
        raise primary

    monkeypatch.setattr(parts.api, "run_repeated_d4_authority_collective", reject)
    with pytest.raises(MdpStateError, match="first WORLD rejected") as caught:
        _run(parts)
    assert caught.value is primary
    assert parts.events == ["world0"]
    assert parts.handle.calls == 1
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS


def test_dynamic_gate2_final_world_rejection_never_enters_post_world_callback(monkeypatch):
    parts = _parts(monkeypatch)
    primary = MdpStateError("final WORLD rejected")

    def reject(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        parts.events.extend(("world0", "domain-status", "world1"))
        raise primary

    monkeypatch.setattr(parts.api, "run_repeated_d4_authority_collective", reject)
    with pytest.raises(MdpStateError, match="final WORLD rejected") as caught:
        _run(parts)
    assert caught.value is primary
    assert parts.events == ["world0", "domain-status", "world1"]
    assert parts.handle.calls == 1
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS


def test_dynamic_gate2_result_substitution_is_task_fatal_and_cleans(monkeypatch):
    parts = _parts(monkeypatch)

    def substitute(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        return object()

    monkeypatch.setattr(parts.api, "run_repeated_d4_authority_collective", substitute)
    with pytest.raises(MdpTaskFatalError, match="returns its exact owner"):
        _run(parts)
    assert parts.handle.calls == 1
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS


def test_dynamic_gate2_partial_leaf_allocation_releases_every_acquired_leaf(monkeypatch):
    parts = _parts(monkeypatch)
    parts.allocator.fail_at = 1
    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted") as caught:
        _run(parts)
    assert str(caught.value.__cause__) == "leaf acquire failed"
    assert parts.allocator.released == [
        *(value for _, value in parts.allocator.acquired),
        *parts.old_buffers,
    ]
    assert parts.handle.calls == 1
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS


def test_dynamic_gate2_rejects_transport_alias_before_copy_and_cleans(monkeypatch):
    parts = _parts(monkeypatch)
    parts.allocator.alias = next(iter(parts.publication.embedding_bundle.received_tensors.values()))
    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted") as caught:
        _run(parts)
    assert "disjoint from owned buffers" in str(caught.value.__cause__)
    assert parts.allocator.released == [parts.allocator.alias, *parts.old_buffers]
    assert parts.handle.calls == 1


def test_dynamic_gate2_post_world_mutation_is_task_fatal_and_trusted_cleanup(monkeypatch):
    parts = _parts(monkeypatch)

    def runner(_binding, _authority, **kwargs):
        owner = kwargs["prepare"]()
        parts.events.extend(("world0", "domain-status", "world1"))
        object.__setattr__(owner.ready, "records", ())
        return kwargs["domain_collective"](owner)

    monkeypatch.setattr(parts.api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpTaskFatalError, match="post-WORLD validation failed"):
        _run(parts)
    assert parts.handle.calls == 1
    assert parts.allocator.released == [
        *(value for _, value in parts.allocator.acquired),
        *parts.old_buffers,
    ]
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS


def test_dynamic_gate2_cleanup_preserves_foreign_registries_and_reentry_is_one_shot(monkeypatch):
    parts = _parts(monkeypatch)
    api = parts.api
    foreign_owner = (object(),)
    foreign_runtime = (object(), lambda: None)
    reentry_errors = []
    captured = {}
    original_claim = api._D4DynamicDecoderReplayOwner._claim_handoff

    def claim(owner, handoff, entry):
        owner_entry = original_claim(owner, handoff, entry)
        captured.update(owner=owner, entry=owner_entry)
        return owner_entry

    monkeypatch.setattr(api._D4DynamicDecoderReplayOwner, "_claim_handoff", claim)

    def mutate():
        identity, reference = next(iter(api._RETIRED_ESCROWS.items()))
        owner = reference()
        api._ACTIVE_OWNERS[identity] = foreign_owner
        api._PENDING_OWNERS[identity] = foreign_owner
        forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign_runtime
        try:
            owner._abort_from_entry(captured["entry"], RuntimeError("reenter"))
        except BaseException as error:
            reentry_errors.append(error)

    parts.allocator.callback = mutate
    primary = RuntimeError("codec failed")
    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted"):
        _run(parts, rebuild=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary))
    assert api._ACTIVE_OWNERS[next(iter(api._ACTIVE_OWNERS))] is foreign_owner
    assert api._PENDING_OWNERS[next(iter(api._PENDING_OWNERS))] is foreign_owner
    assert forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
    assert len(reentry_errors) == len(parts.allocator.released)
    assert all(isinstance(error, MdpStateError) for error in reentry_errors)
    assert api._RETIRED_ESCROWS[id(captured["owner"])]() is captured["owner"]
    api._ACTIVE_OWNERS.clear()
    api._PENDING_OWNERS.clear()
    forward_api._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))


def test_dynamic_gate2_retired_owner_escrow_releases_resources_and_is_weak(monkeypatch):
    parts = _parts(monkeypatch)
    owner = _run(parts)
    identity = id(owner)
    reference = weakref.ref(owner)

    owner.abort()

    assert parts.allocator.released == [
        *(value for _, value in parts.allocator.acquired),
        *parts.old_buffers,
    ]
    assert parts.api._RETIRED_ESCROWS[identity] is reference or (
        parts.api._RETIRED_ESCROWS[identity]() is owner
    )
    assert isinstance(parts.api._RETIRED_ESCROWS[identity], weakref.ReferenceType)
    del owner
    gc.collect()
    assert reference() is None
    assert identity not in parts.api._RETIRED_ESCROWS


def test_dynamic_gate2_abort_uses_independent_exact_entry_escrow(monkeypatch):
    parts = _parts(monkeypatch)
    owner = _run(parts)
    api = parts.api
    exact = api._OWNER_ESCROWS[id(owner)].entry
    bomb_calls = []

    def bomb(_value):
        bomb_calls.append(_value)

    substituted = list(exact)
    substituted[6] = bomb
    substituted = tuple(substituted)
    api._TRUSTED_OWNERS[id(owner)] = substituted
    primary = RuntimeError("abort primary")

    owner.abort(primary)

    assert bomb_calls == []
    assert parts.allocator.released == [
        *(value for _, value in parts.allocator.acquired),
        *parts.old_buffers,
    ]
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert api._TRUSTED_OWNERS[id(owner)] is substituted
    assert id(owner) not in api._OWNER_ESCROWS
    assert any("integrity validation failed" in note for note in primary.__notes__)
    del api._TRUSTED_OWNERS[id(owner)]


def test_dynamic_gate2_abort_rejects_owner_escrow_callback_injection(monkeypatch):
    parts = _parts(monkeypatch)
    owner = _run(parts)
    api = parts.api
    exact = api._OWNER_ESCROWS[id(owner)].entry
    bomb_calls = []
    reentry_errors = []

    def bomb(_value):
        bomb_calls.append(_value)

    bad = list(exact)
    bad[6] = bomb
    bad = tuple(bad)
    foreign = api._OwnerEscrow(weakref.ref(owner), bad, seal=api._OWNER_ESCROW_SEAL)
    api._OWNER_ESCROWS[id(owner)] = foreign

    def reenter():
        try:
            owner.abort(RuntimeError("reenter"))
        except BaseException as error:
            reentry_errors.append(error)

    parts.allocator.callback = reenter
    owner.abort(RuntimeError("primary"))

    assert bomb_calls == []
    assert parts.allocator.released == [
        *(value for _, value in parts.allocator.acquired),
        *parts.old_buffers,
    ]
    assert api._OWNER_ESCROWS[id(owner)] is foreign
    assert len(reentry_errors) == len(parts.allocator.released)
    assert all(isinstance(error, MdpStateError) for error in reentry_errors)
    del api._OWNER_ESCROWS[id(owner)]


def test_storage_intervals_use_bytes_for_cross_dtype_aliases():
    api = _api()
    storage = torch.empty(32, dtype=torch.uint8, device="cuda")
    int64_view = storage.view(torch.int64)[1:2]
    float32_view = storage[8:16].view(torch.float32)

    assert api._storage_interval(int64_view)[1:] == (8, 16)
    assert api._storage_interval(float32_view)[1:] == (8, 16)
    assert api._overlaps(api._storage_interval(int64_view), api._storage_interval(float32_view))


def test_dynamic_gate2_forged_trusted_tuple_is_rejected_during_prepare(monkeypatch):
    parts = _parts(monkeypatch)
    api = parts.api
    original = api._D4DynamicDecoderReplayOwner._seal_ready

    def forge(owner, entry, ready, leaves):
        sealed = original(owner, entry, ready, leaves)
        owner._trusted = tuple(list(sealed))
        return sealed

    monkeypatch.setattr(api._D4DynamicDecoderReplayOwner, "_seal_ready", forge)
    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted"):
        _run(parts)
    assert parts.events == ["world0"]
    assert parts.handle.calls == 1
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS


def test_dynamic_gate2_forged_trusted_tuple_is_rejected_post_final_world(monkeypatch):
    parts = _parts(monkeypatch)

    def runner(_binding, _authority, **kwargs):
        owner = kwargs["prepare"]()
        parts.events.extend(("world0", "domain-status", "world1"))
        owner._trusted = tuple(list(owner._trusted))
        return kwargs["domain_collective"](owner)

    monkeypatch.setattr(parts.api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpTaskFatalError, match="post-WORLD validation failed"):
        _run(parts)
    assert parts.events == ["world0", "domain-status", "world1"]
    assert parts.handle.calls == 1
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS


def test_consumed_handoff_tombstone_is_removed_when_handoff_dies(monkeypatch):
    parts = _parts(monkeypatch)
    api = parts.api
    original = api._retire_consumed_handoff
    captured = {}

    class Dead:
        pass

    dead = Dead()
    stale = weakref.ref(dead)
    del dead
    gc.collect()
    assert stale() is None

    def retire(handoff, entry):
        captured["identity"] = id(handoff)
        captured["reference"] = weakref.ref(handoff)
        forward_api._RETIRED_REPLAY_HANDOFFS[id(handoff)] = stale
        original(handoff, entry)

    monkeypatch.setattr(api, "_retire_consumed_handoff", retire)
    owner = _run(parts)
    gc.collect()

    assert captured["reference"]() is None
    assert captured["identity"] not in forward_api._RETIRED_REPLAY_HANDOFFS
    owner.abort()


def test_dynamic_gate2_cleanup_failure_is_note_and_fresh_runtime_retries(monkeypatch):
    parts = _parts(monkeypatch)
    parts.allocator.fail_release = True
    primary = RuntimeError("codec failed")
    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted") as caught:
        _run(parts, rebuild=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary))
    assert caught.value.__cause__ is primary
    assert any(
        "suppressed dynamic replay buffer release" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS
    parts.allocator.fail_release = False
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    owner = _run(fresh)
    owner.abort()


def test_dynamic_gate2_resnapshots_authority_after_group_getter(monkeypatch):
    parts = _parts(monkeypatch)
    original = parts.group_getter

    def mutate(**kwargs):
        group = original(**kwargs)
        object.__setattr__(parts.authority, "bridge_width", 3)
        return group

    parts.group_getter = mutate
    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted"):
        _run(parts)
    assert not parts.allocator.acquired
    assert parts.handle.calls == 1


def test_dynamic_gate2_route_mutation_during_acquire_never_rebuilds(monkeypatch):
    parts = _parts(monkeypatch)
    calls = []
    embedding = parts.publication.embedding_bundle

    def mutate():
        object.__setattr__(embedding, "received_tensors", MappingProxyType({}))

    parts.allocator.acquire_callback = mutate
    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted"):
        _run(parts, rebuild=lambda *args, **kwargs: calls.append((args, kwargs)))
    assert calls == []
    assert parts.handle.calls == 1
    assert parts.allocator.released == [
        *(value for _, value in parts.allocator.acquired),
        *parts.old_buffers,
    ]


def test_dynamic_gate2_partial_claim_failure_is_failure_total(monkeypatch):
    parts = _parts(monkeypatch)
    original = parts.api._retire_consumed_handoff
    calls = 0

    def retire_then_raise(handoff, entry):
        nonlocal calls
        calls += 1
        original(handoff, entry)
        if calls == 1:
            raise RuntimeError("claim activation failed")

    monkeypatch.setattr(parts.api, "_retire_consumed_handoff", retire_then_raise)
    with pytest.raises(MdpStateError, match="WORLD preparation gate accepted") as caught:
        _run(parts)
    assert str(caught.value.__cause__) == "claim activation failed"
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert parts.api._TRUSTED_OWNERS == {}
    assert id(parts.runtime) not in forward_api._ACTIVE_RUNTIME_OWNERS
