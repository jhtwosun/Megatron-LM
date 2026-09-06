# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gate2 fixed-CP4 replay ownership and cursor contracts."""

import copy
import weakref
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_encoder_forward as forward_api
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as api
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_bridge_transport import PreparedDynamicBridgeExchange
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _AUTHORITY_SEAL,
    _BINDING_SEAL,
    _RepeatedD4GroupAuthority,
    _RepeatedD4GroupBinding,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    DecoderPayloadHeaderV1,
    DecoderPayloadMetadata,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderCpAssignment,
    DecoderDynamicPlan,
    DecoderMicrobatchPlan,
    DecoderSampleMetadata,
    EncoderVisionItemMetadata,
)
from megatron.core.mdp.dynamic_cp_routing import DecoderPayloadRouteKey
from megatron.core.mdp.dynamic_cp_transport import PreparedDecoderPayloadBundle
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams


class _Group:
    def size(self):
        return 4

    def rank(self):
        return 0


class _Allocator:
    def __init__(self):
        self.acquired = []
        self.released = []
        self.fail_release = False
        self.release_callback = None

    def acquire(self, *, rows, width, dtype, device, tag):
        tensor = torch.empty((rows, width), dtype=dtype, device=device)
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
        self.calls = []
        self.callback = None

    def restore(self, primary=None):
        self.calls.append(primary)
        if self.callback is not None:
            self.callback()
        self.active = False


class _Handle:
    def __init__(self):
        self.consumed = False
        self.calls = 0
        self.callback = None

    def release_forward_only(self):
        self.calls += 1
        if self.callback is not None:
            self.callback()
        self.consumed = True


def _binding(group):
    authority = _RepeatedD4GroupAuthority(
        world_ranks=(0, 1, 2, 3),
        domain_ranks=(0, 1, 2, 3),
        global_rank=0,
        expert_parallel_size=1,
        _world_group=object(),
        _domain_group=group,
        _expert_group=None,
        _device=torch.device("cuda"),
        _timeout_seconds=1.0,
        _status_gather_factory=lambda: (lambda *_args, **_kwargs: None),
        _group_ranks_getter=lambda _group: (0, 1, 2, 3),
        _world_pre_gate=lambda *_args, **_kwargs: None,
        _domain_status=lambda *_args, **_kwargs: None,
        _seal=_AUTHORITY_SEAL,
    )
    return _RepeatedD4GroupBinding(
        world_ranks=(0, 1, 2, 3),
        domain_ranks=(0, 1, 2, 3),
        global_rank=0,
        expert_parallel_size=1,
        _authority=authority,
        _seal=_BINDING_SEAL,
    )


def _parts(
    monkeypatch, *, vision=True, selected=None, microbatches=2, runtime=None, publication=False
):
    if selected is None:
        selected = vision
    group = _Group()
    binding = _binding(group)
    samples = []
    payloads = []
    items = []
    plans = []
    payload_views = {}
    embedding_views = {}
    payload_entries = []
    embedding_entries = []
    for index in range(microbatches):
        sample_id = GlobalSampleId(0, index)
        vision_items = ()
        if vision:
            item_id = GlobalVisionItemId(0, index)
            planned_item = EncoderVisionItemMetadata(item_id, sample_id, 0)
            vision_items = (planned_item,)
            items.append(
                DecoderVisionItemMetadata(
                    item_id=item_id,
                    sample_id=sample_id,
                    image_ordinal=0,
                    grid_thw=(1, 2, 2),
                    output_rows=1,
                    decoder_offsets=(index,),
                )
            )
            embedding_key = DynamicBridgeKey(item_id, 0)
            embedding_views[embedding_key] = torch.full((1, 2), float(index + 1))
            embedding_entries.append(SimpleNamespace(key=embedding_key, dst_global_rank=0))
        samples.append(DecoderSampleMetadata(sample_id, 8, 8, vision_items))
        spec = DecoderTensorFieldSpec("input_ids", torch.int64, (1, 8), "cpu")
        header = DecoderPayloadHeaderV1(1, 0, index, 8, 8, 1, 0, -1)
        payloads.append(
            DecoderPayloadMetadata(sample_id, 8, 8, header.to_wire_tuple(), (spec,), ())
        )
        payload_key = DecoderPayloadRouteKey(sample_id, 0, "input_ids")
        payload_views[payload_key] = torch.arange(8).view(1, 8)
        payload_entries.append(SimpleNamespace(key=payload_key, dst_global_rank=0))
        plans.append(
            DecoderMicrobatchPlan(index, (DecoderCpAssignment((sample_id,), (0, 1, 2, 3)),))
        )
    manifest = DecoderGlobalManifest(tuple(samples), tuple(items), tuple(payloads), b"m" * 16)
    plan = DecoderDynamicPlan(tuple(samples), (0, 1, 2, 3), 8, 4, tuple(plans), b"p" * 16)
    authority = SimpleNamespace(
        plan=plan,
        global_manifest=manifest,
        participant_ranks=(0, 1, 2, 3),
        payload_ledger=SimpleNamespace(entries=tuple(payload_entries)),
        embedding_ledger=SimpleNamespace(entries=tuple(embedding_entries)),
        bridge_width=2,
        bridge_dtype=torch.float32,
    )
    payload = PreparedDecoderPayloadBundle(
        b"a" * 16,
        0,
        (0, 1, 2, 3),
        (torch.int64,),
        torch.device("cpu"),
        (),
        MappingProxyType(payload_views),
    )
    embedding = PreparedDynamicBridgeExchange(
        BridgePhase.EMBEDDING,
        torch.float32,
        0,
        (0, 1, 2, 3),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        b"e" * 16,
        torch.empty(0),
        torch.empty(0),
        MappingProxyType(embedding_views),
    )
    allocator = _Allocator() if runtime is None else runtime.allocator
    operations = SimpleNamespace(acquire=allocator.acquire, release=allocator.release)
    runtime = runtime or SimpleNamespace(device=torch.device("cpu"), allocator=allocator)
    binding_owner = _BindingOwner() if selected else None
    handle = _Handle() if selected else None
    old_buffers = (torch.empty(0), torch.empty(0))
    trusted = (
        runtime,
        authority,
        binding,
        (0, 1) if selected else (),
        object() if selected else None,
        object() if selected else None,
        payload,
        embedding,
        object() if selected else None,
        MappingProxyType({}),
        handle,
        not vision,
        selected,
        selected,
        binding_owner,
        old_buffers,
        operations,
    )
    if publication:
        owner = forward_api._D4EncoderPublicationOwner(trusted)
        reference = weakref.ref(owner)
        forward_api._ACTIVE_PUBLICATIONS[id(owner)] = (reference, *trusted)
        forward_api._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
        handoff = None
    else:
        owner = None
        handoff = forward_api._D4EncoderReplayHandoff(trusted)
        reference = weakref.ref(handoff)
        forward_api._ACTIVE_REPLAY_HANDOFFS[id(handoff)] = (reference, *trusted, False)
        forward_api._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    events = []

    monkeypatch.setattr(api, "_snapshot_local_authority", lambda b, a: a)
    monkeypatch.setattr(forward_api, "_snapshot_local_authority", lambda b, a: a)
    monkeypatch.setattr(api, "validate_decoder_dynamic_plan", lambda value: value)
    monkeypatch.setattr(api, "validate_decoder_global_manifest", lambda value: value)
    monkeypatch.setattr(api, "validate_prepared_decoder_payload_bundle", lambda value: value)
    monkeypatch.setattr(api, "validate_prepared_dynamic_bridge_exchange", lambda value: value)
    monkeypatch.setattr(api, "_validate_repeated_d4_group_binding", lambda value: value._authority)

    def runner(_binding, _authority, **kwargs):
        events.append(("gate", kwargs["gate_id"], kwargs["byte_generator"]))
        prepared = kwargs["prepare"]()
        events.append("world0")
        prepared = kwargs["domain_collective"](prepared)
        events.extend(("domain", "world1"))
        return prepared

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)

    calls = []

    def rebuild(actual_manifest, assignment, *, packets, key, cp_group, cp_partition_mode):
        calls.append((actual_manifest, assignment, packets, key, cp_group, cp_partition_mode))
        sample = next(
            value
            for value in actual_manifest.samples
            if value.sample_id == assignment.sample_ids[0]
        )
        vision_records = tuple(
            MdpMicrobatchVisionRecord(
                global_item_id=item.item_id,
                sample_id=0,
                image_ordinal=item.image_ordinal,
                grid_thw=(1, 2, 2),
                output_rows=1,
                decoder_positions=(key.microbatch_index,),
            )
            for item in sample.vision_items
        )
        boundary = torch.tensor([0, 8], dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=boundary,
            cu_seqlens_kv=boundary.clone(),
            cu_seqlens_q_padded=boundary.clone(),
            cu_seqlens_kv_padded=boundary.clone(),
            max_seqlen_q=8,
            max_seqlen_kv=8,
            local_cp_size=4,
            cp_group=group,
            total_tokens=8,
            cp_partition_mode=cp_partition_mode,
        )
        return MdpMicrobatchRecord(
            key.microbatch_index,
            not vision_records,
            vision_records,
            packed,
            MappingProxyType({"input_ids": packets[0].tensor_fields["input_ids"]}),
        )

    return SimpleNamespace(
        runtime=runtime,
        authority=authority,
        publication=owner,
        handoff=handoff,
        binding=binding,
        payload=payload,
        embedding=embedding,
        allocator=allocator,
        binding_owner=binding_owner,
        handle=handle,
        old_buffers=old_buffers,
        rebuild=rebuild,
        calls=calls,
        events=events,
    )


def _run(parts, *, byte_generator=None):
    return api.run_repeated_d4_fixed_decoder_replay(
        parts.handoff,
        parts.authority,
        rebuild_microbatch=parts.rebuild,
        cp_partition_mode="zigzag",
        byte_generator=byte_generator,
    )


def _run_from_publication(parts, *, byte_generator=None):
    return api._run_repeated_d4_fixed_decoder_replay_from_publication(
        parts.publication,
        parts.authority,
        rebuild_microbatch=parts.rebuild,
        cp_partition_mode="zigzag",
        byte_generator=byte_generator,
    )


def _complete(parts):
    owner = _run(parts)
    cursor = owner.replay_cursor()
    for _record in owner.records:
        next(cursor)
    token = torch.tensor(1.0)
    owner.capture_global_num_tokens(token)
    schedule_return = owner.mark_schedule_returned(cursor)
    completion = owner.prepare_completion(cursor, schedule_return)
    return owner, cursor, token, schedule_return, completion


def test_publication_replay_claim_and_successor_transfer_are_atomic(monkeypatch):
    parts = _parts(monkeypatch, publication=True)
    generator = object()

    owner = _run_from_publication(parts, byte_generator=generator)

    assert owner.require() is owner
    assert parts.events == [("gate", 2, generator), "world0", "domain", "world1"]
    assert forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is owner
    with pytest.raises(MdpStateError, match="publication owner is retired"):
        parts.publication.require()
    owner.abort()
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1


@pytest.mark.parametrize("mutated", (False, True))
def test_publication_replay_preclaim_failure_retires_exact_owner_once_and_retries(
    monkeypatch, mutated
):
    parts = _parts(monkeypatch, publication=True)
    if mutated:
        parts.publication.payload_bundle = object()
        authority = parts.authority
        message = "retains sealed fields"
    else:
        authority = copy.copy(parts.authority)
        message = "exact iteration authority"

    with pytest.raises(MdpStateError, match=message):
        api._run_repeated_d4_fixed_decoder_replay_from_publication(
            parts.publication,
            authority,
            rebuild_microbatch=parts.rebuild,
            cp_partition_mode="zigzag",
        )

    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert tuple(parts.allocator.released) == parts.old_buffers
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    with pytest.raises(MdpStateError, match="publication owner is retired"):
        parts.publication.abort()

    fresh = _parts(monkeypatch, runtime=parts.runtime, publication=True)
    owner = _run_from_publication(fresh)
    owner.abort()


def test_publication_replay_preconsume_failure_uses_handoff_cleanup_once(monkeypatch):
    parts = _parts(monkeypatch, publication=True)
    original = api._build_candidate
    primary = RuntimeError("before replay consumption")
    seen = []

    def fail(handoff, *_args, **_kwargs):
        seen.append(handoff)
        handoff.payload_bundle = object()
        raise primary

    monkeypatch.setattr(api, "_build_candidate", fail)
    with pytest.raises(RuntimeError) as raised:
        _run_from_publication(parts)
    assert raised.value is primary
    assert len(seen) == 1
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert tuple(parts.allocator.released) == parts.old_buffers
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    monkeypatch.setattr(api, "_build_candidate", original)
    fresh = _parts(monkeypatch, runtime=parts.runtime, publication=True)
    owner = _run_from_publication(fresh)
    owner.abort()


def test_publication_replay_postconsume_failure_is_not_cleaned_twice(monkeypatch):
    parts = _parts(monkeypatch, publication=True)
    primary = MdpStateError("Gate2 rejected")

    def reject(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", reject)
    with pytest.raises(MdpStateError) as raised:
        _run_from_publication(parts)
    assert raised.value is primary
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    leaves = tuple(tensor for _tag, tensor in parts.allocator.acquired)
    assert tuple(parts.allocator.released) == (*reversed(leaves), *parts.old_buffers)
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


@pytest.mark.parametrize(
    "mutation", ("raise", "delete", "substitute", "substitute_runtime", "mutate")
)
def test_publication_replay_activation_failure_cleans_exact_successor(monkeypatch, mutation):
    parts = _parts(monkeypatch, publication=True)
    original = api._D4FixedDecoderReplayOwner._activate_prepared
    primary = RuntimeError("activation failed after registry installation")
    foreign_owner = _Handle()
    foreign = (parts.runtime, weakref.ref(foreign_owner))
    observed_foreign = []

    def fail(owner, handoff):
        original(owner, handoff)
        if mutation == "delete":
            del api._ACTIVE_OWNERS[id(owner)]
        elif mutation == "substitute":
            api._ACTIVE_OWNERS[id(owner)] = foreign
        elif mutation == "substitute_runtime":
            forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign
            parts.allocator.release_callback = lambda: observed_foreign.append(
                forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is foreign
            )
        elif mutation == "mutate":
            owner.records = ()
            return
        raise primary

    monkeypatch.setattr(api._D4FixedDecoderReplayOwner, "_activate_prepared", fail)
    error_type = MdpStateError if mutation == "mutate" else MdpTaskFatalError
    expected = "sealed resources" if mutation == "mutate" else "activates after Gate2"
    with pytest.raises(error_type, match=expected) as raised:
        _run_from_publication(parts)
    if mutation != "mutate":
        assert raised.value.__cause__ is primary
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    leaves = tuple(tensor for _tag, tensor in parts.allocator.acquired)
    expected_releases = (*leaves, *parts.old_buffers)
    assert len(parts.allocator.released) == len(expected_releases)
    assert all(
        actual is expected
        for actual, expected in zip(parts.allocator.released, expected_releases, strict=True)
    )
    if mutation == "substitute_runtime":
        assert observed_foreign and all(observed_foreign)
        parts.allocator.release_callback = None
        assert forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign
        del forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]
    else:
        assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    if mutation == "substitute":
        assert api._ACTIVE_OWNERS.pop(next(iter(api._ACTIVE_OWNERS))) is foreign
    else:
        assert api._ACTIVE_OWNERS == {}

    monkeypatch.setattr(api._D4FixedDecoderReplayOwner, "_activate_prepared", original)
    fresh = _parts(monkeypatch, runtime=parts.runtime, publication=True)
    owner = _run_from_publication(fresh)
    owner.abort()


def test_publication_replay_cleanup_reentry_observes_retired_owner(monkeypatch):
    parts = _parts(monkeypatch, publication=True)
    reentry = []
    primary = MdpStateError("Gate2 rejected")

    def reject(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", reject)

    def reenter():
        parts.allocator.release_callback = None
        try:
            _run_from_publication(parts)
        except MdpStateError as error:
            reentry.append(error)

    parts.allocator.release_callback = reenter
    with pytest.raises(MdpStateError) as raised:
        _run_from_publication(parts)
    assert raised.value is primary
    assert len(reentry) == 1
    assert "publication owner is retired" in str(reentry[0])
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


@pytest.mark.parametrize(
    "mutation", ("delete_handoff", "substitute_handoff", "delete_runtime", "substitute_runtime")
)
def test_publication_replay_escrow_survives_handoff_registry_mutation(monkeypatch, mutation):
    parts = _parts(monkeypatch, publication=True)
    primary = RuntimeError("handoff callback failed")
    foreign_owner = _Handle()
    foreign_runtime = (parts.runtime, weakref.ref(foreign_owner))
    foreign_handoff = (object(), object())
    observed_foreign = []
    seen_handoff = []

    def fail(_binding, _authority, **kwargs):
        handoff = forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]()
        seen_handoff.append(handoff)
        if mutation == "delete_handoff":
            del forward_api._ACTIVE_REPLAY_HANDOFFS[id(handoff)]
        elif mutation == "substitute_handoff":
            forward_api._ACTIVE_REPLAY_HANDOFFS[id(handoff)] = foreign_handoff
        elif mutation == "delete_runtime":
            del forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]
        else:
            forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign_runtime
            parts.allocator.release_callback = lambda: observed_foreign.append(
                forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is foreign_runtime
            )
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", fail)
    with pytest.raises(RuntimeError) as raised:
        _run_from_publication(parts)
    assert raised.value is primary
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert tuple(parts.allocator.released) == parts.old_buffers
    assert len(seen_handoff) == 1
    if mutation == "substitute_handoff":
        assert forward_api._ACTIVE_REPLAY_HANDOFFS[id(seen_handoff[0])] is foreign_handoff
        del forward_api._ACTIVE_REPLAY_HANDOFFS[id(seen_handoff[0])]
    else:
        assert id(seen_handoff[0]) not in forward_api._ACTIVE_REPLAY_HANDOFFS
    if mutation == "substitute_runtime":
        assert observed_foreign and all(observed_foreign)
        parts.allocator.release_callback = None
        assert forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
        del forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]
    else:
        assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    fresh = _parts(monkeypatch, runtime=parts.runtime, publication=True)
    owner = _run_from_publication(fresh)
    owner.abort()


def test_gate2_transfers_exact_resources_and_exposes_monotonic_cursor(monkeypatch):
    parts = _parts(monkeypatch)
    generator = object()
    owner = _run(parts, byte_generator=generator)

    assert owner.require() is owner
    assert parts.events == [("gate", 2, generator), "world0", "domain", "world1"]
    assert tuple(call[0] is parts.authority.global_manifest for call in parts.calls) == (True, True)
    assert all(call[4] is parts.binding.domain_group for call in parts.calls)
    assert all(call[5] == "zigzag" for call in parts.calls)
    for index, call in enumerate(parts.calls):
        packet = call[2][0]
        assert packet.metadata() == parts.authority.global_manifest.payloads[index]
        key = DecoderPayloadRouteKey(packet.sample_id, 0, "input_ids")
        assert packet.tensor_fields["input_ids"] is parts.payload.received_tensors[key]
    assert forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is owner
    with pytest.raises(MdpStateError, match="replay handoff is retired"):
        parts.handoff.require()

    cursor = owner.replay_cursor()
    with pytest.raises(MdpStateError, match="exactly one cursor"):
        owner.replay_cursor()
    assert next(cursor) is owner.records[0]
    assert next(cursor) is owner.records[1]
    with pytest.raises(MdpStateError, match="permits exactly 2"):
        next(cursor)
    assert tuple(owner.embedding_leaves) == tuple(call[3] for call in parts.calls)
    assert all(leaf.is_leaf and leaf.requires_grad for leaf in owner.embedding_leaves.values())
    with pytest.raises(AttributeError):
        owner.source_window
    owner.abort()
    assert parts.handle.consumed
    assert not parts.binding_owner.active
    assert parts.allocator.released == [
        *(tensor for _, tensor in parts.allocator.acquired),
        *parts.old_buffers,
    ]


def test_completion_requires_exhausted_returned_cursor_and_exact_token(monkeypatch):
    parts = _parts(monkeypatch)
    owner = _run(parts)
    cursor = owner.replay_cursor()
    token = torch.tensor(9.0)

    with pytest.raises(AttributeError):
        owner._schedule_returned = True
    cursor._state = api._RETIRED
    with pytest.raises(MdpStateError, match="cursor exhaustion"):
        owner.mark_schedule_returned(cursor)
    cursor._state = api._ACTIVE
    with pytest.raises(MdpStateError, match="cursor exhaustion"):
        owner.mark_schedule_returned(cursor)
    next(cursor)
    next(cursor)
    with pytest.raises(MdpStateError, match="returned schedule cursor"):
        owner.prepare_completion(cursor, object())
    owner.capture_global_num_tokens(token)
    with pytest.raises(MdpStateError, match="captures global num_tokens once"):
        owner.capture_global_num_tokens(torch.tensor(9.0))
    schedule_return = owner.mark_schedule_returned(cursor)
    with pytest.raises(MdpStateError, match="returns once"):
        owner.mark_schedule_returned(cursor)
    completion = owner.prepare_completion(cursor, schedule_return)
    assert completion is owner.prepare_completion(cursor, schedule_return)
    with pytest.raises(MdpStateError, match="exact schedule return"):
        owner.prepare_completion(cursor, copy.copy(schedule_return))
    assert completion.authority is parts.authority
    assert completion.globally_reduced_num_tokens is token
    assert owner.require_completion(completion) is completion
    token.add_(1)
    with pytest.raises(MdpStateError, match="exact owner and token"):
        owner.prepare_completion(cursor, schedule_return)
    owner.abort()


def test_mutated_or_substituted_token_invalidates_completion(monkeypatch):
    parts = _parts(monkeypatch)
    owner = _run(parts)
    cursor = owner.replay_cursor()
    next(cursor)
    next(cursor)
    token = torch.tensor(3.0)
    owner.capture_global_num_tokens(token)
    schedule_return = owner.mark_schedule_returned(cursor)
    token.add_(1)
    with pytest.raises(MdpStateError, match="exact in-place num_tokens"):
        owner.prepare_completion(cursor, schedule_return)
    owner.abort()

    parts = _parts(monkeypatch)
    owner = _run(parts)
    cursor = owner.replay_cursor()
    next(cursor)
    next(cursor)
    owner.capture_global_num_tokens(torch.tensor(3.0))
    schedule_return = owner.mark_schedule_returned(cursor)
    api._ACTIVE_OWNERS[id(owner)][-1].token = torch.tensor(3.0)
    with pytest.raises(MdpStateError, match="exact in-place num_tokens"):
        owner.prepare_completion(cursor, schedule_return)
    owner.abort()


def test_foreign_equal_authority_and_bundle_substitution_cleanup(monkeypatch):
    parts = _parts(monkeypatch)
    clone = copy.copy(parts.authority)
    for candidate in (object(), clone):
        with pytest.raises(MdpStateError, match="exact iteration authority"):
            api.run_repeated_d4_fixed_decoder_replay(
                parts.handoff,
                candidate,
                rebuild_microbatch=parts.rebuild,
                cp_partition_mode="zigzag",
            )
        assert parts.handoff.require() is parts.handoff

    parts.handoff.payload_bundle = object()
    with pytest.raises(MdpStateError, match="retains sealed fields"):
        _run(parts)
    with pytest.raises(MdpStateError, match="handoff is retired"):
        parts.handoff.require()
    assert parts.handle.consumed
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    parts = _parts(monkeypatch)
    object.__setattr__(parts.payload, "global_rank", 1)
    with pytest.raises(MdpBridgeError, match="rank authority"):
        _run(parts)
    assert parts.handle.consumed
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


@pytest.mark.parametrize(
    ("corruption", "error_type", "message"),
    (
        ("group_order", MdpStateError, "native CP4 group order"),
        ("decoder_order", MdpPlanError, "native CP4 domain order"),
        ("assignments", MdpPlanError, "one ordered CP4 assignment"),
        ("packed_group", MdpConfigurationError, "exact CP4 geometry"),
        ("packed_size", MdpConfigurationError, "exact CP4 geometry"),
        ("packed_boundary", MdpConfigurationError, "exact CP4 geometry"),
    ),
)
def test_fixed_cp4_and_packed_contract_reject_before_later_gate_stages(
    monkeypatch, corruption, error_type, message
):
    parts = _parts(monkeypatch)
    original_rebuild = parts.rebuild

    if corruption == "group_order":
        object.__setattr__(
            parts.binding._authority, "_group_ranks_getter", lambda _group: (3, 2, 1, 0)
        )
    elif corruption == "decoder_order":
        parts.authority.plan = replace(parts.authority.plan, decoder_ranks=(3, 2, 1, 0))
    elif corruption == "assignments":
        first = parts.authority.plan.microbatches[0]
        parts.authority.plan = replace(
            parts.authority.plan,
            microbatches=(
                replace(first, assignments=first.assignments * 2),
                *parts.authority.plan.microbatches[1:],
            ),
        )
    else:

        def corrupt_rebuild(*args, **kwargs):
            record = original_rebuild(*args, **kwargs)
            packed = record.decoder_packed_seq_params
            if corruption == "packed_group":
                packed.cp_group = object()
            elif corruption == "packed_size":
                packed.local_cp_size = 2
            else:
                packed.cu_seqlens_q_padded = torch.tensor([0, 7], dtype=torch.int32)
            return record

        parts.rebuild = corrupt_rebuild

    gate_events = []

    def converge_first_world(_binding, _authority, **kwargs):
        gate_events.append(("begin", kwargs["gate_id"]))
        try:
            kwargs["prepare"]()
        except BaseException as error:
            gate_events.append(("world0", error))
            raise
        raise AssertionError("invalid fixture unexpectedly reached the domain stage")

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", converge_first_world)
    with pytest.raises(error_type, match=message) as raised:
        _run(parts)
    assert gate_events == [("begin", 2), ("world0", raised.value)]
    assert parts.handle.consumed
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


@pytest.mark.parametrize("vision", (False, True))
def test_all_rank_text_and_vision_owners_are_typed(monkeypatch, vision):
    parts = _parts(monkeypatch, vision=vision, microbatches=1)
    owner = _run(parts)
    cursor = owner.replay_cursor()
    record = next(cursor)
    assert record.text_only is (not vision)
    assert bool(owner.embedding_leaves) is vision
    owner.capture_global_num_tokens(torch.tensor(1.0))
    schedule_return = owner.mark_schedule_returned(cursor)
    completion = owner.prepare_completion(cursor, schedule_return)
    assert owner.require_completion(completion) is completion
    owner.abort()


def test_runner_substitution_and_rejection_cleanup_then_fresh_runtime(monkeypatch):
    parts = _parts(monkeypatch)
    primary = MdpStateError("peer rejected")

    def reject(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", reject)
    with pytest.raises(MdpStateError) as raised:
        _run(parts)
    assert raised.value is primary
    assert parts.handle.consumed
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    fresh = _parts(monkeypatch, runtime=parts.runtime)
    assert fresh.runtime is parts.runtime
    owner = _run(fresh)
    owner.abort()

    substituted = _parts(monkeypatch)

    def substitute(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        return object()

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", substitute)
    with pytest.raises(MdpTaskFatalError, match="exact result"):
        _run(substituted)
    assert substituted.handle.consumed


@pytest.mark.parametrize("delete_operations", (False, True))
def test_rejection_uses_escrowed_operations_after_callback_substitution(
    monkeypatch, delete_operations
):
    parts = _parts(monkeypatch)
    primary = MdpStateError("final WORLD rejected")
    original_release_count = len(parts.allocator.released)
    bomb_calls = []

    class BombOperations:
        def release(self, buffer):
            bomb_calls.append(buffer)
            raise AssertionError("substituted release must not run")

    def reject(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        if delete_operations:
            del parts.handoff._operations
        else:
            parts.handoff._operations = BombOperations()
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", reject)
    with pytest.raises(MdpStateError) as raised:
        _run(parts)
    assert raised.value is primary
    assert bomb_calls == []
    leaves = tuple(tensor for _tag, tensor in parts.allocator.acquired)
    expected_releases = (*reversed(leaves), *parts.old_buffers)
    actual_releases = tuple(parts.allocator.released[original_release_count:])
    assert len(actual_releases) == len(expected_releases)
    assert all(
        actual is expected
        for actual, expected in zip(actual_releases, expected_releases, strict=True)
    )
    assert parts.handle.consumed
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None

    fresh = _parts(monkeypatch, runtime=parts.runtime)
    owner = _run(fresh)
    owner.abort()


def test_abort_preserves_primary_and_rejects_replay(monkeypatch):
    parts = _parts(monkeypatch)
    owner = _run(parts)
    primary = MdpStateError("schedule failed")
    parts.allocator.fail_release = True

    owner.abort(primary)

    assert any("buffer release error" in note for note in primary.__notes__)
    with pytest.raises(MdpStateError, match="owner is retired"):
        owner.abort(primary)


@pytest.mark.parametrize(("vision", "selected"), ((False, False), (True, False), (True, True)))
def test_gradient_handoff_transfers_exact_completed_replay_without_release(
    monkeypatch, vision, selected
):
    parts = _parts(monkeypatch, vision=vision, selected=selected, microbatches=1)
    owner, cursor, token, _schedule_return, completion = _complete(parts)
    releases = tuple(parts.allocator.released)

    handoff = owner._claim_for_gradient(parts.authority, completion)

    assert type(handoff) is api._D4FixedDecoderGradientHandoff
    assert handoff.require() is handoff
    assert handoff.authority is parts.authority
    assert handoff.binding is parts.binding
    assert handoff.completion is completion
    assert handoff.records[0].text_only is (not vision)
    assert handoff.text_only is (not vision)
    assert handoff.is_selected is selected
    assert handoff.embedding_leaves is not None
    assert completion._owner is owner
    assert completion.globally_reduced_num_tokens is token
    completion_entry = api._ACTIVE_COMPLETIONS[id(completion)]
    assert completion_entry[0]() is handoff
    assert completion_entry[1] is completion
    assert completion_entry[3] is token
    assert completion_entry[4] == api._tensor_descriptor(token)
    assert forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is handoff
    assert tuple(parts.allocator.released) == releases
    with pytest.raises(MdpStateError, match="owner is retired"):
        owner.require()
    with pytest.raises(MdpStateError, match="exact active cursor"):
        next(cursor)

    handoff.abort()
    assert parts.handle is None or parts.handle.consumed
    assert parts.binding_owner is None or not parts.binding_owner.active
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


def test_gradient_handoff_claim_rejects_substitution_without_consuming(monkeypatch):
    parts = _parts(monkeypatch)
    owner, _cursor, _token, _schedule_return, completion = _complete(parts)

    for authority in (object(), copy.copy(parts.authority)):
        with pytest.raises(MdpStateError, match="exact iteration authority"):
            owner._claim_for_gradient(authority, completion)
        assert owner.require() is owner
    with pytest.raises(MdpStateError, match="exact owner and token"):
        owner._claim_for_gradient(parts.authority, copy.copy(completion))
    assert owner.require() is owner

    records = owner.records
    owner.records = ()
    with pytest.raises(MdpStateError, match="sealed resources"):
        owner._claim_for_gradient(parts.authority, completion)
    owner.records = records
    assert owner.require() is owner

    snapshot_calls = []

    def reject_mutated_authority(binding, authority):
        snapshot_calls.append((binding, authority))
        raise MdpStateError("mutated iteration authority")

    monkeypatch.setattr(api, "_snapshot_local_authority", reject_mutated_authority)
    with pytest.raises(MdpStateError, match="mutated iteration authority"):
        owner._claim_for_gradient(parts.authority, completion)
    assert snapshot_calls == [(parts.binding, parts.authority)]
    assert owner.require() is owner
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda binding, authority: authority)
    handoff = owner._claim_for_gradient(parts.authority, completion)
    handoff.abort()


def test_gradient_handoff_consume_is_exact_and_one_shot(monkeypatch):
    parts = _parts(monkeypatch)
    owner, _cursor, _token, _schedule_return, completion = _complete(parts)
    handoff = owner._claim_for_gradient(parts.authority, completion)

    for authority, candidate in (
        (copy.copy(parts.authority), completion),
        (parts.authority, copy.copy(completion)),
    ):
        with pytest.raises(MdpStateError, match="exact authority and completion"):
            handoff.consume(authority, candidate)
        assert handoff.require() is handoff

    def reject_mutated_authority(_binding, _authority):
        raise MdpStateError("mutated iteration authority")

    monkeypatch.setattr(api, "_snapshot_local_authority", reject_mutated_authority)
    with pytest.raises(MdpStateError, match="mutated iteration authority"):
        handoff.consume(parts.authority, completion)
    assert handoff.require() is handoff
    assert handoff._consumed is False
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda binding, authority: authority)
    assert handoff.consume(parts.authority, completion) is handoff
    with pytest.raises(MdpStateError, match="consumed exactly once"):
        handoff.consume(parts.authority, completion)
    handoff.abort()
    with pytest.raises(MdpStateError, match="handoff is retired"):
        handoff.consume(parts.authority, completion)


def test_gradient_handoff_abort_clears_before_hostile_notes_and_allows_fresh_runtime(monkeypatch):
    parts = _parts(monkeypatch)
    owner, _cursor, _token, _schedule_return, completion = _complete(parts)
    handoff = owner._claim_for_gradient(parts.authority, completion)
    reentry = []

    class HostilePrimary(BaseException):
        def add_note(self, _message):
            try:
                handoff.abort(self)
            except MdpStateError as error:
                reentry.append(error)
            raise RuntimeError("hostile note")

    del handoff.authority
    parts.allocator.fail_release = True
    primary = HostilePrimary("schedule failed")
    handoff.abort(primary)

    assert reentry and all("retired" in str(error) for error in reentry)
    assert parts.handle.consumed
    assert not parts.binding_owner.active
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    with pytest.raises(MdpStateError, match="handoff is retired"):
        handoff.abort(primary)

    parts.allocator.fail_release = False
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    fresh_owner = _run(fresh)
    fresh_owner.abort()


def test_gradient_handoff_abort_preserves_foreign_registries_through_cleanup(monkeypatch):
    parts = _parts(monkeypatch, microbatches=1)
    owner, _cursor, token, _schedule_return, completion = _complete(parts)
    handoff = owner._claim_for_gradient(parts.authority, completion)
    foreign_owner = _Handle()
    foreign_handoff = (weakref.ref(foreign_owner), object())
    foreign_runtime = (parts.runtime, weakref.ref(foreign_owner))

    class BombDescriptor:
        def __eq__(self, _other):
            raise AssertionError("foreign descriptor comparison executed")

    foreign_completion = (
        weakref.ref(foreign_owner),
        completion,
        parts.authority,
        token,
        BombDescriptor(),
    )
    observations = []
    handoff_entry = api._ACTIVE_GRADIENT_HANDOFFS[id(handoff)]
    completion_entry = api._ACTIVE_COMPLETIONS[id(completion)]

    def observe():
        observations.append(
            (
                api._ACTIVE_GRADIENT_HANDOFFS.get(id(handoff)) is foreign_handoff,
                forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is foreign_runtime,
                api._ACTIVE_COMPLETIONS.get(id(completion)) is foreign_completion,
            )
        )

    api._ACTIVE_GRADIENT_HANDOFFS[id(handoff)] = foreign_handoff
    forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign_runtime
    api._ACTIVE_COMPLETIONS[id(completion)] = foreign_completion
    parts.handle.callback = observe
    parts.binding_owner.callback = observe
    parts.allocator.release_callback = observe

    handoff._abort_from_escrow(handoff_entry)

    assert observations and all(all(value) for value in observations)
    assert api._ACTIVE_GRADIENT_HANDOFFS[id(handoff)] is foreign_handoff
    assert forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
    assert api._ACTIVE_COMPLETIONS[id(completion)] is foreign_completion
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    leaves = tuple(tensor for _tag, tensor in parts.allocator.acquired)
    assert tuple(parts.allocator.released) == (*leaves, *parts.old_buffers)

    del api._ACTIVE_GRADIENT_HANDOFFS[id(handoff)]
    del forward_api._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]
    del api._ACTIVE_COMPLETIONS[id(completion)]
    parts.allocator.release_callback = None
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    fresh_owner = _run(fresh)
    fresh_owner.abort()


def test_gradient_handoff_cleanup_rejects_fabricated_escrow_without_consuming(monkeypatch):
    parts = _parts(monkeypatch, microbatches=1)
    owner, _cursor, _token, _schedule_return, completion = _complete(parts)
    handoff = owner._claim_for_gradient(parts.authority, completion)

    with pytest.raises(MdpStateError, match="exact handoff escrow"):
        handoff._abort_from_escrow((object(),))
    assert handoff.require() is handoff
    handoff.abort()


def test_gradient_handoff_abort_uses_escrow_after_completion_field_deletion(monkeypatch):
    parts = _parts(monkeypatch, microbatches=1)
    owner, _cursor, _token, _schedule_return, completion = _complete(parts)
    handoff = owner._claim_for_gradient(parts.authority, completion)

    object.__delattr__(completion, "globally_reduced_num_tokens")
    handoff.abort()

    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    assert api._ACTIVE_COMPLETIONS.get(id(completion)) is None
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    fresh_owner = _run(fresh)
    fresh_owner.abort()


def test_replay_abort_preserves_foreign_completion_through_cleanup(monkeypatch):
    parts = _parts(monkeypatch, microbatches=1)
    owner, _cursor, token, _schedule_return, completion = _complete(parts)
    foreign_owner = _Handle()

    class BombDescriptor:
        def __eq__(self, _other):
            raise AssertionError("foreign descriptor comparison executed")

    foreign_completion = (
        weakref.ref(foreign_owner),
        completion,
        parts.authority,
        token,
        BombDescriptor(),
    )
    observations = []
    api._ACTIVE_COMPLETIONS[id(completion)] = foreign_completion

    def observe():
        observations.append(api._ACTIVE_COMPLETIONS.get(id(completion)) is foreign_completion)

    parts.handle.callback = observe
    parts.binding_owner.callback = observe
    parts.allocator.release_callback = observe
    owner.abort()

    assert observations and all(observations)
    assert api._ACTIVE_COMPLETIONS[id(completion)] is foreign_completion
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    del api._ACTIVE_COMPLETIONS[id(completion)]
    parts.allocator.release_callback = None
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    fresh_owner = _run(fresh)
    fresh_owner.abort()


def test_gradient_handoff_escrow_cleanup_is_one_shot_during_callback_reentry(monkeypatch):
    parts = _parts(monkeypatch, microbatches=1)
    owner, _cursor, _token, _schedule_return, completion = _complete(parts)
    handoff = owner._claim_for_gradient(parts.authority, completion)
    handoff_entry = api._ACTIVE_GRADIENT_HANDOFFS[id(handoff)]
    reentry = []

    def reenter():
        parts.allocator.release_callback = None
        try:
            handoff._abort_from_escrow(handoff_entry)
        except MdpStateError as error:
            reentry.append(error)

    parts.allocator.release_callback = reenter
    handoff._abort_from_escrow(handoff_entry)

    assert len(reentry) == 1
    assert "handoff is retired" in str(reentry[0])
    assert parts.handle.calls == 1
    assert len(parts.binding_owner.calls) == 1
    releases = tuple(parts.allocator.released)
    with pytest.raises(MdpStateError, match="handoff is retired"):
        handoff._abort_from_escrow(handoff_entry)
    assert tuple(parts.allocator.released) == releases
