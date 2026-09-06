# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gate5 selected encoder backward contracts."""

import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_dynamic_decoder_replay as dynamic_replay_api
from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as gradient_api
from megatron.core.mdp import dynamic_cp_d4_encoder_selected_backward as api
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay_api
from megatron.core.mdp.activation import EncoderForwardHandle
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
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment
from megatron.core.mdp.protocols import DynamicEncoderCpBinding
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord


class _Operations:
    def __init__(self):
        self.released = []
        self.on_release = None

        def acquire(**_kwargs):
            raise AssertionError("Gate5 must not acquire replay resources")

        def release(value):
            self.released.append(value)
            if self.on_release is not None:
                callback, self.on_release = self.on_release, None
                callback()

        self.acquire = acquire
        self.release = release


class _DynamicGroup:
    def size(self):
        return 4

    def rank(self):
        return 0


class _DynamicSolver:
    def __init__(self, cp_size):
        self.cp_size = cp_size

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        del max_seq_len_per_rank, min_cp_size
        count = total_gpus // self.cp_size
        selected = sample_seqlens[:count]
        return (
            [value for _sample_id, length in selected for value in [[length]] * self.cp_size],
            sample_seqlens[count:],
            None,
            [value for sample_id, _length in selected for value in [[sample_id]] * self.cp_size],
        )


class _DynamicAllocator:
    def __init__(self):
        self.released = []

        def acquire(*, rows, width, dtype, device, tag):
            del tag
            return torch.empty(rows if width == 0 else (rows, width), dtype=dtype, device=device)

        def release(value):
            self.released.append(value)

        self.acquire = acquire
        self.release = release


def _real_dynamic_authority(cp_size):
    domain_group = _DynamicGroup()
    world_group = object()
    group_authority = _RepeatedD4GroupAuthority(
        world_ranks=tuple(range(8)),
        domain_ranks=(0, 1, 2, 3),
        global_rank=0,
        expert_parallel_size=1,
        _world_group=world_group,
        _domain_group=domain_group,
        _expert_group=None,
        _device=torch.device("cuda"),
        _timeout_seconds=1.0,
        _status_gather_factory=lambda: (lambda *_args, **_kwargs: None),
        _group_ranks_getter=lambda value: (
            tuple(range(8)) if value is world_group else (0, 1, 2, 3)
        ),
        _world_pre_gate=lambda *_args, **_kwargs: None,
        _domain_status=lambda *_args, **_kwargs: None,
        _seal=_AUTHORITY_SEAL,
    )
    binding = _RepeatedD4GroupBinding(
        world_ranks=tuple(range(8)),
        domain_ranks=(0, 1, 2, 3),
        global_rank=0,
        expert_parallel_size=1,
        _authority=group_authority,
        _seal=_BINDING_SEAL,
    )
    samples = []
    items = []
    packets = []
    for index in range(4):
        sample_id = GlobalSampleId(0, index)
        item_id = GlobalVisionItemId(0, index)
        samples.append(
            DecoderSampleMetadata(
                sample_id, 4, 4, (EncoderVisionItemMetadata(item_id, sample_id, 0),)
            )
        )
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
    authority = build_repeated_d4_joint_iteration_authority(
        binding,
        metadata,
        decoder_max_seqlen_per_rank=4,
        decoder_minimum_cp_size=1,
        decoder_solver=_DynamicSolver(cp_size),
        encoder_max_seqlen_per_rank=4,
        encoder_minimum_cp_size=1,
        encoder_workload_query=lambda values, group_size: EncoderWorkEstimate(
            max(1, len(values)), group_size
        ),
        bridge_width=3,
        bridge_dtype=torch.float32,
    )
    return binding, authority


def _real_dynamic_gate4(monkeypatch, cp_size):
    binding, authority = _real_dynamic_authority(cp_size)
    samples = {sample.sample_id: sample for sample in authority.global_manifest.samples}
    items = {item.item_id: item for item in authority.global_manifest.items}
    records = []
    for microbatch in authority.plan.microbatches:
        assignment = next(value for value in microbatch.assignments if 0 in value.endpoint_ranks)
        vision = []
        padded_start = 0
        for local_sample_id, sample_id in enumerate(assignment.sample_ids):
            sample = samples[sample_id]
            for encoder_item in sample.vision_items:
                item = items[encoder_item.item_id]
                vision.append(
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
        records.append(
            MdpMicrobatchRecord(
                microbatch.microbatch_index, False, tuple(vision), object(), MappingProxyType({})
            )
        )
    leaves = MappingProxyType(
        {
            DecoderMicrobatchKey(record.microbatch_id): torch.ones(
                (sum(item.output_rows for item in record.vision_items), 3), requires_grad=True
            )
            for record in records
        }
    )
    for leaf in leaves.values():
        (leaf * 2.0).sum().backward()
    allocator = _DynamicAllocator()
    runtime = SimpleNamespace(device=torch.device("cpu"), allocator=allocator)
    operations = replay_api._forward._D4EncoderForwardOperations(
        allocator,
        allocator.acquire,
        allocator.release,
        runtime.device,
        torch.float32,
        object(),
        object(),
        object(),
        lambda: None,
        lambda *_args: None,
        lambda *_args: None,
        1,
        3,
        torch.float32,
        replay_api._forward._OPERATIONS_SEAL,
    )
    selected_ranks = gate4._selected_ranks(authority)
    output = torch.ones((2 * len(authority.global_manifest.items), 3), requires_grad=True) * 2
    segments = tuple(
        EncoderThdSegment(item.item_id.local_item_id, 0, 0, index, 0, 1, 2 * index, 2, (1, 1, 2))
        for index, item in enumerate(authority.global_manifest.items)
    )
    handle = EncoderForwardHandle(0, 0, (output,), (EncoderThdLayout(0, segments),))
    binding_owner = DynamicEncoderCpBinding(
        SimpleNamespace(ranks=selected_ranks), is_current=lambda: True, restore=lambda *_args: None
    )
    predecessor_buffers = (torch.empty(0),)
    handoff_resources = (binding_owner, predecessor_buffers, operations, handle)
    leaf_bases = tuple(leaves.values())
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
    ready = SimpleNamespace(records=tuple(records), embedding_leaves=leaves)
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
    for record in records:
        assert next(cursor) is record
    token = torch.tensor(4.0)
    replay.capture_global_num_tokens(token)
    returned = replay.mark_schedule_returned(cursor)
    completion = replay.prepare_completion(cursor, returned)
    monkeypatch.setattr(dynamic_replay_api, "_snapshot_local_authority", lambda b, a: a)
    handoff = replay._claim_for_gradient(authority, completion)

    incoming_entries = tuple(
        entry for entry in authority.gradient_ledger.entries if entry.dst_global_rank == 0
    )
    expected_received = MappingProxyType(
        {
            entry.key: torch.full(
                (items[entry.key.item_id].output_rows, 3), float(entry.key.endpoint_rank + 1)
            )
            for entry in incoming_entries
        }
    )
    expected_backward = torch.cat(
        tuple(
            torch.full(
                (item.output_rows, 3),
                sum(
                    float(entry.key.endpoint_rank + 1)
                    for entry in incoming_entries
                    if entry.key.item_id == item.item_id
                ),
            )
            for item in authority.global_manifest.items
        )
    )
    monkeypatch.setattr(
        gradient_api, "_validate_repeated_d4_group_binding", lambda value: value._authority
    )
    monkeypatch.setattr(
        gradient_api,
        "dynamic_bridge_split_sizes",
        lambda *_args, **_kwargs: ((0, 0, 0, 0), (0, 0, 0, 0)),
    )

    def prepare_exchange(*_args, **kwargs):
        return PreparedDynamicBridgeExchange(
            BridgePhase.GRADIENT,
            torch.float32,
            0,
            (0, 1, 2, 3),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            b"g" * 16,
            kwargs["send_buffer"],
            kwargs["receive_buffer"],
            expected_received,
        )

    monkeypatch.setattr(gradient_api, "prepare_dynamic_bridge_exchange", prepare_exchange)
    monkeypatch.setattr(
        gradient_api, "validate_prepared_dynamic_bridge_exchange", lambda value: value
    )
    monkeypatch.setattr(
        gradient_api,
        "_execute_validated_dynamic_bridge_exchange",
        lambda exchange, **_kwargs: exchange.received_tensors,
    )
    runner = lambda _binding, _authority, **kwargs: kwargs["domain_collective"](kwargs["prepare"]())
    monkeypatch.setattr(gradient_api, "run_repeated_d4_authority_collective", runner)
    gate3 = gradient_api.run_repeated_d4_dynamic_encoder_gradient(
        handoff, authority, completion, all_to_all_single=lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(gate4, "_snapshot_local_authority", lambda b, a: a)
    monkeypatch.setattr(gate4, "validate_prepared_dynamic_bridge_exchange", lambda value: value)
    monkeypatch.setattr(gate4, "run_repeated_d4_authority_collective", runner)
    gate4_owner = gate4.run_repeated_d4_encoder_backward_authorization(gate3, authority, completion)
    return (
        gate4_owner,
        authority,
        completion,
        runtime,
        len(records),
        operations,
        expected_backward,
    )


def _parts(
    monkeypatch,
    *,
    rank=0,
    selected_size=2,
    text_only=False,
    chunks=1,
    runtime=None,
    corruption=None,
    decoder_mode="fixed",
):
    selected_ranks = () if text_only else tuple(range(selected_size))
    authority = SimpleNamespace(
        participant_ranks=(0, 1, 2, 3), bridge_width=2, bridge_dtype=torch.float32
    )
    binding = SimpleNamespace(global_rank=rank)
    runtime = runtime or SimpleNamespace()
    release_tracker = _Operations()
    operations = replay_api._forward._D4EncoderForwardOperations(
        object(),
        release_tracker.acquire,
        release_tracker.release,
        torch.device("cpu"),
        torch.float32,
        object(),
        object(),
        object(),
        lambda: None,
        lambda *_args: None,
        lambda *_args: None,
        1,
        2,
        torch.float32,
        replay_api._forward._OPERATIONS_SEAL,
    )
    restored = []
    selected = rank in selected_ranks
    outputs = []
    layouts = []
    items = []
    routed = {}
    canonical_route_keys = []
    expected = []
    if selected:
        for chunk in range(chunks):
            base = torch.arange(1, 9, dtype=torch.float32).reshape(4, 2).requires_grad_()
            output = base * 2.0
            outputs.append(output)
            segments = []
            full = torch.empty_like(output)
            for ordinal, (start, rows) in enumerate(((0, 1), (1, 3))):
                local_id = chunk * 2 + ordinal
                item_id = GlobalVisionItemId(0, local_id)
                items.append(SimpleNamespace(item_id=item_id))
                segments.append(
                    EncoderThdSegment(local_id, chunk, chunk, ordinal, 0, 1, start, rows, (1, 1, 1))
                )
                value = torch.full((rows, 2), float(local_id + 1))
                full.narrow(0, start, rows).copy_(value)
                canonical_route_keys.append(DynamicBridgeKey(item_id, 3))
                if rank == 0:
                    routed[DynamicBridgeKey(item_id, 3)] = value
            expected.append(full)
            layouts.append(EncoderThdLayout(0, tuple(segments)))
        handle = EncoderForwardHandle(0, 0, tuple(outputs), tuple(layouts))
        membership = SimpleNamespace(ranks=selected_ranks)
        binding_owner = DynamicEncoderCpBinding(
            membership, is_current=lambda: True, restore=lambda primary: restored.append(primary)
        )
    else:
        handle = None
        binding_owner = None
    if corruption == "missing":
        routed.pop(next(iter(routed)))
    elif corruption == "extra":
        routed[DynamicBridgeKey(GlobalVisionItemId(0, 99), 3)] = torch.ones(1, 2)
    elif corruption == "geometry":
        key = next(iter(routed))
        routed[key] = torch.ones(2, 3)
    elif corruption == "offset":
        object.__setattr__(handle.chunk_layouts[0].segments[1], "output_row_start", 2)
    elif corruption == "follower_output":
        object.__setattr__(handle, "chunk_outputs", (torch.ones(4, 2),))
    elif corruption == "follower_layout":
        object.__setattr__(handle.chunk_layouts[0].segments[1], "output_row_start", 2)
    elif corruption == "follower_empty":
        object.__setattr__(handle, "chunk_outputs", ())
        object.__setattr__(handle, "chunk_layouts", ())
    elif corruption == "follower_width":
        base = torch.ones(4, 3, requires_grad=True)
        object.__setattr__(handle, "chunk_outputs", (base * 2.0,))
    elif corruption == "follower_dtype":
        base = torch.ones(4, 2, dtype=torch.float64, requires_grad=True)
        object.__setattr__(handle, "chunk_outputs", (base * 2.0,))
    elif corruption == "follower_noncontiguous":
        base = torch.ones(2, 4, requires_grad=True)
        object.__setattr__(handle, "chunk_outputs", (base.t(),))
    elif corruption == "follower_device":
        base = torch.ones(4, 2, device="meta", requires_grad=True)
        object.__setattr__(handle, "chunk_outputs", (base * 2.0,))
    authority.global_manifest = SimpleNamespace(items=tuple(items))
    authority.gradient_ledger = SimpleNamespace(
        entries=tuple(
            SimpleNamespace(key=key, dst_global_rank=selected_ranks[0])
            for key in canonical_route_keys
        )
    )
    if corruption == "duplicate_endpoint":
        authority.gradient_ledger = SimpleNamespace(
            entries=(*authority.gradient_ledger.entries, authority.gradient_ledger.entries[0])
        )
    token = torch.tensor(1.0)
    completion_owner = object()
    predecessor_entry = None if decoder_mode == "fixed" else object()
    completion_lifecycle = None if decoder_mode == "fixed" else object()
    mode = replay_api if decoder_mode == "fixed" else dynamic_replay_api
    if decoder_mode == "fixed":
        completion = replay_api._D4FixedDecoderCompletion(
            authority, token, completion_owner, replay_api._COMPLETION_SEAL
        )
    else:
        completion = dynamic_replay_api._D4DynamicDecoderCompletion(
            authority,
            token,
            completion_owner,
            predecessor_entry,
            completion_lifecycle,
            dynamic_replay_api._COMPLETION_SEAL,
        )
    receipt = gradient_api._D4EncoderOnlyGradientReceipt(
        authority,
        completion,
        SimpleNamespace(received_tensors=MappingProxyType(routed)),
        MappingProxyType(routed),
        gradient_api._RECEIPT_SEAL,
    )
    object.__setattr__(
        receipt, "exchange", SimpleNamespace(received_tensors=receipt.received_tensors)
    )
    if selected:
        carrier = gate4._D4EncoderBackwardMember(
            authority,
            completion,
            receipt,
            selected_ranks,
            rank,
            rank == 0,
            handle,
            receipt.received_tensors,
            gate4._MEMBER_SEAL,
        )
    else:
        carrier = gate4._D4EncoderBackwardEmpty(
            authority, completion, receipt, selected_ranks, text_only, gate4._EMPTY_SEAL
        )
    buffers = (object(), object())
    leaf_bases = (object(),)
    transport = (object(), object())
    resources = (binding_owner, buffers, operations, handle)
    trusted = (
        runtime,
        authority,
        completion,
        receipt,
        carrier,
        resources,
        leaf_bases,
        transport,
        operations,
        binding,
    )
    owner = gate4._D4EncoderBackwardAuthorizationOwner(trusted, gate4._OWNER_SEAL)
    reference = weakref.ref(owner)
    completion_entry = (
        (reference, completion, authority, token, replay_api._tensor_descriptor(token))
        if decoder_mode == "fixed"
        else (
            reference,
            completion,
            completion_owner,
            predecessor_entry,
            completion_lifecycle,
            authority,
            token,
            dynamic_replay_api._tensor_descriptor(token),
            dynamic_replay_api._GRADIENT_COMPLETION_ENTRY_SEAL,
        )
    )
    token_descriptor = completion_entry[4] if decoder_mode == "fixed" else completion_entry[7]
    provenance = gradient_api._D4ReplayGradientProvenance(
        mode,
        completion_owner,
        predecessor_entry,
        completion_lifecycle,
        token,
        token_descriptor,
        operations.acquire,
        operations.release,
        gradient_api._PROVENANCE_SEAL,
    )
    completion_escrow = (
        mode,
        mode._ACTIVE_COMPLETIONS,
        completion_entry,
        provenance.release,
        provenance,
        gradient_api._COMPLETION_ESCROW_SEAL,
    )
    trusted = (*trusted, completion_escrow)
    owner._trusted = trusted
    owner_entry = (reference, *trusted)
    gate4._ACTIVE_OWNERS[id(owner)] = owner_entry
    role = (
        (selected_ranks, rank, rank == 0, handle, receipt.received_tensors)
        if selected
        else (selected_ranks, text_only)
    )
    carrier_entry = (reference, carrier, authority, completion, receipt, *role)
    gate4._ACTIVE_CARRIERS[id(carrier)] = carrier_entry
    receipt_entry = (
        reference,
        receipt,
        authority,
        completion,
        receipt.exchange,
        receipt.received_tensors,
    )
    gradient_api._ACTIVE_RECEIPTS[id(receipt)] = receipt_entry
    mode._ACTIVE_COMPLETIONS[id(completion)] = completion_entry
    owner_escrow = gate4._OwnerEscrow(
        reference,
        owner_entry,
        (reference,),
        receipt_entry,
        receipt_entry,
        carrier_entry,
        (runtime, reference),
        completion_escrow,
        gate4._OWNER_ESCROW_SEAL,
    )
    gate4._OWNER_ESCROWS[id(owner)] = owner_escrow
    gate4._TRUSTED_OWNER_ESCROWS[id(owner)] = owner_escrow
    gate4._ACTIVATED_OWNER_ESCROWS[id(owner)] = owner_escrow
    gate4._TRUSTED_ACTIVATED_OWNER_ESCROWS[id(owner)] = owner_escrow
    replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = owner_escrow.runtime_entry
    assert len(mode._ACTIVE_COMPLETIONS[id(completion)]) == (5 if decoder_mode == "fixed" else 9)
    assert mode._ACTIVE_COMPLETIONS[id(completion)] is completion_escrow[2]
    assert provenance.acquire is operations.acquire
    assert provenance.release is operations.release
    assert completion_escrow[3] is provenance.release
    monkeypatch.setattr(gate4, "_snapshot_local_authority", lambda b, a: a)
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda b, a: a)
    assert owner.require() is owner
    assert all(
        actual is expected
        for actual, expected in zip(owner._trusted[:10], trusted[:10], strict=True)
    )
    events = []

    def runner(_binding, _authority, **kwargs):
        events.append(("gate", kwargs["gate_id"], kwargs["byte_generator"]))
        prepared = kwargs["prepare"]()
        events.append("world0")
        prepared = kwargs["domain_collective"](prepared)
        events.extend(("domain", "world1"))
        return prepared

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    return SimpleNamespace(
        runtime=runtime,
        authority=authority,
        binding=binding,
        owner=owner,
        completion=completion,
        completion_registry=mode._ACTIVE_COMPLETIONS,
        carrier=carrier,
        handle=handle,
        expected=tuple(expected),
        events=events,
        operations=release_tracker,
        forward_operations=operations,
        buffers=(*transport, *leaf_bases, *buffers),
        restored=restored,
    )


@pytest.mark.parametrize("decoder_mode", ("fixed", "dynamic"))
@pytest.mark.parametrize(
    ("rank", "size", "text_only", "selected"),
    (
        (0, 1, False, True),
        (0, 2, False, True),
        (1, 2, False, True),
        (0, 4, False, True),
        (3, 4, False, True),
        (3, 2, False, False),
        (0, 1, True, False),
    ),
)
def test_gate5_runs_backward_only_after_world_domain_world(
    monkeypatch, rank, size, text_only, selected, decoder_mode
):
    parts = _parts(
        monkeypatch,
        rank=rank,
        selected_size=size,
        text_only=text_only,
        chunks=2,
        decoder_mode=decoder_mode,
    )
    marker = object()
    if selected:
        original = parts.handle.backward

        def backward(gradients):
            assert parts.events[-1] == "world1"
            if rank == 0:
                for actual, expected in zip(gradients, parts.expected, strict=True):
                    torch.testing.assert_close(actual, expected)
            else:
                assert all(torch.count_nonzero(value) == 0 for value in gradients)
            return original(gradients)

        parts.handle.backward = backward
    owner = api.run_repeated_d4_encoder_selected_backward(
        parts.owner, parts.authority, parts.completion, byte_generator=marker
    )
    assert parts.events == [("gate", 5, marker), "world0", "domain", "world1"]
    assert owner.require() is owner and owner.binding is parts.binding
    normalized = owner.backward_completion.normalized_completion_escrow
    owner_escrow = api._TRUSTED_OWNER_ESCROWS[id(owner)]
    assert normalized.registry is parts.completion_registry
    assert parts.completion_registry[id(parts.completion)] is owner_escrow.decoder_completion_entry
    assert id(parts.owner) not in gate4._ACTIVE_OWNERS
    assert id(parts.owner) not in gate4._OWNER_ESCROWS
    assert id(parts.owner) not in gate4._TRUSTED_OWNER_ESCROWS
    assert id(parts.owner) not in gate4._ACTIVATED_OWNER_ESCROWS
    assert id(parts.owner) not in gate4._TRUSTED_ACTIVATED_OWNER_ESCROWS
    assert api._CANONICAL_OWNER_ESCROWS[id(owner)] is owner_escrow
    with pytest.raises(MdpStateError, match="encoder backward owner is retired"):
        parts.owner.require()
    assert (parts.handle is not None and parts.handle._backward_done) is selected
    assert parts.restored == [] and parts.operations.released == []
    owner.abort()
    assert id(parts.completion) not in parts.completion_registry


@pytest.mark.parametrize("decoder_cp_size", (1, 2, 4))
def test_real_dynamic_replay_gate3_gate4_gate5_chain(monkeypatch, decoder_cp_size):
    (predecessor, authority, completion, runtime, record_count, operations, expected_backward) = (
        _real_dynamic_gate4(monkeypatch, decoder_cp_size)
    )
    assert record_count == decoder_cp_size
    incoming = tuple(
        entry for entry in authority.gradient_ledger.entries if entry.dst_global_rank == 0
    )
    assert all(
        sum(entry.key.item_id == item.item_id for entry in incoming) == decoder_cp_size
        for item in authority.global_manifest.items
    )
    original_backward = predecessor.carrier.forward_handle.backward

    def backward(gradients):
        assert len(gradients) == 1
        torch.testing.assert_close(gradients[0], expected_backward)
        assert gradients[0].storage_offset() == 0
        return original_backward(gradients)

    predecessor.carrier.forward_handle.backward = backward
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda b, a: a)
    monkeypatch.setattr(
        api,
        "run_repeated_d4_authority_collective",
        lambda _binding, _authority, **kwargs: kwargs["domain_collective"](kwargs["prepare"]()),
    )
    owner = api.run_repeated_d4_encoder_selected_backward(predecessor, authority, completion)
    assert owner.require() is owner
    assert owner.backward_completion.completion is completion
    assert owner.backward_completion.normalized_completion_escrow.registry is (
        dynamic_replay_api._ACTIVE_COMPLETIONS
    )
    assert owner._trusted[10] is operations
    assert type(operations) is replay_api._forward._D4EncoderForwardOperations
    assert operations._seal is replay_api._forward._OPERATIONS_SEAL
    assert operations.allocator is runtime.allocator
    assert operations.acquire is runtime.allocator.acquire
    assert operations.release is runtime.allocator.release
    assert operations.device == torch.device("cpu")
    assert operations.payload_width > 0
    assert operations.bridge_width == authority.bridge_width
    assert operations.bridge_dtype == authority.bridge_dtype
    assert owner.backward_completion.normalized_completion_escrow.release is operations.release
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)][1]() is owner
    owner.abort()


@pytest.mark.parametrize("decoder_mode", ("fixed", "dynamic"))
@pytest.mark.parametrize("timing", ("domain", "final"))
def test_gate5_normalized_provenance_mutation_rejects_before_backward_and_retries(
    monkeypatch, decoder_mode, timing
):
    parts = _parts(monkeypatch, decoder_mode=decoder_mode)
    primary_provenance = parts.owner._trusted[10][4]

    def runner(_binding, _authority, **kwargs):
        prepared = kwargs["prepare"]()
        if timing == "domain":
            object.__setattr__(primary_provenance, "release", lambda _value: None)
        prepared = kwargs["domain_collective"](prepared)
        if timing == "final":
            object.__setattr__(parts.completion, "globally_reduced_num_tokens", torch.tensor(9.0))
        return prepared

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    expected = MdpStateError if timing == "domain" else MdpTaskFatalError
    message = (
        "MDP: Gate5 retains exact normalized decoder completion."
        if timing == "domain"
        else "MDP: selected encoder backward failed after Gate5 final WORLD."
    )
    with pytest.raises(expected) as caught:
        api.run_repeated_d4_encoder_selected_backward(
            parts.owner, parts.authority, parts.completion
        )
    assert str(caught.value) == message
    assert "Gate5 cleanup recovered mutated owner integrity." in caught.value.__notes__
    assert parts.handle._backward_done is False
    assert id(parts.completion) not in parts.completion_registry
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.operations.released == list(parts.buffers)
    fresh = _parts(monkeypatch, runtime=parts.runtime, decoder_mode=decoder_mode)
    monkeypatch.setattr(
        api,
        "run_repeated_d4_authority_collective",
        lambda _binding, _authority, **kwargs: kwargs["domain_collective"](kwargs["prepare"]()),
    )
    owner = api.run_repeated_d4_encoder_selected_backward(
        fresh.owner, fresh.authority, fresh.completion
    )
    owner.abort()


@pytest.mark.parametrize("decoder_mode", ("fixed", "dynamic"))
def test_gate5_abort_preserves_foreign_decoder_completion_registry(monkeypatch, decoder_mode):
    parts = _parts(monkeypatch, decoder_mode=decoder_mode)
    owner = api.run_repeated_d4_encoder_selected_backward(
        parts.owner, parts.authority, parts.completion
    )
    foreign = object()
    parts.completion_registry[id(parts.completion)] = foreign
    owner.abort()
    assert parts.completion_registry[id(parts.completion)] is foreign
    assert parts.operations.released == list(parts.buffers)
    del parts.completion_registry[id(parts.completion)]


def test_gate5_abort_uses_immutable_escrow_and_preserves_foreign_slots(monkeypatch):
    parts = _parts(monkeypatch, decoder_mode="dynamic")
    owner = api.run_repeated_d4_encoder_selected_backward(
        parts.owner, parts.authority, parts.completion
    )
    foreign_owner = object()
    foreign_runtime = object()
    api._OWNER_ESCROWS[id(owner)] = foreign_owner
    replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign_runtime
    owner.abort()
    assert api._OWNER_ESCROWS[id(owner)] is foreign_owner
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
    assert parts.operations.released == list(parts.buffers)
    with pytest.raises(MdpStateError, match="retired"):
        owner.abort()
    del api._OWNER_ESCROWS[id(owner)]
    del replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]


@pytest.mark.parametrize("slot", ("public", "trusted"))
def test_gate5_abort_rejects_same_self_forged_cleanup(monkeypatch, slot):
    parts = _parts(monkeypatch, decoder_mode="dynamic")
    owner = api.run_repeated_d4_encoder_selected_backward(
        parts.owner, parts.authority, parts.completion
    )
    canonical = api._OWNER_ESCROWS[id(owner)]
    injected = []
    forged_normalized = canonical.normalized._replace(release=lambda value: injected.append(value))
    registry = api._OWNER_ESCROWS if slot == "public" else api._TRUSTED_OWNER_ESCROWS
    registry[id(owner)] = canonical._replace(normalized=forged_normalized)
    owner.abort()
    assert injected == []
    assert parts.operations.released == list(parts.buffers)
    with pytest.raises(MdpStateError, match="retired"):
        owner.abort()


def test_gate5_canonical_anchor_disambiguates_valid_foreign_escrows(monkeypatch):
    parts = _parts(monkeypatch, decoder_mode="dynamic")
    owner = api.run_repeated_d4_encoder_selected_backward(
        parts.owner, parts.authority, parts.completion
    )
    canonical = api._CANONICAL_OWNER_ESCROWS[id(owner)]
    public_foreign = canonical._replace()
    trusted_foreign = canonical._replace()
    api._OWNER_ESCROWS[id(owner)] = public_foreign
    api._TRUSTED_OWNER_ESCROWS[id(owner)] = trusted_foreign
    owner.abort()
    assert api._OWNER_ESCROWS[id(owner)] is public_foreign
    assert api._TRUSTED_OWNER_ESCROWS[id(owner)] is trusted_foreign
    assert parts.operations.released == list(parts.buffers)
    del api._OWNER_ESCROWS[id(owner)]
    del api._TRUSTED_OWNER_ESCROWS[id(owner)]


@pytest.mark.parametrize(
    "mutation", ("delete_active", "substitute_active", "delete_field", "trusted")
)
def test_gate5_abort_recovers_live_mutation_and_reentry(monkeypatch, mutation):
    parts = _parts(monkeypatch, decoder_mode="dynamic")
    owner = api.run_repeated_d4_encoder_selected_backward(
        parts.owner, parts.authority, parts.completion
    )
    foreign = object()
    if mutation == "delete_active":
        del api._ACTIVE_OWNERS[id(owner)]
    elif mutation == "substitute_active":
        api._ACTIVE_OWNERS[id(owner)] = foreign
    elif mutation == "delete_field":
        del owner.authority
    else:
        owner._trusted = ()
    reentry = []

    def callback():
        with pytest.raises(MdpStateError, match="retired") as caught:
            owner.abort()
        reentry.append(caught.value)

    parts.operations.on_release = callback
    primary = RuntimeError("later failure")
    owner.abort(primary)
    assert reentry and parts.operations.released == list(parts.buffers)
    assert any("recovered mutated owner integrity" in note for note in primary.__notes__)
    if mutation == "substitute_active":
        assert api._ACTIVE_OWNERS[id(owner)] is foreign
        del api._ACTIVE_OWNERS[id(owner)]
    fresh = _parts(monkeypatch, runtime=parts.runtime, decoder_mode="dynamic")
    replacement = api.run_repeated_d4_encoder_selected_backward(
        fresh.owner, fresh.authority, fresh.completion
    )
    replacement.abort()


@pytest.mark.parametrize(
    "mutation",
    (
        "complete_selected",
        "complete_text",
        "complete_seal",
        "complete_token",
        "provenance_seal",
        "provenance_release",
        "operations_seal",
        "operations_release",
    ),
)
def test_gate5_abort_ignores_nested_callback_authority_mutation(monkeypatch, mutation):
    parts = _parts(monkeypatch, decoder_mode="dynamic")
    owner = api.run_repeated_d4_encoder_selected_backward(
        parts.owner, parts.authority, parts.completion
    )
    complete = owner.backward_completion
    normalized = api._CANONICAL_OWNER_ESCROWS[id(owner)].normalized
    injected = []
    if mutation == "complete_selected":
        object.__setattr__(complete, "selected", not complete.selected)
    elif mutation == "complete_text":
        object.__setattr__(complete, "text_only", not complete.text_only)
    elif mutation == "complete_seal":
        object.__setattr__(complete, "_seal", object())
    elif mutation == "complete_token":
        object.__setattr__(complete, "normalized_completion_escrow", object())
    elif mutation == "provenance_seal":
        object.__setattr__(normalized.provenance, "_seal", object())
    elif mutation == "provenance_release":
        object.__setattr__(normalized.provenance, "release", lambda value: injected.append(value))
    elif mutation == "operations_seal":
        object.__setattr__(parts.forward_operations, "_seal", object())
    else:
        object.__setattr__(
            parts.forward_operations, "release", lambda value: injected.append(value)
        )
    reentry = []

    def callback():
        with pytest.raises(MdpStateError, match="retired") as caught:
            owner.abort()
        reentry.append(caught.value)

    parts.operations.on_release = callback
    primary = RuntimeError("later failure")
    owner.abort(primary)
    assert injected == []
    assert reentry and parts.operations.released == list(parts.buffers)
    assert any("recovered mutated owner integrity" in note for note in primary.__notes__)
    fresh = _parts(monkeypatch, runtime=parts.runtime, decoder_mode="dynamic")
    replacement = api.run_repeated_d4_encoder_selected_backward(
        fresh.owner, fresh.authority, fresh.completion
    )
    replacement.abort()


def test_leader_reconstructs_multi_chunk_manifest_order_and_offsets(monkeypatch):
    parts = _parts(monkeypatch, chunks=2)
    try:
        gradients = api._leader_gradients(parts.carrier, parts.forward_operations)
        assert len(gradients) == 2
        for actual, expected in zip(gradients, parts.expected, strict=True):
            torch.testing.assert_close(actual, expected)
            assert actual.is_contiguous() and actual.storage_offset() == 0
    finally:
        parts.owner.abort()


@pytest.mark.parametrize(
    ("corruption", "error_type", "message"),
    (
        ("missing", MdpPlanError, "exact canonical encoder gradient routes"),
        ("extra", MdpPlanError, "exact canonical encoder gradient routes"),
        ("duplicate_endpoint", MdpPlanError, "exact canonical encoder gradient routes"),
        ("geometry", MdpStateError, "item gradients match encoder output geometry"),
        ("offset", MdpPlanError, "encoder layouts densely cover chunk outputs"),
    ),
)
def test_malformed_routes_fail_before_later_gate_stages(
    monkeypatch, corruption, error_type, message
):
    parts = _parts(monkeypatch, corruption=corruption)
    with pytest.raises(error_type, match=message):
        api.run_repeated_d4_encoder_selected_backward(
            parts.owner, parts.authority, parts.completion
        )
    assert "domain" not in parts.events and parts.handle._backward_done is False
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS


@pytest.mark.parametrize(
    "corruption",
    (
        "follower_output",
        "follower_layout",
        "follower_empty",
        "follower_width",
        "follower_dtype",
        "follower_noncontiguous",
        "follower_device",
    ),
)
def test_follower_handle_is_fully_validated_before_later_gate_stages(monkeypatch, corruption):
    parts = _parts(monkeypatch, rank=1, corruption=corruption)
    with pytest.raises((MdpPlanError, MdpStateError)):
        api.run_repeated_d4_encoder_selected_backward(
            parts.owner, parts.authority, parts.completion
        )
    assert parts.events == [("gate", 5, None)]
    assert parts.handle._backward_done is False
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS


@pytest.mark.parametrize(
    "mode",
    (
        "first",
        "final",
        "substitute",
        "mutate_prepared",
        "reenter",
        "callback_abort",
        "claim_before",
        "claim_after",
        "claim_foreign",
    ),
)
def test_gate5_rejection_substitution_reentry_cleanup_and_retry(monkeypatch, mode):
    parts = _parts(monkeypatch)
    primary = MdpPlanError(f"{mode} rejection")
    foreign = object()
    foreign_identity = []
    if mode in ("claim_before", "claim_after", "claim_foreign"):
        original_claim = gate4._D4EncoderBackwardAuthorizationOwner._claim_for_selected_backward

        def claim(owner, successor, registry, *args):
            foreign_identity.append(id(successor))
            if mode == "claim_foreign":
                registry[id(successor)] = foreign
            if mode != "claim_before" and mode != "claim_foreign":
                original_claim(owner, successor, registry, *args)
            elif mode == "claim_foreign":
                with pytest.raises(MdpStateError, match="normalized Gate4 ownership"):
                    original_claim(owner, successor, registry, *args)
            raise primary

        monkeypatch.setattr(
            gate4._D4EncoderBackwardAuthorizationOwner, "_claim_for_selected_backward", claim
        )

    def runner(_binding, _authority, **kwargs):
        prepared = kwargs["prepare"]()
        assert parts.handle._backward_done is False
        if mode == "reenter":
            with pytest.raises(MdpStateError, match="one-shot"):
                kwargs["prepare"]()
            raise primary
        if mode == "first":
            raise primary
        if mode == "callback_abort":
            prepared.owner.abort(primary)
            with pytest.raises(MdpStateError, match="retired"):
                prepared.owner.require()
            return prepared
        if mode == "mutate_prepared":
            object.__setattr__(prepared, "gradients", ())
        kwargs["domain_collective"](prepared)
        if mode == "final":
            raise primary
        return object()

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    expected = (
        MdpTaskFatalError
        if mode in ("substitute", "mutate_prepared", "callback_abort")
        else MdpPlanError
    )
    with pytest.raises(expected):
        api.run_repeated_d4_encoder_selected_backward(
            parts.owner, parts.authority, parts.completion
        )
    assert parts.handle._backward_done is False
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.operations.released == list(parts.buffers)
    for registry in (
        gate4._OWNER_ESCROWS,
        gate4._TRUSTED_OWNER_ESCROWS,
        gate4._ACTIVATED_OWNER_ESCROWS,
        gate4._TRUSTED_ACTIVATED_OWNER_ESCROWS,
    ):
        assert id(parts.owner) not in registry
    if mode == "claim_foreign":
        assert api._ACTIVE_OWNERS[foreign_identity[0]] is foreign
        del api._ACTIVE_OWNERS[foreign_identity[0]]
    if mode in ("first", "claim_before", "claim_after", "claim_foreign"):
        if mode.startswith("claim_"):
            monkeypatch.setattr(
                gate4._D4EncoderBackwardAuthorizationOwner,
                "_claim_for_selected_backward",
                original_claim,
            )
        fresh = _parts(monkeypatch, runtime=parts.runtime)
        owner = api.run_repeated_d4_encoder_selected_backward(
            fresh.owner, fresh.authority, fresh.completion
        )
        assert owner.require() is owner
        owner.abort()


def test_follower_partial_gradient_allocation_failure_cleans_and_reuses_runtime(monkeypatch):
    parts = _parts(monkeypatch, rank=1, chunks=2)
    original = torch.zeros_like
    calls = 0

    def fail_second(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second gradient allocation failed")
        return original(value)

    monkeypatch.setattr(api.torch, "zeros_like", fail_second)
    with pytest.raises(RuntimeError, match="second gradient allocation failed"):
        api.run_repeated_d4_encoder_selected_backward(
            parts.owner, parts.authority, parts.completion
        )
    assert parts.events == [("gate", 5, None)]
    assert parts.handle._backward_done is False
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    monkeypatch.setattr(api.torch, "zeros_like", original)
    fresh = _parts(monkeypatch, rank=1, runtime=parts.runtime)
    owner = api.run_repeated_d4_encoder_selected_backward(
        fresh.owner, fresh.authority, fresh.completion
    )
    assert owner.require() is owner
    owner.abort()


def test_registry_mutation_during_gradient_preparation_cannot_partially_activate(monkeypatch):
    parts = _parts(monkeypatch, rank=1)
    original = torch.zeros_like

    def mutate_registry(value):
        gate4._ACTIVE_CARRIERS.pop(id(parts.carrier))
        return original(value)

    monkeypatch.setattr(api.torch, "zeros_like", mutate_registry)
    with pytest.raises(MdpStateError, match="exact Gate3 capabilities"):
        api.run_repeated_d4_encoder_selected_backward(
            parts.owner, parts.authority, parts.completion
        )
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert not any(entry[1]() is parts.owner for entry in api._ACTIVE_OWNERS.values())
    monkeypatch.setattr(api.torch, "zeros_like", original)
    fresh = _parts(monkeypatch, rank=1, runtime=parts.runtime)
    owner = api.run_repeated_d4_encoder_selected_backward(
        fresh.owner, fresh.authority, fresh.completion
    )
    assert owner.require() is owner
    owner.abort()


@pytest.mark.parametrize("mode", ("delete", "substitute"))
def test_runtime_slot_must_still_name_exact_predecessor(monkeypatch, mode):
    parts = _parts(monkeypatch)
    foreign = _Operations()
    if mode == "delete":
        replay_api._forward._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))
    else:
        replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = (
            parts.runtime,
            weakref.ref(foreign),
        )
    with pytest.raises(MdpStateError, match="exact predecessor registries"):
        api.run_repeated_d4_encoder_selected_backward(
            parts.owner, parts.authority, parts.completion
        )
    current = replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime))
    if mode == "substitute":
        assert current is not None and current[1]() is foreign
        replay_api._forward._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))
    else:
        assert current is None
    fresh = _parts(monkeypatch, runtime=parts.runtime)
    owner = api.run_repeated_d4_encoder_selected_backward(
        fresh.owner, fresh.authority, fresh.completion
    )
    assert owner.require() is owner
    owner.abort()


def test_backward_baseexception_is_task_fatal_and_binding_cleanup_runs(monkeypatch):
    parts = _parts(monkeypatch)
    primary = KeyboardInterrupt("backward failed")
    parts.handle.backward = lambda _gradients: (_ for _ in ()).throw(primary)
    with pytest.raises(MdpTaskFatalError, match="failed after Gate5") as caught:
        api.run_repeated_d4_encoder_selected_backward(
            parts.owner, parts.authority, parts.completion
        )
    assert caught.value.__cause__ is primary
    assert parts.restored == [caught.value]
    assert parts.operations.released == list(parts.buffers)
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
