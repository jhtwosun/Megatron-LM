# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gate5 selected encoder backward contracts."""

import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as gradient_api
from megatron.core.mdp import dynamic_cp_d4_encoder_selected_backward as api
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay_api
from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.errors import MdpPlanError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment
from megatron.core.mdp.protocols import DynamicEncoderCpBinding


class _Operations:
    def __init__(self):
        self.released = []

    def release(self, value):
        self.released.append(value)


def _parts(
    monkeypatch,
    *,
    rank=0,
    selected_size=2,
    text_only=False,
    chunks=1,
    runtime=None,
    corruption=None,
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
        lambda *_args: None,
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
    token = torch.tensor(1.0)
    completion = replay_api._D4FixedDecoderCompletion(
        authority, token, object(), replay_api._COMPLETION_SEAL
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
    gate4._ACTIVE_OWNERS[id(owner)] = (reference, *trusted)
    role = (
        (selected_ranks, rank, rank == 0, handle, receipt.received_tensors)
        if selected
        else (selected_ranks, text_only)
    )
    gate4._ACTIVE_CARRIERS[id(carrier)] = (
        reference,
        carrier,
        authority,
        completion,
        receipt,
        *role,
    )
    gradient_api._ACTIVE_RECEIPTS[id(receipt)] = (
        reference,
        receipt,
        authority,
        completion,
        receipt.exchange,
        receipt.received_tensors,
    )
    replay_api._ACTIVE_COMPLETIONS[id(completion)] = (
        reference,
        completion,
        authority,
        token,
        replay_api._tensor_descriptor(token),
    )
    replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    monkeypatch.setattr(gate4, "_snapshot_local_authority", lambda b, a: a)
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda b, a: a)
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
        carrier=carrier,
        handle=handle,
        expected=tuple(expected),
        events=events,
        operations=release_tracker,
        forward_operations=operations,
        buffers=(*transport, *leaf_bases, *buffers),
        restored=restored,
    )


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
    monkeypatch, rank, size, text_only, selected
):
    parts = _parts(monkeypatch, rank=rank, selected_size=size, text_only=text_only, chunks=2)
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
    assert (parts.handle is not None and parts.handle._backward_done) is selected
    assert parts.restored == [] and parts.operations.released == []
    owner.abort()


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


@pytest.mark.parametrize("corruption", ("missing", "extra", "geometry", "offset"))
def test_malformed_routes_fail_before_later_gate_stages(monkeypatch, corruption):
    parts = _parts(monkeypatch, corruption=corruption)
    with pytest.raises((MdpPlanError, MdpStateError)):
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
    "mode", ("first", "final", "substitute", "mutate_prepared", "reenter", "callback_abort")
)
def test_gate5_rejection_substitution_reentry_cleanup_and_retry(monkeypatch, mode):
    parts = _parts(monkeypatch)
    primary = MdpPlanError(f"{mode} rejection")

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
    if mode == "first":
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
