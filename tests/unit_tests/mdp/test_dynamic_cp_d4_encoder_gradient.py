# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Encoder-only Gate3 gradient routing and ownership tests."""

import copy
import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as api
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay_api
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_bridge_transport import PreparedDynamicBridgeExchange
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _AUTHORITY_SEAL,
    _BINDING_SEAL,
    _RepeatedD4GroupAuthority,
    _RepeatedD4GroupBinding,
)
from megatron.core.mdp.dynamic_cp_execution import DecoderMicrobatchKey
from megatron.core.mdp.errors import MdpPlanError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord


class _Group:
    def __init__(self, rank=0):
        self._rank = rank

    def size(self):
        return 4

    def rank(self):
        return self._rank


class _Allocator:
    def __init__(self):
        self.acquired = []
        self.released = []
        self.fail_release = False
        self.fail_acquire_at = None
        self.acquire_error = None

    def acquire(self, *, rows, width, dtype, device, tag):
        if self.fail_acquire_at == len(self.acquired):
            raise self.acquire_error
        tensor = torch.empty(rows if width == 0 else (rows, width), dtype=dtype, device=device)
        self.acquired.append((tag, tensor))
        return tensor

    def release(self, tensor):
        self.released.append(tensor)
        if self.fail_release:
            raise RuntimeError("release failed")


class _BindingOwner:
    def __init__(self):
        self.active = True

    def restore(self, _primary=None):
        self.active = False


class _Handle:
    def __init__(self):
        self.consumed = False

    def release_forward_only(self):
        self.consumed = True


def _binding(group, rank=0):
    authority = _RepeatedD4GroupAuthority(
        world_ranks=(0, 1, 2, 3),
        domain_ranks=(0, 1, 2, 3),
        global_rank=rank,
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
):
    group = _Group(rank)
    binding = _binding(group, rank)
    item_id = GlobalVisionItemId(0, 0)
    route_key = DynamicBridgeKey(item_id, rank)
    received_key = DynamicBridgeKey(item_id, (rank - 1) % 4)
    items = (SimpleNamespace(item_id=item_id, output_rows=2),) if vision else ()
    entries = (
        tuple(
            SimpleNamespace(
                key=DynamicBridgeKey(item_id, source),
                src_global_rank=source,
                dst_global_rank=(source + 1) % 4,
            )
            for source in range(4)
        )
        if vision
        else ()
    )
    authority = SimpleNamespace(
        plan=object(),
        global_manifest=SimpleNamespace(items=items),
        producer_rank_by_item=MappingProxyType({item_id: 1}) if vision else MappingProxyType({}),
        output_rows_by_item=MappingProxyType({item_id: 2}) if vision else MappingProxyType({}),
        embedding_ledger=SimpleNamespace(entries=entries),
        gradient_ledger=SimpleNamespace(entries=entries),
        participant_ranks=(0, 1, 2, 3),
        bridge_width=3,
        bridge_dtype=torch.float32,
    )
    vision_records = (
        (MdpMicrobatchVisionRecord(item_id, 0, 0, (1, 1, 2), 2, (0, 1)),) if vision else ()
    )
    record = MdpMicrobatchRecord(0, not vision, vision_records, object(), MappingProxyType({}))
    leaf = torch.ones((2, 3), requires_grad=True) if vision else None
    if leaf_corruption == "geometry":
        leaf = torch.ones((1, 3), requires_grad=True)
    if leaf is not None and not missing_grad:
        (leaf * 2.0).sum().backward()
    leaf_values = {DecoderMicrobatchKey(0): leaf} if vision else {}
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
    leaf_bases = (leaf,) if leaf is not None else ()
    predecessor = SimpleNamespace()
    token = torch.tensor(4.0)
    completion = replay_api._D4FixedDecoderCompletion(
        authority, token, predecessor, replay_api._COMPLETION_SEAL
    )
    trusted = (
        runtime,
        authority,
        binding,
        (record,),
        leaves,
        handoff_resources,
        leaf_bases,
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
    events = []
    prepared_calls = []
    physical_calls = []

    monkeypatch.setattr(replay_api, "_snapshot_local_authority", lambda b, a: a)
    monkeypatch.setattr(api, "_validate_repeated_d4_group_binding", lambda value: value._authority)
    monkeypatch.setattr(
        api,
        "dynamic_bridge_split_sizes",
        lambda *_args, **_kwargs: (
            (
                tuple(6 if index == (rank + 1) % 4 else 0 for index in range(4)),
                tuple(6 if index == (rank - 1) % 4 else 0 for index in range(4)),
            )
            if vision
            else ((0, 0, 0, 0), (0, 0, 0, 0))
        ),
    )

    def prepare_exchange(*args, **kwargs):
        prepared_calls.append((args, kwargs))
        received = (
            MappingProxyType({received_key: torch.empty((2, 3))})
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
        completion=completion,
        leaf=leaf,
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
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS.get(id(parts.runtime)) is None


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
