# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gate4 encoder-only backward authorization contracts."""

import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_dynamic_decoder_replay as dynamic_replay_api
from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as api
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as gradient_api
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay_api
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_plan import (
    EncoderDynamicPlan,
    EncoderExecution,
    EncoderExecutionWave,
)
from megatron.core.mdp.errors import MdpPlanError, MdpStateError, MdpTaskFatalError


class _Operations:
    def __init__(self):
        self.acquire_calls = 0
        self.released = []
        self.observe = lambda: None

        def acquire(**_kwargs):
            self.acquire_calls += 1
            raise AssertionError("Gate4 must not acquire replay resources")

        def release(value):
            self.observe()
            self.released.append(value)

        self.acquire = acquire
        self.release = release


class _TupleClone(tuple):
    pass


class _BindingOwner:
    def __init__(self, ranks):
        self.membership = SimpleNamespace(ranks=ranks)
        self.restored = []

    @property
    def active(self):
        return not self.restored

    def restore(self, primary=None):
        self.restored.append(primary)


class _Handle:
    def __init__(self):
        self.released = 0
        self._released = False
        self._backward_done = False

    def release_forward_only(self):
        self.released += 1
        self._released = True


def _parts(
    monkeypatch, *, rank=0, selected_size=2, text_only=False, runtime=None, decoder_mode="fixed"
):
    ranks = (0, 1, 2, 3)
    selected = () if text_only else ranks[:selected_size]
    item_id = GlobalVisionItemId(0, 0)
    items = () if text_only else (SimpleNamespace(item_id=item_id),)
    execution = EncoderExecution(
        group_size=selected_size,
        group_index=0,
        rank_slots=tuple(range(selected_size)),
        item_ids=tuple(item.item_id for item in items),
        effective_rows_per_rank=1,
        cost_units=1,
    )
    plan = EncoderDynamicPlan(
        source_samples=(),
        pool_ranks=ranks,
        max_seqlen_per_rank=1,
        waves=() if text_only else (EncoderExecutionWave(0, (execution,)),),
        digest=b"e" * 16,
    )
    incoming_key = DynamicBridgeKey(item_id, 3)
    entries = (
        ()
        if text_only
        else (SimpleNamespace(key=incoming_key, src_global_rank=3, dst_global_rank=0),)
    )
    authority = SimpleNamespace(
        encoder_plan=plan,
        participant_ranks=ranks,
        global_manifest=SimpleNamespace(items=items),
        gradient_ledger=SimpleNamespace(entries=entries),
    )
    binding = SimpleNamespace(global_rank=rank)
    runtime = runtime or SimpleNamespace()
    operations = _Operations()
    is_selected = rank in selected
    binding_owner = _BindingOwner(selected) if is_selected else None
    handle = _Handle() if is_selected else None
    predecessor_buffers = (object(), object())
    leaf_bases = (object(),)
    transport_buffers = (object(), object())
    handoff_resources = (binding_owner, predecessor_buffers, operations, handle)
    token = torch.tensor(1.0)
    completion_provenance = object()
    predecessor_entry = None if decoder_mode == "fixed" else object()
    completion_lifecycle = None if decoder_mode == "fixed" else object()
    mode = replay_api if decoder_mode == "fixed" else dynamic_replay_api
    if decoder_mode == "fixed":
        completion = replay_api._D4FixedDecoderCompletion(
            authority, token, completion_provenance, replay_api._COMPLETION_SEAL
        )
    else:
        completion = dynamic_replay_api._D4DynamicDecoderCompletion(
            authority,
            token,
            completion_provenance,
            predecessor_entry,
            completion_lifecycle,
            dynamic_replay_api._COMPLETION_SEAL,
        )
    received = (
        MappingProxyType({incoming_key: torch.ones(2, 3)})
        if rank == 0 and not text_only
        else MappingProxyType({})
    )
    receipt = gradient_api._D4EncoderOnlyGradientReceipt(
        authority,
        completion,
        SimpleNamespace(received_tensors=received),
        received,
        gradient_api._RECEIPT_SEAL,
    )
    trusted = (
        runtime,
        authority,
        binding,
        receipt,
        completion,
        (),
        MappingProxyType({}),
        handoff_resources,
        leaf_bases,
        transport_buffers,
        operations,
        completion_provenance,
        text_only,
        is_selected,
    )
    predecessor = gradient_api._D4EncoderGradientRouteOwner(trusted, gradient_api._OWNER_SEAL)
    reference = weakref.ref(predecessor)
    receipt_entry = (reference, receipt, authority, completion, receipt.exchange, received)
    completion_entry = (
        (reference, completion, authority, token, replay_api._tensor_descriptor(token))
        if decoder_mode == "fixed"
        else (
            reference,
            completion,
            completion_provenance,
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
        completion_provenance,
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
    assert provenance.acquire is operations.acquire
    assert provenance.release is operations.release
    assert completion_escrow[3] is provenance.release
    assert operations.acquire_calls == 0
    trusted = (*trusted, receipt_entry, completion_entry, completion_escrow)
    predecessor._trusted = trusted
    gradient_api._ACTIVE_OWNERS[id(predecessor)] = (reference, *trusted)
    gradient_api._ACTIVE_RECEIPTS[id(receipt)] = receipt_entry
    mode._ACTIVE_COMPLETIONS[id(completion)] = completion_entry
    replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda binding, authority: authority)
    monkeypatch.setattr(api, "validate_encoder_dynamic_plan", lambda plan: plan)
    monkeypatch.setattr(api, "validate_prepared_dynamic_bridge_exchange", lambda value: value)
    monkeypatch.setattr(api, "DynamicEncoderCpBinding", _BindingOwner)
    monkeypatch.setattr(api, "EncoderForwardHandle", _Handle)
    events = []

    def runner(_binding, _authority, **kwargs):
        events.append(("begin", kwargs["gate_id"], kwargs["byte_generator"]))
        candidate = kwargs["prepare"]()
        events.append("world0")
        candidate = kwargs["domain_collective"](candidate)
        events.extend(("domain", "world1"))
        return candidate

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    return SimpleNamespace(
        runtime=runtime,
        authority=authority,
        binding=binding,
        predecessor=predecessor,
        receipt=receipt,
        completion=completion,
        completion_registry=mode._ACTIVE_COMPLETIONS,
        selected=selected,
        binding_owner=binding_owner,
        handle=handle,
        operations=operations,
        all_buffers=(*transport_buffers, *leaf_bases, *predecessor_buffers),
        incoming_key=incoming_key,
        events=events,
    )


@pytest.mark.parametrize("decoder_mode", ("fixed", "dynamic"))
@pytest.mark.parametrize(
    ("rank", "selected_size", "text_only", "member", "leader"),
    (
        (0, 1, False, True, True),
        (0, 2, False, True, True),
        (0, 4, False, True, True),
        (1, 4, False, True, False),
        (2, 2, False, False, False),
        (0, 1, True, False, False),
    ),
)
def test_gate4_arms_exact_member_follower_nonmember_and_text_carriers(
    monkeypatch, rank, selected_size, text_only, member, leader, decoder_mode
):
    parts = _parts(
        monkeypatch,
        rank=rank,
        selected_size=selected_size,
        text_only=text_only,
        decoder_mode=decoder_mode,
    )
    marker = object()
    owner = api.run_repeated_d4_encoder_backward_authorization(
        parts.predecessor, parts.authority, parts.completion, byte_generator=marker
    )
    assert parts.events == [("begin", 4, marker), "world0", "domain", "world1"]
    assert owner.require() is owner
    assert owner.binding is parts.binding
    assert owner.receipt is parts.receipt and owner.completion is parts.completion
    completion_escrow = owner._trusted[10]
    assert completion_escrow[1] is parts.completion_registry
    assert parts.completion_registry[id(parts.completion)] is completion_escrow[2]
    assert completion_escrow[4].release is parts.operations.release
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is owner
    with pytest.raises(MdpStateError, match="gradient route owner is retired"):
        parts.predecessor.require()
    if member:
        carrier = owner.require_carrier(owner.carrier)
        assert type(carrier) is api._D4EncoderBackwardMember
        assert carrier.selected_ranks == parts.selected
        assert carrier.member_index == parts.selected.index(rank)
        assert carrier.is_leader is leader
        assert carrier.forward_handle is parts.handle
        assert tuple(carrier.routed_gradients) == ((parts.incoming_key,) if leader else ())
    else:
        carrier = owner.require_carrier(owner.carrier)
        assert type(carrier) is api._D4EncoderBackwardEmpty
        assert carrier.selected_ranks == parts.selected
        assert carrier.text_only is text_only
    assert parts.handle is None or parts.handle.released == 0
    assert parts.operations.released == []
    owner.abort()
    assert id(parts.completion) not in parts.completion_registry
    assert parts.handle is None or parts.handle.released == 1
    assert parts.operations.released == list(parts.all_buffers)


@pytest.mark.parametrize("decoder_mode", ("fixed", "dynamic"))
@pytest.mark.parametrize("mutation", ("mode", "release", "token"))
def test_gate4_normalized_completion_provenance_mutation_rejects_and_cleans(
    monkeypatch, mutation, decoder_mode
):
    parts = _parts(monkeypatch, decoder_mode=decoder_mode)
    owner = api.run_repeated_d4_encoder_backward_authorization(
        parts.predecessor, parts.authority, parts.completion
    )
    provenance = owner._trusted[10][4]
    if mutation == "mode":
        object.__setattr__(provenance, "mode", object())
    elif mutation == "release":
        object.__setattr__(provenance, "release", lambda _value: None)
    else:
        object.__setattr__(parts.completion, "globally_reduced_num_tokens", torch.tensor(2.0))
    with pytest.raises(MdpStateError, match="normalized completion provenance"):
        owner.require()
    owner.abort()
    assert id(parts.completion) not in parts.completion_registry
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.handle.released == 1
    assert parts.operations.released == list(parts.all_buffers)


def test_gate4_preserves_foreign_completion_registry_during_trusted_cleanup(monkeypatch):
    parts = _parts(monkeypatch)
    owner = api.run_repeated_d4_encoder_backward_authorization(
        parts.predecessor, parts.authority, parts.completion
    )
    foreign = object()
    replay_api._ACTIVE_COMPLETIONS[id(parts.completion)] = foreign
    owner.abort()
    assert replay_api._ACTIVE_COMPLETIONS[id(parts.completion)] is foreign
    assert parts.operations.released == list(parts.all_buffers)
    del replay_api._ACTIVE_COMPLETIONS[id(parts.completion)]


def test_gate4_owner_cleanup_ignores_same_self_registry_and_escrow_substitution(monkeypatch):
    parts = _parts(monkeypatch, decoder_mode="dynamic")
    owner = api.run_repeated_d4_encoder_backward_authorization(
        parts.predecessor, parts.authority, parts.completion
    )
    foreign_escrow = object()
    foreign_entry = (weakref.ref(owner), *owner._trusted[:-1], object())
    foreign_runtime = (parts.runtime, weakref.ref(owner))
    api._OWNER_ESCROWS[id(owner)] = foreign_escrow
    api._TRUSTED_OWNER_ESCROWS[id(owner)] = foreign_escrow
    api._ACTIVATED_OWNER_ESCROWS[id(owner)] = foreign_escrow
    api._ACTIVE_OWNERS[id(owner)] = foreign_entry
    replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = foreign_runtime

    def observe():
        assert api._OWNER_ESCROWS[id(owner)] is foreign_escrow
        assert api._TRUSTED_OWNER_ESCROWS[id(owner)] is foreign_escrow
        assert api._ACTIVATED_OWNER_ESCROWS[id(owner)] is foreign_escrow
        assert api._ACTIVE_OWNERS[id(owner)] is foreign_entry
        assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
        with pytest.raises(MdpStateError, match="retired"):
            owner.abort()

    parts.operations.observe = observe
    owner.abort()
    assert api._OWNER_ESCROWS[id(owner)] is foreign_escrow
    assert api._TRUSTED_OWNER_ESCROWS[id(owner)] is foreign_escrow
    assert api._ACTIVATED_OWNER_ESCROWS[id(owner)] is foreign_escrow
    assert api._ACTIVE_OWNERS[id(owner)] is foreign_entry
    assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] is foreign_runtime
    assert parts.operations.released == list(parts.all_buffers)
    assert id(parts.completion) not in parts.completion_registry
    del api._OWNER_ESCROWS[id(owner)]
    del api._TRUSTED_OWNER_ESCROWS[id(owner)]
    del api._ACTIVATED_OWNER_ESCROWS[id(owner)]
    del api._ACTIVE_OWNERS[id(owner)]
    del replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)]


@pytest.mark.parametrize("mutation", ("delete", "substitute"))
def test_gate4_activation_then_failure_uses_local_escrow_and_allows_retry(monkeypatch, mutation):
    parts = _parts(monkeypatch, decoder_mode="dynamic")
    primary = RuntimeError("activation failed")
    original = api._D4EncoderBackwardAuthorizationOwner._claim_prepared
    foreign = object()
    activated = []

    def claim(owner, predecessor, escrow):
        original(owner, predecessor, escrow)
        activated.append(owner)
        if mutation == "delete":
            del api._ACTIVE_OWNERS[id(owner)]
            del api._OWNER_ESCROWS[id(owner)]
        else:
            api._ACTIVE_OWNERS[id(owner)] = foreign
            api._OWNER_ESCROWS[id(owner)] = foreign
        raise primary

    monkeypatch.setattr(api._D4EncoderBackwardAuthorizationOwner, "_claim_prepared", claim)
    with pytest.raises(RuntimeError) as caught:
        api.run_repeated_d4_encoder_backward_authorization(
            parts.predecessor, parts.authority, parts.completion
        )
    assert caught.value is primary
    if mutation == "substitute":
        assert api._ACTIVE_OWNERS[id(activated[0])] is foreign
        assert api._OWNER_ESCROWS[id(activated[0])] is foreign
        del api._ACTIVE_OWNERS[id(activated[0])]
        del api._OWNER_ESCROWS[id(activated[0])]
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.operations.released == list(parts.all_buffers)
    monkeypatch.setattr(api._D4EncoderBackwardAuthorizationOwner, "_claim_prepared", original)
    fresh = _parts(monkeypatch, runtime=parts.runtime, decoder_mode="dynamic")
    owner = api.run_repeated_d4_encoder_backward_authorization(
        fresh.predecessor, fresh.authority, fresh.completion
    )
    owner.abort()


@pytest.mark.parametrize("mode", ("first", "final", "substitute", "reenter", "callback_abort"))
def test_gate4_rejection_and_runner_substitution_cleanup_without_transfer(monkeypatch, mode):
    parts = _parts(monkeypatch)
    primary = MdpPlanError(f"{mode} rejected")

    def runner(_binding, _authority, **kwargs):
        candidate = kwargs["prepare"]()
        if mode == "reenter":
            with pytest.raises(MdpStateError, match="prepares once"):
                kwargs["prepare"]()
            raise primary
        if mode == "callback_abort":
            with pytest.raises(MdpStateError, match="gradient route owner is retired"):
                parts.predecessor.abort()
            raise primary
        if mode == "substitute":
            return object()
        if mode == "first":
            raise primary
        kwargs["domain_collective"](candidate)
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    expected = MdpTaskFatalError if mode == "substitute" else MdpPlanError
    with pytest.raises(expected) as caught:
        api.run_repeated_d4_encoder_backward_authorization(
            parts.predecessor, parts.authority, parts.completion
        )
    if mode != "substitute":
        assert caught.value is primary
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.handle.released == 1
    assert parts.operations.released == list(parts.all_buffers)
    if mode == "first":
        fresh = _parts(monkeypatch, runtime=parts.runtime)
        owner = api.run_repeated_d4_encoder_backward_authorization(
            fresh.predecessor, fresh.authority, fresh.completion
        )
        assert owner.require() is owner
        owner.abort()


def test_gate4_validation_failure_is_pre_final_and_nonreentrant(monkeypatch):
    parts = _parts(monkeypatch)
    object.__setattr__(parts.authority.encoder_plan.waves[0].executions[0], "rank_slots", (1, 0))
    stages = []

    def runner(_binding, _authority, **kwargs):
        stages.append("begin")
        with pytest.raises(MdpPlanError, match="manifest-order prefix"):
            kwargs["prepare"]()
        stages.append("world0")
        raise MdpPlanError("rejected rank 0 with error code 1")

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpPlanError, match="rejected rank 0"):
        api.run_repeated_d4_encoder_backward_authorization(
            parts.predecessor, parts.authority, parts.completion
        )
    assert stages == ["begin", "world0"]
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS


def test_foreign_inputs_are_nonconsuming_and_live_mutation_cleans_up(monkeypatch):
    parts = _parts(monkeypatch)
    with pytest.raises(MdpStateError, match="exact authority and completion"):
        api.run_repeated_d4_encoder_backward_authorization(
            parts.predecessor, object(), parts.completion
        )
    with pytest.raises(MdpStateError, match="exact authority and completion"):
        api.run_repeated_d4_encoder_backward_authorization(
            parts.predecessor, parts.authority, object()
        )
    assert parts.predecessor.require() is parts.predecessor
    parts.predecessor.is_selected = False
    with pytest.raises(MdpStateError, match="sealed resources"):
        api.run_repeated_d4_encoder_backward_authorization(
            parts.predecessor, parts.authority, parts.completion
        )
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.handle.released == 1


def test_authorized_owner_hostile_field_cleanup_uses_registry_escrow(monkeypatch):
    parts = _parts(monkeypatch)
    owner = api.run_repeated_d4_encoder_backward_authorization(
        parts.predecessor, parts.authority, parts.completion
    )
    primary = RuntimeError("primary")
    del owner.authority
    owner.abort(primary)
    assert parts.handle.released == 1
    assert parts.operations.released == list(parts.all_buffers)
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    with pytest.raises(MdpStateError, match="retired"):
        owner.abort()
    assert any("integrity" in note for note in primary.__notes__)


def test_authorized_owner_rejects_binding_substitution_and_trusted_abort(monkeypatch):
    parts = _parts(monkeypatch)
    owner = api.run_repeated_d4_encoder_backward_authorization(
        parts.predecessor, parts.authority, parts.completion
    )
    owner.binding = SimpleNamespace(global_rank=parts.binding.global_rank)
    with pytest.raises(MdpStateError, match="sealed resources"):
        owner.require()
    owner.abort()
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.operations.released == list(parts.all_buffers)


@pytest.mark.parametrize("mutation", ("binding", "released", "backward", "received"))
def test_status_callback_resource_mutation_is_post_world_task_fatal(monkeypatch, mutation):
    parts = _parts(monkeypatch)

    def runner(_binding, _authority, **kwargs):
        carrier = kwargs["prepare"]()
        kwargs["domain_collective"](carrier)
        if mutation == "binding":
            parts.binding_owner.restored.append(None)
        elif mutation == "released":
            parts.handle._released = True
        elif mutation == "backward":
            parts.handle._backward_done = True
        else:
            object.__setattr__(parts.receipt, "received_tensors", MappingProxyType({}))
        return carrier

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpTaskFatalError, match="post-WORLD result"):
        api.run_repeated_d4_encoder_backward_authorization(
            parts.predecessor, parts.authority, parts.completion
        )
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.operations.released == list(parts.all_buffers)


@pytest.mark.parametrize(
    ("text_only", "field", "value"),
    (
        (False, "selected_ranks", (9,)),
        (False, "selected_ranks", _TupleClone((0,))),
        (False, "member_index", 3),
        (False, "member_index", False),
        (False, "is_leader", False),
        (True, "selected_ranks", (0,)),
        (True, "selected_ranks", _TupleClone(())),
        (True, "text_only", False),
    ),
)
def test_armed_carrier_role_mutation_is_rejected_and_trusted_cleanup_runs(
    monkeypatch, text_only, field, value
):
    parts = _parts(monkeypatch, selected_size=1, text_only=text_only)
    owner = api.run_repeated_d4_encoder_backward_authorization(
        parts.predecessor, parts.authority, parts.completion
    )
    object.__setattr__(owner.carrier, field, value)
    with pytest.raises(MdpStateError, match="sealed authorization|sealed authorization|empty"):
        owner.require_carrier(owner.carrier)
    owner.abort()
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    assert parts.operations.released == list(parts.all_buffers)
