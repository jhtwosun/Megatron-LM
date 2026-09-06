# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gate6 restore and encoder-only finalization contracts."""

import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_bridge_transport as bridge_transport
from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as gradient_api
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient_finalize as api
from megatron.core.mdp import dynamic_cp_d4_encoder_selected_backward as gate5
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay_api
from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp_bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import PreparedDynamicBridgeExchange
from megatron.core.mdp.errors import MdpPlanError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.protocols import DynamicEncoderCpBinding
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState


class _Operations:
    def __init__(self, encoder_ddp, *, fail_release=False):
        self.released = []
        self.fail_release = fail_release
        self.on_release = None
        self.carrier = replay_api._forward._D4EncoderForwardOperations(
            object(),
            lambda *_args: None,
            self.release,
            torch.device("cpu"),
            torch.float32,
            object(),
            encoder_ddp,
            object(),
            lambda: None,
            lambda *_args: None,
            lambda *_args: None,
            1,
            2,
            torch.float32,
            replay_api._forward._OPERATIONS_SEAL,
        )

    def release(self, value):
        self.released.append(value)
        if self.on_release is not None:
            callback, self.on_release = self.on_release, None
            callback()
        if self.fail_release:
            raise RuntimeError("cleanup release failed")


def _runtime():
    runtime = object.__new__(MdpRuntime)
    runtime._state = MdpRuntimeState.EMPTY
    runtime._iteration = 7
    runtime._captured_num_tokens = None
    runtime._token_capture_count = 0
    runtime._token_consumed = False
    return runtime


def _parts(monkeypatch, *, role="member", restore_error=None, cleanup_error=False, runtime=None):
    runtime = runtime or _runtime()
    authority = SimpleNamespace()
    binding = SimpleNamespace()
    token = torch.tensor(9.0)
    encoder_ddp = object()
    operations = _Operations(encoder_ddp, fail_release=cleanup_error)
    restores = []

    def restore(primary):
        restores.append(primary)
        if restore_error is not None:
            raise restore_error

    selected = role == "member"
    text_only = role == "text"
    if selected:
        dynamic_binding = DynamicEncoderCpBinding(
            SimpleNamespace(), is_current=lambda: True, restore=restore
        )
        output = torch.ones(1, 1, requires_grad=True) * 2
        handle = EncoderForwardHandle(7, 0, (output,), (SimpleNamespace(total_output_rows=1),))
        handle._backward_done = True
    else:
        dynamic_binding = None
        handle = None
    completion = replay_api._D4FixedDecoderCompletion(
        authority, token, object(), replay_api._COMPLETION_SEAL
    )
    received = MappingProxyType({})
    exchange = PreparedDynamicBridgeExchange(
        BridgePhase.GRADIENT,
        torch.float32,
        0,
        (0,),
        (0,),
        (0,),
        bytes(16),
        torch.empty(0),
        torch.empty(0),
        received,
    )
    object.__setattr__(exchange, "_authority", bridge_transport._capture_authority(exchange))
    receipt = gradient_api._D4EncoderOnlyGradientReceipt(
        authority, completion, exchange, received, gradient_api._RECEIPT_SEAL
    )
    selected_ranks = (0,) if role != "text" else ()
    if selected:
        carrier = gate4._D4EncoderBackwardMember(
            authority,
            completion,
            receipt,
            selected_ranks,
            0,
            True,
            handle,
            received,
            gate4._MEMBER_SEAL,
        )
    else:
        carrier = gate4._D4EncoderBackwardEmpty(
            authority, completion, receipt, selected_ranks, text_only, gate4._EMPTY_SEAL
        )
    backward_completion = gate5._D4EncoderBackwardComplete(
        authority, completion, carrier, selected, text_only, gate5._COMPLETE_SEAL
    )
    buffers = (object(), object())
    leaf_bases = (object(),)
    transport = (object(),)
    resources = (dynamic_binding, buffers, operations.carrier, handle)
    trusted = (
        runtime,
        binding,
        authority,
        completion,
        carrier,
        backward_completion,
        resources,
        receipt,
        leaf_bases,
        transport,
        operations.carrier,
    )
    owner = gate5._D4EncoderSelectedBackwardOwner(trusted, gate5._OWNER_SEAL)
    owner._backward_done = True
    reference = weakref.ref(owner)
    gate5._ACTIVE_OWNERS[id(owner)] = (reference, *trusted)
    gate5._ACTIVE_COMPLETIONS[id(backward_completion)] = (
        reference,
        backward_completion,
        selected,
        text_only,
    )
    role_entry = (
        (selected_ranks, 0, True, handle, received) if selected else (selected_ranks, text_only)
    )
    gate4._ACTIVE_CARRIERS[id(carrier)] = (
        reference,
        carrier,
        authority,
        completion,
        receipt,
        *role_entry,
    )
    gradient_api._ACTIVE_RECEIPTS[id(receipt)] = (
        reference,
        receipt,
        authority,
        completion,
        exchange,
        received,
    )
    replay_api._ACTIVE_COMPLETIONS[id(completion)] = (
        reference,
        completion,
        authority,
        token,
        replay_api._tensor_descriptor(token),
    )
    replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    original_require = gate5._D4EncoderSelectedBackwardOwner.require
    require_calls = []

    def strict_require(self):
        require_calls.append(self)
        return original_require(self)

    monkeypatch.setattr(gate5._D4EncoderSelectedBackwardOwner, "require", strict_require)
    monkeypatch.setattr(gate5, "_snapshot_local_authority", lambda b, a: a)
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda b, a: a)
    events = []

    def runner(_binding, _authority, **kwargs):
        events.append(("gate", kwargs["gate_id"], kwargs["byte_generator"]))
        value = kwargs["prepare"]()
        events.append("world0")
        value = kwargs["domain_collective"](value)
        events.extend(("domain", "world1"))
        return value

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    finalizes = []
    monkeypatch.setattr(
        api._encoder,
        "finalize_encoder_grads",
        lambda ddp, *, globally_reduced_num_tokens: finalizes.append(
            (ddp, globally_reduced_num_tokens, tuple(events))
        ),
    )
    return SimpleNamespace(
        runtime=runtime,
        authority=authority,
        binding=binding,
        token=token,
        owner=owner,
        dynamic_binding=dynamic_binding,
        handle=handle,
        operations=operations,
        buffers=(*transport, *leaf_bases, *buffers),
        restores=restores,
        finalizes=finalizes,
        events=events,
        encoder_ddp=encoder_ddp,
        require_calls=require_calls,
    )


@pytest.mark.parametrize("role", ("member", "nonmember", "text"))
def test_gate6_restores_then_finalizes_after_world_domain_world(monkeypatch, role):
    parts = _parts(monkeypatch, role=role)
    marker = object()
    ready = api.run_repeated_d4_encoder_gradient_finalize(
        parts.owner, parts.authority, parts.owner.completion, byte_generator=marker
    )
    assert parts.events == [("gate", 6, marker), "world0", "domain", "world1"]
    assert parts.restores == ([None] if role == "member" else [])
    assert parts.dynamic_binding is None or parts.dynamic_binding.active is False
    assert parts.finalizes == [(parts.encoder_ddp, parts.token, tuple(parts.events))]
    assert ready.owner.require_commit_ready(ready) is ready
    assert parts.runtime._captured_num_tokens is parts.token
    assert parts.runtime._token_capture_count == 1 and parts.runtime._token_consumed is True
    assert parts.require_calls and all(value is parts.owner for value in parts.require_calls)
    ready.owner.abort()


@pytest.mark.parametrize("local_failure", (True, False))
def test_restore_failure_converges_at_first_world_and_never_retries(monkeypatch, local_failure):
    local = RuntimeError("restore failed") if local_failure else None
    parts = _parts(monkeypatch, restore_error=local, cleanup_error=local_failure)
    common = MdpPlanError("MDP: Gate6 rejected rank 0 with error code 1.")

    def runner(_binding, _authority, **kwargs):
        retained = None
        try:
            kwargs["prepare"]()
        except BaseException as error:
            retained = error
        parts.events.append("world0")
        if retained is not None:
            raise common from retained
        raise common

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpPlanError, match="rejected rank 0") as caught:
        api.run_repeated_d4_encoder_gradient_finalize(
            parts.owner, parts.authority, parts.owner.completion
        )
    assert caught.value is common
    assert caught.value.__cause__ is local
    assert parts.events == ["world0"]
    assert parts.restores == [None]
    assert parts.dynamic_binding.active is False
    assert parts.finalizes == []
    assert parts.operations.released == list(parts.buffers)
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS


@pytest.mark.parametrize("local_failure", (True, False))
def test_domain_integrity_failure_converges_at_final_world_without_finalize(
    monkeypatch, local_failure
):
    parts = _parts(monkeypatch)
    common = MdpPlanError("MDP: Gate6 rejected rank 0 with error code 1.")
    retained = None

    def runner(_binding, _authority, **kwargs):
        nonlocal retained
        value = kwargs["prepare"]()
        parts.events.append("world0")
        if local_failure:
            owner_entry = api._ACTIVE_OWNERS[id(value.owner)]
            object.__setattr__(owner_entry[5], "authority", object())
        try:
            kwargs["domain_collective"](value)
        except BaseException as error:
            retained = error
        parts.events.extend(("domain", "world1"))
        if retained is not None:
            raise common from retained
        raise common

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpPlanError, match="rejected rank 0") as caught:
        api.run_repeated_d4_encoder_gradient_finalize(
            parts.owner, parts.authority, parts.owner.completion
        )
    assert caught.value is common
    if local_failure:
        assert isinstance(caught.value.__cause__, MdpStateError)
    else:
        assert caught.value.__cause__ is None
    assert parts.events == ["world0", "domain", "world1"]
    assert parts.finalizes == [] and api._ACTIVE_READY == {}
    assert parts.runtime._captured_num_tokens is None
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS


@pytest.mark.parametrize(
    "mutation",
    (
        "iteration",
        "runtime_slot",
        "operations",
        "completion_authority",
        "backward_role",
        "receipt_exchange",
        "owner_registry",
        "carrier_authority",
        "replay_owner",
        "exchange_geometry",
        "handle_state",
        "release_callable",
    ),
)
def test_callback_mutation_blocks_physical_finalize_and_capability(monkeypatch, mutation):
    parts = _parts(monkeypatch)
    foreign = _Operations(object())

    def runner(_binding, _authority, **kwargs):
        value = kwargs["prepare"]()
        kwargs["domain_collective"](value)
        if mutation == "iteration":
            parts.runtime._iteration += 1
        elif mutation == "runtime_slot":
            replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = (
                parts.runtime,
                weakref.ref(foreign),
            )
        elif mutation == "operations":
            owner_entry = api._ACTIVE_OWNERS[id(value.owner)]
            object.__setattr__(owner_entry[11], "encoder_ddp", object())
        elif mutation == "completion_authority":
            object.__setattr__(value.owner.completion, "authority", object())
        elif mutation == "backward_role":
            object.__setattr__(value.owner.backward_completion, "selected", False)
        elif mutation == "receipt_exchange":
            owner_entry = api._ACTIVE_OWNERS[id(value.owner)]
            object.__setattr__(owner_entry[8], "exchange", object())
        elif mutation == "carrier_authority":
            owner_entry = api._ACTIVE_OWNERS[id(value.owner)]
            object.__setattr__(owner_entry[5], "authority", object())
        elif mutation == "replay_owner":
            object.__setattr__(value.owner.completion, "_owner", object())
        elif mutation == "exchange_geometry":
            owner_entry = api._ACTIVE_OWNERS[id(value.owner)]
            object.__setattr__(owner_entry[8].exchange, "route_authority_digest", bytes(16))
            object.__setattr__(owner_entry[8].exchange, "global_rank", 1)
        elif mutation == "handle_state":
            owner_entry = api._ACTIVE_OWNERS[id(value.owner)]
            owner_entry[7][3]._released = True
        elif mutation == "release_callable":
            owner_entry = api._ACTIVE_OWNERS[id(value.owner)]
            object.__setattr__(
                owner_entry[11],
                "release",
                lambda _value: (_ for _ in ()).throw(AssertionError("foreign release")),
            )
        else:
            api._ACTIVE_OWNERS[id(value.owner)] = (weakref.ref(foreign), foreign)
        return value

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpTaskFatalError, match="after Gate6"):
        api.run_repeated_d4_encoder_gradient_finalize(
            parts.owner, parts.authority, parts.owner.completion
        )
    assert parts.finalizes == [] and api._ACTIVE_READY == {}
    assert parts.runtime._captured_num_tokens is None
    if mutation == "runtime_slot":
        assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is foreign
        replay_api._forward._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))


@pytest.mark.parametrize("mode", ("first", "final", "substitute", "reenter"))
def test_rejection_substitution_reentry_cleanup_and_fresh_retry(monkeypatch, mode):
    parts = _parts(monkeypatch)
    primary = MdpPlanError(f"{mode} rejected")

    def runner(_binding, _authority, **kwargs):
        value = kwargs["prepare"]()
        assert parts.restores == [None] and parts.finalizes == []
        if mode == "reenter":
            with pytest.raises(MdpStateError, match="one-shot"):
                kwargs["prepare"]()
            raise primary
        if mode == "first":
            raise primary
        kwargs["domain_collective"](value)
        if mode == "final":
            raise primary
        return object()

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    expected = MdpTaskFatalError if mode == "substitute" else MdpPlanError
    with pytest.raises(expected):
        api.run_repeated_d4_encoder_gradient_finalize(
            parts.owner, parts.authority, parts.owner.completion
        )
    assert parts.restores == [None] and parts.finalizes == []
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    if mode == "first":
        fresh = _parts(monkeypatch, runtime=parts.runtime)
        ready = api.run_repeated_d4_encoder_gradient_finalize(
            fresh.owner, fresh.authority, fresh.owner.completion
        )
        ready.owner.abort()


def test_physical_failure_is_task_fatal_without_capability(monkeypatch):
    parts = _parts(monkeypatch)
    primary = KeyboardInterrupt("finalize failed")
    monkeypatch.setattr(
        api._encoder,
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    with pytest.raises(MdpTaskFatalError, match="after Gate6") as caught:
        api.run_repeated_d4_encoder_gradient_finalize(
            parts.owner, parts.authority, parts.owner.completion
        )
    assert caught.value.__cause__ is primary
    assert api._ACTIVE_READY == {}
    assert parts.runtime._captured_num_tokens is None
    assert parts.runtime._token_capture_count == 0 and parts.runtime._token_consumed is False


@pytest.mark.parametrize("mode", ("abort", "abort_raise", "delete_owner", "runtime_substitute"))
def test_finalizer_reentry_cannot_mint_or_leak_capability(monkeypatch, mode):
    parts = _parts(monkeypatch)
    primary = KeyboardInterrupt("finalizer callback failed")
    foreign = _Operations(object())

    def finalize(_ddp, *, globally_reduced_num_tokens):
        assert globally_reduced_num_tokens is parts.token
        successor = replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]()
        if mode in ("abort", "abort_raise"):
            successor.abort(primary)
            if mode == "abort_raise":
                raise primary
        elif mode == "delete_owner":
            api._ACTIVE_OWNERS.pop(id(successor))
        else:
            replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = (
                parts.runtime,
                weakref.ref(foreign),
            )

    monkeypatch.setattr(api._encoder, "finalize_encoder_grads", finalize)
    with pytest.raises(MdpTaskFatalError, match="after Gate6") as caught:
        api.run_repeated_d4_encoder_gradient_finalize(
            parts.owner, parts.authority, parts.owner.completion
        )
    if mode == "abort_raise":
        assert caught.value.__cause__ is primary
    assert api._ACTIVE_READY == {}
    assert parts.runtime._captured_num_tokens is None
    assert parts.runtime._token_capture_count == 0 and parts.runtime._token_consumed is False
    assert parts.operations.released == list(parts.buffers)
    if mode == "runtime_substitute":
        assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is foreign
        replay_api._forward._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))


def test_commit_capability_mutation_is_rejected_and_trusted_cleanup_rolls_back(monkeypatch):
    parts = _parts(monkeypatch)
    ready = api.run_repeated_d4_encoder_gradient_finalize(
        parts.owner, parts.authority, parts.owner.completion
    )
    object.__setattr__(ready, "token", torch.tensor(10.0))
    with pytest.raises(MdpStateError, match="exact encoder-only commit"):
        ready.owner.require_commit_ready(ready)
    ready.owner.abort()
    assert parts.runtime._captured_num_tokens is None
    assert parts.runtime._token_capture_count == 0 and parts.runtime._token_consumed is False


def test_live_ready_field_substitution_cannot_leak_registered_capability(monkeypatch):
    parts = _parts(monkeypatch)
    ready = api.run_repeated_d4_encoder_gradient_finalize(
        parts.owner, parts.authority, parts.owner.completion
    )
    owner = ready.owner
    owner.commit_ready = None
    owner.abort()
    assert api._ACTIVE_READY == {}
    assert ready.owner is None and ready.runtime is None and ready.token is None
    assert parts.runtime._captured_num_tokens is None


@pytest.mark.parametrize("role", ("member", "nonmember", "text"))
def test_claim_for_commit_releases_gate6_resources_and_preserves_token(monkeypatch, role):
    parts = _parts(monkeypatch, role=role)
    ready = api.run_repeated_d4_encoder_gradient_finalize(
        parts.owner, parts.authority, parts.owner.completion
    )
    owner = ready.owner
    backward_completion = owner.backward_completion
    carrier = owner._trusted[4]
    receipt = owner._trusted[7]
    completion = owner.completion
    handoff = api._claim_for_commit(owner, parts.authority, ready)
    assert handoff.require() is handoff
    assert ready.owner is handoff
    assert parts.runtime._captured_num_tokens is parts.token
    assert parts.runtime._token_capture_count == 1
    assert parts.runtime._token_consumed is True
    assert parts.handle is None or parts.handle._released is True
    assert parts.operations.released == list(parts.buffers)
    with pytest.raises(MdpStateError, match="retired"):
        owner.require()
    assert id(backward_completion) not in gate5._ACTIVE_COMPLETIONS
    assert id(carrier) not in gate4._ACTIVE_CARRIERS
    assert id(receipt) not in gradient_api._ACTIVE_RECEIPTS
    assert id(completion) not in replay_api._ACTIVE_COMPLETIONS
    handoff.abort()
    assert parts.runtime._captured_num_tokens is None
    assert parts.runtime._token_capture_count == 0
    assert parts.runtime._token_consumed is False
    assert api._ACTIVE_COMMIT_HANDOFFS == {}
    with pytest.raises(MdpStateError, match="retired"):
        handoff.require()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("authority", "exact Gate6 authority"),
        ("ready", "exact encoder-only commit authority"),
        ("owner", "exact and active"),
    ),
)
def test_claim_for_commit_rejects_substitution_without_consuming(monkeypatch, mutation, message):
    parts = _parts(monkeypatch)
    ready = api.run_repeated_d4_encoder_gradient_finalize(
        parts.owner, parts.authority, parts.owner.completion
    )
    owner = ready.owner
    owner_entry = api._ACTIVE_OWNERS[id(owner)]
    argument_authority = parts.authority
    argument_ready = ready
    foreign = _Operations(object())
    if mutation == "authority":
        argument_authority = object()
    elif mutation == "ready":
        argument_ready = object.__new__(api._D4EncoderOnlyCommitReady)
    else:
        api._RETIRED_OWNERS.pop(id(owner), None)
        api._ACTIVE_OWNERS[id(owner)] = (weakref.ref(foreign),)
    with pytest.raises(MdpStateError, match=message):
        api._claim_for_commit(owner, argument_authority, argument_ready)
    if mutation == "owner":
        api._ACTIVE_OWNERS[id(owner)] = owner_entry
    assert owner.require_commit_ready(ready) is ready
    owner.abort()


def test_claim_for_commit_cleanup_failure_retires_and_rolls_back_token(monkeypatch):
    parts = _parts(monkeypatch)
    ready = api.run_repeated_d4_encoder_gradient_finalize(
        parts.owner, parts.authority, parts.owner.completion
    )
    owner = ready.owner
    parts.operations.fail_release = True
    with pytest.raises(RuntimeError, match="release failed") as caught:
        api._claim_for_commit(owner, parts.authority, ready)
    assert len(caught.value.__notes__) == len(parts.buffers) - 1
    assert parts.operations.released == list(parts.buffers)
    assert parts.runtime._captured_num_tokens is None
    assert parts.runtime._token_capture_count == 0
    assert parts.runtime._token_consumed is False
    assert api._ACTIVE_COMMIT_HANDOFFS == {}
    assert api._ACTIVE_READY == {}
    assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    with pytest.raises(MdpStateError, match="retired"):
        owner.require()


def test_commit_handoff_live_field_mutation_cannot_block_trusted_abort(monkeypatch):
    parts = _parts(monkeypatch)
    ready = api.run_repeated_d4_encoder_gradient_finalize(
        parts.owner, parts.authority, parts.owner.completion
    )
    handoff = api._claim_for_commit(ready.owner, parts.authority, ready)
    handoff.runtime = object()
    handoff.ready = object()
    handoff.abort()
    assert parts.runtime._captured_num_tokens is None
    assert api._ACTIVE_COMMIT_HANDOFFS == {}
    assert api._ACTIVE_READY == {}


@pytest.mark.parametrize(
    "mutation",
    (
        "delete_handoff",
        "substitute_handoff",
        "delete_runtime",
        "substitute_runtime",
        "delete_ready",
        "substitute_ready",
        "live_fields",
        "reenter",
    ),
)
def test_claim_cleanup_callback_mutation_is_drained_and_same_runtime_retries(monkeypatch, mutation):
    parts = _parts(monkeypatch)
    ready = api.run_repeated_d4_encoder_gradient_finalize(
        parts.owner, parts.authority, parts.owner.completion
    )
    owner = ready.owner
    foreign = _Operations(object())

    def mutate():
        handoff = ready.owner
        if mutation == "delete_handoff":
            api._ACTIVE_COMMIT_HANDOFFS.pop(id(handoff))
        elif mutation == "substitute_handoff":
            api._ACTIVE_COMMIT_HANDOFFS[id(handoff)] = (weakref.ref(foreign), foreign)
        elif mutation == "delete_runtime":
            replay_api._forward._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))
        elif mutation == "substitute_runtime":
            replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = (
                parts.runtime,
                weakref.ref(foreign),
            )
        elif mutation == "delete_ready":
            api._ACTIVE_READY.pop(id(ready))
        elif mutation == "substitute_ready":
            api._ACTIVE_READY[id(ready)] = (weakref.ref(foreign), foreign)
        elif mutation == "live_fields":
            handoff.runtime = handoff.authority = handoff.ready = object()
        else:
            handoff.abort()

    parts.operations.on_release = mutate
    with pytest.raises(MdpStateError, match="Gate7 cleanup handoff"):
        api._claim_for_commit(owner, parts.authority, ready)
    assert parts.operations.released == list(parts.buffers)
    assert parts.runtime._captured_num_tokens is None
    assert parts.runtime._token_capture_count == 0
    assert parts.runtime._token_consumed is False
    assert id(owner) in api._RETIRED_OWNERS
    assert api._ACTIVE_COMMIT_HANDOFFS == {}
    if mutation not in ("substitute_runtime",):
        assert id(parts.runtime) not in replay_api._forward._ACTIVE_RUNTIME_OWNERS
    else:
        assert replay_api._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is foreign
        replay_api._forward._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))
    if mutation == "substitute_ready":
        assert api._ACTIVE_READY[id(ready)][0]() is foreign
        api._ACTIVE_READY.pop(id(ready))
    else:
        assert id(ready) not in api._ACTIVE_READY

    fresh = _parts(monkeypatch, runtime=parts.runtime)
    fresh_ready = api.run_repeated_d4_encoder_gradient_finalize(
        fresh.owner, fresh.authority, fresh.owner.completion
    )
    fresh_handoff = api._claim_for_commit(fresh_ready.owner, fresh.authority, fresh_ready)
    fresh_handoff.abort()
    assert fresh.runtime._captured_num_tokens is None
    assert ready.owner is None and ready.runtime is None and ready.token is None
    assert parts.runtime._captured_num_tokens is None
