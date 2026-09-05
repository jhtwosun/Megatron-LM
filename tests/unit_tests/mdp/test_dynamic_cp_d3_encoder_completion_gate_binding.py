# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 physical gate-4 encoder-completion authorization contracts."""

import gc
import hashlib
import inspect
import os
import weakref
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType

import pytest
import torch

from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding")


@dataclass(frozen=True)
class _Manifest:
    digest: bytes = b"m" * 16


@dataclass(frozen=True)
class _Authority:
    participant_ranks: tuple[int, ...] = (7,)
    global_manifest: object = _Manifest()
    plan: object = object()
    gradient_ledger: object = object()
    embedding_ledger: object = object()
    producer_rank_by_item: object = object()
    output_rows_by_item: object = object()
    bridge_width: int = 4
    bridge_dtype: object = torch.float32
    route_byte: int = 17


@dataclass(frozen=True)
class _Ready:
    authority_digest: bytes = b"q" * 16


@dataclass(frozen=True)
class _Exchange:
    route_authority_digest: bytes = b"r" * 16


@dataclass(frozen=True)
class _ReceiptPrepared:
    ready: object
    exchange: _Exchange = _Exchange()


@dataclass(frozen=True)
class _Receipt:
    prepared: _ReceiptPrepared
    iteration_nonce: bytes


@dataclass(frozen=True)
class _PreAuthority:
    owner: object


class _NativeOwner:
    producer: object


@dataclass(frozen=True)
class _Producer:
    marker: object
    owner: object
    pre_authority: _PreAuthority


@dataclass(frozen=True)
class _NativeCompletion:
    owner: object
    payload: object = None


@dataclass(frozen=True)
class _Prepared:
    authority: _Authority
    producer: _Producer
    workspace: object
    receipt: _Receipt
    native_completion: object
    cp_partition_mode: str = "contiguous"


class _Context:
    def __init__(self, gate_id, authority, phase_value, ready):
        self.gate_id = gate_id
        self.authority = authority
        self.phase_value = phase_value
        self.ready = ready


class _Workspace:
    def __init__(self, authority, rank=7, device=torch.device("cuda")):
        self.authority = authority
        self.rank = rank
        self.device = device
        self._released = False


class _Owner:
    def __init__(self, workspace):
        self.workspace = workspace

    def require_workspace(self, authority):
        if (
            self.workspace is None
            or self.workspace.authority is not authority
            or self.workspace._released
        ):
            raise MdpStateError("exact active workspace")
        return self.workspace


class _Group:
    def __init__(self, ranks=(7,), global_rank=7):
        self.ranks = ranks
        self.global_rank = global_rank

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(self.global_rank)


def _digest(*parts):
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        hasher.update(part if isinstance(part, bytes) else bytes((part,)))
    return hasher.digest()


@pytest.fixture
def gate_api(monkeypatch):
    api = _api()
    monkeypatch.setattr(api, "_D3WorkspaceBindingOwner", _Owner)
    monkeypatch.setattr(api, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(api, "DecoderReadyIteration", _Ready)
    monkeypatch.setattr(api, "DecoderGradientReceipt", _Receipt)
    monkeypatch.setattr(api, "_PreparedD3EncoderCompletion", _Prepared)
    monkeypatch.setattr(api, "_D3GateStatusContext", _Context)
    monkeypatch.setattr(api, "_dynamic_iteration_plan_digest", lambda _authority: b"p" * 16)
    monkeypatch.setattr(
        api, "_validate_retained_decoder_ready_iteration", lambda value, **_kwargs: value
    )

    def validate(prepared, **kwargs):
        if type(prepared) is not _Prepared:
            raise MdpConfigurationError("exact prepared completion")
        if prepared.authority is not kwargs["authority"]:
            raise MdpBridgeError("authority")
        if prepared.producer.marker == "invalid":
            raise MdpBridgeError("carrier seal")
        if prepared.cp_partition_mode != kwargs["cp_partition_mode"]:
            raise MdpConfigurationError("exact CP mode")
        if prepared.workspace is not kwargs["workspace_owner"].require_workspace(
            kwargs["authority"]
        ):
            raise MdpStateError("workspace")
        return prepared

    monkeypatch.setattr(api, "_validate_prepared_d3_encoder_completion", validate)

    def validate_native(completion, *, owner):
        if type(completion) is not _NativeCompletion or completion.owner is not owner:
            raise MdpStateError("exact native completion owner")
        return completion

    monkeypatch.setattr(
        api, "_validate_prepared_native_encoder_completion", validate_native, raising=False
    )
    monkeypatch.setattr(
        api,
        "build_dynamic_bridge_route_authority_digest",
        lambda *_args, **kwargs: (
            _digest(kwargs["global_manifest"].digest, _args[0].route_byte)
            if hasattr(_args[0], "route_byte")
            else b"r" * 16
        ),
    )
    monkeypatch.setattr(
        api,
        "_decoder_gradient_wave_authority_digest",
        lambda ready, nonce: _digest(ready.authority_digest, nonce),
    )
    monkeypatch.setattr(
        api,
        "_dynamic_bridge_gate_authority_digest",
        lambda phase, route, wave: _digest(phase.value.encode(), route, wave),
    )
    return api


def _parts(*, ranks=(7,), rank=7, nonce=b"n" * 16, ready_digest=b"q" * 16, mode="contiguous"):
    authority = _Authority(participant_ranks=ranks)
    workspace = _Workspace(authority, rank=rank)
    owner = _Owner(workspace)
    ready = _Ready(ready_digest)
    receipt = _Receipt(_ReceiptPrepared(ready), nonce)
    native_owner = _NativeOwner()
    pre_authority = _PreAuthority(native_owner)
    native_owner.producer = pre_authority
    producer = _Producer(object(), native_owner, pre_authority)
    native = _NativeCompletion(native_owner)
    prepared = _Prepared(authority, producer, workspace, receipt, native, mode)
    return authority, workspace, owner, ready, receipt, prepared, native


def _binding(
    api, owner, *, ranks=(7,), rank=7, mode="contiguous", events=None, gather=None, fallback=None
):
    events = [] if events is None else events

    def status(wire, *, timeout_seconds):
        events.append(("status", api._PrecollectiveStatus.from_wire_tuple(wire)))
        return (wire,)

    return api._make_d3_encoder_completion_gate_binding(
        workspace_owner=owner,
        cp_partition_mode=mode,
        group=_Group(ranks, rank),
        group_ranks=ranks,
        global_rank=rank,
        device=torch.device("cuda"),
        timeout_seconds=1.0,
        fallback_status_gate=(
            (lambda *args: events.append(("fallback", args))) if fallback is None else fallback
        ),
        all_gather_status=status if gather is None else gather,
        group_ranks_getter=lambda group: group.ranks,
    )


def _context(parts, *, gate=4, value=True):
    authority, _, _, ready, _, prepared, _ = parts
    return _Context(gate, authority, prepared if value else None, ready)


def test_private_factory_signatures_state_and_fallback(gate_api):
    api = gate_api
    parts = _parts()
    _, _, owner, _, _, _, _ = parts
    events = []
    binding = _binding(api, owner, events=events)
    assert api.__all__ == ()
    assert tuple(inspect.signature(api._make_d3_encoder_completion_gate_binding).parameters) == (
        "workspace_owner",
        "cp_partition_mode",
        "group",
        "group_ranks",
        "global_rank",
        "device",
        "timeout_seconds",
        "fallback_status_gate",
        "all_gather_status",
        "group_ranks_getter",
    )
    assert tuple(inspect.signature(binding.status_gate).parameters) == ("context", "local_error")
    assert tuple(inspect.signature(binding.claim_for_backward).parameters) == ("prepared",)
    assert binding.state == "idle" and binding.is_idle and not binding.is_armed
    other = _Context(5, parts[0], object(), parts[3])
    binding.status_gate(other, None)
    assert events == [("fallback", (other, None))]
    with pytest.raises(TypeError):
        binding.status_gate(context=_context(parts), local_error=None)
    with pytest.raises(TypeError):
        binding.claim_for_backward(prepared=parts[5])

    kwargs = dict(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        group=_Group(),
        group_ranks=(7,),
        global_rank=7,
        device=torch.device("cuda"),
        timeout_seconds=1.0,
        fallback_status_gate=lambda *_args: None,
        all_gather_status=lambda *_args, **_kwargs: (),
        group_ranks_getter=lambda group: group.ranks,
    )
    with pytest.raises(MdpStateError, match="factory"):
        api._D3EncoderCompletionGateBinding(**kwargs)
    with pytest.raises(MdpConfigurationError):
        api._make_d3_encoder_completion_gate_binding(**{**kwargs, "timeout_seconds": 10**1000})


def test_external_attempt_prepares_without_collective_and_blocks_claim_or_reentry(gate_api):
    api = gate_api
    parts = _parts()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external preparation must not gather")

    binding = _binding(api, parts[2], gather=forbidden)
    attempt = binding.prepare_status_attempt(_context(parts), None)

    assert type(attempt) is api._D3EncoderCompletionGateAttempt
    assert attempt.status.error_code == 0 and attempt.error is None
    assert binding.state == "claimed" and not binding.is_armed
    with pytest.raises(MdpStateError, match="requires one armed"):
        binding.claim_for_backward(parts[5])
    with pytest.raises(MdpStateError, match="already claimed"):
        binding.prepare_status_attempt(_context(parts), None)
    with pytest.raises(MdpStateError, match="minted by its binding"):
        api._D3EncoderCompletionGateAttempt(None, None, None, None, None, None, None)
    with pytest.raises(MdpStateError, match="minted by its binding"):
        api._D3EncoderCompletionGateAttempt(
            attempt._binding,
            attempt._status,
            attempt._error,
            attempt._authority,
            attempt._ready,
            attempt._armed,
            attempt._factory_seal,
        )


def test_external_attempt_accepts_exact_token_and_rejects_foreign_or_double_resolution(
    gate_api, monkeypatch
):
    api = gate_api
    first = _parts()
    second = _parts(nonce=b"s" * 16)
    first_binding = _binding(api, first[2])
    second_binding = _binding(api, second[2])
    first_attempt = first_binding.prepare_status_attempt(_context(first), None)
    second_attempt = second_binding.prepare_status_attempt(_context(second), None)

    with pytest.raises(MdpStateError, match="active attempt"):
        first_binding.accept_status_attempt(second_attempt)
    assert first_binding.state == "claimed"

    monkeypatch.setattr(
        api._D3EncoderCompletionGateBinding,
        "_validate_gate4",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("accept must not repeat validation")
        ),
    )
    first_binding.accept_status_attempt(first_attempt)
    assert first_binding.is_armed
    assert first_binding.claim_for_backward(first[5]) is first[6]
    with pytest.raises(MdpStateError, match="active attempt"):
        first_binding.accept_status_attempt(first_attempt)

    second_binding.abort_status_attempt(second_attempt, MdpPlanError("peer rejection"))
    assert second_binding.is_idle
    with pytest.raises(MdpStateError, match="active attempt"):
        second_binding.abort_status_attempt(second_attempt, MdpPlanError("again"))


def test_external_attempt_mutation_is_rejected_without_installing_authority(gate_api):
    api = gate_api
    parts = _parts()
    binding = _binding(api, parts[2])
    attempt = binding.prepare_status_attempt(_context(parts), None)
    object.__setattr__(
        attempt,
        "_status",
        api._PrecollectiveStatus(
            global_rank=7,
            global_manifest_digest=b"x" * 16,
            plan_digest=attempt.status.plan_digest,
            error_code=0,
            gate_id=4,
        ),
    )

    with pytest.raises(MdpTaskFatalError, match="sealed fields"):
        binding.accept_status_attempt(attempt)
    assert binding.is_poisoned and not binding.is_armed

    parts = _parts()
    binding = _binding(api, parts[2])
    attempt = binding.prepare_status_attempt(_context(parts), None)
    object.__setattr__(attempt, "_authority", object())
    with pytest.raises(MdpTaskFatalError, match="sealed fields"):
        binding.abort_status_attempt(attempt, MdpPlanError("must not retire forged identity"))
    assert binding.is_poisoned and binding._tombstone is None


def test_external_attempt_error_is_raised_by_guarded_caller_then_aborted(gate_api):
    api = gate_api
    parts = _parts()
    binding = _binding(api, parts[2])
    local_error = RuntimeError("local gate4 preparation")
    attempt = binding.prepare_status_attempt(_context(parts, value=False), local_error)

    assert attempt.error is local_error and attempt.status.error_code == 1
    assert binding.state == "claimed"
    with pytest.raises(RuntimeError, match="local gate4 preparation"):
        raise attempt.error
    binding.abort_status_attempt(attempt, MdpPlanError("WORLD rejected local error"))
    assert binding.is_idle


def test_external_attempt_mint_failure_poison_does_not_strand_claimed(gate_api, monkeypatch):
    api = gate_api
    parts = _parts()
    binding = _binding(api, parts[2])
    monkeypatch.setattr(
        api._D3EncoderCompletionGateBinding,
        "_fingerprint_attempt",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("fingerprint")),
    )

    with pytest.raises(RuntimeError, match="fingerprint"):
        binding.prepare_status_attempt(_context(parts), None)
    assert binding.is_poisoned and binding._attempt is None


@pytest.mark.parametrize("failure", (MdpBridgeError("transport"), RuntimeError("unknown")))
def test_external_attempt_abort_classifies_logical_and_fatal_failures(gate_api, failure):
    api = gate_api
    parts = _parts()
    binding = _binding(api, parts[2])
    attempt = binding.prepare_status_attempt(_context(parts), None)
    binding.abort_status_attempt(attempt, failure)
    assert binding.is_poisoned

    retired = _parts()
    binding = _binding(api, retired[2])
    attempt = binding.prepare_status_attempt(_context(retired), None)
    binding.abort_status_attempt(attempt, MdpPlanError("coordinated rejection"))
    assert binding.is_idle
    with pytest.raises(MdpTaskFatalError, match="cannot be replayed"):
        binding.prepare_status_attempt(_context(retired), None)
    assert binding.is_poisoned


def test_status_gate_resolves_through_external_attempt_seam(gate_api, monkeypatch):
    api = gate_api
    parts = _parts()
    binding = _binding(api, parts[2])
    events = []
    prepare = api._D3EncoderCompletionGateBinding.prepare_status_attempt
    accept = api._D3EncoderCompletionGateBinding.accept_status_attempt

    def tracked_prepare(self, context, local_error, /):
        events.append("prepare")
        return prepare(self, context, local_error)

    def tracked_accept(self, attempt, /):
        events.append("accept")
        return accept(self, attempt)

    monkeypatch.setattr(
        api._D3EncoderCompletionGateBinding, "prepare_status_attempt", tracked_prepare
    )
    monkeypatch.setattr(
        api._D3EncoderCompletionGateBinding, "accept_status_attempt", tracked_accept
    )
    monkeypatch.setattr(
        api, "_run_precollective_consensus", lambda *_args, **_kwargs: events.append("gather")
    )

    binding.status_gate(_context(parts), None)

    assert events == ["prepare", "gather", "accept"]
    assert binding.claim_for_backward(parts[5]) is parts[6]


@pytest.mark.parametrize("mode", ("contiguous", "zigzag"))
def test_valid_digest_arms_and_exact_claim_only_unwraps(gate_api, mode):
    api = gate_api
    parts = _parts(mode=mode, ready_digest=(b"q" if mode == "contiguous" else b"z") * 16)
    _, _, owner, ready, receipt, prepared, native = parts
    events = []
    binding = _binding(api, owner, mode=mode, events=events)

    binding.status_gate(_context(parts), None)

    assert binding.state == "armed" and binding.is_armed
    status = events[0][1]
    route = api.build_dynamic_bridge_route_authority_digest(
        parts[0].gradient_ledger,
        parts[0].embedding_ledger,
        global_manifest=parts[0].global_manifest,
    )
    expected = api._dynamic_bridge_gate_authority_digest(
        api.BridgePhase.GRADIENT,
        route,
        api._decoder_gradient_wave_authority_digest(ready, receipt.iteration_nonce),
    )
    assert status.plan_digest == expected
    assert status.global_manifest_digest == parts[0].global_manifest.digest
    assert status.error_code == 0 and status.gate_id == 4
    assert binding.claim_for_backward(prepared) is native
    assert binding.is_idle and not binding.is_armed
    assert events == [("status", status)]


def test_concrete_native_validation_precedes_status_and_claim_executes_nothing(
    gate_api, monkeypatch
):
    api = gate_api
    parts = _parts()
    native, native_owner = parts[6], parts[5].producer.owner
    calls = []
    validate_native = api._validate_prepared_native_encoder_completion

    def tracked(completion, *, owner):
        calls.append((completion, owner))
        return validate_native(completion, owner=owner)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("gate 4 must not execute encoder completion")

    monkeypatch.setattr(api, "_validate_prepared_native_encoder_completion", tracked)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(
        import_module("megatron.core.mdp.activation").EncoderForwardHandle, "backward", forbidden
    )
    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"), "finalize_encoder_grads", forbidden
    )
    binding = _binding(api, parts[2])

    binding.status_gate(_context(parts), None)
    assert calls == [(native, native_owner)]
    assert binding.claim_for_backward(parts[5]) is native
    assert calls == [(native, native_owner), (native, native_owner)]


@pytest.mark.parametrize("fault", ("validation", "wrong-result"))
def test_invalid_concrete_native_submits_zero_status_and_does_not_arm(gate_api, monkeypatch, fault):
    api = gate_api
    parts = _parts()
    concrete_calls = []
    if fault == "validation":
        validate_native = api._validate_prepared_native_encoder_completion
        object.__setattr__(parts[6], "owner", object())

        def tracked(completion, *, owner):
            concrete_calls.append((completion, owner))
            return validate_native(completion, owner=owner)

        monkeypatch.setattr(api, "_validate_prepared_native_encoder_completion", tracked)
    else:
        monkeypatch.setattr(
            api,
            "_validate_prepared_native_encoder_completion",
            lambda completion, *, owner: _NativeCompletion(owner),
        )
    events = []
    binding = _binding(api, parts[2], events=events)

    with pytest.raises(MdpPlanError):
        binding.status_gate(_context(parts), None)

    assert events[0][1].plan_digest == bytes(16)
    assert events[0][1].error_code == 1
    assert binding.is_idle and not binding.is_armed
    if fault == "validation":
        assert concrete_calls == [(parts[6], parts[5].producer.owner)]


@pytest.mark.parametrize("mismatch", ("owner", "pre-authority"))
def test_pre_status_owner_chain_mismatch_submits_zero_and_does_not_arm(gate_api, mismatch):
    api = gate_api
    parts = _parts()
    if mismatch == "owner":
        object.__setattr__(parts[5].producer, "owner", object())
    else:
        object.__setattr__(
            parts[5].producer, "pre_authority", _PreAuthority(parts[5].producer.owner)
        )
    events = []
    binding = _binding(api, parts[2], events=events)

    with pytest.raises(MdpPlanError):
        binding.status_gate(_context(parts), None)

    assert events[0][1].plan_digest == bytes(16)
    assert events[0][1].error_code == 1
    assert binding.is_idle and not binding.is_armed


@pytest.mark.parametrize("substitution", ("native", "owner", "pre-authority"))
def test_post_status_native_owner_or_pre_authority_substitution_is_task_fatal(
    gate_api, substitution
):
    api = gate_api
    parts = _parts()
    binding = _binding(api, parts[2])
    binding.status_gate(_context(parts), None)
    if substitution == "native":
        object.__setattr__(
            parts[5], "native_completion", _NativeCompletion(parts[5].producer.owner)
        )
    elif substitution == "owner":
        object.__setattr__(parts[5].producer, "owner", object())
    else:
        object.__setattr__(
            parts[5].producer, "pre_authority", _PreAuthority(parts[5].producer.owner)
        )

    with pytest.raises(MdpTaskFatalError, match="exact armed native"):
        binding.claim_for_backward(parts[5])
    assert binding.is_poisoned


def test_digest_uses_only_common_authority_ready_route_and_nonce(gate_api):
    api = gate_api

    def captured(parts, mode="contiguous"):
        wires = []
        binding = _binding(
            api,
            parts[2],
            mode=mode,
            gather=lambda wire, **_kwargs: (wires.append(wire), (wire,))[1],
        )
        binding.status_gate(_context(parts), None)
        return api._PrecollectiveStatus.from_wire_tuple(wires[0]).plan_digest

    first = _parts()
    same = _parts()
    assert captured(first) == captured(same)
    assert first[5].native_completion is not same[5].native_completion
    assert captured(_parts(nonce=b"x" * 16)) != captured(first)
    assert captured(_parts(ready_digest=b"y" * 16)) != captured(first)
    changed_route = _parts()
    object.__setattr__(changed_route[0], "gradient_ledger", type("Route", (), {"route_byte": 99})())
    route_digest = _digest(changed_route[0].global_manifest.digest, 99)
    object.__setattr__(changed_route[4].prepared, "exchange", _Exchange(route_digest))
    assert captured(changed_route) != captured(first)

    zigzag = _parts(mode="zigzag", ready_digest=b"z" * 16)
    assert captured(zigzag, mode="zigzag") != captured(first)


def test_cp_mode_or_rebuilt_route_mismatch_submits_zero_and_never_arms(gate_api):
    api = gate_api
    for fault in ("cp", "route"):
        parts = _parts()
        if fault == "route":
            object.__setattr__(
                parts[4].prepared, "exchange", _Exchange(route_authority_digest=b"x" * 16)
            )
        events = []
        binding = _binding(
            api, parts[2], mode="zigzag" if fault == "cp" else "contiguous", events=events
        )
        with pytest.raises(MdpPlanError):
            binding.status_gate(_context(parts), None)
        assert binding.is_idle and not binding.is_armed
        assert events[0][1].plan_digest == bytes(16)
        assert events[0][1].error_code == 1
        with pytest.raises(MdpStateError, match="requires one armed"):
            binding.claim_for_backward(parts[5])


def test_local_error_uses_zero_digest_converges_and_tombstones_iteration(gate_api):
    api = gate_api
    parts = _parts()
    events = []
    binding = _binding(api, parts[2], events=events)
    error = RuntimeError("local preparation")

    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        binding.status_gate(_context(parts, value=False), error)

    assert caught.value.__cause__ is error
    assert events[0][1].plan_digest == bytes(16)
    assert events[0][1].error_code == 1
    assert binding.is_idle
    with pytest.raises(MdpTaskFatalError, match="cannot be replayed"):
        binding.status_gate(_context(parts, value=False), error)
    assert binding.is_poisoned and len(events) == 1


@pytest.mark.parametrize("fault", ("absent", "nested-ready", "carrier", "rank", "device", "group"))
def test_local_validation_faults_submit_zero_before_any_claim(gate_api, fault):
    api = gate_api
    parts = list(_parts())
    authority, workspace, owner, ready, receipt, prepared, _ = parts
    if fault == "absent":
        value = None
    elif fault == "nested-ready":
        value = _Prepared(
            authority,
            prepared.producer,
            workspace,
            _Receipt(_ReceiptPrepared(_Ready()), receipt.iteration_nonce),
            prepared.native_completion,
            prepared.cp_partition_mode,
        )
    elif fault == "carrier":
        value = _Prepared(
            authority,
            _Producer("invalid", prepared.producer.owner, prepared.producer.pre_authority),
            workspace,
            receipt,
            prepared.native_completion,
            prepared.cp_partition_mode,
        )
    else:
        value = prepared
    if fault == "rank":
        workspace.rank = 8
    elif fault == "device":
        workspace.device = torch.device("cpu")
    group_ranks = (8,) if fault == "group" else (7,)
    events = []
    binding = _binding(api, owner, events=events)
    if fault == "group":
        binding._group.ranks = group_ranks

    with pytest.raises(MdpPlanError):
        binding.status_gate(_Context(4, authority, value, ready), None)

    assert events[0][1].plan_digest == bytes(16)
    assert events[0][1].error_code == 1
    assert binding.is_idle


def test_peer_and_malformed_consensus_rejections_are_idle_and_fresh_iteration_reuses(gate_api):
    api = gate_api
    ranks = (7, 8)
    parts = _parts(ranks=ranks)
    calls = []

    def peer_error(wire, *, timeout_seconds):
        calls.append(wire)
        peer = list(wire)
        peer[0], peer[-2] = 8, 1
        return wire, tuple(peer)

    binding = _binding(api, parts[2], ranks=ranks, gather=peer_error)
    with pytest.raises(MdpPlanError, match="rank 8"):
        binding.status_gate(_context(parts), None)
    assert binding.is_idle and len(calls) == 1

    fresh = _parts(ranks=ranks, nonce=b"f" * 16)
    parts[2].workspace = fresh[1]
    wires = []
    binding._all_gather_status = lambda wire, **_kwargs: (
        wires.append(wire),
        (wire, tuple([8, *wire[1:]])),
    )[1]
    binding.status_gate(_Context(4, fresh[0], fresh[5], fresh[3]), None)
    assert binding.is_armed and len(wires) == 1
    assert binding.claim_for_backward(fresh[5]) is fresh[6]


@pytest.mark.parametrize("fault", ("rank", "digest", "gate", "malformed"))
def test_status_response_faults_converge_without_arming(gate_api, fault):
    api = gate_api
    ranks = (7, 8)
    parts = _parts(ranks=ranks)

    def gather(wire, *, timeout_seconds):
        peer = list(wire)
        peer[0] = 8
        if fault == "rank":
            peer[0] = 9
        elif fault == "digest":
            peer[3] += 1
        elif fault == "gate":
            peer[-1] = 3
        elif fault == "malformed":
            peer.pop()
        return wire, tuple(peer)

    binding = _binding(api, parts[2], ranks=ranks, gather=gather)
    with pytest.raises(MdpPlanError):
        binding.status_gate(_context(parts), None)
    assert binding.is_idle and not binding.is_armed


def test_transport_and_anomalous_acceptance_poison_without_fallback(gate_api, monkeypatch):
    api = gate_api
    parts = _parts()

    def transport(*_args, **_kwargs):
        raise RuntimeError("transport")

    binding = _binding(api, parts[2], gather=transport)
    with pytest.raises(MdpBridgeError):
        binding.status_gate(_context(parts), None)
    assert binding.is_poisoned
    with pytest.raises(MdpTaskFatalError):
        binding.status_gate(_Context(5, parts[0], None, parts[3]), None)

    accepted = _binding(api, parts[2])
    monkeypatch.setattr(api, "_run_precollective_consensus", lambda *_args, **_kwargs: None)
    with pytest.raises(MdpStateError, match="accepted a local error"):
        accepted.status_gate(_context(parts, value=False), RuntimeError("ignored"))
    assert accepted.is_poisoned


def test_armed_reentry_substitution_post_status_mutation_and_double_claim(gate_api):
    api = gate_api
    parts = _parts()
    binding = _binding(api, parts[2])
    binding.status_gate(_context(parts), None)
    with pytest.raises(MdpStateError, match="already armed"):
        binding.status_gate(_context(parts), None)
    with pytest.raises(MdpStateError, match="idle"):
        binding.status_gate(_Context(5, parts[0], None, parts[3]), None)
    substitute = _Prepared(parts[0], parts[5].producer, parts[1], parts[4], parts[6], "contiguous")
    with pytest.raises(MdpTaskFatalError, match="exact armed"):
        binding.claim_for_backward(substitute)
    assert binding.is_poisoned

    parts = _parts()
    binding = _binding(api, parts[2])
    binding.status_gate(_context(parts), None)
    object.__setattr__(
        parts[5],
        "producer",
        _Producer("invalid", parts[5].producer.owner, parts[5].producer.pre_authority),
    )
    with pytest.raises(MdpTaskFatalError, match="post-status"):
        binding.claim_for_backward(parts[5])
    assert binding.is_poisoned

    parts = _parts()
    binding = _binding(api, parts[2])
    binding.status_gate(_context(parts), None)
    assert binding.claim_for_backward(parts[5]) is parts[6]
    with pytest.raises(MdpTaskFatalError, match="cannot be replayed"):
        binding.claim_for_backward(parts[5])
    assert binding.is_poisoned


@pytest.mark.parametrize("fault", ("rank", "device", "participants", "group"))
def test_post_status_workspace_geometry_mutation_is_task_fatal(gate_api, fault):
    api = gate_api
    parts = _parts()
    binding = _binding(api, parts[2])
    binding.status_gate(_context(parts), None)
    if fault == "rank":
        parts[1].rank = 8
    elif fault == "device":
        parts[1].device = torch.device("cuda", 1)
    elif fault == "participants":
        object.__setattr__(parts[0], "participant_ranks", (7, 8))
    else:
        binding._group.ranks = (8,)

    with pytest.raises(MdpTaskFatalError, match="post-status"):
        binding.claim_for_backward(parts[5])
    assert binding.is_poisoned


def _bind_real_native_completion(template_producer, authority):
    from tests.unit_tests.mdp.test_dynamic_cp_d3_producer_owner import _runtime

    producer_owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
    dynamic_api = import_module("megatron.core.mdp.dynamic_cp_runtime")
    native_runtime, native_outputs = _runtime(contributor=False)
    native_runtime.rank_view = template_producer.rank_view
    native_owner = None
    try:
        native_owner = producer_owner_api._capture_d3_producer_owner(
            runtime=native_runtime,
            rank_view=template_producer.rank_view,
            local_manifest=None,
            source_window=None,
            static_plan=None,
            item_outputs=native_outputs,
            sample_location_by_id=MappingProxyType({}),
            forward_only=False,
        )
        bound = dynamic_api._bind_pre_authority_dynamic_producer(
            producer=native_owner.producer,
            authority=authority,
            payload_destination_views=template_producer.payload_destination_views,
            embedding_destination_views=template_producer.embedding_destination_views,
            gradient_destination_views=template_producer.gradient_destination_views,
            summed_gradient_destination_views=template_producer.summed_gradient_destination_views,
        )
    except BaseException:
        if native_owner is not None and native_owner._runtime is not None:
            native_owner.abort()
        raise
    return bound, native_owner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA workspace")
def test_real_pr71_completion_passes_real_preparation_and_gate4_without_execution(monkeypatch):
    api = _api()
    preparation_api = import_module(
        "megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation"
    )
    producer_owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
    from tests.unit_tests.mdp.test_dynamic_cp_d3_encoder_completion_preparation import (
        _real_receipt_inputs,
    )

    authority, owner, workspace, producer, receipt = _real_receipt_inputs()
    validate_receipt = preparation_api._validate_decoder_gradient_receipt
    validate_prepared = preparation_api.validate_prepared_decoder_gradient_exchange
    consume_receipt = preparation_api._consume_decoder_gradient_receipt
    preparation_plan_digests = []
    gate_plan_digests = []

    def tracked_validate_receipt(*args, **kwargs):
        preparation_plan_digests.append(("receipt", kwargs["plan_digest"]))
        return validate_receipt(*args, **kwargs)

    def tracked_validate_prepared(*args, **kwargs):
        preparation_plan_digests.append(("prepared", kwargs["plan_digest"]))
        return validate_prepared(*args, **kwargs)

    def tracked_consume_receipt(*args, **kwargs):
        preparation_plan_digests.append(("consume", kwargs["plan_digest"]))
        return consume_receipt(*args, **kwargs)

    validate_ready = api._validate_retained_decoder_ready_iteration

    def tracked_validate_ready(value, **kwargs):
        gate_plan_digests.append(kwargs["plan_digest"])
        return validate_ready(value, **kwargs)

    monkeypatch.setattr(
        preparation_api, "_validate_decoder_gradient_receipt", tracked_validate_receipt
    )
    monkeypatch.setattr(
        preparation_api, "validate_prepared_decoder_gradient_exchange", tracked_validate_prepared
    )
    monkeypatch.setattr(
        preparation_api, "_consume_decoder_gradient_receipt", tracked_consume_receipt
    )
    monkeypatch.setattr(api, "_validate_retained_decoder_ready_iteration", tracked_validate_ready)
    native_owner = None
    bound_producer = None
    try:
        bound_producer, native_owner = _bind_real_native_completion(producer, authority)
        prepared = preparation_api._make_d3_encoder_completion_preparation_binding(
            workspace_owner=owner, cp_partition_mode="contiguous"
        )(authority, bound_producer, receipt)
        native_completion = prepared.native_completion
        assert type(native_completion) is producer_owner_api._PreparedNativeEncoderCompletion

        def forbidden(*_args, **_kwargs):
            raise AssertionError("gate 4 must not execute native encoder completion")

        monkeypatch.setattr(torch.autograd, "backward", forbidden)
        monkeypatch.setattr(
            import_module("megatron.core.mdp.activation").EncoderForwardHandle,
            "backward",
            forbidden,
        )
        monkeypatch.setattr(
            import_module("megatron.core.mdp.encoder"), "finalize_encoder_grads", forbidden
        )
        ranks = authority.participant_ranks
        binding = api._make_d3_encoder_completion_gate_binding(
            workspace_owner=owner,
            cp_partition_mode="contiguous",
            group=_Group(ranks, workspace.rank),
            group_ranks=ranks,
            global_rank=workspace.rank,
            device=workspace.device,
            timeout_seconds=1.0,
            fallback_status_gate=lambda *_args: None,
            all_gather_status=lambda wire, **_kwargs: tuple(
                tuple((rank, *wire[1:])) for rank in ranks
            ),
            group_ranks_getter=lambda group: group.ranks,
        )
        context = _Context(4, authority, prepared, receipt.prepared.ready)
        monkeypatch.setattr(api, "_D3GateStatusContext", _Context)

        binding.status_gate(context, None)
        assert preparation_plan_digests == [
            ("receipt", authority.plan.digest),
            ("consume", authority.plan.digest),
            ("prepared", authority.plan.digest),
        ]
        assert gate_plan_digests == [authority.plan.digest]
        assert binding.claim_for_backward(prepared) is native_completion
        assert native_completion.handle is None
    finally:
        try:
            if bound_producer is not None and native_owner._runtime is not None:
                bound_producer.cleanup()
            elif native_owner is not None and native_owner._runtime is not None:
                native_owner.abort()
        finally:
            producer.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA workspace")
def test_post_status_resealed_real_route_cannot_change_accepted_digest(monkeypatch):
    api = _api()
    preparation_api = import_module(
        "megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation"
    )
    bridge_api = import_module("megatron.core.mdp.dynamic_cp_bridge_transport")
    runtime_api = import_module("megatron.core.mdp.dynamic_cp_runtime")
    from tests.unit_tests.mdp.test_dynamic_cp_d3_encoder_completion_preparation import (
        _real_receipt_inputs,
    )

    authority, owner, workspace, producer, receipt = _real_receipt_inputs()
    native_owner = None
    bound_producer = None
    try:
        bound_producer, native_owner = _bind_real_native_completion(producer, authority)
        prepared = preparation_api._make_d3_encoder_completion_preparation_binding(
            workspace_owner=owner, cp_partition_mode="contiguous"
        )(authority, bound_producer, receipt)
        ranks = authority.participant_ranks

        def gather(wire, *, timeout_seconds):
            return tuple(tuple((rank, *wire[1:])) for rank in ranks)

        binding = api._make_d3_encoder_completion_gate_binding(
            workspace_owner=owner,
            cp_partition_mode="contiguous",
            group=_Group(ranks, workspace.rank),
            group_ranks=ranks,
            global_rank=workspace.rank,
            device=workspace.device,
            timeout_seconds=1.0,
            fallback_status_gate=lambda *_args: None,
            all_gather_status=gather,
            group_ranks_getter=lambda group: group.ranks,
        )
        context = _Context(4, authority, prepared, receipt.prepared.ready)
        monkeypatch.setattr(api, "_D3GateStatusContext", _Context)
        binding.status_gate(context, None)

        changed_route = b"x" * 16
        exchange = receipt.prepared.exchange
        object.__setattr__(exchange, "route_authority_digest", changed_route)
        object.__setattr__(exchange, "_authority", bridge_api._capture_authority(exchange))
        object.__setattr__(
            receipt.prepared,
            "_authority",
            runtime_api._capture_prepared_decoder_gradient_authority(receipt.prepared),
        )
        object.__setattr__(
            receipt, "_authority", runtime_api._capture_decoder_gradient_receipt_authority(receipt)
        )
        object.__setattr__(
            prepared, "_authority", preparation_api._capture_prepared_authority(prepared)
        )
        monkeypatch.setattr(
            api._D3EncoderCompletionGateBinding,
            "_route_digest",
            lambda self, actual_authority: changed_route,
        )

        with pytest.raises(MdpTaskFatalError, match="post-status") as caught:
            binding.claim_for_backward(prepared)
        assert isinstance(caught.value.__cause__, MdpBridgeError)
        assert "accepted gate authority" in str(caught.value.__cause__)
        assert binding.is_poisoned
    finally:
        try:
            if bound_producer is not None and native_owner._runtime is not None:
                bound_producer.cleanup()
            elif native_owner is not None and native_owner._runtime is not None:
                native_owner.abort()
        finally:
            producer.cleanup()


@pytest.mark.parametrize("shared", ("authority", "ready", "receipt"))
def test_partial_replay_after_coordinated_reject_is_pre_gather_task_fatal(gate_api, shared):
    api = gate_api
    ranks = (7, 8)
    retired = _parts(ranks=ranks)
    calls = []

    def peer_error(wire, *, timeout_seconds):
        calls.append(wire)
        peer = list(wire)
        peer[0], peer[-2] = 8, 1
        return wire, tuple(peer)

    binding = _binding(api, retired[2], ranks=ranks, gather=peer_error)
    with pytest.raises(MdpPlanError):
        binding.status_gate(_context(retired), None)
    assert binding.is_idle and len(calls) == 1

    fresh = list(_parts(ranks=ranks, nonce=b"f" * 16))
    if shared == "authority":
        authority = retired[0]
        workspace = _Workspace(authority)
        ready = fresh[3]
        receipt = _Receipt(_ReceiptPrepared(ready), fresh[4].iteration_nonce)
    else:
        authority = fresh[0]
        workspace = fresh[1]
        ready = retired[3] if shared == "ready" else fresh[3]
        receipt = (
            retired[4]
            if shared == "receipt"
            else _Receipt(_ReceiptPrepared(ready), fresh[4].iteration_nonce)
        )
    native_owner = _NativeOwner()
    pre_authority = _PreAuthority(native_owner)
    native_owner.producer = pre_authority
    producer = _Producer(object(), native_owner, pre_authority)
    replay = _Prepared(authority, producer, workspace, receipt, _NativeCompletion(native_owner))
    retired[2].workspace = workspace
    context = _Context(4, authority, replay, ready)

    with pytest.raises(MdpTaskFatalError, match="cannot be replayed"):
        binding.status_gate(context, None)
    assert binding.is_poisoned and len(calls) == 1


def test_weak_retirement_does_not_retain_completion_or_native_tensor(gate_api):
    api = gate_api
    parts = list(_parts())
    native = torch.empty(1)
    object.__setattr__(
        parts[5], "native_completion", _NativeCompletion(parts[5].producer.owner, native)
    )
    native_ref = weakref.ref(native)
    authority_ref = weakref.ref(parts[0])
    binding = _binding(api, parts[2])
    binding.status_gate(_context(parts), None)
    assert binding.claim_for_backward(parts[5]) is parts[5].native_completion

    fresh = _parts(nonce=b"f" * 16)
    parts[2].workspace = fresh[1]
    parts.clear()
    del native
    gc.collect()
    assert native_ref() is None and authority_ref() is None
    binding.status_gate(_Context(4, fresh[0], fresh[5], fresh[3]), None)
    assert binding.claim_for_backward(fresh[5]) is fresh[6]


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
_WORLD4_RANKS = (0, 1, 2, 3)

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def completion_gate_group():
        Utils.initialize_model_parallel()
        group = torch.distributed.new_group(ranks=list(_WORLD4_RANKS), backend="nccl")
        yield group
        torch.distributed.barrier(group=group)
        torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_common_valid_digest_and_one_rank_failure_converges(
    monkeypatch, gate_api, completion_gate_group
):
    api = gate_api
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    parts = list(_parts(ranks=_WORLD4_RANKS, rank=rank, nonce=b"w" * 16))
    parts[1].device = device
    object.__setattr__(
        parts[5],
        "native_completion",
        _NativeCompletion(parts[5].producer.owner, torch.tensor([rank], device=device)),
    )
    owner = parts[2]
    events = []
    from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather

    gather = make_precollective_status_gather(
        group=completion_gate_group, group_ranks=_WORLD4_RANKS, global_rank=rank, device=device
    )

    def tracked(wire, *, timeout_seconds):
        events.append(api._PrecollectiveStatus.from_wire_tuple(wire))
        return gather(wire, timeout_seconds=timeout_seconds)

    binding = api._make_d3_encoder_completion_gate_binding(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        group=completion_gate_group,
        group_ranks=_WORLD4_RANKS,
        global_rank=rank,
        device=device,
        timeout_seconds=30.0,
        fallback_status_gate=lambda *_args: None,
        all_gather_status=tracked,
        group_ranks_getter=torch.distributed.get_process_group_ranks,
    )
    binding.status_gate(_context(parts), None)
    digests = [None] * 4
    torch.distributed.all_gather_object(
        digests, events[-1].plan_digest, group=completion_gate_group
    )
    assert all(value == events[-1].plan_digest for value in digests)
    assert binding.claim_for_backward(parts[5]) is parts[5].native_completion
    assert len(events) == 1

    fresh = list(_parts(ranks=_WORLD4_RANKS, rank=rank, nonce=b"z" * 16))
    fresh[1].device = device
    owner.workspace = fresh[1]
    error = RuntimeError("rank-2 gate4") if rank == 2 else None
    value = None if error is not None else fresh[5]
    with pytest.raises(MdpPlanError):
        binding.status_gate(_Context(4, fresh[0], value, fresh[3]), error)
    assert binding.is_idle and len(events) == 2
