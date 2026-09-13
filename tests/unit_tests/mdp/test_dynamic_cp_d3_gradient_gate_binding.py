# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 nonce-bound gradient-gate binding contracts."""

import gc
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
    return import_module("megatron.core.mdp.dynamic_cp_d3_gradient_gate_binding")


def test_api_is_private_and_factory_returns_bound_coordinator_callbacks():
    api = _api()
    assert api.__all__ == ()
    assert tuple(inspect.signature(api._make_d3_gradient_gate_binding).parameters) == (
        "workspace_owner",
        "cp_partition_mode",
        "group",
        "group_ranks",
        "global_rank",
        "device",
        "timeout_seconds",
        "fallback_status_gate",
        "nonce_status_gather_factory",
        "nonce_byte_generator",
        "all_gather_status",
        "group_ranks_getter",
        "all_to_all_single",
    )


@dataclass(frozen=True)
class _Authority:
    participant_ranks: tuple[int, ...] = (7,)
    global_manifest: object = None
    plan: object = object()
    gradient_ledger: object = object()
    embedding_ledger: object = object()
    producer_rank_by_item: object = object()
    output_rows_by_item: object = object()
    bridge_width: int = 2
    bridge_dtype: object = torch.float32


class _Manifest:
    digest = b"m" * 16


class _Ready:
    pass


class _Exchange:
    def __init__(self, send, receive, views):
        self.phase = _api().BridgePhase.GRADIENT
        self.send_buffer = send
        self.receive_buffer = receive
        self.received_tensors = views
        self.route_authority_digest = b"r" * 16
        self.global_rank = 7
        self.participant_ranks = (7,)


class _Prepared:
    def __init__(self, ready, exchange):
        self.ready = ready
        self.exchange = exchange


class _Context:
    def __init__(self, gate_id, authority, phase_value, ready):
        self.gate_id = gate_id
        self.authority = authority
        self.phase_value = phase_value
        self.ready = ready


class _Receipt:
    def __init__(self, prepared, received, nonce):
        self.prepared = prepared
        self.received_tensors = received
        self.iteration_nonce = nonce


class _Workspace:
    def __init__(self, authority, *, offset=1):
        self.authority = authority
        self.rank = 7
        self.device = torch.device("cuda")
        self._released = False
        base = torch.empty(12)
        self.gradient_transport_buffers = (torch.empty(4), base)
        self.gradient_views = MappingProxyType({"item": base[offset : offset + 4].view(2, 2)})


class _Owner:
    def __init__(self, workspace):
        self.workspace = workspace

    def require_workspace(self, authority):
        if self.workspace is None or self.workspace.authority is not authority:
            raise MdpStateError("exact active workspace")
        return self.workspace


class _Group:
    def size(self):
        return 1

    def rank(self):
        return 0


@pytest.fixture
def gate_api(monkeypatch):
    api = _api()
    monkeypatch.setattr(api, "_D3WorkspaceBindingOwner", _Owner)
    monkeypatch.setattr(api, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(api, "DecoderReadyIteration", _Ready)
    monkeypatch.setattr(api, "PreparedDecoderGradientExchange", _Prepared)
    monkeypatch.setattr(api, "DecoderGradientReceipt", _Receipt)
    monkeypatch.setattr(api, "_D3GateStatusContext", _Context)
    monkeypatch.setattr(api, "_dynamic_iteration_plan_digest", lambda _authority: b"p" * 16)
    monkeypatch.setattr(
        api, "_validate_retained_decoder_ready_iteration", lambda value, **_kw: value
    )
    monkeypatch.setattr(
        api, "validate_prepared_decoder_gradient_exchange", lambda value, **_kw: value
    )
    monkeypatch.setattr(api, "validate_prepared_dynamic_bridge_exchange", lambda value: value)
    monkeypatch.setattr(
        api, "build_dynamic_bridge_route_authority_digest", lambda *_a, **_k: b"r" * 16
    )
    monkeypatch.setattr(api, "_decoder_gradient_wave_authority_digest", lambda *_a: b"w" * 16)
    monkeypatch.setattr(api, "_dynamic_bridge_gate_authority_digest", lambda *_a: b"g" * 16)
    return api


def _parts():
    authority = _Authority(global_manifest=_Manifest())
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    ready = _Ready()
    exchange = _Exchange(
        workspace.gradient_transport_buffers[0],
        workspace.gradient_transport_buffers[1],
        workspace.gradient_views,
    )
    prepared = _Prepared(ready, exchange)
    return authority, workspace, owner, ready, prepared


def _binding(api, owner, monkeypatch, *, events=None, nonce_values=None, gather=None, a2a=None):
    events = [] if events is None else events
    nonce_values = iter((b"n" * 16,) if nonce_values is None else nonce_values)

    def nonce(**_kwargs):
        events.append("nonce")
        return next(nonce_values)

    monkeypatch.setattr(api, "acquire_d3_iteration_nonce", nonce)

    def status(wire, *, timeout_seconds):
        events.append("status")
        return (wire,)

    def exchange(*_args, **_kwargs):
        events.append("a2a")

    monkeypatch.setattr(
        api,
        "_execute_validated_dynamic_bridge_exchange",
        lambda carrier, **kwargs: ((a2a or exchange)(carrier, **kwargs), carrier.received_tensors)[
            1
        ],
    )

    def receipt(prepared, received, *, iteration_nonce):
        events.append("receipt")
        return _Receipt(prepared, received, iteration_nonce)

    monkeypatch.setattr(api, "_make_decoder_gradient_receipt", receipt)
    return api._make_d3_gradient_gate_binding(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        group=_Group(),
        group_ranks=(7,),
        global_rank=7,
        device=torch.device("cuda"),
        timeout_seconds=1.0,
        fallback_status_gate=lambda *args: events.append(("fallback", args)),
        nonce_status_gather_factory=lambda **_kwargs: None,
        nonce_byte_generator=lambda _width: b"x" * 16,
        all_gather_status=status if gather is None else gather,
        group_ranks_getter=lambda _group: (7,),
        all_to_all_single=lambda *_args, **_kwargs: None,
    )


def test_factory_seal_lifecycle_fallback_and_exact_success_receipt(gate_api, monkeypatch):
    api = gate_api
    authority, workspace, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events)
    kwargs = dict(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        group=_Group(),
        group_ranks=(7,),
        global_rank=7,
        device=torch.device("cuda"),
        timeout_seconds=1.0,
        fallback_status_gate=lambda *_args: None,
        nonce_status_gather_factory=lambda **_kwargs: None,
        nonce_byte_generator=lambda _width: b"x" * 16,
        all_gather_status=lambda *_args, **_kwargs: (),
        group_ranks_getter=lambda _group: (7,),
        all_to_all_single=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(MdpStateError, match="factory"):
        api._D3GradientGateBinding(**kwargs)
    assert binding.state == "idle" and binding.is_idle
    other = _Context(2, authority, object(), None)
    binding.status_gate(other, None)
    assert events == [("fallback", (other, None))]

    context = _Context(3, authority, prepared, ready)
    binding.status_gate(context, None)
    assert binding.state == "armed" and binding.is_armed
    with pytest.raises(TypeError):
        binding.execute_gradient(prepared=prepared)
    receipt = binding.execute_gradient(prepared)
    assert binding.state == "idle" and binding.is_idle
    assert receipt.prepared is prepared
    assert receipt.received_tensors is workspace.gradient_views
    assert receipt.iteration_nonce == b"n" * 16
    assert events[1:] == ["nonce", "status", "a2a", "receipt"]


def test_factory_rejects_unrepresentable_timeout_without_callback(gate_api):
    api = gate_api
    _, _, owner, _, _ = _parts()
    calls = []
    with pytest.raises(MdpConfigurationError, match="finite number"):
        api._make_d3_gradient_gate_binding(
            workspace_owner=owner,
            cp_partition_mode="contiguous",
            group=_Group(),
            group_ranks=(7,),
            global_rank=7,
            device=torch.device("cuda"),
            timeout_seconds=10**1000,
            fallback_status_gate=lambda *_args: calls.append("fallback"),
            nonce_status_gather_factory=lambda **_kwargs: calls.append("nonce factory"),
            nonce_byte_generator=lambda _width: b"x" * 16,
            all_gather_status=lambda *_args, **_kwargs: calls.append("status"),
            group_ranks_getter=lambda _group: (7,),
            all_to_all_single=lambda *_args, **_kwargs: calls.append("a2a"),
        )
    assert calls == []


def test_second_status_and_replay_reject_before_nonce_or_a2a(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events, nonce_values=(b"1" * 16,))
    context = _Context(3, authority, prepared, ready)
    binding.status_gate(context, None)
    with pytest.raises(MdpStateError, match="already armed"):
        binding.status_gate(context, None)
    assert events == ["nonce", "status"]
    binding.execute_gradient(prepared)
    with pytest.raises(MdpTaskFatalError, match="replayed"):
        binding.status_gate(context, None)
    assert binding.is_poisoned
    with pytest.raises(MdpTaskFatalError, match="poisoned"):
        binding.execute_gradient(prepared)
    with pytest.raises(MdpTaskFatalError, match="poisoned"):
        binding.status_gate(_Context(2, authority, None, None), None)
    assert events == ["nonce", "status", "a2a", "receipt"]


def test_gate3_status_carries_exact_manifest_and_route_wave_digest(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    wires = []

    def gather(wire, *, timeout_seconds):
        wires.append((wire, timeout_seconds))
        return (wire,)

    binding = _binding(api, owner, monkeypatch, gather=gather)
    binding.status_gate(_Context(3, authority, prepared, ready), None)
    status = api._PrecollectiveStatus.from_wire_tuple(wires[0][0])
    assert status.global_manifest_digest == b"m" * 16
    assert status.plan_digest == b"g" * 16
    assert status.gate_id == 3 and status.error_code == 0
    assert wires[0][1] == 1.0


def test_idle_execute_of_retired_prepared_poisons_without_collective(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events)
    binding.status_gate(_Context(3, authority, prepared, ready), None)
    binding.execute_gradient(prepared)
    with pytest.raises(MdpTaskFatalError, match="replayed"):
        binding.execute_gradient(prepared)
    assert binding.is_poisoned
    assert events == ["nonce", "status", "a2a", "receipt"]


@pytest.mark.parametrize("scheduled", (False, True))
def test_local_or_scheduled_error_gets_nonce_and_status_but_no_a2a(
    gate_api, monkeypatch, scheduled
):
    api = gate_api
    authority, _, owner, ready, _ = _parts()
    failure = RuntimeError("schedule" if scheduled else "prepare")
    events = []
    binding = _binding(api, owner, monkeypatch, events=events)
    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        binding.status_gate(_Context(3, authority, None, ready), failure)
    assert caught.value.__cause__ is failure
    assert binding.is_idle
    assert events == ["nonce", "status"]


def test_recoverable_rejection_then_fresh_nonce_success(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events, nonce_values=(b"1" * 16, b"2" * 16))
    failure = RuntimeError("prepare")
    with pytest.raises(MdpPlanError) as caught:
        binding.status_gate(_Context(3, authority, None, ready), failure)
    assert caught.value.__cause__ is failure
    binding.status_gate(_Context(3, authority, prepared, ready), None)
    receipt = binding.execute_gradient(prepared)
    assert receipt.iteration_nonce == b"2" * 16
    assert events == ["nonce", "status", "nonce", "status", "a2a", "receipt"]


def test_nonce_plan_rejection_returns_idle_then_fresh_nonce_succeeds(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events)
    outcomes = iter((MdpPlanError("nonce plan"), b"2" * 16))

    def acquire(**_kwargs):
        events.append("nonce")
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(api, "acquire_d3_iteration_nonce", acquire)
    context = _Context(3, authority, prepared, ready)
    with pytest.raises(MdpPlanError, match="nonce plan"):
        binding.status_gate(context, None)
    assert binding.is_idle
    binding.status_gate(context, None)
    receipt = binding.execute_gradient(prepared)
    assert receipt.iteration_nonce == b"2" * 16


def test_received_view_order_and_offset_alias_reject_before_a2a(gate_api, monkeypatch):
    api = gate_api
    authority, workspace, owner, ready, prepared = _parts()
    other = workspace.gradient_transport_buffers[1]
    prepared.exchange.received_tensors = MappingProxyType({"item": other[2:6].view(2, 2)})
    events = []
    binding = _binding(api, owner, monkeypatch, events=events)
    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        binding.status_gate(_Context(3, authority, prepared, ready), None)
    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert binding.is_idle and events == ["nonce", "status"]


def test_wrong_received_view_key_rejects_before_a2a(gate_api, monkeypatch):
    api = gate_api
    authority, workspace, owner, ready, prepared = _parts()
    prepared.exchange.received_tensors = MappingProxyType(
        {"wrong": next(iter(workspace.gradient_views.values()))}
    )
    events = []
    binding = _binding(api, owner, monkeypatch, events=events)
    with pytest.raises(MdpPlanError) as caught:
        binding.status_gate(_Context(3, authority, prepared, ready), None)
    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert events == ["nonce", "status"]


def test_multi_key_order_and_external_buffer_identity_reject_before_a2a(gate_api, monkeypatch):
    api = gate_api
    authority, workspace, owner, ready, prepared = _parts()
    base = workspace.gradient_transport_buffers[1]
    workspace.gradient_views = MappingProxyType(
        {"first": base[0:4].view(2, 2), "second": base[4:8].view(2, 2)}
    )
    prepared.exchange.received_tensors = MappingProxyType(
        {"second": base[4:8].view(2, 2), "first": base[0:4].view(2, 2)}
    )
    events = []
    binding = _binding(api, owner, monkeypatch, events=events)
    with pytest.raises(MdpPlanError) as caught:
        binding.status_gate(_Context(3, authority, prepared, ready), None)
    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert events == ["nonce", "status"]

    external_send = torch.empty_like(workspace.gradient_transport_buffers[0])
    prepared.exchange.received_tensors = workspace.gradient_views
    prepared.exchange.send_buffer = external_send
    binding = _binding(api, owner, monkeypatch, events=events)
    with pytest.raises(MdpPlanError) as caught:
        binding.status_gate(_Context(3, authority, prepared, ready), None)
    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert "a2a" not in events


def test_local_error_still_runs_full_ready_and_native_group_validation(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, _ = _parts()
    events = []

    def validate_ready(value, **_kwargs):
        events.append(("ready", value))
        raise MdpBridgeError("different ready authority")

    monkeypatch.setattr(api, "_validate_retained_decoder_ready_iteration", validate_ready)
    binding = _binding(api, owner, monkeypatch, events=events)
    primary = RuntimeError("scheduled abort")
    with pytest.raises(MdpPlanError) as caught:
        binding.status_gate(_Context(3, authority, None, ready), primary)
    assert caught.value.__cause__ is primary
    assert events[:2] == ["nonce", ("ready", ready)]


def test_different_ready_and_native_group_mismatch_converge_without_a2a(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events)
    with pytest.raises(MdpPlanError) as caught:
        binding.status_gate(_Context(3, authority, prepared, _Ready()), None)
    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert events == ["nonce", "status"]

    binding = api._make_d3_gradient_gate_binding(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        group=_Group(),
        group_ranks=(7,),
        global_rank=7,
        device=torch.device("cuda"),
        timeout_seconds=1.0,
        fallback_status_gate=lambda *_args: None,
        nonce_status_gather_factory=lambda **_kwargs: None,
        nonce_byte_generator=lambda _width: b"x" * 16,
        all_gather_status=lambda wire, **_kwargs: (wire,),
        group_ranks_getter=lambda _group: (9,),
        all_to_all_single=lambda *_args, **_kwargs: events.append("a2a"),
    )
    monkeypatch.setattr(api, "acquire_d3_iteration_nonce", lambda **_kwargs: b"3" * 16)
    with pytest.raises(MdpPlanError) as caught:
        binding.status_gate(_Context(3, authority, prepared, ready), None)
    assert caught.value.__cause__ is not None
    assert "a2a" not in events


def test_post_status_substitution_workspace_release_and_mutation_are_task_fatal(
    gate_api, monkeypatch
):
    api = gate_api
    for mutation in ("substitute", "release", "views"):
        authority, workspace, owner, ready, prepared = _parts()
        events = []
        binding = _binding(api, owner, monkeypatch, events=events)
        binding.status_gate(_Context(3, authority, prepared, ready), None)
        argument = prepared
        if mutation == "substitute":
            argument = _Prepared(ready, prepared.exchange)
        elif mutation == "release":
            workspace._released = True
        else:
            prepared.exchange.received_tensors = MappingProxyType({})
        with pytest.raises(MdpTaskFatalError):
            binding.execute_gradient(argument)
        assert binding.is_poisoned
        with pytest.raises(MdpTaskFatalError, match="poisoned"):
            binding.status_gate(_Context(2, authority, None, None), None)
        assert "a2a" not in events


def test_native_group_mutation_between_status_and_execute_is_task_fatal(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []

    class MutableGroup:
        ranks = (7,)

        def size(self):
            return len(self.ranks)

        def rank(self):
            return 0

    group = MutableGroup()
    monkeypatch.setattr(api, "acquire_d3_iteration_nonce", lambda **_kwargs: b"n" * 16)
    monkeypatch.setattr(
        api,
        "_execute_validated_dynamic_bridge_exchange",
        lambda *_args, **_kwargs: events.append("a2a"),
    )
    binding = api._make_d3_gradient_gate_binding(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        group=group,
        group_ranks=(7,),
        global_rank=7,
        device=torch.device("cuda"),
        timeout_seconds=1.0,
        fallback_status_gate=lambda *_args: None,
        nonce_status_gather_factory=lambda **_kwargs: None,
        nonce_byte_generator=lambda _width: b"x" * 16,
        all_gather_status=lambda wire, **_kwargs: (wire,),
        group_ranks_getter=lambda actual: actual.ranks,
        all_to_all_single=lambda *_args, **_kwargs: events.append("a2a"),
    )
    binding.status_gate(_Context(3, authority, prepared, ready), None)
    group.ranks = (8,)
    with pytest.raises(MdpTaskFatalError, match="task-fatal"):
        binding.execute_gradient(prepared)
    assert binding.is_poisoned and events == []


def test_nonce_status_a2a_and_receipt_failures_poison(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    context = _Context(3, authority, prepared, ready)

    binding = _binding(api, owner, monkeypatch)
    monkeypatch.setattr(
        api,
        "acquire_d3_iteration_nonce",
        lambda **_kwargs: (_ for _ in ()).throw(MdpBridgeError("nonce")),
    )
    with pytest.raises(MdpBridgeError, match="nonce"):
        binding.status_gate(context, None)
    assert binding.is_poisoned

    def status_failure(*_args, **_kwargs):
        raise RuntimeError("status")

    binding = _binding(api, owner, monkeypatch, gather=status_failure)
    with pytest.raises(MdpBridgeError, match="consensus failed"):
        binding.status_gate(context, None)
    assert binding.is_poisoned

    def a2a_failure(*_args, **_kwargs):
        raise RuntimeError("a2a")

    binding = _binding(api, owner, monkeypatch, a2a=a2a_failure)
    binding.status_gate(context, None)
    with pytest.raises(MdpTaskFatalError, match="task-fatal"):
        binding.execute_gradient(prepared)
    assert binding.is_poisoned

    binding = _binding(api, owner, monkeypatch)
    binding.status_gate(context, None)
    monkeypatch.setattr(
        api,
        "_make_decoder_gradient_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("receipt")),
    )
    with pytest.raises(MdpTaskFatalError, match="task-fatal"):
        binding.execute_gradient(prepared)
    assert binding.is_poisoned


def test_anomalous_status_acceptance_of_local_error_does_not_arm(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, _ = _parts()
    binding = _binding(api, owner, monkeypatch)
    monkeypatch.setattr(api, "_run_precollective_consensus", lambda *_args, **_kwargs: None)
    failure = RuntimeError("prepare")
    with pytest.raises(MdpStateError, match="accepted") as caught:
        binding.status_gate(_Context(3, authority, None, ready), failure)
    assert caught.value.__cause__ is failure
    assert binding.is_poisoned

    binding = _binding(api, owner, monkeypatch)
    monkeypatch.setattr(api, "_run_precollective_consensus", lambda *_args, **_kwargs: None)
    fatal = MdpTaskFatalError("task fatal")
    with pytest.raises(MdpStateError, match="accepted") as caught:
        binding.status_gate(_Context(3, authority, None, ready), fatal)
    assert caught.value.__cause__ is fatal
    assert binding.is_poisoned


def test_same_retired_workspace_poison_then_new_binding_and_workspace_succeed(
    gate_api, monkeypatch
):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events, nonce_values=(b"1" * 16,))
    binding.status_gate(_Context(3, authority, prepared, ready), None)
    binding.execute_gradient(prepared)

    fresh_ready_same_workspace = _Ready()
    fresh_prepared_same_workspace = _Prepared(
        fresh_ready_same_workspace,
        _Exchange(
            owner.workspace.gradient_transport_buffers[0],
            owner.workspace.gradient_transport_buffers[1],
            owner.workspace.gradient_views,
        ),
    )
    with pytest.raises(MdpTaskFatalError, match="re-armed"):
        binding.status_gate(
            _Context(3, authority, fresh_prepared_same_workspace, fresh_ready_same_workspace), None
        )
    assert binding.is_poisoned
    assert events == ["nonce", "status", "a2a", "receipt"]

    new_authority = _Authority(global_manifest=_Manifest())
    new_workspace = _Workspace(new_authority)
    owner.workspace = new_workspace
    new_ready = _Ready()
    new_prepared = _Prepared(
        new_ready,
        _Exchange(
            new_workspace.gradient_transport_buffers[0],
            new_workspace.gradient_transport_buffers[1],
            new_workspace.gradient_views,
        ),
    )
    binding = _binding(api, owner, monkeypatch, events=events, nonce_values=(b"2" * 16,))
    binding.status_gate(_Context(3, new_authority, new_prepared, new_ready), None)
    binding.execute_gradient(new_prepared)
    assert events == ["nonce", "status", "a2a", "receipt", "nonce", "status", "a2a", "receipt"]


def test_exact_old_pair_after_workspace_rebind_poisoned_before_nonce_or_a2a(gate_api, monkeypatch):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events, nonce_values=(b"1" * 16,))
    context = _Context(3, authority, prepared, ready)
    binding.status_gate(context, None)
    binding.execute_gradient(prepared)
    new_authority = _Authority(global_manifest=_Manifest())
    owner.workspace = _Workspace(new_authority)
    with pytest.raises(MdpTaskFatalError, match="replayed"):
        binding.status_gate(context, None)
    assert binding.is_poisoned
    assert events == ["nonce", "status", "a2a", "receipt"]


@pytest.mark.parametrize("retired_part", ("ready", "prepared"))
def test_each_retired_carrier_identity_after_rebind_poisons_before_nonce_or_a2a(
    gate_api, monkeypatch, retired_part
):
    api = gate_api
    authority, _, owner, ready, prepared = _parts()
    events = []
    binding = _binding(api, owner, monkeypatch, events=events, nonce_values=(b"1" * 16,))
    binding.status_gate(_Context(3, authority, prepared, ready), None)
    binding.execute_gradient(prepared)

    new_authority = _Authority(global_manifest=_Manifest())
    new_workspace = _Workspace(new_authority)
    owner.workspace = new_workspace
    fresh_ready = _Ready()
    fresh_prepared = _Prepared(
        ready if retired_part == "ready" else fresh_ready,
        _Exchange(
            new_workspace.gradient_transport_buffers[0],
            new_workspace.gradient_transport_buffers[1],
            new_workspace.gradient_views,
        ),
    )
    replay_ready = ready if retired_part == "ready" else fresh_ready
    replay_prepared = fresh_prepared if retired_part == "ready" else prepared
    with pytest.raises(MdpTaskFatalError, match="replayed"):
        binding.status_gate(_Context(3, new_authority, replay_prepared, replay_ready), None)
    assert binding.is_poisoned
    assert events == ["nonce", "status", "a2a", "receipt"]


def test_retirement_tombstone_does_not_retain_workspace_ready_or_cuda_graph(gate_api, monkeypatch):
    api = gate_api
    authority, workspace, owner, ready, prepared = _parts()
    binding = _binding(api, owner, monkeypatch)
    context = _Context(3, authority, prepared, ready)
    binding.status_gate(context, None)
    receipt = binding.execute_gradient(prepared)
    references = tuple(weakref.ref(value) for value in (workspace, ready, prepared))
    owner.workspace = None
    del context, receipt, workspace, ready, prepared
    gc.collect()
    assert all(reference() is None for reference in references)
    assert binding.is_idle


def test_collected_retired_authority_does_not_false_match_none_context(gate_api, monkeypatch):
    api = gate_api
    _, workspace, owner, ready, prepared = _parts()
    binding = _binding(api, owner, monkeypatch)
    retired_authority = _Authority(global_manifest=_Manifest())
    authority_reference = weakref.ref(retired_authority)
    binding._tombstone = (
        authority_reference,
        weakref.ref(workspace),
        weakref.ref(ready),
        weakref.ref(prepared),
    )
    del retired_authority
    gc.collect()
    assert authority_reference() is None
    with pytest.raises(MdpPlanError):
        binding.status_gate(_Context(3, None, None, _Ready()), None)
    assert binding.is_idle


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
_WORLD4_RANKS = (0, 1, 2, 3)

if _DISTRIBUTED:
    from tests.unit_tests.mdp import test_dynamic_cp_runtime as runtime_test
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def gradient_gate_groups():
        Utils.initialize_model_parallel()
        participant = torch.distributed.new_group(ranks=list(_WORLD4_RANKS), backend="nccl")
        cp_rank_1 = torch.distributed.new_group(ranks=[1], backend="nccl")
        cp_rank_2 = torch.distributed.new_group(ranks=[2], backend="nccl")
        cp_ranks_1_2 = torch.distributed.new_group(ranks=[1, 2], backend="nccl")
        yield participant, {1: cp_rank_1, 2: cp_rank_2, (1, 2): cp_ranks_1_2}
        rank = torch.distributed.get_rank()
        if rank in (1, 2):
            torch.distributed.destroy_process_group(cp_ranks_1_2)
        if rank == 2:
            torch.distributed.destroy_process_group(cp_rank_2)
        if rank == 1:
            torch.distributed.destroy_process_group(cp_rank_1)
        torch.distributed.barrier(group=participant)
        torch.distributed.destroy_process_group(participant)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_common_nonce_one_status_one_reverse_a2a_and_error_convergence(
    monkeypatch, gradient_gate_groups
):
    api = _api()
    runtime = import_module("megatron.core.mdp.dynamic_cp_runtime")
    coordinator = import_module("megatron.core.mdp.dynamic_cp_d3_coordinator")
    transport = import_module("megatron.core.mdp.dynamic_cp_transport")
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    participant, _ = gradient_gate_groups
    state = runtime_test._state(device=device)
    ready = runtime_test._run_world4(state, rank, gradient_gate_groups)
    runtime_test._set_leaf_grads(ready)
    buffers = runtime_test._gradient_buffers(state, rank)
    prepared = runtime_test._prepare_gradient(state, ready, rank, buffers=buffers)
    authority = runtime._DynamicIterationAuthority(
        global_manifest=state.manifest,
        plan=state.plan,
        source_rank_by_lane=state.payload_authority["source_rank_by_lane"],
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        payload_ledger=state.payload_ledger,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        participant_ranks=_WORLD4_RANKS,
        bridge_width=runtime_test._WIDTH,
        bridge_dtype=torch.float32,
    )
    validate_ready = api._validate_retained_decoder_ready_iteration
    validate_prepared = api.validate_prepared_decoder_gradient_exchange
    plan_digests = []

    def tracked_validate_ready(value, **kwargs):
        plan_digests.append(("ready", kwargs["plan_digest"]))
        return validate_ready(value, **kwargs)

    def tracked_validate_prepared(value, **kwargs):
        plan_digests.append(("prepared", kwargs["plan_digest"]))
        return validate_prepared(value, **kwargs)

    monkeypatch.setattr(api, "_validate_retained_decoder_ready_iteration", tracked_validate_ready)
    monkeypatch.setattr(api, "validate_prepared_decoder_gradient_exchange", tracked_validate_prepared)

    class WorldWorkspace:
        def __init__(self):
            self.authority = authority
            self.rank = rank
            self.device = device
            self._released = False
            self.gradient_transport_buffers = buffers
            self.gradient_views = prepared.exchange.received_tensors

    class WorldOwner:
        def __init__(self):
            self.workspace = WorldWorkspace()

        def require_workspace(self, actual):
            if actual is not authority:
                raise MdpStateError("exact active workspace")
            return self.workspace

    monkeypatch.setattr(api, "_D3WorkspaceBindingOwner", WorldOwner)
    owner = WorldOwner()
    events = []

    def nonce_factory(**kwargs):
        gather = transport.make_precollective_status_gather(**kwargs)

        def tracked(wire, *, timeout_seconds):
            events.append("nonce")
            return gather(wire, timeout_seconds=timeout_seconds)

        return tracked

    status_gather = transport.make_precollective_status_gather(
        group=participant, group_ranks=_WORLD4_RANKS, global_rank=rank, device=device
    )

    def tracked_status(wire, *, timeout_seconds):
        events.append("status")
        return status_gather(wire, timeout_seconds=timeout_seconds)

    def tracked_a2a(*args, **kwargs):
        events.append("a2a")
        return torch.distributed.all_to_all_single(*args, **kwargs)

    binding = api._make_d3_gradient_gate_binding(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        group=participant,
        group_ranks=_WORLD4_RANKS,
        global_rank=rank,
        device=device,
        timeout_seconds=30.0,
        fallback_status_gate=lambda *_args: None,
        nonce_status_gather_factory=nonce_factory,
        all_gather_status=tracked_status,
        group_ranks_getter=torch.distributed.get_process_group_ranks,
        all_to_all_single=tracked_a2a,
    )

    local_error = RuntimeError("rank-2 preparation") if rank == 2 else None
    failed_value = None if local_error is not None else prepared
    failed_context = coordinator._make_d3_gate_status_context(
        gate_id=3, authority=authority, phase_value=failed_value, ready=ready
    )
    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        binding.status_gate(failed_context, local_error)
    if rank == 2:
        assert caught.value.__cause__ is local_error
    assert binding.is_idle and events == ["nonce", "status"]

    events.clear()
    context = coordinator._make_d3_gate_status_context(
        gate_id=3, authority=authority, phase_value=prepared, ready=ready
    )
    binding.status_gate(context, None)
    receipt = binding.execute_gradient(prepared)
    assert plan_digests == [
        ("ready", authority.plan.digest),
        ("ready", authority.plan.digest),
        ("prepared", authority.plan.digest),
    ]
    assert events == ["nonce", "status", "a2a"]
    assert receipt.received_tensors is prepared.exchange.received_tensors
    assert tuple(receipt.received_tensors) == tuple(owner.workspace.gradient_views)
    nonces = [None] * 4
    torch.distributed.all_gather_object(nonces, receipt.iteration_nonce, group=participant)
    assert all(nonce == receipt.iteration_nonce for nonce in nonces)
