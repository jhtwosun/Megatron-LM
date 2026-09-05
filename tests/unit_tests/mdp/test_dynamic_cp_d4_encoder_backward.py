# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for repeated-D4 Gate-5 encoder backward."""

import os
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp_execution import (
    _COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL,
    _CompletedPrecollectiveConsensus,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError

_MANIFEST = b"m" * 16
_GATE = b"g" * 16
_RUNNER_NONCE = b"d" * 16
_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d4_encoder_backward")


def _completed():
    return _CompletedPrecollectiveConsensus(
        error=None, _seal=_COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL
    )


class _Binding:
    def __init__(self, runner, events):
        self.runner = runner
        self.events = events
        self.global_rank = 0

    def begin_attempt(self, **kwargs):
        self.events.append(("begin", kwargs))
        return self.runner


class _Gate:
    pass


class _Claim:
    pass


class _Ready:
    pass


def _values(api, monkeypatch):
    events = []

    def world_gate(**kwargs):
        events.append(("world", kwargs))
        if kwargs["local_error"] is not None:
            raise MdpPlanError("WORLD rejected") from kwargs["local_error"]

    def domain_status(**kwargs):
        events.append(("domain", kwargs))
        return _completed()

    order = import_module("megatron.core.mdp.dynamic_cp_d4_collective_order")
    runner = order._make_repeated_d4_collective_runner(
        attempt_nonce=_RUNNER_NONCE,
        world_pre_gate=world_gate,
        domain_status_collector=domain_status,
    )
    binding = _Binding(runner, events)
    authority = SimpleNamespace(global_manifest=SimpleNamespace(digest=_MANIFEST))
    prepared = SimpleNamespace(authority=authority, receipt=object())
    gate = _Gate()
    claim = _Claim()
    ready = _Ready()
    ready.prepared = prepared
    monkeypatch.setattr(api, "_RepeatedD4GroupBinding", _Binding)
    monkeypatch.setattr(api, "_PreparedD3EncoderCompletion", SimpleNamespace)
    monkeypatch.setattr(api, "_D3EncoderCompletionGateBinding", _Gate)
    monkeypatch.setattr(api, "_D3EncoderBackwardClaim", _Claim)
    monkeypatch.setattr(api, "_D3EncoderFinalizeReady", _Ready)
    monkeypatch.setattr(
        api,
        "_snapshot_local_authority",
        lambda actual_binding, actual_authority: events.append(
            ("snapshot", actual_binding, actual_authority)
        ),
    )
    monkeypatch.setattr(api, "_candidate_digest", lambda *_args: _MANIFEST)
    monkeypatch.setattr(
        api,
        "_candidate_completion_gate_digest",
        lambda actual_authority, receipt: events.append(("candidate", actual_authority, receipt))
        or _GATE,
    )
    monkeypatch.setattr(
        api,
        "_prepare_d3_encoder_backward_claim",
        lambda actual_gate, actual_prepared: events.append(("claim", actual_gate, actual_prepared))
        or claim,
    )
    monkeypatch.setattr(
        api,
        "_execute_d3_encoder_backward_claim",
        lambda actual_claim: events.append(("execute", actual_claim)) or ready,
    )
    return SimpleNamespace(
        authority=authority,
        binding=binding,
        claim=claim,
        events=events,
        gate=gate,
        prepared=prepared,
        ready=ready,
    )


def _run(api, values):
    return api.run_repeated_d4_encoder_backward(
        values.binding,
        values.authority,
        prepared=values.prepared,
        completion_gate_binding=values.gate,
    )


def test_gate5_begins_attempt_before_candidate_and_guards_exact_inputs(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)

    assert _run(api, values) is values.ready

    assert [event[0] for event in values.events] == [
        "begin",
        "candidate",
        "snapshot",
        "claim",
        "world",
        "domain",
        "world",
        "execute",
    ]
    assert values.events[4][1]["gate_id"] == 5


def test_gate5_signature_returns_exact_finalize_ready_type():
    api = _api()

    assert (
        api.run_repeated_d4_encoder_backward.__annotations__["return"]
        is api._D3EncoderFinalizeReady
    )


def test_gate5_malformed_candidate_still_enters_first_world(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)

    class _MalformedPrepared:
        @property
        def receipt(self):
            raise RuntimeError("malformed preparation")

    values.prepared = _MalformedPrepared()
    monkeypatch.setattr(api, "_candidate_completion_gate_digest", api._candidate_gate5_digest)

    with pytest.raises(MdpPlanError, match="WORLD rejected"):
        _run(api, values)

    assert [event[0] for event in values.events] == ["begin", "world"]
    assert isinstance(values.events[-1][1]["local_error"], MdpPlanError)


def test_gate5_exact_validation_failure_is_guarded_by_first_world(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    values.gate = object()

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        _run(api, values)

    assert isinstance(caught.value.__cause__, MdpConfigurationError)
    assert [event[0] for event in values.events] == ["begin", "candidate", "snapshot", "world"]


def test_gate5_prepares_and_retains_exact_backward_claim_without_executing(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    claim = _Claim()
    monkeypatch.setattr(
        api,
        "_prepare_d3_encoder_backward_claim",
        lambda gate, prepared: values.events.append(("claim", gate, prepared)) or claim,
    )

    assert _run(api, values) is values.ready

    assert [event[0] for event in values.events] == [
        "begin",
        "candidate",
        "snapshot",
        "claim",
        "world",
        "domain",
        "world",
        "execute",
    ]
    assert values.events[3] == ("claim", values.gate, values.prepared)
    assert values.events[-1] == ("execute", claim)


@pytest.mark.parametrize(
    "failure_stage",
    ["world-preparation", "domain", "world-outcome", "world-preparation-abort-fails"],
)
def test_gate5_rejection_aborts_exact_retained_claim_once(monkeypatch, failure_stage):
    api = _api()
    primary = RuntimeError(failure_stage)
    events = []
    world_calls = 0

    def world_gate(**kwargs):
        nonlocal world_calls
        world_calls += 1
        events.append(("world", world_calls))
        if failure_stage.startswith("world-preparation") and world_calls == 1:
            raise primary
        if failure_stage == "world-outcome" and world_calls == 2:
            raise primary

    def domain_status(**_kwargs):
        events.append(("domain",))
        if failure_stage == "domain":
            raise primary
        return _completed()

    order = import_module("megatron.core.mdp.dynamic_cp_d4_collective_order")
    runner = order._make_repeated_d4_collective_runner(
        attempt_nonce=_RUNNER_NONCE,
        world_pre_gate=world_gate,
        domain_status_collector=domain_status,
    )
    binding = _Binding(runner, events)
    authority = SimpleNamespace(global_manifest=SimpleNamespace(digest=_MANIFEST))
    prepared = SimpleNamespace(authority=authority, receipt=object())
    gate = _Gate()
    claim = _Claim()
    monkeypatch.setattr(api, "_RepeatedD4GroupBinding", _Binding)
    monkeypatch.setattr(api, "_PreparedD3EncoderCompletion", SimpleNamespace)
    monkeypatch.setattr(api, "_D3EncoderCompletionGateBinding", _Gate)
    monkeypatch.setattr(api, "_D3EncoderBackwardClaim", _Claim)
    monkeypatch.setattr(api, "_D3EncoderFinalizeReady", _Ready)
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda *_args: None)
    monkeypatch.setattr(api, "_candidate_digest", lambda *_args: _MANIFEST)
    monkeypatch.setattr(api, "_candidate_completion_gate_digest", lambda *_args: _GATE)
    monkeypatch.setattr(api, "_prepare_d3_encoder_backward_claim", lambda *_args: claim)
    monkeypatch.setattr(
        api,
        "_execute_d3_encoder_backward_claim",
        lambda _claim: pytest.fail("rejected Gate 5 must not execute encoder backward"),
    )

    def abort(actual_claim, error):
        events.append(("abort", actual_claim, error))
        if failure_stage.endswith("abort-fails"):
            raise RuntimeError("abort failed")

    monkeypatch.setattr(api, "_abort_d3_encoder_backward_claim", abort)

    with pytest.raises(RuntimeError, match=failure_stage) as caught:
        api.run_repeated_d4_encoder_backward(
            binding, authority, prepared=prepared, completion_gate_binding=gate
        )

    assert caught.value is primary
    assert [event for event in events if event[0] == "abort"] == [("abort", claim, primary)]
    if failure_stage.endswith("abort-fails"):
        assert any("abort failed" in note for note in caught.value.__notes__)


def test_gate5_physical_failure_relies_on_claim_seam_cleanup_once(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    primary = MdpPlanError("physical backward failed")
    aborts = []
    monkeypatch.setattr(
        api, "_execute_d3_encoder_backward_claim", lambda claim: (_ for _ in ()).throw(primary)
    )
    monkeypatch.setattr(
        api, "_abort_d3_encoder_backward_claim", lambda claim, error: aborts.append((claim, error))
    )

    with pytest.raises(MdpPlanError, match="physical backward failed") as caught:
        _run(api, values)

    assert caught.value is primary
    assert aborts == []


def test_gate5_prepare_reentry_rejects_before_second_claim_and_aborts_first(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    runner = values.binding.runner
    aborts = []

    class _Runner:
        def run(self, **kwargs):
            prepare = kwargs["prepare"]

            def reenter():
                prepare()
                return prepare()

            kwargs["prepare"] = reenter
            return runner.run(**kwargs)

    values.binding.runner = _Runner()
    monkeypatch.setattr(
        api, "_abort_d3_encoder_backward_claim", lambda claim, error: aborts.append((claim, error))
    )
    monkeypatch.setattr(
        api,
        "_execute_d3_encoder_backward_claim",
        lambda _claim: pytest.fail("prepare reentry must not enter backward"),
    )

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        _run(api, values)

    assert isinstance(caught.value.__cause__, MdpStateError)
    assert [event[0] for event in values.events].count("claim") == 1
    assert len(aborts) == 1 and aborts[0][0] is values.claim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA real ECP follower")
def test_gate5_real_ecp_follower_runs_zero_gradient_backward_after_final_world(monkeypatch):
    from tests.unit_tests.mdp.test_dynamic_cp_d3_encoder_backward import _real_gate4_parts

    api = _api()
    gate, context, prepared, bound, owner, original = _real_gate4_parts(monkeypatch, follower=True)
    native = prepared.native_completion
    handle = native.handle
    for output in handle.chunk_outputs:
        output.retain_grad()
    events = []

    def world_gate(**_kwargs):
        events.append("world")

    def domain_status(**_kwargs):
        events.append("domain")
        return _completed()

    order = import_module("megatron.core.mdp.dynamic_cp_d4_collective_order")
    runner = order._make_repeated_d4_collective_runner(
        attempt_nonce=_RUNNER_NONCE,
        world_pre_gate=world_gate,
        domain_status_collector=domain_status,
    )
    d4_binding = _Binding(runner, events)
    monkeypatch.setattr(api, "_RepeatedD4GroupBinding", _Binding)
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda *_args: None)
    backward = EncoderForwardHandle.backward

    def tracked_backward(candidate, gradients):
        events.append("backward")
        return backward(candidate, gradients)

    monkeypatch.setattr(EncoderForwardHandle, "backward", tracked_backward)
    try:
        gate.status_gate(context, None)
        ready = api.run_repeated_d4_encoder_backward(
            d4_binding, prepared.authority, prepared=prepared, completion_gate_binding=gate
        )

        assert events == [("begin", {}), "world", "domain", "world", "backward"]
        assert ready.prepared is prepared and ready.owner is owner
        assert ready.handle is handle
        assert owner._state == "backward-complete" and owner._runtime is not None
        assert handle._backward_done is True and handle._released is False
        assert native.gradient_views and all(
            not gradient.any() for gradient in native.gradient_views
        )
        for output in handle.chunk_outputs:
            torch.testing.assert_close(output.grad, torch.zeros_like(output))
    finally:
        if owner._runtime is not None:
            bound.cleanup()
        original.cleanup()


@pytest.mark.parametrize(
    "substitution", ["payload", "result", "callback-reentry", "runner-return", "no-callback"]
)
def test_gate5_post_world_substitution_is_task_fatal(monkeypatch, substitution):
    api = _api()
    values = _values(api, monkeypatch)
    aborts = []
    executes = []

    class _Runner:
        def __init__(self, runner):
            self.runner = runner

        def run(self, **kwargs):
            callback = kwargs["domain_collective"]
            if substitution == "payload":
                kwargs["domain_collective"] = lambda _claim: callback(object())
            elif substitution == "callback-reentry":

                def reenter(claim):
                    callback(claim)
                    return callback(claim)

                kwargs["domain_collective"] = reenter
            elif substitution == "no-callback":
                kwargs["domain_collective"] = lambda claim: claim
            result = self.runner.run(**kwargs)
            return object() if substitution == "runner-return" else result

    values.binding.runner = _Runner(values.binding.runner)
    monkeypatch.setattr(
        api,
        "_execute_d3_encoder_backward_claim",
        lambda claim: executes.append(claim)
        or (object() if substitution == "result" else values.ready),
    )
    monkeypatch.setattr(
        api, "_abort_d3_encoder_backward_claim", lambda claim, error: aborts.append((claim, error))
    )

    with pytest.raises(api.MdpTaskFatalError):
        _run(api, values)

    assert [event[0] for event in values.events[:7]] == [
        "begin",
        "candidate",
        "snapshot",
        "claim",
        "world",
        "domain",
        "world",
    ]
    if substitution in ("payload", "no-callback"):
        assert executes == []
        assert len(aborts) == 1 and aborts[0][0] is values.claim
    else:
        assert executes == [values.claim]
        assert aborts == []


if _WORLD8:
    from tests.unit_tests.mdp.test_dynamic_cp_d4_authority_construction import groups


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
@pytest.mark.parametrize("ep", (1, 4))
def test_world8_gate5_executes_real_encoder_backward_after_final_world(groups, ep):
    from tests.unit_tests.mdp.test_dynamic_cp_d4_decoder_composition import _world8_composition
    from tests.unit_tests.mdp.test_dynamic_cp_d4_encoder_completion import (
        _bind_real_world8_producer,
    )

    api = _api()
    backward_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_backward")
    completion_api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    gate_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding")
    context = _world8_composition(groups, ep)
    bound_producer = native_owner = None
    try:
        decoder_ready = context.coordinator.begin_iteration(context.authority)
        for leaf in decoder_ready.embedding_leaves.values():
            leaf.grad = torch.full_like(leaf, dist.get_rank() + 1)
        context.coordinator.mark_decoder_complete(decoder_ready)
        receipt = context.coordinator.end_decoder_phase(decoder_ready)
        bound_producer, native_owner = _bind_real_world8_producer(context)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("external D4 status must not run a second collective")

        gate = gate_api._make_d3_encoder_completion_gate_binding(
            workspace_owner=context.workspace_owner,
            cp_partition_mode="contiguous",
            group=context.binding.domain_group,
            group_ranks=context.domain_ranks,
            global_rank=dist.get_rank(),
            device=context.workspace.device,
            timeout_seconds=30.0,
            fallback_status_gate=forbidden,
            all_gather_status=forbidden,
            group_ranks_getter=dist.get_process_group_ranks,
        )
        prepared = completion_api.run_repeated_d4_encoder_completion(
            context.binding,
            context.authority,
            workspace_owner=context.workspace_owner,
            producer=bound_producer,
            receipt=receipt,
            cp_partition_mode="contiguous",
            completion_gate_binding=gate,
        )
        runtime_handle = native_owner._handle
        if runtime_handle is not None:
            for output in runtime_handle.chunk_outputs:
                output.retain_grad()

        finalize_ready = api.run_repeated_d4_encoder_backward(
            context.binding, context.authority, prepared=prepared, completion_gate_binding=gate
        )

        assert backward_api._validate_d3_encoder_finalize_ready(finalize_ready) is finalize_ready
        assert finalize_ready.prepared is prepared
        assert finalize_ready.owner is native_owner
        assert gate._state == "idle" and not gate.is_armed
        assert native_owner._state == "backward-complete"
        assert native_owner._runtime is not None
        assert all(key.item_id.source_dp_lane == context.lane for key in receipt.received_tensors)
        assert all(key.endpoint_rank in context.domain_ranks for key in receipt.received_tensors)
        assert all(group is context.binding.domain_group for group in context.all_to_all_calls)
        if runtime_handle is None:
            assert finalize_ready.handle is None
        else:
            assert finalize_ready.handle is runtime_handle
            assert runtime_handle._backward_done is True
            assert runtime_handle._released is False
            for output, gradient in zip(
                runtime_handle.chunk_outputs, prepared.native_completion.gradient_views
            ):
                torch.testing.assert_close(output.grad, gradient)
    finally:
        if bound_producer is not None and native_owner._runtime is not None:
            bound_producer.cleanup()
        elif native_owner is not None and native_owner._runtime is not None:
            native_owner.abort()
        context.producer.cleanup()
