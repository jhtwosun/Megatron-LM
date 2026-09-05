# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for repeated-D4 Gate-6 encoder-finalization preparation."""

import os
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.mdp.dynamic_cp_execution import (
    _COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL,
    _CompletedPrecollectiveConsensus,
    _PrecollectiveStatus,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpPlanError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.rank_mapping import MdpRankView

_D4_MANIFEST = b"m" * 16
_D3_TOPOLOGY = b"t" * 16
_GATE = b"g" * 16
_RUNNER_NONCE = b"d" * 16
_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d4_encoder_finalize")


def _completed(error=None):
    return _CompletedPrecollectiveConsensus(
        error=error, _seal=_COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL
    )


class _Binding:
    def __init__(self, runner, events):
        self.runner = runner
        self.events = events
        self.global_rank = 0

    def begin_attempt(self, **kwargs):
        self.events.append(("begin", kwargs))
        return self.runner


class _FinalizeBinding:
    def __init__(self, attempt, commit, events, *, accept_error=None, finalize_error=None):
        self.attempt = attempt
        self.commit = commit
        self.events = events
        self.accept_error = accept_error
        self.finalize_error = finalize_error
        self._group_ranks = (0,)

    def prepare_status_attempt(self, context, local_error, /):
        self.events.append(("finalize-prepare", context, local_error))
        return self.attempt

    def abort_status_attempt(self, attempt, error, /):
        self.events.append(("abort", attempt, error))

    def accept_status_attempt(self, attempt, /):
        self.events.append(("accept", attempt))
        if self.accept_error is not None:
            raise self.accept_error

    def finalize(self, ready, /):
        self.events.append(("finalize", ready))
        if self.finalize_error is not None:
            raise self.finalize_error
        return self.commit


class _Attempt:
    def __init__(self, status, error=None):
        self.status = status
        self.error = error


class _Ready:
    pass


class _Commit:
    pass


def _values(
    api,
    monkeypatch,
    *,
    attempt_error=None,
    world_error=None,
    domain_error=None,
    accept_error=None,
    finalize_error=None,
):
    events = []

    def world_gate(**kwargs):
        events.append(("world", kwargs))
        if world_error is not None or kwargs["local_error"] is not None:
            raise world_error or MdpPlanError("WORLD rejected") from kwargs["local_error"]

    def domain_status(**kwargs):
        events.append(("domain", kwargs))
        return _completed(domain_error)

    order = import_module("megatron.core.mdp.dynamic_cp_d4_collective_order")
    runner = order._make_repeated_d4_collective_runner(
        attempt_nonce=_RUNNER_NONCE,
        world_pre_gate=world_gate,
        domain_status_collector=domain_status,
    )
    binding = _Binding(runner, events)
    authority = SimpleNamespace(global_manifest=SimpleNamespace(digest=_D4_MANIFEST))
    ready = _Ready()
    ready.owner = SimpleNamespace(_iteration=7)
    decoder_ready = object()
    ready.prepared = SimpleNamespace(
        authority=authority, receipt=SimpleNamespace(prepared=SimpleNamespace(ready=decoder_ready))
    )
    status = _PrecollectiveStatus(
        global_rank=0,
        global_manifest_digest=_D3_TOPOLOGY if attempt_error is None else bytes(16),
        plan_digest=_GATE if attempt_error is None else bytes(16),
        error_code=int(attempt_error is not None),
        gate_id=5,
    )
    attempt = _Attempt(status, attempt_error)
    commit = _Commit()
    finalize_binding = _FinalizeBinding(
        attempt, commit, events, accept_error=accept_error, finalize_error=finalize_error
    )

    monkeypatch.setattr(api, "_RepeatedD4GroupBinding", _Binding)
    monkeypatch.setattr(api, "_D3EncoderFinalizeBinding", _FinalizeBinding)
    monkeypatch.setattr(api, "_D3EncoderFinalizeAttempt", _Attempt)
    monkeypatch.setattr(api, "_D3EncoderFinalizeReady", _Ready)
    monkeypatch.setattr(api, "_D3IterationCommitReady", _Commit)
    monkeypatch.setattr(api, "_DynamicIterationAuthority", SimpleNamespace)
    monkeypatch.setattr(
        api,
        "_snapshot_local_authority",
        lambda actual_binding, actual_authority: events.append(
            ("snapshot", actual_binding, actual_authority)
        ),
    )
    monkeypatch.setattr(
        api,
        "_candidate_digest",
        lambda actual, field: events.append(("candidate-manifest", actual, field)) or _D4_MANIFEST,
    )
    monkeypatch.setattr(
        api,
        "_candidate_gate6_digest",
        lambda actual_binding, actual_ready: events.append(
            ("candidate-gate", actual_binding, actual_ready)
        )
        or _GATE,
    )
    monkeypatch.setattr(
        api,
        "_digest",
        lambda label, iteration, ranks: _D3_TOPOLOGY if label == b"topology" else _GATE,
    )
    monkeypatch.setattr(
        api, "_make_d3_gate_status_context", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    return SimpleNamespace(
        attempt=attempt,
        authority=authority,
        binding=binding,
        commit=commit,
        events=events,
        finalize_binding=finalize_binding,
        ready=ready,
        decoder_ready=decoder_ready,
    )


def _run(api, values):
    return api.run_repeated_d4_encoder_finalize(
        values.binding,
        values.authority,
        ready=values.ready,
        finalize_binding=values.finalize_binding,
    )


def test_gate6_begins_before_candidate_and_finalizes_only_after_final_world(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)

    assert _run(api, values) is values.commit

    assert [event[0] for event in values.events] == [
        "begin",
        "candidate-manifest",
        "candidate-gate",
        "snapshot",
        "finalize-prepare",
        "world",
        "domain",
        "world",
        "accept",
        "finalize",
    ]
    context = values.events[4][1]
    assert context.gate_id == 5
    assert context.authority is values.authority
    assert context.phase_value is values.ready
    assert context.ready is values.decoder_ready
    worlds = [event[1] for event in values.events if event[0] == "world"]
    assert all(world["global_manifest_digest"] == _D4_MANIFEST for world in worlds)
    assert all(world["gate_id"] == 6 for world in worlds)
    assert all(type(world["plan_digest"]) is bytes for world in worlds)
    domain = next(event[1] for event in values.events if event[0] == "domain")
    assert domain["global_manifest_digest"] == _D4_MANIFEST
    assert domain["gate_id"] == 6 and type(domain["plan_digest"]) is bytes
    assert values.attempt.status.global_manifest_digest == _D3_TOPOLOGY
    assert _D4_MANIFEST != _D3_TOPOLOGY
    assert values.events[-2:] == [("accept", values.attempt), ("finalize", values.ready)]


def test_gate6_signature_returns_exact_iteration_commit_type():
    api = _api()

    assert (
        api.run_repeated_d4_encoder_finalize.__annotations__["return"]
        is api._D3IterationCommitReady
    )


def test_gate6_malformed_candidate_still_enters_first_world(monkeypatch):
    api = _api()
    candidate = api._candidate_gate6_digest
    values = _values(api, monkeypatch)
    values.ready = object()
    monkeypatch.setattr(api, "_candidate_gate6_digest", candidate)

    with pytest.raises(MdpPlanError, match="WORLD rejected"):
        _run(api, values)

    assert [event[0] for event in values.events] == ["begin", "candidate-manifest", "world"]
    assert values.events[-1][1]["local_error"] is not None


def test_gate6_status_lineage_failure_is_guarded_and_aborts_exact_attempt(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    values.attempt.status = _PrecollectiveStatus(1, _D3_TOPOLOGY, _GATE, 0, 5)

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        _run(api, values)

    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert [event[0] for event in values.events] == [
        "begin",
        "candidate-manifest",
        "candidate-gate",
        "snapshot",
        "finalize-prepare",
        "world",
        "abort",
    ]
    assert values.events[-1][1:] == (values.attempt, caught.value)


def test_gate6_local_attempt_error_uses_zero_status_and_converges_before_abort(monkeypatch):
    api = _api()
    local_error = RuntimeError("local finalize preparation failed")
    values = _values(api, monkeypatch, attempt_error=local_error)

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        _run(api, values)

    assert caught.value.__cause__ is local_error
    world = next(event for event in values.events if event[0] == "world")
    assert world[1]["local_error"] is local_error
    assert values.events[-1] == ("abort", values.attempt, caught.value)


def test_gate6_candidate_digest_uses_ready_iteration_and_binding_world_ranks(monkeypatch):
    api = _api()
    binding = SimpleNamespace(_group_ranks=(0, 1, 2, 3))
    ready = SimpleNamespace(owner=SimpleNamespace(_iteration=11))
    calls = []
    monkeypatch.setattr(
        api,
        "_digest",
        lambda label, iteration, ranks: calls.append((label, iteration, ranks)) or _GATE,
    )

    assert api._candidate_gate6_digest(binding, ready) == _GATE
    assert calls == [(b"gate-5", 11, (0, 1, 2, 3))]
    assert api._candidate_gate6_digest(binding, object()) is None


def test_gate6_peer_rejection_aborts_without_accept_or_finalize(monkeypatch):
    api = _api()
    rejection = MdpPlanError("peer rejected Gate 6")
    values = _values(api, monkeypatch, world_error=rejection)

    with pytest.raises(MdpPlanError, match="peer rejected") as caught:
        _run(api, values)

    assert caught.value is rejection
    assert values.events[-1] == ("abort", values.attempt, rejection)
    assert not any(event[0] in ("accept", "finalize") for event in values.events)


def test_gate6_domain_outcome_failure_aborts_exact_attempt(monkeypatch):
    api = _api()
    domain_error = MdpPlanError("domain Gate 6 failed")
    values = _values(api, monkeypatch, domain_error=domain_error)

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        _run(api, values)

    assert caught.value.__cause__ is domain_error
    assert values.events[-1] == ("abort", values.attempt, caught.value)
    assert not any(event[0] in ("accept", "finalize") for event in values.events)


def test_gate6_forwards_byte_generator_only_to_begin_attempt(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    byte_generator = object()

    assert (
        api.run_repeated_d4_encoder_finalize(
            values.binding,
            values.authority,
            ready=values.ready,
            finalize_binding=values.finalize_binding,
            byte_generator=byte_generator,
        )
        is values.commit
    )

    assert values.events[0] == ("begin", {"byte_generator": byte_generator})


@pytest.mark.parametrize("stage", ("accept", "finalize"))
def test_gate6_post_world_failure_is_task_fatal_without_second_abort(monkeypatch, stage):
    api = _api()
    primary = RuntimeError(f"{stage} failed")
    values = _values(
        api,
        monkeypatch,
        accept_error=primary if stage == "accept" else None,
        finalize_error=primary if stage == "finalize" else None,
    )

    with pytest.raises(MdpTaskFatalError, match="after repeated-D4 status") as caught:
        _run(api, values)

    assert caught.value.__cause__ is primary
    assert [event[0] for event in values.events].count("abort") == 0
    assert [event[0] for event in values.events].count("accept") == 1
    assert [event[0] for event in values.events].count("finalize") == int(stage == "finalize")


def test_gate6_rejects_substituted_post_world_result_before_ownership_transfer(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    real_runner = values.binding.runner

    class SubstitutingRunner:
        def run(self, **kwargs):
            prepare = kwargs["prepare"]
            kwargs["prepare"] = lambda: (prepare(), object())[1]
            return real_runner.run(**kwargs)

    values.binding.runner = SubstitutingRunner()

    with pytest.raises(MdpTaskFatalError, match="exact retained status attempt"):
        _run(api, values)

    assert values.events[-1][0] == "abort"
    assert not any(event[0] in ("accept", "finalize") for event in values.events)


def test_gate6_rejects_substituted_runner_return_after_one_finalize(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    real_runner = values.binding.runner

    class SubstitutingRunner:
        def run(self, **kwargs):
            real_runner.run(**kwargs)
            return object()

    values.binding.runner = SubstitutingRunner()

    with pytest.raises(MdpTaskFatalError, match="post-WORLD commit result"):
        _run(api, values)

    assert [event[0] for event in values.events].count("accept") == 1
    assert [event[0] for event in values.events].count("finalize") == 1
    assert [event[0] for event in values.events].count("abort") == 0


def test_gate6_prepare_reentry_rejects_second_attempt_and_aborts_first(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    real_runner = values.binding.runner

    class ReenteringRunner:
        def run(self, **kwargs):
            prepare = kwargs["prepare"]

            def reenter():
                prepare()
                return prepare()

            kwargs["prepare"] = reenter
            return real_runner.run(**kwargs)

    values.binding.runner = ReenteringRunner()

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        _run(api, values)

    assert isinstance(caught.value.__cause__, MdpStateError)
    assert [event[0] for event in values.events].count("finalize-prepare") == 1
    assert [event[0] for event in values.events].count("abort") == 1
    assert [event[0] for event in values.events][-2:] == ["world", "abort"]
    assert not any(event[0] in ("accept", "finalize") for event in values.events)


def test_gate6_callback_reentry_after_final_world_finalizes_once(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    real_runner = values.binding.runner

    class ReenteringRunner:
        def run(self, **kwargs):
            callback = kwargs["domain_collective"]

            def reenter(value):
                callback(value)
                return callback(value)

            kwargs["domain_collective"] = reenter
            return real_runner.run(**kwargs)

    values.binding.runner = ReenteringRunner()

    with pytest.raises(MdpTaskFatalError, match="exactly once"):
        _run(api, values)

    assert [event[0] for event in values.events[:8]][-3:] == ["world", "domain", "world"]
    assert [event[0] for event in values.events].count("accept") == 1
    assert [event[0] for event in values.events].count("finalize") == 1
    assert [event[0] for event in values.events].count("abort") == 0


def test_gate6_prepare_after_finalize_is_task_fatal_without_second_attempt(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    real_runner = values.binding.runner

    class ReenteringRunner:
        def run(self, **kwargs):
            result = real_runner.run(**kwargs)
            kwargs["prepare"]()
            return result

    values.binding.runner = ReenteringRunner()

    with pytest.raises(MdpTaskFatalError, match="prepare after finalization"):
        _run(api, values)

    assert [event[0] for event in values.events].count("finalize-prepare") == 1
    assert [event[0] for event in values.events].count("accept") == 1
    assert [event[0] for event in values.events].count("finalize") == 1
    assert [event[0] for event in values.events].count("abort") == 0


def test_gate6_runner_return_without_callback_aborts_retained_attempt(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    real_runner = values.binding.runner

    class SkippingRunner:
        def run(self, **kwargs):
            kwargs["domain_collective"] = lambda value: value
            return real_runner.run(**kwargs)

    values.binding.runner = SkippingRunner()

    with pytest.raises(MdpTaskFatalError, match="post-WORLD commit result") as caught:
        _run(api, values)

    assert [event[0] for event in values.events].count("abort") == 1
    assert values.events[-1] == ("abort", values.attempt, caught.value)
    assert not any(event[0] in ("accept", "finalize") for event in values.events)


def test_gate6_rejects_noncommit_finalize_result_after_ownership_transfer(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    values.finalize_binding.commit = object()

    with pytest.raises(MdpTaskFatalError, match="iteration-commit capability"):
        _run(api, values)

    assert [event[0] for event in values.events].count("accept") == 1
    assert [event[0] for event in values.events].count("finalize") == 1
    assert [event[0] for event in values.events].count("abort") == 0


@pytest.mark.parametrize(
    ("contributor", "follower"), ((True, False), (False, True), (False, False))
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_gate6_real_d3_finalize_contributor_follower_and_empty(monkeypatch, contributor, follower):
    from tests.unit_tests.mdp.test_dynamic_cp_d3_encoder_finalize import _binding, _ready

    api = _api()
    runtime, _outputs, owner, _native, ready, context, group = _ready(
        monkeypatch, contributor=contributor, follower=follower
    )
    finalize_binding = _binding(monkeypatch, runtime, group)
    events = []

    def world_gate(**kwargs):
        events.append(("world", kwargs["local_error"]))
        if kwargs["local_error"] is not None:
            raise MdpPlanError("WORLD rejected") from kwargs["local_error"]

    def domain_status(**_kwargs):
        events.append(("domain", None))
        return _completed()

    order = import_module("megatron.core.mdp.dynamic_cp_d4_collective_order")
    runner = order._make_repeated_d4_collective_runner(
        attempt_nonce=_RUNNER_NONCE,
        world_pre_gate=world_gate,
        domain_status_collector=domain_status,
    )
    d4_binding = _Binding(runner, events)
    authority = ready.prepared.authority
    monkeypatch.setattr(api, "_RepeatedD4GroupBinding", _Binding)
    monkeypatch.setattr(api, "_DynamicIterationAuthority", type(authority))
    monkeypatch.setattr(api, "_snapshot_local_authority", lambda *_args: None)
    monkeypatch.setattr(api, "_candidate_digest", lambda *_args: _D4_MANIFEST)
    monkeypatch.setattr(
        api,
        "_make_d3_gate_status_context",
        lambda **kwargs: type(context)(
            kwargs["gate_id"], kwargs["authority"], kwargs["phase_value"], kwargs["ready"]
        ),
    )
    try:
        commit = api.run_repeated_d4_encoder_finalize(
            d4_binding, authority, ready=ready, finalize_binding=finalize_binding
        )

        assert type(commit) is api._D3IterationCommitReady
        assert events == [("begin", {}), ("world", None), ("domain", None), ("world", None)]
        assert owner._state == "retired" and owner._runtime is None
        assert runtime._token_consumed is True
        assert finalize_binding.is_idle
    finally:
        if owner._runtime is not None:
            owner.abort()


if _WORLD8:
    from tests.unit_tests.mdp.test_dynamic_cp_d4_authority_construction import groups


def _bind_world8_encoder_with_follower(context):
    from tests.unit_tests.mdp.test_dynamic_cp_d3_producer_owner import _runtime
    from tests.unit_tests.mdp.test_dynamic_cp_d4_encoder_completion import (
        _bind_real_world8_producer,
    )

    rank = dist.get_rank()
    if rank != context.domain_ranks[1]:
        return _bind_real_world8_producer(context)

    owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
    runtime_api = import_module("megatron.core.mdp.dynamic_cp_runtime")
    runtime, outputs = _runtime(contributor=False, follower=True)
    runtime.device = context.workspace.device
    runtime._captured_num_tokens = torch.zeros((), device=runtime.device)
    runtime._chunk_payload_bases = (torch.empty(1, 1, device=runtime.device),)
    rank_view = MdpRankView(
        global_rank=rank,
        outer_dp_rank=context.lane,
        lane_id=None,
        my_worker_id=0,
        endpoint_rank=rank,
        planning_group_ranks=(rank,),
        worker_ids=(0,),
    )
    runtime.rank_view = rank_view
    owner = owner_api._capture_d3_producer_owner(
        runtime=runtime,
        rank_view=rank_view,
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs=outputs,
        sample_location_by_id=MappingProxyType({}),
        forward_only=False,
        encoder_cp_follower=True,
    )
    producer = runtime_api._bind_pre_authority_dynamic_producer(
        producer=owner.producer,
        authority=context.authority,
        payload_destination_views=context.producer.payload_destination_views,
        embedding_destination_views=context.producer.embedding_destination_views,
        gradient_destination_views=context.producer.gradient_destination_views,
        summed_gradient_destination_views=context.producer.summed_gradient_destination_views,
    )
    return producer, owner


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
@pytest.mark.parametrize("ep", (1, 4))
def test_world8_gate6_finalizes_real_repeated_d4_chain_with_ecp_follower(groups, ep):
    from tests.unit_tests.mdp.test_dynamic_cp_d4_decoder_composition import _world8_composition

    api = _api()
    completion_api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    backward_api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_backward")
    gate_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding")
    finalize_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_finalize")
    commit_api = import_module("megatron.core.mdp.dynamic_cp_d3_iteration_commit")
    context = _world8_composition(groups, ep)
    producer = owner = None
    try:
        decoder_ready = context.coordinator.begin_iteration(context.authority)
        for leaf in decoder_ready.embedding_leaves.values():
            leaf.grad = torch.full_like(leaf, dist.get_rank() + 1)
        context.coordinator.mark_decoder_complete(decoder_ready)
        receipt = context.coordinator.end_decoder_phase(decoder_ready)
        producer, owner = _bind_world8_encoder_with_follower(context)
        runtime = owner._runtime
        world, _domain = groups
        world_ranks = tuple(range(8))
        runtime.process_groups = SimpleNamespace(encoder_reduction_group=world, world_group=world)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("external repeated-D4 gate must not nest status collectives")

        completion_gate = gate_api._make_d3_encoder_completion_gate_binding(
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
            producer=producer,
            receipt=receipt,
            cp_partition_mode="contiguous",
            completion_gate_binding=completion_gate,
        )
        handle = owner._handle
        if handle is not None:
            for output in handle.chunk_outputs:
                output.retain_grad()
        ready = backward_api.run_repeated_d4_encoder_backward(
            context.binding,
            context.authority,
            prepared=prepared,
            completion_gate_binding=completion_gate,
        )
        finalize_binding = finalize_api._make_d3_encoder_finalize_binding(
            group=world,
            group_ranks=world_ranks,
            global_rank=dist.get_rank(),
            device=context.workspace.device,
            timeout_seconds=30.0,
            fallback_status_gate=forbidden,
            all_gather_status=forbidden,
            group_ranks_getter=dist.get_process_group_ranks,
        )
        commit = api.run_repeated_d4_encoder_finalize(
            context.binding, context.authority, ready=ready, finalize_binding=finalize_binding
        )

        assert commit_api._validate_ready(commit) is commit
        assert commit.runtime is runtime and commit.iteration == 0
        assert owner._state == "retired" and owner._runtime is None
        assert runtime._token_consumed is True
        assert owner._encoder_cp_follower is (dist.get_rank() == context.domain_ranks[1])
        assert (handle is not None) is (dist.get_rank() in context.domain_ranks[:2])
        if handle is not None:
            assert handle._backward_done is True and handle._released is True
        assert all(group is context.binding.domain_group for group in context.all_to_all_calls)
        assert all(key.endpoint_rank in context.domain_ranks for key in receipt.received_tensors)
    finally:
        if owner is not None and owner._runtime is not None:
            producer.cleanup()
        context.producer.cleanup()
