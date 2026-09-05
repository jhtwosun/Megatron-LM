# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for repeated-D4 gate-4 encoder-completion authorization."""

import os
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.dynamic_cp_execution import (
    _COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL,
    _CompletedPrecollectiveConsensus,
    _PrecollectiveStatus,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpPlanError, MdpTaskFatalError
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment, RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankView

_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8

_MANIFEST = b"m" * 16
_GATE = b"g" * 16
_RECEIPT_NONCE = b"r" * 16
_RUNNER_NONCE = b"d" * 16


def _completed(error=None):
    return _CompletedPrecollectiveConsensus(
        error=error, _seal=_COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL
    )


class _Binding:
    def __init__(self, runner):
        self.runner = runner
        self.global_rank = 0

    def begin_attempt(self, **_kwargs):
        return self.runner


class _Gate:
    def __init__(self, attempt, events, *, accept_error=None, abort_error=None):
        self.attempt = attempt
        self.events = events
        self.accept_error = accept_error
        self.abort_error = abort_error

    def prepare_status_attempt(self, context, local_error, /):
        self.events.append(("gate-prepare", context, local_error))
        return self.attempt

    def accept_status_attempt(self, attempt, /):
        self.events.append(("accept", attempt))
        if self.accept_error is not None:
            raise self.accept_error

    def abort_status_attempt(self, attempt, error, /):
        self.events.append(("abort", attempt, error))
        if self.abort_error is not None:
            raise self.abort_error


def _dependencies(api, monkeypatch, *, world_error=None, domain_error=None, attempt_error=None):
    events = []
    authority = SimpleNamespace(global_manifest=SimpleNamespace(digest=_MANIFEST))
    ready = object()
    receipt = SimpleNamespace(prepared=SimpleNamespace(ready=ready), iteration_nonce=_RECEIPT_NONCE)
    producer = object()
    workspace_owner = object()
    prepared = SimpleNamespace(receipt=receipt)
    attempt = SimpleNamespace(
        status=_PrecollectiveStatus(
            global_rank=0,
            global_manifest_digest=_MANIFEST,
            plan_digest=_GATE if attempt_error is None else bytes(16),
            error_code=int(attempt_error is not None),
            gate_id=4,
        ),
        error=attempt_error,
    )

    def world_gate(**kwargs):
        events.append(("world", kwargs))
        if world_error is not None or kwargs["local_error"] is not None:
            error = world_error or MdpPlanError("WORLD rejected")
            raise error from kwargs["local_error"]

    def domain_status(**kwargs):
        events.append(("domain", kwargs))
        if isinstance(domain_error, MdpBridgeError):
            raise domain_error
        return _completed(domain_error)

    order = import_module("megatron.core.mdp.dynamic_cp_d4_collective_order")
    runner = order._make_repeated_d4_collective_runner(
        attempt_nonce=_RUNNER_NONCE,
        world_pre_gate=world_gate,
        domain_status_collector=domain_status,
    )
    binding = _Binding(runner)
    gate = _Gate(attempt, events)
    monkeypatch.setattr(api, "_RepeatedD4GroupBinding", _Binding)
    monkeypatch.setattr(api, "_D3EncoderCompletionGateBinding", _Gate)
    monkeypatch.setattr(
        api,
        "_snapshot_local_authority",
        lambda actual, value: events.append(("snapshot", actual, value)),
    )

    def candidate(actual_authority, actual_ready, nonce):
        events.append(("candidate", actual_authority, actual_ready, nonce))
        assert actual_authority is authority and actual_ready is ready
        assert nonce == _RECEIPT_NONCE and nonce != _RUNNER_NONCE
        return b"route".ljust(16, b"0"), _GATE

    monkeypatch.setattr(api, "_candidate_gradient_gate_digest", candidate)
    monkeypatch.setattr(api, "_candidate_digest", lambda *_args: _MANIFEST)
    monkeypatch.setattr(
        api, "_make_d3_gate_status_context", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    def make_preparation(**kwargs):
        assert kwargs == {"workspace_owner": workspace_owner, "cp_partition_mode": "contiguous"}

        def prepare(actual_authority, actual_producer, actual_receipt, /):
            events.append(("completion-prepare", actual_authority, actual_producer, actual_receipt))
            assert (actual_authority, actual_producer, actual_receipt) == (
                authority,
                producer,
                receipt,
            )
            return prepared

        return prepare

    monkeypatch.setattr(api, "_make_d3_encoder_completion_preparation_binding", make_preparation)
    return SimpleNamespace(
        attempt=attempt,
        authority=authority,
        binding=binding,
        events=events,
        gate=gate,
        prepared=prepared,
        producer=producer,
        receipt=receipt,
        workspace_owner=workspace_owner,
    )


def _run(api, values):
    return api.run_repeated_d4_encoder_completion(
        values.binding,
        values.authority,
        workspace_owner=values.workspace_owner,
        producer=values.producer,
        receipt=values.receipt,
        cp_partition_mode="contiguous",
        completion_gate_binding=values.gate,
    )


def test_gate4_uses_receipt_nonce_and_arms_only_after_world_domain_world(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    values = _dependencies(api, monkeypatch)

    assert _run(api, values) is values.prepared

    assert [event[0] for event in values.events] == [
        "candidate",
        "snapshot",
        "completion-prepare",
        "gate-prepare",
        "world",
        "domain",
        "world",
        "accept",
    ]
    assert values.events[-1] == ("accept", values.attempt)
    assert not hasattr(api, "_execute_d3_encoder_backward")
    assert not hasattr(api, "_run_precollective_consensus")


def test_gate4_malformed_receipt_candidate_still_enters_world(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    values = _dependencies(api, monkeypatch)

    class _RaisingReceipt:
        @property
        def prepared(self):
            raise RuntimeError("malformed receipt")

        @property
        def iteration_nonce(self):
            raise AssertionError("candidate stops after the first malformed field")

    values.receipt = _RaisingReceipt()

    with pytest.raises(MdpPlanError):
        _run(api, values)

    assert [event[0] for event in values.events] == ["world"]
    assert isinstance(values.events[0][1]["local_error"], MdpPlanError)


def test_gate4_rejects_wrong_d3_status_rank_before_accept(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    values = _dependencies(api, monkeypatch)
    values.attempt.status = _PrecollectiveStatus(
        global_rank=1, global_manifest_digest=_MANIFEST, plan_digest=_GATE, error_code=0, gate_id=4
    )

    with pytest.raises(MdpPlanError):
        _run(api, values)

    assert [event[0] for event in values.events][-2:] == ["world", "abort"]
    assert not any(event[0] == "accept" for event in values.events)


@pytest.mark.parametrize(
    ("failure_kind", "failure_type"),
    (("logical", MdpPlanError), ("transport", MdpBridgeError), ("local", MdpPlanError)),
)
def test_gate4_failure_aborts_exact_attempt_with_caught_error(
    monkeypatch, failure_kind, failure_type
):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    kwargs = {}
    if failure_kind == "logical":
        kwargs["domain_error"] = MdpPlanError("domain rejected")
    elif failure_kind == "transport":
        kwargs["domain_error"] = MdpBridgeError("domain transport")
    else:
        kwargs["attempt_error"] = RuntimeError("local preparation")
    values = _dependencies(api, monkeypatch, **kwargs)

    with pytest.raises(failure_type):
        _run(api, values)

    abort = [event for event in values.events if event[0] == "abort"]
    assert len(abort) == 1 and abort[0][1] is values.attempt
    assert isinstance(abort[0][2], failure_type)
    assert not any(event[0] == "accept" for event in values.events)


def test_gate4_post_world_accept_failure_is_task_fatal_and_poisoned(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    values = _dependencies(api, monkeypatch)
    values.gate.accept_error = RuntimeError("accept")

    with pytest.raises(MdpTaskFatalError, match="after repeated-D4 status") as caught:
        _run(api, values)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert [event[0] for event in values.events][-2:] == ["accept", "abort"]
    assert isinstance(values.events[-1][2], MdpTaskFatalError)


def test_gate4_pre_world_abort_failure_preserves_runner_error(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    primary = MdpPlanError("WORLD rejected")
    values = _dependencies(api, monkeypatch, world_error=primary)
    values.gate.abort_error = RuntimeError("abort")

    with pytest.raises(MdpPlanError) as caught:
        _run(api, values)

    assert caught.value is primary
    assert any("attempt abort" in note for note in getattr(primary, "__notes__", ()))
    assert not any(event[0] == "accept" for event in values.events)


if _WORLD8:
    from tests.unit_tests.mdp.test_dynamic_cp_d4_authority_construction import groups


def _bind_real_world8_producer(context):
    from tests.unit_tests.mdp.test_dynamic_cp_d3_producer_owner import _runtime

    owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
    runtime_api = import_module("megatron.core.mdp.dynamic_cp_runtime")
    runtime, _ = _runtime(contributor=False)
    device = context.workspace.device
    runtime.device = device
    runtime._captured_num_tokens = torch.zeros((), device=device)
    runtime._chunk_payload_bases = (torch.empty(1, 1, device=device),)
    is_source = dist.get_rank() == context.domain_ranks[0]
    outputs = MappingProxyType({})
    if is_source:
        rows = context.authority.output_rows_by_item[context.item_id]
        weight = torch.ones(
            rows,
            context.authority.bridge_width,
            dtype=context.authority.bridge_dtype,
            device=device,
            requires_grad=True,
        )
        output = weight * (context.lane + 1)
        segment = EncoderThdSegment(
            global_item_id=0,
            microbatch_id=0,
            sample_id=0,
            image_ordinal=0,
            payload_row_start=0,
            payload_rows=rows,
            output_row_start=0,
            output_rows=rows,
            grid_thw=(1, 1, rows),
        )
        layout = EncoderThdLayout(producer_worker_id=0, segments=(segment,))
        runtime._handle = EncoderForwardHandle(
            iteration=0, producer_worker_id=0, chunk_outputs=(output,), chunk_layouts=(layout,)
        )
        runtime._chunk_layouts = (layout,)
        runtime._chunk_of_item = {0: (0, segment)}
        outputs = MappingProxyType({0: runtime._handle.detached_outputs()[0]})

    rank = dist.get_rank()
    rank_view = MdpRankView(
        global_rank=rank,
        outer_dp_rank=context.lane,
        lane_id=context.lane if is_source else None,
        my_worker_id=0,
        endpoint_rank=rank,
        planning_group_ranks=(rank,),
        worker_ids=(0,),
    )
    runtime.rank_view = rank_view
    locations = (
        MappingProxyType({GlobalSampleId(context.lane, 0): (0, 0)})
        if is_source
        else MappingProxyType({})
    )
    static_plan = None
    if is_source:
        item = context.producer.local_manifest.items[0]
        static_plan = MdpPlanner(
            rank_view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()
        ).build_plan(
            0,
            (
                VisionDescriptor(
                    global_item_id=0,
                    sample_id=0,
                    image_ordinal=item.image_ordinal,
                    owner_dp_lane=context.lane,
                    microbatch_id=0,
                    estimated_cost_units=1,
                    payload_rows=1,
                    output_rows=item.output_rows,
                    grid_thw=item.grid_thw,
                    owner_worker_id=0,
                ),
            ),
            (0,),
        )
    native_owner = owner_api._capture_d3_producer_owner(
        runtime=runtime,
        rank_view=rank_view,
        local_manifest=context.producer.local_manifest if is_source else None,
        source_window=context.producer.source_window if is_source else None,
        static_plan=static_plan,
        item_outputs=outputs,
        sample_location_by_id=locations,
        forward_only=False,
    )
    bound = runtime_api._bind_pre_authority_dynamic_producer(
        producer=native_owner.producer,
        authority=context.authority,
        payload_destination_views=context.producer.payload_destination_views,
        embedding_destination_views=context.producer.embedding_destination_views,
        gradient_destination_views=context.producer.gradient_destination_views,
        summed_gradient_destination_views=context.producer.summed_gradient_destination_views,
    )
    return bound, native_owner


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
@pytest.mark.parametrize("ep", (1, 4))
def test_world8_gate4_arms_real_d3_completion_without_claim_or_backward(groups, ep):
    from tests.unit_tests.mdp.test_dynamic_cp_d4_decoder_composition import _world8_composition

    api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    gate_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding")
    owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
    context = _world8_composition(groups, ep)
    bound_producer = native_owner = None
    try:
        ready = context.coordinator.begin_iteration(context.authority)
        for leaf in ready.embedding_leaves.values():
            leaf.grad = torch.full_like(leaf, dist.get_rank() + 1)
        context.coordinator.mark_decoder_complete(ready)
        receipt = context.coordinator.end_decoder_phase(ready)
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
        prepared = api.run_repeated_d4_encoder_completion(
            context.binding,
            context.authority,
            workspace_owner=context.workspace_owner,
            producer=bound_producer,
            receipt=receipt,
            cp_partition_mode="contiguous",
            completion_gate_binding=gate,
        )

        runtime_handle = native_owner._handle
        assert gate.is_armed
        assert type(prepared.native_completion) is owner_api._PreparedNativeEncoderCompletion
        assert prepared.native_completion.owner is native_owner
        assert prepared.native_completion.handle is runtime_handle
        assert (runtime_handle is not None) is (dist.get_rank() == context.domain_ranks[0])
        if runtime_handle is not None:
            assert runtime_handle._backward_done is False
            assert runtime_handle._released is False
        assert not hasattr(api, "_execute_d3_encoder_backward")
    finally:
        if bound_producer is not None and native_owner._runtime is not None:
            bound_producer.cleanup()
        elif native_owner is not None and native_owner._runtime is not None:
            native_owner.abort()
        context.producer.cleanup()
