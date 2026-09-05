# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for repeated-D4 Gate-7 cleanup and iteration commit."""

import os
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.mdp.dynamic_cp_execution import (
    _COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL,
    _CompletedPrecollectiveConsensus,
)
from megatron.core.mdp.errors import MdpPlanError, MdpStateError, MdpTaskFatalError

_MANIFEST = bytes.fromhex("00112233445566778899aabbccddeeff")
_PLAN = bytes.fromhex("ffeeddccbbaa99887766554433221100")
_NONCE = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
_WORLD = tuple(range(8))
_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8

if _WORLD8:
    from tests.unit_tests.mdp.test_dynamic_cp_d4_authority_construction import groups


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d4_iteration_commit")


def _completed(error=None):
    return _CompletedPrecollectiveConsensus(
        error=error, _seal=_COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL
    )


class _Binding:
    def __init__(self, runner, events, *, begin_error=None):
        self.runner = runner
        self.events = events
        self.begin_error = begin_error
        self.global_rank = 0
        self.world_ranks = _WORLD

    def begin_attempt(self, **kwargs):
        self.events.append(("begin", kwargs))
        if self.begin_error is not None:
            raise self.begin_error
        return self.runner


class _Authority:
    def __init__(self):
        self.global_manifest = SimpleNamespace(digest=_MANIFEST)
        self.plan = SimpleNamespace(digest=_PLAN)


class _Workspace:
    def __init__(self, authority):
        self.authority = authority
        self.rank = 0
        self.payload_views = object()
        self.embedding_views = object()
        self.gradient_views = object()
        self.summed_gradient_views = object()


class _Producer:
    def __init__(self, authority, workspace, *, cleanup=None):
        self.authority = authority
        self.rank_view = SimpleNamespace(global_rank=0)
        self.payload_destination_views = workspace.payload_views
        self.embedding_destination_views = workspace.embedding_views
        self.gradient_destination_views = workspace.gradient_views
        self.summed_gradient_destination_views = workspace.summed_gradient_views
        self.cleanup = cleanup or (lambda: None)


class _WorkspaceOwner:
    def __init__(self, authority, producer, workspace, events, *, cleanup_error=None):
        self.authority = authority
        self.producer = producer
        self.workspace = workspace
        self.events = events
        self.cleanup_error = cleanup_error
        self.cleanup_calls = 0

    @property
    def is_idle(self):
        return self.workspace is None and self.producer is None

    def require_bound_producer(self, authority, producer, /):
        self.events.append(("require-producer", authority, producer))
        if (
            authority is not self.authority
            or producer is not self.producer
            or producer.authority is not authority
            or self.workspace is None
            or self.workspace.authority is not authority
        ):
            raise MdpStateError("exact bound producer")
        return producer

    def require_workspace(self, authority):
        self.events.append(("require-workspace", authority))
        if authority is not self.authority or self.workspace is None:
            raise MdpStateError("exact active workspace")
        return self.workspace

    def cleanup_bound_producer(self, authority, producer, /):
        self.events.append(("owner-cleanup", authority, producer))
        if authority is not self.authority or producer is not self.producer:
            raise MdpStateError("exact bound producer")
        self.cleanup_calls += 1
        self.producer = None
        self.workspace = None
        if self.cleanup_error is not None:
            raise self.cleanup_error


class _Commit:
    def __init__(self, iteration=19):
        self.iteration = iteration


class _RunnerWrapper:
    def __init__(self, runner, transform):
        self.runner = runner
        self.transform = transform

    def run(self, **kwargs):
        return self.transform(self.runner, kwargs)


def _values(
    api,
    monkeypatch,
    *,
    begin_error=None,
    cleanup_error=None,
    domain_error=None,
    commit_error=None,
    runner_transform=None,
):
    events = []

    def world_gate(**kwargs):
        events.append(("world", kwargs))
        if kwargs["local_error"] is not None:
            raise MdpPlanError("WORLD rejected") from kwargs["local_error"]

    def domain_status(**kwargs):
        events.append(("domain", kwargs))
        return _completed(domain_error)

    order = import_module("megatron.core.mdp.dynamic_cp_d4_collective_order")
    runner = order._make_repeated_d4_collective_runner(
        attempt_nonce=_NONCE, world_pre_gate=world_gate, domain_status_collector=domain_status
    )
    if runner_transform is not None:
        runner = _RunnerWrapper(runner, runner_transform)
    binding = _Binding(runner, events, begin_error=begin_error)
    authority = _Authority()
    workspace = _Workspace(authority)
    callback_calls = []
    producer = _Producer(authority, workspace, cleanup=lambda: callback_calls.append("direct"))
    owner = _WorkspaceOwner(authority, producer, workspace, events, cleanup_error=cleanup_error)
    commit = _Commit()

    monkeypatch.setattr(api, "_RepeatedD4GroupBinding", _Binding)
    monkeypatch.setattr(api, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(api, "_D3WorkspaceBindingOwner", _WorkspaceOwner)
    monkeypatch.setattr(api, "_DynamicIterationWorkspace", _Workspace)
    monkeypatch.setattr(api, "_DynamicProducerCarrier", _Producer)
    monkeypatch.setattr(api, "_D3IterationCommitReady", _Commit)
    monkeypatch.setattr(
        api,
        "_snapshot_local_authority",
        lambda actual_binding, actual_authority: events.append(
            ("snapshot", actual_binding, actual_authority)
        )
        or actual_authority,
    )
    terminal_candidate = api._candidate_terminal_digest
    monkeypatch.setattr(
        api,
        "_candidate_digest",
        lambda actual, field: events.append(("candidate-manifest", actual, field)) or _MANIFEST,
    )
    monkeypatch.setattr(
        api,
        "_candidate_terminal_digest",
        lambda actual_binding, actual_authority, actual_commit: events.append(
            ("candidate-terminal", actual_binding, actual_authority, actual_commit)
        )
        or terminal_candidate(actual_binding, actual_authority, actual_commit),
    )
    monkeypatch.setattr(
        api,
        "_validate_d3_iteration_commit_ready",
        lambda actual: events.append(("validate-commit", actual)) or actual,
    )

    def execute(actual):
        events.append(("commit", actual))
        if commit_error is not None:
            raise commit_error

    monkeypatch.setattr(api, "_execute_d3_iteration_commit", execute)
    return SimpleNamespace(
        authority=authority,
        binding=binding,
        callback_calls=callback_calls,
        commit=commit,
        events=events,
        owner=owner,
        producer=producer,
        workspace=workspace,
    )


def _run(api, values, **kwargs):
    return api.run_repeated_d4_iteration_commit(
        values.binding,
        values.authority,
        workspace_owner=values.owner,
        producer=values.producer,
        commit_ready=values.commit,
        **kwargs,
    )


def test_gate7_begins_before_candidates_cleans_through_owner_and_commits_last(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    generator = lambda size: _NONCE if size == 16 else bytes(size)

    assert _run(api, values, byte_generator=generator) is None

    assert [event[0] for event in values.events] == [
        "begin",
        "candidate-manifest",
        "candidate-terminal",
        "snapshot",
        "require-producer",
        "require-workspace",
        "validate-commit",
        "owner-cleanup",
        "world",
        "domain",
        "world",
        "commit",
    ]
    assert values.events[0][1] == {"byte_generator": generator}
    assert values.callback_calls == []
    assert values.owner.cleanup_calls == 1
    assert values.owner.is_idle
    assert all(event[1]["gate_id"] == 7 for event in values.events if event[0] == "world")
    assert next(event[1] for event in values.events if event[0] == "domain")["gate_id"] == 7


def test_terminal_digest_is_deterministic_binds_all_fields_and_differs_from_gate6():
    api = _api()
    kwargs = dict(
        global_manifest_digest=_MANIFEST, plan_digest=_PLAN, iteration=19, world_ranks=_WORLD
    )
    digest = api._terminal_commit_digest(**kwargs)

    assert digest == api._terminal_commit_digest(**kwargs)
    assert type(digest) is bytes and len(digest) == 16
    assert digest != api._terminal_commit_digest(
        **{**kwargs, "global_manifest_digest": bytes(reversed(_MANIFEST))}
    )
    assert digest != api._terminal_commit_digest(
        **{**kwargs, "plan_digest": bytes(reversed(_PLAN))}
    )
    assert digest != api._terminal_commit_digest(**{**kwargs, "iteration": 20})
    assert digest != api._terminal_commit_digest(**{**kwargs, "world_ranks": tuple(range(4))})
    gate6 = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_finalize")._digest(
        b"gate-5", kwargs["iteration"], kwargs["world_ranks"]
    )
    assert digest != gate6


def test_begin_failure_preserves_primary_and_owner_cleans_once(monkeypatch):
    api = _api()
    primary = RuntimeError("begin")
    values = _values(api, monkeypatch, begin_error=primary)

    with pytest.raises(RuntimeError, match="begin") as caught:
        _run(api, values)

    assert caught.value is primary
    assert values.owner.cleanup_calls == 1
    assert values.owner.is_idle
    assert values.callback_calls == []
    assert [event[0] for event in values.events] == ["begin", "owner-cleanup"]


def test_begin_failure_keeps_primary_when_exact_owner_cleanup_also_fails(monkeypatch):
    api = _api()
    primary = RuntimeError("begin")
    cleanup = RuntimeError("cleanup")
    values = _values(api, monkeypatch, begin_error=primary, cleanup_error=cleanup)

    with pytest.raises(RuntimeError, match="begin") as caught:
        _run(api, values)

    assert caught.value is primary
    assert values.owner.cleanup_calls == 1
    assert values.owner.is_idle
    assert any("terminal cleanup error" in note for note in primary.__notes__)


def test_candidate_failure_enters_first_world_then_owner_cleans(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    monkeypatch.setattr(api, "_candidate_terminal_digest", lambda *_args: None)

    with pytest.raises(MdpPlanError, match="WORLD rejected"):
        _run(api, values)

    assert [event[0] for event in values.events] == [
        "begin",
        "candidate-manifest",
        "world",
        "owner-cleanup",
    ]
    assert values.owner.cleanup_calls == 1
    assert values.callback_calls == []
    assert not any(event[0] == "commit" for event in values.events)


@pytest.mark.parametrize(
    "mutation",
    (
        "clone",
        "authority",
        "workspace-authority",
        "rank",
        "payload-view",
        "embedding-view",
        "gradient-view",
        "summed-gradient-view",
    ),
)
def test_foreign_producer_or_workspace_authority_is_rejected_without_callback(
    monkeypatch, mutation
):
    api = _api()
    values = _values(api, monkeypatch)
    if mutation == "clone":
        clone_calls = []
        values.producer = _Producer(
            values.authority, values.workspace, cleanup=lambda: clone_calls.append("clone")
        )
    elif mutation == "authority":
        values.producer.authority = _Authority()
    elif mutation == "workspace-authority":
        values.workspace.authority = _Authority()
    elif mutation == "rank":
        values.producer.rank_view.global_rank = 1
    elif mutation == "payload-view":
        values.producer.payload_destination_views = object()
    elif mutation == "embedding-view":
        values.producer.embedding_destination_views = object()
    elif mutation == "gradient-view":
        values.producer.gradient_destination_views = object()
    else:
        values.producer.summed_gradient_destination_views = object()

    with pytest.raises(MdpPlanError, match="WORLD rejected"):
        _run(api, values)

    assert values.callback_calls == []
    assert not any(event[0] == "commit" for event in values.events)
    if mutation == "clone":
        assert clone_calls == []
        assert values.owner.cleanup_calls == 0
        assert not values.owner.is_idle
    else:
        assert values.owner.cleanup_calls == 1
        assert values.owner.is_idle


def test_local_cleanup_error_is_not_retried_and_commits_nowhere(monkeypatch):
    api = _api()
    primary = RuntimeError("cleanup")
    values = _values(api, monkeypatch, cleanup_error=primary)

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        _run(api, values)

    assert caught.value.__cause__ is primary
    assert values.owner.cleanup_calls == 1
    assert values.owner.is_idle
    assert not any(event[0] == "commit" for event in values.events)


def test_peer_cleanup_failure_prevents_commit_after_local_cleanup(monkeypatch):
    api = _api()
    peer = MdpPlanError("peer cleanup")
    values = _values(api, monkeypatch, domain_error=peer)

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        _run(api, values)

    assert caught.value.__cause__ is peer
    assert values.owner.cleanup_calls == 1
    assert values.owner.is_idle
    assert not any(event[0] == "commit" for event in values.events)


def test_prepare_reentry_cleans_once_and_never_commits(monkeypatch):
    def reenter(runner, kwargs):
        original = kwargs["prepare"]

        def twice():
            value = original()
            original()
            return value

        return runner.run(**{**kwargs, "prepare": twice})

    api = _api()
    values = _values(api, monkeypatch, runner_transform=reenter)

    with pytest.raises(MdpPlanError, match="WORLD rejected"):
        _run(api, values)

    assert values.owner.cleanup_calls == 1
    assert not any(event[0] == "commit" for event in values.events)


def test_post_world_callback_substitution_never_commits(monkeypatch):
    def substitute(runner, kwargs):
        callback = kwargs["domain_collective"]
        return runner.run(**{**kwargs, "domain_collective": lambda _value: callback(object())})

    api = _api()
    values = _values(api, monkeypatch, runner_transform=substitute)

    with pytest.raises(MdpTaskFatalError, match="exact retained commit"):
        _run(api, values)

    assert values.owner.cleanup_calls == 1
    assert values.callback_calls == []
    assert not any(event[0] == "commit" for event in values.events)


def test_commit_reentry_and_runner_result_substitution_never_commit_twice(monkeypatch):
    def reenter(runner, kwargs):
        callback = kwargs["domain_collective"]

        def twice(value):
            result = callback(value)
            callback(value)
            return result

        return runner.run(**{**kwargs, "domain_collective": twice})

    api = _api()
    values = _values(api, monkeypatch, runner_transform=reenter)
    with pytest.raises(MdpTaskFatalError, match="commit exactly once"):
        _run(api, values)
    assert [event[0] for event in values.events].count("commit") == 1

    def substitute(runner, kwargs):
        runner.run(**kwargs)
        return object()

    values = _values(api, monkeypatch, runner_transform=substitute)
    with pytest.raises(MdpTaskFatalError, match="post-WORLD commit result"):
        _run(api, values)
    assert [event[0] for event in values.events].count("commit") == 1


def test_runner_return_without_callbacks_cleans_once_and_never_commits(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch, runner_transform=lambda _runner, _kwargs: object())

    with pytest.raises(MdpTaskFatalError, match="post-WORLD commit result"):
        _run(api, values)

    assert values.owner.cleanup_calls == 1
    assert values.owner.is_idle
    assert values.callback_calls == []
    assert not any(event[0] == "commit" for event in values.events)


def test_replay_rejects_without_second_cleanup_or_commit(monkeypatch):
    api = _api()
    values = _values(api, monkeypatch)
    _run(api, values)

    with pytest.raises(MdpPlanError, match="WORLD rejected"):
        _run(api, values)

    assert values.owner.cleanup_calls == 1
    assert [event[0] for event in values.events].count("commit") == 1


def test_post_world_commit_failure_is_task_fatal(monkeypatch):
    api = _api()
    primary = RuntimeError("commit")
    values = _values(api, monkeypatch, commit_error=primary)

    with pytest.raises(MdpTaskFatalError, match="failed after repeated-D4 status") as caught:
        _run(api, values)

    assert caught.value.__cause__ is primary
    assert values.owner.cleanup_calls == 1
    assert [event[0] for event in values.events].count("commit") == 1


def _build_world8_terminal_parts(groups, ep, resources, *, fail_cleanup=False):
    from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
    from megatron.core.mdp.activation import EncoderForwardHandle
    from megatron.core.mdp.dynamic_cp import GlobalSampleId
    from megatron.core.mdp.dynamic_cp_d4_decoder_composition import (
        _D4DecoderCompositionBindings,
        _make_d4_decoder_composition,
    )
    from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment, RowCapacityPolicy
    from megatron.core.mdp.planner import MdpPlanner
    from megatron.core.mdp.protocols import VisionDescriptor
    from megatron.core.mdp.rank_mapping import MdpRankView
    from tests.unit_tests.mdp.test_dynamic_cp_d3_producer_owner import _runtime
    from tests.unit_tests.mdp.test_dynamic_cp_d4_decoder_composition import _world8_composition

    context = _world8_composition(groups, ep)
    resources.context = context
    context.producer.cleanup()
    assert context.workspace_owner.is_idle

    owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
    completion_api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_completion")
    backward_api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_backward")
    finalize_adapter_api = import_module("megatron.core.mdp.dynamic_cp_d4_encoder_finalize")
    gate_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding")
    finalize_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_finalize")
    commit_api = import_module("megatron.core.mdp.dynamic_cp_d3_iteration_commit")
    rank = dist.get_rank()
    is_source = rank == context.domain_ranks[0]
    is_follower = rank == context.domain_ranks[1]
    runtime, _ = _runtime(contributor=False, follower=is_follower)
    runtime.device = context.workspace.device
    runtime._captured_num_tokens = torch.zeros((), device=runtime.device)
    runtime._chunk_payload_bases = (torch.empty(1, 1, device=runtime.device),)
    outputs = MappingProxyType({})
    if is_source:
        rows = context.authority.output_rows_by_item[context.item_id]
        weight = torch.ones(
            rows,
            context.authority.bridge_width,
            dtype=context.authority.bridge_dtype,
            device=runtime.device,
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
        encoder_cp_follower=is_follower,
    )
    resources.native_owner = native_owner
    producer = context.workspace_owner.bind(
        authority=context.authority, producer=native_owner.producer
    )
    resources.producer = producer

    def decoder_group_getter(*, group_size):
        assert group_size == len(context.domain_ranks)
        return context.binding.domain_group

    all_to_all_calls = []

    def tracked_all_to_all(*args, **kwargs):
        all_to_all_calls.append(kwargs.get("group"))
        return dist.all_to_all_single(*args, **kwargs)

    def cleanup(actual):
        if actual is not context.authority:
            raise MdpStateError("foreign cleanup authority")
        producer.cleanup()

    coordinator = _make_d4_decoder_composition(
        bindings=_D4DecoderCompositionBindings(
            binding=context.binding,
            authority=context.authority,
            workspace_owner=context.workspace_owner,
            producer=producer,
            cp_partition_mode="contiguous",
            decoder_group_getter=decoder_group_getter,
            decoder_group_ranks_getter=dist.get_process_group_ranks,
            rebuild_microbatch=MultimodalDecoderPayloadCodec().rebuild_microbatch,
            all_to_all_single=tracked_all_to_all,
            byte_generator=None,
            failure_boundary=lambda *_args: None,
            cleanup=cleanup,
        )
    )
    decoder_ready = coordinator.begin_iteration(context.authority)
    for leaf in decoder_ready.embedding_leaves.values():
        leaf.grad = torch.full_like(leaf, rank + 1)
    coordinator.mark_decoder_complete(decoder_ready)
    receipt = coordinator.end_decoder_phase(decoder_ready)

    world, _domain = groups
    runtime.process_groups = SimpleNamespace(encoder_reduction_group=world, world_group=world)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external repeated-D4 gate must not nest status collectives")

    completion_gate = gate_api._make_d3_encoder_completion_gate_binding(
        workspace_owner=context.workspace_owner,
        cp_partition_mode="contiguous",
        group=context.binding.domain_group,
        group_ranks=context.domain_ranks,
        global_rank=rank,
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
    handle = native_owner._handle
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
        group_ranks=_WORLD,
        global_rank=rank,
        device=context.workspace.device,
        timeout_seconds=30.0,
        fallback_status_gate=forbidden,
        all_gather_status=forbidden,
        group_ranks_getter=dist.get_process_group_ranks,
    )
    commit_ready = finalize_adapter_api.run_repeated_d4_encoder_finalize(
        context.binding, context.authority, ready=ready, finalize_binding=finalize_binding
    )
    assert commit_api._validate_ready(commit_ready) is commit_ready
    assert commit_ready.runtime is runtime
    assert native_owner._state == "retired" and native_owner._runtime is None
    assert native_owner._encoder_cp_follower is is_follower
    assert (handle is not None) is (rank in context.domain_ranks[:2])
    if handle is not None:
        assert handle._backward_done is True and handle._released is True
    assert all(group is context.binding.domain_group for group in all_to_all_calls)
    assert all(key.endpoint_rank in context.domain_ranks for key in receipt.received_tensors)

    cleanup_calls = []
    real_cleanup = context.workspace_owner._bound_cleanup

    def tracked_cleanup():
        cleanup_calls.append(rank)
        real_cleanup()
        if fail_cleanup and rank == 1:
            raise RuntimeError("rank-1 cleanup failure")

    context.workspace_owner._bound_cleanup = tracked_cleanup
    return context, producer, native_owner, runtime, commit_ready, cleanup_calls


def _cleanup_world8_terminal_resources(resources, primary=None):
    errors = []
    context = resources.context
    producer = resources.producer
    native_owner = resources.native_owner
    if context is not None and producer is not None and not context.workspace_owner.is_idle:
        try:
            context.workspace_owner.cleanup_bound_producer(context.authority, producer)
        except BaseException as error:
            errors.append(error)
    if native_owner is not None and native_owner._runtime is not None:
        try:
            native_owner.abort()
        except BaseException as error:
            errors.append(error)
    if context is not None:
        try:
            context.producer.cleanup()
        except BaseException as error:
            errors.append(error)
    if primary is not None:
        for error in errors:
            try:
                primary.add_note(f"suppressed world8 terminal fixture cleanup error: {error!r}")
            except BaseException:
                pass
    elif errors:
        raise errors[0]


def _world8_terminal_parts(groups, ep, *, fail_cleanup=False):
    resources = SimpleNamespace(context=None, producer=None, native_owner=None)
    try:
        parts = _build_world8_terminal_parts(groups, ep, resources, fail_cleanup=fail_cleanup)
    except BaseException as error:
        _cleanup_world8_terminal_resources(resources, error)
        raise
    return (*parts, resources)


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
@pytest.mark.parametrize("ep", (1, 4))
def test_world8_gate7_cleans_real_bound_workspace_and_commits_once(groups, ep):
    api = _api()
    context, producer, _native_owner, runtime, commit_ready, cleanup_calls, resources = (
        _world8_terminal_parts(groups, ep)
    )
    try:
        iteration = runtime.iteration
        api.run_repeated_d4_iteration_commit(
            context.binding,
            context.authority,
            workspace_owner=context.workspace_owner,
            producer=producer,
            commit_ready=commit_ready,
        )

        assert cleanup_calls == [dist.get_rank()]
        assert context.workspace_owner.is_idle
        assert runtime.iteration == iteration + 1
        assert commit_ready.runtime is None and commit_ready.token is None
        assert commit_ready.token_authority == ()
    finally:
        _cleanup_world8_terminal_resources(resources)


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_rank1_cleanup_failure_rejects_all_ranks_before_fresh_success(groups):
    api = _api()
    context, producer, _native_owner, runtime, commit_ready, cleanup_calls, resources = (
        _world8_terminal_parts(groups, 4, fail_cleanup=True)
    )
    retry_resources = None
    try:
        iteration = runtime.iteration
        with pytest.raises(MdpPlanError, match="WORLD"):
            api.run_repeated_d4_iteration_commit(
                context.binding,
                context.authority,
                workspace_owner=context.workspace_owner,
                producer=producer,
                commit_ready=commit_ready,
            )

        assert cleanup_calls == [dist.get_rank()]
        assert context.workspace_owner.is_idle
        assert runtime.iteration == iteration
        assert commit_ready.runtime is runtime

        (
            retry_context,
            retry_producer,
            _retry_native_owner,
            retry_runtime,
            retry_ready,
            retry_cleanup,
            retry_resources,
        ) = _world8_terminal_parts(groups, 4)
        retry_iteration = retry_runtime.iteration
        api.run_repeated_d4_iteration_commit(
            retry_context.binding,
            retry_context.authority,
            workspace_owner=retry_context.workspace_owner,
            producer=retry_producer,
            commit_ready=retry_ready,
        )
        assert retry_cleanup == [dist.get_rank()]
        assert retry_context.workspace_owner.is_idle
        assert retry_runtime.iteration == retry_iteration + 1
        assert retry_ready.runtime is None and retry_ready.token is None
        assert retry_ready.token_authority == ()
    finally:
        if retry_resources is not None:
            _cleanup_world8_terminal_resources(retry_resources)
        _cleanup_world8_terminal_resources(resources)
