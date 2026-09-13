# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for the passive repeated-D4 transaction owner."""

import gc
import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d3_metadata_transport as transport_api
from megatron.core.mdp import dynamic_cp_d4_authority_construction as authority_api
from megatron.core.mdp import dynamic_cp_d4_dynamic_decoder_replay as dynamic_replay_api
from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_capture as capture_api
from megatron.core.mdp import dynamic_cp_d4_encoder_execution as execution_api
from megatron.core.mdp import dynamic_cp_d4_encoder_forward as forward_api
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as gradient_api
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient_finalize as gate6
from megatron.core.mdp import dynamic_cp_d4_encoder_selected_backward as gate5
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay_api
from megatron.core.mdp import dynamic_cp_d4_group_binding as binding_api
from megatron.core.mdp import dynamic_cp_d4_native_schedule as native_api
from megatron.core.mdp import dynamic_cp_d4_source_catalog as catalog_api
from megatron.core.mdp import dynamic_cp_d4_transaction as api
from megatron.core.mdp import dynamic_cp_runtime as runtime_api
from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.vision_locator import build_vision_locator_catalog


@pytest.fixture(autouse=True)
def _clean():
    api._FACTORY_SEALS.clear()
    api._ACTIVE_TRANSACTIONS.clear()
    api._TRUSTED_TRANSACTIONS.clear()
    api._RETIRED_TRANSACTIONS.clear()
    api._ACTIVE_LEASES.clear()
    api._ACTIVE_CHECKPOINT_LIFECYCLES.clear()
    api._RETIRED_CHECKPOINT_LIFECYCLES.clear()
    api._ACTIVE_CHECKPOINT_SNAPSHOTS.clear()
    yield
    api._FACTORY_SEALS.clear()
    api._ACTIVE_TRANSACTIONS.clear()
    api._TRUSTED_TRANSACTIONS.clear()
    api._RETIRED_TRANSACTIONS.clear()
    api._ACTIVE_LEASES.clear()
    api._ACTIVE_CHECKPOINT_LIFECYCLES.clear()
    api._RETIRED_CHECKPOINT_LIFECYCLES.clear()
    api._ACTIVE_CHECKPOINT_SNAPSHOTS.clear()


class _Group:
    def __init__(self, ranks):
        self.ranks = tuple(ranks)


class _Solver:
    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size):
        ids = [sample_id for sample_id, _length in sample_seqlens]
        lengths = [length for _sample_id, length in sample_seqlens]
        return ([lengths] * total_gpus, [], None, [ids] * total_gpus)


def _binding(rank=0):
    domain = tuple(range(rank // 4 * 4, rank // 4 * 4 + 4))
    return binding_api._make_repeated_d4_group_binding(
        world_group=_Group(range(8)),
        domain_group=_Group(domain),
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=torch.device("cuda", 0),
        timeout_seconds=5.0,
        group_ranks_getter=lambda group: group.ranks,
        status_gather_factory=lambda **_kwargs: lambda *_args, **_kwargs: None,
    )


def _manifest(lane):
    sample_id = GlobalSampleId(lane, 0)
    sample = DecoderSampleMetadata(sample_id, 4, 4, ())
    tensor = torch.arange(4, dtype=torch.int64).view(1, 4)
    fields = (DecoderTensorFieldSpec("input_ids", tensor.dtype, (1, 4), "cpu"),)
    packet = DecoderPayloadPacket(
        DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id,
        4,
        4,
        DecoderPayloadHeaderV1(
            DECODER_EXECUTION_SCHEMA_VERSION, lane, 0, 4, 4, 1, 1, -1
        ).to_wire_tuple(),
        fields,
        MappingProxyType({"input_ids": tensor}),
        ("position_ids",),
    )
    return finalize_decoder_source_window(
        source_dp_lane=lane, samples=(sample,), items=(), packets=(packet,)
    ).metadata_manifest()


def _capture(monkeypatch, binding, events, manifest=None):
    owner = object.__new__(capture_api._D4EncoderCaptureOwner)
    owner._trusted_binding = binding
    owner._trusted_manifest = manifest
    owner._trusted_error = None
    owner._trusted_capture_mode = VisionCaptureMode.SOURCE_PIXEL_SIDECAR
    owner._trusted_locator_catalog = build_vision_locator_catalog((), ())
    state = {"active": True}

    def require(self):
        if not state["active"]:
            raise MdpStateError("MDP: D4 encoder capture owner is retired.")
        return self

    def abort(self, primary=None):
        if not state["active"]:
            raise MdpStateError("MDP: D4 encoder capture owner is retired.")
        state["active"] = False
        events.append(("abort", "capture", primary))

    monkeypatch.setattr(capture_api._D4EncoderCaptureOwner, "require", require)
    monkeypatch.setattr(capture_api._D4EncoderCaptureOwner, "abort", abort)
    return owner, state


def test_real_source_projection_and_joint_authority_attach(monkeypatch):
    binding = _binding(0)
    manifests = (_manifest(0), _manifest(1))
    wires = tuple(transport_api.encode_decoder_source_manifest(value) for value in manifests)
    statuses = []

    def status_factory(**_kwargs):
        def gather(value, *, timeout_seconds):
            del timeout_seconds
            statuses.append(value)
            if len(statuses) == 1:
                return tuple(
                    (
                        value[0],
                        rank,
                        0,
                        int(rank in (0, 4)),
                        rank // 4 if rank in (0, 4) else -1,
                        len(wires[rank // 4]) if rank in (0, 4) else 0,
                        value[-1],
                    )
                    for rank in range(8)
                )
            return tuple((value[0], rank, *value[2:]) for rank in range(8))

        return gather

    monkeypatch.setattr(transport_api, "make_precollective_status_gather", status_factory)
    monkeypatch.setattr(
        transport_api,
        "_gather_body",
        lambda _body, **_kwargs: (wires[0], (), (), (), wires[1], (), (), ()),
    )
    capture, _state = _capture(monkeypatch, binding, [], manifests[0])
    projection = catalog_api._gather_d4_source_catalog(capture, binding)
    metadata = projection.metadata
    authority_kwargs = dict(
        decoder_max_seqlen_per_rank=8,
        decoder_minimum_cp_size=1,
        decoder_solver=_Solver(),
        encoder_max_seqlen_per_rank=8,
        encoder_minimum_cp_size=1,
        encoder_workload_query=lambda *_args, **_kwargs: pytest.fail("text-only queried"),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    authority = authority_api.build_repeated_d4_joint_iteration_authority(
        binding, metadata, **authority_kwargs
    )
    located = authority_api.build_repeated_d4_joint_iteration_authority(
        binding, metadata, locator_catalog_digest=b"l" * 16, **authority_kwargs
    )
    transaction = api._begin_d4_transaction(capture)
    assert transaction.attach_source_catalog(projection) is transaction
    with pytest.raises(MdpStateError, match="locator|joint authority"):
        transaction.attach_authority(located)
    assert transaction.attach_authority(authority) is transaction
    assert transaction.require() is transaction
    assert len(statuses) == 2
    object.__setattr__(projection, "local_source_manifest", manifests[1])
    with pytest.raises(MdpStateError, match="projected source metadata"):
        transaction.require()
    object.__setattr__(projection, "local_source_manifest", manifests[0])
    equal_metadata = DecoderMetadataGatherResult(
        metadata.global_manifest, dict(metadata.source_rank_by_lane)
    )
    object.__setattr__(projection, "metadata", equal_metadata)
    with pytest.raises(MdpStateError, match="projected source metadata"):
        transaction.require()


def test_transaction_rejects_equal_but_substituted_projected_locator_catalog(monkeypatch):
    binding = _binding(0)
    manifest = _manifest(0)
    locator_catalog = build_vision_locator_catalog((), ())
    entry = catalog_api._D4SourceCatalogEntry(0, 0, manifest, locator_catalog)
    world_catalog = catalog_api._seal_catalog((entry,))
    metadata = DecoderMetadataGatherResult(
        catalog_api.build_decoder_global_manifest((manifest,)), MappingProxyType({0: 0})
    )
    projection = catalog_api._D4SourceCatalogProjection(
        world_catalog, manifest, metadata, locator_catalog, locator_catalog.digest
    )
    capture, _state = _capture(monkeypatch, binding, [])
    transaction = api._begin_d4_transaction(capture)
    transaction.attach_source_catalog(projection)
    assert transaction.require() is transaction

    kwargs = dict(
        decoder_max_seqlen_per_rank=8,
        decoder_minimum_cp_size=1,
        decoder_solver=_Solver(),
        encoder_max_seqlen_per_rank=8,
        encoder_minimum_cp_size=1,
        encoder_workload_query=lambda *_args, **_kwargs: pytest.fail("text-only queried"),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    foreign = authority_api.build_repeated_d4_joint_iteration_authority(
        binding, metadata, locator_catalog_digest=b"x" * 16, **kwargs
    )
    with pytest.raises(MdpStateError, match="locator|joint authority"):
        transaction.attach_authority(foreign)
    source = authority_api.build_repeated_d4_joint_iteration_authority(binding, metadata, **kwargs)
    with pytest.raises(MdpStateError, match="locator|joint authority"):
        transaction.attach_authority(source)
    authority = authority_api.build_repeated_d4_joint_iteration_authority(
        binding, metadata, locator_catalog_digest=locator_catalog.digest, **kwargs
    )
    assert transaction.attach_authority(authority) is transaction
    assert transaction.require() is transaction

    object.__setattr__(projection, "local_locator_catalog", build_vision_locator_catalog((), ()))
    with pytest.raises(MdpStateError, match="projected source metadata|locator"):
        transaction.require()


def _fake_parts(monkeypatch, *, locator=False):
    events = []
    binding = object()
    capture, capture_state = _capture(monkeypatch, binding, events)
    catalog = SimpleNamespace(entries=(), digest=b"catalog")
    metadata = SimpleNamespace(global_manifest=object(), source_rank_by_lane=object())
    locator_catalog = build_vision_locator_catalog((), ()) if locator else None
    projection = catalog_api._D4SourceCatalogProjection(
        catalog,
        object(),
        metadata,
        locator_catalog,
        None if locator_catalog is None else locator_catalog.digest,
    )
    authority = object.__new__(runtime_api._DynamicIterationAuthority)
    monkeypatch.setattr(api, "_validate_projection", lambda value, actual: value)
    monkeypatch.setattr(api, "_validate_authority", lambda actual, value, candidate: candidate)
    transaction = api._begin_d4_transaction(capture)
    transaction.attach_source_catalog(projection).attach_authority(authority)
    return SimpleNamespace(
        transaction=transaction,
        binding=binding,
        authority=authority,
        capture=capture,
        capture_state=capture_state,
        events=events,
        projection=projection,
    )


def _install(monkeypatch, cls, label, parts, *, completion=None):
    value = object.__new__(cls)
    state = {"active": True, "aborts": 0}
    for name, item in (("authority", parts.authority), ("binding", parts.binding)):
        if name in getattr(cls, "__slots__", ()):
            setattr(value, name, item)
    if "capture_mode" in getattr(cls, "__slots__", ()):
        value.capture_mode = (
            VisionCaptureMode.STABLE_LOCATOR_CATALOG
            if parts.projection.local_locator_catalog is not None
            else VisionCaptureMode.SOURCE_PIXEL_SIDECAR
        )
    if "locator_catalog" in getattr(cls, "__slots__", ()):
        value.locator_catalog = parts.projection.local_locator_catalog
    if "completion" in getattr(cls, "__slots__", ()):
        value.completion = completion

    def require(self):
        if not state["active"]:
            raise MdpStateError(api._retired_message(cls))
        return self

    def abort(self, primary=None):
        if not state["active"]:
            raise MdpStateError(api._retired_message(cls))
        state["active"] = False
        state["aborts"] += 1
        parts.events.append(("abort", label, primary))

    monkeypatch.setattr(cls, "require", require)
    monkeypatch.setattr(cls, "abort", abort)
    return value, state


def test_execution_adoption_binds_exact_projected_locator_catalog(monkeypatch):
    parts = _fake_parts(monkeypatch, locator=True)
    assert "locator_catalog" in execution_api._D4EncoderExecutionClaim.__slots__
    execution, _state = _install(
        monkeypatch, execution_api._D4EncoderExecutionClaim, "execution", parts
    )
    foreign = build_vision_locator_catalog((), ())
    assert foreign == parts.projection.local_locator_catalog
    assert foreign is not parts.projection.local_locator_catalog
    execution.locator_catalog = foreign

    with pytest.raises(MdpStateError, match="locator|projected|catalog"):
        parts.transaction.begin_execution().adopt(execution)


def _adopt_to_replay(monkeypatch, parts, *, dynamic):
    execution, execution_state = _install(
        monkeypatch, execution_api._D4EncoderExecutionClaim, "execution", parts
    )
    assert parts.transaction.begin_execution().adopt(execution) is execution
    forward, _ = _install(monkeypatch, forward_api._D4EncoderForwardOwner, "forward", parts)
    parts.transaction.begin_forward().adopt(forward)
    publication, _ = _install(
        monkeypatch, forward_api._D4EncoderPublicationOwner, "publication", parts
    )
    parts.transaction.begin_publication().adopt(publication)
    replay_type = (
        dynamic_replay_api._D4DynamicDecoderReplayOwner
        if dynamic
        else replay_api._D4FixedDecoderReplayOwner
    )
    replay, replay_state = _install(monkeypatch, replay_type, "replay", parts)
    parts.transaction.begin_replay().adopt(replay)
    return SimpleNamespace(
        execution=execution,
        execution_state=execution_state,
        replay=replay,
        replay_state=replay_state,
        replay_type=replay_type,
    )


def _completion(replay, authority, *, dynamic):
    token = torch.tensor(1.0)
    if dynamic:
        completion = object.__new__(dynamic_replay_api._D4DynamicDecoderCompletion)
        for name, value in (
            ("authority", authority),
            ("globally_reduced_num_tokens", token),
            ("_owner", replay),
            ("_owner_entry", object()),
            ("_lifecycle", object()),
            ("_seal", dynamic_replay_api._COMPLETION_SEAL),
        ):
            object.__setattr__(completion, name, value)
    else:
        completion = replay_api._D4FixedDecoderCompletion(
            authority, token, replay, replay_api._COMPLETION_SEAL
        )
    return completion


def _adopt_to_native(monkeypatch, parts, *, dynamic=False):
    values = _adopt_to_replay(monkeypatch, parts, dynamic=dynamic)
    completion = _completion(values.replay, parts.authority, dynamic=dynamic)

    def require_completion(self, actual):
        self.require()
        if actual is not completion:
            raise MdpStateError("completion substituted")
        return actual

    monkeypatch.setattr(values.replay_type, "require_completion", require_completion)
    success_type = (
        native_api._D4DynamicNativeScheduleSuccess
        if dynamic
        else native_api._D4NativeScheduleSuccess
    )
    success = success_type(object(), completion, native_api._SUCCESS_SEAL)
    parts.transaction.begin_native_schedule().adopt(success)
    values.completion = completion
    return values


@pytest.mark.parametrize("dynamic", (False, True))
def test_replay_and_native_transition_retain_exact_owner_and_completion(monkeypatch, dynamic):
    parts = _fake_parts(monkeypatch)
    values = _adopt_to_native(monkeypatch, parts, dynamic=dynamic)
    entry = api._TRUSTED_TRANSACTIONS[id(parts.transaction)]
    assert entry.capability == (values.replay,)
    assert entry.completion is values.completion
    assert parts.transaction.require() is parts.transaction


@pytest.mark.parametrize(
    "replay_type",
    (replay_api._D4FixedDecoderReplayOwner, dynamic_replay_api._D4DynamicDecoderReplayOwner),
)
def test_replay_transition_rejects_subclasses(monkeypatch, replay_type):
    parts = _fake_parts(monkeypatch)
    execution, _ = _install(monkeypatch, execution_api._D4EncoderExecutionClaim, "execution", parts)
    parts.transaction.begin_execution().adopt(execution)
    forward, _ = _install(monkeypatch, forward_api._D4EncoderForwardOwner, "forward", parts)
    parts.transaction.begin_forward().adopt(forward)
    publication, publication_state = _install(
        monkeypatch, forward_api._D4EncoderPublicationOwner, "publication", parts
    )
    parts.transaction.begin_publication().adopt(publication)

    class DerivedReplayOwner(replay_type):
        pass

    with pytest.raises(MdpStateError, match="exact next phase") as caught:
        parts.transaction.begin_replay().adopt(object.__new__(DerivedReplayOwner))
    assert publication_state["aborts"] == 1
    assert parts.events[-1] == ("abort", "publication", caught.value)
    assert api._ACTIVE_TRANSACTIONS == {}


@pytest.mark.parametrize("dynamic", (False, True))
def test_native_transition_rejects_other_replay_mode_and_allows_retry(monkeypatch, dynamic):
    parts = _fake_parts(monkeypatch)
    values = _adopt_to_replay(monkeypatch, parts, dynamic=dynamic)
    fixed_completion = _completion(values.replay, parts.authority, dynamic=False)
    dynamic_completion = _completion(values.replay, parts.authority, dynamic=True)
    wrong = (
        native_api._D4NativeScheduleSuccess(object(), fixed_completion, native_api._SUCCESS_SEAL)
        if dynamic
        else native_api._D4DynamicNativeScheduleSuccess(
            object(), dynamic_completion, native_api._SUCCESS_SEAL
        )
    )
    with pytest.raises(MdpStateError, match="exact native schedule success"):
        parts.transaction.begin_native_schedule().adopt(wrong)
    assert values.replay_state["aborts"] == 1
    assert api._ACTIVE_TRANSACTIONS == {}
    fresh = _fake_parts(monkeypatch)
    assert fresh.transaction.require() is fresh.transaction


@pytest.mark.parametrize("dynamic", (False, True))
def test_native_transition_rejects_same_mode_completion_from_other_owner(monkeypatch, dynamic):
    parts = _fake_parts(monkeypatch)
    values = _adopt_to_replay(monkeypatch, parts, dynamic=dynamic)
    other_owner = object.__new__(values.replay_type)
    completion = _completion(other_owner, parts.authority, dynamic=dynamic)
    monkeypatch.setattr(values.replay_type, "require_completion", lambda _self, actual: actual)
    success_type = (
        native_api._D4DynamicNativeScheduleSuccess
        if dynamic
        else native_api._D4NativeScheduleSuccess
    )
    success = success_type(object(), completion, native_api._SUCCESS_SEAL)

    with pytest.raises(MdpStateError, match="exact replay owner"):
        parts.transaction.begin_native_schedule().adopt(success)
    assert values.replay_state["aborts"] == 1
    assert api._ACTIVE_TRANSACTIONS == {}


@pytest.mark.parametrize("gate7_state", ["active", "consumed", "corrupt"])
def test_passive_lease_chain_terminal_completion(monkeypatch, gate7_state):
    parts = _fake_parts(monkeypatch)
    values = _adopt_to_native(monkeypatch, parts)
    gradient, _ = _install(
        monkeypatch,
        gradient_api._D4EncoderGradientRouteOwner,
        "gradient",
        parts,
        completion=values.completion,
    )
    parts.transaction.begin_gradient().adopt(gradient)
    authorized, _ = _install(
        monkeypatch,
        gate4._D4EncoderBackwardAuthorizationOwner,
        "authorized",
        parts,
        completion=values.completion,
    )
    parts.transaction.begin_backward_authorization().adopt(authorized)
    backward, _ = _install(
        monkeypatch,
        gate5._D4EncoderSelectedBackwardOwner,
        "backward",
        parts,
        completion=values.completion,
    )
    parts.transaction.begin_backward().adopt(backward)
    finalized, finalized_state = _install(
        monkeypatch, gate6._D4EncoderFinalizeOwner, "finalized", parts, completion=values.completion
    )
    ready = gate6._D4EncoderOnlyCommitReady(
        finalized, object(), parts.authority, torch.tensor(1.0), 0, (), gate6._READY_SEAL
    )

    def require_ready(self, actual):
        self.require()
        if actual is not ready:
            raise MdpStateError("ready substituted")
        return actual

    monkeypatch.setattr(gate6._D4EncoderFinalizeOwner, "require_commit_ready", require_ready)
    parts.transaction.begin_finalize().adopt(ready)
    terminal = parts.transaction.begin_commit()
    assert terminal.arguments() == (parts.binding, parts.authority, finalized, ready)
    assert finalized_state["active"] is True
    assert id(parts.transaction) in api._ACTIVE_TRANSACTIONS
    if gate7_state == "consumed":
        finalized_state["active"] = False
        terminal.complete(None)
        assert finalized_state["active"] is False
    elif gate7_state == "active":
        with pytest.raises(MdpStateError, match="Gate7 commit consumes"):
            terminal.complete(None)
        assert finalized_state["active"] is False
    else:
        monkeypatch.setattr(
            gate6._D4EncoderFinalizeOwner,
            "require",
            lambda _self: (_ for _ in ()).throw(MdpStateError("corrupt Gate6 state")),
        )
        with pytest.raises(MdpStateError, match="corrupt Gate6 state"):
            terminal.complete(None)
        assert finalized_state["active"] is False
    assert api._ACTIVE_TRANSACTIONS == {}
    assert api._ACTIVE_LEASES == {}
    with pytest.raises(MdpStateError, match="lease retains"):
        terminal.arguments()


def test_preconsume_failure_aborts_prior_once(monkeypatch):
    parts = _fake_parts(monkeypatch)
    lease = parts.transaction.begin_execution()
    owner_entry = api._TRUSTED_TRANSACTIONS[id(parts.transaction)]
    lease_entry = api._ACTIVE_LEASES[id(lease)]
    original_abort = capture_api._D4EncoderCaptureOwner.abort

    def abort(self, primary=None):
        assert owner_entry.capability == ()
        assert owner_entry.binding is owner_entry.authority is None
        assert lease_entry.owner is lease_entry.prior is lease_entry.owner_entry is None
        original_abort(self, primary)

    monkeypatch.setattr(capture_api._D4EncoderCaptureOwner, "abort", abort)
    primary = RuntimeError("preconsume")
    lease.fail(primary)
    assert parts.capture_state == {"active": False}
    assert parts.events == [("abort", "capture", primary)]
    assert api._ACTIVE_TRANSACTIONS == {}


def test_postconsume_failure_suppresses_only_canonical_retired(monkeypatch):
    parts = _fake_parts(monkeypatch)
    lease = parts.transaction.begin_execution()
    parts.capture_state["active"] = False
    primary = RuntimeError("postconsume")
    lease.fail(primary)
    assert "capture owner is retired" in primary.__notes__[0]
    assert parts.events == []


def test_adoption_validation_failure_cleans_successor_not_prior(monkeypatch):
    parts = _fake_parts(monkeypatch)
    successor, state = _install(
        monkeypatch, execution_api._D4EncoderExecutionClaim, "execution", parts
    )
    successor.binding = object()
    with pytest.raises(MdpStateError, match="authority and binding") as caught:
        parts.transaction.begin_execution().adopt(successor)
    assert state["aborts"] == 1
    assert parts.capture_state["active"] is True
    assert parts.events == [("abort", "execution", None)]


def test_final_successor_require_failure_uses_intact_escrow_then_scrubs(monkeypatch):
    parts = _fake_parts(monkeypatch)
    successor, state = _install(
        monkeypatch, execution_api._D4EncoderExecutionClaim, "execution", parts
    )
    lease = parts.transaction.begin_execution()
    lease_entry = api._ACTIVE_LEASES[id(lease)]
    calls = 0

    def require(self):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MdpStateError("second successor validation failed")
        return self

    def abort(self):
        assert lease.prior is lease.owner is lease._seal is None
        assert lease_entry.prior is lease_entry.owner is lease_entry.owner_entry is None
        state["active"] = False
        state["aborts"] += 1

    monkeypatch.setattr(execution_api._D4EncoderExecutionClaim, "require", require)
    monkeypatch.setattr(execution_api._D4EncoderExecutionClaim, "abort", abort)
    with pytest.raises(MdpStateError, match="second successor validation failed"):
        lease.adopt(successor)
    assert calls == 2
    assert state["aborts"] == 1
    assert api._ACTIVE_TRANSACTIONS == {}


def test_wrong_successor_aborts_prior_and_replay_is_rejected(monkeypatch):
    parts = _fake_parts(monkeypatch)
    lease = parts.transaction.begin_execution()
    with pytest.raises(MdpStateError, match="exact next phase") as caught:
        lease.adopt(object())
    assert parts.events == [("abort", "capture", caught.value)]
    with pytest.raises(MdpStateError, match="lease retains"):
        lease.adopt(object())


@pytest.mark.parametrize("target", ("gradient", "authorized", "backward", "finalized"))
@pytest.mark.parametrize("dynamic", (False, True))
def test_decoder_completion_lineage_cannot_be_spliced(monkeypatch, target, dynamic):
    parts = _fake_parts(monkeypatch)
    values = _adopt_to_native(monkeypatch, parts, dynamic=dynamic)
    stages = (
        ("gradient", gradient_api._D4EncoderGradientRouteOwner, parts.transaction.begin_gradient),
        (
            "authorized",
            gate4._D4EncoderBackwardAuthorizationOwner,
            parts.transaction.begin_backward_authorization,
        ),
        ("backward", gate5._D4EncoderSelectedBackwardOwner, parts.transaction.begin_backward),
    )
    for label, owner_type, begin in stages:
        completion = object() if label == target else values.completion
        successor, state = _install(monkeypatch, owner_type, label, parts, completion=completion)
        lease = begin()
        if label == target:
            with pytest.raises(MdpStateError, match="completion lineage") as caught:
                lease.adopt(successor)
            assert state["aborts"] == 1
            assert parts.events[-1] == ("abort", label, caught.value)
            return
        lease.adopt(successor)
    assert target == "finalized"
    finalized, state = _install(
        monkeypatch, gate6._D4EncoderFinalizeOwner, "finalized", parts, completion=object()
    )
    ready = gate6._D4EncoderOnlyCommitReady(
        finalized, object(), parts.authority, torch.tensor(1.0), 0, (), gate6._READY_SEAL
    )
    monkeypatch.setattr(
        gate6._D4EncoderFinalizeOwner,
        "require_commit_ready",
        lambda self, actual: actual if self.require() else actual,
    )
    with pytest.raises(MdpStateError, match="completion lineage") as caught:
        parts.transaction.begin_finalize().adopt(ready)
    assert state["aborts"] == 1
    assert parts.events[-1] == ("abort", "finalized", caught.value)


def test_mutated_lease_and_transaction_abort_from_trusted_prior(monkeypatch):
    parts = _fake_parts(monkeypatch)
    lease = parts.transaction.begin_execution()
    object.__setattr__(lease, "prior", object())
    primary = RuntimeError("bad")
    lease.fail(primary)
    assert parts.events == [("abort", "capture", primary)]
    assert any("lease retains" in note for note in primary.__notes__)
    assert api._ACTIVE_TRANSACTIONS == {}


@pytest.mark.parametrize("registry", ["lease", "transaction"])
def test_pending_abort_uses_trusted_escrow_and_preserves_foreign_registry(monkeypatch, registry):
    parts = _fake_parts(monkeypatch)
    lease = parts.transaction.begin_execution()
    foreign = object()
    if registry == "lease":
        api._ACTIVE_LEASES[id(lease)] = foreign
    else:
        api._ACTIVE_TRANSACTIONS[id(parts.transaction)] = foreign
    primary = RuntimeError("stop")
    if registry == "lease":
        lease.fail(primary)
    else:
        parts.transaction.abort(primary)
    assert parts.events == [("abort", "capture", primary)]
    table = api._ACTIVE_LEASES if registry == "lease" else api._ACTIVE_TRANSACTIONS
    key = id(lease) if registry == "lease" else id(parts.transaction)
    assert table[key] is foreign


def test_pending_owner_abort_scrubs_lease_before_cleanup_callback(monkeypatch):
    parts = _fake_parts(monkeypatch)
    lease = parts.transaction.begin_execution()
    lease_entry = api._ACTIVE_LEASES[id(lease)]
    original_abort = capture_api._D4EncoderCaptureOwner.abort

    def abort(self, primary=None):
        assert lease.owner is lease.prior is lease._seal is None
        assert lease_entry.owner is lease_entry.prior is lease_entry.owner_entry is None
        with pytest.raises(MdpStateError, match="retired"):
            parts.transaction.abort(primary)
        original_abort(self, primary)

    monkeypatch.setattr(capture_api._D4EncoderCaptureOwner, "abort", abort)
    parts.transaction.abort(RuntimeError("stop"))
    assert parts.capture_state["active"] is False


def test_dropped_pending_lease_is_scrubbed_by_transaction_abort(monkeypatch):
    parts = _fake_parts(monkeypatch)
    lease = parts.transaction.begin_execution()
    lease_entry = api._ACTIVE_LEASES[id(lease)]
    reference = weakref.ref(lease)
    del lease
    gc.collect()
    assert reference() is None
    parts.transaction.abort(RuntimeError("stop"))
    assert api._ACTIVE_LEASES == {}
    assert api._TRUSTED_TRANSACTIONS == {}
    assert lease_entry.identity == 0
    assert lease_entry.owner is lease_entry.prior is lease_entry.owner_entry is None
    fresh = _fake_parts(monkeypatch)
    assert fresh.transaction.require() is fresh.transaction


@pytest.mark.parametrize("mutation", ("delete", "substitute"))
def test_stable_stage_abort_recovers_independent_transaction_escrow(monkeypatch, mutation):
    parts = _fake_parts(monkeypatch)
    successor, state = _install(
        monkeypatch, execution_api._D4EncoderExecutionClaim, "execution", parts
    )
    parts.transaction.begin_execution().adopt(successor)
    entry = api._TRUSTED_TRANSACTIONS[id(parts.transaction)]
    foreign = object()
    if mutation == "delete":
        del api._ACTIVE_TRANSACTIONS[id(parts.transaction)]
    else:
        api._ACTIVE_TRANSACTIONS[id(parts.transaction)] = foreign
    parts.transaction.abort(RuntimeError("stop"))
    assert state["aborts"] == 1
    assert entry.capability == ()
    assert entry.binding is entry.projection is entry.authority is None
    if mutation == "substitute":
        assert api._ACTIVE_TRANSACTIONS[id(parts.transaction)] is foreign
    fresh = _fake_parts(monkeypatch)
    assert fresh.transaction.require() is fresh.transaction


def test_abort_callback_reentry_sees_retired_transaction(monkeypatch):
    parts = _fake_parts(monkeypatch)
    original = capture_api._D4EncoderCaptureOwner.abort

    def abort(self, primary=None):
        with pytest.raises(MdpStateError, match="retired"):
            parts.transaction.abort(primary)
        original(self, primary)

    monkeypatch.setattr(capture_api._D4EncoderCaptureOwner, "abort", abort)
    entry = api._ACTIVE_TRANSACTIONS[id(parts.transaction)]
    parts.transaction.abort(RuntimeError("stop"))
    assert entry.capability == ()
    assert entry.binding is entry.authority is None
    assert parts.capture_state["active"] is False


@pytest.mark.parametrize(
    "owner_type",
    (
        execution_api._D4EncoderExecutionClaim,
        forward_api._D4EncoderForwardOwner,
        forward_api._D4EncoderPublicationOwner,
        replay_api._D4FixedDecoderReplayOwner,
        gradient_api._D4EncoderGradientRouteOwner,
        gate4._D4EncoderBackwardAuthorizationOwner,
        gate5._D4EncoderSelectedBackwardOwner,
        gate6._D4EncoderFinalizeOwner,
    ),
)
@pytest.mark.parametrize("variant", ("canonical", "subclass", "wrong"))
def test_retired_reconciliation_is_exact(monkeypatch, owner_type, variant):
    value = object.__new__(owner_type)
    message = api._retired_message(owner_type)

    class DerivedStateError(MdpStateError):
        pass

    error_type = DerivedStateError if variant == "subclass" else MdpStateError
    error_message = "different" if variant == "wrong" else message

    def abort(*_args, **_kwargs):
        raise error_type(error_message)

    monkeypatch.setattr(owner_type, "abort", abort)
    primary = RuntimeError("primary")
    api._abort(value, primary)
    notes = getattr(primary, "__notes__", ())
    assert (notes == ()) is (variant == "canonical")


class _CheckpointRuntime:
    pass


def _checkpoint_lifecycle(rank=0):
    binding = _binding(rank)
    runtime = _CheckpointRuntime()
    runtime.dynamic_group_binding = binding
    owner = api._bind_d4_checkpoint_lifecycle(runtime, binding)
    return runtime, binding, owner


def _assert_checkpoint_snapshot(snapshot, *, state, may_load, may_save):
    assert type(snapshot) is api._D4CheckpointLifecycleSnapshot
    assert not hasattr(snapshot, "_owner")
    assert snapshot.state is state
    assert snapshot.may_load is may_load
    assert snapshot.may_save is may_save


def test_checkpoint_lifecycle_has_closed_initial_commit_and_poison_states():
    runtime, binding, owner = _checkpoint_lifecycle()
    initial = owner.snapshot(runtime, binding)
    _assert_checkpoint_snapshot(initial, state=api._INITIAL_IDLE, may_load=True, may_save=False)

    owner.begin_iteration(runtime, binding)
    _assert_checkpoint_snapshot(
        owner.snapshot(runtime, binding),
        state=api._CHECKPOINT_ACTIVE,
        may_load=False,
        may_save=False,
    )
    with pytest.raises(MdpStateError, match="active|iteration"):
        owner.begin_iteration(runtime, binding)
    _assert_checkpoint_snapshot(
        owner.snapshot(runtime, binding),
        state=api._CHECKPOINT_ACTIVE,
        may_load=False,
        may_save=False,
    )

    owner.commit_iteration(runtime, binding)
    _assert_checkpoint_snapshot(
        owner.snapshot(runtime, binding), state=api._COMMITTED_IDLE, may_load=False, may_save=True
    )

    owner.begin_iteration(runtime, binding)
    primary = RuntimeError("capture failed")
    owner.poison(primary, runtime, binding)
    _assert_checkpoint_snapshot(
        owner.snapshot(runtime, binding),
        state=api._CHECKPOINT_POISONED,
        may_load=False,
        may_save=False,
    )
    for transition in (owner.begin_iteration, owner.commit_iteration):
        with pytest.raises(MdpStateError, match="poison"):
            transition(runtime, binding)


def test_checkpoint_lifecycle_rejects_substitution_mutation_duplicate_and_forgery():
    runtime, binding, owner = _checkpoint_lifecycle()
    active_before = dict(api._ACTIVE_CHECKPOINT_LIFECYCLES)
    with pytest.raises(MdpConfigurationError, match="bound|once|active"):
        api._bind_d4_checkpoint_lifecycle(runtime, binding)
    assert api._ACTIVE_CHECKPOINT_LIFECYCLES == active_before
    with pytest.raises(MdpConfigurationError, match="privately minted"):
        api._D4CheckpointLifecycleOwner(runtime, binding, object())
    with pytest.raises(MdpConfigurationError, match="privately minted"):
        api._D4CheckpointLifecycleSnapshot(api._INITIAL_IDLE, 0, True, False, object())
    with pytest.raises(MdpStateError, match="runtime|binding|exact"):
        owner.snapshot(_CheckpointRuntime(), binding)
    with pytest.raises(MdpStateError, match="runtime|binding|exact"):
        owner.snapshot(runtime, _binding(1))

    snapshot = owner.snapshot(runtime, binding)
    object.__setattr__(snapshot, "may_save", True)
    with pytest.raises(MdpStateError, match="snapshot|exact"):
        snapshot.require()
    object.__setattr__(owner, "_state", object())
    with pytest.raises(MdpStateError, match="state|exact"):
        owner.snapshot(runtime, binding)


def test_checkpoint_lifecycle_retirement_is_one_shot_and_scrubs_references():
    runtime, binding, owner = _checkpoint_lifecycle()
    owner.retire(runtime, binding)
    assert owner._runtime is owner._binding is None
    with pytest.raises(MdpStateError, match="retired"):
        owner.snapshot(runtime, binding)
    with pytest.raises(MdpStateError, match="retired"):
        owner.retire(runtime, binding)

    fresh_runtime = _CheckpointRuntime()
    fresh_runtime.dynamic_group_binding = binding
    fresh = api._bind_d4_checkpoint_lifecycle(fresh_runtime, binding)
    _assert_checkpoint_snapshot(
        fresh.snapshot(fresh_runtime, binding),
        state=api._INITIAL_IDLE,
        may_load=True,
        may_save=False,
    )
