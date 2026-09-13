# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 physical Gate-5 and encoder-finalization contracts."""

import gc
import os
import weakref
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from megatron.core.mdp.errors import (
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)
from megatron.core.mdp.runtime import MdpRuntimeState
from tests.unit_tests.mdp.test_dynamic_cp_d3_encoder_backward import _gate, _parts
from tests.unit_tests.mdp.test_dynamic_cp_d3_producer_owner import _capture, _runtime


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_encoder_finalize")


@pytest.fixture(autouse=True)
def _attempt_access_registry_is_drained():
    registry = _api()._ACTIVE_ATTEMPT_ACCESS
    assert registry == {}
    yield
    assert registry == {}


class _Group:
    def __init__(self, ranks, global_rank):
        self.ranks = ranks
        self.global_rank = global_rank

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(self.global_rank)


class _Context:
    def __init__(self, gate_id, authority, phase_value, ready):
        self.gate_id = gate_id
        self.authority = authority
        self.phase_value = phase_value
        self.ready = ready


class _HostileFinalizePrimary(BaseException):
    def __init__(self, message="primary"):
        super().__init__(message)
        self.add_note_calls = 0

    def add_note(self, _note):
        self.add_note_calls += 1
        raise BaseException("hostile add_note")


def _ready(monkeypatch, *, contributor=False, follower=False):
    monkeypatch.setattr(_api(), "_DynamicProducerCarrier", SimpleNamespace)
    backward_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_backward")
    runtime, outputs, owner, native, prepared = _parts(contributor=contributor, follower=follower)
    ready = backward_api._execute_d3_encoder_backward(_gate(monkeypatch, prepared), prepared)
    context = _Context(5, ready.prepared.authority, ready, ready.prepared.receipt.prepared.ready)
    runtime.rank_view.global_rank = 0
    group = _Group((0,), 0)
    runtime.process_groups = SimpleNamespace(encoder_reduction_group=group, world_group=group)
    return runtime, outputs, owner, native, ready, context, group


def _binding(monkeypatch, runtime, group, *, gather=None, ranks=(0,)):
    api = _api()
    monkeypatch.setattr(api, "_D3GateStatusContext", _Context)
    return api._make_d3_encoder_finalize_binding(
        group=group,
        group_ranks=ranks,
        global_rank=0,
        device=runtime.device,
        timeout_seconds=1.0,
        fallback_status_gate=lambda *_args: None,
        all_gather_status=(
            gather or (lambda wire, **_kwargs: tuple((rank, *wire[1:]) for rank in ranks))
        ),
        group_ranks_getter=lambda value: value.ranks,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_prepare_status_attempt_builds_local_gate5_status_without_collective_or_finalize(
    monkeypatch,
):
    api = _api()
    runtime, _outputs, owner, native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    monkeypatch.setattr(
        api,
        "_run_precollective_consensus",
        lambda *_args, **_kwargs: pytest.fail("external preparation must not run consensus"),
    )
    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"),
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: pytest.fail("preparation must not finalize encoder grads"),
    )

    attempt = binding.prepare_status_attempt(context, None)
    prepared = attempt._prepared
    try:
        assert type(attempt) is api._D3EncoderFinalizeAttempt
        assert attempt._binding is binding
        assert attempt._ready is ready and prepared.owner is owner
        assert attempt.error is None
        assert attempt.status == api._PrecollectiveStatus(
            0,
            api._digest(b"topology", prepared.iteration, (0,)),
            api._digest(b"gate-5", prepared.iteration, (0,)),
            0,
            5,
        )
        assert attempt.status is not attempt._status
        assert binding._state == "claimed"
        assert binding._attempt is attempt
        assert binding._attempt_resources == (prepared, owner)
        assert binding._attempt_resources_fingerprint == binding._fingerprint_resources(
            (prepared, owner)
        )
        assert native.handle is None
        assert runtime._ddp_calls == []
    finally:
        binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))


def test_finalize_attempt_direct_construction_is_rejected():
    api = _api()

    with pytest.raises(MdpStateError, match="minted"):
        api._D3EncoderFinalizeAttempt(object(), object(), None, object(), object(), object())


@pytest.mark.parametrize(("field", "accessor"), (("_status", "status"), ("_error", "error")))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_finalize_attempt_accessors_never_expose_substituted_live_fields(
    monkeypatch, field, accessor
):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    calls = []
    aborts = []
    owner_type = type(owner)
    real_abort = owner_type.abort

    class Bomb:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("substituted attempt field must not be observed")

    def tracked_abort(candidate, error=None):
        aborts.append((candidate, error))
        return real_abort(candidate, error)

    monkeypatch.setattr(owner_type, "abort", tracked_abort)
    object.__setattr__(attempt, field, Bomb())

    with pytest.raises(MdpTaskFatalError, match="sealed"):
        getattr(attempt, accessor)

    assert calls == []
    assert len(aborts) == 1 and aborts[0][0] is owner
    assert type(aborts[0][1]) is MdpTaskFatalError
    assert owner._runtime is None and binding.is_poisoned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_finalize_attempt_accessor_rejects_foreign_binding_without_calling_it(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    calls = []

    class BombBinding:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("substituted binding must not be invoked")

    object.__setattr__(attempt, "_binding", BombBinding())
    with pytest.raises(MdpTaskFatalError, match="sealed"):
        _ = attempt.status

    assert calls == []
    assert owner._runtime is None and binding.is_poisoned
    assert id(attempt) not in _api()._ACTIVE_ATTEMPT_ACCESS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_finalize_attempt_access_registry_collision_cleans_prepared_owner(monkeypatch):
    api = _api()
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)

    class CollisionRegistry(dict):
        def __contains__(self, _key):
            return True

    monkeypatch.setattr(api, "_ACTIVE_ATTEMPT_ACCESS", CollisionRegistry())

    with pytest.raises(MdpStateError, match="collided"):
        binding.prepare_status_attempt(context, None)

    assert owner._runtime is None and binding.is_poisoned
    assert binding._attempt is binding._attempt_trusted is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_prepare_status_attempt_reentry_does_not_create_second_cleanup_authority(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    prepared = attempt._prepared
    try:
        with pytest.raises(MdpStateError, match="claimed"):
            binding.prepare_status_attempt(context, None)
        assert binding._attempt_resources == (prepared, owner)
        assert owner._state == "finalization-prepared"
    finally:
        binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_finalize_attempt_mutation_poisons_and_cleans_binding_trusted_owner(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    object.__setattr__(attempt, "_prepared", object())

    with pytest.raises(MdpTaskFatalError, match="sealed"):
        binding._require_active_attempt(attempt)

    assert binding.is_poisoned
    assert binding._attempt is binding._attempt_resources is None
    assert owner._runtime is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_finalize_attempt_resource_substitution_cleans_original_not_live_foreign_owner(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    (
        foreign_runtime,
        _outputs,
        foreign_owner,
        _native,
        _ready_value,
        foreign_context,
        foreign_group,
    ) = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    foreign_binding = _binding(monkeypatch, foreign_runtime, foreign_group)
    attempt = binding.prepare_status_attempt(context, None)
    foreign_attempt = foreign_binding.prepare_status_attempt(foreign_context, None)
    binding._attempt_resources = foreign_binding._attempt_resources
    try:
        with pytest.raises(MdpTaskFatalError, match="sealed"):
            binding._require_active_attempt(attempt)

        assert binding.is_poisoned
        assert binding._attempt is binding._attempt_trusted_resources is None
        assert owner._runtime is None
        assert foreign_owner._runtime is foreign_runtime
        assert foreign_binding._attempt is foreign_attempt
    finally:
        if foreign_owner._runtime is not None:
            foreign_binding.abort_status_attempt(foreign_attempt, RuntimeError("test cleanup"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_foreign_finalize_binding_rejects_active_attempt(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    foreign = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    try:
        with pytest.raises(MdpStateError, match="active attempt"):
            foreign._require_active_attempt(attempt)
        assert foreign.is_idle
        assert binding._attempt_resources == (attempt._prepared, owner)
    finally:
        binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_prepare_failure_never_cleans_owner_substituted_by_release_callback(monkeypatch):
    api = _api()
    runtime, _outputs, owner, native, ready, context, group = _ready(monkeypatch, contributor=True)
    (foreign_runtime, _outputs, foreign_owner, _native, _ready_value, _context, _group) = _ready(
        monkeypatch
    )
    binding = _binding(monkeypatch, runtime, group)
    primary = RuntimeError("release failed after owner substitution")
    first = native.allocation_bases[0]
    aborts = []
    resets = []
    owner_type = type(owner)
    real_abort = owner_type.abort
    runtime_type = type(runtime)
    real_reset = runtime_type._abort_failed_iteration

    def tracked_abort(candidate, *args, **kwargs):
        aborts.append(candidate)
        return real_abort(candidate, *args, **kwargs)

    def tracked_reset(candidate, error):
        resets.append((candidate, error))
        return real_reset(candidate, error)

    def release(value):
        if value is first:
            object.__setattr__(ready, "owner", foreign_owner)
            raise primary
        return original_release(value)

    monkeypatch.setattr(owner_type, "abort", tracked_abort)
    monkeypatch.setattr(runtime_type, "_abort_failed_iteration", tracked_reset)
    original_release = runtime.allocator.release
    runtime.allocator.release = release
    attempt = None
    try:
        attempt = binding.prepare_status_attempt(context, None)
        assert attempt.error is primary
        assert aborts == []
        assert resets == [(runtime, primary)]
        assert owner._runtime is None
        assert foreign_owner._runtime is foreign_runtime
        assert binding._attempt_resources is None
    finally:
        runtime.allocator.release = original_release
        if attempt is not None and binding._attempt_trusted is attempt:
            binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))
        if foreign_owner._runtime is not None:
            foreign_owner.abort()


@pytest.mark.parametrize("fault", ("digest", "status", "mint", "fingerprint"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_postprepare_failure_poisons_and_cleans_exact_binding_resources(monkeypatch, fault):
    api = _api()
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    primary = RuntimeError(f"postprepare {fault} failed")
    if fault == "digest":
        real_digest = api._digest

        def digest(label, *args):
            if label == b"gate-5":
                raise primary
            return real_digest(label, *args)

        monkeypatch.setattr(api, "_digest", digest)
    elif fault == "status":
        monkeypatch.setattr(
            api, "_PrecollectiveStatus", lambda *_args: (_ for _ in ()).throw(primary)
        )
    elif fault == "mint":
        monkeypatch.setattr(
            api, "_D3EncoderFinalizeAttempt", lambda *_args: (_ for _ in ()).throw(primary)
        )
    else:
        monkeypatch.setattr(
            type(binding),
            "_fingerprint_attempt",
            staticmethod(lambda _attempt: (_ for _ in ()).throw(primary)),
        )

    with pytest.raises(RuntimeError, match=f"postprepare {fault} failed") as caught:
        binding.prepare_status_attempt(context, None)

    assert caught.value is primary
    assert binding.is_poisoned
    assert binding._attempt is binding._attempt_resources is None
    assert binding._attempt_resources_fingerprint is None
    assert owner._runtime is None
    assert api._PENDING_ATTEMPT_SEALS == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_accept_finalize_attempt_only_installs_exact_armed_state(monkeypatch):
    api = _api()
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    prepared = attempt._prepared
    status = attempt.status
    monkeypatch.setattr(
        api,
        "_run_precollective_consensus",
        lambda *_args, **_kwargs: pytest.fail("accept must not run consensus"),
    )
    monkeypatch.setattr(
        api,
        "_validate_native_group_context",
        lambda *_args, **_kwargs: pytest.fail("accept must not query groups"),
    )
    monkeypatch.setattr(
        api, "_digest", lambda *_args: pytest.fail("accept must not rebuild authority")
    )
    try:
        binding.accept_status_attempt(attempt)

        assert binding.is_armed
        assert binding._armed.ready is ready
        assert binding._armed.prepared is prepared
        assert binding._armed.owner is owner
        assert binding._armed.digest == status.plan_digest
        assert binding._attempt is binding._attempt_resources is None
        assert owner._state == "finalization-prepared" and owner._runtime is runtime
        with pytest.raises(MdpStateError, match="active attempt"):
            binding.accept_status_attempt(attempt)
    finally:
        if owner._runtime is not None:
            binding._abort(prepared, owner=owner)


@pytest.mark.parametrize("primary_type", (MdpPlanError, RuntimeError))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_abort_finalize_attempt_retires_and_cleans_exact_owner_once(monkeypatch, primary_type):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    primary = primary_type("external consensus rejected")

    binding.abort_status_attempt(attempt, primary)

    assert binding.is_idle is (primary_type is MdpPlanError)
    assert binding.is_poisoned is (primary_type is RuntimeError)
    assert binding._tombstone is (ready if primary_type is MdpPlanError else None)
    assert binding._attempt is binding._attempt_trusted_resources is None
    assert id(attempt) not in _api()._ACTIVE_ATTEMPT_ACCESS
    assert owner._runtime is None
    with pytest.raises(MdpStateError, match="active attempt"):
        binding.abort_status_attempt(attempt, primary)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_abort_mutated_finalize_attempt_preserves_primary_and_cleans_original(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    primary = MdpPlanError("peer rejected mutated attempt")
    abort_errors = []
    owner_type = type(owner)
    real_abort = owner_type.abort

    def tracked_abort(candidate, error=None):
        abort_errors.append(error)
        return real_abort(candidate, error)

    monkeypatch.setattr(owner_type, "abort", tracked_abort)
    object.__setattr__(attempt, "_prepared", object())

    binding.abort_status_attempt(attempt, primary)

    assert binding.is_poisoned and owner._runtime is None
    assert abort_errors == [primary]
    assert any("sealed" in note for note in primary.__notes__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_abort_mutated_attempt_never_tests_caller_primary_truthiness(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    bool_calls = []
    abort_errors = []
    owner_type = type(owner)
    real_abort = owner_type.abort

    class BombBoolError(MdpPlanError):
        def __bool__(self):
            bool_calls.append(self)
            raise AssertionError("caller primary truthiness must not be tested")

    primary = BombBoolError("peer rejected mutated attempt")

    def tracked_abort(candidate, error=None):
        abort_errors.append(error)
        return real_abort(candidate, error)

    monkeypatch.setattr(owner_type, "abort", tracked_abort)
    object.__setattr__(attempt, "_prepared", object())

    binding.abort_status_attempt(attempt, primary)

    assert bool_calls == []
    assert abort_errors == [primary]
    assert owner._runtime is None and binding.is_poisoned
    assert any("sealed" in note for note in primary.__notes__)


@pytest.mark.parametrize("mutation", ("binding", "state"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_abort_early_invalid_exact_attempt_cleans_trusted_resources(monkeypatch, mutation):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    primary = MdpPlanError("peer rejected corrupted exact attempt")
    if mutation == "binding":
        object.__setattr__(attempt, "_binding", object())
    else:
        binding._state = "idle"

    binding.abort_status_attempt(attempt, primary)

    assert binding.is_poisoned and owner._runtime is None
    assert binding._attempt is binding._attempt_trusted_resources is None
    assert any("sealed fields" in note for note in primary.__notes__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_accept_early_invalid_exact_attempt_cleans_trusted_resources(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    object.__setattr__(attempt, "_binding", object())

    with pytest.raises(MdpTaskFatalError, match="sealed"):
        binding.accept_status_attempt(attempt)

    assert binding.is_poisoned and owner._runtime is None
    assert binding._attempt is binding._attempt_trusted_resources is None


@pytest.mark.parametrize("resolution", ("accept", "abort"))
@pytest.mark.parametrize("substitution", ("status", "resources"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_attempt_substitution_never_invokes_untrusted_callbacks(
    monkeypatch, resolution, substitution
):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    calls = []
    aborts = []
    owner_type = type(owner)
    real_abort = owner_type.abort

    class BombStatus:
        def to_wire_tuple(self):
            calls.append("status")
            raise AssertionError("untrusted status callback must not run")

    class BombResources:
        def __iter__(self):
            calls.append("resources")
            raise AssertionError("untrusted resources must not be unpacked")

    def tracked_abort(candidate, *args, **kwargs):
        aborts.append(candidate)
        return real_abort(candidate, *args, **kwargs)

    monkeypatch.setattr(owner_type, "abort", tracked_abort)
    if substitution == "status":
        object.__setattr__(attempt, "_status", BombStatus())
    else:
        binding._attempt_resources = BombResources()

    primary = MdpPlanError("peer rejected substituted attempt")
    if resolution == "accept":
        with pytest.raises(MdpTaskFatalError, match="sealed"):
            binding.accept_status_attempt(attempt)
    else:
        binding.abort_status_attempt(attempt, primary)
        assert any("sealed fields" in note for note in primary.__notes__)

    assert calls == []
    assert aborts == [owner]
    assert owner._runtime is None and binding.is_poisoned
    assert binding._attempt is binding._attempt_trusted_resources is None


@pytest.mark.parametrize("resolution", ("accept", "abort"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_foreign_attempt_rejection_never_reads_candidate_attributes(monkeypatch, resolution):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    calls = []

    class BombAttempt:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("foreign attempt attributes must not be read")

    foreign = BombAttempt()
    primary = MdpPlanError("peer rejected a foreign attempt")
    try:
        with pytest.raises(MdpStateError, match="active attempt"):
            if resolution == "accept":
                binding.accept_status_attempt(foreign)
            else:
                binding.abort_status_attempt(foreign, primary)

        assert calls == []
        assert binding._state == "claimed" and binding._attempt is attempt
        assert binding._attempt_trusted_resources == (attempt._prepared, owner)
        assert owner._runtime is runtime
    finally:
        binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))


@pytest.mark.parametrize("resolution", ("accept", "abort"))
@pytest.mark.parametrize("active_slot", ("cleared", "foreign"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_original_attempt_slot_substitution_cleans_only_trusted_resources(
    monkeypatch, resolution, active_slot
):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    foreign_binding = foreign_owner = None
    if active_slot == "cleared":
        binding._attempt = None
    else:
        (
            foreign_runtime,
            _outputs,
            foreign_owner,
            _native,
            _ready_value,
            foreign_context,
            foreign_group,
        ) = _ready(monkeypatch)
        foreign_binding = _binding(monkeypatch, foreign_runtime, foreign_group)
        binding._attempt = foreign_binding.prepare_status_attempt(foreign_context, None)

    primary = MdpPlanError("peer rejected after attempt-slot substitution")
    if resolution == "accept":
        with pytest.raises(MdpTaskFatalError, match="sealed"):
            binding.accept_status_attempt(attempt)
    else:
        binding.abort_status_attempt(attempt, primary)
        assert any("sealed fields" in note for note in primary.__notes__)

    assert binding.is_poisoned and owner._runtime is None
    assert binding._attempt is binding._attempt_trusted_resources is None
    if foreign_owner is not None:
        assert foreign_owner._runtime is foreign_runtime
        foreign_binding.abort_status_attempt(
            foreign_binding._attempt_trusted, RuntimeError("test cleanup")
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_foreign_attempt_caller_does_not_clean_valid_active_attempt(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    (
        foreign_runtime,
        _outputs,
        foreign_owner,
        _native,
        _ready_value,
        foreign_context,
        foreign_group,
    ) = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    foreign_binding = _binding(monkeypatch, foreign_runtime, foreign_group)
    attempt = binding.prepare_status_attempt(context, None)
    foreign_attempt = foreign_binding.prepare_status_attempt(foreign_context, None)
    try:
        with pytest.raises(MdpStateError, match="active attempt"):
            binding.accept_status_attempt(foreign_attempt)
        assert binding._attempt is attempt and binding._state == "claimed"
        assert owner._runtime is runtime and foreign_owner._runtime is foreign_runtime
    finally:
        binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))
        foreign_binding.abort_status_attempt(foreign_attempt, RuntimeError("test cleanup"))


def test_abort_local_error_attempt_has_no_cleanup_authority(monkeypatch):
    api = _api()
    runtime, _outputs = _runtime(contributor=False)
    group = _Group((0,), 0)
    binding = _binding(monkeypatch, runtime, group)
    local_error = RuntimeError("local Gate-5 failure")
    context = _Context(5, object(), None, object())
    attempt = binding.prepare_status_attempt(context, local_error)

    assert attempt.error is local_error
    assert attempt._prepared is None and binding._attempt_resources is None
    binding.abort_status_attempt(attempt, MdpPlanError("peer rejected"))
    assert binding.is_idle and binding._tombstone is None


def test_accept_local_error_attempt_poisons_without_cleanup_authority(monkeypatch):
    runtime, _outputs = _runtime(contributor=False)
    group = _Group((0,), 0)
    binding = _binding(monkeypatch, runtime, group)
    local_error = RuntimeError("local Gate-5 failure")
    attempt = binding.prepare_status_attempt(_Context(5, object(), None, object()), local_error)

    with pytest.raises(MdpStateError, match="accepted a local error") as caught:
        binding.accept_status_attempt(attempt)

    assert caught.value.__cause__ is local_error
    assert binding.is_poisoned
    assert binding._attempt is binding._attempt_resources is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_local_error_with_valid_ready_cleans_exact_owner_once(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    local_error = RuntimeError("local Gate-5 failure with ready")
    owner_type = type(owner)
    real_abort = owner_type.abort
    aborts = []

    def tracked_abort(candidate, *args, **kwargs):
        aborts.append(candidate)
        return real_abort(candidate, *args, **kwargs)

    monkeypatch.setattr(owner_type, "abort", tracked_abort)
    attempt = binding.prepare_status_attempt(context, local_error)

    assert attempt.error is local_error and attempt._prepared is None
    assert aborts == [owner] and owner._runtime is None
    assert binding._attempt_resources is None
    binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_hostile_supplied_local_error_survives_secondary_preparation_failure(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    primary = _HostileFinalizePrimary("hostile supplied local error")

    attempt = binding.prepare_status_attempt(context, primary)

    assert attempt.error is primary
    assert primary.add_note_calls == 1
    assert owner._runtime is None
    binding.abort_status_attempt(attempt, MdpPlanError("peer rejection"))
    assert binding.is_idle

    fresh_error = RuntimeError("fresh local failure")
    fresh = binding.prepare_status_attempt(_Context(5, object(), None, object()), fresh_error)
    assert fresh.error is fresh_error
    binding.abort_status_attempt(fresh, MdpPlanError("fresh peer rejection"))
    assert binding.is_idle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_hostile_preparation_error_survives_owner_cleanup_failure(monkeypatch):
    api = _api()
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    primary = _HostileFinalizePrimary("hostile preparation failure")
    owner_type = type(owner)
    real_abort = owner_type.abort
    abort_calls = []

    def abort_then_fail(candidate, error=None):
        abort_calls.append((candidate, error))
        real_abort(candidate, error)
        raise RuntimeError("injected owner cleanup failure")

    monkeypatch.setattr(
        api, "_validate_native_group_context", lambda *_args: (_ for _ in ()).throw(primary)
    )
    monkeypatch.setattr(owner_type, "abort", abort_then_fail)
    attempt = binding.prepare_status_attempt(context, None)

    assert attempt.error is primary
    assert primary.add_note_calls == 1
    assert abort_calls == [(owner, primary)]
    assert owner._runtime is None
    binding.abort_status_attempt(attempt, MdpPlanError("peer rejection"))
    assert binding.is_idle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_local_error_never_cleans_substituted_ready_owner(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    (
        foreign_runtime,
        _outputs,
        foreign_owner,
        _native,
        _foreign_ready,
        _foreign_context,
        _foreign_group,
    ) = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    object.__setattr__(ready, "owner", foreign_owner)

    attempt = binding.prepare_status_attempt(context, RuntimeError("local Gate-5 failure"))

    assert attempt._prepared is None
    assert owner._runtime is runtime
    assert foreign_owner._runtime is foreign_runtime
    binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))
    owner.abort()
    foreign_owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_invalid_local_error_type_with_ready_cleans_exact_owner(monkeypatch):
    api = _api()
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    assert type(ready.native_completion) is api._PreparedNativeEncoderCompletion
    assert type(ready.prepared) is api._PreparedD3EncoderCompletion
    assert type(ready.prepared.producer) is api._DynamicProducerCarrier

    attempt = binding.prepare_status_attempt(context, object())

    assert isinstance(attempt.error, MdpConfigurationError)
    assert owner._runtime is None and binding._attempt_resources is None
    binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))


@pytest.mark.parametrize("substituted", (False, True))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_group_validation_failure_uses_precaptured_exact_owner(monkeypatch, substituted):
    api = _api()
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    foreign_runtime = foreign_owner = None
    if substituted:
        (foreign_runtime, _outputs, foreign_owner, _native, _ready_value, _context, _group) = (
            _ready(monkeypatch)
        )
        object.__setattr__(ready, "owner", foreign_owner)
    binding = _binding(monkeypatch, runtime, group)
    primary = RuntimeError("group validation callback failed")
    monkeypatch.setattr(
        api, "_validate_native_group_context", lambda *_args: (_ for _ in ()).throw(primary)
    )

    attempt = binding.prepare_status_attempt(context, None)

    assert attempt.error is primary and attempt._prepared is None
    assert owner._runtime is (runtime if substituted else None)
    binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))
    if foreign_owner is not None:
        assert foreign_owner._runtime is foreign_runtime
        owner.abort()
        foreign_owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_wrong_gate_context_with_ready_cleans_precaptured_exact_owner(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    wrong_gate = _Context(6, context.authority, ready, context.ready)

    attempt = binding.prepare_status_attempt(wrong_gate, None)

    assert isinstance(attempt.error, MdpConfigurationError)
    assert owner._runtime is None and binding._attempt_resources is None
    binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))


@pytest.mark.parametrize("nested", ("native", "producer"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_raising_nested_owner_accessor_is_never_invoked_or_trusted(monkeypatch, nested):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    (foreign_runtime, _outputs, foreign_owner, _native, _ready_value, _context, _group) = _ready(
        monkeypatch
    )
    binding = _binding(monkeypatch, runtime, group)
    primary = RuntimeError(f"raising {nested} owner accessor")
    accesses = []

    class RaisingOwner:
        @property
        def owner(self):
            accesses.append(nested)
            raise primary

    if nested == "native":
        object.__setattr__(ready, "native_completion", RaisingOwner())
    else:
        object.__setattr__(ready.prepared, "producer", RaisingOwner())

    attempt = binding.prepare_status_attempt(context, None)

    assert isinstance(attempt.error, MdpConfigurationError) and accesses == []
    assert owner._runtime is runtime
    assert foreign_owner._runtime is foreign_runtime
    binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))
    owner.abort()
    foreign_owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_joint_foreign_owner_authority_and_raising_nested_carrier_cannot_redirect_cleanup(
    monkeypatch,
):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    (foreign_runtime, _outputs, foreign_owner, _native, _ready_value, _context, _group) = _ready(
        monkeypatch
    )
    binding = _binding(monkeypatch, runtime, group)
    accesses = []

    class RaisingCompletion:
        @property
        def owner(self):
            accesses.append("owner")
            raise RuntimeError("must not access foreign nested carrier")

    authority = list(ready._authority)
    authority[5] = id(foreign_owner)
    object.__setattr__(ready, "owner", foreign_owner)
    object.__setattr__(ready, "_authority", tuple(authority))
    object.__setattr__(ready, "native_completion", RaisingCompletion())

    attempt = binding.prepare_status_attempt(context, None)

    assert isinstance(attempt.error, MdpConfigurationError) and accesses == []
    assert owner._runtime is runtime and foreign_owner._runtime is foreign_runtime
    binding.abort_status_attempt(attempt, RuntimeError("test cleanup"))
    owner.abort()
    foreign_owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_abort_cleanup_failure_is_only_a_note_on_caller_primary(monkeypatch):
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    attempt = binding.prepare_status_attempt(context, None)
    primary = MdpPlanError("peer rejected")
    cleanup_error = RuntimeError("owner abort failed")
    owner_type = type(owner)
    real_abort = owner_type.abort
    monkeypatch.setattr(
        owner_type, "abort", lambda *_args, **_kwargs: (_ for _ in ()).throw(cleanup_error)
    )
    try:
        binding.abort_status_attempt(attempt, primary)
        assert binding.is_idle
        assert owner._runtime is runtime
        assert any("owner abort failed" in note for note in primary.__notes__)
    finally:
        real_abort(owner)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_status_gate_routes_prepare_consensus_accept_in_exact_order(monkeypatch):
    api = _api()
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    binding_type = type(binding)
    prepare = binding_type.prepare_status_attempt
    accept = binding_type.accept_status_attempt
    events = []

    def tracked_prepare(candidate, *args):
        events.append("prepare")
        return prepare(candidate, *args)

    def tracked_accept(candidate, *args):
        events.append("accept")
        return accept(candidate, *args)

    monkeypatch.setattr(binding_type, "prepare_status_attempt", tracked_prepare)
    monkeypatch.setattr(binding_type, "accept_status_attempt", tracked_accept)
    monkeypatch.setattr(
        api, "_run_precollective_consensus", lambda *_args, **_kwargs: events.append("consensus")
    )
    binding.status_gate(context, None)
    try:
        assert events == ["prepare", "consensus", "accept"]
        assert binding.is_armed and binding._armed.ready is ready
    finally:
        if owner._runtime is not None:
            binding._abort(binding._armed.prepared, owner=owner)


def test_status_gate_invalid_locals_accepted_by_consensus_are_task_fatal(monkeypatch):
    api = _api()
    runtime, _outputs = _runtime(contributor=False)
    group = _Group((0,), 0)
    binding = _binding(monkeypatch, runtime, group)
    local_error = RuntimeError("invalid local preparation")
    monkeypatch.setattr(api, "_run_precollective_consensus", lambda *_args, **_kwargs: None)

    with pytest.raises(MdpTaskFatalError, match="accepted an invalid") as caught:
        binding.status_gate(_Context(5, object(), None, object()), local_error)

    assert caught.value.__cause__ is local_error
    assert binding.is_poisoned


def test_status_gate_plan_rejection_preserves_local_error_cause(monkeypatch):
    api = _api()
    runtime, _outputs = _runtime(contributor=False)
    group = _Group((0,), 0)
    binding = _binding(monkeypatch, runtime, group)
    local_error = RuntimeError("invalid local preparation")
    rejection = MdpPlanError("WORLD rejected")
    monkeypatch.setattr(
        api,
        "_run_precollective_consensus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(rejection),
    )

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        binding.status_gate(_Context(5, object(), None, object()), local_error)

    assert caught.value is rejection and caught.value.__cause__ is local_error
    assert binding.is_idle


def test_status_gate_plan_rejection_uses_precallback_local_error_snapshot(monkeypatch):
    api = _api()
    runtime, _outputs = _runtime(contributor=False)
    group = _Group((0,), 0)
    binding = _binding(monkeypatch, runtime, group)
    local_error = RuntimeError("original local preparation error")
    substituted = RuntimeError("substituted callback error")
    rejection = MdpPlanError("WORLD rejected")
    binding_type = type(binding)
    abort = binding_type.abort_status_attempt

    def mutate_then_abort(candidate, attempt, primary):
        object.__setattr__(attempt, "_error", substituted)
        return abort(candidate, attempt, primary)

    monkeypatch.setattr(
        api,
        "_run_precollective_consensus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(rejection),
    )
    monkeypatch.setattr(binding_type, "abort_status_attempt", mutate_then_abort)

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        binding.status_gate(_Context(5, object(), None, object()), local_error)

    assert caught.value is rejection and caught.value.__cause__ is local_error
    assert substituted is not caught.value.__cause__
    assert binding.is_poisoned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_status_gate_success_with_cleared_attempt_slot_fails_closed(monkeypatch):
    api = _api()
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)

    def clear_then_accept(*_args, **_kwargs):
        binding._attempt = None

    monkeypatch.setattr(api, "_run_precollective_consensus", clear_then_accept)

    with pytest.raises(MdpTaskFatalError, match="sealed"):
        binding.status_gate(context, None)

    assert binding.is_poisoned and owner._runtime is None


@pytest.mark.parametrize("active_slot", ("cleared", "foreign"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_status_gate_peer_error_after_attempt_slot_substitution_preserves_primary(
    monkeypatch, active_slot
):
    api = _api()
    runtime, _outputs, owner, _native, _ready_value, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    foreign_binding = foreign_owner = foreign_attempt = None
    if active_slot == "foreign":
        (
            foreign_runtime,
            _outputs,
            foreign_owner,
            _native,
            _ready_value,
            foreign_context,
            foreign_group,
        ) = _ready(monkeypatch)
        foreign_binding = _binding(monkeypatch, foreign_runtime, foreign_group)
        foreign_attempt = foreign_binding.prepare_status_attempt(foreign_context, None)
    rejection = MdpPlanError("peer rejected after slot substitution")

    def substitute_then_reject(*_args, **_kwargs):
        binding._attempt = None if active_slot == "cleared" else foreign_attempt
        raise rejection

    monkeypatch.setattr(api, "_run_precollective_consensus", substitute_then_reject)
    try:
        with pytest.raises(MdpPlanError, match="peer rejected") as caught:
            binding.status_gate(context, None)
        assert caught.value is rejection
        assert any("sealed fields" in note for note in rejection.__notes__)
        assert binding.is_poisoned and owner._runtime is None
        if foreign_owner is not None:
            assert foreign_owner._runtime is foreign_runtime
    finally:
        if foreign_owner is not None:
            foreign_binding.abort_status_attempt(foreign_attempt, RuntimeError("test cleanup"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
@pytest.mark.parametrize(
    ("contributor", "follower"), ((True, False), (False, True), (False, False))
)
def test_gate5_releases_locals_then_finalizes_exactly_once(monkeypatch, contributor, follower):
    api = _api()
    runtime, _outputs, owner, native, ready, context, group = _ready(
        monkeypatch, contributor=contributor, follower=follower
    )
    handle, bases, pixels, token = (
        native.handle,
        native.allocation_bases,
        runtime._chunk_payload_bases,
        runtime._captured_num_tokens,
    )
    events = []
    encoder_api = import_module("megatron.core.mdp.encoder")
    real_finalize = encoder_api.finalize_encoder_grads

    def finalize(ddp, *, globally_reduced_num_tokens):
        assert owner._state == "finalization-prepared"
        assert handle is None or handle._released
        assert all(
            any(value is released for released in runtime.allocator.released)
            for value in bases + pixels
        )
        events.append((ddp, globally_reduced_num_tokens))
        return real_finalize(ddp, globally_reduced_num_tokens=globally_reduced_num_tokens)

    monkeypatch.setattr(encoder_api, "finalize_encoder_grads", finalize)
    binding = _binding(monkeypatch, runtime, group)
    binding.status_gate(context, None)
    commit_ready = binding.finalize(ready)

    assert events == [(runtime.encoder_domain.encoder_ddp, token)]
    assert runtime._ddp_calls == ["finish", "scale"]
    assert runtime._token_consumed is True and runtime._captured_num_tokens is token
    assert commit_ready.runtime is runtime and commit_ready.token is token
    assert owner._state == "retired" and owner._runtime is None
    assert native.handle is None and native.gradient_views == () and native.allocation_bases == ()
    with pytest.raises(MdpTaskFatalError, match="replay"):
        binding.finalize(ready)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_local_release_failure_converges_before_zero_finalize(monkeypatch):
    runtime, _outputs, owner, native, ready, context, group = _ready(monkeypatch, contributor=True)
    released = []
    original = runtime.allocator.release
    first = native.allocation_bases[0]
    all_bases = native.allocation_bases

    def release(value):
        released.append(value)
        if value is first:
            raise RuntimeError("injected local release failure")
        return original(value)

    runtime.allocator.release = release
    statuses = []
    binding = _binding(
        monkeypatch, runtime, group, gather=lambda wire, **_kwargs: statuses.append(wire) or (wire,)
    )
    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"),
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: pytest.fail("local cleanup failure must precede WORLD finalize"),
    )
    with pytest.raises(MdpPlanError):
        binding.status_gate(context, None)
    assert statuses and statuses[0][-2:] == (1, 5)
    assert owner._state == "retired"
    assert all(any(value is candidate for candidate in released) for value in all_bases)
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._captured_num_tokens is None
    assert runtime._token_capture_count == 0 and runtime._token_consumed is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_iteration_outside_signed_int64_rejects_capture_before_ownership():
    owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
    runtime, outputs = _runtime(contributor=False)
    runtime._iteration = 2**63
    with pytest.raises(MdpStateError, match="runtime iteration"):
        _capture(owner_api, runtime, outputs)
    assert runtime._pre_authority_dynamic_producer is None
    assert runtime._handle is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_adversarial_huge_iteration_converges_zero_status_and_cleans(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    owner._iteration = runtime._iteration = 2**63
    statuses = []
    binding = _binding(
        monkeypatch, runtime, group, gather=lambda wire, **_kwargs: statuses.append(wire) or (wire,)
    )
    with pytest.raises(MdpPlanError):
        binding.status_gate(context, None)
    assert len(statuses) == 1 and statuses[0][1:5] == (0, 0, 0, 0)
    assert statuses[0][-2:] == (1, 5)
    assert binding.is_idle and owner._runtime is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_substituted_ready_owner_is_not_adopted_as_preparation_cleanup_authority(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    (
        foreign_runtime,
        _outputs,
        foreign_owner,
        _native,
        _foreign_ready,
        _foreign_context,
        _foreign_group,
    ) = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    object.__setattr__(ready, "owner", foreign_owner)

    with pytest.raises(MdpPlanError):
        binding.status_gate(context, None)

    assert owner._runtime is runtime
    assert foreign_owner._runtime is foreign_runtime
    assert binding.is_idle
    owner.abort()
    foreign_owner.abort()
    assert runtime.state is MdpRuntimeState.EMPTY


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_peer_failure_retires_without_finalize(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    gathers = []

    def peer_failure(wire, **_kwargs):
        gathers.append(wire)
        peer = (1, *wire[1:-2], 1, wire[-1])
        return wire, peer

    group.ranks = (0, 1)
    binding = _binding(monkeypatch, runtime, group, gather=peer_failure, ranks=(0, 1))
    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"),
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: pytest.fail("peer failure must prevent finalize"),
    )
    with pytest.raises(MdpPlanError):
        binding.status_gate(context, None)
    assert owner._state == "retired" and binding.is_idle
    with pytest.raises(MdpTaskFatalError, match="replayed"):
        binding.status_gate(context, None)
    assert binding.is_poisoned and len(gathers) == 1


@pytest.mark.parametrize("fault", ("ready", "owner", "token", "iteration", "group"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_post_status_substitution_is_task_fatal_before_finalize(monkeypatch, fault):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    binding.status_gate(context, None)
    candidate = ready
    if fault == "ready":
        candidate = object()
    elif fault == "owner":
        object.__setattr__(binding._armed.prepared, "owner", object())
    elif fault == "token":
        runtime._captured_num_tokens = torch.zeros((), device=runtime.device)
    elif fault == "iteration":
        runtime._iteration += 1
    else:
        runtime.process_groups.encoder_reduction_group = object()
    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"),
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: pytest.fail("mutation must precede finalize"),
    )
    with pytest.raises(MdpTaskFatalError):
        binding.finalize(candidate)
    assert binding.is_poisoned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_world4_status_uses_iteration_and_static_world_topology(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    group.ranks = (0, 1, 2, 3)
    wires = []

    def gather(wire, **_kwargs):
        wires.append(wire)
        return tuple((rank, *wire[1:]) for rank in group.ranks)

    binding = _binding(monkeypatch, runtime, group, gather=gather, ranks=group.ranks)
    binding.status_gate(context, None)
    assert wires[0][-1] == 5
    assert binding.is_armed
    owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_finalizer_failure_is_task_fatal_and_never_consumes_token(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    binding.status_gate(context, None)
    primary = RuntimeError("injected WORLD finalizer failure")

    def fail(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(import_module("megatron.core.mdp.encoder"), "finalize_encoder_grads", fail)
    with pytest.raises(MdpTaskFatalError, match="post-Gate-5") as caught:
        binding.finalize(ready)
    assert caught.value.__cause__ is primary
    assert runtime._token_consumed is False
    assert binding.is_poisoned and owner._runtime is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_hostile_finalizer_primary_survives_post_cleanup_failure_and_retires_owner(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    binding.status_gate(context, None)
    primary = _HostileFinalizePrimary("hostile WORLD finalizer failure")
    finalize_calls = []
    owner_type = type(owner)
    real_abort = owner_type.abort
    abort_calls = []

    def fail_finalize(*_args, **_kwargs):
        finalize_calls.append(primary)
        raise primary

    def abort_then_fail(candidate, error=None):
        abort_calls.append((candidate, error))
        real_abort(candidate, error)
        raise RuntimeError("injected post-finalize cleanup failure")

    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"), "finalize_encoder_grads", fail_finalize
    )
    monkeypatch.setattr(owner_type, "abort", abort_then_fail)
    with pytest.raises(MdpTaskFatalError, match="post-Gate-5") as caught:
        binding.finalize(ready)

    assert caught.value.__cause__ is primary
    assert finalize_calls == [primary]
    assert abort_calls == [(owner, primary)]
    assert primary.add_note_calls == 1
    assert owner._runtime is None
    assert runtime._token_consumed is False
    assert binding.is_poisoned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_recursive_finalize_claim_poisons_outer_without_consuming_token(monkeypatch):
    runtime, _outputs, owner, _native, ready, context, group = _ready(monkeypatch)
    binding = _binding(monkeypatch, runtime, group)
    binding.status_gate(context, None)

    def reenter(*_args, **_kwargs):
        with pytest.raises(MdpTaskFatalError, match="reentered"):
            binding.finalize(ready)

    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"), "finalize_encoder_grads", reenter
    )
    with pytest.raises(MdpTaskFatalError, match="one-shot"):
        binding.finalize(ready)
    assert binding.is_poisoned and runtime._token_consumed is False
    assert owner._runtime is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA retained state")
def test_success_scrubs_capabilities_and_does_not_pin_runtime_ddp_or_token(monkeypatch):
    runtime, _outputs, owner, native, ready, context, group = _ready(monkeypatch)
    runtime_ref = weakref.ref(runtime)
    ddp_ref = weakref.ref(runtime.encoder_domain.encoder_ddp)
    token_ref = weakref.ref(runtime._captured_num_tokens)
    binding = _binding(monkeypatch, runtime, group)
    binding.status_gate(context, None)
    binding.finalize(ready)

    assert all(
        getattr(ready, name) is None
        for name in (
            "prepared",
            "native_completion",
            "owner",
            "runtime",
            "handle",
            "encoder_domain",
            "encoder_ddp",
            "globally_reduced_num_tokens",
            "_authority",
        )
    )
    assert native.owner is native.runtime is native.encoder_ddp is None
    monkeypatch.undo()
    del runtime, owner, native, context, ready
    gc.collect()
    assert runtime_ref() is ddp_ref() is token_ref() is None


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
_WORLD4_RANKS = (0, 1, 2, 3)

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def finalize_group():
        Utils.initialize_model_parallel()
        group = torch.distributed.new_group(ranks=list(_WORLD4_RANKS), backend="nccl")
        yield group
        torch.distributed.barrier(group=group)
        torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_accept_reject_and_iteration_skew(monkeypatch, finalize_group):
    api = _api()
    transport = import_module("megatron.core.mdp.dynamic_cp_transport")
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(api, "_D3GateStatusContext", _Context)
    gathered = transport.make_precollective_status_gather(
        group=finalize_group, group_ranks=_WORLD4_RANKS, global_rank=rank, device=device
    )
    wires = []

    def status_gather(wire, *, timeout_seconds):
        wires.append(api._PrecollectiveStatus.from_wire_tuple(wire))
        return gathered(wire, timeout_seconds=timeout_seconds)

    def parts():
        runtime, _outputs, owner, _native, ready, context, _group = _ready(monkeypatch)
        runtime.rank_view.global_rank = rank
        runtime.process_groups = SimpleNamespace(
            encoder_reduction_group=finalize_group, world_group=finalize_group
        )
        return runtime, owner, ready, context

    def binding(runtime):
        return api._make_d3_encoder_finalize_binding(
            group=finalize_group,
            group_ranks=_WORLD4_RANKS,
            global_rank=rank,
            device=runtime.device,
            timeout_seconds=30.0,
            fallback_status_gate=lambda *_args: None,
            all_gather_status=status_gather,
            group_ranks_getter=torch.distributed.get_process_group_ranks,
        )

    # Each rank deliberately has distinct local lineage; only iteration and
    # ordered WORLD topology are cross-rank Gate-5 authority.
    runtime, owner, ready, context = parts()
    accepted = binding(runtime)
    accepted.status_gate(context, None)
    assert accepted.is_armed
    digests = [None] * 4
    torch.distributed.all_gather_object(digests, wires[-1].plan_digest, group=finalize_group)
    assert len(set(digests)) == 1
    accepted._abort(accepted._armed.prepared, owner=owner)

    runtime, owner, _ready_value, context = parts()
    peer_reject = binding(runtime)
    peer_context = _Context(5, object(), None, object()) if rank == 2 else context
    with pytest.raises(MdpPlanError):
        peer_reject.status_gate(
            peer_context, RuntimeError("rank-2 local cleanup") if rank == 2 else None
        )
    assert peer_reject.is_idle
    if owner._runtime is not None:
        owner.abort()

    runtime, owner, _ready_value, context = parts()
    if rank == 3:
        owner._iteration = runtime._iteration = 2**63
    skew = binding(runtime)
    with pytest.raises(MdpPlanError):
        skew.status_gate(context, None)
    assert skew.is_idle
    assert owner._runtime is None
