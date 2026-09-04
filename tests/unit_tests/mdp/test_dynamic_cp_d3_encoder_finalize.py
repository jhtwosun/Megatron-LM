# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 physical Gate-5 and encoder-finalization contracts."""

import gc
import os
import weakref
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from megatron.core.mdp.errors import MdpPlanError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.runtime import MdpRuntimeState
from tests.unit_tests.mdp.test_dynamic_cp_d3_encoder_backward import _gate, _parts
from tests.unit_tests.mdp.test_dynamic_cp_d3_producer_owner import _capture, _runtime


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_encoder_finalize")


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


def _ready(monkeypatch, *, contributor=False, follower=False):
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
    binding.finalize(ready)

    assert events == [(runtime.encoder_domain.encoder_ddp, token)]
    assert runtime._ddp_calls == ["finish", "scale"]
    assert runtime._token_consumed is True and runtime._captured_num_tokens is token
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

    def binding():
        return api._make_d3_encoder_finalize_binding(
            group=finalize_group,
            group_ranks=_WORLD4_RANKS,
            global_rank=rank,
            device=device,
            timeout_seconds=30.0,
            fallback_status_gate=lambda *_args: None,
            all_gather_status=status_gather,
            group_ranks_getter=torch.distributed.get_process_group_ranks,
        )

    class Prepared:
        def __init__(self, iteration):
            self.iteration = iteration
            self.owner = object()

    class FakeReady:
        def __init__(self, iteration):
            self.owner = SimpleNamespace(iteration_nonce=bytes([rank]), _iteration=iteration)

    # Each rank deliberately has distinct local lineage; only iteration and
    # ordered WORLD topology are cross-rank Gate-5 authority.
    monkeypatch.setattr(api, "_D3EncoderFinalizeReady", FakeReady)
    contexts = [
        _Context(5, object(), FakeReady(17), object()),
        _Context(5, object(), FakeReady(17), object()),
        _Context(5, object(), FakeReady(18 if rank == 3 else 17), object()),
    ]
    monkeypatch.setattr(
        api,
        "_prepare",
        lambda context, *_args, **_kwargs: Prepared(context.phase_value.owner._iteration),
    )

    accepted = binding()
    accepted.status_gate(contexts[0], None)
    assert accepted.is_armed
    digests = [None] * 4
    torch.distributed.all_gather_object(digests, wires[-1].plan_digest, group=finalize_group)
    assert len(set(digests)) == 1

    peer_reject = binding()
    peer_context = _Context(5, object(), None, object()) if rank == 2 else contexts[1]
    with pytest.raises(MdpPlanError):
        peer_reject.status_gate(
            peer_context, RuntimeError("rank-2 local cleanup") if rank == 2 else None
        )
    assert peer_reject.is_idle

    skew = binding()
    with pytest.raises(MdpPlanError):
        skew.status_gate(contexts[2], None)
    assert skew.is_idle
