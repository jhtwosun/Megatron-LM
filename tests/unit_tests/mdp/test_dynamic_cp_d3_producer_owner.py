# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 production producer-owner contracts."""

import gc
import weakref
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")


class _TrackingAllocator(DirectBufferAllocator):
    def __init__(self, *, fail_after=None, invalid_at=None):
        super().__init__()
        self.fail_after = fail_after
        self.invalid_at = invalid_at
        self.acquired = []
        self.released = []

    def acquire(self, **kwargs):
        if self.fail_after is not None and len(self.acquired) == self.fail_after:
            raise RuntimeError("injected allocation failure")
        tensor = super().acquire(**kwargs)
        if len(self.acquired) == self.invalid_at:
            tensor = tensor[:1]
        self.acquired.append(tensor)
        return tensor

    def release(self, tensor):
        self.released.append(tensor)
        super().release(tensor)


def _segment(item_id, rows, output_start):
    return EncoderThdSegment(
        global_item_id=item_id,
        microbatch_id=0,
        sample_id=item_id,
        image_ordinal=0,
        payload_row_start=output_start,
        payload_rows=rows,
        output_row_start=output_start,
        output_rows=rows,
        grid_thw=(1, 1, rows),
    )


def _runtime(*, contributor=True, follower=False, mixed_dtype=False, allocator=None):
    runtime = object.__new__(MdpRuntime)
    runtime.allocator = allocator or _TrackingAllocator()
    runtime.device = torch.device("cuda", 0)
    runtime._pre_authority_dynamic_producer = None
    runtime._retired_pre_authority_dynamic_producers = {}
    runtime._window = None
    runtime._plan = None
    runtime._iter_specs = {}
    runtime._iter_ledgers = {}
    runtime._eval_outputs = ()
    runtime._captured_num_tokens = None
    runtime._token_capture_count = 0
    runtime._token_consumed = False
    runtime._state = MdpRuntimeState.EMPTY
    runtime._chunk_payload_bases = (torch.empty(3, 4, device="cuda"),)

    if not contributor and not follower:
        runtime._handle = None
        runtime._chunk_layouts = ()
        runtime._chunk_of_item = {}
        return runtime, MappingProxyType({})

    layouts = (
        EncoderThdLayout(producer_worker_id=0, segments=(_segment(0, 2, 0), _segment(1, 1, 2))),
        EncoderThdLayout(producer_worker_id=0, segments=(_segment(2, 2, 0),)),
    )
    weight0 = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    weight0.requires_grad_(True)
    output0 = weight0 * 2
    dtype1 = torch.float64 if mixed_dtype else torch.float32
    weight1 = torch.arange(8, dtype=dtype1, device="cuda").reshape(2, 4)
    weight1.requires_grad_(True)
    output1 = weight1 * 3
    runtime._handle = EncoderForwardHandle(
        iteration=0, producer_worker_id=0, chunk_outputs=(output0, output1), chunk_layouts=layouts
    )
    runtime._chunk_layouts = layouts
    runtime._chunk_of_item = {
        segment.global_item_id: (chunk_index, segment)
        for chunk_index, layout in enumerate(layouts)
        for segment in layout.segments
    }
    if follower:
        return runtime, MappingProxyType({})
    detached = runtime._handle.detached_outputs()
    outputs = MappingProxyType(
        {
            item_id: detached[chunk_index][
                segment.output_row_start : segment.output_row_start + segment.output_rows
            ]
            for item_id, (chunk_index, segment) in runtime._chunk_of_item.items()
        }
    )
    return runtime, outputs


def _capture(api, runtime, outputs, *, follower=False, bind=True):
    contributor = bool(outputs)
    rank_view = SimpleNamespace(global_rank=5, lane_id=0 if contributor else None)
    runtime.rank_view = rank_view
    metadata = object() if contributor else None
    locations = (
        MappingProxyType({GlobalSampleId(0, item_id): (0, item_id) for item_id in outputs})
        if contributor
        else MappingProxyType({})
    )
    owner = api._capture_d3_producer_owner(
        runtime=runtime,
        rank_view=rank_view,
        local_manifest=metadata,
        source_window=metadata,
        static_plan=metadata,
        item_outputs=outputs,
        sample_location_by_id=locations,
        forward_only=False,
        encoder_cp_follower=follower,
    )
    if bind:
        runtime._consume_pre_authority_dynamic_producer(owner, owner.producer)
    return owner


def _gradients(outputs):
    return MappingProxyType(
        {item_id: torch.full_like(output, item_id + 1) for item_id, output in outputs.items()}
    )


def test_factory_captures_and_registers_exact_contributor():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs, bind=False)

    assert owner._runtime is runtime
    assert owner.producer.owner is owner
    assert runtime._pre_authority_dynamic_producer is owner.producer
    assert tuple(owner.producer.item_outputs) == (0, 1, 2)
    assert all(owner.producer.item_outputs[key] is outputs[key] for key in outputs)
    with pytest.raises(MdpStateError, match="factory"):
        api._D3ProducerOwner(
            runtime=runtime,
            rank_view=runtime.rank_view,
            handle=runtime._handle,
            layouts=runtime._chunk_layouts,
            geometry=api._capture_geometry(runtime, runtime._handle, runtime._chunk_layouts),
            item_outputs=outputs,
            pixel_bases=runtime._chunk_payload_bases,
            encoder_cp_follower=False,
        )


def test_capture_requires_exact_rank_view_and_pre_routing_p2_state():
    api = _api()
    runtime, outputs = _runtime()
    runtime.rank_view = SimpleNamespace(global_rank=5, lane_id=0)
    kwargs = dict(
        runtime=runtime,
        rank_view=SimpleNamespace(global_rank=5, lane_id=0),
        local_manifest=object(),
        source_window=object(),
        static_plan=object(),
        item_outputs=outputs,
        sample_location_by_id=MappingProxyType({GlobalSampleId(0, 0): (0, 0)}),
        forward_only=False,
    )
    with pytest.raises(MdpConfigurationError, match="rank view"):
        api._capture_d3_producer_owner(**kwargs)
    kwargs["rank_view"] = runtime.rank_view
    runtime._state = MdpRuntimeState.DECODER_READY
    with pytest.raises(MdpConfigurationError, match="pre-routing"):
        api._capture_d3_producer_owner(**kwargs)
    assert runtime._pre_authority_dynamic_producer is None


@pytest.mark.parametrize("follower", (False, True))
def test_empty_noncontributor_and_future_ecp_follower_capture(follower):
    api = _api()
    runtime, outputs = _runtime(contributor=False, follower=follower)
    owner = _capture(api, runtime, outputs, follower=follower)
    assert not owner.producer.item_outputs
    completion = owner.prepare_dynamic_completion(MappingProxyType({}))
    assert completion.handle is runtime._handle
    assert len(completion.gradient_views) == (2 if follower else 0)
    assert all(torch.count_nonzero(view).item() == 0 for view in completion.gradient_views)
    assert api._validate_prepared_native_encoder_completion(completion, owner=owner) is completion


def test_exact_chunk_regroup_is_plan_ordered_and_does_not_run_backward(monkeypatch):
    api = _api()
    runtime, outputs = _runtime(mixed_dtype=True)
    owner = _capture(api, runtime, outputs)
    backward_calls = []
    monkeypatch.setattr(
        torch.autograd, "backward", lambda *args, **kwargs: backward_calls.append(args)
    )
    for name in ("all_reduce", "all_gather_into_tensor", "all_to_all_single"):
        monkeypatch.setattr(
            torch.distributed,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(f"unexpected {_name}"),
        )
    encoder_api = import_module("megatron.core.mdp.encoder")
    monkeypatch.setattr(
        encoder_api,
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: pytest.fail("unexpected encoder finalization"),
    )
    ddp_calls = []
    runtime.encoder_domain = SimpleNamespace(
        encoder_ddp=SimpleNamespace(
            finish_grad_sync=lambda: ddp_calls.append("finish"),
            scale_gradients=lambda _scale: ddp_calls.append("scale"),
        )
    )

    gradients = _gradients(outputs)
    completion = owner.prepare_dynamic_completion(gradients)

    assert backward_calls == []
    assert ddp_calls == []
    assert tuple(completion.gradient_views[0][:, 0].tolist()) == (1.0, 1.0, 2.0)
    assert tuple(completion.gradient_views[1][:, 0].tolist()) == (3.0, 3.0)
    assert completion.gradient_views[0].dtype == torch.float32
    assert completion.gradient_views[1].dtype == torch.float64
    assert all(
        base is acquired
        for base, acquired in zip(completion.allocation_bases, runtime.allocator.acquired)
    )
    assert api._validate_prepared_native_encoder_completion(completion, owner=owner) is completion


def test_native_completion_requires_factory_seal():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    with pytest.raises(MdpStateError, match="factory"):
        api._PreparedNativeEncoderCompletion(
            owner=owner,
            runtime=runtime,
            handle=runtime._handle,
            gradient_views=(),
            allocation_bases=(),
            encoder_cp_follower=False,
        )


def test_existing_binder_uses_real_owner_without_fake_completion(monkeypatch):
    api = _api()
    dynamic = import_module("megatron.core.mdp.dynamic_cp_runtime")
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs, bind=False)
    monkeypatch.setattr(dynamic, "_validate_local_singleton_producer_proof", lambda **_kwargs: None)
    authority = object.__new__(dynamic._DynamicIterationAuthority)
    global_items = tuple(GlobalVisionItemId(0, item_id) for item_id in outputs)
    for name, value in {
        "source_rank_by_lane": MappingProxyType({0: 5}),
        "producer_rank_by_item": MappingProxyType({item_id: 5 for item_id in global_items}),
        "output_rows_by_item": MappingProxyType(
            {item_id: outputs[item_id.local_item_id].shape[0] for item_id in global_items}
        ),
        "participant_ranks": (5,),
        "bridge_width": 4,
        "bridge_dtype": torch.float32,
    }.items():
        object.__setattr__(authority, name, value)
    bound = dynamic._bind_pre_authority_dynamic_producer(
        producer=owner.producer,
        authority=authority,
        payload_destination_views=MappingProxyType({}),
        embedding_destination_views=MappingProxyType({}),
        gradient_destination_views=MappingProxyType({}),
        summed_gradient_destination_views=MappingProxyType({}),
    )
    completion = bound.backward(
        MappingProxyType(
            {item_id: torch.ones_like(outputs[item_id.local_item_id]) for item_id in global_items}
        )
    )
    assert type(completion) is api._PreparedNativeEncoderCompletion
    assert runtime._pre_authority_dynamic_producer is None


@pytest.mark.parametrize("fault", ("missing", "extra", "reordered", "shape", "dtype", "grad"))
def test_gradient_faults_reject_before_allocation_or_mutation(fault):
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    gradients = dict(_gradients(outputs))
    if fault == "missing":
        gradients.pop(1)
    elif fault == "extra":
        gradients[3] = torch.ones(1, 4, device="cuda")
    elif fault == "reordered":
        gradients = {key: gradients[key] for key in (2, 1, 0)}
    elif fault == "shape":
        gradients[1] = torch.ones(2, 4, device="cuda")
    elif fault == "dtype":
        gradients[1] = gradients[1].to(torch.float64)
    elif fault == "grad":
        gradients[1].requires_grad_(True)
    with pytest.raises((MdpConfigurationError, MdpStateError)):
        owner.prepare_dynamic_completion(MappingProxyType(gradients))
    assert runtime.allocator.acquired == []
    assert runtime._handle is None


def test_alias_and_wrong_device_reject_before_allocation():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    shared = torch.ones(3, 4, device="cuda")
    aliased = MappingProxyType({0: shared[:2], 1: shared[2:], 2: shared[:2]})
    with pytest.raises(MdpStateError, match="overlap"):
        owner.prepare_dynamic_completion(aliased)
    assert owner._runtime is None

    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    wrong_device = dict(_gradients(outputs))
    wrong_device[1] = torch.ones_like(outputs[1], device="cpu")
    with pytest.raises(MdpStateError, match="matches its item output"):
        owner.prepare_dynamic_completion(MappingProxyType(wrong_device))
    assert runtime.allocator.acquired == []


def test_foreign_runtime_handle_and_output_mutation_reject_before_prepare():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    foreign, _ = _runtime()
    runtime._handle = foreign._handle
    with pytest.raises(MdpStateError, match="handle"):
        owner.prepare_dynamic_completion(_gradients(outputs))

    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    owner._runtime = foreign
    with pytest.raises(MdpStateError, match="bound exactly once"):
        owner.prepare_dynamic_completion(_gradients(outputs))

    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    object.__setattr__(owner.producer, "item_outputs", MappingProxyType(dict(outputs)))
    with pytest.raises(MdpStateError, match="producer inputs"):
        owner.prepare_dynamic_completion(_gradients(outputs))


@pytest.mark.parametrize("field", ("local_manifest", "source_window", "static_plan"))
def test_producer_source_metadata_identity_mutation_rejects(field):
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    object.__setattr__(owner.producer, field, object())
    with pytest.raises(MdpStateError, match="producer inputs"):
        owner.prepare_dynamic_completion(_gradients(outputs))
    assert owner._runtime is None


def test_producer_sample_location_identity_and_content_mutation_rejects():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    locations = owner.producer.sample_location_by_id
    object.__setattr__(owner.producer, "sample_location_by_id", MappingProxyType(dict(locations)))
    with pytest.raises(MdpStateError, match="producer inputs"):
        owner.prepare_dynamic_completion(_gradients(outputs))

    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    object.__setattr__(owner.producer, "sample_location_by_id", None)
    with pytest.raises(MdpStateError, match="immutable mapping"):
        owner.prepare_dynamic_completion(_gradients(outputs))

    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    key = next(iter(owner.producer.sample_location_by_id))
    object.__setattr__(key, "local_sample_order", 99)
    with pytest.raises(MdpStateError, match="producer inputs"):
        owner.prepare_dynamic_completion(_gradients(outputs))


def test_duplicate_prepare_and_completion_mutation_are_rejected():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    completion = owner.prepare_dynamic_completion(_gradients(outputs))
    with pytest.raises(MdpStateError, match="exactly once"):
        owner.prepare_dynamic_completion(_gradients(outputs))
    assert owner._runtime is None

    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    completion = owner.prepare_dynamic_completion(_gradients(outputs))
    object.__setattr__(completion, "gradient_views", tuple(reversed(completion.gradient_views)))
    with pytest.raises(MdpStateError, match="seal"):
        api._validate_prepared_native_encoder_completion(completion, owner=owner)


def test_completion_replay_after_abort_and_foreign_owner_rejects():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    completion = owner.prepare_dynamic_completion(_gradients(outputs))
    foreign_runtime, foreign_outputs = _runtime()
    foreign = _capture(api, foreign_runtime, foreign_outputs)
    with pytest.raises(MdpStateError):
        api._validate_prepared_native_encoder_completion(completion, owner=foreign)

    owner.abort()
    assert completion.handle is None
    with pytest.raises(MdpStateError):
        api._validate_prepared_native_encoder_completion(completion, owner=owner)


def test_partial_allocation_failure_releases_owned_bases_and_retires():
    api = _api()
    allocator = _TrackingAllocator(fail_after=1)
    runtime, outputs = _runtime(allocator=allocator)
    owner = _capture(api, runtime, outputs)
    with pytest.raises(RuntimeError, match="allocation failure"):
        owner.prepare_dynamic_completion(_gradients(outputs))
    assert all(
        any(base is released for released in allocator.released) for base in allocator.acquired
    )
    with pytest.raises(MdpStateError, match="bound exactly once"):
        owner.prepare_dynamic_completion(_gradients(outputs))
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._handle is None


def test_failed_prepare_preserves_primary_and_attempts_all_cleanup(monkeypatch):
    api = _api()
    allocator = _TrackingAllocator(fail_after=1)
    runtime, outputs = _runtime(allocator=allocator)
    owner = _capture(api, runtime, outputs)
    release_calls = []

    def release(tensor):
        release_calls.append(tensor)
        if len(release_calls) == 1:
            raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(allocator, "release", release)
    with pytest.raises(RuntimeError, match="allocation failure") as raised:
        owner.prepare_dynamic_completion(_gradients(outputs))
    assert any("cleanup failure" in note for note in getattr(raised.value, "__notes__", ()))
    assert len(release_calls) == 2  # acquired regroup base plus packed-pixel base
    assert owner._runtime is None
    with pytest.raises(MdpStateError, match="bound exactly once"):
        owner.prepare_dynamic_completion(_gradients(outputs))


def test_invalid_allocator_result_is_released_and_failed_prepare_cannot_replay():
    api = _api()
    allocator = _TrackingAllocator(invalid_at=0)
    runtime, outputs = _runtime(allocator=allocator)
    owner = _capture(api, runtime, outputs)
    with pytest.raises(MdpStateError, match="allocation"):
        owner.prepare_dynamic_completion(_gradients(outputs))
    assert len(allocator.acquired) == 1
    assert allocator.released[0] is allocator.acquired[0]
    assert owner._runtime is None
    with pytest.raises(MdpStateError, match="bound exactly once"):
        owner.prepare_dynamic_completion(_gradients(outputs))


@pytest.mark.parametrize(
    ("first_start", "second_start", "match"),
    ((-1, 1, "contiguous"), (0, 1, "contiguous"), (0, 3, "contiguous")),
)
def test_segment_negative_overlap_and_gap_reject_before_capture(first_start, second_start, match):
    api = _api()
    runtime, outputs = _runtime()
    first, second = runtime._chunk_layouts[0].segments
    object.__setattr__(first, "output_row_start", first_start)
    object.__setattr__(second, "output_row_start", second_start)
    with pytest.raises(MdpStateError, match=match):
        _capture(api, runtime, outputs)
    assert runtime._pre_authority_dynamic_producer is None


@pytest.mark.parametrize("alias", ("gradient", "output", "pixel", "prior"))
def test_allocator_aliases_are_rejected_and_released(alias):
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    allocator = runtime.allocator
    gradients = dict(_gradients(outputs))
    calls = 0
    if alias == "gradient":
        shared = torch.ones(3, 4, device="cuda")
        gradients[0], gradients[1] = shared[:2], shared[2:]
        targets = (shared,)
    elif alias == "output":
        targets = (runtime._handle.chunk_outputs[0],)
    elif alias == "pixel":
        targets = (runtime._chunk_payload_bases[0],)
    else:
        targets = ()
    original_acquire = allocator.acquire

    def acquire(**kwargs):
        nonlocal calls
        if alias == "prior" and calls == 0:
            value = original_acquire(**kwargs)
        elif alias == "prior":
            value = allocator.acquired[0][: kwargs["rows"]]
            allocator.acquired.append(value)
        else:
            value = targets[0]
            allocator.acquired.append(value)
        calls += 1
        return value

    allocator.acquire = acquire
    with pytest.raises(MdpStateError, match="non-aliased"):
        owner.prepare_dynamic_completion(MappingProxyType(gradients))
    assert owner._runtime is None
    assert allocator.released


def test_abort_retires_registry_releases_all_state_and_is_exactly_once():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    completion = owner.prepare_dynamic_completion(_gradients(outputs))
    bases = completion.allocation_bases
    packed = runtime._chunk_payload_bases
    primary = RuntimeError("primary")

    owner.abort(primary)

    assert owner._runtime is None
    assert runtime._pre_authority_dynamic_producer is None
    assert runtime._handle is None
    assert runtime.state is MdpRuntimeState.EMPTY
    assert all(any(value is released for released in runtime.allocator.released) for value in bases)
    assert all(
        any(value is released for released in runtime.allocator.released) for value in packed
    )
    assert completion.gradient_views == completion.allocation_bases == ()
    with pytest.raises(MdpStateError, match="exactly once"):
        owner.abort()


@pytest.mark.parametrize("prepared", (False, True))
def test_abort_releases_exact_handle_once_through_forward_only_path(monkeypatch, prepared):
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs, bind=prepared)
    if prepared:
        owner.prepare_dynamic_completion(_gradients(outputs))
    handle = runtime._handle
    forward_only_calls = []
    release_calls = []
    original_forward_only = EncoderForwardHandle.release_forward_only

    def release_forward_only(candidate):
        if candidate is handle:
            forward_only_calls.append(candidate)
        return original_forward_only(candidate)

    def release(candidate):
        if candidate is handle:
            release_calls.append(candidate)
        raise AssertionError("generic handle release is forbidden during D3 owner abort")

    monkeypatch.setattr(EncoderForwardHandle, "release_forward_only", release_forward_only)
    monkeypatch.setattr(EncoderForwardHandle, "release", release)
    owner.abort()

    assert forward_only_calls == [handle]
    assert release_calls == []
    assert handle._released is True


def test_abort_preserves_primary_and_attaches_cleanup_note(monkeypatch):
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    owner.prepare_dynamic_completion(_gradients(outputs))
    original_release = runtime.allocator.release
    calls = 0

    def fail_first(tensor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("cleanup")
        original_release(tensor)

    monkeypatch.setattr(runtime.allocator, "release", fail_first)
    primary = RuntimeError("primary")
    owner.abort(primary)
    assert str(primary) == "primary"
    assert any("cleanup" in note for note in getattr(primary, "__notes__", ()))
    assert calls == 3


def test_payload_substitution_is_rejected_and_only_owned_payload_is_released():
    api = _api()
    runtime, outputs = _runtime()
    owned_payload = runtime._chunk_payload_bases[0]
    owner = _capture(api, runtime, outputs)
    foreign_payload = torch.empty_like(owned_payload)
    runtime._chunk_payload_bases = (foreign_payload,)
    with pytest.raises(MdpStateError, match="packed-pixel"):
        owner.prepare_dynamic_completion(_gradients(outputs))
    assert any(value is owned_payload for value in runtime.allocator.released)
    assert all(value is not foreign_payload for value in runtime.allocator.released)


def test_follower_rejects_handle_output_substitution_and_already_backward_state():
    api = _api()
    runtime, outputs = _runtime(contributor=False, follower=True)
    owner = _capture(api, runtime, outputs, follower=True)
    runtime._handle.chunk_outputs = tuple(
        output.clone() for output in runtime._handle.chunk_outputs
    )
    with pytest.raises(MdpStateError, match="exact encoder outputs"):
        owner.prepare_dynamic_completion(MappingProxyType({}))

    runtime, outputs = _runtime(contributor=False, follower=True)
    owner = _capture(api, runtime, outputs, follower=True)
    runtime._handle._backward_done = True
    with pytest.raises(MdpStateError, match="fresh exact encoder outputs"):
        owner.prepare_dynamic_completion(MappingProxyType({}))


def test_fresh_same_runtime_capture_after_abort_and_weak_gc():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs)
    producer_ref = weakref.ref(owner.producer)
    owner.abort()
    del owner
    gc.collect()
    assert producer_ref() is None

    registry_identity = id(runtime._retired_pre_authority_dynamic_producers)
    allocator_identity = id(runtime.allocator)
    runtime2, outputs2 = _runtime()
    # Install fresh P2 state without replacing the runtime-owned registry or allocator.
    runtime._handle = runtime2._handle
    runtime._chunk_layouts = runtime2._chunk_layouts
    runtime._chunk_of_item = runtime2._chunk_of_item
    runtime._chunk_payload_bases = runtime2._chunk_payload_bases
    runtime._state = runtime2._state
    fresh = _capture(api, runtime, outputs2, bind=False)
    assert fresh._runtime is runtime
    assert runtime._pre_authority_dynamic_producer is fresh.producer
    assert id(runtime._retired_pre_authority_dynamic_producers) == registry_identity
    assert id(runtime.allocator) == allocator_identity


def test_owner_requires_successful_runtime_consume_before_prepare():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs, bind=False)
    runtime._abort_pre_authority_dynamic_producer(owner)

    with pytest.raises(MdpStateError, match="bound exactly once"):
        owner.prepare_dynamic_completion(_gradients(outputs))

    assert runtime.allocator.acquired == []
    assert owner._runtime is None
    assert runtime._handle is None


def test_owner_binding_hook_rejects_forged_and_double_transition():
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs, bind=False)

    with pytest.raises(MdpStateError, match="exact registered producer once"):
        owner._mark_pre_authority_dynamic_producer_bound(object())
    assert runtime._pre_authority_dynamic_producer is owner.producer

    runtime._consume_pre_authority_dynamic_producer(owner, owner.producer)
    with pytest.raises(MdpStateError, match="exact registered producer once"):
        owner._mark_pre_authority_dynamic_producer_bound(owner.producer)
    owner.abort()


def test_runtime_consume_hook_failure_keeps_registry_active(monkeypatch):
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs, bind=False)

    def fail_binding(_self, _producer):
        raise RuntimeError("injected owner binding failure")

    monkeypatch.setattr(
        api._D3ProducerOwner, "_mark_pre_authority_dynamic_producer_bound", fail_binding
    )
    with pytest.raises(RuntimeError, match="binding failure"):
        runtime._consume_pre_authority_dynamic_producer(owner, owner.producer)
    assert runtime._pre_authority_dynamic_producer is owner.producer
    assert owner._state == "registered"
    owner.abort()


def test_runtime_consume_rejects_noncallable_owner_hook_without_retiring(monkeypatch):
    api = _api()
    runtime, outputs = _runtime()
    owner = _capture(api, runtime, outputs, bind=False)
    monkeypatch.setattr(
        api._D3ProducerOwner, "_mark_pre_authority_dynamic_producer_bound", object()
    )

    with pytest.raises(MdpStateError, match="binding hook is callable"):
        runtime._consume_pre_authority_dynamic_producer(owner, owner.producer)
    assert runtime._pre_authority_dynamic_producer is owner.producer
    owner.abort()


def test_capture_rejects_malformed_roles_without_registering():
    api = _api()
    runtime, outputs = _runtime()
    cases = (
        {"encoder_cp_follower": True},
        {"item_outputs": MappingProxyType({})},
        {"forward_only": True},
    )
    for override in cases:
        fresh_runtime, fresh_outputs = _runtime()
        fresh_runtime.rank_view = SimpleNamespace(global_rank=5, lane_id=0)
        kwargs = {
            "runtime": fresh_runtime,
            "rank_view": fresh_runtime.rank_view,
            "local_manifest": object(),
            "source_window": object(),
            "static_plan": object(),
            "item_outputs": fresh_outputs,
            "sample_location_by_id": MappingProxyType({GlobalSampleId(0, 0): (0, 0)}),
            "forward_only": False,
            "encoder_cp_follower": False,
        }
        kwargs.update(override)
        with pytest.raises((MdpConfigurationError, MdpStateError)):
            api._capture_d3_producer_owner(**kwargs)
        assert fresh_runtime._pre_authority_dynamic_producer is None

    assert runtime._pre_authority_dynamic_producer is None
    assert outputs


@pytest.mark.parametrize(
    "locations",
    (MappingProxyType({object(): (0, 0)}), MappingProxyType({GlobalSampleId(0, 0): (0, True)})),
)
def test_capture_rejects_noncanonical_sample_locations_before_ownership(locations):
    api = _api()
    runtime, outputs = _runtime()
    runtime.rank_view = SimpleNamespace(global_rank=5, lane_id=0)
    handle, payload = runtime._handle, runtime._chunk_payload_bases

    with pytest.raises(MdpStateError, match="canonical integer tuples"):
        api._capture_d3_producer_owner(
            runtime=runtime,
            rank_view=runtime.rank_view,
            local_manifest=object(),
            source_window=object(),
            static_plan=object(),
            item_outputs=outputs,
            sample_location_by_id=locations,
            forward_only=False,
        )

    assert runtime._pre_authority_dynamic_producer is None
    assert runtime._handle is handle
    assert runtime._chunk_payload_bases is payload
    assert runtime.allocator.acquired == runtime.allocator.released == []


def test_encoder_cp_follower_requires_retained_handle_and_exact_empty_geometry():
    api = _api()
    empty_runtime, empty_outputs = _runtime(contributor=False)
    with pytest.raises(MdpStateError, match="empty producer"):
        _capture(api, empty_runtime, empty_outputs, follower=True)

    follower_runtime, _ = _runtime(contributor=False, follower=True)
    with pytest.raises(MdpStateError, match="canonical local item order"):
        _capture(api, follower_runtime, MappingProxyType({}), follower=False)

    broken_runtime, _ = _runtime(contributor=False, follower=True)
    broken_runtime._chunk_of_item = dict(reversed(tuple(broken_runtime._chunk_of_item.items())))
    with pytest.raises(MdpStateError, match="exact canonical order"):
        _capture(api, broken_runtime, MappingProxyType({}), follower=True)
