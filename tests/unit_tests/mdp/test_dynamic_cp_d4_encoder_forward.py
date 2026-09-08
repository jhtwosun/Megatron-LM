# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gate0 selected-forward ownership and ordering contracts."""

import copy
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_encoder_execution as execution_api
from megatron.core.mdp import dynamic_cp_d4_encoder_forward as api
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.errors import (
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment
from megatron.core.mdp.protocols import DynamicEncoderCpBinding, VisionCaptureMode
from megatron.core.mdp.vision_locator import (
    VisionDataLocator,
    VisionLocatorCatalogEntry,
    VisionLocatorIndexSentinel,
    VisionLocatorKind,
    build_vision_locator_catalog,
)


class _Allocator:
    def __init__(self):
        self.acquired = []
        self.released = []
        self.fail_at = None
        self.fail_release = False

    def acquire(self, *, rows, width, dtype, device, tag):
        if self.fail_at == len(self.acquired):
            raise RuntimeError("allocator failed")
        tensor = torch.empty((rows,) if width == 0 else (rows, width), dtype=dtype, device=device)
        self.acquired.append((tag, tensor))
        return tensor

    def release(self, tensor):
        self.released.append(tensor)
        if self.fail_release:
            raise RuntimeError("release failed")


class _EncoderDdp:
    def __init__(self, events):
        self.module = object()
        self.events = events

    def zero_grad_buffer(self):
        self.events.append("zero")


class _Adapter:
    payload_width = 4

    def __init__(self, events):
        self.events = events
        self.restored = []
        self.encoded_with = []
        self.invalid_output = False
        self.fail_restore = False
        self.fail_encode = False

    def bind_dynamic_encoder_cp(self, encoder, *, membership, global_rank):
        assert encoder is self.encoder
        self.events.append(("bind", membership, global_rank))
        current = [True]

        def restore(primary):
            self.restored.append(primary)
            self.events.append("restore")
            if self.fail_restore:
                raise RuntimeError("restore failed")

        return DynamicEncoderCpBinding(membership, is_current=lambda: current[0], restore=restore)

    def encode(self, encoder, payload, layout):
        self.encoded_with.append(encoder)
        self.events.append(("encode", payload, layout))
        if self.fail_encode:
            raise RuntimeError("encode failed")
        if self.invalid_output:
            return object()
        leaf = (
            payload[: sum(segment.output_rows for segment in layout.segments), :2]
            .detach()
            .requires_grad_()
        )
        return leaf * 2

    def prepare_materialized_vision_payloads(self, locators, payloads):
        self.events.append(("prepare-locator-payloads", locators, payloads))
        return torch.arange(16.0).view(4, 4)


def _layout():
    return EncoderThdLayout(
        producer_worker_id=0,
        segments=(
            EncoderThdSegment(
                global_item_id=0,
                microbatch_id=3,
                sample_id=0,
                image_ordinal=0,
                payload_row_start=0,
                payload_rows=4,
                output_row_start=0,
                output_rows=1,
                grid_thw=(1, 2, 2),
            ),
        ),
    )


def _claim(*, rank=0, selected=True, leader=True, text_only=False, runtime=None):
    if runtime is None:
        events = []
        allocator = _Allocator()
        ddp = _EncoderDdp(events)
        adapter = _Adapter(events)
        adapter.encoder = ddp.module
        adapter.ddp = ddp
        runtime = SimpleNamespace(
            allocator=allocator,
            adapter=adapter,
            encoder_domain=SimpleNamespace(encoder_ddp=ddp),
            params_dtype=torch.float32,
            device=torch.device("cpu"),
            _iteration=0,
        )
    else:
        events = runtime.encoder_domain.encoder_ddp.events
    binding = SimpleNamespace(global_rank=rank, domain_group=object())
    membership = SimpleNamespace(group=object()) if selected else None
    layout = _layout() if leader else None
    authority = SimpleNamespace(
        source_rank_by_lane={0: 0},
        payload_ledger=SimpleNamespace(entries=()),
        plan=object(),
        global_manifest=SimpleNamespace(
            items=(
                ()
                if text_only
                else (
                    SimpleNamespace(
                        item_id=GlobalVisionItemId(0, 0),
                        image_ordinal=0,
                        grid_thw=(1, 2, 2),
                        output_rows=1,
                    ),
                )
            )
        ),
        participant_ranks=(0, 1, 2, 3),
        bridge_width=2,
        bridge_dtype=torch.float32,
    )
    selected_ranks = () if text_only else ((0, 1) if selected else (0, 1))
    token = object()
    values = (authority, binding, selected_ranks, membership, layout)
    execution_api._PENDING[token] = tuple(id(value) for value in values)
    claim = execution_api._D4EncoderExecutionClaim(
        authority=authority,
        binding=binding,
        selected_ranks=selected_ranks,
        membership=membership,
        layout=layout,
        is_selected=selected and not text_only,
        is_leader=leader and not text_only,
        text_only=text_only,
        _factory_seal=token,
    )
    pixels = {} if text_only or not leader else {0: torch.arange(16.0).view(4, 4)}
    claim._activate(
        (
            runtime,
            binding,
            object() if rank == 0 else None,
            object() if rank == 0 else None,
            MappingProxyType({}),
            pixels,
        )
    )
    return runtime, authority, claim, events


def _locator_claim(*, rank=0, selected=True, leader=True, text_only=False):
    runtime, authority, _source_claim, events = _claim(
        rank=rank, selected=selected, leader=leader, text_only=text_only
    )
    _source_claim.abort()
    entries = ()
    item_ids = ()
    if not text_only:
        item_id = authority.global_manifest.items[0].item_id
        locator = VisionDataLocator(
            VisionLocatorKind.SHARED_FILE,
            "/datasets/image.jpg",
            None,
            None,
            VisionLocatorIndexSentinel.UNUSED,
            (1, 2, 2),
        )
        item_ids = (item_id,)
        entries = (VisionLocatorCatalogEntry(item_id, locator),)
    catalog = build_vision_locator_catalog(item_ids, entries)
    binding = SimpleNamespace(global_rank=rank, domain_group=object())
    membership = SimpleNamespace(group=object()) if selected and not text_only else None
    layout = _layout() if leader and not text_only else None
    selected_ranks = () if text_only else (0, 1)
    token = object()
    values = (authority, binding, selected_ranks, membership, layout, catalog)
    execution_api._PENDING[token] = tuple(id(value) for value in values)
    claim = execution_api._D4EncoderExecutionClaim(
        authority=authority,
        binding=binding,
        selected_ranks=selected_ranks,
        membership=membership,
        layout=layout,
        is_selected=selected and not text_only,
        is_leader=leader and not text_only,
        text_only=text_only,
        capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        locator_catalog=catalog,
        _factory_seal=token,
    )
    claim._activate(
        (
            runtime,
            binding,
            object() if rank == 0 else None,
            object() if rank == 0 else None,
            MappingProxyType({}),
            {},
            VisionCaptureMode.STABLE_LOCATOR_CATALOG,
            catalog,
        )
    )
    return runtime, authority, claim, catalog, events


def _install_gate0(monkeypatch, events, *, reject=None, physical_error=None):
    bundle = object()

    def run(binding, authority, **kwargs):
        del binding, authority
        events.append(("gate", kwargs["gate_id"], kwargs["byte_generator"]))
        prepared = kwargs["prepare"]()
        events.append("world0")
        if reject is not None:
            raise reject
        events.extend(("domain-status", "world1"))
        if physical_error is not None:
            raise physical_error
        return kwargs["domain_collective"](prepared)

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(
        api,
        "_prepare_repeated_d4_decoder_payload",
        lambda *_args, **_kwargs: events.append("payload-prepare") or bundle,
    )
    monkeypatch.setattr(
        api,
        "_execute_repeated_d4_decoder_payload",
        lambda *_args, **_kwargs: events.append("payload-a2a") or bundle,
    )
    return bundle


def test_gate0_encodes_with_typed_encoder_beneath_ddp_and_float16_layers(monkeypatch):
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    class TypedEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    runtime, _authority, claim, _events = _claim()
    claim.abort()
    typed_encoder = TypedEncoder()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    runtime.encoder_domain.encoder_ddp.module = Float16Module(config, typed_encoder)
    runtime.adapter.encoder = typed_encoder

    runtime, authority, claim, events = _claim(runtime=runtime)
    _install_gate0(monkeypatch, events)
    owner = api.run_repeated_d4_encoder_forward(
        runtime, claim, authority, broadcast=lambda *_args, **_kwargs: None
    )

    assert runtime.adapter.encoded_with == [typed_encoder]
    owner.abort()


def _forward_owner(
    monkeypatch,
    *,
    runtime=None,
    invalid_output=False,
    text_only=False,
    selected=True,
    leader=True,
    rank=0,
):
    runtime, authority, claim, events = _claim(
        runtime=runtime, text_only=text_only, selected=selected, leader=leader, rank=rank
    )
    runtime.adapter.invalid_output = invalid_output
    _install_gate0(monkeypatch, events)
    owner = api.run_repeated_d4_encoder_forward(
        runtime, claim, authority, broadcast=lambda *_args, **_kwargs: None
    )
    return runtime, authority, owner, events


def _publication(monkeypatch, **kwargs):
    runtime, authority, owner, events = _forward_owner(monkeypatch, **kwargs)
    _install_gate1(monkeypatch, events)
    publication = api.run_repeated_d4_encoder_publication(owner)
    calls = []

    def snapshot(binding, candidate):
        calls.append((binding, candidate))
        if candidate.participant_ranks != (0, 1, 2, 3):
            raise MdpStateError("mutated authority")
        return candidate

    monkeypatch.setattr(api, "_snapshot_local_authority", snapshot)
    return runtime, authority, publication, calls


def _install_gate1(
    monkeypatch,
    events,
    *,
    reject_at=None,
    physical_error=None,
    substitute=None,
    converge_prepare_error=False,
):
    embedding = object()

    def run(binding, authority, **kwargs):
        del binding, authority
        events.append(("gate", kwargs["gate_id"], kwargs["byte_generator"]))
        try:
            prepared = kwargs["prepare"]()
        except Exception as local_error:
            if not converge_prepare_error:
                raise
            events.append(("first-world-error", local_error))
            raise MdpPlanError("Gate1 rejected the common plan") from local_error
        events.append("world0")
        if reject_at == "first":
            raise RuntimeError("first WORLD rejected")
        events.extend(("domain-status", "world1"))
        if reject_at == "final":
            raise RuntimeError("final WORLD rejected")
        if physical_error is not None:
            raise physical_error
        result = kwargs["domain_collective"](prepared)
        return substitute if substitute is not None else result

    def prepare(binding, authority, **kwargs):
        del binding, authority
        events.append("embedding-prepare")
        assert isinstance(kwargs["item_outputs"], MappingProxyType)
        return embedding

    def execute(binding, prepared, **kwargs):
        del binding, kwargs
        assert prepared is embedding
        events.append("embedding-a2a")
        return prepared

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(api, "_embedding_buffers", lambda owner: (object(), object()))
    monkeypatch.setattr(api, "_prepare_repeated_d4_embedding", prepare)
    monkeypatch.setattr(api, "_execute_repeated_d4_embedding", execute)
    return embedding


def test_gate0_begins_before_prepare_then_layout_pixels_and_encode(monkeypatch):
    runtime, authority, claim, events = _claim()
    bundle = _install_gate0(monkeypatch, events)
    generator = object()
    membership = claim.membership
    monkeypatch.setattr(
        api,
        "run_repeated_d4_decoder_payload",
        lambda *_args, **_kwargs: pytest.fail("nested Gate0 payload wrapper was called"),
        raising=False,
    )

    def broadcast(tensor, *, src, group):
        assert src == 0
        assert group is membership.group
        events.append("layout-broadcast" if tensor.dtype == torch.int64 else "pixel-broadcast")

    owner = api.run_repeated_d4_encoder_forward(
        runtime, claim, authority, broadcast=broadcast, byte_generator=generator
    )

    assert owner.require() is owner
    assert owner.payload_bundle is bundle
    assert owner.output.shape == (1, 2)
    assert tuple(owner.item_outputs) == (authority.global_manifest.items[0].item_id,)
    assert events[0] == ("gate", 0, generator)
    assert events[1:5] == ["zero", ("bind", membership, 0), "payload-prepare", "world0"]
    assert events[5:9] == ["domain-status", "world1", "payload-a2a", "layout-broadcast"]
    assert events[9] == "pixel-broadcast"
    assert events[10][0] == "encode"
    with pytest.raises(MdpStateError, match="retired"):
        claim.require()
    owner.abort()
    assert runtime.adapter.restored
    assert len(runtime.allocator.released) == len(runtime.allocator.acquired)


def test_locator_gate0_materializes_prepares_and_copies_before_world_then_skips_pixels(monkeypatch):
    runtime, authority, claim, catalog, events = _locator_claim()
    _install_gate0(monkeypatch, events)

    class Materialized:
        def _claim_for_gate0(self):
            events.append("claim-locator-bytes")
            return (b"encoded-image",)

    monkeypatch.setattr(
        api,
        "materialize_d4_selected_locator_catalog",
        lambda actual_runtime, actual_authority, actual_catalog: (
            events.append(("materialize", actual_runtime, actual_authority, actual_catalog))
            or Materialized()
        ),
        raising=False,
    )
    broadcasts = []

    owner = api.run_repeated_d4_encoder_forward(
        runtime,
        claim,
        authority,
        broadcast=lambda tensor, **_kwargs: broadcasts.append(tensor.dtype),
    )

    assert events[0] == ("gate", 0, None)
    materialize_index = next(
        i
        for i, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "materialize"
    )
    prepare_index = next(
        i
        for i, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "prepare-locator-payloads"
    )
    world_index = events.index("world0")
    bind_index = next(
        i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "bind"
    )
    assert materialize_index < prepare_index < world_index < bind_index
    assert events[materialize_index][1:] == (runtime, authority, catalog)
    assert broadcasts == [torch.int64]
    assert events.count("zero") == 1
    locator_buffers = tuple(
        tensor
        for tag, tensor in runtime.allocator.acquired
        if tag == "dynamic_cp_gate0_locator_pixels"
    )
    assert len(locator_buffers) == 1
    assert locator_buffers[0].shape == (4, 4)
    torch.testing.assert_close(locator_buffers[0], torch.arange(16.0).view(4, 4))
    encode_event = next(
        event for event in events if isinstance(event, tuple) and event[0] == "encode"
    )
    assert encode_event[1] is locator_buffers[0]
    owner.abort()


@pytest.mark.parametrize(("rank", "text_only"), ((3, False), (0, True)))
def test_locator_gate0_clears_all_rank_grads_before_world_without_nonselected_work(
    monkeypatch, rank, text_only
):
    runtime, authority, claim, _catalog, events = _locator_claim(
        rank=rank, selected=False, leader=False, text_only=text_only
    )
    _install_gate0(monkeypatch, events)

    class EmptyMaterialized:
        def _claim_for_gate0(self):
            return ()

    monkeypatch.setattr(
        api,
        "materialize_d4_selected_locator_catalog",
        lambda *_args: EmptyMaterialized(),
        raising=False,
    )

    owner = api.run_repeated_d4_encoder_forward(runtime, claim, authority)

    assert events.index("zero") < events.index("world0")
    assert events.count("zero") == 1
    assert not any(
        isinstance(event, tuple) and event[0] in ("bind", "encode", "prepare-locator-payloads")
        for event in events
    )
    assert not any(
        tag == "dynamic_cp_gate0_locator_pixels" for tag, _tensor in runtime.allocator.acquired
    )
    owner.abort()


def test_locator_gate0_h2d_allocation_error_converges_before_world(monkeypatch):
    runtime, authority, claim, _catalog, events = _locator_claim()

    class Materialized:
        def _claim_for_gate0(self):
            return (b"encoded-image",)

    monkeypatch.setattr(
        api, "materialize_d4_selected_locator_catalog", lambda *_args: Materialized(), raising=False
    )
    runtime.allocator.fail_at = 0

    def converge(_binding, _authority, **kwargs):
        with pytest.raises(RuntimeError, match="allocator failed"):
            kwargs["prepare"]()
        events.append("first-world-error")
        raise MdpPlanError("common Gate0 preparation error")

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", converge)

    with pytest.raises(MdpPlanError, match="common Gate0"):
        api.run_repeated_d4_encoder_forward(runtime, claim, authority)

    assert events[-1] == "first-world-error"
    assert "world0" not in events and "zero" not in events
    assert not any(isinstance(event, tuple) and event[0] in ("bind", "encode") for event in events)


def test_locator_gate0_local_read_error_converges_before_zero_bind_or_encode(monkeypatch):
    runtime, authority, claim, catalog, events = _locator_claim()
    original = OSError("missing image")
    monkeypatch.setattr(
        api,
        "materialize_d4_selected_locator_catalog",
        lambda *_args: (_ for _ in ()).throw(original),
        raising=False,
    )

    def converge(_binding, _authority, **kwargs):
        try:
            kwargs["prepare"]()
        except OSError as error:
            assert error is original
            events.append("first-world-error")
            raise MdpPlanError("common Gate0 preparation error") from error
        pytest.fail("locator read unexpectedly succeeded")

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", converge)

    with pytest.raises(MdpPlanError, match="common Gate0"):
        api.run_repeated_d4_encoder_forward(runtime, claim, authority)

    assert "first-world-error" in events
    assert not any(
        event == "zero" or (isinstance(event, tuple) and event[0] in ("bind", "encode"))
        for event in events
    )
    assert runtime.allocator.acquired == []
    with pytest.raises(MdpStateError, match="retired"):
        claim.require()


def test_selected_follower_receives_source_layout_without_fabricating_locations(monkeypatch):
    runtime, authority, claim, events = _claim(rank=1, selected=True, leader=False)
    _install_gate0(monkeypatch, events)
    expected = api._layout_values(_layout())

    def broadcast(tensor, *, src, group):
        del group
        assert src == 0
        if tensor.dtype == torch.int64:
            tensor.copy_(torch.tensor(expected, dtype=torch.int64))
        else:
            tensor.fill_(3)

    owner = api.run_repeated_d4_encoder_forward(runtime, claim, authority, broadcast=broadcast)

    assert owner.layout == _layout()
    assert owner.output is not None
    assert owner.item_outputs == {}
    owner.abort()


def test_peer_rejection_restores_and_releases_without_physical_work(monkeypatch):
    runtime, authority, claim, events = _claim()
    primary = MdpStateError("peer rejected")
    _install_gate0(monkeypatch, events, reject=primary)

    with pytest.raises(MdpStateError) as raised:
        api.run_repeated_d4_encoder_forward(
            runtime, claim, authority, broadcast=lambda *_a, **_k: None
        )

    assert raised.value is primary
    assert "payload-a2a" not in events
    assert runtime.adapter.restored == [primary]
    assert len(runtime.allocator.released) == len(runtime.allocator.acquired)


@pytest.mark.parametrize("stage", ("first-world", "final-world"))
def test_each_world_rejection_cleans_without_physical_work(monkeypatch, stage):
    runtime, authority, claim, events = _claim()
    primary = MdpStateError(stage)
    monkeypatch.setattr(api, "_prepare_repeated_d4_decoder_payload", lambda *_a, **_k: object())

    def run(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        events.append("first-world")
        if stage == "first-world":
            raise primary
        events.append("domain-status")
        events.append("final-world")
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)

    with pytest.raises(MdpStateError) as raised:
        api.run_repeated_d4_encoder_forward(runtime, claim, authority)
    assert raised.value is primary
    assert "payload-a2a" not in events
    assert runtime.adapter.restored == [primary]


def test_prepare_reentry_rejects_and_cleans_once(monkeypatch):
    runtime, authority, claim, events = _claim()

    def run(_binding, _authority, **kwargs):
        first = kwargs["prepare"]()
        assert first is not None
        return kwargs["prepare"]()

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(api, "_prepare_repeated_d4_decoder_payload", lambda *_a, **_k: object())

    with pytest.raises(MdpStateError, match="one-shot"):
        api.run_repeated_d4_encoder_forward(runtime, claim, authority)
    assert len(runtime.adapter.restored) == 1
    assert len(runtime.allocator.released) == len(runtime.allocator.acquired)


def test_text_only_zeros_grad_but_skips_binding_pixels_and_encode(monkeypatch):
    runtime, authority, claim, events = _claim(selected=False, leader=False, text_only=True)
    _install_gate0(monkeypatch, events)

    owner = api.run_repeated_d4_encoder_forward(runtime, claim, authority)

    assert events == [
        ("gate", 0, None),
        "zero",
        "payload-prepare",
        "world0",
        "domain-status",
        "world1",
        "payload-a2a",
    ]
    assert owner.output is None
    assert owner.item_outputs == {}
    owner.abort()


def test_physical_payload_failure_is_taskfatal_and_cleans(monkeypatch):
    runtime, authority, claim, events = _claim()

    def run(_binding, _authority, **kwargs):
        return kwargs["domain_collective"](kwargs["prepare"]())

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(api, "_prepare_repeated_d4_decoder_payload", lambda *_a, **_k: object())
    monkeypatch.setattr(
        api,
        "_execute_repeated_d4_decoder_payload",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("a2a failed")),
    )

    with pytest.raises(MdpTaskFatalError, match="physical execution failed") as raised:
        api.run_repeated_d4_encoder_forward(runtime, claim, authority)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert runtime.adapter.restored == [raised.value]
    assert len(runtime.allocator.released) == len(runtime.allocator.acquired)


def test_partial_payload_buffer_acquire_failure_releases_first_buffer(monkeypatch):
    runtime, authority, claim, events = _claim()
    authority.payload_ledger.entries = (SimpleNamespace(dtype=torch.float32),)
    runtime.allocator.fail_at = 1
    monkeypatch.setattr(api, "decoder_payload_split_sizes", lambda *_a, **_k: ((1,), (1,)))

    def run(_binding, _authority, **kwargs):
        return kwargs["prepare"]()

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)

    with pytest.raises(RuntimeError, match="allocator failed"):
        api.run_repeated_d4_encoder_forward(runtime, claim, authority)
    assert len(runtime.allocator.acquired) == 1
    assert runtime.allocator.released == [runtime.allocator.acquired[0][1]]
    with pytest.raises(MdpStateError, match="retired"):
        claim.require()


def test_physical_callback_reentry_is_taskfatal_without_second_collective(monkeypatch):
    runtime, authority, claim, events = _claim()
    monkeypatch.setattr(api, "_prepare_repeated_d4_decoder_payload", lambda *_a, **_k: object())
    monkeypatch.setattr(
        api, "_execute_repeated_d4_decoder_payload", lambda *_a, **_k: events.append("payload-a2a")
    )

    def run(_binding, _authority, **kwargs):
        prepared = kwargs["prepare"]()
        kwargs["domain_collective"](prepared)
        return kwargs["domain_collective"](prepared)

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)

    with pytest.raises(MdpTaskFatalError, match="physical callback is one-shot"):
        api.run_repeated_d4_encoder_forward(
            runtime, claim, authority, broadcast=lambda *_a, **_k: None
        )
    assert events.count("payload-a2a") == 1
    assert len(runtime.adapter.restored) == 1


def test_runner_result_substitution_is_taskfatal_and_retires_resources(monkeypatch):
    runtime, authority, claim, events = _claim()

    def run(_binding, _authority, **kwargs):
        kwargs["prepare"]()
        return object()

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(api, "_prepare_repeated_d4_decoder_payload", lambda *_a, **_k: object())

    with pytest.raises(MdpTaskFatalError, match="exact owner"):
        api.run_repeated_d4_encoder_forward(runtime, claim, authority)
    assert len(runtime.adapter.restored) == 1
    assert len(runtime.allocator.released) == len(runtime.allocator.acquired)


def test_live_runtime_ddp_allocator_and_adapter_substitution_cannot_redirect(monkeypatch):
    runtime, authority, claim, events = _claim()
    original = runtime.adapter
    _install_gate0(monkeypatch, events)

    def run(_binding, _authority, **kwargs):
        prepared = kwargs["prepare"]()
        runtime.allocator = object()
        runtime.encoder_domain = object()
        runtime.adapter = SimpleNamespace(
            encode=lambda *_a, **_k: pytest.fail("substituted adapter encode called")
        )
        return kwargs["domain_collective"](prepared)

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)

    owner = api.run_repeated_d4_encoder_forward(
        runtime, claim, authority, broadcast=lambda *_a, **_k: None
    )
    assert owner.output is not None
    assert any(event[0] == "encode" for event in events if type(event) is tuple)
    owner.abort()
    assert original.restored


def test_malformed_runtime_is_captured_by_first_world_without_consuming_claim(monkeypatch):
    runtime, authority, claim, events = _claim()
    runtime.encoder_domain = object()

    def run(_binding, _authority, **kwargs):
        try:
            kwargs["prepare"]()
        except BaseException as error:
            events.append(("first-world", error))
            raise MdpStateError("first WORLD rejected") from error
        pytest.fail("malformed preparation unexpectedly succeeded")

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)

    with pytest.raises(MdpStateError, match="first WORLD rejected"):
        api.run_repeated_d4_encoder_forward(runtime, claim, authority)
    assert isinstance(events[-1][1], MdpConfigurationError)
    assert claim.require() is claim
    claim.abort()


@pytest.mark.parametrize("failure", ("layout", "pixels", "encode"))
def test_layout_pixel_and_encode_failures_are_taskfatal(monkeypatch, failure):
    runtime, authority, claim, events = _claim()
    _install_gate0(monkeypatch, events)
    calls = 0

    def broadcast(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if (failure == "layout" and calls == 1) or (failure == "pixels" and calls == 2):
            raise RuntimeError(f"{failure} failed")

    runtime.adapter.fail_encode = failure == "encode"
    with pytest.raises(MdpTaskFatalError, match="physical execution failed") as raised:
        api.run_repeated_d4_encoder_forward(runtime, claim, authority, broadcast=broadcast)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert runtime.adapter.restored == [raised.value]


def test_physical_cleanup_failures_note_exact_taskfatal_primary(monkeypatch):
    runtime, authority, claim, events = _claim()
    runtime.adapter.fail_restore = True
    runtime.allocator.fail_release = True
    _install_gate0(monkeypatch, events)
    runtime.adapter.fail_encode = True

    with pytest.raises(MdpTaskFatalError) as raised:
        api.run_repeated_d4_encoder_forward(
            runtime, claim, authority, broadcast=lambda *_a, **_k: None
        )

    assert runtime.adapter.restored == [raised.value]
    assert any("restore error" in note for note in raised.value.__notes__)
    assert any("buffer release error" in note for note in raised.value.__notes__)


def test_invalid_post_encode_result_is_retained_for_gate1(monkeypatch):
    runtime, authority, claim, events = _claim()
    _install_gate0(monkeypatch, events)
    runtime.adapter.invalid_output = True

    owner = api.run_repeated_d4_encoder_forward(
        runtime, claim, authority, broadcast=lambda *_a, **_k: None
    )

    assert isinstance(owner.local_forward_error, MdpStateError)
    assert owner.item_outputs == {}
    owner.abort(owner.local_forward_error)


def test_abort_cleanup_notes_primary_then_replay_and_same_runtime_retry(monkeypatch):
    runtime, authority, claim, events = _claim()
    _install_gate0(monkeypatch, events)
    owner = api.run_repeated_d4_encoder_forward(
        runtime, claim, authority, broadcast=lambda *_a, **_k: None
    )
    primary = MdpStateError("caller abort")
    runtime.adapter.fail_restore = True
    runtime.allocator.fail_release = True

    owner.abort(primary)

    assert runtime.adapter.restored == [primary]
    assert any("restore error" in note for note in primary.__notes__)
    assert any("buffer release error" in note for note in primary.__notes__)
    with pytest.raises(MdpStateError, match="retired"):
        owner.abort(primary)

    runtime.adapter.fail_restore = False
    runtime.allocator.fail_release = False
    _, fresh_authority, fresh_claim, fresh_events = _claim(runtime=runtime)
    _install_gate0(monkeypatch, fresh_events)
    fresh = api.run_repeated_d4_encoder_forward(
        runtime, fresh_claim, fresh_authority, broadcast=lambda *_a, **_k: None
    )
    assert fresh.require() is fresh
    fresh.abort()


def test_gate1_publishes_once_then_transfers_exact_graph_owner(monkeypatch):
    runtime, authority, owner, events = _forward_owner(monkeypatch)
    handle = owner.forward_handle
    assert handle.consumed is False
    generator = object()
    embedding = _install_gate1(monkeypatch, events)
    publication = api.run_repeated_d4_encoder_publication(owner, byte_generator=generator)

    assert publication.require() is publication
    assert publication.authority is authority
    assert publication.embedding_bundle is embedding
    assert publication.output is not None
    assert publication.forward_handle is handle
    assert handle.consumed is False
    assert events[-6:] == [
        ("gate", 1, generator),
        "embedding-prepare",
        "world0",
        "domain-status",
        "world1",
        "embedding-a2a",
    ]
    with pytest.raises(MdpStateError, match="forward owner is retired"):
        owner.require()
    assert owner.forward_handle is None
    publication.abort()
    assert handle.consumed is True
    assert len(runtime.adapter.restored) == 1


def test_gate1_text_only_transfers_typed_empty_publication(monkeypatch):
    runtime, authority, owner, events = _forward_owner(monkeypatch, text_only=True)
    embedding = _install_gate1(monkeypatch, events)

    publication = api.run_repeated_d4_encoder_publication(owner)

    assert publication.authority is authority
    assert publication.embedding_bundle is embedding
    assert publication.text_only is True
    assert publication.output is None
    assert publication.forward_handle is None
    assert publication.item_outputs == {}
    publication.abort()
    assert runtime.adapter.restored == []


def test_gate1_begins_before_retained_forward_error_and_skips_a2a(monkeypatch):
    runtime, _, owner, events = _forward_owner(monkeypatch, invalid_output=True)
    retained = owner.local_forward_error
    _install_gate1(monkeypatch, events, converge_prepare_error=True)

    with pytest.raises(MdpPlanError, match="common plan") as raised:
        api.run_repeated_d4_encoder_publication(owner)

    gate1_events = events[events.index(("gate", 1, None)) :]
    assert raised.value.__cause__ is retained
    assert gate1_events[0:2] == [("gate", 1, None), ("first-world-error", retained)]
    assert "embedding-prepare" not in gate1_events
    assert "embedding-a2a" not in gate1_events
    assert "domain-status" not in gate1_events
    assert "world1" not in gate1_events
    assert runtime.adapter.restored == [raised.value]


@pytest.mark.parametrize("boundary", ("first", "final"))
def test_gate1_each_world_rejection_cleans_without_embedding_a2a(monkeypatch, boundary):
    runtime, _, owner, events = _forward_owner(monkeypatch)
    _install_gate1(monkeypatch, events, reject_at=boundary)

    with pytest.raises(RuntimeError, match=f"{boundary} WORLD rejected") as raised:
        api.run_repeated_d4_encoder_publication(owner)

    assert "embedding-prepare" in events
    assert "embedding-a2a" not in events
    assert runtime.adapter.restored == [raised.value]
    assert len(runtime.allocator.released) == len(runtime.allocator.acquired)


def test_gate1_prepare_reentry_and_result_substitution_retire_owner(monkeypatch):
    runtime, _, owner, events = _forward_owner(monkeypatch)
    _install_gate1(monkeypatch, events)
    with pytest.raises(AttributeError):
        object.__setattr__(owner, "_publication_started", False)

    def reenter(binding, authority, **kwargs):
        del binding, authority
        kwargs["prepare"]()
        kwargs["prepare"]()

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", reenter)
    with pytest.raises(MdpStateError, match="preparation is one-shot"):
        api.run_repeated_d4_encoder_publication(owner)
    assert len(runtime.adapter.restored) == 1

    runtime, _, owner, events = _forward_owner(monkeypatch, runtime=runtime)
    _install_gate1(monkeypatch, events, substitute=object())
    with pytest.raises(MdpTaskFatalError, match="runner returned its owner"):
        api.run_repeated_d4_encoder_publication(owner)
    assert len(runtime.adapter.restored) == 2


def test_gate1_physical_reentry_cannot_reset_invocation_state(monkeypatch):
    runtime, _, owner, events = _forward_owner(monkeypatch)
    callback = {}
    calls = []

    def run(_binding, _authority, **kwargs):
        prepared = kwargs["prepare"]()
        callback["physical"] = kwargs["domain_collective"]
        return callback["physical"](prepared)

    def execute(*_args, **_kwargs):
        calls.append("a2a")
        with pytest.raises(AttributeError):
            object.__setattr__(owner, "_publication_physical_started", False)
        callback["physical"](owner)

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(api, "_embedding_buffers", lambda _owner: (object(), object()))
    monkeypatch.setattr(api, "_prepare_repeated_d4_embedding", lambda *_a, **_k: object())
    monkeypatch.setattr(api, "_execute_repeated_d4_embedding", execute)

    with pytest.raises(MdpTaskFatalError, match="physical callback is one-shot"):
        api.run_repeated_d4_encoder_publication(owner)
    assert calls == ["a2a"]
    assert len(runtime.adapter.restored) == 1


def test_gate1_physical_failure_is_taskfatal_and_cleanup_notes_exact_primary(monkeypatch):
    runtime, _, owner, events = _forward_owner(monkeypatch)
    runtime.adapter.fail_restore = True
    runtime.allocator.fail_release = True
    _install_gate1(monkeypatch, events)

    def fail_execute(*_args, **_kwargs):
        raise RuntimeError("embedding A2A failed")

    monkeypatch.setattr(api, "_execute_repeated_d4_embedding", fail_execute)
    with pytest.raises(MdpTaskFatalError, match="physical execution failed") as raised:
        api.run_repeated_d4_encoder_publication(owner)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert runtime.adapter.restored == [raised.value]
    notes = getattr(raised.value, "__notes__", ())
    assert any("restore error" in note for note in notes)
    assert any("buffer release error" in note for note in notes)


def test_gate1_rejection_uses_forward_handle_graph_release_and_notes_failure(monkeypatch):
    runtime, _, owner, events = _forward_owner(monkeypatch)
    handle = owner.forward_handle
    original = handle.release_forward_only

    def release_then_fail():
        original()
        raise RuntimeError("graph release failed")

    handle.release_forward_only = release_then_fail
    _install_gate1(monkeypatch, events, reject_at="final")
    with pytest.raises(RuntimeError, match="final WORLD rejected") as raised:
        api.run_repeated_d4_encoder_publication(owner)

    assert handle.consumed
    assert runtime.adapter.restored == [raised.value]
    assert any("graph release error" in note for note in raised.value.__notes__)


def test_gate1_rejection_allows_fresh_same_runtime_forward(monkeypatch):
    runtime, _, owner, events = _forward_owner(monkeypatch)
    _install_gate1(monkeypatch, events, reject_at="final")
    with pytest.raises(RuntimeError) as raised:
        api.run_repeated_d4_encoder_publication(owner)
    assert runtime.adapter.restored == [raised.value]

    runtime.adapter.fail_restore = False
    runtime.allocator.fail_release = False
    fresh_runtime, _, fresh, _ = _forward_owner(monkeypatch, runtime=runtime)
    assert fresh_runtime is runtime
    fresh.abort()


def test_publication_claims_registered_replay_handoff_without_releasing(monkeypatch):
    runtime, authority, publication, calls = _publication(monkeypatch)
    handle = publication.forward_handle
    binding_owner = publication._binding_owner
    buffers = publication._buffers

    handoff = publication._claim_for_replay(authority)

    assert type(handoff) is api._D4EncoderReplayHandoff
    assert handoff.require() is handoff
    assert handoff.authority is authority
    assert handoff.forward_handle is handle and not handle.consumed
    assert handoff._binding_owner is binding_owner and binding_owner.active
    assert handoff._buffers is buffers
    assert runtime.allocator.released == []
    assert calls == [(handoff.binding, authority)]
    assert api._ACTIVE_RUNTIME_OWNERS[id(runtime)][1]() is handoff
    with pytest.raises(MdpStateError, match="publication owner is retired"):
        publication.require()
    with pytest.raises(MdpStateError, match="publication owner is retired"):
        publication.abort()
    with pytest.raises(MdpStateError, match="publication owner is retired"):
        publication._claim_for_replay(authority)
    assert publication.authority is None
    assert publication.forward_handle is None

    assert handoff.consume(authority) is handoff
    assert handoff.require() is handoff
    with pytest.raises(MdpStateError, match="consumed exactly once"):
        handoff.consume(authority)
    handoff.abort()
    assert handle.consumed
    assert binding_owner.active is False
    assert runtime.allocator.released == [tensor for _, tensor in runtime.allocator.acquired]
    with pytest.raises(MdpStateError, match="replay handoff is retired"):
        handoff.require()


def test_foreign_clone_and_mutated_authority_do_not_consume_publication(monkeypatch):
    _, authority, publication, calls = _publication(monkeypatch)
    clone = copy.copy(authority)

    for candidate in (object(), clone):
        with pytest.raises(MdpStateError, match="exact iteration authority"):
            publication._claim_for_replay(candidate)
        assert publication.require() is publication
    assert calls == []

    original = authority.participant_ranks
    authority.participant_ranks = (0, 1)
    with pytest.raises(MdpStateError, match="mutated authority"):
        publication._claim_for_replay(authority)
    assert publication.require() is publication
    authority.participant_ranks = original
    handoff = publication._claim_for_replay(authority)
    handoff.abort()


@pytest.mark.parametrize(
    ("rank", "selected", "leader", "text_only", "has_graph", "has_binding"),
    (
        (0, True, True, False, True, True),
        (3, False, False, False, False, False),
        (0, True, True, True, False, False),
    ),
)
def test_replay_handoff_preserves_selected_nonselected_and_text_only_resources(
    monkeypatch, rank, selected, leader, text_only, has_graph, has_binding
):
    _, authority, publication, _ = _publication(
        monkeypatch, rank=rank, selected=selected, leader=leader, text_only=text_only
    )
    payload = publication.payload_bundle
    embedding = publication.embedding_bundle
    layout = publication.layout
    handoff = publication._claim_for_replay(authority)

    assert handoff.payload_bundle is payload
    assert handoff.embedding_bundle is embedding
    assert handoff.layout is layout
    assert (handoff.forward_handle is not None) is has_graph
    assert (handoff._binding_owner is not None) is has_binding
    assert handoff.text_only is text_only
    handoff.abort()


def test_replay_abort_cleans_trusted_resources_after_live_mutation_and_reuses_runtime(monkeypatch):
    runtime, authority, publication, _ = _publication(monkeypatch)
    handoff = publication._claim_for_replay(authority)
    handle = handoff.forward_handle
    binding_owner = handoff._binding_owner
    original_restore = binding_owner._restore
    callback_states = []

    def restore(primary):
        callback_states.append(
            (handoff.authority, api._ACTIVE_RUNTIME_OWNERS.get(id(runtime)), primary)
        )
        original_restore(primary)
        raise RuntimeError("restore failed")

    binding_owner._restore = restore
    runtime.allocator.fail_release = True
    note_states = []

    class HostilePrimary(MdpStateError):
        def add_note(self, note):
            del note
            try:
                handoff.abort(self)
            except BaseException as error:
                reentry = error
            else:
                reentry = None
            note_states.append(
                (handoff.authority, api._ACTIVE_RUNTIME_OWNERS.get(id(runtime)), reentry)
            )
            raise RuntimeError("hostile add_note")

    primary = HostilePrimary("replay failed")
    del handoff.authority

    handoff.abort(primary)

    assert callback_states == [(None, None, primary)]
    assert handle.consumed is True
    assert binding_owner.active is False
    assert note_states
    assert all(authority is None and owner is None for authority, owner, _ in note_states)
    assert all(
        isinstance(reentry, MdpStateError) and "retired" in str(reentry)
        for _, _, reentry in note_states
    )
    with pytest.raises(MdpStateError, match="replay handoff is retired"):
        handoff.abort(primary)

    runtime.adapter.fail_restore = False
    runtime.allocator.fail_release = False
    _, fresh_authority, fresh_publication, _ = _publication(monkeypatch, runtime=runtime)
    fresh = fresh_publication._claim_for_replay(fresh_authority)
    assert fresh.require() is fresh
    fresh.abort()
