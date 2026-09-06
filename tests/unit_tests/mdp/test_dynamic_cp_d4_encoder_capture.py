# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for the capture-only repeated-D4 encoder source owner."""

import gc
import os
import weakref
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
from megatron.core.mdp import dynamic_cp_d4_encoder_capture as capture_api
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.runtime import MdpRuntimeState
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams

_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8

if _WORLD8:
    from tests.unit_tests.test_utilities import Utils


class _Group:
    def __init__(self, ranks):
        self.ranks = tuple(ranks)


class _Adapter:
    spatial_merge_size = 2

    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def get_batch(self, iterator):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return next(iterator)

    def estimate_cost(self, grid_thw):
        return int(grid_thw[0])


class _Codec:
    def __init__(self, source_window, locations, error=None):
        self.source_window = source_window
        self.locations = locations
        self.error = error
        self.calls = []

    def build_source_window_with_locations(self, records, *, source_dp_lane):
        self.calls.append((records, source_dp_lane))
        if self.error is not None:
            raise self.error
        return self.source_window, self.locations


class _LocalBaseException(BaseException):
    pass


class _ComparisonBomb:
    def __init__(self):
        self.calls = 0

    def __eq__(self, _other):
        self.calls += 1
        raise AssertionError("comparison callback invoked")

    def __ne__(self, _other):
        self.calls += 1
        raise AssertionError("comparison callback invoked")


def _binding(rank):
    domain = tuple(range(0 if rank < 4 else 4, 4 if rank < 4 else 8))
    return _make_repeated_d4_group_binding(
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


def _runtime(rank, *, device=torch.device("cuda", 0)):
    domain = tuple(range(0 if rank < 4 else 4, 4 if rank < 4 else 8))
    runtime = object.__new__(capture_api.MdpRuntime)
    runtime._state = MdpRuntimeState.EMPTY
    runtime._window = None
    runtime._plan = None
    runtime.num_vpp_chunks = 1
    runtime.device = device
    runtime.rank_view = SimpleNamespace(
        global_rank=rank,
        outer_dp_rank=rank // 4,
        lane_id=rank // 4 if rank == domain[0] else None,
        my_worker_id=0,
        endpoint_rank=domain[0],
        planning_group_ranks=domain,
        worker_ids=(0,),
    )
    runtime.rank_map = SimpleNamespace(
        spec=SimpleNamespace(world_size=8, tp=1, pp=1, cp=4, encoder_cp=4)
    )
    runtime._d4_encoder_capture_owner = None
    runtime._d4_encoder_capture_trusted_owner = None
    runtime._retired_d4_encoder_capture_owners = {}
    return runtime


def _window(monkeypatch, *, pixels=None):
    pixels = {0: torch.ones(4, 4)} if pixels is None else pixels
    window = SimpleNamespace(
        records=lambda: ("record",),
        payload_sidecar=lambda: dict(pixels),
        release_pixels=lambda: pixels.clear(),
    )
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["adapter"].get_batch(args[0])
        assert kwargs["adapter"].estimate_cost((1,)) == 1
        assert kwargs["adapter"].spatial_merge_size == 2
        return window

    monkeypatch.setattr(capture_api.MdpIterationWindow, "capture", capture)
    return window, calls


def _source_window(lane):
    boundaries = torch.tensor((0, 1), dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries.clone(),
        cu_seqlens_q_padded=boundaries.clone(),
        cu_seqlens_kv_padded=boundaries.clone(),
        max_seqlen_q=1,
        max_seqlen_kv=1,
        total_tokens=1,
    )
    payload = MappingProxyType(
        {
            "input_ids": torch.zeros((1, 1), dtype=torch.int64),
            "labels": torch.zeros((1, 1), dtype=torch.int64),
            "loss_mask": torch.ones((1, 1)),
            "padding_mask": torch.zeros((1, 1), dtype=torch.bool),
            "position_ids": torch.zeros((1, 1), dtype=torch.int64),
            "attention_mask": None,
            "image_grid_thw": torch.tensor(((1, 2, 2),), dtype=torch.int64),
        }
    )
    source_window, locations = MultimodalDecoderPayloadCodec().build_source_window_with_locations(
        (
            MdpMicrobatchRecord(
                microbatch_id=0,
                text_only=False,
                vision_items=(
                    MdpMicrobatchVisionRecord(
                        global_item_id=0,
                        sample_id=0,
                        image_ordinal=0,
                        grid_thw=(1, 2, 2),
                        output_rows=1,
                        decoder_positions=(0,),
                    ),
                ),
                decoder_packed_seq_params=packed,
                model_payload=payload,
            ),
        ),
        source_dp_lane=lane,
    )
    return source_window, source_window.metadata_manifest(), locations


@pytest.mark.parametrize("rank", range(8))
def test_all_ranks_get_typed_owner_and_only_domain_source_captures(monkeypatch, rank):
    runtime = _runtime(rank)
    source_window, manifest, locations = _source_window(rank // 4)
    adapter = _Adapter()
    codec = _Codec(source_window, locations)
    operations = capture_api._snapshot_d4_encoder_capture_operations(adapter, codec)
    _, capture_calls = _window(monkeypatch)

    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(rank),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )

    assert type(owner) is capture_api._D4EncoderCaptureOwner
    assert owner.require() is owner
    assert (len(capture_calls), len(codec.calls)) == ((1, 1) if rank in (0, 4) else (0, 0))
    if rank in (0, 4):
        kwargs = capture_calls[0][1]
        assert kwargs["my_worker_id"] == 0
        assert kwargs["num_workers"] == 1
        assert kwargs["is_worker_leader"] is True
        assert kwargs["data_loader_source_worker_ids"] == (0,)
        assert kwargs["lane_id"] == rank // 4
        assert kwargs["capture_error_consensus"] is None
        assert owner.local_manifest == manifest
        assert owner.source_window is source_window
        assert owner.sample_locations is not locations
        assert owner.sample_locations == locations
        assert tuple(owner.sample_locations) == (GlobalSampleId(rank // 4, 0),)
        assert next(iter(owner.sample_locations)) is next(iter(locations))
        assert owner.sample_locations[GlobalSampleId(rank // 4, 0)] == (0, 0)
        assert source_window.items[0].item_id == GlobalVisionItemId(rank // 4, 0)
        assert set(owner.pixel_sidecar) == {0}
    else:
        assert owner.local_manifest is None
        assert owner.source_window is None
        assert dict(owner.pixel_sidecar) == {}
    owner.abort()
    assert runtime._d4_encoder_capture_owner is None


def test_snapshotted_operations_ignore_later_live_adapter_and_codec_mutation(monkeypatch):
    runtime = _runtime(0)
    source_window, manifest, locations = _source_window(0)
    adapter = _Adapter()
    codec = _Codec(source_window, locations)
    operations = capture_api._snapshot_d4_encoder_capture_operations(adapter, codec)
    adapter.get_batch = lambda *_args: (_ for _ in ()).throw(AssertionError("live adapter"))
    adapter.estimate_cost = lambda *_args: (_ for _ in ()).throw(AssertionError("live adapter"))
    adapter.spatial_merge_size = "mutated"
    codec.build_source_window_with_locations = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("live codec")
    )
    _window(monkeypatch)

    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )

    assert owner.local_manifest == manifest
    owner.abort()


def test_replaced_operation_snapshot_is_rejected_before_iterator(monkeypatch):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations)
    )
    forged = replace(
        operations,
        get_batch=lambda *_args: (_ for _ in ()).throw(AssertionError("forged callback")),
    )
    _, calls = _window(monkeypatch)

    with pytest.raises(MdpStateError, match="exact snapshotted operations"):
        capture_api._capture_d4_encoder_source(
            runtime=runtime,
            binding=_binding(0),
            data_iterators=iter((object(),)),
            num_microbatches=1,
            operations=forged,
        )

    assert calls == []


def test_mutated_exact_operation_snapshot_is_rejected_before_iterator(monkeypatch):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations)
    )
    object.__setattr__(
        operations,
        "get_batch",
        lambda *_args: (_ for _ in ()).throw(AssertionError("mutated callback")),
    )
    _, calls = _window(monkeypatch)

    with pytest.raises(MdpStateError, match="exact snapshotted operations"):
        capture_api._capture_d4_encoder_source(
            runtime=runtime,
            binding=_binding(0),
            data_iterators=iter((object(),)),
            num_microbatches=1,
            operations=operations,
        )

    assert calls == []


def test_mutated_width_bomb_is_rejected_without_comparison_or_iterator(monkeypatch):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations)
    )
    bomb = _ComparisonBomb()
    object.__setattr__(operations, "spatial_merge_size", bomb)
    _, calls = _window(monkeypatch)

    with pytest.raises(MdpStateError, match="exact snapshotted operations"):
        capture_api._capture_d4_encoder_source(
            runtime=runtime,
            binding=_binding(0),
            data_iterators=iter((object(),)),
            num_microbatches=1,
            operations=operations,
        )

    assert bomb.calls == 0
    assert calls == []


def test_capture_owner_cannot_be_minted_outside_private_factory():
    with pytest.raises(MdpConfigurationError, match="private factory"):
        capture_api._D4EncoderCaptureOwner(_runtime(0), _binding(0), _factory_seal=object())


@pytest.mark.parametrize("failure_site", ("capture", "codec"))
def test_source_failure_is_retained_for_future_consensus_and_fresh_retry(monkeypatch, failure_site):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    original = _LocalBaseException(f"{failure_site} failed")
    adapter = _Adapter(error=original if failure_site == "capture" else None)
    codec = _Codec(source_window, locations, error=original if failure_site == "codec" else None)
    operations = capture_api._snapshot_d4_encoder_capture_operations(adapter, codec)
    pixels = {0: torch.ones(4, 4)}
    _, capture_calls = _window(monkeypatch, pixels=pixels)
    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )

    assert type(owner.local_prepare_error) is MdpStateError
    assert owner.local_prepare_error.__cause__ is original
    assert runtime._window is None and runtime._plan is None
    owner.abort(owner.local_prepare_error)
    assert runtime._d4_encoder_capture_owner is None

    adapter.error = None
    codec.error = None
    fresh_operations = capture_api._snapshot_d4_encoder_capture_operations(adapter, codec)
    _window(monkeypatch, pixels={0: torch.ones(4, 4)})
    fresh = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=fresh_operations,
    )
    assert fresh.local_prepare_error is None
    fresh.abort()


def test_normal_exception_identity_is_retained(monkeypatch):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    original = RuntimeError("codec failed")
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations, error=original)
    )
    _window(monkeypatch)

    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )

    assert owner.local_prepare_error is original
    owner.abort(original)


def test_owner_mutation_cleans_only_trusted_pixels_and_rejects_replay(monkeypatch):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    original_pixels = {0: torch.ones(4, 4)}
    foreign_pixels = {9: torch.ones(3)}
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations)
    )
    _window(monkeypatch, pixels=original_pixels)
    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )
    trusted_pixels = owner._trusted_pixels
    object.__setattr__(owner, "_pixel_sidecar", foreign_pixels)

    with pytest.raises(MdpStateError, match="sealed capture fields"):
        owner.require()
    owner.abort()

    assert trusted_pixels == {}
    assert set(foreign_pixels) == {9}
    with pytest.raises(MdpStateError, match="retired"):
        owner.abort()


def test_hostile_state_substitution_cannot_interrupt_primary_cleanup(monkeypatch):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations)
    )
    _window(monkeypatch)
    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )
    trusted_pixels = owner._trusted_pixels
    bomb = _ComparisonBomb()
    primary = RuntimeError("WORLD primary")
    object.__setattr__(owner, "_state", bomb)

    owner.abort(primary)

    assert bomb.calls == 0
    assert trusted_pixels == {}
    assert runtime._d4_encoder_capture_owner is None
    assert any("integrity" in note for note in primary.__notes__)


def test_active_slot_substitution_still_cleans_original_owner_once(monkeypatch):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    foreign = object()
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations)
    )
    _window(monkeypatch)
    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )
    trusted_pixels = owner._trusted_pixels
    runtime._d4_encoder_capture_owner = foreign

    with pytest.raises(MdpStateError, match="trusted owner"):
        owner.require()
    owner.abort()

    assert trusted_pixels == {}
    assert runtime._d4_encoder_capture_owner is None
    assert runtime._d4_encoder_capture_trusted_owner is None


def test_abort_releases_last_trusted_pixel_reference(monkeypatch):
    runtime = _runtime(0)
    source_window, _, locations = _source_window(0)
    tensor = torch.ones(4, 4)
    reference = weakref.ref(tensor)
    pixels = {0: tensor}
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations)
    )
    _window(monkeypatch, pixels=pixels)
    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )
    del tensor

    owner.abort()
    gc.collect()

    assert reference() is None


def test_invalid_runtime_or_binding_fails_before_iterator(monkeypatch):
    runtime = _runtime(0)
    runtime.rank_view.global_rank = 1
    source_window, _, locations = _source_window(0)
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _Adapter(), _Codec(source_window, locations)
    )
    _, calls = _window(monkeypatch)

    with pytest.raises(MdpStateError, match="runtime rank"):
        capture_api._capture_d4_encoder_source(
            runtime=runtime,
            binding=_binding(0),
            data_iterators=iter((object(),)),
            num_microbatches=1,
            operations=operations,
        )

    assert calls == []


def test_real_window_capture_transfers_records_and_pixels_after_owner_registration(monkeypatch):
    runtime = _runtime(0)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor((0, 4), dtype=torch.int32),
        cu_seqlens_kv=torch.tensor((0, 4), dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor((0, 4), dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor((0, 4), dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=4,
    )
    pixels = torch.arange(16, dtype=torch.float32).view(4, 4)
    payload = MappingProxyType(
        {
            "input_ids": torch.arange(4).view(1, 4),
            "labels": torch.arange(4).view(1, 4),
            "loss_mask": torch.ones(1, 4),
            "padding_mask": torch.zeros(1, 4, dtype=torch.bool),
            "position_ids": torch.arange(4).view(1, 4),
            "attention_mask": None,
            "image_grid_thw": torch.tensor(((1, 2, 2),), dtype=torch.int64),
        }
    )
    captured = CapturedMicrobatch(
        decoder_packed_seq_params=packed,
        vision_items=(
            CapturedVisionItem(
                sample_id=0,
                image_ordinal=0,
                grid_thw=(1, 2, 2),
                payload_row_start=0,
                payload_rows=4,
                decoder_positions=(1,),
            ),
        ),
        flat_pixel_payload=pixels,
        model_payload=payload,
    )

    class _RealAdapter:
        spatial_merge_size = 2

        def get_batch(self, iterator):
            next(iterator)
            assert runtime._d4_encoder_capture_owner is not None
            assert runtime._d4_encoder_capture_owner is runtime._d4_encoder_capture_trusted_owner
            return captured

        @staticmethod
        def estimate_cost(item):
            return item.payload_rows

    released = []
    release_pixels = capture_api.MdpIterationWindow.release_pixels

    def observe_release(window):
        released.append(window.payload_sidecar())
        release_pixels(window)
        assert window.payload_sidecar() == {}

    monkeypatch.setattr(capture_api.MdpIterationWindow, "release_pixels", observe_release)
    operations = capture_api._snapshot_d4_encoder_capture_operations(
        _RealAdapter(), MultimodalDecoderPayloadCodec()
    )
    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=_binding(0),
        data_iterators=iter((0,)),
        num_microbatches=1,
        operations=operations,
    )

    assert owner.local_prepare_error is None
    assert len(owner.source_window.samples) == 1
    assert len(owner.source_window.items) == 1
    assert tuple(owner.sample_locations.values()) == ((0, 0),)
    assert tuple(owner.pixel_sidecar) == (0,)
    assert tuple(released[0]) == (0,)
    assert (
        owner.pixel_sidecar[0].untyped_storage().data_ptr() == pixels.untyped_storage().data_ptr()
    )
    owner.abort()


@pytest.fixture(scope="module")
def _world8_groups():
    if not _WORLD8:
        yield None
        return
    Utils.initialize_model_parallel()
    domains = (dist.new_group(ranks=(0, 1, 2, 3)), dist.new_group(ranks=(4, 5, 6, 7)))
    yield dist.group.WORLD, domains[dist.get_rank() // 4]
    for group in domains:
        dist.destroy_process_group(group)
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_source_failure_is_asymmetric_owner_state_and_all_ranks_retry(
    monkeypatch, _world8_groups
):
    world, domain = _world8_groups
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    binding = _make_repeated_d4_group_binding(
        world_group=world,
        domain_group=domain,
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=device,
        timeout_seconds=30.0,
    )
    runtime = _runtime(rank, device=device)
    source_window, _, locations = _source_window(rank // 4)
    error = _LocalBaseException("rank-0 capture failure") if rank == 0 else None
    adapter = _Adapter(error=error)
    codec = _Codec(source_window, locations)
    operations = capture_api._snapshot_d4_encoder_capture_operations(adapter, codec)
    _, calls = _window(monkeypatch)

    owner = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=binding,
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=operations,
    )
    local_error = owner.local_prepare_error
    observations = [None] * 8
    dist.all_gather_object(
        observations,
        (type(owner) is capture_api._D4EncoderCaptureOwner, local_error is not None, len(calls)),
    )
    assert observations == [
        (True, candidate == 0, 1 if candidate in (0, 4) else 0) for candidate in range(8)
    ]
    if rank == 0:
        assert local_error.__cause__ is error
    owner.abort(local_error)

    adapter.error = None
    fresh_operations = capture_api._snapshot_d4_encoder_capture_operations(adapter, codec)
    _window(monkeypatch, pixels={0: torch.ones(4, 4)})
    fresh = capture_api._capture_d4_encoder_source(
        runtime=runtime,
        binding=binding,
        data_iterators=iter((object(),)),
        num_microbatches=1,
        operations=fresh_operations,
    )
    assert fresh.local_prepare_error is None
    fresh.abort()
