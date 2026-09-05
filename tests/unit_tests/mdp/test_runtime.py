# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime phase-machine tests with a stub adapter and tiny encoder.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_runtime.py
"""

import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import BridgePhase, ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import (
    build_encoder_domain,
    build_encoder_pg_collection,
    finalize_encoder_grads,
)
from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import pixel_capture_suppressed
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.transformer_config import TransformerConfig

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1
pytestmark = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


WIDTH = 8  # payload width == hidden size, identity encoder
MERGE = 2
GRIDS = ((1, 4, 4), (1, 8, 8), (2, 4, 4))  # 16/64/32 payload; 4/16/8 output rows


def _sentinel(lane, item_index):
    return float(10 * (lane + 1) + item_index)


class _TinyEncoder(torch.nn.Module):
    def __init__(self, config, cp_group, *, divergent_init=False):
        super().__init__()
        self.config = config
        self.cp_group = cp_group
        self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            if divergent_init:
                self.proj.weight.fill_(torch.distributed.get_rank() + 1)
            else:
                self.proj.weight.copy_(torch.eye(WIDTH))

    def forward(self, x):
        if self.cp_group.size() == 1:
            return self.proj(x)
        cp_size = self.cp_group.size()
        cp_rank = self.cp_group.rank()
        rows_per_rank, remainder = divmod(x.size(0), cp_size)
        split_sizes = [rows_per_rank + (rank < remainder) for rank in range(cp_size)]
        start = sum(split_sizes[:cp_rank])
        local_output = self.proj(x.narrow(0, start, split_sizes[cp_rank]))
        return gather_from_sequence_parallel_region(
            local_output,
            tensor_parallel_output_grad=True,
            group=self.cp_group,
            output_split_sizes=split_sizes if len(set(split_sizes)) > 1 else None,
        )


class _StubAdapter:
    """Deterministic capture: two microbatches, three items in mb0, mb1 text-only."""

    payload_width = WIDTH
    spatial_merge_size = MERGE

    def __init__(self, lane, *, divergent_encoder_init=False):
        self._lane = lane if lane is not None else 0
        self._divergent_encoder_init = divergent_encoder_init
        self.materialized_count = 0
        self.input_grad_events = []
        self.output_grad_events = []

    def get_batch(self, iterator):
        mb = next(iterator)
        if mb != 0:
            return CapturedMicrobatch(
                decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
                vision_items=(),
                flat_pixel_payload=None,
                model_payload=MappingProxyType({"microbatch": mb}),
            )
        items = []
        payload_chunks = []
        payload_start = 0
        for index, grid in enumerate(GRIDS):
            t, h, w = grid
            rows = t * h * w
            output_rows = t * (h // MERGE) * (w // MERGE)
            items.append(
                CapturedVisionItem(
                    sample_id=index,
                    image_ordinal=0,
                    grid_thw=grid,
                    payload_row_start=payload_start,
                    payload_rows=rows,
                    decoder_positions=tuple(range(output_rows)),
                )
            )
            if not pixel_capture_suppressed():
                payload_chunks.append(
                    torch.full((rows, WIDTH), _sentinel(self._lane, index), device="cuda")
                )
            payload_start += rows
        if payload_chunks:
            self.materialized_count += 1
        return CapturedMicrobatch(
            decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
            vision_items=tuple(items),
            flat_pixel_payload=torch.cat(payload_chunks) if payload_chunks else None,
            model_payload=MappingProxyType({"microbatch": mb}),
        )

    def estimate_cost(self, item):
        return item.payload_rows

    def build_encoder(self, model_config, *, pg_collection):
        return _TinyEncoder(
            model_config,
            pg_collection.cp,
            divergent_init=self._divergent_encoder_init,
        )

    def encode(self, encoder, payload, layout):
        pieces = []
        for segment in layout.segments:
            piece = payload[
                segment.payload_row_start : segment.payload_row_start
                + segment.output_rows
            ]
            if torch.is_grad_enabled():
                piece = piece.detach().requires_grad_(True)

                def _record_input_grad(grad, item_id=segment.global_item_id):
                    self.input_grad_events.append((item_id, grad.detach().clone()))
                    return grad

                piece.register_hook(_record_input_grad)
            pieces.append(encoder(piece))
        output = torch.cat(pieces) if pieces else payload[:0]
        if output.requires_grad:

            def _record_output_grad(grad):
                self.output_grad_events.append(grad.detach().clone())
                return grad

            output.register_hook(_record_output_grad)
        return output


class _TrackingAllocator(DirectBufferAllocator):
    """Track exact releases and inject one rank-local allocation failure."""

    def __init__(self):
        super().__init__()
        self._tags = {}
        self._release_counts = {}
        self._failure = None
        self._release_failure_tag = None

    @staticmethod
    def _storage_key(buffer):
        # Leaves are stored as a valid-row view of the acquired capacity base.
        # Storage identity is stable across that view and remains unique for
        # zero-row bridge buffers, unlike data_ptr().
        return buffer.untyped_storage()._cdata

    def fail_once(self, *, rank, tag):
        self._failure = (rank, tag)

    def fail_release_once(self, *, tag):
        self._release_failure_tag = tag

    def acquire(self, *, rows, width, dtype, device, tag):
        if self._failure == (torch.distributed.get_rank(), tag):
            self._failure = None
            raise RuntimeError(f"injected rank-local {tag} failure")
        buffer = super().acquire(
            rows=rows, width=width, dtype=dtype, device=device, tag=tag
        )
        key = self._storage_key(buffer)
        self._tags[key] = tag
        self._release_counts[key] = 0
        return buffer

    def release(self, buffer):
        key = self._storage_key(buffer)
        self._release_counts[key] += 1
        super().release(buffer)
        if self._release_failure_tag == self._tags[key]:
            self._release_failure_tag = None
            raise RuntimeError(f"injected {self._tags[key]} cleanup failure")

    def all_released_once(self):
        return all(count == 1 for count in self._release_counts.values())


def _build_runtime(
    *,
    decoder_cp=1,
    decoder_pp=2,
    encoder_cp=1,
    allocator=None,
    divergent_encoder_init=False,
):
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(
            world_size=world,
            tp=1,
            pp=decoder_pp,
            cp=decoder_cp,
            ep=1,
            encoder_cp=encoder_cp,
        )
    )
    view = rank_map.view(rank)
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=encoder_cp, process_groups=groups
    )
    adapter = _StubAdapter(
        view.outer_dp_rank, divergent_encoder_init=divergent_encoder_init
    )
    model_config = TransformerConfig(
        num_layers=1,
        hidden_size=WIDTH,
        num_attention_heads=1,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    domain = build_encoder_domain(
        adapter=adapter,
        model_config=model_config,
        mdp_config=MdpConfig(enable=True, encoder_cp=encoder_cp),
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1e-3, use_distributed_optimizer=True, clip_grad=1.0
        ),
        encoder_pgs=encoder_pgs,
        wrap_mixed_precision=False,
    )
    allocator = allocator or DirectBufferAllocator()
    config = MdpConfig(enable=True, encoder_cp=encoder_cp)
    runtime = MdpRuntime(
        config=config,
        rank_map=rank_map,
        rank_view=view,
        process_groups=groups,
        adapter=adapter,
        encoder_domain=domain,
        planner=MdpPlanner(
            view,
            locality_slack_permille=config.locality_slack_permille,
            capacity_policy=RowCapacityPolicy(config.row_alignment),
        ),
        bridge=ModalityBridge(allocator),
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=WIDTH,
        params_dtype=torch.float32,
        num_vpp_chunks=1,
    )
    return runtime, view


def _drive_decoder(runtime, view, replay_iters, *, backward):
    """Consume the replay iterator like the native schedule would."""
    records = [next(replay_iters[0]) for _ in range(2)]
    assert [r.model_payload["microbatch"] for r in records] == [0, 1]
    if view.decoder_endpoint_id is not None:
        leaf = runtime.storage.get_leaf(0)
        assert leaf is not None
        assert runtime.storage.get_leaf(1) is None  # text-only
        # Forward routing correctness: every leaf row carries its item's
        # sentinel (identity encoder, sentinel pixels).
        offset = 0
        for index, grid in enumerate(GRIDS):
            t, h, w = grid
            rows = t * (h // MERGE) * (w // MERGE)
            block = leaf[offset : offset + rows]
            assert (block == _sentinel(view.outer_dp_rank, index)).all(), index
            offset += rows
        if backward:
            (leaf * 2.0).sum().backward()
    return records


def _reconstructed_reduced_param_grad(runtime):
    """Rebuild the reduced DistOpt parameter gradient from its owned shards."""
    encoder_ddp = runtime.encoder_domain.encoder_ddp
    param = next(encoder_ddp.module.parameters())
    bucket_group = next(
        group
        for group in encoder_ddp.bucket_groups
        + encoder_ddp.expert_parallel_bucket_groups
        if param in group.param_to_bucket
    )
    bucket = bucket_group.param_to_bucket[param]
    bucket_index = bucket_group.buckets.index(bucket)
    shard_views = bucket_group.cached_grad_buffer_shard_list[bucket_index]
    group = bucket_group.intra_distributed_optimizer_instance_group
    group_rank = bucket_group.intra_distributed_optimizer_instance_rank
    local_shard = shard_views[group_rank].detach().clone()
    gathered = [torch.empty_like(local_shard) for _ in shard_views]
    torch.distributed.all_gather(gathered, local_shard, group=group)
    full_bucket = torch.cat(gathered)
    start, end = bucket.param_to_index[param]
    return full_bucket[start:end].reshape_as(param).clone()


def _all_gather_object(value):
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _endpoint_leaves(runtime, view):
    leaf = runtime.storage.get_leaf(0)
    local = None
    if view.decoder_endpoint_id is not None:
        local = (
            view.decoder_endpoint_id,
            None if leaf is None else leaf.detach().cpu(),
        )
    leaves = {
        endpoint_id: value
        for entry in _all_gather_object(local)
        if entry is not None
        for endpoint_id, value in (entry,)
    }
    assert leaves and all(value is not None for value in leaves.values())
    return leaves


def _global_input_grads(adapter):
    local = tuple((item_id, grad.cpu()) for item_id, grad in adapter.input_grad_events)
    combined = {}
    for events in _all_gather_object(local):
        for item_id, grad in events:
            combined[item_id] = combined.get(item_id, torch.zeros_like(grad)) + grad
    return combined


def _finish_training(runtime, view, replay):
    leaves = _endpoint_leaves(runtime, view)
    _backward_decoder_without_assertions(runtime, view, replay)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    return leaves, _global_input_grads(runtime.adapter), _reconstructed_reduced_param_grad(runtime)


def _backward_decoder_without_assertions(runtime, view, replay_iters):
    """Enter backward only after rank-symmetric leaf validation has completed."""
    records = [next(replay_iters[0]) for _ in range(2)]
    if view.decoder_endpoint_id is not None:
        runtime.storage.get_leaf(0).mul(2.0).sum().backward()
    return records


def _record_encoder_cp_broadcasts(monkeypatch, runtime):
    calls = []
    original = torch.distributed.broadcast

    def _record(tensor, src, group=None, async_op=False):
        result = original(tensor, src=src, group=group, async_op=async_op)
        if group is runtime.process_groups.encoder_cp_group:
            calls.append((src, tuple(tensor.shape), tensor.detach().clone()))
        return result

    monkeypatch.setattr(torch.distributed, "broadcast", _record)
    return calls


def _capture_runtime_error(operation):
    try:
        operation()
    except RuntimeError as error:
        return str(error)
    return None


def _capture_runtime_exception(operation):
    try:
        operation()
    except RuntimeError as error:
        return error
    return None


def _runtime_cleanup_observation(runtime, allocator):
    return {
        "state": runtime.state,
        "leaves": len(runtime.storage._leaves),
        "bridge_in_flight": runtime.bridge._in_flight,
        "outstanding": allocator._outstanding,
        "handle": runtime._handle is not None,
        "eval_outputs": len(runtime._eval_outputs),
        "payload_bases": len(runtime._chunk_payload_bases),
        "all_released_once": allocator.all_released_once(),
    }


def _assert_all_ranks_clean(runtime, allocator):
    for entry in _all_gather_object(_runtime_cleanup_observation(runtime, allocator)):
        assert entry == {
            "state": MdpRuntimeState.EMPTY,
            "leaves": 0,
            "bridge_in_flight": False,
            "outstanding": 0,
            "handle": False,
            "eval_outputs": 0,
            "payload_bases": 0,
            "all_released_once": True,
        }


def _assert_rank_local_failure_converged(error, *, failed_rank, expected):
    errors = _all_gather_object(error)
    assert all(message is not None for message in errors)
    assert expected in errors[failed_rank]


def _complete_training_iteration(runtime, view):
    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    _backward_decoder_without_assertions(runtime, view, replay)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()


class _DynamicSourceCodec:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def build_source_window_with_locations(self, records, *, source_dp_lane):
        self.calls.append((records, source_dp_lane))
        if self.error is not None:
            raise self.error
        manifest = ("manifest", source_dp_lane)
        source = SimpleNamespace(metadata_manifest=lambda: manifest)
        locations = MappingProxyType(
            {
                GlobalSampleId(source_dp_lane, sample_id): (0, sample_id)
                for sample_id in range(len(GRIDS))
            }
        )
        return source, locations


class _LocalPreparationBaseException(BaseException):
    pass


class _HostileLocalPreparationBaseException(BaseException):
    def __init__(self, message):
        super().__init__(message)
        self.add_note_calls = 0

    def add_note(self, note):
        del note
        self.add_note_calls += 1
        raise AssertionError("untrusted BaseException.add_note was invoked")


def test_dynamic_producer_preparation_stops_after_p2_and_owns_exact_state():
    runtime, view = _build_runtime(decoder_pp=1)
    codec = _DynamicSourceCodec()

    producer = runtime._prepare_dynamic_encoder_producer(
        iter(range(10)), num_microbatches=2, forward_only=False, codec=codec
    )

    assert producer.local_prepare_error is None
    assert producer.owner is runtime._pre_authority_dynamic_producer.owner
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._handle is not None
    assert not runtime.storage._leaves
    assert tuple(codec.calls[0][0]) == tuple(runtime._window.records())
    assert codec.calls[0][1] == view.lane_id
    assert set(producer.item_outputs) == set(range(len(GRIDS)))
    assert set(producer.sample_location_by_id) == {
        GlobalSampleId(view.lane_id, sample_id) for sample_id in range(len(GRIDS))
    }

    producer.owner.abort()
    assert runtime.state is MdpRuntimeState.EMPTY


def test_dynamic_producer_codec_failure_returns_clean_rendezvous_carrier():
    runtime, _ = _build_runtime(decoder_pp=1)
    error = RuntimeError("injected source codec failure")

    producer = runtime._prepare_dynamic_encoder_producer(
        iter(range(10)), num_microbatches=2, forward_only=False, codec=_DynamicSourceCodec(error)
    )

    assert producer.local_prepare_error is error
    assert producer.owner is None
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._handle is None
    assert not runtime._chunk_payload_bases
    assert runtime._pre_authority_dynamic_producer is None


@pytest.mark.parametrize("failure_stage", ("capture", "codec"))
def test_dynamic_producer_normalizes_local_baseexception_and_allows_fresh_retry(
    monkeypatch, failure_stage
):
    runtime, _ = _build_runtime(decoder_pp=1)
    original = _LocalPreparationBaseException(f"injected {failure_stage} failure")
    codec = _DynamicSourceCodec(original if failure_stage == "codec" else None)
    if failure_stage == "capture":
        capture = runtime._capture_window
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise original
            return capture(*args, **kwargs)

        monkeypatch.setattr(runtime, "_capture_window", fail_once)

    failed = runtime._prepare_dynamic_encoder_producer(
        iter(range(10)), num_microbatches=2, forward_only=False, codec=codec
    )

    assert type(failed.local_prepare_error) is MdpStateError
    assert failed.local_prepare_error.__cause__ is original
    assert failed.owner is None
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._handle is None
    assert not runtime._chunk_payload_bases
    assert runtime._pre_authority_dynamic_producer is None

    recovered = runtime._prepare_dynamic_encoder_producer(
        iter(range(10)), num_microbatches=2, forward_only=False, codec=_DynamicSourceCodec()
    )
    assert recovered.local_prepare_error is None
    recovered.owner.abort()
    assert runtime.state is MdpRuntimeState.EMPTY


def test_dynamic_producer_cleanup_notes_only_the_typed_baseexception_wrapper(monkeypatch):
    runtime, _ = _build_runtime(decoder_pp=1)
    original = _HostileLocalPreparationBaseException("injected hostile codec failure")
    cleanup_error = RuntimeError("injected packed-pixel cleanup failure")

    def fail_cleanup():
        raise cleanup_error

    monkeypatch.setattr(runtime, "_release_chunk_payload_bases", fail_cleanup)
    failed = runtime._prepare_dynamic_encoder_producer(
        iter(range(10)), num_microbatches=2, forward_only=False, codec=_DynamicSourceCodec(original)
    )

    error = failed.local_prepare_error
    assert type(error) is MdpStateError
    assert error.__cause__ is original
    assert original.add_note_calls == 0
    assert any("packed-pixel buffers" in note for note in error.__notes__)
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._pre_authority_dynamic_producer is None


def test_full_training_iteration_and_state_machine():
    runtime, view = _build_runtime()
    assert runtime.state is MdpRuntimeState.EMPTY

    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    assert runtime.state is MdpRuntimeState.DECODER_READY
    _drive_decoder(runtime, view, replay, backward=True)

    tokens = torch.tensor(20.0, device="cuda")
    runtime.capture_global_num_tokens(tokens)
    assert runtime.consumed_num_tokens() is tokens  # same object, never cloned
    assert runtime.consumed_num_tokens().data_ptr() == tokens.data_ptr()
    runtime.mark_decoder_complete()
    assert runtime.state is MdpRuntimeState.DECODER_DONE
    runtime.end_iteration()
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime.iteration == 1

    # Encoder gradients exist (ZeRO-1 leaves each rank its reduced shard);
    # after one optimizer step the all-gathered parameters must be identical
    # on every rank, proving the WORLD reduction and shared update.
    param = next(runtime.encoder_domain.encoder_ddp.module.parameters())
    assert param.main_grad.abs().sum() > 0
    success, _, _ = runtime.encoder_domain.encoder_optimizer.step()
    assert success
    world = torch.distributed.get_world_size()
    gathered = [torch.empty_like(param.data) for _ in range(world)]
    torch.distributed.all_gather(gathered, param.data)
    for other in gathered[1:]:
        assert torch.equal(other, gathered[0])
    # The step moved the identity weights: gradients were really applied.
    assert not torch.equal(param.data, torch.eye(WIDTH, device=param.device))


def test_forward_only_iteration_captures_nothing_and_cleans_up():
    runtime, view = _build_runtime()
    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=True
    )
    _drive_decoder(runtime, view, replay, backward=False)
    runtime.mark_decoder_complete()  # eval requires no token capture
    runtime.end_iteration()
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()


def test_invalid_transitions_raise():
    runtime, view = _build_runtime()
    with pytest.raises(MdpStateError, match="mark_decoder_complete"):
        runtime.mark_decoder_complete()
    with pytest.raises(MdpStateError, match="end_iteration"):
        runtime.end_iteration()

    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    with pytest.raises(MdpStateError, match="begin_iteration"):
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)

    # Training decoder completion without a token capture is an error.
    _drive_decoder(runtime, view, replay, backward=True)
    with pytest.raises(MdpStateError, match="exactly one global"):
        runtime.mark_decoder_complete()
    # Recover: capture once, then a second capture must fail.
    tokens = torch.tensor(8.0, device="cuda")
    runtime.capture_global_num_tokens(tokens)
    with pytest.raises(MdpStateError, match="more than once"):
        runtime.capture_global_num_tokens(tokens)
    runtime.mark_decoder_complete()
    runtime.end_iteration()


def test_iteration_metrics_are_populated():
    runtime, view = _build_runtime()
    assert runtime.last_iteration_metrics() is None
    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    _drive_decoder(runtime, view, replay, backward=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    metrics = runtime.last_iteration_metrics()
    assert metrics.iteration == 0
    assert metrics.outer_dp_rank == view.outer_dp_rank
    assert metrics.plan_build_ms >= 0.0
    assert metrics.decoder_schedule_ms >= 0.0
    assert len(metrics.worker_loads) == len(view.worker_ids)
    assert sum(metrics.worker_loads) == 16 + 64 + 32  # all payload rows
    assert set(metrics.bridge_stats) == {"pixel", "embedding", "gradient"}
    assert metrics.bridge_stats["pixel"].total_bytes > 0
    assert all(count == 0 for count in metrics.allocator_reuse.values())


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for ECP4 spanning PP2",
)
def test_encoder_params_are_broadcast_before_optimizer_for_ecp4_across_pp():
    runtime, _ = _build_runtime(
        decoder_cp=2, encoder_cp=4, divergent_encoder_init=True
    )
    parameter = next(runtime.encoder_domain.encoder_ddp.module.parameters()).detach()
    gathered = [torch.empty_like(parameter) for _ in range(4)]
    torch.distributed.all_gather(gathered, parameter)
    for candidate in gathered:
        torch.testing.assert_close(candidate, gathered[0], rtol=0, atol=0)
    torch.testing.assert_close(
        gathered[0], torch.ones_like(gathered[0]), rtol=0, atol=0
    )


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for TP1/PP2/decoder-CP2/encoder-CP2",
)
def test_encoder_cp2_one_rank_capture_failure_converges_and_reuses():
    allocator = _TrackingAllocator()
    runtime, view = _build_runtime(
        decoder_cp=2, encoder_cp=2, allocator=allocator
    )
    original_get_batch = runtime.adapter.get_batch
    fail_once = True

    def _get_batch(iterator):
        nonlocal fail_once
        if torch.distributed.get_rank() == 1 and fail_once:
            fail_once = False
            raise RuntimeError("injected rank-local capture failure")
        return original_get_batch(iterator)

    runtime.adapter.get_batch = _get_batch
    error = _capture_runtime_error(
        lambda: runtime.begin_iteration(
            iter(range(10)), num_microbatches=2, forward_only=False
        )
    )
    runtime.adapter.get_batch = original_get_batch
    _assert_rank_local_failure_converged(
        error, failed_rank=1, expected="injected rank-local capture failure"
    )
    _assert_all_ranks_clean(runtime, allocator)
    _complete_training_iteration(runtime, view)
    _assert_all_ranks_clean(runtime, allocator)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for TP1/PP2/decoder-CP2/encoder-CP2",
)
def test_encoder_cp2_one_rank_p2_allocation_failure_converges_and_reuses():
    allocator = _TrackingAllocator()
    allocator.fail_once(rank=1, tag="packed_pixels")
    runtime, view = _build_runtime(
        decoder_cp=2, encoder_cp=2, allocator=allocator
    )
    error = _capture_runtime_error(
        lambda: runtime.begin_iteration(
            iter(range(10)), num_microbatches=2, forward_only=False
        )
    )
    _assert_rank_local_failure_converged(
        error, failed_rank=1, expected="injected rank-local packed_pixels failure"
    )
    _assert_all_ranks_clean(runtime, allocator)
    _complete_training_iteration(runtime, view)
    _assert_all_ranks_clean(runtime, allocator)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for TP1/PP2/decoder-CP2/encoder-CP2",
)
def test_encoder_cp2_symmetric_pixel_exchange_failure_cleans_and_reuses():
    allocator = _TrackingAllocator()
    runtime, view = _build_runtime(
        decoder_cp=2, encoder_cp=2, allocator=allocator
    )
    original_exchange = runtime.bridge.exchange_all_to_all
    fail_once = True

    def _exchange(ledger, *args, **kwargs):
        nonlocal fail_once
        if fail_once and ledger.phase is BridgePhase.PIXEL:
            fail_once = False
            raise RuntimeError("injected symmetric PIXEL exchange failure")
        return original_exchange(ledger, *args, **kwargs)

    runtime.bridge.exchange_all_to_all = _exchange
    error = _capture_runtime_error(
        lambda: runtime.begin_iteration(
            iter(range(10)), num_microbatches=2, forward_only=False
        )
    )
    assert _all_gather_object(error) == [
        "injected symmetric PIXEL exchange failure"
    ] * 4
    _assert_all_ranks_clean(runtime, allocator)
    _complete_training_iteration(runtime, view)
    _assert_all_ranks_clean(runtime, allocator)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for TP1/PP2/decoder-CP2/encoder-CP2",
)
def test_encoder_cp2_symmetric_p2_encode_failure_preserves_primary_and_reuses():
    allocator = _TrackingAllocator()
    allocator.fail_release_once(tag="packed_pixels")
    runtime, view = _build_runtime(
        decoder_cp=2, encoder_cp=2, allocator=allocator
    )
    original_encode = runtime.adapter.encode

    def _fail_encode(*_args, **_kwargs):
        raise RuntimeError("injected symmetric P2 encode failure")

    runtime.adapter.encode = _fail_encode
    error = _capture_runtime_exception(
        lambda: runtime.begin_iteration(
            iter(range(10)), num_microbatches=2, forward_only=False
        )
    )
    runtime.adapter.encode = original_encode
    gathered = _all_gather_object(
        (
            None if error is None else str(error),
            tuple(getattr(error, "__notes__", ())) if error is not None else (),
        )
    )
    assert all(message == "injected symmetric P2 encode failure" for message, _ in gathered)
    assert all(
        any("injected packed_pixels cleanup failure" in note for note in notes)
        for _, notes in gathered
    )
    _assert_all_ranks_clean(runtime, allocator)
    _complete_training_iteration(runtime, view)
    _assert_all_ranks_clean(runtime, allocator)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for TP1/PP2/decoder-CP2/encoder-CP2",
)
def test_encoder_cp2_endpoint_follower_p3_failure_converges_and_reuses():
    allocator = _TrackingAllocator()
    allocator.fail_once(rank=1, tag="leaf")
    runtime, view = _build_runtime(
        decoder_cp=2, encoder_cp=2, allocator=allocator
    )
    error = _capture_runtime_error(
        lambda: runtime.begin_iteration(
            iter(range(10)), num_microbatches=2, forward_only=False
        )
    )
    _assert_rank_local_failure_converged(
        error, failed_rank=1, expected="injected rank-local leaf failure"
    )
    _assert_all_ranks_clean(runtime, allocator)
    _complete_training_iteration(runtime, view)
    _assert_all_ranks_clean(runtime, allocator)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for TP1/PP2/decoder-CP2/encoder-CP2",
)
def test_encoder_cp2_endpoint_follower_missing_grad_blocks_bridge_and_reuses():
    allocator = _TrackingAllocator()
    runtime, view = _build_runtime(
        decoder_cp=2, encoder_cp=2, allocator=allocator
    )
    bridge_phases = []
    original_exchange = runtime.bridge.exchange_all_to_all

    def _exchange(ledger, *args, **kwargs):
        bridge_phases.append(ledger.phase)
        return original_exchange(ledger, *args, **kwargs)

    runtime.bridge.exchange_all_to_all = _exchange
    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    _backward_decoder_without_assertions(runtime, view, replay)
    if torch.distributed.get_rank() == 1:
        runtime.storage.get_leaf(0).grad = None
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    error = _capture_runtime_error(runtime.end_iteration)
    _assert_rank_local_failure_converged(
        error, failed_rank=1, expected="must have a gradient"
    )
    assert all(
        BridgePhase.GRADIENT not in phases
        for phases in _all_gather_object(tuple(bridge_phases))
    )
    _assert_all_ranks_clean(runtime, allocator)
    _complete_training_iteration(runtime, view)
    _assert_all_ranks_clean(runtime, allocator)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for two TP1/PP1/decoder-CP2/encoder-CP2 groups",
)
def test_encoder_cp2_one_rank_post_backward_failure_blocks_finalize_and_reuses(
    monkeypatch,
):
    allocator = _TrackingAllocator()
    runtime, view = _build_runtime(
        decoder_cp=2, decoder_pp=1, encoder_cp=2, allocator=allocator
    )
    assert len(view.planning_group_ranks) == 2
    assert runtime.process_groups.planning_group is not runtime.process_groups.world_group
    assert (
        runtime.process_groups.encoder_reduction_group
        is runtime.process_groups.world_group
    )
    finalize_calls = []
    original_finalize = finalize_encoder_grads

    def _record_finalize(*args, **kwargs):
        finalize_calls.append(True)
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(
        "megatron.core.mdp.runtime.finalize_encoder_grads", _record_finalize
    )
    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    _backward_decoder_without_assertions(runtime, view, replay)
    if torch.distributed.get_rank() == 1:
        original_release = runtime._handle.release

        def _release_then_fail():
            original_release()
            raise RuntimeError("injected rank-local post-backward release failure")

        runtime._handle.release = _release_then_fail
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    error = _capture_runtime_error(runtime.end_iteration)
    _assert_rank_local_failure_converged(
        error,
        failed_rank=1,
        expected="injected rank-local post-backward release failure",
    )
    assert _all_gather_object(len(finalize_calls)) == [0, 0, 0, 0]
    _assert_all_ranks_clean(runtime, allocator)
    _complete_training_iteration(runtime, view)
    assert _all_gather_object(len(finalize_calls)) == [1, 1, 1, 1]
    _assert_all_ranks_clean(runtime, allocator)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for TP1/PP2/decoder-CP2/encoder-CP2",
)
def test_encoder_cp2_decoder_cp2_composes_leader_transport_and_backward(monkeypatch):
    """Public transport stays leader-only while both ECP ranks execute autograd."""
    reference, reference_view = _build_runtime(decoder_cp=2, encoder_cp=1)
    reference_replay = reference.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    reference_leaves, reference_inputs, reference_grad = _finish_training(
        reference, reference_view, reference_replay
    )
    reference_step_success, _, _ = reference.encoder_domain.encoder_optimizer.step()
    assert reference_step_success
    reference_parameter = next(
        reference.encoder_domain.encoder_ddp.module.parameters()
    ).detach().clone()

    allocator = DirectBufferAllocator()
    runtime, view = _build_runtime(
        decoder_cp=2, encoder_cp=2, allocator=allocator
    )
    broadcasts = _record_encoder_cp_broadcasts(monkeypatch, runtime)
    bridge_calls = []
    original_exchange = runtime.bridge.exchange_all_to_all

    def _exchange(ledger, local_tensors, **kwargs):
        rank = view.global_rank
        bridge_calls.append(
            {
                "phase": ledger.phase,
                "local": tuple(
                    sorted(
                        (key.global_item_id, key.slice_id) for key in local_tensors
                    )
                ),
                "dest": tuple(
                    sorted(
                        (key.global_item_id, key.slice_id)
                        for key in (kwargs.get("dest_views") or {})
                    )
                ),
                "incoming": tuple(
                    (entry.key.global_item_id, entry.key.slice_id)
                    for entry in ledger.entries
                    if entry.dst_global_rank == rank
                ),
            }
        )
        return original_exchange(ledger, local_tensors, **kwargs)

    runtime.bridge.exchange_all_to_all = _exchange
    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    candidate_leaves = _endpoint_leaves(runtime, view)
    assert candidate_leaves.keys() == reference_leaves.keys()
    for endpoint_id in candidate_leaves:
        torch.testing.assert_close(
            candidate_leaves[endpoint_id], reference_leaves[endpoint_id], rtol=0, atol=0
        )

    phase_counts = {
        phase: sum(call["phase"] is phase for call in bridge_calls)
        for phase in (BridgePhase.PIXEL, BridgePhase.EMBEDDING)
    }
    gathered_phase_counts = _all_gather_object(phase_counts)
    assert all(
        counts == {BridgePhase.PIXEL: 1, BridgePhase.EMBEDDING: 1}
        for counts in gathered_phase_counts
    )

    is_leader = view.global_rank == runtime.process_groups.encoder_cp_leader_rank
    produced_items = tuple(
        route.global_item_id
        for route in runtime._plan.routes
        if route.producer_worker_id == view.my_worker_id
    )
    pixel_call = next(call for call in bridge_calls if call["phase"] is BridgePhase.PIXEL)
    embedding_call = next(
        call for call in bridge_calls if call["phase"] is BridgePhase.EMBEDDING
    )
    gathered_pre = _all_gather_object(
        {
            "rank": view.global_rank,
            "worker": view.my_worker_id,
            "leader": is_leader,
            "encoder_cp_leader": runtime.process_groups.encoder_cp_leader_rank,
            "expt_dp_is_singleton": (
                runtime.encoder_domain.encoder_ddp.expt_dp_group
                is runtime.process_groups.singleton_group
            ),
            "endpoint": view.decoder_endpoint_id,
            "has_leaf": runtime.storage.get_leaf(0) is not None,
            "materialized": runtime.adapter.materialized_count,
            "chunks": len(runtime._chunk_layouts),
            "chunk_shapes": tuple(
                (chunk.total_payload_rows, WIDTH) for chunk in runtime._chunk_layouts
            ),
            "broadcasts": tuple(
                (src, shape, tensor.cpu()) for src, shape, tensor in broadcasts
            ),
            "pixel_local": len(pixel_call["local"]),
            "pixel_incoming": len(pixel_call["incoming"]),
            "embedding_local": len(embedding_call["local"]),
            "embedding_incoming": len(embedding_call["incoming"]),
            "produced": len(produced_items),
        }
    )

    assert [entry["leader"] for entry in gathered_pre] == [True, False, True, False]
    assert [entry["endpoint"] for entry in gathered_pre] == [0, 1, None, None]
    assert [entry["has_leaf"] for entry in gathered_pre] == [True, True, False, False]
    assert [entry["materialized"] for entry in gathered_pre] == [1, 0, 0, 0]
    assert all(entry["expt_dp_is_singleton"] for entry in gathered_pre)
    assert [entry["pixel_local"] for entry in gathered_pre] == [len(GRIDS), 0, 0, 0]
    for entry in gathered_pre:
        assert len(entry["broadcasts"]) == entry["chunks"]
        assert tuple((src, shape) for src, shape, _ in entry["broadcasts"]) == tuple(
            (entry["encoder_cp_leader"], shape) for shape in entry["chunk_shapes"]
        )
        assert entry["pixel_incoming"] == (
            entry["produced"] if entry["leader"] else 0
        )
        assert entry["embedding_local"] == (
            2 * entry["produced"] if entry["leader"] else 0
        )
        assert entry["embedding_incoming"] == (
            len(GRIDS) if entry["endpoint"] is not None else 0
        )
    for worker_id in (0, 1):
        members = [entry for entry in gathered_pre if entry["worker"] == worker_id]
        assert len(members) == 2
        assert members[0]["chunk_shapes"] == members[1]["chunk_shapes"]
        for leader_call, follower_call in zip(
            members[0]["broadcasts"], members[1]["broadcasts"]
        ):
            torch.testing.assert_close(leader_call[2], follower_call[2], rtol=0, atol=0)

    _backward_decoder_without_assertions(runtime, view, replay)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    phase_counts = {
        phase: sum(call["phase"] is phase for call in bridge_calls)
        for phase in (BridgePhase.PIXEL, BridgePhase.EMBEDDING, BridgePhase.GRADIENT)
    }
    gathered_phase_counts = _all_gather_object(phase_counts)
    assert all(
        counts
        == {
            BridgePhase.PIXEL: 1,
            BridgePhase.EMBEDDING: 1,
            BridgePhase.GRADIENT: 1,
        }
        for counts in gathered_phase_counts
    )
    gradient_call = next(
        call for call in bridge_calls if call["phase"] is BridgePhase.GRADIENT
    )
    candidate_inputs = _global_input_grads(runtime.adapter)
    assert candidate_inputs.keys() == reference_inputs.keys()
    for item_id in candidate_inputs:
        torch.testing.assert_close(
            candidate_inputs[item_id], reference_inputs[item_id], rtol=0, atol=0
        )

    gathered_post = _all_gather_object(
        {
            "rank": view.global_rank,
            "leader": is_leader,
            "endpoint": view.decoder_endpoint_id,
            "output_grads": tuple(
                (int(torch.count_nonzero(grad)), float(grad.sum()))
                for grad in runtime.adapter.output_grad_events
            ),
            "input_nonzero": sum(
                int(torch.count_nonzero(grad))
                for _, grad in runtime.adapter.input_grad_events
            ),
            "gradient_local": len(gradient_call["local"]),
            "gradient_incoming": len(gradient_call["incoming"]),
            "produced": len(produced_items),
            "outstanding": allocator._outstanding,
        }
    )
    assert gathered_post[1]["endpoint"] == 1 and not gathered_post[1]["leader"]
    assert gathered_post[1]["gradient_local"] == len(GRIDS)
    assert gathered_post[1]["gradient_incoming"] == 0
    for entry in gathered_post:
        assert entry["gradient_local"] == (
            len(GRIDS) if entry["endpoint"] is not None else 0
        )
        assert entry["gradient_incoming"] == (
            2 * entry["produced"] if entry["leader"] else 0
        )
        assert entry["outstanding"] == 0
        if entry["produced"]:
            assert entry["input_nonzero"] > 0
            assert entry["output_grads"]
            if entry["leader"]:
                assert all(nonzero > 0 for nonzero, _ in entry["output_grads"])
            else:
                assert all(nonzero == 0 for nonzero, _ in entry["output_grads"])

    candidate_grad = _reconstructed_reduced_param_grad(runtime)
    torch.testing.assert_close(candidate_grad, reference_grad, rtol=0, atol=0)
    candidate_step_success, _, _ = runtime.encoder_domain.encoder_optimizer.step()
    assert candidate_step_success
    candidate_parameter = next(
        runtime.encoder_domain.encoder_ddp.module.parameters()
    ).detach()
    grad_max_delta = float((candidate_grad - reference_grad).abs().max())
    torch.testing.assert_close(
        candidate_parameter,
        reference_parameter,
        rtol=0,
        atol=0,
        msg=f"optimizer-step parameter parity failed; pre-step grad max delta={grad_max_delta}",
    )
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
