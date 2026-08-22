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
from megatron.core.mdp.bridge import BridgeBufferKey, BridgePhase, ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import build_encoder_domain, build_encoder_pg_collection
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState, _allocate_endpoint_leaves
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
ENCODER_CP_GRIDS = (
    (1, 8, 8),
    (2, 4, 6),
    (1, 6, 6),
    (2, 4, 4),
    (1, 4, 6),
    (1, 4, 4),
)  # payload rows 64/48/36/32/24/16


def _sentinel(lane, item_index):
    return float(10 * (lane + 1) + item_index)


class _TinyEncoder(torch.nn.Module):
    """CP-aware identity linear: shard rows, project locally, autograd-gather."""

    def __init__(self, config, cp_group):
        super().__init__()
        self.config = config
        self.cp_group = cp_group
        self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(WIDTH))

    def forward(self, x):
        cp_size = self.cp_group.size()
        cp_rank = self.cp_group.rank()
        base, remainder = divmod(x.size(0), cp_size)
        split_sizes = [base + (rank < remainder) for rank in range(cp_size)]
        local_start = sum(split_sizes[:cp_rank])
        local = x.narrow(0, local_start, split_sizes[cp_rank])
        local_output = self.proj(local)
        return gather_from_sequence_parallel_region(
            local_output,
            tensor_parallel_output_grad=True,
            group=self.cp_group,
            output_split_sizes=split_sizes if len(set(split_sizes)) > 1 else None,
        )


class _RecordingBridge(ModalityBridge):
    def __init__(self, allocator):
        super().__init__(allocator)
        self.calls = []
        self.p2p_calls = []

    @staticmethod
    def _call(ledger, local_tensors, kwargs):
        return (
            ledger.phase,
            tuple(sorted(key.global_item_id for key in local_tensors)),
            tuple(sorted(key.global_item_id for key in (kwargs.get("dest_views") or {}))),
        )

    def exchange(self, ledger, local_tensors, **kwargs):
        self.p2p_calls.append(self._call(ledger, local_tensors, kwargs))
        return super().exchange(ledger, local_tensors, **kwargs)

    def exchange_all_to_all(self, ledger, local_tensors, **kwargs):
        self.calls.append(self._call(ledger, local_tensors, kwargs))
        return super().exchange_all_to_all(ledger, local_tensors, **kwargs)


class _IdentityRecordingAllocator(DirectBufferAllocator):
    def __init__(self):
        super().__init__()
        self.acquired = []
        self.released = []

    def acquire(self, *, rows, width, dtype, device, tag):
        base = super().acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)
        self.acquired.append((tag, base))
        return base

    def release(self, tensor):
        self.released.append(tensor)
        super().release(tensor)

    def bases_for_tag(self, tag):
        return tuple(base for acquired_tag, base in self.acquired if acquired_tag == tag)

    def release_count(self, base):
        return sum(released is base for released in self.released)


class _FailOnceReleaseAllocator(_IdentityRecordingAllocator):
    def __init__(self, fail_tags):
        super().__init__()
        self.fail_tags = frozenset(fail_tags)
        self.release_attempts = []
        self._failed_bases = []

    def release(self, tensor):
        self.release_attempts.append(tensor)
        tag = next(tag for tag, base in self.acquired if base is tensor)
        already_failed = any(base is tensor for base in self._failed_bases)
        if tag in self.fail_tags and not already_failed:
            self._failed_bases.append(tensor)
            raise RuntimeError(f"injected {tag} release failure")
        self.released.append(tensor)
        DirectBufferAllocator.release(self, tensor)


class _FailSecondLeafAllocator(_IdentityRecordingAllocator):
    def __init__(self):
        super().__init__()
        self._leaf_calls = 0

    def acquire(self, *, rows, width, dtype, device, tag):
        if tag == "leaf":
            self._leaf_calls += 1
            if self._leaf_calls == 2:
                raise ValueError("injected second leaf allocation failure")
        return super().acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)


class _FailThirdLeafAndFirstReleaseAllocator(_IdentityRecordingAllocator):
    def __init__(self):
        super().__init__()
        self._leaf_calls = 0
        self.release_attempts = []

    def acquire(self, *, rows, width, dtype, device, tag):
        if tag == "leaf":
            self._leaf_calls += 1
            if self._leaf_calls == 3:
                raise ValueError("injected third leaf allocation failure")
        return super().acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)

    def release(self, tensor):
        self.release_attempts.append(tensor)
        if len(self.release_attempts) == 1:
            raise RuntimeError("injected first leaf release failure")
        super().release(tensor)


class _StubAdapter:
    """Deterministic capture: two microbatches, three items in mb0, mb1 text-only."""

    payload_width = WIDTH
    spatial_merge_size = MERGE

    def __init__(self, lane, grids=GRIDS):
        self._lane = lane if lane is not None else 0
        self.grids = tuple(grids)
        self.materialized_count = 0
        self.encoded_chunks = []
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
        for index, grid in enumerate(self.grids):
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
        return _TinyEncoder(model_config, pg_collection.cp)

    def encode(self, encoder, payload, layout):
        pieces = []
        sentinels = []
        for segment in layout.segments:
            piece = payload[
                segment.payload_row_start : segment.payload_row_start + segment.output_rows
            ]
            pieces.append(piece)
            sentinels.append(float(piece[0, 0].item()))
        self.encoded_chunks.append(
            (tuple(segment.global_item_id for segment in layout.segments), tuple(sentinels))
        )
        inputs = torch.cat(pieces) if pieces else payload[:0]
        output = encoder(inputs)
        if output.requires_grad:
            chunk_index = len(self.encoded_chunks) - 1

            def _record_grad(grad, index=chunk_index):
                self.output_grad_events.append((index, grad.detach().clone()))
                return grad

            output.register_hook(_record_grad)
        return output


def _build_runtime(
    *,
    encoder_cp=1,
    pp=2,
    grids=GRIDS,
    pixel_owner_shard=False,
    encoder_max_payload_rows=None,
    allocator=None,
):
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=pp, cp=1, ep=1, encoder_cp=encoder_cp)
    )
    view = rank_map.view(rank)
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=encoder_cp, process_groups=groups
    )
    adapter = _StubAdapter(view.outer_dp_rank, grids=grids)
    config = MdpConfig(
        enable=True,
        encoder_cp=encoder_cp,
        encoder_max_payload_rows=encoder_max_payload_rows,
        pixel_owner_shard=pixel_owner_shard,
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
        mdp_config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False, overlap_param_gather=False
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1e-3, use_distributed_optimizer=True, clip_grad=1.0
        ),
        encoder_pgs=encoder_pgs,
        wrap_mixed_precision=False,
    )
    if allocator is None:
        allocator = DirectBufferAllocator()
    bridge = _RecordingBridge(allocator)
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
        bridge=bridge,
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=WIDTH,
        params_dtype=torch.float32,
        num_vpp_chunks=1,
    )
    return runtime, view


def _drive_decoder(
    runtime, view, replay_iters, *, backward, weighted_loss=False, expect_text_only=False
):
    """Consume the replay iterator like the native schedule would."""
    records = [next(replay_iters[0]) for _ in range(2)]
    assert [r.model_payload["microbatch"] for r in records] == [0, 1]
    if view.lane_id is not None:
        leaf = runtime.storage.get_leaf(0)
        if expect_text_only:
            assert leaf is None
            assert runtime.storage.get_leaf(1) is None
            return records
        assert leaf is not None
        assert runtime.storage.get_leaf(1) is None  # text-only
        # Forward routing correctness: every leaf row carries its item's
        # sentinel (identity encoder, sentinel pixels).
        offset = 0
        for index, grid in enumerate(runtime.adapter.grids):
            t, h, w = grid
            rows = t * (h // MERGE) * (w // MERGE)
            block = leaf[offset : offset + rows]
            assert (block == _sentinel(view.outer_dp_rank, index)).all(), index
            offset += rows
        if backward:
            if weighted_loss:
                weight = torch.linspace(
                    0.25, 1.75, leaf.numel(), dtype=leaf.dtype, device=leaf.device
                ).view_as(leaf)
                (leaf * weight).sum().backward()
            else:
                (leaf * 2.0).sum().backward()
    return records


def _reconstructed_reduced_param_grad(runtime):
    """Rebuild one reduced DistOpt gradient from the actually owned shards."""
    encoder_ddp = runtime.encoder_domain.encoder_ddp
    param = next(encoder_ddp.module.parameters())
    bucket_group = next(
        group
        for group in encoder_ddp.bucket_groups + encoder_ddp.expert_parallel_bucket_groups
        if param in group.param_to_bucket
    )
    bucket = bucket_group.param_to_bucket[param]
    bucket_index = bucket_group.buckets.index(bucket)
    shard_views = bucket_group.cached_grad_buffer_shard_list[bucket_index]
    assert shard_views is not None
    group = bucket_group.intra_distributed_optimizer_instance_group
    group_rank = bucket_group.intra_distributed_optimizer_instance_rank
    group_size = bucket_group.intra_distributed_optimizer_instance_size
    local_shard = shard_views[group_rank].detach().clone()
    reduced_shards = [torch.empty_like(local_shard) for _ in range(group_size)]
    torch.distributed.all_gather(reduced_shards, local_shard, group=group)
    reduced_bucket = torch.cat(reduced_shards)
    start, end = bucket.param_to_index[param]
    return reduced_bucket[start:end].reshape_as(param).clone()


def _finish_training(runtime, view):
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    _drive_decoder(runtime, view, replay, backward=True, weighted_loss=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    return _reconstructed_reduced_param_grad(runtime)


def test_full_training_iteration_and_state_machine():
    runtime, view = _build_runtime()
    assert runtime.state is MdpRuntimeState.EMPTY

    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
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
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=True)
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
    assert runtime.state is MdpRuntimeState.EMPTY

    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    plan_before = runtime._plan
    handle_before = runtime._handle
    payload_bases_before = runtime._chunk_payload_bases
    leaf_before = runtime.storage.get_leaf(0)
    with pytest.raises(MdpStateError, match="begin_iteration"):
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    assert runtime.state is MdpRuntimeState.DECODER_READY
    assert runtime._plan is plan_before
    assert runtime._handle is handle_before
    assert runtime._chunk_payload_bases is payload_bases_before
    assert all(
        current is previous
        for current, previous in zip(
            runtime._chunk_payload_bases, payload_bases_before, strict=True
        )
    )
    assert runtime.storage.get_leaf(0) is leaf_before
    assert (leaf_before is not None) is (view.lane_id is not None)

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


def test_cleanup_failure_does_not_mask_original_begin_error(monkeypatch):
    runtime, _ = _build_runtime()

    def _original_failure(*args, **kwargs):
        raise ValueError("original begin failure")

    def _cleanup_failure():
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(runtime, "_begin_iteration", _original_failure)
    monkeypatch.setattr(runtime, "_cleanup_failed_iteration", _cleanup_failure)
    with pytest.raises(ValueError, match="original begin failure"):
        runtime.begin_iteration(iter(range(2)), num_microbatches=2, forward_only=False)


def test_cleanup_failure_does_not_mask_original_end_error(monkeypatch):
    runtime, _ = _build_runtime()
    runtime._state = MdpRuntimeState.DECODER_DONE

    def _original_failure():
        raise ValueError("original end failure")

    def _cleanup_failure():
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(runtime, "_end_iteration", _original_failure)
    monkeypatch.setattr(runtime, "_cleanup_failed_iteration", _cleanup_failure)
    with pytest.raises(ValueError, match="original end failure"):
        runtime.end_iteration()


def test_partial_leaf_allocation_failure_releases_exact_prior_base():
    allocator = _FailSecondLeafAllocator()
    layouts = tuple(
        SimpleNamespace(
            microbatch_id=microbatch_id,
            text_only=False,
            total_output_rows=2,
            segments=(
                SimpleNamespace(global_item_id=microbatch_id, leaf_row_start=0, output_rows=2),
            ),
        )
        for microbatch_id in range(2)
    )
    plan = SimpleNamespace(layouts=layouts, capacity_policy=RowCapacityPolicy(1))
    with pytest.raises(ValueError, match="injected second leaf allocation failure"):
        _allocate_endpoint_leaves(
            plan,
            allocator=allocator,
            hidden_size=WIDTH,
            params_dtype=torch.float32,
            device=torch.device("cpu"),
        )
    first_base = allocator.bases_for_tag("leaf")
    assert len(first_base) == 1
    assert allocator.release_count(first_base[0]) == 1
    assert allocator.released[0] is first_base[0]


def test_leaf_cleanup_failure_preserves_acquire_error_and_attempts_every_base():
    allocator = _FailThirdLeafAndFirstReleaseAllocator()
    layouts = tuple(
        SimpleNamespace(
            microbatch_id=microbatch_id,
            text_only=False,
            total_output_rows=2,
            segments=(
                SimpleNamespace(global_item_id=microbatch_id, leaf_row_start=0, output_rows=2),
            ),
        )
        for microbatch_id in range(3)
    )
    plan = SimpleNamespace(layouts=layouts, capacity_policy=RowCapacityPolicy(1))
    with pytest.raises(ValueError, match="injected third leaf allocation failure"):
        _allocate_endpoint_leaves(
            plan,
            allocator=allocator,
            hidden_size=WIDTH,
            params_dtype=torch.float32,
            device=torch.device("cpu"),
        )
    bases = allocator.bases_for_tag("leaf")
    assert len(bases) == 2
    assert len(allocator.release_attempts) == 3
    assert allocator.release_attempts[0] is bases[0]
    assert allocator.release_attempts[1] is bases[1]
    assert allocator.release_attempts[2] is bases[0]
    assert allocator.release_count(bases[0]) == 1
    assert allocator.release_count(bases[1]) == 1


def _assert_failed_iteration_cleanup(runtime, allocator, caught, error_type, message):
    local_errors = []
    if not isinstance(caught, error_type) or message not in str(caught):
        local_errors.append(f"unexpected error: {type(caught).__name__}: {caught}")
    if runtime.state is not MdpRuntimeState.EMPTY:
        local_errors.append(f"runtime state is {runtime.state}")
    for label, check in (
        ("storage", runtime.storage.assert_empty),
        ("bridge", runtime.bridge.assert_idle),
    ):
        try:
            check()
        except Exception as error:
            local_errors.append(f"{label}: {error}")
    for tag, base in allocator.acquired:
        release_count = allocator.release_count(base)
        if release_count != 1:
            local_errors.append(f"{tag} release count is {release_count}")
    errors_by_rank = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors_by_rank, tuple(local_errors))
    assert errors_by_rank == [()] * torch.distributed.get_world_size()


def test_p3_failure_preserves_original_and_retries_exact_bases(monkeypatch):
    allocator = _FailOnceReleaseAllocator({"leaf"})
    runtime, _ = _build_runtime(allocator=allocator)
    original_exchange = runtime.bridge.exchange_all_to_all

    def _fail_embedding(ledger, local_tensors, **kwargs):
        if ledger.phase is BridgePhase.EMBEDDING:
            raise ValueError("injected P3 embedding failure")
        return original_exchange(ledger, local_tensors, **kwargs)

    monkeypatch.setattr(runtime.bridge, "exchange_all_to_all", _fail_embedding)
    caught = None
    try:
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    except BaseException as error:
        caught = error
    _assert_failed_iteration_cleanup(
        runtime, allocator, caught, ValueError, "injected P3 embedding failure"
    )


def test_eval_release_failure_retries_every_exact_packed_base():
    allocator = _FailOnceReleaseAllocator({"packed_pixels"})
    runtime, view = _build_runtime(allocator=allocator)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=True)
    _drive_decoder(runtime, view, replay, backward=False)
    runtime.mark_decoder_complete()
    caught = None
    try:
        runtime.end_iteration()
    except BaseException as error:
        caught = error
    _assert_failed_iteration_cleanup(
        runtime, allocator, caught, RuntimeError, "injected packed_pixels release failure"
    )


def test_p5_failure_preserves_original_and_retries_grad_and_pixel_bases(monkeypatch):
    allocator = _FailOnceReleaseAllocator({"grad_regroup", "packed_pixels"})
    runtime, view = _build_runtime(allocator=allocator)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    _drive_decoder(runtime, view, replay, backward=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    original_exchange = runtime.bridge.exchange_all_to_all

    def _fail_gradient(ledger, local_tensors, **kwargs):
        if ledger.phase is BridgePhase.GRADIENT:
            raise ValueError("injected P5 gradient failure")
        return original_exchange(ledger, local_tensors, **kwargs)

    monkeypatch.setattr(runtime.bridge, "exchange_all_to_all", _fail_gradient)
    caught = None
    try:
        runtime.end_iteration()
    except BaseException as error:
        caught = error
    _assert_failed_iteration_cleanup(
        runtime, allocator, caught, ValueError, "injected P5 gradient failure"
    )


def test_iteration_metrics_are_populated():
    runtime, view = _build_runtime()
    assert runtime.last_iteration_metrics() is None
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
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
    reason="requires world4 for the PP4/C1/E2 runtime topology",
)
def test_encoder_cp_e2_runtime_matches_e1_and_reuses_lifecycle():
    reference, reference_view = _build_runtime(
        encoder_cp=1,
        pp=4,
        grids=ENCODER_CP_GRIDS,
        pixel_owner_shard=True,
        encoder_max_payload_rows=50,
    )
    reference_grad = _finish_training(reference, reference_view)

    allocator = _IdentityRecordingAllocator()
    runtime, view = _build_runtime(
        encoder_cp=2,
        pp=4,
        grids=ENCODER_CP_GRIDS,
        pixel_owner_shard=True,
        encoder_max_payload_rows=50,
        allocator=allocator,
    )
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    expected_chunk_rows = {0: [64, 48], 1: [48, 36, 24]}[view.my_worker_id]
    expected_chunk_items = {0: [(0,), (3, 5)], 1: [(1,), (2,), (4,)]}[view.my_worker_id]
    assert [chunk.total_payload_rows for chunk in runtime._chunk_layouts] == (expected_chunk_rows)
    assert [entry[0] for entry in runtime.adapter.encoded_chunks] == expected_chunk_items
    assert runtime._handle is not None

    materialized = torch.tensor(
        runtime.adapter.materialized_count, dtype=torch.int64, device="cuda"
    )
    materialized_by_rank = [torch.empty_like(materialized) for _ in range(4)]
    torch.distributed.all_gather(materialized_by_rank, materialized)
    assert [int(value.item()) for value in materialized_by_rank] == [1, 0, 0, 0]

    pixel_ledger = runtime._iter_ledgers[BridgePhase.PIXEL]
    assert len(pixel_ledger.entries) == len(ENCODER_CP_GRIDS)
    assert len({entry.key.global_item_id for entry in pixel_ledger.entries}) == len(
        ENCODER_CP_GRIDS
    )
    route_by_item = {route.global_item_id: route for route in runtime._plan.routes}
    for entry in pixel_ledger.entries:
        route = route_by_item[entry.key.global_item_id]
        assert entry.src_global_rank == 0
        assert entry.dst_global_rank == runtime.rank_map.worker_leader_rank(
            0, route.producer_worker_id
        )
    assert {entry.dst_global_rank for entry in pixel_ledger.entries} == {0, 2}

    assert [call[0] for call in runtime.bridge.calls] == [BridgePhase.PIXEL, BridgePhase.EMBEDDING]
    produced_items = tuple(
        sorted(
            route.global_item_id
            for route in runtime._plan.routes
            if route.producer_worker_id == view.my_worker_id
        )
    )
    is_worker_leader = view.global_rank == runtime.process_groups.encoder_cp_leader_rank
    assert runtime.bridge.calls[1][1] == (produced_items if is_worker_leader else ())

    _drive_decoder(runtime, view, replay, backward=True, weighted_loss=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    assert [call[0] for call in runtime.bridge.calls] == [
        BridgePhase.PIXEL,
        BridgePhase.EMBEDDING,
        BridgePhase.GRADIENT,
    ]
    assert runtime.bridge.calls[-1][2] == (produced_items if is_worker_leader else ())
    grad_order = [index for index, _ in runtime.adapter.output_grad_events]
    assert sorted(grad_order) == list(range(len(expected_chunk_rows)))
    gathered_orders = [None, None]
    torch.distributed.all_gather_object(
        gathered_orders, grad_order, group=runtime.process_groups.encoder_cp_group
    )
    assert gathered_orders[0] == gathered_orders[1]
    grad_magnitude = sum(
        float(grad.abs().sum().item()) for _, grad in runtime.adapter.output_grad_events
    )
    reduced_grad = _reconstructed_reduced_param_grad(runtime)
    local_errors = []
    if (grad_magnitude > 0.0) is not is_worker_leader:
        local_errors.append("leader/follower gradient role mismatch")
    try:
        torch.testing.assert_close(reduced_grad, reference_grad, rtol=1e-6, atol=1e-6)
    except AssertionError as error:
        local_errors.append(f"reduced gradient mismatch: {error}")
    for tag in ("packed_pixels", "grad_regroup", "leaf"):
        counts = [allocator.release_count(base) for base in allocator.bases_for_tag(tag)]
        if any(count != 1 for count in counts):
            local_errors.append(f"{tag} release counts: {counts}")
    if not all(
        any(released is base for _, base in allocator.acquired) for released in allocator.released
    ):
        local_errors.append("allocator received a non-base release")
    if runtime.state is not MdpRuntimeState.EMPTY:
        local_errors.append(f"runtime state is {runtime.state}")
    for label, check in (
        ("storage", runtime.storage.assert_empty),
        ("bridge", runtime.bridge.assert_idle),
    ):
        try:
            check()
        except Exception as error:
            local_errors.append(f"{label}: {error}")
    errors_by_rank = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors_by_rank, tuple(local_errors))
    assert errors_by_rank == [()] * torch.distributed.get_world_size()

    # Forward-only reuse leaves worker 1 empty, then text-only training keeps
    # every rank in the phase collectives and WORLD encoder finalizer.
    runtime.adapter.grids = (ENCODER_CP_GRIDS[-1],)
    runtime.adapter.materialized_count = 0
    runtime.adapter.encoded_chunks.clear()
    runtime.adapter.output_grad_events.clear()
    runtime.bridge.calls.clear()
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=True)
    assert bool(runtime.adapter.encoded_chunks) is (view.my_worker_id == 0)
    _drive_decoder(runtime, view, replay, backward=False)
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    assert runtime.state is MdpRuntimeState.EMPTY
    for tag in ("packed_pixels", "leaf"):
        assert all(allocator.release_count(base) == 1 for base in allocator.bases_for_tag(tag))

    runtime.adapter.grids = ()
    runtime.adapter.encoded_chunks.clear()
    runtime.bridge.calls.clear()
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    assert runtime.adapter.encoded_chunks == []
    _drive_decoder(runtime, view, replay, backward=True, expect_text_only=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    assert runtime.iteration == 3
    runtime.storage.assert_empty()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires world4 for the PP4/C1/E2 runtime topology",
)
def test_encoder_cp_e2_default_pixel_p2p_with_empty_worker_matches_e1():
    one_item = (ENCODER_CP_GRIDS[0],)
    reference, reference_view = _build_runtime(
        encoder_cp=1, pp=4, grids=one_item, pixel_owner_shard=False
    )
    reference_grad = _finish_training(reference, reference_view)

    runtime, view = _build_runtime(encoder_cp=2, pp=4, grids=one_item, pixel_owner_shard=False)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    assert [call[0] for call in runtime.bridge.p2p_calls] == [BridgePhase.PIXEL]
    assert [call[0] for call in runtime.bridge.calls] == [BridgePhase.EMBEDDING]
    assert len(runtime._iter_ledgers[BridgePhase.PIXEL].entries) == 1
    materialized = torch.tensor(
        runtime.adapter.materialized_count, dtype=torch.int64, device="cuda"
    )
    materialized_by_rank = [torch.empty_like(materialized) for _ in range(4)]
    torch.distributed.all_gather(materialized_by_rank, materialized)
    assert [int(value.item()) for value in materialized_by_rank] == [1, 0, 0, 0]

    if view.my_worker_id == 0:
        assert runtime._handle is not None
    else:
        assert runtime._handle is None
        assert runtime.adapter.encoded_chunks == []
    _drive_decoder(runtime, view, replay, backward=True, weighted_loss=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    if view.my_worker_id == 0:
        magnitude = sum(
            float(grad.abs().sum().item()) for _, grad in runtime.adapter.output_grad_events
        )
        is_worker_leader = view.global_rank == runtime.process_groups.encoder_cp_leader_rank
        assert (magnitude > 0.0) is is_worker_leader
    else:
        assert runtime.adapter.output_grad_events == []
    torch.testing.assert_close(
        _reconstructed_reduced_param_grad(runtime), reference_grad, rtol=1e-6, atol=1e-6
    )
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
