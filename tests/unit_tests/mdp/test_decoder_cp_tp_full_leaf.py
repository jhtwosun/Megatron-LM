# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-CP full-leaf routing with tensor-parallel decoder replicas.

Run the distributed cases with exactly four ranks: TP2 x decoder-CP2 x PP1.
"""

import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.forward_step import pack_or_pad_batch
from examples.multimodal_dev.models.base import MultimodalModel
from megatron.core import parallel_state, tensor_parallel
from megatron.core.context_parallel_layout import get_thd_context_parallel_rank_indices
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp import bridge as bridge_module
from megatron.core.mdp import decoder_cp as decoder_cp_module
from megatron.core.mdp import runtime as runtime_module
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import BridgeBufferKey, BridgePhase, BridgeTensorSpec, ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import build_encoder_domain, build_encoder_pg_collection
from megatron.core.mdp.errors import MdpPlanError, MdpStateError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import (
    PLAN_SCHEMA_VERSION,
    EncoderThdLayout,
    EncoderThdSegment,
    LayoutSegment,
    MdpBatchPlan,
    MicrobatchLayout,
    RouteSlice,
    RowCapacityPolicy,
)
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem, VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import MdpIterationWindow
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig


class _NoopAllocator:
    def acquire(self, **kwargs):  # pragma: no cover - ledger construction allocates nothing
        raise AssertionError(kwargs)

    def release(self, tensor):  # pragma: no cover - ledger construction releases nothing
        raise AssertionError(tensor)


def _descriptor(item_id=0):
    return VisionDescriptor(
        global_item_id=item_id,
        sample_id=0,
        image_ordinal=item_id,
        owner_dp_lane=0,
        microbatch_id=0,
        estimated_cost_units=16,
        payload_rows=16,
        output_rows=4,
        grid_thw=(1, 4, 4),
        owner_worker_id=0,
    )


def test_full_leaf_plan_and_wire_expand_over_cp_not_tp():
    """One item has C decoder routes while PIXEL remains exactly-once."""
    rank_map = build_rank_map(MdpRankSpec(world_size=4, tp=2, pp=1, cp=2, ep=1, encoder_cp=1))
    view = rank_map.view(0)
    plan = MdpPlanner(
        view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()
    ).build_plan(0, (_descriptor(),), (0,))

    assert PLAN_SCHEMA_VERSION >= 7
    assert tuple(
        (route.global_item_id, route.slice_id, route.endpoint_rank) for route in plan.routes
    ) == ((0, 0, 0), (0, 1, 2))

    device = torch.device("cpu")
    pixel_specs = {BridgeBufferKey(0, 0): BridgeTensorSpec(16, 16, 8, torch.float32, device)}
    io_specs = {
        BridgeBufferKey(0, slice_id): BridgeTensorSpec(4, 4, 8, torch.float32, device)
        for slice_id in range(2)
    }
    bridge = ModalityBridge(_NoopAllocator())
    pixel = bridge.build_ledger(BridgePhase.PIXEL, plan, rank_map, pixel_specs)
    embedding = bridge.build_ledger(BridgePhase.EMBEDDING, plan, rank_map, io_specs)
    gradient = bridge.build_ledger(BridgePhase.GRADIENT, plan, rank_map, io_specs)

    assert tuple(entry.key for entry in pixel.entries) == (BridgeBufferKey(0, 0),)
    assert tuple(entry.key for entry in embedding.entries) == (
        BridgeBufferKey(0, 0),
        BridgeBufferKey(0, 1),
    )
    assert tuple(entry.key for entry in gradient.entries) == (
        BridgeBufferKey(0, 0),
        BridgeBufferKey(0, 1),
    )
    assert {entry.src_global_rank for entry in gradient.entries} == {0, 2}
    assert len(gradient.entries) == rank_map.spec.cp
    assert len(gradient.entries) != rank_map.spec.tp * rank_map.spec.cp


def test_route_product_ledger_uses_complete_item_slice_identity():
    """Two producers and noncontiguous endpoints retain zero-row plan slices."""

    class _RankMap:
        @staticmethod
        def worker_ranks(outer_dp_rank, worker_id):
            assert outer_dp_rank == 0
            return ((3,), (7,))[worker_id]

    routes = (
        RouteSlice(global_item_id=0, producer_worker_id=0, endpoint_rank=3, slice_id=0),
        RouteSlice(global_item_id=0, producer_worker_id=0, endpoint_rank=9, slice_id=1),
        RouteSlice(global_item_id=1, producer_worker_id=1, endpoint_rank=3, slice_id=0),
        RouteSlice(global_item_id=1, producer_worker_id=1, endpoint_rank=9, slice_id=1),
    )
    segments = (
        EncoderThdSegment(0, 0, 0, 0, 0, 16, 0, 4, (1, 4, 4)),
        EncoderThdSegment(1, 0, 0, 1, 16, 16, 4, 2, (1, 4, 4)),
    )
    plan = MdpBatchPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        iteration=0,
        outer_dp_rank=0,
        capacity_policy=RowCapacityPolicy(),
        routes=routes,
        layouts=(
            MicrobatchLayout(
                microbatch_id=0,
                text_only=False,
                total_output_rows=6,
                segments=(LayoutSegment(0, 0, 4), LayoutSegment(1, 4, 2)),
            ),
        ),
        encoder_layouts=(
            EncoderThdLayout(producer_worker_id=0, segments=(segments[0],)),
            EncoderThdLayout(producer_worker_id=1, segments=(segments[1],)),
        ),
        digest=b"\x00" * 16,
    )
    specs = {
        BridgeBufferKey(0, 0): BridgeTensorSpec(4, 4, WIDTH, torch.float32, torch.device("cpu")),
        BridgeBufferKey(0, 1): BridgeTensorSpec(4, 4, WIDTH, torch.float32, torch.device("cpu")),
        BridgeBufferKey(1, 0): BridgeTensorSpec(2, 2, WIDTH, torch.float32, torch.device("cpu")),
        BridgeBufferKey(1, 1): BridgeTensorSpec(0, 0, WIDTH, torch.float32, torch.device("cpu")),
    }
    bridge = ModalityBridge(_NoopAllocator())

    embedding = bridge.build_ledger(BridgePhase.EMBEDDING, plan, _RankMap(), specs)
    gradient = bridge.build_ledger(BridgePhase.GRADIENT, plan, _RankMap(), specs)

    assert plan.route_for_item_slice(1, 1) is routes[3]
    assert tuple(entry.key for entry in embedding.entries) == (
        BridgeBufferKey(0, 0),
        BridgeBufferKey(0, 1),
        BridgeBufferKey(1, 0),
    )
    assert tuple(entry.key for entry in gradient.entries) == (
        BridgeBufferKey(0, 0),
        BridgeBufferKey(1, 0),
        BridgeBufferKey(0, 1),
    )
    assert embedding.total_bytes == gradient.total_bytes == 10 * WIDTH * 4
    assert embedding.remote_bytes == gradient.remote_bytes == 6 * WIDTH * 4
    assert any(entry.src_global_rank == entry.dst_global_rank for entry in embedding.entries)
    assert any(entry.src_global_rank != entry.dst_global_rank for entry in embedding.entries)


def test_full_leaf_remains_the_d3_default_oracle():
    assert MdpConfig(enable=True).decoder_cp_routing == "full_leaf"
    assert MdpConfig(enable=True, decoder_cp_routing="full_leaf").decoder_cp_routing == "full_leaf"


_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
_DISTRIBUTED = _WORLD == 4
WIDTH = 8
SEQ = 8
IMAGE_TOKEN_ID = 7
IMAGE_POSITIONS = (0, 2, 5, 7)
ROW_OWNER = (0, 1, 1, 0)
D3_POSITIONS = tuple(range(SEQ))

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _model_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1, context_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


class _TpOwnerShardPackAdapter:
    """Exercise the native TP broadcast under the window ownership context."""

    payload_width = WIDTH
    spatial_merge_size = 2

    def __init__(self, use_packed_sequence):
        self.use_packed_sequence = use_packed_sequence
        self.pixel_presence = []
        self.grid_values = []

    @staticmethod
    def _sample(sample_id):
        return {
            "input_ids": torch.tensor([IMAGE_TOKEN_ID] * 4, dtype=torch.long),
            "labels": torch.full((4,), -100, dtype=torch.long),
            "loss_mask": torch.zeros(4, dtype=torch.float32),
            "pixel_values": torch.full((16, WIDTH), float(sample_id + 1)),
            "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
        }

    def get_batch(self, iterator):
        tp_source = parallel_state.get_tensor_model_parallel_rank() == 0
        sample_id = next(iterator) if tp_source else None
        raw_batch = [self._sample(sample_id)] if tp_source else None
        batch = pack_or_pad_batch(
            raw_batch, use_packed_sequence=self.use_packed_sequence, seq_length=4, device="cuda"
        )
        self.pixel_presence.append("pixel_values" in batch)
        self.grid_values.append(tuple(int(v) for v in batch["image_grid_thw"][0]))
        item_id = len(self.pixel_presence) - 1
        item = CapturedVisionItem(
            sample_id=item_id,
            image_ordinal=0,
            grid_thw=(1, 4, 4),
            payload_row_start=0,
            payload_rows=16,
            decoder_positions=(0, 1, 2, 3),
        )
        return CapturedMicrobatch(
            decoder_packed_seq_params=batch.get("packed_seq_params"),
            decoder_input_shape=tuple(batch["input_ids"].shape),
            vision_items=(item,),
            flat_pixel_payload=batch.get("pixel_values"),
            model_payload=MappingProxyType({"input_ids": batch["input_ids"]}),
        )

    def estimate_cost(self, item):
        return item.payload_rows


@pytest.mark.parametrize("use_packed_sequence", [False, True])
@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 native TP2 x CP2")
def test_tp2_owner_shard_keeps_pixels_on_only_the_selected_tp0_source(use_packed_sequence):
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(MdpRankSpec(world_size=4, tp=2, pp=1, cp=2, ep=1, encoder_cp=1))
    view = rank_map.view(rank)
    adapter = _TpOwnerShardPackAdapter(use_packed_sequence)
    source_iterator = iter((0, 1)) if parallel_state.get_tensor_model_parallel_rank() == 0 else None
    window = MdpIterationWindow.capture(
        source_iterator,
        num_microbatches=2,
        adapter=adapter,
        num_vpp_chunks=1,
        lane_id=view.lane_id,
        pixel_owner_shard=True,
        my_worker_id=view.my_worker_id,
        num_logical_workers=len(view.worker_ids),
        data_loader_source_worker_ids=view.data_loader_source_worker_ids,
    )

    assert view.data_loader_source_worker_ids == (0, 2)
    assert adapter.grid_values == [(1, 4, 4), (1, 4, 4)]
    assert adapter.pixel_presence == [rank == 0, rank == 2]
    assert set(window.payload_sidecar()) == ({0} if rank == 0 else {1} if rank == 2 else set())


class _IdentityEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(WIDTH))

    def forward(self, value):
        return self.proj(value)


class _RuntimeAdapter:
    payload_width = WIDTH
    spatial_merge_size = 2

    def __init__(self, positions=IMAGE_POSITIONS):
        self.positions = tuple(positions)
        self.output_grads = []
        self.output_grad_events = []

    def get_batch(self, iterator):
        next(iterator)
        items = []
        pixels = []
        for item_id, position in enumerate(self.positions):
            items.append(
                CapturedVisionItem(
                    sample_id=0,
                    image_ordinal=item_id,
                    grid_thw=(1, 2, 2),
                    payload_row_start=4 * item_id,
                    payload_rows=4,
                    decoder_positions=(position,),
                )
            )
            pixels.append(torch.full((4, WIDTH), float(item_id + 1), device="cuda"))
        input_ids = torch.zeros(1, SEQ, dtype=torch.long, device="cuda")
        input_ids[0, list(self.positions)] = IMAGE_TOKEN_ID
        return CapturedMicrobatch(
            decoder_packed_seq_params=None,
            decoder_input_shape=(1, SEQ),
            vision_items=tuple(items),
            flat_pixel_payload=torch.cat(pixels),
            model_payload=MappingProxyType({"input_ids": input_ids}),
        )

    def estimate_cost(self, item):
        return item.payload_rows

    def build_encoder(self, model_config, *, pg_collection):
        del pg_collection
        return _IdentityEncoder(model_config)

    def encode(self, encoder, payload, layout):
        rows = [
            encoder(
                payload[segment.payload_row_start : segment.payload_row_start + segment.output_rows]
            )
            for segment in layout.segments
        ]
        output = torch.cat(rows) if rows else payload[:0]
        item_ids = tuple(segment.global_item_id for segment in layout.segments)

        def _record_grad(grad):
            value = grad.detach().clone()
            self.output_grads.append(value)
            self.output_grad_events.append((item_ids, value))

        if output.requires_grad:
            output.register_hook(_record_grad)
        return output


class _RecordingAllocator(DirectBufferAllocator):
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


class _FailingAllocator(_RecordingAllocator):
    def __init__(self, fail_tag):
        super().__init__()
        self.fail_tag = fail_tag

    def acquire(self, *, rows, width, dtype, device, tag):
        if tag == self.fail_tag:
            raise RuntimeError(f"injected allocator failure for {tag}")
        return super().acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)


class _RecordingBridge(ModalityBridge):
    def __init__(self, allocator):
        super().__init__(allocator)
        self.calls = []

    def exchange_all_to_all(self, ledger, local_tensors, **kwargs):
        keys = tuple(sorted(local_tensors, key=lambda key: (key.global_item_id, key.slice_id)))
        self.calls.append((ledger.phase, keys))
        return super().exchange_all_to_all(ledger, local_tensors, **kwargs)


def _build_runtime(
    *,
    decoder_cp_routing="full_leaf",
    positions=IMAGE_POSITIONS,
    encoder_max_payload_rows=None,
    plan_check_interval=1,
    allocator=None,
    bridge=None,
):
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(MdpRankSpec(world_size=4, tp=2, pp=1, cp=2, ep=1, encoder_cp=1))
    view = rank_map.view(rank)
    groups = install_mdp_process_groups(
        rank_map,
        group_registry=MdpGroupRegistry(),
        decoder_tp_group=parallel_state.get_tensor_model_parallel_group(),
    )
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
    adapter = _RuntimeAdapter(positions)
    config = MdpConfig(
        enable=True,
        decoder_cp_routing=decoder_cp_routing,
        encoder_max_payload_rows=encoder_max_payload_rows,
        plan_check_interval=plan_check_interval,
    )
    encoder_domain = build_encoder_domain(
        adapter=adapter,
        model_config=TransformerConfig(
            num_layers=1,
            hidden_size=WIDTH,
            num_attention_heads=1,
            calculate_per_token_loss=True,
            use_cpu_initialization=True,
        ),
        mdp_config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False, overlap_param_gather=False
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1.0e-3, use_distributed_optimizer=True, clip_grad=1.0
        ),
        encoder_pgs=encoder_pgs,
        wrap_mixed_precision=False,
    )
    allocator = allocator or _RecordingAllocator()
    bridge = bridge or _RecordingBridge(allocator)
    runtime = MdpRuntime(
        config=config,
        rank_map=rank_map,
        rank_view=view,
        process_groups=groups,
        adapter=adapter,
        encoder_domain=encoder_domain,
        planner=MdpPlanner(view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()),
        bridge=bridge,
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=WIDTH,
        params_dtype=torch.float32,
        num_vpp_chunks=1,
    )
    return runtime, allocator, bridge


def _visual_weights(slice_id):
    column_scale = torch.linspace(0.5, 1.5, WIDTH, device="cuda")
    weights = torch.zeros(len(IMAGE_POSITIONS), WIDTH, device="cuda")
    for row, owner in enumerate(ROW_OWNER):
        if owner == slice_id:
            weights[row] = (row + 1) * column_scale
    return weights


def _native_bshd_owner_by_position():
    """Bind the compact oracle to MCore's native zigzag index helper."""
    cu_seqlens = torch.tensor((0, SEQ), dtype=torch.long)
    owner = {}
    for cp_rank in range(2):
        for position in get_thd_context_parallel_rank_indices(
            cu_seqlens, 2, cp_rank, "zigzag"
        ).tolist():
            owner[int(position)] = cp_rank
    assert set(owner) == set(range(SEQ))
    return owner


def _d3_weights(slice_id, positions):
    owner = _native_bshd_owner_by_position()
    column_scale = torch.linspace(0.5, 1.5, WIDTH, device="cuda")
    weights = torch.zeros(len(positions), WIDTH, device="cuda")
    for item_id, position in enumerate(positions):
        if owner[position] == slice_id:
            weights[item_id] = (item_id + 1) * column_scale
    return weights


def _drive_routing_decoder(runtime, *, routing, positions):
    """Direct leaf loss isolates transport while still using native TP autograd."""
    leaf = runtime.storage.get_leaf(0)
    assert leaf is not None
    slice_id = runtime._local_decoder_slice_id()
    full_weights = _d3_weights(slice_id, positions)
    if routing == "full_leaf":
        row_item_ids = tuple(range(len(positions)))
        expected_leaf_grad = full_weights
    else:
        local_slice = runtime.decoder_cp_microbatch_slice(0)
        assert local_slice is not None
        row_item_ids = tuple(
            item.global_item_id for item in local_slice.items for _ in item.source_row_ids
        )
        expected_leaf_grad = full_weights[list(row_item_ids)]

    expected_values = torch.tensor(
        tuple(item_id + 1 for item_id in row_item_ids), dtype=leaf.dtype, device=leaf.device
    ).view(-1, 1)
    torch.testing.assert_close(leaf, expected_values.expand_as(leaf), rtol=0.0, atol=0.0)

    replicated = tensor_parallel.copy_to_tensor_model_parallel_region(leaf)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    shard = WIDTH // 2
    start = tp_rank * shard
    stop = start + shard
    local_loss = (replicated[:, start:stop] * expected_leaf_grad[:, start:stop]).sum()
    local_loss.backward()
    assert leaf.grad is not None
    torch.testing.assert_close(leaf.grad, expected_leaf_grad, rtol=0.0, atol=0.0)

    loss = local_loss.detach().clone()
    torch.distributed.all_reduce(loss, group=parallel_state.get_tensor_model_parallel_group())
    expected_loss = (leaf.detach() * expected_leaf_grad).sum()
    torch.testing.assert_close(loss, expected_loss, rtol=0.0, atol=0.0)
    return loss, leaf.grad.detach().clone(), row_item_ids


def _producer_grad_by_item(events):
    result = {}
    for item_ids, grad in events:
        assert grad.shape[0] == len(item_ids)
        for row, item_id in enumerate(item_ids):
            assert item_id not in result
            result[item_id] = grad[row].clone()
    return result


def _run_d3_routing_mode(routing, *, positions=D3_POSITIONS):
    runtime, allocator, bridge = _build_runtime(
        decoder_cp_routing=routing, positions=positions, encoder_max_payload_rows=4
    )
    replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=False)
    next(replay[0])
    ledgers = {
        phase: runtime._iter_ledgers[phase]
        for phase in (BridgePhase.PIXEL, BridgePhase.EMBEDDING, BridgePhase.GRADIENT)
    }
    loss, leaf_grad, row_item_ids = _drive_routing_decoder(
        runtime, routing=routing, positions=positions
    )
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    return SimpleNamespace(
        runtime=runtime,
        allocator=allocator,
        bridge=bridge,
        ledgers=ledgers,
        loss=loss,
        leaf_grad=leaf_grad,
        row_item_ids=row_item_ids,
        producer_grads=_producer_grad_by_item(runtime.adapter.output_grad_events),
        metrics=runtime.last_iteration_metrics(),
    )


def _assert_tag_released_once(allocator, *tags):
    matched = [(tag, base) for tag, base in allocator.acquired if tag in tags]
    assert matched
    for _, base in matched:
        assert sum(released is base for released in allocator.released) == 1


def _run_chunked_route_product_mode(routing):
    """Observe one two-row encoder chunk per physical producer without local asserts."""
    runtime, allocator, bridge = _build_runtime(
        decoder_cp_routing=routing, positions=D3_POSITIONS, encoder_max_payload_rows=None
    )
    replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=False)
    next(replay[0])
    leaf = runtime.storage.get_leaf(0)
    slice_id = runtime._local_decoder_slice_id()
    owned_item_ids = ((0, 1, 6, 7), (2, 3, 4, 5))[slice_id]
    row_item_ids = D3_POSITIONS if routing == "full_leaf" else owned_item_ids

    column_scale = torch.linspace(0.5, 1.5, WIDTH, device="cuda")
    expected_grad = torch.zeros_like(leaf)
    for row, item_id in enumerate(row_item_ids):
        if routing == "cp_local" or item_id in owned_item_ids:
            expected_grad[row] = (item_id + 1) * column_scale
    replicated = tensor_parallel.copy_to_tensor_model_parallel_region(leaf)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    shard = WIDTH // 2
    start = tp_rank * shard
    stop = start + shard
    (replicated[:, start:stop] * expected_grad[:, start:stop]).sum().backward()

    observation = {
        "row_item_ids": tuple(row_item_ids),
        "leaf": leaf.detach().cpu(),
        "leaf_grad": leaf.grad.detach().cpu(),
        "ledgers": {
            phase.value: tuple(
                (
                    entry.src_global_rank,
                    entry.dst_global_rank,
                    entry.key.global_item_id,
                    entry.key.slice_id,
                    entry.element_count,
                )
                for entry in runtime._iter_ledgers[phase].entries
            )
            for phase in (BridgePhase.EMBEDDING, BridgePhase.GRADIENT)
        },
    }
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    observation.update(
        producer_events=tuple(
            (item_ids, grad.cpu()) for item_ids, grad in runtime.adapter.output_grad_events
        ),
        bridge_calls=tuple(
            (phase.value, tuple((key.global_item_id, key.slice_id) for key in keys))
            for phase, keys in bridge.calls
        ),
        allocator_outstanding=allocator._outstanding,
        stored_leaf_count=len(runtime.storage._leaves),
        bridge_in_flight=runtime.bridge._in_flight,
        state=runtime.state.value,
    )
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, observation)
    return tuple(gathered)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_multi_source_multi_endpoint_chunked_forward_and_reverse_are_exact():
    """Four producers reconstruct two encoder rows each in original LPT order."""
    column_scale = torch.linspace(0.5, 1.5, WIDTH)
    for routing in ("full_leaf", "cp_local"):
        observations = _run_chunked_route_product_mode(routing)
        for rank, observation in enumerate(observations):
            slice_id = rank // 2
            owned_item_ids = ((0, 1, 6, 7), (2, 3, 4, 5))[slice_id]
            expected_item_ids = D3_POSITIONS if routing == "full_leaf" else owned_item_ids
            assert observation["row_item_ids"] == expected_item_ids
            expected_leaf = torch.tensor(
                tuple(item_id + 1 for item_id in expected_item_ids), dtype=torch.float32
            ).view(-1, 1)
            torch.testing.assert_close(
                observation["leaf"], expected_leaf.expand(-1, WIDTH), rtol=0.0, atol=0.0
            )
            expected_leaf_grad = torch.zeros(len(expected_item_ids), WIDTH)
            for row, item_id in enumerate(expected_item_ids):
                if routing == "cp_local" or item_id in owned_item_ids:
                    expected_leaf_grad[row] = (item_id + 1) * column_scale
            torch.testing.assert_close(
                observation["leaf_grad"], expected_leaf_grad, rtol=0.0, atol=0.0
            )

            assert len(observation["producer_events"]) == 1
            producer_item_ids, producer_grad = observation["producer_events"][0]
            assert producer_item_ids == (rank, rank + 4)
            expected_producer_grad = torch.stack(
                ((rank + 1) * column_scale, (rank + 5) * column_scale)
            )
            torch.testing.assert_close(producer_grad, expected_producer_grad, rtol=0.0, atol=0.0)

            gradient_call = next(
                keys for phase, keys in observation["bridge_calls"] if phase == "gradient"
            )
            if rank in (0, 2):
                source_item_ids = D3_POSITIONS if routing == "full_leaf" else owned_item_ids
                assert gradient_call == tuple((item_id, slice_id) for item_id in source_item_ids)
            else:
                assert gradient_call == ()
            assert observation["allocator_outstanding"] == 0
            assert observation["stored_leaf_count"] == 0
            assert not observation["bridge_in_flight"]
            assert observation["state"] == MdpRuntimeState.EMPTY.value

        entries = observations[0]["ledgers"]
        for phase in ("embedding", "gradient"):
            assert any(src == dst for src, dst, *_ in entries[phase])
            assert any(src != dst for src, dst, *_ in entries[phase])


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_asymmetric_p3_preparation_failure_converges_before_embedding_exchange(monkeypatch):
    """A rank-local allocation error must not let peers enter the P3 collective."""
    runtime, allocator, bridge = _build_runtime(
        decoder_cp_routing="cp_local", positions=D3_POSITIONS, encoder_max_payload_rows=None
    )
    rank = torch.distributed.get_rank()
    original_acquire = allocator.acquire

    def _fail_one_rank(*, rows, width, dtype, device, tag):
        if rank == 1 and tag == "embedding_compact_staging":
            raise RuntimeError("injected rank-local P3 preparation failure")
        return original_acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)

    monkeypatch.setattr(allocator, "acquire", _fail_one_rank)
    original_exchange = bridge.exchange_all_to_all
    embedding_exchange_calls = 0

    def _intercept_embedding_exchange(ledger, local_tensors, **kwargs):
        nonlocal embedding_exchange_calls
        if ledger.phase is BridgePhase.EMBEDDING:
            embedding_exchange_calls += 1
            raise RuntimeError("unsafe peer entered P3 embedding exchange")
        return original_exchange(ledger, local_tensors, **kwargs)

    monkeypatch.setattr(bridge, "exchange_all_to_all", _intercept_embedding_exchange)
    try:
        runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=True)
    except BaseException as error:
        error_observation = (type(error).__name__, str(error))
    else:
        error_observation = (None, None)

    local = {
        "error": error_observation,
        "embedding_exchange_calls": embedding_exchange_calls,
        "allocator_outstanding": allocator._outstanding,
        "stored_leaf_count": len(runtime.storage._leaves),
        "bridge_in_flight": runtime.bridge._in_flight,
    }
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)

    assert gathered[1]["error"] == ("RuntimeError", "injected rank-local P3 preparation failure")
    for peer_rank in (0, 2, 3):
        error_type, message = gathered[peer_rank]["error"]
        assert error_type == "MdpStateError"
        assert "P3 embedding preparation failed on a planning-group peer" in message
    assert all(observation["embedding_exchange_calls"] == 0 for observation in gathered)
    assert all(observation["allocator_outstanding"] == 0 for observation in gathered)
    assert all(observation["stored_leaf_count"] == 0 for observation in gathered)
    assert all(not observation["bridge_in_flight"] for observation in gathered)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_asymmetric_p5_preparation_failure_converges_before_tp_collapse(monkeypatch):
    """A rank-local regroup error must not let peers enter a TP collective."""
    runtime, allocator, _ = _prepare_compact_training_failure_runtime()
    rank = torch.distributed.get_rank()
    original_acquire = allocator.acquire

    def _fail_one_rank(*, rows, width, dtype, device, tag):
        if rank == 1 and tag == "grad_compact_scratch":
            raise RuntimeError("injected rank-local P5 preparation failure")
        return original_acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)

    monkeypatch.setattr(allocator, "acquire", _fail_one_rank)
    original_broadcast = torch.distributed.broadcast
    tp_broadcast_calls = 0

    def _intercept_tp_broadcast(tensor, src, group, **kwargs):
        nonlocal tp_broadcast_calls
        if group is runtime.process_groups.decoder_tp_group:
            tp_broadcast_calls += 1
            raise RuntimeError("unsafe peer entered P5 TP gradient collapse")
        return original_broadcast(tensor, src=src, group=group, **kwargs)

    monkeypatch.setattr(torch.distributed, "broadcast", _intercept_tp_broadcast)
    try:
        runtime.end_iteration()
    except BaseException as error:
        error_observation = (type(error).__name__, str(error))
    else:
        error_observation = (None, None)

    local = {
        "error": error_observation,
        "tp_broadcast_calls": tp_broadcast_calls,
        "allocator_outstanding": allocator._outstanding,
        "stored_leaf_count": len(runtime.storage._leaves),
        "bridge_in_flight": runtime.bridge._in_flight,
    }
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)

    assert gathered[1]["error"] == ("RuntimeError", "injected rank-local P5 preparation failure")
    for peer_rank in (0, 2, 3):
        error_type, message = gathered[peer_rank]["error"]
        assert error_type == "MdpStateError"
        assert "P5 gradient preparation failed on a planning-group peer" in message
    assert all(observation["tp_broadcast_calls"] == 0 for observation in gathered)
    assert all(observation["allocator_outstanding"] == 0 for observation in gathered)
    assert all(observation["stored_leaf_count"] == 0 for observation in gathered)
    assert all(not observation["bridge_in_flight"] for observation in gathered)


@pytest.mark.parametrize("phase", (BridgePhase.EMBEDDING, BridgePhase.GRADIENT))
@pytest.mark.parametrize("failure_point", ("recv_acquire", "pack", "missing_dest"))
@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_asymmetric_bridge_preparation_failure_converges_before_a2a(
    monkeypatch, phase, failure_point
):
    """Bridge allocations and packing must converge before the real A2A."""
    if phase is BridgePhase.EMBEDDING:
        runtime, allocator, bridge = _build_runtime(
            decoder_cp_routing="cp_local", positions=D3_POSITIONS
        )
    else:
        runtime, allocator, bridge = _prepare_compact_training_failure_runtime()
    rank = torch.distributed.get_rank()
    injected_rank = [1 if phase is BridgePhase.EMBEDDING else 0]
    active_phase = [None]

    original_exchange = bridge.exchange_all_to_all

    def _track_phase(ledger, local_tensors, **kwargs):
        active_phase[0] = ledger.phase
        try:
            if failure_point == "missing_dest":
                injected_rank[0] = min(entry.dst_global_rank for entry in ledger.entries)
                if rank == injected_rank[0]:
                    missing_key = next(
                        entry.key
                        for entry in ledger.entries
                        if entry.dst_global_rank == injected_rank[0]
                    )
                    kwargs["dest_views"] = dict(kwargs["dest_views"])
                    kwargs["dest_views"].pop(missing_key, None)
            return original_exchange(ledger, local_tensors, **kwargs)
        finally:
            active_phase[0] = None

    monkeypatch.setattr(bridge, "exchange_all_to_all", _track_phase)
    original_acquire = allocator.acquire

    def _inject_recv_failure(*, rows, width, dtype, device, tag):
        if (
            failure_point == "recv_acquire"
            and rank == injected_rank[0]
            and active_phase[0] is phase
            and tag == "bridge_a2a_recv"
        ):
            raise RuntimeError(f"injected {phase.value} bridge recv allocation failure")
        return original_acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)

    monkeypatch.setattr(allocator, "acquire", _inject_recv_failure)
    original_foreach_copy = bridge_module.torch._foreach_copy_

    def _inject_pack_failure(destinations, sources):
        if failure_point == "pack" and rank == injected_rank[0] and active_phase[0] is phase:
            raise RuntimeError(f"injected {phase.value} bridge pack failure")
        return original_foreach_copy(destinations, sources)

    monkeypatch.setattr(bridge_module.torch, "_foreach_copy_", _inject_pack_failure)
    original_a2a = torch.distributed.all_to_all_single
    a2a_calls = 0

    def _intercept_a2a(*args, **kwargs):
        nonlocal a2a_calls
        if active_phase[0] is phase:
            a2a_calls += 1
            raise RuntimeError(f"unsafe peer entered {phase.value} A2A")
        return original_a2a(*args, **kwargs)

    monkeypatch.setattr(torch.distributed, "all_to_all_single", _intercept_a2a)
    try:
        if phase is BridgePhase.EMBEDDING:
            runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=True)
        else:
            runtime.end_iteration()
    except BaseException as error:
        error_observation = (type(error).__name__, str(error))
    else:
        error_observation = (None, None)

    bridge_bases = [
        (tag, base)
        for tag, base in allocator.acquired
        if tag in ("bridge_a2a_send", "bridge_a2a_recv")
    ]
    local = {
        "error": error_observation,
        "a2a_calls": a2a_calls,
        "bridge_bases": tuple(
            (tag, sum(released is base for released in allocator.released))
            for tag, base in bridge_bases
        ),
        "allocator_outstanding": allocator._outstanding,
        "stored_leaf_count": len(runtime.storage._leaves),
        "bridge_in_flight": runtime.bridge._in_flight,
    }
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)

    if failure_point == "missing_dest":
        error_type, message = gathered[injected_rank[0]]["error"]
        assert error_type == "MdpBridgeError"
        assert "requires a caller-owned destination view" in message
    else:
        expected = f"injected {phase.value} bridge "
        expected += "recv allocation failure" if failure_point == "recv_acquire" else "pack failure"
        assert gathered[injected_rank[0]]["error"] == ("RuntimeError", expected)
    for peer_rank, observation in enumerate(gathered):
        if peer_rank == injected_rank[0]:
            continue
        error_type, message = observation["error"]
        assert error_type == "MdpStateError"
        assert f"{phase.value} bridge preparation failed" in message
    assert all(observation["a2a_calls"] == 0 for observation in gathered)
    assert all(
        all(release_count == 1 for _, release_count in observation["bridge_bases"])
        for observation in gathered
    )
    assert all(observation["allocator_outstanding"] == 0 for observation in gathered)
    assert all(observation["stored_leaf_count"] == 0 for observation in gathered)
    assert all(not observation["bridge_in_flight"] for observation in gathered)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_asymmetric_compact_reconstruction_index_failure_precedes_tp_and_grad_exchange(monkeypatch):
    runtime, allocator, bridge = _prepare_compact_training_failure_runtime()
    rank = torch.distributed.get_rank()
    original_tensor = runtime_module.torch.tensor

    def _fail_one_rank(data, *args, **kwargs):
        if rank == 1 and isinstance(data, tuple) and kwargs.get("dtype") is torch.long:
            raise RuntimeError("injected rank-local reconstruction-index failure")
        return original_tensor(data, *args, **kwargs)

    monkeypatch.setattr(runtime_module.torch, "tensor", _fail_one_rank)
    original_broadcast = torch.distributed.broadcast
    tp_broadcast_calls = 0

    def _track_broadcast(tensor, src, group, **kwargs):
        nonlocal tp_broadcast_calls
        if group is runtime.process_groups.decoder_tp_group:
            tp_broadcast_calls += 1
        return original_broadcast(tensor, src=src, group=group, **kwargs)

    monkeypatch.setattr(torch.distributed, "broadcast", _track_broadcast)
    original_exchange = bridge.exchange_all_to_all
    grad_exchange_calls = 0

    def _track_exchange(ledger, local_tensors, **kwargs):
        nonlocal grad_exchange_calls
        if ledger.phase is BridgePhase.GRADIENT:
            grad_exchange_calls += 1
        return original_exchange(ledger, local_tensors, **kwargs)

    monkeypatch.setattr(bridge, "exchange_all_to_all", _track_exchange)
    backward_calls = 0
    original_backward = runtime._handle.backward

    def _stop_peer_after_exchange(chunk_grads):
        nonlocal backward_calls
        backward_calls += 1
        if rank != 1:
            raise RuntimeError("unsafe peer passed reconstruction preparation")
        return original_backward(chunk_grads)

    monkeypatch.setattr(runtime._handle, "backward", _stop_peer_after_exchange)
    try:
        runtime.end_iteration()
    except BaseException as error:
        error_observation = (type(error).__name__, str(error))
    else:
        error_observation = (None, None)

    local = {
        "error": error_observation,
        "tp_broadcast_calls": tp_broadcast_calls,
        "grad_exchange_calls": grad_exchange_calls,
        "backward_calls": backward_calls,
        "allocator_outstanding": allocator._outstanding,
        "stored_leaf_count": len(runtime.storage._leaves),
    }
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)

    assert gathered[1]["error"] == (
        "RuntimeError",
        "injected rank-local reconstruction-index failure",
    )
    for peer_rank in (0, 2, 3):
        error_type, message = gathered[peer_rank]["error"]
        assert error_type == "MdpStateError"
        assert "P5 gradient preparation failed on a planning-group peer" in message
    assert all(observation["tp_broadcast_calls"] == 0 for observation in gathered)
    assert all(observation["grad_exchange_calls"] == 0 for observation in gathered)
    assert all(observation["backward_calls"] == 0 for observation in gathered)
    assert all(observation["allocator_outstanding"] == 0 for observation in gathered)
    assert all(observation["stored_leaf_count"] == 0 for observation in gathered)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_packed_pixel_bases_are_released_after_training_and_eval():
    trained = _run_d3_routing_mode("full_leaf")
    _assert_tag_released_once(trained.allocator, "packed_pixels")

    runtime, allocator, _ = _build_runtime(decoder_cp_routing="full_leaf")
    replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=True)
    next(replay[0])
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    _assert_tag_released_once(allocator, "packed_pixels")


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_packed_pixel_bases_are_released_after_encoder_backward_failure(monkeypatch):
    runtime, allocator, _ = _prepare_compact_training_failure_runtime()
    assert runtime._handle is not None

    def _fail_backward(_chunk_grads):
        raise RuntimeError("injected encoder backward failure")

    monkeypatch.setattr(runtime._handle, "backward", _fail_backward)
    with pytest.raises(RuntimeError, match="injected encoder backward failure"):
        runtime.end_iteration()
    _assert_tag_released_once(allocator, "packed_pixels")


def _prepare_compact_training_failure_runtime():
    runtime, allocator, bridge = _build_runtime(
        decoder_cp_routing="cp_local", positions=D3_POSITIONS, encoder_max_payload_rows=4
    )
    replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=False)
    next(replay[0])
    _drive_routing_decoder(runtime, routing="cp_local", positions=D3_POSITIONS)
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    return runtime, allocator, bridge


def _prepare_full_leaf_training_failure_runtime():
    runtime, allocator, bridge = _build_runtime(
        decoder_cp_routing="full_leaf", positions=D3_POSITIONS, encoder_max_payload_rows=4
    )
    replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=False)
    next(replay[0])
    _drive_routing_decoder(runtime, routing="full_leaf", positions=D3_POSITIONS)
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    return runtime, allocator, bridge


def _drive_tp_decoder(runtime, record):
    """Small native-TP autograd oracle: TP shards columns, backward reduces input."""
    leaf = runtime.storage.get_leaf(0)
    assert leaf is not None
    expected_values = torch.arange(1, 5, device="cuda", dtype=leaf.dtype).view(-1, 1)
    torch.testing.assert_close(leaf, expected_values.expand_as(leaf), rtol=0.0, atol=0.0)

    slice_id = runtime._local_decoder_slice_id()
    expected_leaf_grad = _visual_weights(slice_id)
    input_ids = record.model_payload["input_ids"]
    text = torch.zeros(SEQ, 1, WIDTH, device="cuda")
    combined = MultimodalModel._scatter_vision_embeddings(
        SimpleNamespace(
            config=SimpleNamespace(sequence_parallel=False), image_token_id=IMAGE_TOKEN_ID
        ),
        input_ids,
        text,
        leaf,
    )
    # The copy mapping is identity in forward and all-reduces its input gradient
    # in backward, matching a native tensor-parallel layer's replicated input.
    combined = tensor_parallel.copy_to_tensor_model_parallel_region(combined)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    shard = WIDTH // 2
    start = tp_rank * shard
    end = start + shard
    local_scale = torch.linspace(0.5, 1.5, WIDTH, device="cuda")[start:end].clone()
    local_scale.requires_grad_(True)
    token_weight = torch.zeros(SEQ, 1, shard, device="cuda")
    for row, position in enumerate(IMAGE_POSITIONS):
        if ROW_OWNER[row] == slice_id:
            token_weight[position] = float(row + 1)
    local_loss = (combined[:, :, start:end] * token_weight * local_scale.view(1, 1, -1)).sum()
    local_loss.backward()

    torch.testing.assert_close(leaf.grad, expected_leaf_grad, rtol=0.0, atol=0.0)
    expected_scale_grad = torch.zeros_like(local_scale)
    for row, owner in enumerate(ROW_OWNER):
        if owner == slice_id:
            expected_scale_grad.add_(float((row + 1) ** 2))
    torch.testing.assert_close(local_scale.grad, expected_scale_grad, rtol=0.0, atol=0.0)

    loss_sum = local_loss.detach().clone()
    torch.distributed.all_reduce(loss_sum, group=parallel_state.get_tensor_model_parallel_group())
    expected_loss = (leaf.detach() * expected_leaf_grad).sum()
    torch.testing.assert_close(loss_sum, expected_loss, rtol=0.0, atol=0.0)
    return expected_leaf_grad


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_tp2_cp2_full_leaf_forward_backward_matches_analytic_reference():
    runtime, allocator, bridge = _build_runtime()
    replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=False)
    record = next(replay[0])

    plan = runtime._plan
    item_count = len(IMAGE_POSITIONS)
    assert len(runtime._iter_ledgers[BridgePhase.PIXEL].entries) == item_count
    assert len(runtime._iter_ledgers[BridgePhase.EMBEDDING].entries) == item_count * 2
    assert len(runtime._iter_ledgers[BridgePhase.GRADIENT].entries) == item_count * 2
    assert len(runtime._iter_ledgers[BridgePhase.GRADIENT].entries) != item_count * 2 * 2
    assert plan.schema_version == PLAN_SCHEMA_VERSION

    expected_local_leaf_grad = _drive_tp_decoder(runtime, record)
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    assert len(runtime.adapter.output_grads) == 1
    producer_item = runtime.rank_view.my_worker_id
    expected_producer_grad = _visual_weights(0)[producer_item] + _visual_weights(1)[producer_item]
    torch.testing.assert_close(
        runtime.adapter.output_grads[0][0], expected_producer_grad, rtol=0.0, atol=0.0
    )
    assert torch.count_nonzero(expected_local_leaf_grad) > 0

    grad_call = next(keys for phase, keys in bridge.calls if phase is BridgePhase.GRADIENT)
    if runtime.rank_view.is_decoder_endpoint:
        assert len(grad_call) == item_count
        assert {key.slice_id for key in grad_call} == {runtime._local_decoder_slice_id()}
    else:
        assert grad_call == ()

    leaf_bases = [base for tag, base in allocator.acquired if tag == "leaf"]
    assert len(leaf_bases) == 1
    assert sum(released is leaf_bases[0] for released in allocator.released) == 1


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_tp2_cp2_cp_local_matches_full_leaf_and_reduces_transport_and_storage():
    full = _run_d3_routing_mode("full_leaf")
    compact = _run_d3_routing_mode("cp_local")

    torch.testing.assert_close(compact.loss, full.loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        compact.leaf_grad, full.leaf_grad[list(compact.row_item_ids)], rtol=0.0, atol=0.0
    )
    assert compact.producer_grads.keys() == full.producer_grads.keys()
    for item_id in compact.producer_grads:
        torch.testing.assert_close(
            compact.producer_grads[item_id], full.producer_grads[item_id], rtol=0.0, atol=0.0
        )

    assert (
        compact.ledgers[BridgePhase.PIXEL].total_bytes
        == full.ledgers[BridgePhase.PIXEL].total_bytes
    )
    # Independent fixture oracle: eight float32 rows of width eight are routed
    # once by cp_local and replicated to both C=2 endpoints by full_leaf.
    fixture_rows = 8
    decoder_cp = 2
    float32_bytes = 4
    expected_compact_bytes = fixture_rows * WIDTH * float32_bytes
    expected_full_bytes = decoder_cp * expected_compact_bytes
    assert D3_POSITIONS == tuple(range(fixture_rows))
    for phase in (BridgePhase.EMBEDDING, BridgePhase.GRADIENT):
        assert compact.ledgers[phase].total_bytes == expected_compact_bytes
        assert full.ledgers[phase].total_bytes == expected_full_bytes
        assert full.ledgers[phase].total_bytes == decoder_cp * compact.ledgers[phase].total_bytes
    assert len(compact.ledgers[BridgePhase.EMBEDDING].entries) == fixture_rows
    assert len(full.ledgers[BridgePhase.EMBEDDING].entries) == decoder_cp * fixture_rows
    assert compact.metrics.endpoint_leaf_valid_rows == fixture_rows // decoder_cp
    assert compact.metrics.endpoint_leaf_capacity_rows == fixture_rows // decoder_cp
    assert full.metrics.endpoint_leaf_valid_rows == fixture_rows
    assert full.metrics.endpoint_leaf_capacity_rows == fixture_rows

    full_leaf_bases = [base for tag, base in full.allocator.acquired if tag == "leaf"]
    compact_leaf_bases = [base for tag, base in compact.allocator.acquired if tag == "leaf"]
    assert len(full_leaf_bases) == len(compact_leaf_bases) == 1
    assert tuple(full_leaf_bases[0].shape) == (fixture_rows, WIDTH)
    assert tuple(compact_leaf_bases[0].shape) == (fixture_rows // decoder_cp, WIDTH)

    compact_tags = [tag for tag, _ in compact.allocator.acquired]
    assert compact_tags.count("embedding_compact_staging") == 1
    assert compact_tags.count("grad_compact_scratch") == 2
    assert compact_tags.count("grad_regroup") == 2
    assert "grad_endpoint_slice" not in compact_tags
    for tag, base in compact.allocator.acquired:
        if tag in ("embedding_compact_staging", "grad_compact_scratch", "grad_regroup"):
            assert sum(released is base for released in compact.allocator.released) == 1

    # Both native TP peers own the same compact CP leaf and produce equal grads;
    # only TP0 contributes its C-local rows to the planning-group wire.
    grad_call = next(keys for phase, keys in compact.bridge.calls if phase is BridgePhase.GRADIENT)
    if compact.runtime.rank_view.is_decoder_endpoint:
        assert len(grad_call) == len(compact.row_item_ids)
        assert {key.slice_id for key in grad_call} == {compact.runtime._local_decoder_slice_id()}
    else:
        assert grad_call == ()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_cp_local_zero_row_tp_endpoint_keeps_empty_leaf_graph_and_omits_wire_entries():
    result = _run_d3_routing_mode("cp_local", positions=(0, 1))
    slice_id = result.runtime._local_decoder_slice_id()
    expected_rows = 2 if slice_id == 0 else 0
    assert result.leaf_grad.shape == (expected_rows, WIDTH)
    assert result.row_item_ids == ((0, 1) if slice_id == 0 else ())
    assert len(result.ledgers[BridgePhase.EMBEDDING].entries) == 2
    assert len(result.ledgers[BridgePhase.GRADIENT].entries) == 2
    assert all(entry.key.slice_id == 0 for entry in result.ledgers[BridgePhase.EMBEDDING].entries)
    assert result.metrics.endpoint_leaf_valid_rows == expected_rows


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_cp_local_combined_digest_runs_every_iteration_before_bridge(monkeypatch):
    runtime, _, bridge = _build_runtime(decoder_cp_routing="cp_local", plan_check_interval=97)
    original = runtime_module.assert_consistent_decoder_cp_iteration
    observations = []

    def _record(plan, slice_plan, *, planning_group):
        observations.append((plan.iteration, tuple(bridge.calls)))
        return original(plan, slice_plan, planning_group=planning_group)

    monkeypatch.setattr(runtime_module, "assert_consistent_decoder_cp_iteration", _record)
    for iteration in range(2):
        replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=True)
        next(replay[0])
        runtime.mark_decoder_complete()
        runtime.end_iteration()
        bridge.calls.clear()
    assert observations == [(0, ()), (1, ())]


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_cp_local_combined_digest_mismatch_fails_before_bridge(monkeypatch):
    runtime, _, bridge = _build_runtime(decoder_cp_routing="cp_local")
    original = decoder_cp_module.compute_decoder_cp_iteration_digest

    def _rank_dependent_digest(base_digest, slice_digest):
        digest = bytearray(original(base_digest, slice_digest))
        if torch.distributed.get_rank() == 0:
            digest[0] ^= 1
        return bytes(digest)

    monkeypatch.setattr(
        decoder_cp_module, "compute_decoder_cp_iteration_digest", _rank_dependent_digest
    )
    with pytest.raises(MdpPlanError, match="combined decoder CP plan digest mismatch"):
        runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=True)
    assert bridge.calls == []
    runtime.storage.assert_empty()


@pytest.mark.parametrize("failure_point", ("packing", "bridge"))
@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_cp_local_p3_failure_releases_staging_and_unhanded_leaf(monkeypatch, failure_point):
    runtime, allocator, bridge = _build_runtime(
        decoder_cp_routing="cp_local", positions=D3_POSITIONS, encoder_max_payload_rows=4
    )
    if failure_point == "packing":
        original = runtime_module.torch.index_select

        def _fail_pack(input_tensor, dim, index, *, out=None):
            if out is not None:
                raise RuntimeError("injected compact embedding packing failure")
            return original(input_tensor, dim, index, out=out)

        monkeypatch.setattr(runtime_module.torch, "index_select", _fail_pack)
        message = "injected compact embedding packing failure"
    else:
        original = bridge.exchange_all_to_all

        def _fail_exchange(ledger, local_tensors, **kwargs):
            if ledger.phase is BridgePhase.EMBEDDING:
                raise RuntimeError("injected compact embedding bridge failure")
            return original(ledger, local_tensors, **kwargs)

        monkeypatch.setattr(bridge, "exchange_all_to_all", _fail_exchange)
        message = "injected compact embedding bridge failure"

    with pytest.raises(RuntimeError, match=message):
        runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=True)
    _assert_tag_released_once(allocator, "embedding_compact_staging")
    _assert_tag_released_once(allocator, "leaf")
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_cp_local_p3_allocator_failure_releases_unhanded_leaf():
    allocator = _FailingAllocator("embedding_compact_staging")
    runtime, _, _ = _build_runtime(
        decoder_cp_routing="cp_local",
        positions=D3_POSITIONS,
        encoder_max_payload_rows=4,
        allocator=allocator,
    )
    with pytest.raises(RuntimeError, match="allocator failure for embedding_compact_staging"):
        runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=True)
    _assert_tag_released_once(allocator, "leaf")
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@pytest.mark.parametrize("failure_point", ("bridge", "reconstruction"))
@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_cp_local_p5_failure_releases_scratch_and_regroup(monkeypatch, failure_point):
    runtime, allocator, bridge = _prepare_compact_training_failure_runtime()
    if failure_point == "bridge":
        original = bridge.exchange_all_to_all

        def _fail_exchange(ledger, local_tensors, **kwargs):
            if ledger.phase is BridgePhase.GRADIENT:
                raise RuntimeError("injected compact gradient bridge failure")
            return original(ledger, local_tensors, **kwargs)

        monkeypatch.setattr(bridge, "exchange_all_to_all", _fail_exchange)
        message = "injected compact gradient bridge failure"
    else:
        original = runtime_module.torch.tensor

        def _fail_reconstruction(data, *args, **kwargs):
            if isinstance(data, tuple) and kwargs.get("dtype") is torch.long:
                raise RuntimeError("injected compact gradient reconstruction failure")
            return original(data, *args, **kwargs)

        monkeypatch.setattr(runtime_module.torch, "tensor", _fail_reconstruction)
        message = "injected compact gradient reconstruction failure"

    with pytest.raises(RuntimeError, match=message):
        runtime.end_iteration()
    _assert_tag_released_once(allocator, "grad_compact_scratch", "grad_regroup")
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_cp_local_p5_allocator_failure_releases_regroup_and_leaf():
    allocator = _FailingAllocator("grad_compact_scratch")
    runtime, _, _ = _build_runtime(
        decoder_cp_routing="cp_local",
        positions=D3_POSITIONS,
        encoder_max_payload_rows=4,
        allocator=allocator,
    )
    replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=False)
    next(replay[0])
    _drive_routing_decoder(runtime, routing="cp_local", positions=D3_POSITIONS)
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    with pytest.raises(RuntimeError, match="allocator failure for grad_compact_scratch"):
        runtime.end_iteration()
    _assert_tag_released_once(allocator, "grad_regroup")
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 TP2 x CP2")
def test_full_leaf_p5_partial_endpoint_allocation_failure_releases_exact_bases(monkeypatch):
    runtime, allocator, _ = _prepare_full_leaf_training_failure_runtime()
    original_acquire = allocator.acquire
    endpoint_acquires = 0

    def _fail_second_endpoint_slice(*, rows, width, dtype, device, tag):
        nonlocal endpoint_acquires
        if tag == "grad_endpoint_slice":
            endpoint_acquires += 1
            if endpoint_acquires == 2:
                raise RuntimeError("injected second endpoint-slice allocation failure")
        return original_acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)

    monkeypatch.setattr(allocator, "acquire", _fail_second_endpoint_slice)
    with pytest.raises(RuntimeError, match="injected second endpoint-slice allocation failure"):
        runtime.end_iteration()

    assert endpoint_acquires == 2
    _assert_tag_released_once(
        allocator, "grad_endpoint_slice", "grad_regroup", "leaf", "packed_pixels"
    )
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs world4 native TP2")
def test_sequence_parallel_scatter_produces_identical_full_leaf_gradients():
    """The existing gather/scatter path returns a full equal leaf grad on TP peers."""
    leaf = torch.arange(1, 5, device="cuda", dtype=torch.float32).view(-1, 1)
    leaf = leaf.expand(-1, WIDTH).clone().requires_grad_(True)
    input_ids = torch.zeros(1, SEQ, dtype=torch.long, device="cuda")
    input_ids[0, list(IMAGE_POSITIONS)] = IMAGE_TOKEN_ID
    full_text = torch.zeros(SEQ, 1, WIDTH, device="cuda")
    local_text = tensor_parallel.scatter_to_sequence_parallel_region(full_text)
    local = MultimodalModel._scatter_vision_embeddings(
        SimpleNamespace(
            config=SimpleNamespace(sequence_parallel=True), image_token_id=IMAGE_TOKEN_ID
        ),
        input_ids,
        local_text,
        leaf,
    )
    full_weight = torch.arange(1, SEQ + 1, device="cuda", dtype=torch.float32).view(SEQ, 1, 1)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    local_weight = full_weight.chunk(2, dim=0)[tp_rank]
    (local * local_weight).sum().backward()
    expected = full_weight[list(IMAGE_POSITIONS), 0].expand(-1, WIDTH)
    torch.testing.assert_close(leaf.grad, expected, rtol=0.0, atol=0.0)

    peer = torch.empty_like(leaf.grad)
    source = parallel_state.get_tensor_model_parallel_src_rank()
    if torch.distributed.get_rank() == source:
        peer.copy_(leaf.grad)
    torch.distributed.broadcast(
        peer, src=source, group=parallel_state.get_tensor_model_parallel_group()
    )
    torch.testing.assert_close(leaf.grad, peer, rtol=0.0, atol=0.0)
