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
from megatron.core.mdp.encoder import build_encoder_domain, build_encoder_pg_collection
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem, MdpEncoderOutput
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.schedule import wrap_forward_backward
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.optimizer import OptimizerConfig
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(WIDTH))

    def forward(self, x):
        return self.proj(x)


class _StubAdapter:
    """Deterministic capture: two microbatches, three items in mb0, mb1 text-only."""

    payload_width = WIDTH
    output_plane_widths = (WIDTH,)
    spatial_merge_size = MERGE

    def __init__(self, lane):
        self._lane = lane if lane is not None else 0

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
            payload_chunks.append(
                torch.full((rows, WIDTH), _sentinel(self._lane, index), device="cuda")
            )
            payload_start += rows
        return CapturedMicrobatch(
            decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
            vision_items=tuple(items),
            flat_pixel_payload=torch.cat(payload_chunks),
            model_payload=MappingProxyType({"microbatch": mb}),
        )

    def estimate_cost(self, item):
        return item.payload_rows

    def build_encoder(self, model_config, *, pg_collection):
        return _TinyEncoder(model_config)

    def encode(self, encoder, payload, layout):
        pieces = []
        for segment in layout.segments:
            pieces.append(
                encoder(
                    payload[
                        segment.payload_row_start : segment.payload_row_start + segment.output_rows
                    ]
                )
            )
        return torch.cat(pieces) if pieces else payload[:0]


class _MultiPlaneEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.final = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        self.deepstack = torch.nn.Linear(WIDTH, WIDTH // 2, bias=False)
        with torch.no_grad():
            self.final.weight.copy_(torch.eye(WIDTH))
            self.deepstack.weight.copy_(3.0 * torch.eye(WIDTH)[: WIDTH // 2])

    def forward(self, x):
        return self.final(x), self.deepstack(x)


class _MultiPlaneAdapter(_StubAdapter):
    """Final plane first, then one canonical DeepStack sentinel plane."""

    output_plane_widths = (WIDTH, WIDTH // 2)

    def __init__(self, lane):
        super().__init__(lane)
        self.grad_events = []

    def build_encoder(self, model_config, *, pg_collection):
        return _MultiPlaneEncoder(model_config)

    def encode(self, encoder, payload, layout):
        pieces = [[], []]
        for segment in layout.segments:
            outputs = encoder(
                payload[segment.payload_row_start : segment.payload_row_start + segment.output_rows]
            )
            for plane_id, (plane, output) in enumerate(zip(pieces, outputs)):
                output.register_hook(
                    lambda grad, plane_id=plane_id: self.grad_events.append(
                        (plane_id, grad.detach().clone())
                    )
                )
                plane.append(output)
        return MdpEncoderOutput(
            planes=tuple(torch.cat(plane) if plane else payload[:0] for plane in pieces)
        )


class _AliasEvalAdapter(_StubAdapter):
    def encode(self, encoder, payload, layout):
        del encoder
        return payload[: layout.total_output_rows]


class _FailingEncodeAdapter(_StubAdapter):
    def encode(self, encoder, payload, layout):
        if torch.distributed.get_rank() in (0, 1):
            raise _InjectedOwnershipFailure("injected encoder failure")
        return super().encode(encoder, payload, layout)


class _InvalidShapeAdapter(_StubAdapter):
    def encode(self, encoder, payload, layout):
        output = super().encode(encoder, payload, layout)
        if torch.distributed.get_rank() in (0, 1):
            return output[:-1]
        return output


class _InjectedOwnershipFailure(RuntimeError):
    pass


class _IdentityAllocator(DirectBufferAllocator):
    def __init__(self, *, fail_tag=None, fail_nth=None):
        super().__init__()
        self.fail_tag = fail_tag
        self.fail_nth = fail_nth
        self.tag_calls = {}
        self.acquired = []
        self.released = []

    def acquire(self, **kwargs):
        tag = kwargs["tag"]
        self.tag_calls[tag] = self.tag_calls.get(tag, 0) + 1
        if tag == self.fail_tag and self.tag_calls[tag] == self.fail_nth:
            raise _InjectedOwnershipFailure(f"injected {tag} acquire failure")
        base = super().acquire(**kwargs)
        self.acquired.append((tag, base))
        return base

    def release(self, tensor):
        matches = [base for _tag, base in self.acquired if base is tensor]
        if len(matches) != 1:
            raise AssertionError("allocator release must receive one exact acquired base")
        if any(base is tensor for base in self.released):
            raise AssertionError("allocator exact base released twice")
        self.released.append(tensor)
        super().release(tensor)


class _PhaseFailBridge(ModalityBridge):
    def __init__(self, allocator, phase):
        super().__init__(allocator)
        self.phase = phase

    def exchange_all_to_all(self, ledger, *args, **kwargs):
        if ledger.phase is self.phase:
            raise _InjectedOwnershipFailure(f"injected {self.phase.value} exchange failure")
        return super().exchange_all_to_all(ledger, *args, **kwargs)

    def exchange(self, ledger, *args, **kwargs):
        if ledger.phase is self.phase:
            raise _InjectedOwnershipFailure(f"injected {self.phase.value} exchange failure")
        return super().exchange(ledger, *args, **kwargs)


class _AsymmetricPhaseFailBridge(ModalityBridge):
    """Raise rank-specific bridge errors without entering the transport."""

    def __init__(self, allocator, phase):
        super().__init__(allocator)
        self.phase = phase
        self.failed = False

    def exchange_all_to_all(self, ledger, *args, **kwargs):
        if ledger.phase is not self.phase:
            return super().exchange_all_to_all(ledger, *args, **kwargs)
        self.failed = True
        if kwargs["global_rank"] == kwargs["group_ranks"][0]:
            raise _InjectedOwnershipFailure(
                f"injected asymmetric {self.phase.value} exchange failure"
            )
        raise MdpStateError(f"peer observed {self.phase.value} exchange failure")


class _FailingStorage(MdpEmbeddingStorage):
    def put_leaves(self, *args, **kwargs):
        raise _InjectedOwnershipFailure("injected storage handoff failure")


def _build_runtime(
    adapter_type=_StubAdapter,
    *,
    allocator_factory=None,
    bridge_factory=None,
    storage_factory=None,
    mdp_config=None,
):
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1))
    view = rank_map.view(rank)
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
    adapter = adapter_type(view.outer_dp_rank)
    model_config = TransformerConfig(
        num_layers=1,
        hidden_size=WIDTH,
        num_attention_heads=1,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    config = MdpConfig(enable=True) if mdp_config is None else mdp_config
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
    allocator = DirectBufferAllocator() if allocator_factory is None else allocator_factory(view)
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
        bridge=(ModalityBridge(allocator) if bridge_factory is None else bridge_factory(allocator)),
        storage=(
            MdpEmbeddingStorage(allocator)
            if storage_factory is None
            else storage_factory(allocator)
        ),
        allocator=allocator,
        hidden_size=WIDTH,
        params_dtype=torch.float32,
        num_vpp_chunks=1,
    )
    return runtime, view


def _assert_exact_release_for_tags(allocator, tags):
    acquired = [base for tag, base in allocator.acquired if tag in tags]
    released = [base for base in allocator.released if any(base is item for item in acquired)]
    assert len(released) == len(acquired)
    assert {id(base) for base in released} == {id(base) for base in acquired}
    assert len({id(base) for base in released}) == len(released)


def _drive_decoder(runtime, view, replay_iters, *, backward):
    """Consume the replay iterator like the native schedule would."""
    records = [next(replay_iters[0]) for _ in range(2)]
    assert [r.model_payload["microbatch"] for r in records] == [0, 1]
    if view.lane_id is not None:
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


def test_full_training_iteration_and_state_machine():
    runtime, view = _build_runtime(
        allocator_factory=lambda _view: _IdentityAllocator(),
        mdp_config=MdpConfig(enable=True, encoder_max_payload_rows=32),
    )
    assert runtime.state is MdpRuntimeState.EMPTY

    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    assert runtime.state is MdpRuntimeState.DECODER_READY
    expected_embedding_bytes = sum(
        t * (h // MERGE) * (w // MERGE) * WIDTH * torch.float32.itemsize for t, h, w in GRIDS
    )
    embedding_ledger = runtime._iter_ledgers[BridgePhase.EMBEDDING]
    assert len(embedding_ledger.entries) == len(GRIDS)
    assert embedding_ledger.total_bytes == expected_embedding_bytes
    packed_bases = [base for tag, base in runtime.allocator.acquired if tag == "packed_pixels"]
    if view.my_worker_id == view.worker_ids[1]:
        assert len(packed_bases) == 2
    assert not any(
        base is released for base in packed_bases for released in runtime.allocator.released
    )
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
    _assert_exact_release_for_tags(runtime.allocator, {"packed_pixels"})
    assert runtime.allocator._outstanding == 0

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
    runtime, view = _build_runtime(allocator_factory=lambda _view: _IdentityAllocator())
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=True)
    _drive_decoder(runtime, view, replay, backward=False)
    runtime.mark_decoder_complete()  # eval requires no token capture
    runtime.end_iteration()
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    _assert_exact_release_for_tags(runtime.allocator, {"packed_pixels"})
    assert runtime.allocator._outstanding == 0


def test_forward_only_alias_output_releases_exact_packed_base_after_p3():
    runtime, view = _build_runtime(
        _AliasEvalAdapter, allocator_factory=lambda _view: _IdentityAllocator()
    )
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=True)
    records = [next(replay[0]) for _ in range(2)]
    assert [record.microbatch_id for record in records] == [0, 1]
    _assert_exact_release_for_tags(runtime.allocator, {"packed_pixels"})
    if view.lane_id is not None:
        assert runtime.storage.get_leaf(0) is not None
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    assert runtime.allocator._outstanding == 0


def test_second_packed_acquire_failure_is_coordinated_and_releases_first_base():
    def allocator_factory(view):
        return _IdentityAllocator(
            fail_tag="packed_pixels" if view.my_worker_id == view.worker_ids[1] else None,
            fail_nth=2,
        )

    runtime, view = _build_runtime(
        allocator_factory=allocator_factory,
        mdp_config=MdpConfig(enable=True, encoder_max_payload_rows=32),
    )
    with pytest.raises(Exception) as error:
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    if view.my_worker_id == view.worker_ids[1]:
        assert isinstance(error.value, _InjectedOwnershipFailure)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "P2 payload assembly" in str(error.value)
    assert runtime.state is MdpRuntimeState.EMPTY
    _assert_exact_release_for_tags(runtime.allocator, {"packed_pixels"})


def test_rank_local_encode_failure_is_coordinated_and_releases_packed_base():
    runtime, view = _build_runtime(
        _FailingEncodeAdapter, allocator_factory=lambda _view: _IdentityAllocator()
    )
    with pytest.raises(Exception) as error:
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    if view.lane_id is not None:
        assert isinstance(error.value, _InjectedOwnershipFailure)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "P2 encoder forward" in str(error.value)
    assert runtime.state is MdpRuntimeState.EMPTY
    _assert_exact_release_for_tags(runtime.allocator, {"packed_pixels"})


def test_rank_local_output_shape_failure_is_coordinated_before_p3():
    runtime, view = _build_runtime(
        _InvalidShapeAdapter, allocator_factory=lambda _view: _IdentityAllocator()
    )
    with pytest.raises(Exception) as error:
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    if view.lane_id is not None:
        assert isinstance(error.value, MdpStateError)
        assert "plane 0 shape" in str(error.value)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "P2 encoder forward" in str(error.value)
    assert runtime.state is MdpRuntimeState.EMPTY
    _assert_exact_release_for_tags(runtime.allocator, {"packed_pixels"})


def test_pixel_exchange_failure_releases_exact_packed_base():
    runtime, _ = _build_runtime(
        allocator_factory=lambda _view: _IdentityAllocator(),
        bridge_factory=lambda allocator: _PhaseFailBridge(allocator, BridgePhase.PIXEL),
    )
    with pytest.raises(_InjectedOwnershipFailure, match="pixel exchange failure"):
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"packed_pixels"})


def test_multiple_output_planes_route_store_replay_and_backprop_in_order():
    runtime, view = _build_runtime(_MultiPlaneAdapter)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    records = [next(replay[0]) for _ in range(2)]
    assert [record.model_payload["microbatch"] for record in records] == [0, 1]

    # The item plan remains single-copy; only its transport dimension expands.
    assert len(runtime._plan.routes) == len(GRIDS)
    embedding_ledger = runtime._iter_ledgers[BridgePhase.EMBEDDING]
    assert len(embedding_ledger.entries) == 2 * len(GRIDS)
    expected_single_plane_bytes = sum(
        t * (h // MERGE) * (w // MERGE) * WIDTH * torch.float32.itemsize for t, h, w in GRIDS
    )
    expected_deepstack_bytes = expected_single_plane_bytes // 2
    assert embedding_ledger.total_bytes == (expected_single_plane_bytes + expected_deepstack_bytes)
    if view.lane_id is not None:
        leaves = runtime.storage.get_leaves(0)
        assert len(leaves) == 2
        assert runtime.storage.get_leaves(1) is None
        offset = 0
        for index, grid in enumerate(GRIDS):
            t, h, w = grid
            rows = t * (h // MERGE) * (w // MERGE)
            expected = _sentinel(view.outer_dp_rank, index)
            assert torch.equal(
                leaves[0][offset : offset + rows],
                torch.full_like(leaves[0][offset : offset + rows], expected),
            )
            assert torch.equal(
                leaves[1][offset : offset + rows],
                torch.full_like(leaves[1][offset : offset + rows], 3.0 * expected),
            )
            offset += rows
        # Ordered model replay: final and DeepStack planes receive distinct weights.
        (2.0 * leaves[0].sum() + 5.0 * leaves[1].sum()).backward()

    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    runtime.storage.assert_empty()

    events_by_plane = {0: [], 1: []}
    for plane_id, grad in runtime.adapter.grad_events:
        events_by_plane[plane_id].append(grad)
    assert len(events_by_plane[0]) == len(events_by_plane[1])
    for grad in events_by_plane[0]:
        assert torch.equal(grad, torch.full_like(grad, 2.0))
    for grad in events_by_plane[1]:
        assert torch.equal(grad, torch.full_like(grad, 5.0))

    module = runtime.encoder_domain.encoder_ddp.module
    final_grad = module.final.weight.main_grad
    deepstack_grad = module.deepstack.weight.main_grad
    assert torch.isfinite(final_grad).all() and final_grad.abs().sum() > 0
    assert torch.isfinite(deepstack_grad).all() and deepstack_grad.abs().sum() > 0


def test_p3_later_plane_acquire_failure_is_coordinated_and_releases_exact_bases():
    def allocator_factory(view):
        return _IdentityAllocator(fail_tag="leaf" if view.lane_id is not None else None, fail_nth=2)

    runtime, view = _build_runtime(_MultiPlaneAdapter, allocator_factory=allocator_factory)
    with pytest.raises(Exception) as error:
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    if view.lane_id is not None:
        assert isinstance(error.value, _InjectedOwnershipFailure)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "P3 leaf assembly" in str(error.value)
    assert BridgePhase.EMBEDDING.value not in runtime.bridge.last_stats()
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"leaf", "packed_pixels"})


def test_p3_exchange_failure_releases_every_unhanded_leaf_base():
    runtime, _ = _build_runtime(
        _MultiPlaneAdapter,
        allocator_factory=lambda _view: _IdentityAllocator(),
        bridge_factory=lambda allocator: _PhaseFailBridge(allocator, BridgePhase.EMBEDDING),
    )
    with pytest.raises(_InjectedOwnershipFailure, match="embedding exchange"):
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"leaf", "packed_pixels"})


def test_p3_asymmetric_bridge_failure_never_launches_post_failure_collective(monkeypatch):
    runtime, view = _build_runtime(
        _MultiPlaneAdapter,
        allocator_factory=lambda _view: _IdentityAllocator(),
        bridge_factory=lambda allocator: _AsymmetricPhaseFailBridge(
            allocator, BridgePhase.EMBEDDING
        ),
    )
    original_all_reduce = torch.distributed.all_reduce
    post_failure_collectives = []

    def record_all_reduce(*args, **kwargs):
        if runtime.bridge.failed:
            post_failure_collectives.append("all_reduce")
        return original_all_reduce(*args, **kwargs)

    monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce)
    with pytest.raises(Exception) as error:
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    if view.global_rank == view.planning_group_ranks[0]:
        assert isinstance(error.value, _InjectedOwnershipFailure)
        assert "asymmetric embedding exchange failure" in str(error.value)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "peer observed embedding exchange failure" in str(error.value)
    assert post_failure_collectives == []
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"leaf", "packed_pixels"})


def test_p3_storage_handoff_failure_is_coordinated_and_releases_exact_bases():
    runtime, view = _build_runtime(
        _MultiPlaneAdapter,
        allocator_factory=lambda _view: _IdentityAllocator(),
        storage_factory=_FailingStorage,
    )
    with pytest.raises(Exception) as error:
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    if view.lane_id is not None:
        assert isinstance(error.value, _InjectedOwnershipFailure)
        assert "storage handoff" in str(error.value)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "P3 embedding handoff" in str(error.value)
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"leaf", "packed_pixels"})


def test_p5_later_regroup_acquire_failure_stops_bridge_and_releases_exact_bases():
    def allocator_factory(view):
        return _IdentityAllocator(
            fail_tag="grad_regroup" if view.my_worker_id == view.worker_ids[0] else None, fail_nth=2
        )

    runtime, view = _build_runtime(_MultiPlaneAdapter, allocator_factory=allocator_factory)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    records = [next(replay[0]) for _ in range(2)]
    assert len(records) == 2
    if view.lane_id is not None:
        leaves = runtime.storage.get_leaves(0)
        (leaves[0].sum() + leaves[1].sum()).backward()
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    with pytest.raises(Exception) as error:
        runtime.end_iteration()
    if view.my_worker_id == view.worker_ids[0]:
        assert isinstance(error.value, _InjectedOwnershipFailure)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "P5 gradient preparation" in str(error.value)
    assert BridgePhase.GRADIENT.value not in runtime.bridge.last_stats()
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"leaf", "grad_regroup", "packed_pixels"})


def test_native_schedule_failure_aborts_locally_and_runtime_is_reusable():
    runtime, view = _build_runtime(allocator_factory=lambda _view: _IdentityAllocator())

    def failed_schedule(*, data_iterator, num_microbatches, forward_only):
        assert next(data_iterator).microbatch_id == 0
        raise _InjectedOwnershipFailure("injected native schedule failure")

    wrapped = wrap_forward_backward(failed_schedule, runtime)
    with pytest.raises(_InjectedOwnershipFailure, match="native schedule failure"):
        wrapped(data_iterator=iter(range(10)), num_microbatches=2, forward_only=False)
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"leaf", "packed_pixels"})

    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=True)
    _drive_decoder(runtime, view, replay, backward=False)
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    assert runtime.state is MdpRuntimeState.EMPTY


def test_rank_local_p5_backward_failure_is_coordinated_and_releases_payload():
    runtime, view = _build_runtime(allocator_factory=lambda _view: _IdentityAllocator())
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    _drive_decoder(runtime, view, replay, backward=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()

    if view.lane_id is not None:

        def fail_backward(_grads):
            raise _InjectedOwnershipFailure("injected encoder backward failure")

        runtime._handle.backward = fail_backward
    else:

        def finish_without_autograd(_grads):
            runtime._handle._backward_done = True

        runtime._handle.backward = finish_without_autograd

    with pytest.raises(Exception) as error:
        runtime.end_iteration()
    if view.lane_id is not None:
        assert isinstance(error.value, _InjectedOwnershipFailure)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "P5 encoder backward" in str(error.value)
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"leaf", "grad_regroup", "packed_pixels"})


def test_p5_asymmetric_bridge_failure_never_launches_post_failure_collective(monkeypatch):
    runtime, view = _build_runtime(
        allocator_factory=lambda _view: _IdentityAllocator(),
        bridge_factory=lambda allocator: _AsymmetricPhaseFailBridge(
            allocator, BridgePhase.GRADIENT
        ),
    )
    original_all_reduce = torch.distributed.all_reduce
    post_failure_collectives = []

    def record_all_reduce(*args, **kwargs):
        if runtime.bridge.failed:
            post_failure_collectives.append("all_reduce")
        return original_all_reduce(*args, **kwargs)

    monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    _drive_decoder(runtime, view, replay, backward=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    with pytest.raises(Exception) as error:
        runtime.end_iteration()
    if view.global_rank == view.planning_group_ranks[0]:
        assert isinstance(error.value, _InjectedOwnershipFailure)
        assert "asymmetric gradient exchange failure" in str(error.value)
    else:
        assert isinstance(error.value, MdpStateError)
        assert "peer observed gradient exchange failure" in str(error.value)
    assert post_failure_collectives == []
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    _assert_exact_release_for_tags(runtime.allocator, {"leaf", "grad_regroup", "packed_pixels"})


def test_invalid_transitions_raise():
    runtime, view = _build_runtime()
    with pytest.raises(MdpStateError, match="mark_decoder_complete"):
        runtime.mark_decoder_complete()
    with pytest.raises(MdpStateError, match="end_iteration"):
        runtime.end_iteration()

    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
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
