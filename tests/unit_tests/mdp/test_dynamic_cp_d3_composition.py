# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction and world-4 lifecycle tests for the concrete private D3 composition."""

import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
from megatron.core import parallel_state
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.dynamic_cp_d3_composition import _build_d3_runtime_facade
from megatron.core.mdp.dynamic_cp_d3_coordinator import _D3Coordinator
from megatron.core.mdp.dynamic_cp_d3_private_facade import _D3PrivateFacade
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.runtime import MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import pixel_capture_suppressed
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.mdp import test_runtime as runtime_harness
from tests.unit_tests.test_utilities import Utils

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_WORLD_RANKS = (0, 1, 2, 3)
_DTYPE = torch.bfloat16


class _Group:
    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0


class _Codec:
    @staticmethod
    def rebuild_microbatch(*_args, **_kwargs):
        return None


def _runtime(*, tp=1):
    allocator = DirectBufferAllocator()
    return SimpleNamespace(
        rank_map=SimpleNamespace(spec=SimpleNamespace(tp=tp, ep=1, pp=1, cp=1, encoder_cp=1)),
        num_vpp_chunks=1,
        config=SimpleNamespace(dynamic_encoder_cp=False, overlap_window_capture=False),
        device=torch.device("cuda", 0),
        params_dtype=torch.bfloat16,
        hidden_size=8,
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
        _prepare_dynamic_encoder_producer=lambda *_args, **_kwargs: None,
    )


def _build(runtime):
    group = _Group()
    return _build_d3_runtime_facade(
        producer_runtime=runtime,
        codec=_Codec(),
        group=group,
        participant_ranks=(0,),
        global_rank=0,
        device=runtime.device,
        expected_source_lanes=(0,),
        decoder_solver=lambda *_args, **_kwargs: None,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        decoder_group_getter=lambda _size: group,
        decoder_group_ranks_getter=lambda _group: (0,),
        timeout_seconds=5.0,
        group_ranks_getter=lambda _group: (0,),
        all_to_all_single=lambda *_args, **_kwargs: None,
    )


def test_factory_builds_exact_config_and_concrete_coordinator_bindings():
    facade = _build(_runtime())

    assert type(facade) is _D3PrivateFacade
    config = facade._config_factory(None, num_microbatches=2, forward_only=False)
    coordinator = facade._coordinator_factory(None, num_microbatches=2, forward_only=False)

    assert type(coordinator) is _D3Coordinator
    assert coordinator.is_idle
    assert config.participant_ranks == (0,)
    assert config.tensor_parallel_size == 1
    assert all(
        callable(getattr(coordinator._bindings, name))
        for name in coordinator._bindings.__dataclass_fields__
    )


def test_factory_rejects_nonlegacy_topology_before_facade_construction():
    with pytest.raises(MdpConfigurationError, match="TP1/EP1/PP1/CP1/ECP1/VPP1"):
        _build(_runtime(tp=2))


@pytest.fixture(scope="module")
def _world4_model_parallel():
    if _WORLD_SIZE != 4:
        pytest.skip("requires torchrun world size 4")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        dynamic_context_parallel=True,
        min_dynamic_context_parallel_size=1,
    )
    try:
        yield
    finally:
        Utils.destroy_model_parallel()
        assert not parallel_state.model_parallel_is_initialized()


def _packed(length):
    boundaries = torch.tensor((0, length), dtype=torch.int32, device="cuda")
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries.clone(),
        cu_seqlens_q_padded=boundaries.clone(),
        cu_seqlens_kv_padded=boundaries.clone(),
        max_seqlen_q=length,
        max_seqlen_kv=length,
        total_tokens=length,
    )


class _D3Adapter(runtime_harness._StubAdapter):
    """One source-local Qwen-style sample with one vision item."""

    def get_batch(self, iterator):
        microbatch = next(iterator)
        assert microbatch == 0
        grid = (1, 4, 4)
        payload_rows = 16
        output_rows = 4
        pixels = None
        if not pixel_capture_suppressed():
            pixels = torch.full(
                (payload_rows, runtime_harness.WIDTH),
                runtime_harness._sentinel(self._lane, 0),
                dtype=_DTYPE,
                device="cuda",
            )
            self.materialized_count += 1
        tokens = torch.arange(8, dtype=torch.int64, device="cuda").view(1, -1)
        return CapturedMicrobatch(
            decoder_packed_seq_params=_packed(8),
            vision_items=(
                CapturedVisionItem(
                    sample_id=0,
                    image_ordinal=0,
                    grid_thw=grid,
                    payload_row_start=0,
                    payload_rows=payload_rows,
                    decoder_positions=tuple(range(output_rows)),
                ),
            ),
            flat_pixel_payload=pixels,
            model_payload=MappingProxyType(
                {
                    "input_ids": tokens,
                    "labels": tokens + 100,
                    "loss_mask": torch.ones((1, 8), dtype=torch.float32, device="cuda"),
                    "padding_mask": torch.zeros((1, 8), dtype=torch.bool, device="cuda"),
                    "position_ids": tokens + 200,
                    "attention_mask": None,
                    "image_grid_thw": torch.tensor((grid,), dtype=torch.int64),
                }
            ),
        )

    def encode(self, encoder, payload, layout):
        """Model the mixed-precision boundary without mutating a built DDP."""
        return super().encode(encoder, payload.float(), layout).to(_DTYPE)


def _full_world_solver(sample_seqlens, total_gpus, **_kwargs):
    sample_ids = [sample_id for sample_id, _ in sample_seqlens]
    lengths = [length for _, length in sample_seqlens]
    return (
        [list(lengths) for _ in range(total_gpus)],
        [],
        None,
        [list(sample_ids) for _ in range(total_gpus)],
    )


@pytest.mark.usefixtures("_world4_model_parallel")
def test_world4_composition_executes_native_producer_through_gate6(monkeypatch):
    rank = dist.get_rank()
    monkeypatch.setattr(runtime_harness, "_StubAdapter", _D3Adapter)
    runtime, view = runtime_harness._build_runtime(decoder_pp=1)
    assert view.lane_id == rank
    runtime.params_dtype = _DTYPE
    group = dist.group.WORLD
    facade = _build_d3_runtime_facade(
        producer_runtime=runtime,
        codec=MultimodalDecoderPayloadCodec(),
        group=group,
        participant_ranks=_WORLD_RANKS,
        global_rank=rank,
        device=torch.device("cuda", torch.cuda.current_device()),
        expected_source_lanes=_WORLD_RANKS,
        decoder_solver=_full_world_solver,
        max_seqlen_per_rank=32,
        minimum_cp_size=1,
        decoder_group_getter=lambda *, group_size: group if group_size == 4 else None,
        decoder_group_ranks_getter=lambda selected: tuple(dist.get_process_group_ranks(selected)),
        timeout_seconds=30.0,
    )

    ready = facade.begin_iteration(iter((0,)), num_microbatches=1, forward_only=False)
    assert len(ready.records) == 1
    assert len(ready.embedding_leaves) == 1
    leaf = next(iter(ready.embedding_leaves.values()))
    assert leaf.is_leaf and leaf.requires_grad
    (leaf * float(rank + 1)).sum().backward()
    runtime.capture_global_num_tokens(torch.tensor(32.0, device="cuda"))
    facade.mark_decoder_complete(ready)
    facade.end_iteration(ready)

    assert facade.is_idle
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime.iteration == 1
    runtime.storage.assert_empty()
    assert runtime._handle is None
    assert runtime.adapter.output_grad_events
    assert all(gradient.abs().sum() > 0 for gradient in runtime.adapter.output_grad_events)
    assert next(runtime.encoder_domain.encoder_ddp.module.parameters()).main_grad.abs().sum() > 0
