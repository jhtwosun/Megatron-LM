# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction and world-4 lifecycle tests for the concrete private D3 composition."""

import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
from megatron.core import parallel_state
from megatron.core.mdp import dynamic_cp_d3_metadata_transport as metadata_transport
from megatron.core.mdp import integration
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.dynamic_cp_d3_composition import _build_d3_runtime_facade
from megatron.core.mdp.dynamic_cp_d3_coordinator import _D3Coordinator
from megatron.core.mdp.dynamic_cp_d3_private_facade import _D3PrivateFacade
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.runtime import MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import pixel_capture_suppressed
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.mdp import test_runtime as runtime_harness
from tests.unit_tests.test_utilities import Utils

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
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

    @staticmethod
    def build_dynamic_decoder_payload_codec():
        return MultimodalDecoderPayloadCodec()

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


class _Rank2LocalPreparationBaseException(BaseException):
    pass


class _Rank2FailOnceCodec(MultimodalDecoderPayloadCodec):
    def __init__(self):
        self.failed = False

    def build_source_window_with_locations(self, records, *, source_dp_lane):
        if dist.get_rank() == 2 and not self.failed:
            self.failed = True
            raise _Rank2LocalPreparationBaseException("injected rank-2 codec failure")
        return super().build_source_window_with_locations(records, source_dp_lane=source_dp_lane)


class _FailOnceD3Adapter(_D3Adapter):
    @staticmethod
    def build_dynamic_decoder_payload_codec():
        return _Rank2FailOnceCodec()


@pytest.mark.usefixtures("_world4_model_parallel")
def test_world4_composition_executes_native_producer_through_gate6(monkeypatch):
    rank = dist.get_rank()
    monkeypatch.setattr(runtime_harness, "_StubAdapter", _D3Adapter)
    runtime, view = runtime_harness._build_runtime(decoder_pp=1)
    assert view.lane_id == rank
    runtime.params_dtype = _DTYPE
    config = SimpleNamespace(
        dynamic_context_parallel=True,
        sequence_packing_scheduler="default_dynamic_cp",
        max_seqlen_per_dp_cp_rank=32,
        min_dynamic_context_parallel_size=1,
        finalize_model_grads_func=lambda *_args, **_kwargs: None,
    )
    integration.reset_for_testing()
    monkeypatch.setattr(integration, "_RUNTIME", runtime)

    def native_schedule(*, data_iterator, num_microbatches, forward_only):
        assert num_microbatches == 1 and forward_only is False
        record = next(data_iterator)
        leaf = runtime.storage.get_leaf(record.microbatch_id)
        assert leaf is not None and leaf.is_leaf and leaf.requires_grad
        (leaf * float(rank + 1)).sum().backward()
        runtime.capture_global_num_tokens(torch.tensor(32.0, device="cuda"))
        return "native-result"

    wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
    try:
        result = wrapped(data_iterator=iter((0,)), num_microbatches=1, forward_only=False)
    finally:
        facade = integration._D3_FACADE
        integration.reset_for_testing()

    assert result == "native-result"
    assert type(facade) is _D3PrivateFacade and facade.is_idle
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime.iteration == 1
    runtime.storage.assert_empty()
    assert runtime._handle is None
    assert runtime.adapter.output_grad_events
    assert all(gradient.abs().sum() > 0 for gradient in runtime.adapter.output_grad_events)
    assert next(runtime.encoder_domain.encoder_ddp.module.parameters()).main_grad.abs().sum() > 0


@pytest.mark.usefixtures("_world4_model_parallel")
def test_world4_local_baseexception_converges_before_body_and_fresh_retry_succeeds(monkeypatch):
    rank = dist.get_rank()
    monkeypatch.setattr(runtime_harness, "_StubAdapter", _FailOnceD3Adapter)
    runtime, view = runtime_harness._build_runtime(decoder_pp=1)
    assert view.lane_id == rank
    runtime.params_dtype = _DTYPE
    config = SimpleNamespace(
        dynamic_context_parallel=True,
        sequence_packing_scheduler="default_dynamic_cp",
        max_seqlen_per_dp_cp_rank=32,
        min_dynamic_context_parallel_size=1,
        finalize_model_grads_func=lambda *_args, **_kwargs: None,
    )
    integration.reset_for_testing()
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    schedule_calls = 0
    metadata_body_calls = 0
    gather_body = metadata_transport._gather_body

    def count_metadata_body(*args, **kwargs):
        nonlocal metadata_body_calls
        metadata_body_calls += 1
        return gather_body(*args, **kwargs)

    monkeypatch.setattr(metadata_transport, "_gather_body", count_metadata_body)

    def native_schedule(*, data_iterator, num_microbatches, forward_only):
        nonlocal schedule_calls
        schedule_calls += 1
        assert num_microbatches == 1 and forward_only is False
        record = next(data_iterator)
        leaf = runtime.storage.get_leaf(record.microbatch_id)
        assert leaf is not None and leaf.is_leaf and leaf.requires_grad
        (leaf * float(rank + 1)).sum().backward()
        runtime.capture_global_num_tokens(torch.tensor(32.0, device="cuda"))
        return "native-result"

    wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
    try:
        with pytest.raises(MdpPlanError) as caught:
            wrapped(data_iterator=iter((0,)), num_microbatches=1, forward_only=False)

        facade = integration._D3_FACADE
        observation = (
            type(caught.value).__name__,
            schedule_calls,
            facade.is_idle,
            runtime.state.name,
            runtime.iteration,
            runtime._handle is None,
            metadata_body_calls,
        )
        observations = [None] * dist.get_world_size()
        dist.all_gather_object(observations, observation)
        assert (
            observations == [("MdpPlanError", 0, True, "EMPTY", 0, True, 0)] * dist.get_world_size()
        )
        runtime.storage.assert_empty()

        result = wrapped(data_iterator=iter((0,)), num_microbatches=1, forward_only=False)
        assert result == "native-result"
        assert schedule_calls == 1
        assert metadata_body_calls > 0
        assert facade.is_idle
        assert runtime.state is MdpRuntimeState.EMPTY
        assert runtime.iteration == 1
        runtime.storage.assert_empty()
        assert runtime._handle is None
    finally:
        integration.reset_for_testing()
