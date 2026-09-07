# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Qwen encoder through both public repeated-D4 schedule modes."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev import forward_step
from examples.multimodal_dev import mdp_adapter as mdp_adapter_module
from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.mdp import integration
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
from megatron.core.mdp.dynamic_encoder_adapter_capability import (
    claim_dynamic_encoder_adapter_capability,
    mint_dynamic_encoder_adapter_capability,
)
from megatron.core.mdp.encoder import EncoderDomain, build_encoder_pg_collection
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8
pytestmark = pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")

_HIDDEN = 128
_PATCH_DIM = 3 * 2 * 16 * 16
_SEQ = 64


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=4,
        dynamic_context_parallel=True,
        min_dynamic_context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    yield
    integration.reset_for_testing()
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _reset_integration():
    integration.reset_for_testing()
    yield
    integration.reset_for_testing()


def _vision_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=2,
        num_query_groups=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        context_parallel_size=4,
        sequence_parallel=False,
        apply_rope_fusion=False,
        mrope_section=[0, 8, 8],
    )


def _batch():
    device = torch.device("cuda", torch.cuda.current_device())
    boundaries = torch.tensor((0, _SEQ), dtype=torch.int32, device=device)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries.clone(),
        cu_seqlens_q_padded=boundaries.clone(),
        cu_seqlens_kv_padded=boundaries.clone(),
        max_seqlen_q=_SEQ,
        max_seqlen_kv=_SEQ,
        total_tokens=_SEQ,
    )
    generator = torch.Generator(device=device).manual_seed(700 + dist.get_rank() // 4)
    return {
        "input_ids": torch.arange(_SEQ, dtype=torch.int64, device=device).view(1, -1),
        "labels": torch.arange(_SEQ, dtype=torch.int64, device=device).view(1, -1),
        "loss_mask": torch.ones((1, _SEQ), device=device),
        "padding_mask": torch.zeros((1, _SEQ), dtype=torch.bool, device=device),
        "position_ids": torch.arange(_SEQ, dtype=torch.int64, device=device).view(1, -1),
        "attention_mask": None,
        "image_grid_thw": torch.tensor(((1, 8, 8),), dtype=torch.int64, device=device),
        "pixel_values": torch.randn(
            (64, _PATCH_DIM), dtype=torch.bfloat16, device=device, generator=generator
        ),
        "vision_item_meta": torch.tensor(((0, 0, 1, 8, 8, 0),), dtype=torch.int64, device=device),
        "vision_decoder_positions": torch.arange(16, dtype=torch.int64, device=device),
        "packed_seq_params": packed,
    }


def _runtime():
    rank = dist.get_rank()
    rank_map = build_rank_map(MdpRankSpec(world_size=8, tp=1, pp=1, cp=4, ep=1, encoder_cp=4))
    groups = install_mdp_process_groups(
        rank_map,
        group_registry=MdpGroupRegistry(),
        decoder_pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        dynamic_encoder_cp=True,
        min_dynamic_encoder_cp_size=1,
    )
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=4, process_groups=groups)
    adapter = Qwen35VLMdpAdapter(
        out_hidden_size=_HIDDEN,
        vision_kwargs={
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "max_num_positions": 2304,
        },
    )
    capability = mint_dynamic_encoder_adapter_capability(adapter)
    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)
    torch.manual_seed(4321)
    model_parallel_cuda_manual_seed(4321)
    encoder = operations.build_encoder(_vision_config(), pg_collection=encoder_pgs)
    encoder = encoder.bfloat16().cuda()
    encoder_ddp = DistributedDataParallel(
        config=_vision_config(),
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
        ),
        module=encoder,
        pg_collection=encoder_pgs,
    )
    allocator = DirectBufferAllocator()
    config = MdpConfig(
        enable=True,
        encoder_cp=4,
        encoder_max_payload_rows=16,
        dynamic_encoder_cp=True,
        min_dynamic_encoder_cp_size=1,
    )
    binding = _make_repeated_d4_group_binding(
        world_group=groups.world_group,
        domain_group=groups.encoder_cp_group,
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=torch.device("cuda", torch.cuda.current_device()),
        timeout_seconds=30.0,
    )
    return MdpRuntime(
        config=config,
        rank_map=rank_map,
        rank_view=rank_map.view(rank),
        process_groups=groups,
        adapter=operations,
        encoder_domain=EncoderDomain(encoder_ddp, None, _vision_config()),
        planner=MdpPlanner(
            rank_map.view(rank), capacity_policy=RowCapacityPolicy(), locality_slack_permille=10
        ),
        bridge=ModalityBridge(allocator),
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=_HIDDEN,
        params_dtype=torch.bfloat16,
        num_vpp_chunks=1,
        dynamic_adapter_capability=capability,
        dynamic_adapter_owner=adapter,
        dynamic_group_binding=binding,
    )


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
def test_public_repeated_d4_runs_actual_qwen_backward_finalize_and_commit(
    monkeypatch, dynamic_decoder
):
    runtime = _runtime()
    integration._RUNTIME = runtime
    selected_encoder_cp = []
    native_bind = mdp_adapter_module._bind_dynamic_encoder_cp

    def observe_encoder_cp(*args, membership, **kwargs):
        selected_encoder_cp.append(membership.group_size)
        return native_bind(*args, membership=membership, **kwargs)

    def get_batch(iterator):
        next(iterator)
        return _batch()

    monkeypatch.setattr(forward_step, "get_batch", get_batch)
    monkeypatch.setattr(mdp_adapter_module, "_bind_dynamic_encoder_cp", observe_encoder_cp)

    def finalize(_model, tokens):
        dist.all_reduce(tokens)

    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=16,
        finalize_model_grads_func=finalize,
    )
    sentinel = object()

    class Decoder(torch.nn.Module):
        vp_stage = None

        def forward(self, *, input_ids, vision_embeddings, packed_seq_params, **_kwargs):
            assert vision_embeddings is not None and vision_embeddings.requires_grad
            scalar = vision_embeddings.float().square().mean()
            local_tokens = input_ids.shape[-1] // packed_seq_params.cp_group.size()
            return scalar.expand(input_ids.shape[0], local_tokens)

    decoder = Decoder()

    def native_schedule(data_iterator, num_microbatches, forward_only):
        assert forward_only is False
        assert num_microbatches == 1
        with pytest.raises(MdpStateError, match="just-yielded record"):
            data_iterator.vision_embedding_leaf(object())
        output, output_loss_func = forward_step.forward_step(data_iterator, decoder)
        loss, tokens, _ = output_loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        assert int(tokens) == 128
        return sentinel

    wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
    result = wrapped(data_iterator=iter((object(),)), num_microbatches=1, forward_only=False)

    assert result is sentinel
    assert runtime.iteration == 1
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime.storage.get_leaf(0) is None
    assert selected_encoder_cp == [4]
    grads = [parameter.main_grad for parameter in runtime.encoder_domain.encoder_ddp.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in grads)
