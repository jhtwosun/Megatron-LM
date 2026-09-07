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
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
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


@pytest.fixture(scope="module")
def mdp_group_registry():
    """Reuse immutable startup groups while rebuilding every parity arm."""
    registry = MdpGroupRegistry()
    yield registry
    registry.assert_no_leak()


@pytest.fixture(scope="module")
def expert_replica_group():
    """Expert-DP replicas pair equal expert slots across the two D4 domains."""
    groups = tuple(dist.new_group(ranks=(index, index + 4)) for index in range(4))
    yield groups[dist.get_rank() % 4]
    for group in reversed(groups):
        dist.destroy_process_group(group)


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


def _batch(*, vision=True):
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
    batch = {
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
    if not vision:
        batch.update(
            image_grid_thw=torch.empty((0, 3), dtype=torch.int64, device=device),
            pixel_values=torch.empty((0, _PATCH_DIM), dtype=torch.bfloat16, device=device),
            vision_item_meta=torch.empty((0, 6), dtype=torch.int64, device=device),
            vision_decoder_positions=torch.empty(0, dtype=torch.int64, device=device),
        )
    return batch


def _runtime(*, group_registry, encoder_capacity=16, expert_parallel_size=1):
    rank = dist.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=8, tp=1, pp=1, cp=4, ep=expert_parallel_size, encoder_cp=4)
    )
    groups = install_mdp_process_groups(
        rank_map,
        group_registry=group_registry,
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
        encoder_max_payload_rows=encoder_capacity,
        dynamic_encoder_cp=True,
        min_dynamic_encoder_cp_size=1,
    )
    binding = _make_repeated_d4_group_binding(
        world_group=groups.world_group,
        domain_group=groups.encoder_cp_group,
        expert_group=groups.encoder_cp_group if expert_parallel_size == 4 else None,
        global_rank=rank,
        expert_parallel_size=expert_parallel_size,
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
@pytest.mark.parametrize(
    ("encoder_capacity", "expected_encoder_cp"),
    ((64, 1), (32, 2), (16, 4)),
    ids=("ecp1", "ecp2", "ecp4"),
)
def test_public_repeated_d4_runs_actual_qwen_backward_finalize_and_commit(
    monkeypatch, mdp_group_registry, dynamic_decoder, encoder_capacity, expected_encoder_cp
):
    runtime = _runtime(group_registry=mdp_group_registry, encoder_capacity=encoder_capacity)
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
    selected = dist.get_rank() % 4 < expected_encoder_cp
    assert selected_encoder_cp == ([expected_encoder_cp] if selected else [])
    grads = [parameter.main_grad for parameter in runtime.encoder_domain.encoder_ddp.parameters()]
    # Encoder parameters are WORLD-replicated. Gate6 synchronizes the selected
    # subgroup's contribution onto selected and idle ranks alike.
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in grads)


class _OptimizerParityDecoder(torch.nn.Module):
    """One real decoder-domain parameter around the D4 vision leaf."""

    vp_stage = None

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.bfloat16, device="cuda"))

    def forward(self, *, input_ids, vision_embeddings, packed_seq_params, **_kwargs):
        assert packed_seq_params.cp_group.size() == 4
        scalar = self.scale.float() * vision_embeddings.float().square().mean()
        local_tokens = input_ids.shape[-1] // packed_seq_params.cp_group.size()
        return scalar.expand(input_ids.shape[0], local_tokens)


class _DomainEp4MoeDecoder(torch.nn.Module):
    """Tiny native MCore MoE using one expert per rank in a D4 domain."""

    vp_stage = None

    def __init__(self, pg_collection):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=_HIDDEN,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_ffn_hidden_size=256,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="none",
            moe_router_topk=2,
            moe_grouped_gemm=False,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=1,
            context_parallel_size=4,
            expert_model_parallel_size=4,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False).mlp
        )
        self.moe = MoELayer(config, submodules, layer_number=1, pg_collection=pg_collection)
        with torch.no_grad():
            self.moe.router.weight.zero_()
            self.moe.router.weight[:, :4].copy_(
                4 * torch.eye(4, dtype=self.moe.router.weight.dtype, device="cuda")
            )

    def forward(self, *, input_ids, vision_embeddings, packed_seq_params, **_kwargs):
        probes = 4 * torch.eye(4, _HIDDEN, dtype=torch.bfloat16, device="cuda")
        hidden = probes if vision_embeddings is None else torch.cat((vision_embeddings, probes))
        hidden = hidden.unsqueeze(1)
        output, _ = self.moe(hidden)
        local_tokens = input_ids.shape[-1] // packed_seq_params.cp_group.size()
        return output.float().square().mean().expand(input_ids.shape[0], local_tokens)


def _clone_named_parameters(module):
    return {
        name: parameter.detach().float().clone() for name, parameter in module.named_parameters()
    }


def _clone_optimizer_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().clone()
    if isinstance(value, dict):
        return {key: _clone_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_optimizer_state(item) for item in value)
    return value


def _optimizer_steps(value):
    if isinstance(value, dict):
        return tuple(int(item) for key, item in value.items() if key == "step") + tuple(
            step for item in value.values() for step in _optimizer_steps(item)
        )
    if isinstance(value, (list, tuple)):
        return tuple(step for item in value for step in _optimizer_steps(item))
    return ()


def _snapshot_batch(batch):
    packed = batch["packed_seq_params"]
    return {
        "tensors": {
            name: value.detach().clone() for name, value in batch.items() if torch.is_tensor(value)
        },
        "packed": {
            name: value.detach().clone() if torch.is_tensor(value) else value
            for name in (
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_q_padded",
                "cu_seqlens_kv_padded",
                "max_seqlen_q",
                "max_seqlen_kv",
                "total_tokens",
            )
            if (value := getattr(packed, name)) is not None
        },
    }


def _run_optimizer_parity_mode(
    monkeypatch,
    *,
    dynamic_decoder,
    native_bind,
    group_registry,
    encoder_capacity=16,
    expected_encoder_cp=4,
    adam_eps=1e-8,
):
    from megatron.core.mdp.optimizer import build_mdp_composite_optimizer
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    runtime = _runtime(group_registry=group_registry, encoder_capacity=encoder_capacity)
    integration._RUNTIME = runtime
    selected_encoder_cp = []
    captured_batches = []

    def observe_encoder_cp(*args, membership, **kwargs):
        selected_encoder_cp.append(membership.group_size)
        return native_bind(*args, membership=membership, **kwargs)

    def get_batch(iterator):
        next(iterator)
        batch = _batch()
        assert not captured_batches
        captured_batches.append(_snapshot_batch(batch))
        return batch

    monkeypatch.setattr(forward_step, "get_batch", get_batch)
    monkeypatch.setattr(mdp_adapter_module, "_bind_dynamic_encoder_cp", observe_encoder_cp)
    decoder = _OptimizerParityDecoder()
    decoder_ddp = DistributedDataParallel(
        config=_vision_config(),
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
        ),
        module=decoder,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam", lr=1e-2, adam_eps=adam_eps, bf16=True, clip_grad=0.1, weight_decay=0.0
    )
    decoder_optimizer = get_megatron_optimizer(
        optimizer_config, [decoder_ddp], use_gloo_process_groups=False
    )
    encoder_optimizer = get_megatron_optimizer(
        optimizer_config, [runtime.encoder_domain.encoder_ddp], use_gloo_process_groups=False
    )
    composite = build_mdp_composite_optimizer(decoder_optimizer, encoder_optimizer)
    decoder_ddp.zero_grad_buffer()
    runtime.encoder_domain.encoder_ddp.zero_grad_buffer()
    initial_decoder = _clone_named_parameters(decoder_ddp)
    initial_encoder = _clone_named_parameters(runtime.encoder_domain.encoder_ddp)
    initial_optimizer = _clone_optimizer_state(composite.state_dict())
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = torch.cuda.get_rng_state().clone()

    def finalize(_model, tokens):
        decoder_ddp.finish_grad_sync()
        dist.all_reduce(tokens)

    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=16,
        finalize_model_grads_func=finalize,
    )

    def native_schedule(data_iterator, num_microbatches, forward_only):
        assert num_microbatches == 1 and forward_only is False
        output, output_loss_func = forward_step.forward_step(data_iterator, decoder_ddp)
        loss, tokens, _ = output_loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        assert int(tokens) == 128
        return loss.detach()

    try:
        wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
        loss = wrapped(data_iterator=iter((object(),)), num_microbatches=1, forward_only=False)
        decoder_grads = {
            name: parameter.main_grad.detach().float().clone()
            for name, parameter in decoder_ddp.named_parameters()
        }
        encoder_grads = {
            name: parameter.main_grad.detach().float().clone()
            for name, parameter in runtime.encoder_domain.encoder_ddp.named_parameters()
        }
        success, grad_norm, _ = composite.step()
        assert success and grad_norm > optimizer_config.clip_grad
        selected = dist.get_rank() % 4 < expected_encoder_cp
        assert selected_encoder_cp == ([expected_encoder_cp] if selected else [])
        decoder_state = _clone_named_parameters(decoder_ddp)
        encoder_state = _clone_named_parameters(runtime.encoder_domain.encoder_ddp)
        assert any(
            not torch.equal(decoder_state[name], initial_decoder[name]) for name in decoder_state
        )
        assert any(
            not torch.equal(encoder_state[name], initial_encoder[name]) for name in encoder_state
        )
        optimizer_state = _clone_optimizer_state(composite.state_dict())
        steps = _optimizer_steps(optimizer_state)
        assert len(steps) >= 2 and all(step == 1 for step in steps)
        return {
            "loss": loss.float(),
            "grad_norm": float(grad_norm),
            "initial_decoder": initial_decoder,
            "initial_encoder": initial_encoder,
            "initial_optimizer": initial_optimizer,
            "initial_cpu_rng": initial_cpu_rng,
            "initial_cuda_rng": initial_cuda_rng,
            "captured_batches": tuple(captured_batches),
            "decoder_grads": decoder_grads,
            "encoder_grads": encoder_grads,
            "decoder_state": decoder_state,
            "encoder_state": encoder_state,
            "optimizer_state": optimizer_state,
        }
    finally:
        integration.reset_for_testing()


def _assert_gradient_parity(actual, reference, section):
    assert actual.keys() == reference.keys(), section
    for name in actual:
        candidate = actual[name].float()
        baseline = reference[name].float()
        assert candidate.shape == baseline.shape, (section, name)
        assert torch.isfinite(candidate).all(), (section, name)
        assert torch.isfinite(baseline).all(), (section, name)
        candidate_norm = float(candidate.norm())
        baseline_norm = float(baseline.norm())
        baseline_max = float(baseline.abs().max())
        assert candidate_norm > 0 and baseline_norm > 0, (section, name)
        delta = candidate - baseline
        l2_relative = float(delta.norm()) / baseline_norm
        max_abs_relative = float(delta.abs().max()) / baseline_max
        cosine = float(
            torch.nn.functional.cosine_similarity(candidate.flatten(), baseline.flatten(), dim=0)
        )
        norm_ratio = candidate_norm / baseline_norm
        diagnostic = (
            f"{section}.{name}: l2_relative={l2_relative}, "
            f"max_abs_relative={max_abs_relative}, cosine={cosine}, "
            f"norm_ratio={norm_ratio}"
        )
        assert l2_relative <= 0.01, diagnostic
        assert max_abs_relative <= 0.015, diagnostic
        assert cosine >= 0.999, diagnostic
        assert 0.99 <= norm_ratio <= 1.01, diagnostic


def _vector_diagnostic(candidate, baseline):
    candidate = candidate.double()
    baseline = baseline.double()
    delta = candidate - baseline
    baseline_norm = float(baseline.norm())
    candidate_norm = float(candidate.norm())
    baseline_max = float(baseline.abs().max())
    assert baseline_norm > 0 and candidate_norm > 0
    return {
        "l2_relative": float(delta.norm()) / baseline_norm,
        "max_abs_relative": float(delta.abs().max()) / baseline_max,
        "cosine": float(candidate.dot(baseline)) / (candidate_norm * baseline_norm),
        "norm_ratio": candidate_norm / baseline_norm,
    }


def _tensor_leaves(value, path=()):
    if torch.is_tensor(value):
        return {path: value}
    if isinstance(value, dict):
        return {
            key: tensor
            for name, item in value.items()
            for key, tensor in _tensor_leaves(item, (*path, name)).items()
        }
    if isinstance(value, (list, tuple)):
        return {
            key: tensor
            for index, item in enumerate(value)
            for key, tensor in _tensor_leaves(item, (*path, index)).items()
        }
    return {}


def _assert_optimizer_state_parity(actual, reference):
    actual_leaves = _tensor_leaves(actual)
    reference_leaves = _tensor_leaves(reference)
    assert actual_leaves.keys() == reference_leaves.keys()
    assert _optimizer_steps(actual) == _optimizer_steps(reference)
    for path, candidate in actual_leaves.items():
        baseline = reference_leaves[path]
        assert candidate.shape == baseline.shape, path
        assert torch.isfinite(candidate).all(), path
        assert torch.isfinite(baseline).all(), path
        if torch.equal(candidate, baseline):
            continue
        diagnostic = _vector_diagnostic(candidate.flatten(), baseline.flatten())
        message = f"optimizer_state{path}: {diagnostic}"
        assert diagnostic["l2_relative"] <= 0.03, message
        assert diagnostic["cosine"] >= 0.999, message
        assert 0.97 <= diagnostic["norm_ratio"] <= 1.03, message


def _assert_optimizer_parity(actual, reference, *, allow_partition_variance=False):
    if not allow_partition_variance:
        assert torch.equal(actual["loss"], reference["loss"]), (actual["loss"], reference["loss"])
    else:
        torch.testing.assert_close(actual["loss"], reference["loss"], rtol=8e-3, atol=2e-3)
    relative_grad_norm_error = (
        abs(actual["grad_norm"] - reference["grad_norm"]) / reference["grad_norm"]
    )
    grad_norm_tolerance = 0.01 if allow_partition_variance else 1e-6
    assert relative_grad_norm_error < grad_norm_tolerance, relative_grad_norm_error
    for section in (
        "initial_decoder",
        "initial_encoder",
        "initial_optimizer",
        "initial_cpu_rng",
        "initial_cuda_rng",
        "captured_batches",
    ):
        torch.testing.assert_close(actual[section], reference[section], rtol=1e-6, atol=1e-7)
    if allow_partition_variance:
        for section in ("decoder_grads", "encoder_grads"):
            _assert_gradient_parity(actual[section], reference[section], section)
    else:
        for section in ("decoder_grads", "encoder_grads"):
            torch.testing.assert_close(actual[section], reference[section], rtol=1e-6, atol=1e-7)
    relative_rtol = 8e-3 if allow_partition_variance else 1e-6
    relative_atol = 2e-3 if allow_partition_variance else 1e-7
    for section in ("decoder_state", "encoder_state"):
        if allow_partition_variance and section == "encoder_state":
            selected_update = torch.cat(
                tuple(
                    (actual[section][name] - actual["initial_encoder"][name]).flatten()
                    for name in actual[section]
                )
            )
            reference_update = torch.cat(
                tuple(
                    (reference[section][name] - reference["initial_encoder"][name]).flatten()
                    for name in reference[section]
                )
            )
            diagnostic = _vector_diagnostic(selected_update, reference_update)
            message = f"encoder update: {diagnostic}"
            assert diagnostic["l2_relative"] <= 0.01, message
            assert diagnostic["cosine"] >= 0.999, message
            assert 0.99 <= diagnostic["norm_ratio"] <= 1.01, message
        torch.testing.assert_close(
            actual[section], reference[section], rtol=relative_rtol, atol=relative_atol
        )
    if allow_partition_variance:
        _assert_optimizer_state_parity(actual["optimizer_state"], reference["optimizer_state"])
    else:
        torch.testing.assert_close(actual["optimizer_state"], reference["optimizer_state"])


def test_joint_dcp_one_step_optimizer_matches_fixed_cp4_reference(monkeypatch, mdp_group_registry):
    native_bind = mdp_adapter_module._bind_dynamic_encoder_cp
    fixed = _run_optimizer_parity_mode(
        monkeypatch,
        dynamic_decoder=False,
        native_bind=native_bind,
        group_registry=mdp_group_registry,
    )
    assert integration.get_runtime() is None
    joint = _run_optimizer_parity_mode(
        monkeypatch,
        dynamic_decoder=True,
        native_bind=native_bind,
        group_registry=mdp_group_registry,
    )
    assert integration.get_runtime() is None
    _assert_optimizer_parity(joint, fixed)


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
@pytest.mark.parametrize(
    ("encoder_capacity", "expected_encoder_cp"), ((64, 1), (32, 2)), ids=("ecp1", "ecp2")
)
def test_selected_encoder_cp_optimizer_matches_ecp4_reference(
    monkeypatch, mdp_group_registry, dynamic_decoder, encoder_capacity, expected_encoder_cp
):
    native_bind = mdp_adapter_module._bind_dynamic_encoder_cp
    reference = _run_optimizer_parity_mode(
        monkeypatch,
        dynamic_decoder=dynamic_decoder,
        adam_eps=1e-6,
        native_bind=native_bind,
        group_registry=mdp_group_registry,
    )
    assert integration.get_runtime() is None
    selected = _run_optimizer_parity_mode(
        monkeypatch,
        dynamic_decoder=dynamic_decoder,
        encoder_capacity=encoder_capacity,
        expected_encoder_cp=expected_encoder_cp,
        adam_eps=1e-6,
        native_bind=native_bind,
        group_registry=mdp_group_registry,
    )
    assert integration.get_runtime() is None
    _assert_optimizer_parity(selected, reference, allow_partition_variance=True)


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
@pytest.mark.parametrize(
    ("vision", "encoder_capacity", "expected_encoder_cp"),
    ((True, 64, 1), (True, 32, 2), (True, 16, 4), (False, 16, None)),
    ids=("vision-ecp1", "vision-ecp2", "vision-ecp4", "text-empty-plan"),
)
def test_domain_ep4_public_qwen_execution_uses_native_moe(
    monkeypatch,
    mdp_group_registry,
    expert_replica_group,
    dynamic_decoder,
    vision,
    encoder_capacity,
    expected_encoder_cp,
):
    runtime = _runtime(
        group_registry=mdp_group_registry, encoder_capacity=encoder_capacity, expert_parallel_size=4
    )
    integration._RUNTIME = runtime
    domain_group = runtime.process_groups.encoder_cp_group
    domain_ranks = tuple(dist.get_process_group_ranks(domain_group))
    expected_domain = tuple(range((dist.get_rank() // 4) * 4, (dist.get_rank() // 4 + 1) * 4))
    assert domain_ranks == expected_domain
    assert runtime.dynamic_group_binding.expert_group is domain_group

    moe_pgs = ProcessGroupCollection(
        tp=runtime.process_groups.singleton_group,
        cp=domain_group,
        tp_cp=domain_group,
        ep=domain_group,
        expt_tp=runtime.process_groups.singleton_group,
        tp_ep=domain_group,
        expt_dp=expert_replica_group,
        tp_dp_cp=dist.group.WORLD,
    )
    decoder = _DomainEp4MoeDecoder(moe_pgs).bfloat16().cuda()
    assert decoder.moe.config.context_parallel_size == domain_group.size()
    assert decoder.moe.local_expert_indices == [dist.get_rank() % 4]
    assert (
        tuple(dist.get_process_group_ranks(decoder.moe.token_dispatcher.ep_group)) == domain_ranks
    )
    assert tuple(dist.get_process_group_ranks(expert_replica_group)) == (
        dist.get_rank() % 4,
        dist.get_rank() % 4 + 4,
    )

    selected_encoder_cp = []
    saw_vision_embeddings = []
    native_bind = mdp_adapter_module._bind_dynamic_encoder_cp
    native_decoder_forward = decoder.forward

    def observe_encoder_cp(*args, membership, **kwargs):
        selected_encoder_cp.append(membership.group_size)
        return native_bind(*args, membership=membership, **kwargs)

    def get_batch(iterator):
        next(iterator)
        return _batch(vision=vision)

    def observe_decoder_forward(**kwargs):
        saw_vision_embeddings.append(kwargs["vision_embeddings"] is not None)
        return native_decoder_forward(**kwargs)

    monkeypatch.setattr(forward_step, "get_batch", get_batch)
    monkeypatch.setattr(mdp_adapter_module, "_bind_dynamic_encoder_cp", observe_encoder_cp)
    monkeypatch.setattr(decoder, "forward", observe_decoder_forward)

    def finalize(_model, tokens):
        dist.all_reduce(tokens)

    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=16,
        finalize_model_grads_func=finalize,
    )

    def native_schedule(data_iterator, num_microbatches, forward_only):
        assert num_microbatches == 1 and forward_only is False
        output, output_loss_func = forward_step.forward_step(data_iterator, decoder)
        loss, tokens, _ = output_loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        assert int(tokens) == 128
        return loss.detach()

    wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
    loss = wrapped(data_iterator=iter((object(),)), num_microbatches=1, forward_only=False)

    assert torch.isfinite(loss)
    assert saw_vision_embeddings == [vision]
    selected = vision and dist.get_rank() % 4 < expected_encoder_cp
    assert selected_encoder_cp == ([expected_encoder_cp] if selected else [])
    expert_parameters = tuple(decoder.moe.experts.local_experts[0].parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in expert_parameters
    )
    encoder_gradients = tuple(
        parameter.main_grad for parameter in runtime.encoder_domain.encoder_ddp.parameters()
    )
    assert (
        any(
            gradient is not None and torch.count_nonzero(gradient) for gradient in encoder_gradients
        )
        is vision
    )
    assert runtime.iteration == 1
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime.storage.get_leaf(0) is None
