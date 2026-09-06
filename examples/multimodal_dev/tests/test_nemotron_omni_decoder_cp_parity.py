# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual WORLD4 Nemotron Omni decoder-CP parity at the native schedule boundary."""

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from examples.multimodal_dev.models.base import (
    split_multimodal_inputs_for_context_parallel,
)
from examples.multimodal_dev.models.nemotron_omni.configuration import IMAGE_TOKEN_ID
from examples.multimodal_dev.models.nemotron_omni.model import NemotronOmniModel
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


_WORLD_SIZE = 4
_TOTAL_ROWS = 16
_VOCAB_SIZE = 64
_SEED = 19937
_ATOL = 4.0e-2
_RTOL = 6.0e-2


def _require_world4():
    if int(os.environ.get("WORLD_SIZE", "1")) != _WORLD_SIZE:
        pytest.skip("Nemotron Omni decoder-CP parity requires torchrun WORLD_SIZE=4")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def _initialize_parallel(cp_size):
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    model_parallel_cuda_manual_seed(_SEED)
    return ProcessGroupCollection.use_mpu_process_groups()


def _config(cp_size, pattern):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=32,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1 if kind == "G" else 0 for kind in pattern],
        linear_cp_mode="chunkwise",
        gdn_pre_gated_delta_rule_fusion=False,
        deterministic_mode=False,
        transformer_impl="transformer_engine",
    )
    config.hybrid_layer_pattern = pattern
    config.finalize_model_grads_func = finalize_model_grads
    return config


def _build_model(cp_size, pattern, pg_collection, state=None):
    config = _config(cp_size, pattern)
    model = NemotronOmniModel(
        language_config=config,
        language_spec=hybrid_stack_spec,
        vision_config=SimpleNamespace(hidden_size=32),
        vision_spec=None,
        projection_config=config,
        projection_submodules=None,
        hybrid_layer_pattern=pattern,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_TOTAL_ROWS,
        image_token_id=IMAGE_TOKEN_ID,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        build_vision_encoder=False,
        pg_collection=pg_collection,
    ).cuda()
    if state is not None:
        model.load_state_dict(state, strict=True)
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
        ),
        module=model,
        pg_collection=pg_collection,
    )
    return ddp


def _packed(cp_size, cp_group):
    padded = torch.tensor((0, 8, 16), dtype=torch.int32, device="cuda")
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=padded.clone(),
        cu_seqlens_kv=padded.clone(),
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded.clone(),
        max_seqlen_q=8,
        max_seqlen_kv=8,
        local_cp_size=cp_size,
        cp_group=cp_group,
        total_tokens=_TOTAL_ROWS,
        cp_partition_mode="contiguous",
    )


def _batch(case, cp_size, cp_group):
    generator = torch.Generator(device="cpu").manual_seed(_SEED + 1)
    input_ids = torch.randint(1, _VOCAB_SIZE, (1, _TOTAL_ROWS), generator=generator).cuda()
    input_ids.masked_fill_(input_ids == IMAGE_TOKEN_ID, 1)
    labels = torch.randint(0, _VOCAB_SIZE, (1, _TOTAL_ROWS), generator=generator).cuda()
    loss_mask = torch.zeros((1, _TOTAL_ROWS), dtype=torch.float32, device="cuda")
    # Two unequal logical records occupy distinct physical [0,8) and [8,16) regions.
    loss_mask[0, :5] = 1
    loss_mask[0, 8:14] = 1
    vision_embeddings = None
    if case == "image":
        input_ids[0, (1, 3, 9)] = IMAGE_TOKEN_ID
        vision_embeddings = torch.randn(
            (3, 128), generator=generator, dtype=torch.float32
        ).cuda().to(torch.bfloat16)
        vision_embeddings.requires_grad_(True)
    else:
        assert not torch.any(input_ids == IMAGE_TOKEN_ID)
    return {
        "input_ids": input_ids,
        "position_ids": torch.arange(8, device="cuda").repeat(2).view(1, -1),
        "labels": labels,
        "loss_mask": loss_mask,
        "packed_seq_params": _packed(cp_size, cp_group),
        "vision_embeddings": vision_embeddings,
    }


def _projected_rows(batch):
    row_ids = torch.arange(_TOTAL_ROWS, device="cuda", dtype=torch.int64).view(
        _TOTAL_ROWS, 1, 1
    )
    local, *_ = split_multimodal_inputs_for_context_parallel(
        decoder_input=row_ids,
        input_ids=None,
        labels=None,
        loss_mask=None,
        attention_mask=None,
        position_ids=batch["position_ids"],
        packed_seq_params=batch["packed_seq_params"],
        sequence_parallel=False,
        padding_mask=None,
    )
    return local.flatten().long()


def _run_schedule(model, batch, pg_collection):
    captured = {}
    local_rows = _projected_rows(batch)
    local_mask = batch["loss_mask"].index_select(1, local_rows)
    original_finalize = model.config.finalize_model_grads_func

    def finalizer(models, num_tokens, **kwargs):
        captured["num_tokens"] = num_tokens
        captured["token_value_before_finalize"] = int(num_tokens.item())
        return original_finalize(models, num_tokens, **kwargs)

    model.config.finalize_model_grads_func = finalizer

    def capture_decoder_input(_module, _inputs, output):
        output.retain_grad()
        captured["decoder_input"] = output

    hook = model.module.language_model.embedding.register_forward_hook(capture_decoder_input)
    layer_hooks = []
    captured["layer_groups"] = []

    def capture_layer_group(module, _inputs, kwargs):
        packed = kwargs["packed_seq_params"]
        captured["layer_groups"].append(
            (type(module), module.pg_collection.cp, packed.cp_group)
        )

    for module in model.module.modules():
        if type(module) in (SelfAttention, GatedDeltaNet):
            layer_hooks.append(
                module.register_forward_pre_hook(capture_layer_group, with_kwargs=True)
            )

    def forward_step(data_iterator, scheduled_model):
        scheduled_batch = next(data_iterator)
        output = scheduled_model(**scheduled_batch)
        captured["output"] = output.detach().float()

        def loss_func(per_token_loss):
            loss = (per_token_loss.float() * local_mask).sum()
            tokens = local_mask.sum().to(dtype=torch.int)
            return loss, tokens, {"lm loss": loss.detach().view(1)}

        return output, loss_func

    model.zero_grad_buffer()
    try:
        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter((batch,)),
            model=model,
            num_microbatches=1,
            seq_length=_TOTAL_ROWS,
            micro_batch_size=1,
            forward_only=False,
            pg_collection=pg_collection,
        )
    finally:
        hook.remove()
        for layer_hook in layer_hooks:
            layer_hook.remove()
    captured["local_rows"] = local_rows
    loss_numerator = (captured["output"] * local_mask).sum().detach()
    logical_tokens = torch.tensor(
        captured["token_value_before_finalize"], dtype=torch.int, device="cuda"
    )
    dist.all_reduce(loss_numerator, group=pg_collection.cp)
    dist.all_reduce(logical_tokens, group=pg_collection.cp)
    captured["loss"] = loss_numerator / logical_tokens.clamp(min=1)
    captured["parameter_grads"] = {
        name: parameter.main_grad.detach().float().cpu().clone()
        for name, parameter in model.module.named_parameters()
        if parameter.requires_grad
    }
    decoder_input_grad = captured["decoder_input"].grad.detach().float()
    assert decoder_input_grad.shape[0] == _TOTAL_ROWS
    captured["decoder_input_grad"] = decoder_input_grad.index_select(0, local_rows).cpu()
    del captured["decoder_input"]
    vision = batch["vision_embeddings"]
    if vision is not None:
        vision_grad = vision.grad.detach().float().clone()
        dist.all_reduce(vision_grad, group=pg_collection.cp)
        captured["vision_grad"] = vision_grad.cpu()
    return captured


def _cpu_state(model):
    return {
        name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in model.module.state_dict().items()
    }


@pytest.mark.internal
@pytest.mark.parametrize("pattern", ("G*", "*G"))
@pytest.mark.parametrize("case", ("image", "text"))
def test_nemotron_omni_native_schedule_cp4_matches_independently_reset_cp1(pattern, case):
    _require_world4()
    reference = trial = None
    try:
        reference_pgs = _initialize_parallel(1)
        assert reference_pgs.cp.size() == 1
        assert parallel_state.get_context_parallel_world_size() == 1
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed_all(_SEED)
        reference = _build_model(1, pattern, reference_pgs)
        reference_batch = _batch(case, 1, reference_pgs.cp)
        state = _cpu_state(reference)
        reference_result = _run_schedule(reference, reference_batch, reference_pgs)
        del reference
        reference = None
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()

        trial_pgs = _initialize_parallel(4)
        assert trial_pgs.cp is parallel_state.get_context_parallel_group()
        assert trial_pgs.cp.size() == 4
        assert tuple(dist.get_process_group_ranks(trial_pgs.cp)) == (0, 1, 2, 3)
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed_all(_SEED)
        trial = _build_model(4, pattern, trial_pgs, state=state)
        trial_batch = _batch(case, 4, trial_pgs.cp)
        assert torch.equal(trial_batch["input_ids"], reference_batch["input_ids"])
        assert torch.equal(trial_batch["labels"], reference_batch["labels"])
        assert torch.equal(trial_batch["loss_mask"], reference_batch["loss_mask"])
        if case == "image":
            assert torch.equal(
                trial_batch["vision_embeddings"], reference_batch["vision_embeddings"]
            )
        assert trial_batch["packed_seq_params"].cu_seqlens_q_padded.tolist() == [0, 8, 16]
        trial_result = _run_schedule(trial, trial_batch, trial_pgs)

        for result, group in (
            (reference_result, reference_pgs.cp),
            (trial_result, trial_pgs.cp),
        ):
            assert [kind for kind, _module_group, _packed_group in result["layer_groups"]] == [
                GatedDeltaNet if kind == "G" else SelfAttention for kind in pattern
            ]
            assert all(
                module_group is group and packed_group is group
                for _kind, module_group, packed_group in result["layer_groups"]
            )

        rows = trial_result["local_rows"].cpu()
        expected_output = reference_result["output"].flatten().cpu().index_select(0, rows)
        torch.testing.assert_close(
            trial_result["output"].flatten().cpu(), expected_output, atol=_ATOL, rtol=_RTOL
        )
        torch.testing.assert_close(
            trial_result["loss"].cpu(), reference_result["loss"].cpu(), atol=_ATOL, rtol=_RTOL
        )
        assert torch.isfinite(trial_result["decoder_input_grad"]).all()
        assert torch.count_nonzero(trial_result["decoder_input_grad"])
        expected_input_grad = reference_result["decoder_input_grad"].index_select(0, rows)
        torch.testing.assert_close(
            trial_result["decoder_input_grad"],
            expected_input_grad,
            atol=7.0e-2,
            rtol=8.0e-2,
        )
        assert reference_result["token_value_before_finalize"] == 11
        assert trial_result["token_value_before_finalize"] == (4, 1, 4, 2)[dist.get_rank()]
        assert int(reference_result["num_tokens"].item()) == 44
        assert int(trial_result["num_tokens"].item()) == 11

        assert trial_result["parameter_grads"].keys() == reference_result[
            "parameter_grads"
        ].keys()
        for name, expected in reference_result["parameter_grads"].items():
            actual = trial_result["parameter_grads"][name]
            assert torch.isfinite(actual).all(), name
            torch.testing.assert_close(actual, expected, atol=7.0e-2, rtol=8.0e-2)
        if case == "image":
            torch.testing.assert_close(
                trial_result["vision_grad"],
                reference_result["vision_grad"],
                atol=7.0e-2,
                rtol=8.0e-2,
            )
    finally:
        del reference, trial
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()
    assert not parallel_state.model_parallel_is_initialized()
