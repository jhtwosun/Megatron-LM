# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Nemotron RADIO transformer parity in two independent D4 domains."""

import os

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev.models.nemotron_omni.vision_encoder import (
    _NemotronEncoderCpRADIOViTModel,
)
from examples.multimodal_dev.tests.test_mdp_encoder_cp_nemotron import _packed
from examples.multimodal_dev.tests.test_mdp_encoder_cp_qwen35_world8 import encoder_cp_groups
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8
pytestmark = pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")


def _config(cp_size):
    return TransformerConfig(
        num_layers=1,
        hidden_size=32,
        ffn_hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        context_parallel_size=cp_size,
        sequence_parallel=False,
    )


def _build_model(cp_size, group=None):
    torch.manual_seed(2026)
    model_parallel_cuda_manual_seed(2026)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    groups.cp = group
    return (
        _NemotronEncoderCpRADIOViTModel(
            _config(cp_size),
            get_gpt_layer_with_transformer_engine_spec(),
            img_h=64,
            img_w=64,
            max_img_h=64,
            max_img_w=64,
            class_token_len=10,
            patch_dim=16,
            add_class_token=True,
            embedder_bias=False,
            dynamic_resolution=True,
            pg_collection=groups,
            encoder_cp_group=group,
        )
        .bfloat16()
        .cuda()
    )


def _reduced_clone(value, group):
    result = value.detach().clone()
    dist.all_reduce(result, group=group)
    return result


@pytest.mark.parametrize("cp_size", (2, 4))
def test_actual_radio_transformer_cp_matches_e1_in_each_d4_domain(cp_size, encoder_cp_groups):
    group = encoder_cp_groups[cp_size]
    group_rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    reference = _build_model(1)
    candidate = _build_model(cp_size, group)
    candidate.load_state_dict(reference.state_dict())

    generator = torch.Generator(device=device).manual_seed(17)
    reference_input = torch.randn(
        1, 48, 32, dtype=torch.bfloat16, device=device, generator=generator, requires_grad=True
    )
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    reference_output = reference._forward_transformer(reference_input, None, _packed(device))
    candidate_output = candidate._forward_transformer(candidate_input, None, _packed(device))
    torch.testing.assert_close(candidate_output, reference_output, rtol=0.02, atol=0.02)

    reference_output.float().square().sum().backward()
    candidate_loss = (
        candidate_output.float().square().sum()
        if group_rank == 0
        else candidate_output.float().sum() * 0
    )
    candidate_loss.backward()
    torch.testing.assert_close(
        _reduced_clone(candidate_input.grad, group), reference_input.grad, rtol=0.03, atol=0.03
    )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        if parameter.grad is None:
            assert reference_parameters[name].grad is None
            continue
        torch.testing.assert_close(
            _reduced_clone(parameter.grad, group),
            reference_parameters[name].grad,
            rtol=0.03,
            atol=0.03,
            msg=name,
        )
