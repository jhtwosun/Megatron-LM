# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual scaled TE acceptance for canonical Qwen3-VL DeepStack."""

from functools import partial

import pytest
import torch

from examples.multimodal_dev.models.qwen3_vl.configuration import MROPE_SECTION
from examples.multimodal_dev.models.qwen3_vl.factory import post_language_config
from examples.multimodal_dev.models.qwen3_vl.model import Qwen3VLModel
from examples.multimodal_dev.models.qwen3_vl.specs import (
    get_qwen3_vl_language_spec,
    get_qwen3_vl_vision_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

IMAGE_TOKEN_ID = 7
VIDEO_TOKEN_ID = 8
VISION_START_TOKEN_ID = 6
VOCAB = 128
SEQ = 16
LANGUAGE_HIDDEN = 128


def _language_config():
    config = TransformerConfig(
        num_layers=4,
        hidden_size=LANGUAGE_HIDDEN,
        ffn_hidden_size=256,
        num_attention_heads=1,
        num_query_groups=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        calculate_per_token_loss=True,
    )
    post_language_config(config, type("Args", (), {"mtp_num_layers": None})())
    assert config.kv_channels == 128
    assert sum(config.mrope_section) == config.kv_channels // 2
    return config


def _vision_config(recompute):
    kwargs = dict(
        num_layers=25,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=2,
        num_query_groups=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        activation_func=partial(torch.nn.functional.gelu, approximate="tanh"),
    )
    if recompute == "selective":
        kwargs.update(recompute_granularity="selective", recompute_modules=["core_attn"])
    elif recompute == "block":
        kwargs.update(
            recompute_granularity="full", recompute_method="block", recompute_num_layers=25
        )
    elif recompute == "uniform":
        kwargs.update(
            recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
        )
    config = TransformerConfig(**kwargs)
    config.mrope_section = [0, 8, 8]
    config.mrope_interleaved = False
    config.apply_rope_fusion = False
    config.deepstack_visual_indexes = [8, 16, 24]
    return config


def _batch(with_image, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    input_ids = torch.randint(9, VOCAB, (1, SEQ), generator=generator, device="cuda")
    labels = torch.randint(9, VOCAB, (1, SEQ), generator=generator, device="cuda")
    loss_mask = torch.ones(1, SEQ, device="cuda")
    if not with_image:
        return input_ids, labels, loss_mask, None, None

    input_ids[0, 1] = VISION_START_TOKEN_ID
    input_ids[0, 2:6] = IMAGE_TOKEN_ID
    loss_mask[0, 1:6] = 0
    pixels = torch.randn(
        16, 3 * 2 * 16 * 16, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    grid = torch.tensor([[1, 4, 4]], device="cuda")
    return input_ids, labels, loss_mask, pixels, grid


@pytest.mark.parametrize("vision_recompute", [None, "selective", "block", "uniform"])
def test_actual_25_layer_vision_and_four_layer_decoder_image_plus_text_backward(vision_recompute):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        language_config = _language_config()
        vision_config = _vision_config(vision_recompute)
        model = (
            Qwen3VLModel(
                language_config=language_config,
                language_spec=get_qwen3_vl_language_spec(language_config),
                vision_config=vision_config,
                vision_spec=get_qwen3_vl_vision_spec(),
                vocab_size=VOCAB,
                max_sequence_length=SEQ,
                image_token_id=IMAGE_TOKEN_ID,
                video_token_id=VIDEO_TOKEN_ID,
                vision_start_token_id=VISION_START_TOKEN_ID,
                parallel_output=False,
                pre_process=True,
                post_process=True,
            )
            .bfloat16()
            .cuda()
        )
        model.train()

        captured_planes = []

        def capture_planes(_module, _inputs, output):
            assert isinstance(output, tuple) and len(output) == 4
            captured_planes.extend(output)
            for plane in output:
                plane.retain_grad()

        hook = model.vision_model.register_forward_hook(capture_planes)
        image = _batch(True, 41)
        text = _batch(False, 42)
        image_loss = model(
            input_ids=image[0],
            position_ids=None,
            labels=image[1],
            loss_mask=image[2],
            pixel_values=image[3],
            image_grid_thw=image[4],
        )
        text_loss = model(
            input_ids=text[0],
            position_ids=None,
            labels=text[1],
            loss_mask=text[2],
            pixel_values=None,
            image_grid_thw=None,
        )
        loss = (image_loss * image[2]).sum() + (text_loss * text[2]).sum()
        assert torch.isfinite(loss)
        loss.backward()
        hook.remove()

        assert language_config.mrope_section == list(MROPE_SECTION)
        assert language_config.apply_rope_fusion is False
        assert len(model.vision_model.decoder.layers) == 25
        assert len(model.language_model.decoder.layers) == 4
        assert len(captured_planes) == 4
        assert all(tuple(plane.shape) == (4, LANGUAGE_HIDDEN) for plane in captured_planes)
        assert all(
            plane.grad is not None and torch.isfinite(plane.grad).all() for plane in captured_planes
        )
        plane_grad_sums = [float(plane.grad.abs().sum()) for plane in captured_planes]
        assert all(value > 0 for value in plane_grad_sums), plane_grad_sums

        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        missing = [name for name, parameter in trainable.items() if parameter.grad is None]
        assert not missing, f"missing trainable gradients: {missing}"
        assert all(torch.isfinite(parameter.grad).all() for parameter in trainable.values())
        merger_names = [
            name
            for name in trainable
            if "deepstack_merger_list" in name and name.endswith("weight")
        ]
        assert merger_names
        assert all(float(trainable[name].grad.abs().sum()) > 0 for name in merger_names)
        state_keys = set(model.state_dict())
        deepstack_keys = {name for name in state_keys if "deepstack_merger_list" in name}
        assert deepstack_keys
        assert all(
            name.startswith("vision_model.decoder.deepstack_merger_list.")
            for name in deepstack_keys
        )
        assert not any("deepstack" in name for name, _ in model.language_model.named_parameters())
    finally:
        Utils.destroy_model_parallel()
