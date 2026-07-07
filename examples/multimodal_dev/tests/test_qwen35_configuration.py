# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    MROPE_SECTION,
    get_qwen35_vl_language_config,
    get_qwen35_vl_vision_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import post_language_config


def test_language_post_config_disables_incompatible_rope_fusion():
    args = SimpleNamespace()
    config = SimpleNamespace(
        mrope_section=None,
        mrope_interleaved=False,
        apply_rope_fusion=True,
        attention_softmax_in_fp32=False,
    )

    post_language_config(config, args)

    assert config.mrope_section == MROPE_SECTION
    assert config.mrope_interleaved is True
    assert config.apply_rope_fusion is False
    assert config.attention_softmax_in_fp32 is True


def test_direct_configs_match_reference_precision_settings():
    language = get_qwen35_vl_language_config(
        "35b_a3b", moe_permute_fusion=False
    )
    vision = get_qwen35_vl_vision_config(variant="35b_a3b")

    assert language.attention_softmax_in_fp32 is True
    assert language.apply_rope_fusion is False
    assert vision.attention_softmax_in_fp32 is True
    assert vision.apply_rope_fusion is False
    assert vision.bias_activation_fusion is False
    assert vision.bias_dropout_fusion is False
