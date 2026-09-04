# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical constants and RADIO configuration for Nemotron Omni."""

from typing import Optional

from megatron.core.activations import fast_gelu
from megatron.core.transformer.transformer_config import TransformerConfig

EXPANDED_SEQUENCE_CONTRACT = "expanded_sequence_v1"
IMAGE_TOKEN_ID = 18
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
CLASS_TOKEN_LEN = 10
PIXEL_PAYLOAD_WIDTH = 3 * PATCH_SIZE * PATCH_SIZE
PROJECTOR_FFN_HIDDEN_SIZE = 20480


def get_nemotron_omni_vision_config(
    num_layers_override: Optional[int] = None, variant: Optional[str] = None
) -> TransformerConfig:
    """Return the canonical dynamic-resolution RADIO Transformer config."""
    del variant
    return TransformerConfig(
        num_layers=num_layers_override or 32,
        hidden_size=1280,
        ffn_hidden_size=5120,
        num_attention_heads=16,
        num_query_groups=16,
        kv_channels=80,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-6,
        normalization="LayerNorm",
        add_bias_linear=True,
        add_qkv_bias=True,
        gated_linear_unit=False,
        activation_func=fast_gelu,
        bias_activation_fusion=False,
        apply_query_key_layer_scaling=False,
        attention_softmax_in_fp32=True,
        qk_layernorm=False,
        bf16=False,
    )
