# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical constants and vision configuration for Qwen3-VL."""

from typing import Optional

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    VISION_KWARGS as _SHARED_VISION_KWARGS,
)
from examples.multimodal_dev.models.qwen35_vl.configuration import get_qwen35_vl_vision_config

VISION_START_TOKEN_ID = 151652
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VOCAB_SIZE = 151936

ROTARY_BASE = 5_000_000
ROTARY_PERCENT = 1.0
MROPE_SECTION = (24, 20, 20)
DEEPSTACK_VISUAL_INDEXES = (8, 16, 24)

# Qwen3-VL and Qwen3.5-VL share the patch geometry. Keep a private copy so
# model-specific output widths cannot mutate the Qwen3.5 registry constants.
VISION_KWARGS = dict(_SHARED_VISION_KWARGS)


def get_qwen3_vl_vision_config(
    num_layers_override: Optional[int] = None, variant: Optional[str] = None
):
    """Build the canonical Qwen3-VL ViT config.

    Qwen3-VL uses the shared 27-layer, 1152-hidden vision stack. Language
    variants do not select a different vision backbone.
    """
    del variant
    config = get_qwen35_vl_vision_config(num_layers_override=num_layers_override, variant=None)
    if config.num_layers <= max(DEEPSTACK_VISUAL_INDEXES):
        raise ValueError(
            "Qwen3-VL vision requires layer indexes 8, 16, and 24; "
            f"got num_layers={config.num_layers}."
        )
    config.deepstack_visual_indexes = list(DEEPSTACK_VISUAL_INDEXES)
    return config
