# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical Qwen3-VL model support for multimodal_dev."""

from examples.multimodal_dev.models.qwen3_vl.configuration import (
    DEEPSTACK_VISUAL_INDEXES,
    IMAGE_TOKEN_ID,
    MROPE_SECTION,
    VIDEO_TOKEN_ID,
    VISION_START_TOKEN_ID,
    get_qwen3_vl_vision_config,
)
from examples.multimodal_dev.models.qwen3_vl.model import Qwen3VLModel

__all__ = [
    "DEEPSTACK_VISUAL_INDEXES",
    "IMAGE_TOKEN_ID",
    "MROPE_SECTION",
    "Qwen3VLModel",
    "VIDEO_TOKEN_ID",
    "VISION_START_TOKEN_ID",
    "get_qwen3_vl_vision_config",
]
