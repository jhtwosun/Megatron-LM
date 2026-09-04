# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical image-only Qwen3-VL model."""

from typing import Optional

import torch
from torch import Tensor

from examples.multimodal_dev.models.base import MultimodalModel
from examples.multimodal_dev.models.qwen3_vl.configuration import (
    IMAGE_TOKEN_ID,
    ROTARY_BASE,
    ROTARY_PERCENT,
    VIDEO_TOKEN_ID,
    VISION_KWARGS,
    VISION_START_TOKEN_ID,
    VOCAB_SIZE,
)
from examples.multimodal_dev.models.qwen3_vl.specs import get_qwen3_vl_vision_spec
from examples.multimodal_dev.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def prepare_qwen3_vl_decoder_inputs(
    *,
    input_ids: Tensor,
    text_embeddings: Tensor,
    output_planes: tuple,
    image_token_id: int,
    video_token_id: int,
):
    """Scatter the final plane and retain compact ordered DeepStack context."""
    if bool((input_ids == video_token_id).any()):
        raise ValueError("Qwen3-VL video inputs are not supported in this image-only path.")
    if torch.is_tensor(output_planes):
        if output_planes.ndim != 2 or output_planes.shape[1] % 4:
            raise ValueError("Qwen3-VL packed vision output must be [rows, 4 * decoder width].")
        output_planes = tuple(output_planes.chunk(4, dim=-1))
    if not isinstance(output_planes, tuple) or len(output_planes) != 4:
        raise ValueError("Qwen3-VL vision output must contain exactly four ordered planes.")
    if input_ids.ndim != 2 or text_embeddings.ndim != 3:
        raise ValueError("Qwen3-VL expects [batch, sequence] IDs and [sequence, batch, hidden].")
    image_mask = input_ids == image_token_id
    row_count = int(image_mask.sum())
    hidden_size = text_embeddings.shape[-1]
    for plane_index, plane in enumerate(output_planes):
        if plane.ndim != 2 or plane.shape[0] != row_count:
            raise ValueError(
                f"Qwen3-VL plane {plane_index} row count must equal {row_count} image slots."
            )
        if plane.shape[1] != hidden_size:
            raise ValueError(
                f"Qwen3-VL plane {plane_index} width must equal decoder width {hidden_size}."
            )

    combined = text_embeddings.transpose(0, 1).contiguous()
    mask_expanded = image_mask.unsqueeze(-1).expand_as(combined)
    combined = combined.masked_scatter(mask_expanded, output_planes[0].to(combined.dtype))
    decoder_input = combined.transpose(0, 1).contiguous()
    deepstack_context = torch.stack(output_planes[1:], dim=0)
    return decoder_input, deepstack_context, image_mask


class Qwen3VLModel(MultimodalModel):
    """Qwen3-VL with canonical MRoPE and four-plane DeepStack vision output."""

    def __init__(
        self,
        language_config: TransformerConfig,
        language_spec: ModuleSpec,
        vision_config: TransformerConfig,
        vision_spec: ModuleSpec = None,
        vocab_size: int = VOCAB_SIZE,
        max_sequence_length: int = 262144,
        image_token_id: int = IMAGE_TOKEN_ID,
        video_token_id: int = VIDEO_TOKEN_ID,
        vision_start_token_id: int = VISION_START_TOKEN_ID,
        spatial_merge_size: int = 2,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
        build_vision_encoder: bool = True,
    ):
        if vision_spec is None:
            vision_spec = get_qwen3_vl_vision_spec()
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = spatial_merge_size

        if pre_process and build_vision_encoder:
            vision_kwargs = dict(VISION_KWARGS)
            vision_kwargs["spatial_merge_size"] = spatial_merge_size
            vision_kwargs["out_hidden_size"] = language_config.hidden_size
            vision_encoder = Qwen3VLVisionEncoder(
                config=vision_config,
                transformer_layer_spec=vision_spec,
                in_channels=vision_kwargs["in_channels"],
                patch_size=vision_kwargs["patch_size"],
                temporal_patch_size=vision_kwargs["temporal_patch_size"],
                spatial_merge_size=vision_kwargs["spatial_merge_size"],
                out_hidden_size=vision_kwargs["out_hidden_size"],
                max_num_positions=vision_kwargs["max_num_positions"],
            )
        else:
            vision_encoder = None

        super().__init__(
            language_config=language_config,
            language_spec=language_spec,
            vision_encoder=vision_encoder,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            image_token_id=image_token_id,
            position_embedding_type="mrope",
            rotary_percent=ROTARY_PERCENT,
            rotary_base=ROTARY_BASE,
            mrope_section=language_config.mrope_section,
            mtp_block_spec=None,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    def compute_position_ids(
        self, input_ids: Tensor, image_grid_thw: Optional[Tensor] = None, packed_seq_params=None
    ) -> Tensor:
        if bool((input_ids == self.video_token_id).any()):
            raise ValueError("Qwen3-VL video inputs are not supported in this image-only path.")
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            packed_seq_params=packed_seq_params,
        )
        return position_ids

    def prepare_decoder_inputs(self, input_ids, text_embeddings, vision_embeddings):
        decoder_input, context, visual_mask = prepare_qwen3_vl_decoder_inputs(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            output_planes=vision_embeddings,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )
        return decoder_input, {"context": context, "context_mask": visual_mask}

    def forward(self, input_ids, *args, **kwargs):
        if input_ids is not None and bool((input_ids == self.video_token_id).any()):
            raise ValueError("Qwen3-VL video inputs are not supported in this image-only path.")
        return super().forward(input_ids, *args, **kwargs)
