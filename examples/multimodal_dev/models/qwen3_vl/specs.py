# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""All-SDPA language and vision specs for Qwen3-VL."""

from typing import Optional

import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


def _inject_deepstack_plane(
    hidden_states: torch.Tensor, context: torch.Tensor, visual_mask: torch.Tensor, plane_index: int
) -> torch.Tensor:
    """Out-of-place add of one compact plane at visual-token positions."""
    if context.ndim != 3 or context.shape[0] != 3:
        raise ValueError("Qwen3-VL DeepStack context must have shape [3, visual_rows, hidden].")
    if visual_mask.ndim != 2 or visual_mask.dtype != torch.bool:
        raise ValueError("Qwen3-VL visual context_mask must be a [batch, sequence] bool tensor.")
    hidden_bsh = hidden_states.transpose(0, 1)
    if tuple(visual_mask.shape) != tuple(hidden_bsh.shape[:2]):
        raise ValueError(
            "Qwen3-VL visual context_mask shape must match decoder batch/sequence dimensions."
        )
    plane = context[plane_index]
    row_count = int(visual_mask.sum())
    if tuple(plane.shape) != (row_count, hidden_states.shape[-1]):
        raise ValueError(
            "Qwen3-VL DeepStack plane row/width mismatch: expected "
            f"({row_count}, {hidden_states.shape[-1]}), got {tuple(plane.shape)}."
        )
    flat_hidden = hidden_bsh.reshape(-1, hidden_states.shape[-1])
    row_ids = visual_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    flat_hidden = flat_hidden.index_add(0, row_ids, plane.to(flat_hidden.dtype))
    return flat_hidden.view_as(hidden_bsh).transpose(0, 1).contiguous()


class Qwen3VLTransformerLayer(TransformerLayer):
    """Stock GPT layer plus canonical DeepStack injection after layers 0/1/2."""

    def forward(self, *args, **kwargs):
        context = kwargs.get("context")
        visual_mask = kwargs.get("context_mask")
        hidden_states, returned_context = super().forward(*args, **kwargs)
        global_layer_index = self.layer_number - 1
        if context is not None and global_layer_index < 3:
            if returned_context is not context:
                raise RuntimeError(
                    "Qwen3-VL requires IdentityOp cross-attention to preserve DeepStack context."
                )
            if visual_mask is None:
                raise ValueError("Qwen3-VL DeepStack context requires a visual context_mask.")
            hidden_states = _inject_deepstack_plane(
                hidden_states, context, visual_mask, global_layer_index
            )
        return hidden_states, returned_context


def _install_qwen3_vl_layer(block_spec):
    """Install the parameter-free layer wrapper after validating cross-attention."""
    for layer_spec in block_spec.layer_specs:
        if layer_spec.submodules.cross_attention is not IdentityOp:
            raise ValueError(
                "Qwen3-VL DeepStack requires IdentityOp cross-attention in the all-SDPA GPT spec."
            )
        if not issubclass(layer_spec.module, TransformerLayer):
            raise ValueError("Qwen3-VL language spec must contain TransformerLayer modules.")
        layer_spec.module = Qwen3VLTransformerLayer
    return block_spec


def get_qwen3_vl_language_spec(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
):
    """Return the stock Transformer Engine all-SDPA block with DeepStack layers."""
    block_spec = get_gpt_decoder_block_spec(
        config=config, use_transformer_engine=True, vp_stage=vp_stage, pp_rank=pp_rank
    )
    return _install_qwen3_vl_layer(block_spec)


def get_qwen3_vl_vision_spec():
    """Return the stock Transformer Engine ViT layer spec."""
    from examples.multimodal_dev.models.qwen35_vl.specs import Qwen35VLVisionSelfAttention

    spec = get_vit_layer_with_transformer_engine_spec()
    spec.submodules.self_attention.module = Qwen35VLVisionSelfAttention
    return spec
