# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Layer spec helpers for Qwen3.5-VL vision encoder and language decoder.

Provides ModuleSpec builders that define the transformer layer composition.
Both the standalone and MIMO training paths import from here.
"""

import dataclasses
import math
from typing import Optional

import torch.nn.functional as F

from examples.multimodal_dev.models.base import _NO_CP_GROUP
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


def _apply_rope_fp32(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
    """Apply rotary positional embedding in fp32, then cast back to original dtype.

    Mirrors ``Qwen3VLSelfAttention.apply_rotary_pos_emb_absolute`` in Megatron-Bridge
    with ``apply_rotary_pos_emb_in_fp32=True``.
    """
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_bshd,
        _apply_rotary_pos_emb_thd,
    )

    orig_dtype = t.dtype
    t_fp32 = t.float()

    if cu_seqlens is None:
        out = _apply_rotary_pos_emb_bshd(
            t_fp32,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=getattr(config, 'multi_latent_attention', False),
            mscale=mscale,
        )
    else:
        if (
            t_fp32.dim() == 3
            and freqs.dim() >= 1
            and freqs.size(0) == t_fp32.size(0)
        ):
            # The vision encoder builds frequencies in exact packed-token
            # order. Treating that tensor as BSHD avoids the generic THD
            # helper's cu_seqlens host synchronization while preserving the
            # same elementwise rotary calculation used by b436.
            out = _apply_rotary_pos_emb_bshd(
                t_fp32.unsqueeze(1),
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=getattr(
                    config, 'multi_latent_attention', False
                ),
                mscale=mscale,
            ).squeeze(1)
        else:
            if cp_group is None:
                cp_group = parallel_state.get_context_parallel_group()
            out = _apply_rotary_pos_emb_thd(
                t_fp32,
                cu_seqlens,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=getattr(
                    config, 'multi_latent_attention', False
                ),
                mscale=mscale,
                cp_group=cp_group,
            )
    return out.to(orig_dtype)


def _apply_rope_fp32_no_cp(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
    """Same as ``_apply_rope_fp32`` but forces CP-size=1.

    The vision encoder uses THD packed sequences for variable-resolution
    images.  When the language model uses CP>1, the global CP group would
    incorrectly split the vision seqlens.  This wrapper substitutes a
    trivial group so the vision RoPE sees the full packed sequence.
    """
    return _apply_rope_fp32(
        t, freqs, config, cu_seqlens, mscale, cp_group=_NO_CP_GROUP,
    )


class Qwen35VLVisionSelfAttention(SelfAttention):
    """ViT self-attention with RoPE applied in fp32.

    Matches Bridge's ``Qwen3VLSelfAttention`` behaviour when
    ``apply_rotary_pos_emb_in_fp32=True``:  query and key are cast to float32
    before the rotary multiply and cast back to bf16 afterwards.  The
    monkey-patch approach avoids duplicating the 300-line ``SelfAttention.forward``
    while keeping the change local to this class.
    """

    def forward(self, *args, **kwargs):
        import megatron.core.transformer.attention as _attn_mod

        _orig = _attn_mod.apply_rotary_pos_emb
        _attn_mod.apply_rotary_pos_emb = _apply_rope_fp32_no_cp
        try:
            return super().forward(*args, **kwargs)
        finally:
            _attn_mod.apply_rotary_pos_emb = _orig


def get_qwen35_vl_language_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Transformer block spec for the Qwen3.5-VL language decoder.

    Uses the experimental attention variant infrastructure to build hybrid
    GatedDeltaNet + full-attention layers with optional MoE interleaving.

    Args:
        config: Language decoder TransformerConfig.
        vp_stage: Virtual pipeline stage.
        pp_rank: Pipeline parallel rank.

    Returns:
        TransformerBlockSubmodules with per-layer specs.
    """
    return get_transformer_block_with_experimental_attention_variant_spec(
        config=config,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )


class PaddedHeadDimDotProductAttention(TEDotProductAttention):
    """Reference fast-pass attention-only padding; projections keep their real size.

    Adapted from BestJuly 5885d65a9c, qwen35_vl/specs.py. Zero padding
    preserves QK products only if softmax scaling retains the real head size.
    """

    def __init__(self, config, *args, **kwargs):
        self._real_kv = config.kv_channels
        self._pad_kv = 128 if self._real_kv == 72 else self._real_kv
        self._pad_width = self._pad_kv - self._real_kv
        if self._pad_width:
            config = dataclasses.replace(config, kv_channels=self._pad_kv)
            if kwargs.get("softmax_scale") is None:
                kwargs["softmax_scale"] = 1.0 / math.sqrt(self._real_kv)
        super().__init__(config, *args, **kwargs)

    def forward(self, query, key, value, *args, **kwargs):
        if not self._pad_width:
            return super().forward(query, key, value, *args, **kwargs)
        query, key, value = (F.pad(t, (0, self._pad_width)) for t in (query, key, value))
        out = super().forward(query, key, value, *args, **kwargs)
        return out.view(*out.shape[:-1], -1, self._pad_kv)[..., :self._real_kv].reshape(
            *out.shape[:-1], -1
        )


def get_qwen35_vl_vision_spec() -> ModuleSpec:
    """ModuleSpec for vision encoder transformer layers.

    Uses ``TEDotProductAttention`` which supports packed-sequence (THD)
    attention via ``PackedSeqParams`` for variable-length images.

    ``Qwen35VLVisionSelfAttention`` replaces the default ``SelfAttention`` so
    that RoPE is applied in fp32, matching Bridge's
    ``apply_rotary_pos_emb_in_fp32=True`` behaviour.
    """
    spec = get_vit_layer_with_transformer_engine_spec()
    spec.submodules.self_attention.module = Qwen35VLVisionSelfAttention
    spec.submodules.self_attention.submodules.core_attention = PaddedHeadDimDotProductAttention
    return spec
