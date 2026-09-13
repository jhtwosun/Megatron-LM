# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3-VL vision encoder with canonical DeepStack output planes."""

import torch
from torch import Tensor

from examples.multimodal_dev.models.qwen3_vl.configuration import DEEPSTACK_VISUAL_INDEXES
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig


class Qwen3VLDeepStackPatchMerger(MegatronModule):
    """Qwen3-VL merger with final or DeepStack normalization order."""

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        use_postshuffle_norm: bool = True,
    ):
        super().__init__(config=config)
        merge_dim = hidden_size * spatial_merge_size**2
        self.merge_dim = merge_dim
        self.use_postshuffle_norm = use_postshuffle_norm
        self.activation_func = config.activation_func
        norm_size = merge_dim if use_postshuffle_norm else hidden_size
        self.patch_norm = TENorm(config=config, hidden_size=norm_size, eps=1e-6)
        self.linear_fc1 = build_module(
            ColumnParallelLinear,
            merge_dim,
            merge_dim,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
        )
        self.linear_fc2 = build_module(
            RowParallelLinear,
            merge_dim,
            out_hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.merge_dim)
        hidden_states = self.patch_norm(hidden_states)
        if not self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.merge_dim)
        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states, _ = self.linear_fc2(hidden_states)
        return hidden_states


class Qwen3VLVisionEncoder(Qwen35VLVisionEncoder):
    """Shared Qwen vision stack returning final then DeepStack 8/16/24 planes."""

    def __init__(self, *args, **kwargs):
        out_hidden_size = int(kwargs.get("out_hidden_size", 3584))
        super().__init__(*args, **kwargs)
        if self.config.num_layers <= max(DEEPSTACK_VISUAL_INDEXES):
            raise ValueError(
                "Qwen3-VL vision requires at least 25 layers for DeepStack indexes 8, 16, 24."
            )
        self.deepstack_visual_indexes = DEEPSTACK_VISUAL_INDEXES
        self.merger = Qwen3VLDeepStackPatchMerger(
            config=self.config,
            hidden_size=self.hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            use_postshuffle_norm=False,
        )
        self.decoder.deepstack_merger_list = torch.nn.ModuleList(
            [
                Qwen3VLDeepStackPatchMerger(
                    config=self.config,
                    hidden_size=self.hidden_size,
                    out_hidden_size=out_hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                )
                for _ in self.deepstack_visual_indexes
            ]
        )

    def forward(self, pixel_values: Tensor, grid_thw: Tensor):
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self._fast_pos_embed_interpolate(grid_thw)
        rot_freqs = self._compute_rotary_pos_emb(grid_thw)
        if getattr(self.config, "mrope_section", None) is None:
            rot_freqs = torch.cat((rot_freqs, rot_freqs), dim=-1).unsqueeze(1).unsqueeze(1)
        packed_seq_params = self._build_packed_seq_params(grid_thw)
        hidden_states, intermediate_states = self.decoder(
            hidden_states=hidden_states.unsqueeze(1),
            attention_mask=None,
            rotary_pos_emb=rot_freqs,
            packed_seq_params=packed_seq_params,
            extract_layer_indices=set(self.deepstack_visual_indexes),
        )
        hidden_states = hidden_states.squeeze(1)
        if len(intermediate_states) != len(self.deepstack_visual_indexes):
            raise RuntimeError(
                "Qwen3-VL vision DeepStack extraction did not return layers 8, 16, and 24."
            )
        final_plane = self.merger(hidden_states)
        deepstack_planes = tuple(
            merger(state.squeeze(1))
            for merger, state in zip(self.decoder.deepstack_merger_list, intermediate_states)
        )
        return (final_plane, *deepstack_planes)
