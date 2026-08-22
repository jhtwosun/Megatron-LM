# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3-VL configuration validation and model factory."""

from examples.multimodal_dev.models.qwen3_vl.configuration import (
    IMAGE_TOKEN_ID,
    MROPE_SECTION,
    ROTARY_BASE,
    ROTARY_PERCENT,
    VIDEO_TOKEN_ID,
    VISION_START_TOKEN_ID,
)


def validate_qwen3_vl_support(args, language_config, vision_config) -> None:
    """Fail before model/runtime construction outside the M2 support boundary."""
    tp_size = int(getattr(language_config, "tensor_model_parallel_size", 1))
    if tp_size > 1 and getattr(language_config, "sequence_parallel", False):
        raise ValueError("Qwen3-VL M2 supports sequence parallel only in the degenerate TP=1 case.")
    if tp_size != 1:
        raise ValueError("Qwen3-VL M2 does not support native tensor parallel sharding.")
    if int(getattr(language_config, "context_parallel_size", 1)) != 1:
        raise ValueError("Qwen3-VL M2 does not support decoder context parallel sharding.")
    if getattr(language_config, "virtual_pipeline_model_parallel_size", None) is not None:
        raise ValueError("Qwen3-VL M2 does not support virtual pipeline parallelism.")
    if getattr(language_config, "pipeline_model_parallel_layout", None) is not None:
        raise ValueError("Qwen3-VL M2 does not support a custom pipeline layout.")
    if getattr(args, "mtp_num_layers", None) or getattr(language_config, "mtp_num_layers", None):
        raise ValueError("Qwen3-VL M2 does not support MTP with DeepStack context.")
    pp_size = int(getattr(language_config, "pipeline_model_parallel_size", 1))
    first_stage_layers = getattr(language_config, "num_layers_in_first_pipeline_stage", None)
    if first_stage_layers is None:
        if int(language_config.num_layers) % pp_size != 0:
            raise ValueError(
                "Qwen3-VL M2 non-interleaved PP requires an even layer split or an "
                "explicit first-stage layer count."
            )
        first_stage_layers = int(language_config.num_layers) // pp_size
    if getattr(language_config, "account_for_embedding_in_pipeline_split", False):
        first_stage_layers -= 1
    if int(first_stage_layers) < 3:
        raise ValueError(
            "Qwen3-VL non-interleaved PP requires PP0 to own the first three decoder layers."
        )
    if vision_config is not None:
        if int(vision_config.num_layers) <= 24:
            raise ValueError("Qwen3-VL vision requires at least 25 layers for DeepStack.")
        recompute_granularity = getattr(vision_config, "recompute_granularity", None)
        if recompute_granularity == "full":
            recompute_method = getattr(vision_config, "recompute_method", None)
            if recompute_method == "uniform":
                if getattr(vision_config, "recompute_num_layers", None) != 1:
                    raise ValueError(
                        "Qwen3-VL vision full-uniform recompute requires "
                        "recompute_num_layers=1 so layers 8, 16, and 24 are exact "
                        "checkpoint boundaries."
                    )
            elif recompute_method != "block":
                raise ValueError(
                    "Qwen3-VL vision full recompute supports only uniform or block methods."
                )


def post_language_config(language_config, args):
    """Apply only the canonical Qwen3-VL language settings."""
    language_config.mrope_section = list(MROPE_SECTION)
    language_config.mrope_interleaved = True
    language_config.rotary_percent = ROTARY_PERCENT
    language_config.rotary_base = ROTARY_BASE
    language_config.apply_rope_fusion = False
    language_config.linear_attention_freq = None
    language_config.qk_layernorm = True
    head_dim = getattr(language_config, "kv_channels", None)
    if head_dim is None and all(
        hasattr(language_config, name) for name in ("hidden_size", "num_attention_heads")
    ):
        head_dim = language_config.hidden_size // language_config.num_attention_heads
    if head_dim is not None and int(head_dim * ROTARY_PERCENT) != 2 * sum(MROPE_SECTION):
        raise ValueError(
            "Qwen3-VL canonical mRoPE sections [24, 20, 20] require rotary_dim=128; "
            f"got rotary_dim={int(head_dim * ROTARY_PERCENT)}."
        )
    args.image_token_id = IMAGE_TOKEN_ID
    args.video_token_id = VIDEO_TOKEN_ID
    args.vision_start_token_id = VISION_START_TOKEN_ID
    validate_qwen3_vl_support(args, language_config, None)


def build_model(
    args, language_config, vision_config, pre_process=True, post_process=True, **kwargs
):
    """Build the image-only canonical Qwen3-VL model."""
    from examples.multimodal_dev.models.qwen3_vl.model import Qwen3VLModel
    from examples.multimodal_dev.models.qwen3_vl.specs import get_qwen3_vl_language_spec
    from megatron.core import parallel_state

    validate_qwen3_vl_support(args, language_config, vision_config)
    vp_stage = kwargs.get("vp_stage", None)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    language_spec = get_qwen3_vl_language_spec(language_config, vp_stage=vp_stage, pp_rank=pp_rank)
    return Qwen3VLModel(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        build_vision_encoder=not getattr(args, "mdp_enable", False),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        parallel_output=True,
        share_embeddings_and_output_weights=not getattr(
            args, "untie_embeddings_and_output_weights", False
        ),
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )
