# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configuration validation and construction for image-only Nemotron Omni."""

import copy

from examples.multimodal_dev.models.nemotron_omni.configuration import (
    EXPANDED_SEQUENCE_CONTRACT,
    IMAGE_TOKEN_ID,
    PROJECTOR_FFN_HIDDEN_SIZE,
)
from megatron.core.activations import squared_relu
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import get_submodules


def get_nemotron_omni_specs(hybrid_layer_pattern):
    """Return canonical language/vision specs and an independent projector spec."""
    if not hybrid_layer_pattern:
        raise ValueError("Nemotron Omni requires a nonempty hybrid layer pattern.")
    language_submodules = get_submodules(hybrid_stack_spec)
    mlp_layer_submodules = get_submodules(language_submodules.mlp_layer)
    projector_submodules = copy.deepcopy(get_submodules(mlp_layer_submodules.mlp))
    return (hybrid_stack_spec, get_vit_layer_with_transformer_engine_spec(), projector_submodules)


def get_nemotron_omni_projector_config(language_config, vision_config, *, hybrid_layer_pattern):
    """Build an independent squared-ReLU projection config and MLP spec."""
    _language_spec, _vision_spec, projector_submodules = get_nemotron_omni_specs(
        hybrid_layer_pattern
    )
    projection_config = copy.deepcopy(language_config)
    projection_config.sequence_parallel = False
    projection_config.context_parallel_size = 1
    projection_config.tp_comm_overlap = False
    projection_config.recompute_granularity = None
    projection_config.recompute_method = None
    projection_config.recompute_num_layers = None
    projection_config.ffn_hidden_size = PROJECTOR_FFN_HIDDEN_SIZE
    projection_config.activation_func = squared_relu
    projection_config.bias_activation_fusion = False
    projection_config.pipeline_model_parallel_size = 1
    return projection_config, projector_submodules, 4 * vision_config.hidden_size


def validate_nemotron_omni_support(args, language_config, vision_config) -> None:
    """Fail before model or MDP resources outside the M3 image-only boundary."""
    contract = getattr(args, "nemotron_omni_input_contract", EXPANDED_SEQUENCE_CONTRACT)
    if contract != EXPANDED_SEQUENCE_CONTRACT:
        raise ValueError(
            "Nemotron Omni requires the expanded_sequence_v1 input contract; " f"got {contract!r}."
        )
    if getattr(args, "nemotron_omni_enable_sound", False):
        raise ValueError(
            "Nemotron Omni sound/audio input is not supported by this image-only path."
        )

    pattern = getattr(language_config, "hybrid_layer_pattern", None) or getattr(
        args, "hybrid_layer_pattern", None
    )
    if not pattern:
        raise ValueError("Nemotron Omni requires a nonempty hybrid layer pattern.")
    if int(getattr(language_config, "tensor_model_parallel_size", 1)) != 1:
        raise ValueError("Nemotron Omni M3 does not support tensor parallel execution.")
    if int(getattr(language_config, "pipeline_model_parallel_size", 1)) != 1:
        raise ValueError("Nemotron Omni M3 does not support pipeline parallel execution.")
    if int(getattr(language_config, "context_parallel_size", 1)) != 1:
        raise ValueError("Nemotron Omni M3 does not support decoder context parallel execution.")
    if getattr(language_config, "sequence_parallel", False):
        raise ValueError("Nemotron Omni M3 does not support sequence parallel execution.")
    if (
        getattr(language_config, "virtual_pipeline_model_parallel_size", None) is not None
        or getattr(args, "virtual_pipeline_model_parallel_size", None) is not None
    ):
        raise ValueError("Nemotron Omni M3 does not support virtual pipeline parallelism.")
    if (
        getattr(language_config, "pipeline_model_parallel_layout", None) is not None
        or getattr(args, "pipeline_model_parallel_layout", None) is not None
    ):
        raise ValueError("Nemotron Omni M3 does not support a custom pipeline layout.")
    if getattr(args, "mtp_num_layers", None) or getattr(language_config, "mtp_num_layers", None):
        raise ValueError("Nemotron Omni M3 does not support MTP.")
    if int(getattr(args, "mdp_encoder_cp", 1)) != 1:
        raise ValueError("Nemotron Omni M3 requires encoder_cp=1; encoder CP is unsupported.")
    if getattr(language_config, "recompute_granularity", None) is not None:
        raise ValueError("Nemotron Omni M3 does not support language recompute.")
    if (
        vision_config is not None
        and getattr(vision_config, "recompute_granularity", None) is not None
    ):
        raise ValueError("Nemotron Omni M3 does not support vision recompute.")


def post_language_config(language_config, args):
    """Pin the expanded image token and validate topology before construction."""
    pattern = getattr(language_config, "hybrid_layer_pattern", None) or getattr(
        args, "hybrid_layer_pattern", None
    )
    if pattern:
        language_config.hybrid_layer_pattern = pattern
    args.image_token_id = IMAGE_TOKEN_ID
    validate_nemotron_omni_support(args, language_config, None)


def build_model(
    args, language_config, vision_config, pre_process=True, post_process=True, **kwargs
):
    """Build the canonical image-only expanded-sequence model."""
    from examples.multimodal_dev.models.nemotron_omni.model import NemotronOmniModel

    validate_nemotron_omni_support(args, language_config, vision_config)
    pattern = getattr(language_config, "hybrid_layer_pattern", None) or getattr(
        args, "hybrid_layer_pattern", None
    )
    language_spec, vision_spec, projection_submodules = get_nemotron_omni_specs(pattern)
    projection_config, projection_submodules, _input_size = get_nemotron_omni_projector_config(
        language_config, vision_config, hybrid_layer_pattern=pattern
    )
    return NemotronOmniModel(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        vision_spec=vision_spec,
        projection_config=projection_config,
        projection_submodules=projection_submodules,
        hybrid_layer_pattern=pattern,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        image_token_id=IMAGE_TOKEN_ID,
        parallel_output=True,
        share_embeddings_and_output_weights=not getattr(
            args, "untie_embeddings_and_output_weights", False
        ),
        pre_process=pre_process,
        post_process=post_process,
        build_vision_encoder=not getattr(args, "mdp_enable", False),
        pg_collection=kwargs.get("pg_collection"),
    )
