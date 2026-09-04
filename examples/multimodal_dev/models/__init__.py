# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model registry for multimodal_dev training.

Maps ``--model-arch`` to a set of factory functions that fully encapsulate
model-specific logic.  The training entry point (``pretrain_multimodal.py``)
remains model-agnostic — adding a new architecture only requires a new
registry entry (and its backing module) without touching the entry point.

Registry entry fields
---------------------
``model_factory_fn``  *(required)*
    ``(args, language_config, vision_config, **kwargs) -> MegatronModule``
    Builds and returns the complete model instance.

``vision_config_fn``  *(required)*
    ``(num_layers_override=None, variant=None) -> TransformerConfig``
    Returns the vision encoder TransformerConfig.

``post_language_config_fn``  *(optional)*
    ``(language_config, args) -> None``
    Mutates the language TransformerConfig in-place with model-specific
    fields (e.g. ``mrope_section``).

``vision_flops_fn``  *(optional)*
    ``(args, language_config, vision_config) -> None``
    Sets vision FLOPs metadata on ``args`` for training throughput logging.

``mdp_adapter_factory``  *(required when MDP is enabled)*
    Callable or lazily resolved dotted path with signature
    ``(args, language_config) -> MdpModelAdapter``.

``dataset_providers``  *(optional)*
    ``Dict[str, str | callable]``
    Maps ``--dataset-provider`` names to callables (or dotted import paths
    resolved lazily) with signature
    ``(train_val_test_num_samples) -> (train_ds, val_ds, test_ds)``.

``energon_task_encoder_factory``  *(required by the ``energon`` provider)*
    Lazy callable with signature ``(*, args, energon_api) -> TaskEncoder``.

``energon_image_materializer_factory``  *(required for Energon images)*
    Lazy callable with signature ``(*, args) -> callable``. The returned
    callable decodes and patchifies selected descriptors on their pixel owner.

``energon_image_metadata_validator``  *(optional for Energon images)*
    Lazy callable with signature ``(descriptors, image_grid_thw) -> object``.
    Validates model-specific image metadata before materializer construction.
"""

from examples.multimodal_dev.models.qwen35_vl.configuration import get_qwen35_vl_vision_config
from examples.multimodal_dev.models.qwen35_vl.factory import build_model as _build_qwen35_vl_model
from examples.multimodal_dev.models.qwen35_vl.factory import (
    post_language_config as _qwen35_vl_post_language_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import (
    set_vision_flops_metadata as _qwen35_vl_vision_flops,
)


def _build_nemotron_omni_model(*args, **kwargs):
    from examples.multimodal_dev.models.nemotron_omni.factory import build_model

    return build_model(*args, **kwargs)


def _nemotron_omni_vision_config(*args, **kwargs):
    from examples.multimodal_dev.models.nemotron_omni.configuration import (
        get_nemotron_omni_vision_config,
    )

    return get_nemotron_omni_vision_config(*args, **kwargs)


def _nemotron_omni_post_language_config(*args, **kwargs):
    from examples.multimodal_dev.models.nemotron_omni.factory import post_language_config

    return post_language_config(*args, **kwargs)


def _build_qwen3_vl_model(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.factory import build_model

    return build_model(*args, **kwargs)


def _qwen3_vl_vision_config(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.configuration import get_qwen3_vl_vision_config

    return get_qwen3_vl_vision_config(*args, **kwargs)


def _qwen3_vl_post_language_config(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.factory import post_language_config

    return post_language_config(*args, **kwargs)


MODEL_REGISTRY = {
    "qwen35_vl": {
        "model_factory_fn": _build_qwen35_vl_model,
        "vision_config_fn": get_qwen35_vl_vision_config,
        "post_language_config_fn": _qwen35_vl_post_language_config,
        "vision_flops_fn": _qwen35_vl_vision_flops,
        "mdp_adapter_factory": "examples.multimodal_dev.mdp_adapter.build_mdp_adapter",
        "energon_task_encoder_factory": (
            "examples.multimodal_dev.models.qwen35_vl.energon.build_task_encoder"
        ),
        "energon_image_materializer_factory": (
            "examples.multimodal_dev.models.qwen35_vl.energon.build_image_materializer"
        ),
        "energon_image_metadata_validator": (
            "examples.multimodal_dev.models.qwen35_vl.energon.validate_image_metadata"
        ),
        "dataset_providers": {
            "mock": (
                "examples.multimodal_dev.data.mock"
                ".train_valid_test_datasets_provider"
            ),
            "cord_v2": (
                "examples.multimodal_dev.data.cord_v2"
                ".train_valid_test_datasets_provider"
            ),
            "mdp_mock": (
                "examples.multimodal_dev.data.mdp_mock"
                ".train_valid_test_datasets_provider"
            ),
            "energon": (
                "examples.multimodal_dev.data.energon.provider"
                ".train_valid_test_datasets_provider"
            ),
        },
    },
    "qwen3_vl": {
        "model_factory_fn": _build_qwen3_vl_model,
        "vision_config_fn": _qwen3_vl_vision_config,
        "post_language_config_fn": _qwen3_vl_post_language_config,
        "mdp_adapter_factory": (
            "examples.multimodal_dev.models.qwen3_vl.mdp.build_mdp_adapter"
        ),
        "dataset_providers": {
            "mock": (
                "examples.multimodal_dev.models.qwen3_vl.data"
                ".mock_dataset_provider"
            ),
            "mdp_mock": (
                "examples.multimodal_dev.models.qwen3_vl.data"
                ".mdp_mock_dataset_provider"
            ),
        },
    },
    "nemotron_omni": {
        "model_factory_fn": _build_nemotron_omni_model,
        "vision_config_fn": _nemotron_omni_vision_config,
        "post_language_config_fn": _nemotron_omni_post_language_config,
        "mdp_adapter_factory": (
            "examples.multimodal_dev.models.nemotron_omni.mdp.build_mdp_adapter"
        ),
        "dataset_providers": {
            "mock": (
                "examples.multimodal_dev.models.nemotron_omni.data"
                ".mock_dataset_provider"
            ),
            "mdp_mock": (
                "examples.multimodal_dev.models.nemotron_omni.data"
                ".mdp_mock_dataset_provider"
            ),
        },
    },
}
