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

``dataset_providers``  *(optional)*
    ``Dict[str, str | callable]``
    Maps ``--dataset-provider`` names to callables (or dotted import paths
    resolved lazily) with signature
    ``(train_val_test_num_samples) -> (train_ds, val_ds, test_ds)``.

``mdp_adapter_factory_fn`` / ``mdp_replay_fn``  *(required when MDP is used)*
    Model-owned encoder adapter construction and decoder replay. Core MDP
    transports ordered tensor planes and never imports model packages.

``mdp_vision_config_validator_fn``  *(optional)*
    Validates the effective MDP vision config before core MDP resources are
    constructed. Architectures without this hook retain their existing path.
"""

from examples.multimodal_dev.models.qwen35_vl.configuration import get_qwen35_vl_vision_config
from examples.multimodal_dev.models.qwen35_vl.factory import build_model as _build_qwen35_vl_model
from examples.multimodal_dev.models.qwen35_vl.factory import (
    post_language_config as _qwen35_vl_post_language_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import (
    set_vision_flops_metadata as _qwen35_vl_vision_flops,
)


def _build_qwen35_mdp_adapter(args, language_config):
    from examples.multimodal_dev.mdp_adapter import build_mdp_adapter

    return build_mdp_adapter(args, language_config)


def _qwen35_mdp_replay(model, batch, record, encoder_leaves):
    from examples.multimodal_dev.mdp_adapter import qwen35_mdp_replay

    return qwen35_mdp_replay(model, batch, record, encoder_leaves)


def _build_qwen3_vl_model(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.factory import build_model

    return build_model(*args, **kwargs)


def _qwen3_vl_vision_config(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.configuration import get_qwen3_vl_vision_config

    return get_qwen3_vl_vision_config(*args, **kwargs)


def _qwen3_vl_post_language_config(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.factory import post_language_config

    return post_language_config(*args, **kwargs)


def _build_qwen3_vl_mdp_adapter(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.mdp import build_mdp_adapter

    return build_mdp_adapter(*args, **kwargs)


def _qwen3_vl_mdp_replay(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.mdp import qwen3_vl_mdp_replay

    return qwen3_vl_mdp_replay(*args, **kwargs)


def _qwen3_vl_validate_mdp_vision_config(*args, **kwargs):
    from examples.multimodal_dev.models.qwen3_vl.factory import validate_qwen3_vl_support

    return validate_qwen3_vl_support(*args, **kwargs)


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


def _build_nemotron_omni_mdp_adapter(*args, **kwargs):
    from examples.multimodal_dev.models.nemotron_omni.mdp import build_mdp_adapter

    return build_mdp_adapter(*args, **kwargs)


def _nemotron_omni_mdp_replay(*args, **kwargs):
    from examples.multimodal_dev.models.nemotron_omni.mdp import nemotron_omni_mdp_replay

    return nemotron_omni_mdp_replay(*args, **kwargs)


def _nemotron_omni_validate_mdp_vision_config(*args, **kwargs):
    from examples.multimodal_dev.models.nemotron_omni.factory import validate_nemotron_omni_support

    return validate_nemotron_omni_support(*args, **kwargs)


def _nemotron_omni_validate_raw_batch(*args, **kwargs):
    from examples.multimodal_dev.models.nemotron_omni.mdp import validate_nemotron_omni_raw_batch

    return validate_nemotron_omni_raw_batch(*args, **kwargs)


MODEL_REGISTRY = {
    "qwen35_vl": {
        "model_factory_fn": _build_qwen35_vl_model,
        "vision_config_fn": get_qwen35_vl_vision_config,
        "post_language_config_fn": _qwen35_vl_post_language_config,
        "vision_flops_fn": _qwen35_vl_vision_flops,
        "mdp_adapter_factory_fn": _build_qwen35_mdp_adapter,
        "mdp_replay_fn": _qwen35_mdp_replay,
        "dataset_providers": {
            "mock": ("examples.multimodal_dev.data.mock" ".train_valid_test_datasets_provider"),
            "cord_v2": (
                "examples.multimodal_dev.data.cord_v2" ".train_valid_test_datasets_provider"
            ),
            "mdp_mock": (
                "examples.multimodal_dev.data.mdp_mock" ".train_valid_test_datasets_provider"
            ),
        },
    },
    "qwen3_vl": {
        "model_factory_fn": _build_qwen3_vl_model,
        "vision_config_fn": _qwen3_vl_vision_config,
        "post_language_config_fn": _qwen3_vl_post_language_config,
        "mdp_adapter_factory_fn": _build_qwen3_vl_mdp_adapter,
        "mdp_replay_fn": _qwen3_vl_mdp_replay,
        "mdp_vision_config_validator_fn": _qwen3_vl_validate_mdp_vision_config,
        "dataset_providers": {
            "mock": ("examples.multimodal_dev.models.qwen3_vl.data" ".mock_dataset_provider"),
            "mdp_mock": (
                "examples.multimodal_dev.models.qwen3_vl.data" ".mdp_mock_dataset_provider"
            ),
        },
    },
    "nemotron_omni": {
        "model_factory_fn": _build_nemotron_omni_model,
        "vision_config_fn": _nemotron_omni_vision_config,
        "post_language_config_fn": _nemotron_omni_post_language_config,
        "mdp_adapter_factory_fn": _build_nemotron_omni_mdp_adapter,
        "mdp_replay_fn": _nemotron_omni_mdp_replay,
        "mdp_vision_config_validator_fn": _nemotron_omni_validate_mdp_vision_config,
        "raw_batch_validator_fn": _nemotron_omni_validate_raw_batch,
        "dataset_providers": {
            "mock": ("examples.multimodal_dev.models.nemotron_omni.data.mock_dataset_provider"),
            "mdp_mock": (
                "examples.multimodal_dev.models.nemotron_omni.data.mdp_mock_dataset_provider"
            ),
        },
    },
}


def resolve_mdp_model_hooks(model_arch):
    """Resolve required model-owned MDP hooks before core runtime construction."""
    if model_arch not in MODEL_REGISTRY:
        raise RuntimeError(
            f"MDP: unknown model arch {model_arch!r}; available: " f"{sorted(MODEL_REGISTRY)}."
        )
    entry = MODEL_REGISTRY[model_arch]
    required = ("mdp_adapter_factory_fn", "mdp_replay_fn")
    missing = [name for name in required if not callable(entry.get(name))]
    if missing:
        raise RuntimeError(
            f"MDP: model arch {model_arch!r} is missing callable registry hook(s): "
            f"{', '.join(missing)}. MDP cannot construct planning, transport, "
            "storage, or the encoder without model-owned hooks."
        )
    return entry[required[0]], entry[required[1]]
