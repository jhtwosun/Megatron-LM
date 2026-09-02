# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-agnostic MDP adapter registration contracts."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev import pretrain_multimodal
from examples.multimodal_dev.models import MODEL_REGISTRY


def test_qwen_registry_declares_a_lazy_mdp_adapter_factory():
    """The built-in Qwen entry resolves its adapter only when MDP is enabled."""
    factory = MODEL_REGISTRY["qwen35_vl"]["mdp_adapter_factory"]
    assert isinstance(factory, str)
    assert callable(pretrain_multimodal._resolve_provider_fn(factory))


def test_mdp_adapter_builder_uses_selected_registry_factory(monkeypatch):
    """Adapter construction dispatches through the selected model entry."""
    calls = []

    def adapter_factory(args, language_config):
        calls.append((args, language_config))
        return "adapter"

    vision_config = SimpleNamespace()
    language_config = SimpleNamespace(
        bf16=True,
        fp16=False,
        apply_rope_fusion=True,
        params_dtype="bf16",
        calculate_per_token_loss=True,
    )
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "test_mdp_arch",
        {
            "mdp_adapter_factory": adapter_factory,
            "vision_config_fn": lambda **kwargs: vision_config,
        },
    )
    monkeypatch.setattr(
        pretrain_multimodal, "core_transformer_config_from_args", lambda args: language_config
    )
    args = SimpleNamespace(model_arch="test_mdp_arch", vision_num_layers=None, model_variant=None)

    adapter, built_vision_config = pretrain_multimodal._mdp_adapter_builder(args)

    assert adapter == "adapter"
    assert built_vision_config is vision_config
    assert calls == [(args, language_config)]
    assert vision_config.bf16 is True
    assert vision_config.fp16 is False
    assert vision_config.apply_rope_fusion is True
    assert vision_config.params_dtype == "bf16"
    assert vision_config.calculate_per_token_loss is True


def test_mdp_adapter_builder_rejects_arch_without_registry_factory(monkeypatch):
    """An MDP-enabled model must declare its adapter factory explicitly."""
    language_config = SimpleNamespace(
        bf16=True,
        fp16=False,
        apply_rope_fusion=True,
        params_dtype="bf16",
        calculate_per_token_loss=True,
    )
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "missing_mdp_factory",
        {"vision_config_fn": lambda **kwargs: SimpleNamespace()},
    )
    monkeypatch.setattr(
        pretrain_multimodal, "core_transformer_config_from_args", lambda args: language_config
    )
    args = SimpleNamespace(
        model_arch="missing_mdp_factory", vision_num_layers=None, model_variant=None
    )

    with pytest.raises(ValueError, match="has no MDP adapter factory"):
        pretrain_multimodal._mdp_adapter_builder(args)
