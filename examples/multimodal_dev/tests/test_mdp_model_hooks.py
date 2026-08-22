# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sentinel contracts for model-owned MDP adapter and replay hooks."""

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from examples.multimodal_dev import forward_step as forward_step_module
from examples.multimodal_dev.mdp_adapter import qwen35_mdp_replay
from examples.multimodal_dev.models import MODEL_REGISTRY, resolve_mdp_model_hooks
from megatron.core.mdp import integration as mdp_integration
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.protocols import validate_output_plane_widths


def test_qwen35_replay_preserves_the_existing_single_leaf_call():
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        return "unchanged-output"

    leaf = torch.randn(3, 8, requires_grad=True)
    packed = object()
    batch = MappingProxyType(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
            "labels": torch.tensor([[2, 3, 4]]),
            "loss_mask": torch.ones(1, 3),
            "padding_mask": torch.zeros(1, 3, dtype=torch.bool),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    )
    record = SimpleNamespace(decoder_packed_seq_params=packed)

    output = qwen35_mdp_replay(model, batch, record, (leaf,))

    assert output == "unchanged-output"
    assert len(calls) == 1
    assert calls[0]["vision_embeddings"] is leaf
    assert calls[0]["packed_seq_params"] is packed
    assert calls[0]["pixel_values"] is None
    assert calls[0]["input_ids"] is batch["input_ids"]
    assert calls[0]["attention_mask"] is None


def test_qwen35_replay_rejects_multiple_planes():
    with pytest.raises(RuntimeError, match="at most one encoder output plane"):
        qwen35_mdp_replay(
            lambda **kwargs: kwargs,
            MappingProxyType({}),
            SimpleNamespace(decoder_packed_seq_params=object()),
            (torch.empty(0, 8), torch.empty(0, 8)),
        )


def test_forward_step_dispatches_ordered_leaves_to_registered_replay(monkeypatch):
    leaves = (torch.randn(2, 8, requires_grad=True), torch.randn(2, 8, requires_grad=True))
    observed = []

    def replay(model, batch, record, encoder_leaves):
        observed.append((model, batch, record, encoder_leaves))
        return torch.tensor(7.0)

    record = SimpleNamespace(
        microbatch_id=3,
        text_only=False,
        model_payload=MappingProxyType(
            {"input_ids": torch.tensor([[1, 2]]), "loss_mask": torch.ones(1, 2)}
        ),
        decoder_packed_seq_params=object(),
    )
    runtime = SimpleNamespace(storage=SimpleNamespace(get_leaves=lambda microbatch_id: leaves))
    model = object()
    monkeypatch.setattr(forward_step_module, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(forward_step_module, "is_pipeline_last_stage", lambda: False)
    mdp_integration.set_model_replay(replay)
    try:
        output, _ = forward_step_module.mdp_forward_step(runtime, iter((record,)), model)
    finally:
        mdp_integration.reset_for_testing()

    assert output.item() == 7.0
    assert len(observed) == 1
    assert observed[0][0] is model
    assert observed[0][2] is record
    assert observed[0][3][0] is leaves[0]
    assert observed[0][3][1] is leaves[1]


def test_registry_dispatches_model_owned_mdp_hooks(monkeypatch):
    def factory_a(*args):
        return args

    def replay_a(*args):
        return args

    def factory_b(*args):
        return args

    def replay_b(*args):
        return args

    monkeypatch.setitem(
        MODEL_REGISTRY,
        "sentinel_a",
        {"mdp_adapter_factory_fn": factory_a, "mdp_replay_fn": replay_a},
    )
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "sentinel_b",
        {"mdp_adapter_factory_fn": factory_b, "mdp_replay_fn": replay_b},
    )

    assert resolve_mdp_model_hooks("sentinel_a") == (factory_a, replay_a)
    assert resolve_mdp_model_hooks("sentinel_b") == (factory_b, replay_b)


@pytest.mark.parametrize("missing", ["mdp_adapter_factory_fn", "mdp_replay_fn"])
def test_missing_model_hook_fails_before_any_runtime_work(monkeypatch, missing):
    calls = []

    def sentinel(*args, **kwargs):
        calls.append((args, kwargs))

    entry = {"mdp_adapter_factory_fn": sentinel, "mdp_replay_fn": sentinel}
    del entry[missing]
    monkeypatch.setitem(MODEL_REGISTRY, "missing_hook", entry)

    with pytest.raises(RuntimeError, match=missing):
        resolve_mdp_model_hooks("missing_hook")
    assert calls == []  # no builder, planner, bridge, storage, or encoder work


def test_core_rejects_missing_replay_before_adapter_or_ledgers():
    calls = []
    mdp_integration.reset_for_testing()
    mdp_integration.set_adapter_builder(lambda args: calls.append(args))
    try:
        with pytest.raises(MdpConfigurationError, match="no model replay hook"):
            mdp_integration.maybe_build_mdp_domain(
                args=SimpleNamespace(mdp_enable=True),
                model=[],
                optimizer=object(),
                optimizer_config=None,
                ddp_config=None,
            )
        assert calls == []
    finally:
        mdp_integration.reset_for_testing()


@pytest.mark.parametrize("widths", [None, (), [8], (0,), (True,), (8, 1.5)])
def test_invalid_output_plane_widths_fail_before_runtime(widths):
    with pytest.raises(ValueError, match="non-empty tuple of positive integers"):
        validate_output_plane_widths(SimpleNamespace(output_plane_widths=widths))


def test_mdp_core_has_no_model_package_imports():
    core_dir = Path(__file__).parents[3] / "megatron" / "core" / "mdp"
    forbidden = ("examples.multimodal_dev", "qwen", "nemotron")
    violations = []
    for path in sorted(core_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(token in name.lower() for token in forbidden):
                    violations.append((path.name, node.lineno, name))
    assert violations == []
