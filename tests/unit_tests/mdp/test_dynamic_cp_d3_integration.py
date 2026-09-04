# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware
from megatron.core.mdp import integration
from megatron.core.mdp.dynamic_cp_d3_private_facade import _D3PrivateFacade
from megatron.core.mdp.errors import MdpConfigurationError


def _config(**overrides):
    values = {
        "dynamic_context_parallel": True,
        "sequence_packing_scheduler": "default_dynamic_cp",
        "max_seqlen_per_dp_cp_rank": 8192,
        "min_dynamic_context_parallel_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_d3_mcore_factory_binds_exact_native_dependencies(monkeypatch):
    group = object()
    codec = object()
    adapter = SimpleNamespace(build_dynamic_decoder_payload_codec=lambda: codec)
    runtime = SimpleNamespace(
        adapter=adapter,
        process_groups=SimpleNamespace(world_group=group),
        device=torch.device("cuda", 3),
    )
    captured = {}
    facade = object()

    def build(**kwargs):
        captured.update(kwargs)
        return facade

    monkeypatch.setattr(integration, "_build_d3_runtime_facade", build)
    monkeypatch.setattr(
        integration.torch.distributed,
        "get_process_group_ranks",
        lambda selected: [0, 1, 2, 3] if selected is group else pytest.fail("wrong group"),
    )
    monkeypatch.setattr(integration.torch.distributed, "get_rank", lambda: 2)

    result = integration._build_d3_facade_from_mcore(runtime, _config())

    assert result is facade
    assert captured == {
        "producer_runtime": runtime,
        "codec": codec,
        "group": group,
        "participant_ranks": (0, 1, 2, 3),
        "global_rank": 2,
        "device": torch.device("cuda", 3),
        "expected_source_lanes": (0, 1, 2, 3),
        "decoder_solver": next_hdp_group_packing_aware,
        "max_seqlen_per_rank": 8192,
        "minimum_cp_size": 2,
        "decoder_group_getter": parallel_state.get_dynamic_data_context_parallel_groups,
        "decoder_group_ranks_getter": integration.torch.distributed.get_process_group_ranks,
        "timeout_seconds": 30.0,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dynamic_context_parallel": False}, "native Dynamic-CP groups"),
        ({"sequence_packing_scheduler": None}, "planning contract"),
        ({"max_seqlen_per_dp_cp_rank": None}, "max sequence length"),
        ({"min_dynamic_context_parallel_size": 0}, "minimum CP size"),
    ],
)
def test_d3_mcore_factory_rejects_invalid_native_config(overrides, message):
    runtime = SimpleNamespace(adapter=object())
    with pytest.raises(MdpConfigurationError, match=message):
        integration._build_d3_facade_from_mcore(runtime, _config(**overrides))


def test_d3_mcore_factory_requires_model_codec():
    runtime = SimpleNamespace(adapter=object())
    with pytest.raises(MdpConfigurationError, match="decoder codec"):
        integration._build_d3_facade_from_mcore(runtime, _config())


def test_dynamic_mdp_activates_one_reused_training_facade(monkeypatch):
    integration.reset_for_testing()
    runtime = object()
    facade = object.__new__(_D3PrivateFacade)
    config = _config(finalize_model_grads_func=lambda *args: None)
    built = []

    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(integration, "wrap_finalize_model_grads", lambda *args: None)
    monkeypatch.setattr(
        integration,
        "_build_d3_facade_from_mcore",
        lambda selected_runtime, selected_config: (
            built.append((selected_runtime, selected_config)) or facade
        ),
    )
    monkeypatch.setattr(
        integration,
        "_wrap_d3_forward_backward",
        lambda schedule, selected_facade: ("wrapped", schedule, selected_facade),
    )
    monkeypatch.setattr(
        integration,
        "wrap_forward_backward",
        lambda schedule, selected_runtime: ("static", schedule, selected_runtime),
    )
    schedule_a = object()
    schedule_b = object()

    assert integration.d3_owns_data_schedule(config)
    assert integration.maybe_wrap_forward_backward(schedule_a, config) == (
        "wrapped",
        schedule_a,
        facade,
    )
    assert integration.maybe_wrap_forward_backward(schedule_b, config) == (
        "wrapped",
        schedule_b,
        facade,
    )
    assert built == [(runtime, config)]

    assert integration.maybe_wrap_forward_backward(schedule_a, config, training=False) == (
        "static",
        schedule_a,
        runtime,
    )
    integration.reset_for_testing()
    assert integration._D3_FACADE is None


def test_training_loop_gives_d3_exclusive_data_schedule_ownership():
    training_path = Path(__file__).parents[3] / "megatron" / "training" / "training.py"
    tree = ast.parse(training_path.read_text())
    train_step = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_step"
    )
    guarded_native_calls = [
        node
        for node in ast.walk(train_step)
        if isinstance(node, ast.If)
        and "wrap_data_iterator" in ast.dump(node)
        and "mdp_d3_owns_data_schedule" in ast.unparse(node.test)
    ]
    assert len(guarded_native_calls) == 1

    evaluate = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
    )
    mdp_wrap = next(
        node
        for node in ast.walk(evaluate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "maybe_wrap_forward_backward"
    )
    assert any(
        keyword.arg == "training"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in mdp_wrap.keywords
    )
    guarded_eval_calls = [
        node
        for node in ast.walk(evaluate)
        if isinstance(node, ast.If)
        and "wrap_data_iterator" in ast.dump(node)
        and "mdp_owns_data_schedule" in ast.unparse(node.test)
    ]
    assert len(guarded_eval_calls) == 1
