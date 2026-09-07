# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware
from megatron.core.mdp import integration
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.dynamic_cp_d3_private_facade import _D3PrivateFacade
from megatron.core.mdp.dynamic_encoder_adapter_capability import (
    _reset_dynamic_encoder_adapter_capabilities_for_tests,
    register_dynamic_encoder_adapter_class,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpTaskFatalError


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


class _DynamicAdapter:
    payload_width = 8
    embedding_width = 16
    spatial_merge_size = 2

    def get_batch(self, iterator):
        return iterator

    def estimate_cost(self, item):
        return item

    def build_dynamic_decoder_payload_codec(self):
        return object()

    def estimate_dynamic_encoder_workload(self, items, *, group_size):
        return items, group_size

    def build_encoder(self, model_config, *, pg_collection):
        return model_config, pg_collection

    def bind_dynamic_encoder_cp(self, encoder, *, membership, global_rank):
        return encoder, membership, global_rank

    def encode(self, encoder, payload, layout):
        return encoder, payload, layout


register_dynamic_encoder_adapter_class(
    _DynamicAdapter,
    get_batch=_DynamicAdapter.get_batch,
    estimate_cost=_DynamicAdapter.estimate_cost,
    build_dynamic_decoder_payload_codec=_DynamicAdapter.build_dynamic_decoder_payload_codec,
    estimate_dynamic_encoder_workload=_DynamicAdapter.estimate_dynamic_encoder_workload,
    build_encoder=_DynamicAdapter.build_encoder,
    bind_dynamic_encoder_cp=_DynamicAdapter.bind_dynamic_encoder_cp,
    encode=_DynamicAdapter.encode,
)


def _dynamic_args():
    return SimpleNamespace(
        mdp_enable=True,
        world_size=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=4,
        expert_model_parallel_size=1,
        bf16=True,
        fp16=False,
        hidden_size=16,
    )


def _patch_dynamic_prefix(monkeypatch):
    rank_view = SimpleNamespace(outer_dp_rank=0, my_worker_id=0, endpoint_rank=0, worker_ids=(0,))
    rank_map = SimpleNamespace(view=lambda rank: rank_view)
    monkeypatch.setattr(
        integration,
        "mdp_config_from_args",
        lambda args: MdpConfig(enable=True, encoder_cp=4, dynamic_encoder_cp=True),
    )
    monkeypatch.setattr(integration, "compatibility_options_from_args", lambda args: object())
    monkeypatch.setattr(integration, "validate_mdp_config", lambda config, options: None)
    monkeypatch.setattr(integration, "build_rank_map", lambda spec: rank_map)
    monkeypatch.setattr(integration.torch.distributed, "get_rank", lambda: 0)
    return rank_map, rank_view


@pytest.mark.parametrize("failure_stage", ("builder", "mint", "claim"))
def test_repeated_d4_pre_group_rejection_converges_before_group_creation(
    monkeypatch, failure_stage
):
    integration.reset_for_testing()
    _reset_dynamic_encoder_adapter_capabilities_for_tests()
    _patch_dynamic_prefix(monkeypatch)
    adapter = _DynamicAdapter()
    original_claim = integration.claim_dynamic_encoder_adapter_capability
    if failure_stage == "builder":
        integration.set_adapter_builder(
            lambda args: (_ for _ in ()).throw(RuntimeError("builder failed"))
        )
    elif failure_stage == "mint":
        integration.set_adapter_builder(lambda args: (object(), object()))
    else:
        integration.set_adapter_builder(lambda args: (adapter, object()))
        monkeypatch.setattr(
            integration,
            "claim_dynamic_encoder_adapter_capability",
            lambda selected, capability: (_ for _ in ()).throw(RuntimeError("claim failed")),
        )

    observed = []

    def converge(error):
        observed.append(error)
        raise MdpPlanError("WORLD rejected")

    monkeypatch.setattr(integration, "_converge_repeated_d4_adapter_prevalidation", converge)
    monkeypatch.setattr(
        integration,
        "install_mdp_process_groups",
        lambda *args, **kwargs: pytest.fail("group construction must not start"),
    )
    with pytest.raises(MdpConfigurationError, match="prevalidation rejected"):
        integration.maybe_build_mdp_domain(
            args=_dynamic_args(),
            model=[object()],
            optimizer=object(),
            optimizer_config=object(),
            ddp_config=object(),
        )
    assert len(observed) == 1 and isinstance(observed[0], BaseException)
    assert integration.get_runtime() is None

    integration.set_adapter_builder(lambda args: (_DynamicAdapter(), object()))
    monkeypatch.setattr(integration, "claim_dynamic_encoder_adapter_capability", original_claim)
    monkeypatch.setattr(
        integration, "_converge_repeated_d4_adapter_prevalidation", lambda error: None
    )
    fresh_adapter, _, capability, operations = integration._prepare_repeated_d4_adapter(
        _dynamic_args()
    )
    assert operations._adapter() is fresh_adapter
    integration.retire_dynamic_encoder_adapter_capability(capability)
    _reset_dynamic_encoder_adapter_capabilities_for_tests()


def test_repeated_d4_gate_configuration_failure_retires_before_group_creation(monkeypatch):
    integration.reset_for_testing()
    _reset_dynamic_encoder_adapter_capabilities_for_tests()
    _patch_dynamic_prefix(monkeypatch)
    integration.set_adapter_builder(lambda args: (_DynamicAdapter(), object()))
    monkeypatch.setattr(
        integration,
        "_converge_repeated_d4_adapter_prevalidation",
        lambda error: (_ for _ in ()).throw(MdpConfigurationError("gate setup failed")),
    )
    monkeypatch.setattr(
        integration,
        "install_mdp_process_groups",
        lambda *args, **kwargs: pytest.fail("group construction must not start"),
    )

    with pytest.raises(MdpConfigurationError, match="gate setup failed"):
        integration.maybe_build_mdp_domain(
            args=_dynamic_args(),
            model=[object()],
            optimizer=object(),
            optimizer_config=object(),
            ddp_config=object(),
        )

    integration.set_adapter_builder(lambda args: (_DynamicAdapter(), object()))
    monkeypatch.setattr(
        integration, "_converge_repeated_d4_adapter_prevalidation", lambda error: None
    )
    adapter, _, capability, operations = integration._prepare_repeated_d4_adapter(_dynamic_args())
    assert operations._adapter() is adapter
    integration.retire_dynamic_encoder_adapter_capability(capability)
    _reset_dynamic_encoder_adapter_capabilities_for_tests()


def _patch_dynamic_construction(monkeypatch, events, *, fail_at=None):
    rank_map, rank_view = _patch_dynamic_prefix(monkeypatch)
    adapter = _DynamicAdapter()
    integration.set_adapter_builder(lambda args: (events.append("builder") or adapter, object()))
    monkeypatch.setattr(
        integration,
        "_converge_repeated_d4_adapter_prevalidation",
        lambda error: events.append("consensus") if error is None else pytest.fail(str(error)),
    )
    process_groups = SimpleNamespace(world_group=object(), encoder_cp_group=object())

    def install(selected, **kwargs):
        events.append("groups")
        assert selected is rank_map
        assert kwargs["dynamic_encoder_cp"] is True
        assert kwargs["min_dynamic_encoder_cp_size"] == 1
        if fail_at == "groups":
            raise RuntimeError("groups failed")
        return process_groups

    monkeypatch.setattr(integration, "install_mdp_process_groups", install)
    monkeypatch.setattr(
        integration, "build_encoder_pg_collection", lambda *args, **kwargs: "encoder-pgs"
    )

    def bind_groups(**kwargs):
        events.append("binding")
        assert kwargs["world_group"] is process_groups.world_group
        assert kwargs["domain_group"] is process_groups.encoder_cp_group
        assert kwargs["expert_group"] is None
        assert kwargs["expert_parallel_size"] == 1
        return "binding"

    monkeypatch.setattr(integration, "_make_repeated_d4_group_binding", bind_groups)
    encoder_domain = SimpleNamespace(encoder_ddp=object(), encoder_optimizer="encoder-optimizer")

    def build_encoder_domain(**kwargs):
        events.append("encoder")
        assert kwargs["adapter"].estimate_cost(7) == 7
        if fail_at == "encoder":
            raise RuntimeError("encoder failed")
        return encoder_domain

    monkeypatch.setattr(integration, "build_encoder_domain", build_encoder_domain)
    monkeypatch.setattr(integration, "assert_parameter_disjointness", lambda *args: None)
    monkeypatch.setattr(integration, "MdpPlanner", lambda *args, **kwargs: "planner")
    captured = {}

    def build_runtime(**kwargs):
        events.append("runtime")
        captured.update(kwargs)
        if fail_at == "runtime":
            raise RuntimeError("runtime failed")
        return SimpleNamespace(dynamic_adapter_capability=kwargs["dynamic_adapter_capability"])

    monkeypatch.setattr(integration, "MdpRuntime", build_runtime)
    from megatron.core.mdp import optimizer as mdp_optimizer

    def build_optimizer(decoder, encoder):
        events.append("optimizer")
        assert integration.get_runtime() is None
        if fail_at == "optimizer":
            raise RuntimeError("optimizer failed")
        return decoder, encoder

    monkeypatch.setattr(mdp_optimizer, "build_mdp_composite_optimizer", build_optimizer)
    return adapter, rank_view, captured


def test_repeated_d4_publishes_runtime_only_after_atomic_construction(monkeypatch):
    integration.reset_for_testing()
    _reset_dynamic_encoder_adapter_capabilities_for_tests()
    events = []
    adapter, _, captured = _patch_dynamic_construction(monkeypatch, events)

    result = integration.maybe_build_mdp_domain(
        args=_dynamic_args(),
        model=[object()],
        optimizer="decoder-optimizer",
        optimizer_config=object(),
        ddp_config=object(),
    )

    assert result == ("decoder-optimizer", "encoder-optimizer")
    assert events == [
        "builder",
        "consensus",
        "groups",
        "binding",
        "encoder",
        "runtime",
        "optimizer",
    ]
    assert captured["adapter"]._adapter() is adapter
    assert captured["dynamic_adapter_owner"] is adapter
    assert captured["dynamic_group_binding"] == "binding"
    assert integration.get_runtime() is not None
    integration.reset_for_testing()
    integration.reset_for_testing()
    _reset_dynamic_encoder_adapter_capabilities_for_tests()


@pytest.mark.parametrize("failure_stage", ("groups", "encoder", "runtime", "optimizer"))
def test_repeated_d4_post_group_failures_are_task_fatal(monkeypatch, failure_stage):
    integration.reset_for_testing()
    _reset_dynamic_encoder_adapter_capabilities_for_tests()
    events = []
    _patch_dynamic_construction(monkeypatch, events, fail_at=failure_stage)

    with pytest.raises(MdpTaskFatalError, match="repeated-D4"):
        integration.maybe_build_mdp_domain(
            args=_dynamic_args(),
            model=[object()],
            optimizer=object(),
            optimizer_config=object(),
            ddp_config=object(),
        )
    assert integration.get_runtime() is None
    _reset_dynamic_encoder_adapter_capabilities_for_tests()
