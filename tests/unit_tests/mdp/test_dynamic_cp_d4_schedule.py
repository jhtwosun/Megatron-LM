# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Public-integration boundary tests for the private repeated-D4 schedule."""

from types import SimpleNamespace

import pytest

from megatron.core.mdp import integration
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.schedule import _wrap_d4_forward_backward, wrap_finalize_model_grads


def _native_schedule(*, data_iterator, num_microbatches, forward_only=False):
    return data_iterator, num_microbatches, forward_only


def test_d4_wrapper_runs_one_native_schedule_and_preserves_its_result():
    calls = []

    def facade(**kwargs):
        calls.append((kwargs["data_iterators"], kwargs["num_microbatches"]))
        kwargs["forward_backward_func"](
            *kwargs["native_schedule_args"], **kwargs["native_schedule_kwargs"]
        )

    wrapped = _wrap_d4_forward_backward(_native_schedule, facade)
    iterator = iter(("raw",))

    assert wrapped(data_iterator=iterator, num_microbatches=3, forward_only=False) == (
        iterator,
        3,
        False,
    )
    assert calls == [(iterator, 3)]
    with pytest.raises(MdpConfigurationError, match="wrapped twice"):
        _wrap_d4_forward_backward(wrapped, facade)


def test_d4_wrapper_preserves_one_element_vpp1_container():
    def facade(**kwargs):
        kwargs["forward_backward_func"](
            *kwargs["native_schedule_args"], **kwargs["native_schedule_kwargs"]
        )

    wrapped = _wrap_d4_forward_backward(_native_schedule, facade)
    iterator = [iter(("raw",))]

    assert wrapped(data_iterator=iterator, num_microbatches=1, forward_only=False) == (
        iterator,
        1,
        False,
    )


@pytest.mark.parametrize(
    ("data_iterator", "forward_only", "message"),
    [([iter((0,)), iter((1,))], False, "VPP1"), (iter((0,)), True, "training-only")],
)
def test_d4_wrapper_rejects_unsupported_schedule_before_facade(
    data_iterator, forward_only, message
):
    wrapped = _wrap_d4_forward_backward(
        _native_schedule, lambda **_kwargs: pytest.fail("facade entered")
    )

    with pytest.raises(MdpConfigurationError, match=message):
        wrapped(data_iterator=data_iterator, num_microbatches=1, forward_only=forward_only)


@pytest.mark.parametrize("native_calls", (0, 2))
def test_d4_wrapper_requires_exactly_one_native_invocation(native_calls):
    def facade(**kwargs):
        for _ in range(native_calls):
            kwargs["forward_backward_func"](
                *kwargs["native_schedule_args"], **kwargs["native_schedule_kwargs"]
            )

    wrapped = _wrap_d4_forward_backward(_native_schedule, facade)
    with pytest.raises(MdpStateError, match="exactly once"):
        wrapped(data_iterator=iter((0,)), num_microbatches=1, forward_only=False)


def test_d4_finalizer_uses_stable_native_sink_wrapper():
    captured = []
    token = object()

    def native(_model, selected):
        return "native", selected

    config = SimpleNamespace(finalize_model_grads_func=native)
    runtime = SimpleNamespace(
        config=MdpConfig(enable=True, dynamic_encoder_cp=True),
        capture_global_num_tokens=captured.append,
    )

    wrap_finalize_model_grads(config, runtime)
    wrapped = config.finalize_model_grads_func
    wrap_finalize_model_grads(config, runtime)

    assert config.finalize_model_grads_func is wrapped
    assert wrapped("model", token) == ("native", token)
    assert captured == [token]
    assert config.finalize_model_grads_func._mdp_native is native


def _runtime(*, dynamic_encoder_cp):
    return SimpleNamespace(config=MdpConfig(enable=True, dynamic_encoder_cp=dynamic_encoder_cp))


def test_no_runtime_preserves_callable_and_ignores_mode_selection():
    integration.reset_for_testing()
    config = SimpleNamespace(dynamic_context_parallel=True)

    assert integration.maybe_wrap_forward_backward(_native_schedule, config) is _native_schedule
    assert not integration.mdp_owns_data_schedule(config)
    assert (
        integration.maybe_wrap_forward_backward(_native_schedule, config, training="train")
        is _native_schedule
    )


@pytest.mark.parametrize(
    ("dynamic_decoder", "dynamic_encoder", "expected"),
    [(False, False, "static"), (True, False, "d3"), (False, True, "d4"), (True, True, "d4")],
)
def test_dispatches_each_mode_once(monkeypatch, dynamic_decoder, dynamic_encoder, expected):
    integration.reset_for_testing()
    runtime = _runtime(dynamic_encoder_cp=dynamic_encoder)
    config = SimpleNamespace(dynamic_context_parallel=dynamic_decoder)
    facade = object()
    calls = []
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(
        integration, "wrap_finalize_model_grads", lambda *args: calls.append("final")
    )
    monkeypatch.setattr(
        integration,
        "_build_d3_facade_from_mcore",
        lambda *_args: calls.append("build-d3") or facade,
    )
    monkeypatch.setattr(
        integration,
        "_build_d4_facade_from_mcore",
        lambda *_args: calls.append("build-d4") or facade,
    )
    monkeypatch.setattr(
        integration,
        "_wrap_d3_forward_backward",
        lambda schedule, selected: (calls.append("d3") or ("d3", schedule, selected)),
    )
    monkeypatch.setattr(
        integration,
        "_wrap_d4_forward_backward",
        lambda schedule, selected: (calls.append("d4") or ("d4", schedule, selected)),
    )
    monkeypatch.setattr(
        integration,
        "wrap_forward_backward",
        lambda schedule, selected: (calls.append("static") or ("static", schedule, selected)),
    )

    result = integration.maybe_wrap_forward_backward(_native_schedule, config)

    assert result[0] == expected
    assert calls.count(expected) == 1
    assert integration.d3_owns_data_schedule(config) is (dynamic_decoder and not dynamic_encoder)
    assert integration.mdp_owns_data_schedule(config) is (dynamic_decoder or dynamic_encoder)


def test_d4_evaluation_rejects_before_any_mutation(monkeypatch):
    integration.reset_for_testing()
    monkeypatch.setattr(integration, "_RUNTIME", _runtime(dynamic_encoder_cp=True))
    for name in (
        "_build_d4_facade_from_mcore",
        "wrap_finalize_model_grads",
        "_wrap_d4_forward_backward",
    ):
        monkeypatch.setattr(
            integration, name, lambda *_args, _name=name: pytest.fail(f"{_name} called")
        )

    with pytest.raises(MdpConfigurationError, match="training-only"):
        integration.maybe_wrap_forward_backward(_native_schedule, object(), training=False)
    assert integration._D4_FACADE is None


def test_d4_reuses_only_the_exact_finalized_training_config(monkeypatch):
    integration.reset_for_testing()
    runtime = _runtime(dynamic_encoder_cp=True)
    config = SimpleNamespace(
        dynamic_context_parallel=False,
        max_seqlen_per_dp_cp_rank=8192,
        min_dynamic_context_parallel_size=1,
    )
    facade = object()
    builds = []
    finalizers = []
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(
        integration,
        "_build_d4_facade_from_mcore",
        lambda selected_runtime, selected_config: (
            builds.append((selected_runtime, selected_config)) or facade
        ),
    )
    monkeypatch.setattr(
        integration, "_wrap_d4_forward_backward", lambda schedule, selected: (schedule, selected)
    )
    monkeypatch.setattr(
        integration, "wrap_finalize_model_grads", lambda *args: finalizers.append(args)
    )

    assert integration.maybe_wrap_forward_backward(_native_schedule, config) == (
        _native_schedule,
        facade,
    )
    assert integration.maybe_wrap_forward_backward(_native_schedule, config) == (
        _native_schedule,
        facade,
    )
    assert builds == [(runtime, config)]
    assert finalizers == [(config, runtime), (config, runtime)]

    other = SimpleNamespace(**vars(config))
    with pytest.raises(MdpConfigurationError, match="exact finalized training config"):
        integration.maybe_wrap_forward_backward(_native_schedule, other)
    config.max_seqlen_per_dp_cp_rank = 4096
    with pytest.raises(MdpConfigurationError, match="exact finalized training config"):
        integration.maybe_wrap_forward_backward(_native_schedule, config)


def test_runtime_requires_exact_bool_before_dispatch(monkeypatch):
    integration.reset_for_testing()
    monkeypatch.setattr(integration, "_RUNTIME", _runtime(dynamic_encoder_cp=True))
    monkeypatch.setattr(
        integration, "_build_d4_facade_from_mcore", lambda *_args: pytest.fail("facade constructed")
    )

    with pytest.raises(MdpConfigurationError, match="exact bool"):
        integration.maybe_wrap_forward_backward(_native_schedule, object(), training=1)


@pytest.mark.parametrize("dynamic_decoder", (False, True))
@pytest.mark.parametrize(
    "capture_mode",
    (VisionCaptureMode.SOURCE_PIXEL_SIDECAR, VisionCaptureMode.STABLE_LOCATOR_CATALOG),
)
def test_d4_mcore_factory_binds_fixed_or_joint_j1_facade(
    monkeypatch, dynamic_decoder, capture_mode
):
    binding = object()
    codec = SimpleNamespace(rebuild_microbatch=lambda *_args, **_kwargs: None)
    adapter = SimpleNamespace(
        build_dynamic_decoder_payload_codec=lambda: codec,
        estimate_dynamic_encoder_workload=lambda *_args, **_kwargs: None,
    )
    runtime = SimpleNamespace(
        config=MdpConfig(
            enable=True,
            encoder_cp=4,
            encoder_max_payload_rows=16384,
            dynamic_encoder_cp=True,
            min_dynamic_encoder_cp_size=2,
        ),
        adapter=adapter,
        dynamic_group_binding=binding,
        vision_capture_mode=capture_mode,
        hidden_size=4096,
        params_dtype=object(),
    )
    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        max_seqlen_per_dp_cp_rank=8192,
        min_dynamic_context_parallel_size=2,
    )
    capture_operations = object()
    capture_owner = object()
    captured = {}
    monkeypatch.setattr(
        integration,
        "_snapshot_d4_encoder_capture_operations",
        lambda selected_adapter, selected_codec: (
            capture_operations
            if (selected_adapter, selected_codec) == (adapter, codec)
            else pytest.fail("wrong capture dependencies")
        ),
    )

    def capture(**kwargs):
        captured["capture"] = kwargs
        return capture_owner

    def fixed(*args, **kwargs):
        captured["run"] = ("fixed", args, kwargs)

    def joint(*args, **kwargs):
        captured["run"] = ("joint", args, kwargs)

    monkeypatch.setattr(integration, "_capture_d4_encoder_source", capture)
    monkeypatch.setattr(integration, "_run_repeated_d4_fixed_iteration", fixed)
    monkeypatch.setattr(integration, "_run_repeated_d4_joint_iteration", joint)

    facade = integration._build_d4_facade_from_mcore(runtime, config)
    iterator = object()
    native_args = (object(),)
    native_kwargs = {"forward_only": False}
    facade(
        data_iterators=iterator,
        num_microbatches=7,
        forward_backward_func=_native_schedule,
        native_schedule_args=native_args,
        native_schedule_kwargs=native_kwargs,
    )

    assert captured["capture"] == {
        "runtime": runtime,
        "binding": binding,
        "data_iterators": iterator,
        "num_microbatches": 7,
        "operations": capture_operations,
        "capture_mode": capture_mode,
    }
    selected, args, kwargs = captured["run"]
    assert selected == ("joint" if dynamic_decoder else "fixed")
    assert args == (runtime, capture_owner)
    assert kwargs["decoder_max_seqlen_per_rank"] == 8192
    assert kwargs["encoder_max_seqlen_per_rank"] == 16384
    assert kwargs["encoder_minimum_cp_size"] == 2
    assert kwargs["bridge_width"] == 4096
    assert kwargs["cp_partition_mode"] == "contiguous"
    assert kwargs["forward_backward_func"] is _native_schedule
    assert kwargs["native_schedule_args"] is native_args
    assert kwargs["native_schedule_kwargs"] is native_kwargs
    assert ("decoder_minimum_cp_size" in kwargs) is dynamic_decoder
