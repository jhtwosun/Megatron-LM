# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Public-integration boundary tests for the private repeated-D4 schedule."""

import inspect
from types import SimpleNamespace

import pytest

from megatron.core.mdp import dynamic_cp_d4_transaction as transaction_api
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
    monkeypatch.setattr(transaction_api, "_validate_repeated_d4_group_binding", lambda value: value)
    lifecycle = transaction_api._bind_d4_checkpoint_lifecycle(runtime, binding)
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", lifecycle)

    facade = integration._build_d4_facade_from_mcore(runtime, config)
    initial = integration.get_d4_checkpoint_lifecycle_snapshot()
    assert initial.state is transaction_api._INITIAL_IDLE
    assert initial.may_load and not initial.may_save
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
    committed = integration.get_d4_checkpoint_lifecycle_snapshot()
    assert committed.state is transaction_api._COMMITTED_IDLE
    assert committed.may_save and not committed.may_load
    with pytest.raises(MdpStateError, match="stale|snapshot"):
        initial.require()

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


@pytest.mark.parametrize("failure_stage", ("begin", "capture", "run", "commit"))
def test_d4_shared_facade_poison_preserves_primary(monkeypatch, failure_stage):
    binding = object()
    runtime = SimpleNamespace(
        config=MdpConfig(
            enable=True,
            encoder_cp=4,
            encoder_max_payload_rows=16,
            dynamic_encoder_cp=True,
            min_dynamic_encoder_cp_size=1,
        ),
        adapter=SimpleNamespace(
            build_dynamic_decoder_payload_codec=lambda: SimpleNamespace(
                rebuild_microbatch=lambda *_args, **_kwargs: None
            ),
            estimate_dynamic_encoder_workload=lambda *_args, **_kwargs: None,
        ),
        dynamic_group_binding=binding,
        vision_capture_mode=VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
        hidden_size=8,
        params_dtype=object(),
    )
    config = SimpleNamespace(
        dynamic_context_parallel=False,
        max_seqlen_per_dp_cp_rank=8,
        min_dynamic_context_parallel_size=1,
    )
    primary = RuntimeError(f"{failure_stage} failed")
    monkeypatch.setattr(transaction_api, "_validate_repeated_d4_group_binding", lambda value: value)
    lifecycle = transaction_api._bind_d4_checkpoint_lifecycle(runtime, binding)
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", lifecycle)
    monkeypatch.setattr(
        integration, "_snapshot_d4_encoder_capture_operations", lambda *_args: object()
    )

    def capture(**_kwargs):
        assert failure_stage != "begin"
        if failure_stage == "capture":
            raise primary
        return object()

    def run(*_args, **_kwargs):
        if failure_stage == "run":
            raise primary

    if failure_stage in ("begin", "commit"):
        monkeypatch.setattr(
            transaction_api._D4CheckpointLifecycleOwner,
            f"{failure_stage}_iteration",
            lambda *_args: (_ for _ in ()).throw(primary),
        )

    monkeypatch.setattr(integration, "_capture_d4_encoder_source", capture)
    monkeypatch.setattr(integration, "_run_repeated_d4_fixed_iteration", run)
    facade = integration._build_d4_facade_from_mcore(runtime, config)

    with pytest.raises(RuntimeError, match=failure_stage) as raised:
        facade(
            data_iterators=iter((object(),)),
            num_microbatches=1,
            forward_backward_func=_native_schedule,
            native_schedule_args=(),
            native_schedule_kwargs={},
        )
    assert raised.value is primary
    snapshot = integration.get_d4_checkpoint_lifecycle_snapshot()
    assert snapshot.state is transaction_api._CHECKPOINT_POISONED
    assert not snapshot.may_load and not snapshot.may_save


@pytest.mark.parametrize("dynamic_decoder", (False, True))
def test_d4_facade_freezes_selected_iteration_runner(monkeypatch, dynamic_decoder):
    binding = object()
    codec = SimpleNamespace(rebuild_microbatch=lambda *_args, **_kwargs: None)
    runtime = SimpleNamespace(
        config=MdpConfig(
            enable=True,
            encoder_cp=4,
            encoder_max_payload_rows=16,
            dynamic_encoder_cp=True,
            min_dynamic_encoder_cp_size=1,
        ),
        adapter=SimpleNamespace(
            build_dynamic_decoder_payload_codec=lambda: codec,
            estimate_dynamic_encoder_workload=lambda *_args, **_kwargs: None,
        ),
        dynamic_group_binding=binding,
        vision_capture_mode=VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
        hidden_size=8,
        params_dtype=object(),
    )
    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        max_seqlen_per_dp_cp_rank=8,
        min_dynamic_context_parallel_size=1,
    )
    calls = []
    monkeypatch.setattr(transaction_api, "_validate_repeated_d4_group_binding", lambda value: value)
    lifecycle = transaction_api._bind_d4_checkpoint_lifecycle(runtime, binding)
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", lifecycle)
    monkeypatch.setattr(
        integration, "_snapshot_d4_encoder_capture_operations", lambda *_args: object()
    )
    monkeypatch.setattr(integration, "_capture_d4_encoder_source", lambda **_kwargs: object())

    def selected(*_args, **_kwargs):
        calls.append("selected")

    selected_name = (
        "_run_repeated_d4_joint_iteration"
        if dynamic_decoder
        else "_run_repeated_d4_fixed_iteration"
    )
    monkeypatch.setattr(integration, selected_name, selected)
    facade = integration._build_d4_facade_from_mcore(runtime, config)
    monkeypatch.setattr(
        integration,
        selected_name,
        lambda *_args, **_kwargs: pytest.fail("late runner substitution was observed"),
    )

    facade(
        data_iterators=iter((object(),)),
        num_microbatches=1,
        forward_backward_func=_native_schedule,
        native_schedule_args=(),
        native_schedule_kwargs={},
    )
    assert calls == ["selected"]


def test_d4_lifecycle_mint_is_between_optimizer_success_and_runtime_publication():
    source = inspect.getsource(integration.maybe_build_mdp_domain)
    optimizer = source.index("build_mdp_composite_optimizer")
    mint = source.index("_install_d4_checkpoint_lifecycle")
    publication = source.index("_RUNTIME = runtime")
    assert optimizer < mint < publication


def test_d4_lifecycle_supports_second_iteration_and_reset_retires_snapshot(monkeypatch):
    binding = object()
    runtime = SimpleNamespace(
        config=MdpConfig(enable=True, dynamic_encoder_cp=True),
        dynamic_adapter_capability=None,
        dynamic_group_binding=binding,
    )
    monkeypatch.setattr(transaction_api, "_validate_repeated_d4_group_binding", lambda value: value)
    lifecycle = transaction_api._bind_d4_checkpoint_lifecycle(runtime, binding)
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", lifecycle)

    lifecycle.begin_iteration(runtime, binding)
    lifecycle.commit_iteration(runtime, binding)
    first = integration.get_d4_checkpoint_lifecycle_snapshot()
    lifecycle.begin_iteration(runtime, binding)
    lifecycle.commit_iteration(runtime, binding)
    second = integration.get_d4_checkpoint_lifecycle_snapshot()
    assert first.generation == 1 and second.generation == 2
    with pytest.raises(MdpStateError, match="stale|snapshot"):
        first.require()

    integration.reset_for_testing()
    assert integration.get_d4_checkpoint_lifecycle_snapshot() is None
    with pytest.raises(MdpStateError, match="retired|stale"):
        second.require()


def test_static_runtime_has_no_repeated_d4_checkpoint_capability(monkeypatch):
    integration.reset_for_testing()
    monkeypatch.setattr(integration, "_RUNTIME", _runtime(dynamic_encoder_cp=False))
    assert integration.get_d4_checkpoint_lifecycle_snapshot() is None


@pytest.mark.parametrize(
    "pairing", ("runtime-only-d4", "lifecycle-only", "static-lifecycle", "foreign-pair")
)
def test_checkpoint_snapshot_query_rejects_incomplete_runtime_lifecycle_pair(monkeypatch, pairing):
    integration.reset_for_testing()
    runtime = _runtime(dynamic_encoder_cp=pairing != "static-lifecycle")
    lifecycle = object()
    monkeypatch.setattr(integration, "_RUNTIME", None if pairing == "lifecycle-only" else runtime)
    monkeypatch.setattr(
        integration, "_D4_CHECKPOINT_LIFECYCLE", None if pairing == "runtime-only-d4" else lifecycle
    )
    with pytest.raises(MdpStateError, match="runtime|lifecycle|pair"):
        integration.get_d4_checkpoint_lifecycle_snapshot()


def test_checkpoint_lifecycle_install_is_atomic_on_stale_global_and_mint_failure(monkeypatch):
    integration.reset_for_testing()
    runtime = SimpleNamespace(dynamic_group_binding=object())
    stale = object()
    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", stale)
    active_before = dict(transaction_api._ACTIVE_CHECKPOINT_LIFECYCLES)
    monkeypatch.setattr(
        transaction_api,
        "_bind_d4_checkpoint_lifecycle",
        lambda *_args: pytest.fail("mint called with stale lifecycle installed"),
    )
    with pytest.raises(MdpStateError, match="installed once|lifecycle"):
        integration._install_d4_checkpoint_lifecycle(runtime)
    assert integration._D4_CHECKPOINT_LIFECYCLE is stale
    assert transaction_api._ACTIVE_CHECKPOINT_LIFECYCLES == active_before

    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", None)
    primary = RuntimeError("mint failed")
    monkeypatch.setattr(
        transaction_api,
        "_bind_d4_checkpoint_lifecycle",
        lambda *_args: (_ for _ in ()).throw(primary),
    )
    with pytest.raises(RuntimeError, match="mint failed") as raised:
        integration._install_d4_checkpoint_lifecycle(runtime)
    assert raised.value is primary
    assert integration._RUNTIME is None
    assert integration._D4_CHECKPOINT_LIFECYCLE is None
    assert transaction_api._ACTIVE_CHECKPOINT_LIFECYCLES == active_before


def test_d4_facade_rejects_method_compatible_fake_lifecycle_before_capture(monkeypatch):
    binding = object()
    runtime = SimpleNamespace(
        config=MdpConfig(
            enable=True,
            encoder_cp=4,
            encoder_max_payload_rows=16,
            dynamic_encoder_cp=True,
            min_dynamic_encoder_cp_size=1,
        ),
        adapter=SimpleNamespace(
            build_dynamic_decoder_payload_codec=lambda: SimpleNamespace(
                rebuild_microbatch=lambda *_args, **_kwargs: None
            ),
            estimate_dynamic_encoder_workload=lambda *_args, **_kwargs: None,
        ),
        dynamic_group_binding=binding,
        vision_capture_mode=VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
        hidden_size=8,
        params_dtype=object(),
    )
    config = SimpleNamespace(
        dynamic_context_parallel=False,
        max_seqlen_per_dp_cp_rank=8,
        min_dynamic_context_parallel_size=1,
    )
    fake = SimpleNamespace(
        snapshot=lambda *_args: pytest.fail("fake snapshot called"),
        begin_iteration=lambda *_args: pytest.fail("fake begin called"),
        commit_iteration=lambda *_args: pytest.fail("fake commit called"),
        poison=lambda *_args: pytest.fail("fake poison called"),
    )
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", fake)
    monkeypatch.setattr(
        integration, "_snapshot_d4_encoder_capture_operations", lambda *_args: object()
    )
    monkeypatch.setattr(
        integration, "_capture_d4_encoder_source", lambda **_kwargs: pytest.fail("capture entered")
    )
    monkeypatch.setattr(
        integration,
        "_run_repeated_d4_fixed_iteration",
        lambda *_args, **_kwargs: pytest.fail("iteration entered"),
    )

    with pytest.raises(MdpStateError, match="exact checkpoint lifecycle pair"):
        integration._build_d4_facade_from_mcore(runtime, config)


def test_reset_retires_checkpoint_lifecycle_before_unpublishing_globals(monkeypatch):
    integration.reset_for_testing()
    runtime = SimpleNamespace(dynamic_adapter_capability=None, dynamic_group_binding=object())

    class Lifecycle:
        def retire(self, selected_runtime, selected_binding):
            assert selected_runtime is runtime
            assert selected_binding is runtime.dynamic_group_binding
            assert integration._RUNTIME is runtime
            assert integration._D4_CHECKPOINT_LIFECYCLE is self

    lifecycle = Lifecycle()
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", lifecycle)

    integration.reset_for_testing()
    assert integration._RUNTIME is None
    assert integration._D4_CHECKPOINT_LIFECYCLE is None
