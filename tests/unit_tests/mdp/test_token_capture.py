# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Global token capture contract tests (API design 13.2). CPU only.

The capture must hold the same storage the native finalizer reduced in place,
happen exactly once per iteration, reject None, stay absent on the evaluation
path, and leave the config untouched when MDP is off.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.schedule import (
    _wrap_d3_forward_backward,
    unwrap_finalize_model_grads,
    wrap_finalize_model_grads,
    wrap_forward_backward,
)


class _CaptureProbe:
    """Stands in for MdpRuntime's capture surface."""

    def __init__(self):
        self.captured = None
        self.count = 0

    def capture_global_num_tokens(self, token_tensor):
        if token_tensor is None:
            raise MdpConfigurationError("calculate_per_token_loss must be True")
        if self.count:
            raise MdpStateError("captured more than once")
        self.captured = token_tensor
        self.count += 1


def _native_finalizer(calls):
    def native(model, num_tokens=None, *args, **kwargs):
        calls.append((model, num_tokens, args, kwargs))
        if num_tokens is not None:
            num_tokens.mul_(4)  # in-place, like the real PP/DP-CP reduction
        return "native-result"

    return native


def test_capture_holds_the_in_place_reduced_storage():
    calls = []
    config = SimpleNamespace(finalize_model_grads_func=_native_finalizer(calls))
    probe = _CaptureProbe()
    wrap_finalize_model_grads(config, probe)

    tokens = torch.tensor(5.0)
    original_ptr = tokens.data_ptr()
    result = config.finalize_model_grads_func(
        ["model"], tokens, pg_collection="pgc", force_all_reduce=False
    )
    assert result == "native-result"
    # Extra dev-schedule kwargs pass through to the native finalizer.
    assert calls[0][3] == {"pg_collection": "pgc", "force_all_reduce": False}
    # Capture happened AFTER the in-place reduction, on the same storage.
    assert probe.captured is tokens
    assert probe.captured.data_ptr() == original_ptr
    assert float(probe.captured) == 20.0
    # A clone taken before the native call would have missed the reduction —
    # the fault this contract exists to catch.
    assert float(tokens.clone() / 4) != 20.0


def test_capture_is_exactly_once_and_rejects_none():
    config = SimpleNamespace(finalize_model_grads_func=_native_finalizer([]))
    probe = _CaptureProbe()
    wrap_finalize_model_grads(config, probe)
    tokens = torch.tensor(1.0)
    config.finalize_model_grads_func(["m"], tokens)
    with pytest.raises(MdpStateError, match="more than once"):
        config.finalize_model_grads_func(["m"], tokens)
    fresh = SimpleNamespace(finalize_model_grads_func=_native_finalizer([]))
    wrap_finalize_model_grads(fresh, _CaptureProbe())
    with pytest.raises(MdpConfigurationError, match="per_token_loss"):
        fresh.finalize_model_grads_func(["m"], None)


def test_wrap_is_idempotent_and_unwrap_restores_native():
    native = _native_finalizer([])
    config = SimpleNamespace(finalize_model_grads_func=native)
    probe = _CaptureProbe()
    wrap_finalize_model_grads(config, probe)
    wrapped = config.finalize_model_grads_func
    wrap_finalize_model_grads(config, probe)  # idempotent per config object
    assert config.finalize_model_grads_func is wrapped
    unwrap_finalize_model_grads(config)
    assert config.finalize_model_grads_func is native


def test_wrap_requires_an_installed_native_finalizer():
    config = SimpleNamespace(finalize_model_grads_func=None)
    with pytest.raises(MdpConfigurationError, match="finalize_model_grads_func"):
        wrap_finalize_model_grads(config, _CaptureProbe())


def test_mdp_off_leaves_every_integration_point_untouched():
    from megatron.core.mdp import integration

    integration.reset_for_testing()
    config = SimpleNamespace(finalize_model_grads_func=_native_finalizer([]))
    native = config.finalize_model_grads_func

    def schedule(data_iterator, num_microbatches, forward_only):
        return "result"

    same = integration.maybe_wrap_forward_backward(schedule, config)
    assert same is schedule
    assert config.finalize_model_grads_func is native
    assert not getattr(config, "_mdp_finalize_wrapped", False)
    args = SimpleNamespace(mdp_enable=False)
    assert (
        integration.maybe_build_mdp_domain(
            args=args, model=[], optimizer="opt", optimizer_config=None, ddp_config=None
        )
        == "opt"
    )


class _FakeRuntime:
    """Order-verifying stand-in for MdpRuntime in the schedule wrapper."""

    def __init__(self):
        self.events = []

    def begin_iteration(self, data_iterator, *, num_microbatches, forward_only):
        self.events.append(("begin", num_microbatches, forward_only))
        return [iter([data_iterator])]

    def mark_decoder_complete(self):
        self.events.append(("decoder_done",))

    def end_iteration(self):
        self.events.append(("end",))


def test_forward_backward_wrapper_order_and_double_wrap():
    runtime = _FakeRuntime()

    def schedule(*, data_iterator, num_microbatches, forward_only):
        runtime.events.append(("schedule", next(data_iterator)))
        return {"loss": 1.0}

    wrapped = wrap_forward_backward(schedule, runtime)
    result = wrapped(data_iterator="real-iter", num_microbatches=3, forward_only=False)
    assert result == {"loss": 1.0}
    assert runtime.events == [
        ("begin", 3, False),
        ("schedule", "real-iter"),
        ("decoder_done",),
        ("end",),
    ]
    with pytest.raises(MdpConfigurationError, match="wrapped twice"):
        wrap_forward_backward(wrapped, runtime)


def _d3_facade(events, *, records=("record-0", "record-1")):
    from megatron.core.mdp.dynamic_cp_d3_private_facade import _D3PrivateFacade

    facade = object.__new__(_D3PrivateFacade)
    facade._active = None

    def begin(data, **kwargs):
        events.append(("begin", data, kwargs))
        return SimpleNamespace(records=records)

    facade.begin_iteration = begin
    facade.decoder_replay_cursor = lambda ready, **kwargs: iter(ready.records)
    facade.mark_decoder_complete = lambda ready: events.append(("decoder_done", ready.records))
    facade.end_iteration = lambda ready: events.append(("end", ready.records))

    def abort(ready, error):
        events.append(("abort", ready.records, error))
        raise error

    facade.abort_scheduled_iteration = abort
    return facade


def test_d3_wrapper_replays_dynamic_count_through_native_schedule():
    events = []
    facade = _d3_facade(events)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        events.append(("schedule", tuple(data_iterator), num_microbatches, forward_only))
        return "native-result"

    wrapped = _wrap_d3_forward_backward(schedule, facade)
    result = wrapped(data_iterator=iter(("raw",)), num_microbatches=7, forward_only=False)

    assert result == "native-result"
    assert events[0][0] == "begin"
    assert events[0][2] == {"num_microbatches": 7, "forward_only": False}
    assert events[1:] == [
        ("schedule", ("record-0", "record-1"), 2, False),
        ("decoder_done", ("record-0", "record-1")),
        ("end", ("record-0", "record-1")),
    ]
    with pytest.raises(MdpConfigurationError, match="wrapped twice"):
        _wrap_d3_forward_backward(wrapped, facade)


def test_d3_wrapper_preserves_native_schedule_error_and_aborts():
    events = []
    facade = _d3_facade(events, records=("record",))
    primary = RuntimeError("native schedule failed")

    def schedule(*, data_iterator, num_microbatches, forward_only):
        del data_iterator, num_microbatches, forward_only
        raise primary

    wrapped = _wrap_d3_forward_backward(schedule, facade)
    with pytest.raises(RuntimeError, match="native schedule failed") as caught:
        wrapped(data_iterator=iter(("raw",)), num_microbatches=1, forward_only=False)

    assert caught.value is primary
    assert events[0][0] == "begin"
    assert events[1:] == [("abort", ("record",), primary)]


def test_d3_wrapper_rejects_vpp_before_starting_iteration():
    events = []
    facade = _d3_facade(events)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        del data_iterator, num_microbatches, forward_only

    wrapped = _wrap_d3_forward_backward(schedule, facade)
    with pytest.raises(MdpConfigurationError, match="VPP1"):
        wrapped(data_iterator=[iter((0,)), iter((1,))], num_microbatches=1, forward_only=False)
    assert events == []
