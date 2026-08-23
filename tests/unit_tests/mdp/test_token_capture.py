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


class _MultiChunkRuntime:
    """VPP-shaped runtime probe with independently tracked replay cursors."""

    def __init__(self, num_chunks):
        self.num_chunks = num_chunks
        self.events = []
        self.cursor_reads = [0] * num_chunks
        self.active = False

    def begin_iteration(self, data_iterator, *, num_microbatches, forward_only):
        if isinstance(data_iterator, list):
            assert len(data_iterator) == self.num_chunks
            source_iterators = tuple(data_iterator)
        else:
            assert self.num_chunks == 1
            source_iterators = (data_iterator,)
        assert not self.active
        self.active = True
        self.events.append(("begin", num_microbatches, forward_only, source_iterators))

        def replay(chunk):
            for microbatch in range(num_microbatches):
                self.cursor_reads[chunk] += 1
                yield (chunk, microbatch)

        return [replay(chunk) for chunk in range(self.num_chunks)]

    def mark_decoder_complete(self):
        assert self.active
        self.events.append(("decoder_done",))

    def end_iteration(self):
        assert self.active
        self.events.append(("end",))
        self.active = False


def test_forward_backward_wrapper_vpp_consumes_every_replay_cursor_once():
    runtime = _MultiChunkRuntime(num_chunks=2)
    real_iterators = [iter(("raw-0",)), iter(("raw-1",))]
    models = [object(), object()]
    shape_adjuster = object()
    observed = {}

    def interleaved_schedule(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        seq_length,
        micro_batch_size,
        decoder_seq_length=None,
        forward_only=False,
        collect_non_loss_data=False,
        first_val_step=None,
        adjust_tensor_shapes_fn=None,
        pg_collection=None,
        force_all_reduce=False,
    ):
        observed.update(
            model=model,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )
        observed["replayed"] = tuple(
            tuple(next(cursor) for _ in range(num_microbatches)) for cursor in data_iterator
        )
        return {"loss": 1.0}

    wrapped = wrap_forward_backward(interleaved_schedule, runtime)
    result = wrapped(
        forward_step_func=object(),
        data_iterator=real_iterators,
        model=models,
        num_microbatches=3,
        seq_length=32,
        micro_batch_size=1,
        decoder_seq_length=24,
        forward_only=False,
        collect_non_loss_data=True,
        first_val_step=False,
        adjust_tensor_shapes_fn=shape_adjuster,
        pg_collection="pgc",
        force_all_reduce=True,
    )

    assert result == {"loss": 1.0}
    assert observed == {
        "model": models,
        "seq_length": 32,
        "micro_batch_size": 1,
        "decoder_seq_length": 24,
        "collect_non_loss_data": True,
        "first_val_step": False,
        "adjust_tensor_shapes_fn": shape_adjuster,
        "pg_collection": "pgc",
        "force_all_reduce": True,
        "replayed": (((0, 0), (0, 1), (0, 2)), ((1, 0), (1, 1), (1, 2))),
    }
    assert runtime.cursor_reads == [3, 3]
    assert [event[0] for event in runtime.events] == ["begin", "decoder_done", "end"]
    assert runtime.events[0][3] == tuple(real_iterators)
    assert not runtime.active


def test_forward_backward_wrapper_preserves_mtp_losses_and_schedule_kwargs():
    runtime = _MultiChunkRuntime(num_chunks=1)
    model = SimpleNamespace(config=SimpleNamespace(mtp_num_layers=1))
    mtp_result = [{"lm loss": 1.5, "mtp_1 loss": 0.25}]
    observed = {}

    def mtp_schedule(
        *,
        data_iterator,
        model,
        num_microbatches,
        forward_only,
        collect_non_loss_data=False,
        first_val_step=None,
        pg_collection=None,
        force_all_reduce=False,
    ):
        assert model.config.mtp_num_layers == 1
        observed.update(
            replayed=tuple(next(data_iterator) for _ in range(num_microbatches)),
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )
        return mtp_result

    wrapped = wrap_forward_backward(mtp_schedule, runtime)
    result = wrapped(
        data_iterator=iter(("raw",)),
        model=model,
        num_microbatches=2,
        forward_only=False,
        collect_non_loss_data=True,
        first_val_step=True,
        pg_collection="mtp-pgc",
        force_all_reduce=True,
    )

    assert result is mtp_result
    assert result[0]["mtp_1 loss"] == 0.25
    assert observed == {
        "replayed": ((0, 0), (0, 1)),
        "collect_non_loss_data": True,
        "first_val_step": True,
        "pg_collection": "mtp-pgc",
        "force_all_reduce": True,
    }
    assert runtime.cursor_reads == [2]
    assert [event[0] for event in runtime.events] == ["begin", "decoder_done", "end"]
    assert not runtime.active
