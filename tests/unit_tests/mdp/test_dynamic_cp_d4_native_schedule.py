# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for the private repeated-D4 native-schedule boundary."""

import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_dynamic_decoder_replay as dynamic_api
from megatron.core.mdp import dynamic_cp_d4_encoder_forward as forward_api
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay_api
from megatron.core.mdp import dynamic_cp_d4_native_schedule as api
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError


def _owner(monkeypatch, *, records=("record-0", "record-1")):
    owner = object.__new__(replay_api._D4FixedDecoderReplayOwner)
    state = SimpleNamespace(
        records=records,
        cursor=None,
        token=None,
        descriptor=None,
        schedule_return=None,
        completion=None,
        events=[],
    )
    owner.records = records

    def require(self):
        state.events.append("require")
        return self

    def replay_cursor(self):
        state.events.append("cursor")
        if state.cursor is not None:
            raise MdpStateError("cursor already exists")
        cursor = replay_api._D4FixedDecoderReplayCursor(self)
        replay_api._ACTIVE_CURSORS[id(cursor)] = (
            weakref.ref(cursor),
            weakref.ref(self),
            state.records,
            0,
        )
        state.cursor = cursor
        return cursor

    def capture(self, token):
        state.events.append(("capture", token))
        if state.token is not None:
            raise MdpStateError("token already captured")
        state.token = token
        state.descriptor = replay_api._tensor_descriptor(token)

    def returned(self, cursor):
        state.events.append(("returned", cursor))
        cursor_entry = replay_api._ACTIVE_CURSORS.get(id(cursor))
        if cursor is not state.cursor or cursor_entry is None or cursor_entry[3] != len(records):
            raise MdpStateError("cursor not exhausted")
        value = object()
        state.schedule_return = value
        return value

    def prepare(self, cursor, schedule_return):
        state.events.append(("prepare", cursor, schedule_return))
        if (
            cursor is not state.cursor
            or schedule_return is not state.schedule_return
            or state.token is None
            or replay_api._tensor_descriptor(state.token) != state.descriptor
        ):
            raise MdpStateError("completion token changed")
        completion = replay_api._D4FixedDecoderCompletion(
            SimpleNamespace(), state.token, self, replay_api._COMPLETION_SEAL
        )
        state.completion = completion
        return completion

    def require_completion(self, completion):
        state.events.append(("require_completion", completion))
        if completion is not state.completion:
            raise MdpStateError("completion substituted")
        return completion

    monkeypatch.setattr(replay_api._D4FixedDecoderReplayOwner, "require", require)
    monkeypatch.setattr(replay_api._D4FixedDecoderReplayOwner, "replay_cursor", replay_cursor)
    monkeypatch.setattr(replay_api._D4FixedDecoderReplayOwner, "capture_global_num_tokens", capture)
    monkeypatch.setattr(replay_api._D4FixedDecoderReplayOwner, "mark_schedule_returned", returned)
    monkeypatch.setattr(replay_api._D4FixedDecoderReplayOwner, "prepare_completion", prepare)
    monkeypatch.setattr(
        replay_api._D4FixedDecoderReplayOwner, "require_completion", require_completion
    )
    return owner, state


def _aborter(events):
    def abort(owner, error):
        assert api._ACTIVE_TOKEN_SINK is None
        assert api._TOKEN_SINKS == {}
        events.append((owner, error))

    return abort


def _dynamic_owner(monkeypatch, *, records=("record-0", "record-1")):
    runtime = SimpleNamespace(device=torch.device("cpu"))
    authority = object()
    binding = object()
    released = []
    restored = []
    graph_releases = []
    leaf = torch.empty(1)
    buffer = torch.empty(1)
    operations = SimpleNamespace(release=released.append)
    handle = SimpleNamespace(release_forward_only=lambda: graph_releases.append(True))
    binding_owner = SimpleNamespace(restore=lambda primary=None: restored.append(primary))
    handoff_trusted = (
        runtime,
        authority,
        binding,
        (0,),
        object(),
        object(),
        object(),
        SimpleNamespace(receive_buffer=buffer),
        object(),
        MappingProxyType({}),
        handle,
        False,
        True,
        True,
        binding_owner,
        (buffer,),
        operations,
    )
    trusted = (
        runtime,
        authority,
        binding,
        handoff_trusted,
        lambda **_kwargs: None,
        operations.release,
        runtime.device,
    )
    owner = dynamic_api._D4DynamicDecoderReplayOwner(trusted, seal=dynamic_api._OWNER_SEAL)
    reference = weakref.ref(owner)
    escrow = dynamic_api._OwnerEscrow(reference, None, seal=dynamic_api._OWNER_ESCROW_SEAL)
    ready = SimpleNamespace(records=records, embedding_leaves=MappingProxyType({"leaf": leaf}))
    cleanup = dynamic_api._CleanupEscrow(seal=dynamic_api._CLEANUP_ESCROW_SEAL)
    entry = (reference, *trusted, cleanup, escrow, (leaf,), ready)
    escrow.entry = entry
    owner.ready = ready
    owner.records = records
    owner.embedding_leaves = ready.embedding_leaves
    owner._state = dynamic_api._ACTIVE
    owner._trusted = entry
    dynamic_api._ACTIVE_OWNERS[id(owner)] = entry
    dynamic_api._TRUSTED_OWNERS[id(owner)] = entry
    dynamic_api._OWNER_ESCROWS[id(owner)] = escrow
    forward_api._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    monkeypatch.setattr(dynamic_api, "_snapshot_local_authority", lambda actual, expected: expected)
    state = SimpleNamespace(
        records=records,
        runtime=runtime,
        released=released,
        restored=restored,
        graph_releases=graph_releases,
        resources=(leaf, buffer),
    )
    return owner, state


@pytest.mark.parametrize("container", (lambda value: value, lambda value: [value]))
def test_dynamic_frontend_preserves_native_result_record_order_and_token(monkeypatch, container):
    owner, state = _dynamic_owner(monkeypatch)
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)
    native_result = object()
    seen = []

    def schedule(*, data_iterator, num_microbatches, forward_only):
        assert forward_only is False
        assert num_microbatches == len(state.records)
        iterator = data_iterator[0] if isinstance(data_iterator, list) else data_iterator
        seen.extend(next(iterator) for _ in range(num_microbatches))
        finalizer("model", torch.tensor(5.0))
        return native_result

    result = api._run_d4_dynamic_native_schedule(
        schedule,
        owner,
        _aborter([]),
        data_iterator=container(iter(("raw",))),
        num_microbatches=99,
        forward_only=False,
    )

    assert type(result) is api._D4DynamicNativeScheduleSuccess
    assert result.native_result is native_result
    lifecycle = dynamic_api._ACTIVE_OWNERS[id(owner)][-3]
    assert result.completion is lifecycle.completion
    assert result.completion.globally_reduced_num_tokens is lifecycle.token
    assert seen == list(state.records)
    owner.abort()


@pytest.mark.parametrize(
    ("data_iterator", "forward_only", "message"),
    (([iter(()), iter(())], False, "VPP1 exactly"), (iter(()), True, "training-only")),
)
def test_dynamic_frontend_rejects_vpp_and_forward_only_before_cursor(
    monkeypatch, data_iterator, forward_only, message
):
    owner, state = _dynamic_owner(monkeypatch)
    schedule_calls = []
    aborts = []

    def schedule(*, data_iterator, num_microbatches, forward_only):
        schedule_calls.append((data_iterator, num_microbatches, forward_only))

    with pytest.raises(MdpConfigurationError, match=message):
        api._run_d4_dynamic_native_schedule(
            schedule,
            owner,
            _aborter(aborts),
            data_iterator=data_iterator,
            num_microbatches=1,
            forward_only=forward_only,
        )
    assert dynamic_api._ACTIVE_OWNERS[id(owner)][-3].cursor is None
    assert schedule_calls == []
    assert aborts == []
    owner.abort()


@pytest.mark.parametrize(
    "failure",
    (
        "before",
        "finalizer",
        "after",
        "missing",
        "duplicate",
        "underconsume",
        "overconsume",
        "mutated",
    ),
)
def test_dynamic_frontend_failure_retires_sink_aborts_once_and_allows_fresh_run(
    monkeypatch, failure
):
    owner, _state = _dynamic_owner(monkeypatch, records=("record",))
    aborts = []
    nested_aborts = []
    primary = RuntimeError(f"dynamic native {failure}")

    def native_finalizer(_model, _token):
        if failure == "finalizer":
            raise primary

    finalizer = api._wrap_d4_native_finalizer(native_finalizer)

    def scheduled_abort(actual, error):
        assert api._ACTIVE_TOKEN_SINK is None
        assert api._TOKEN_SINKS == {}
        if failure == "before":
            with pytest.raises(MdpStateError, match="one active invocation"):
                api._run_d4_dynamic_native_schedule(
                    schedule,
                    actual,
                    _aborter(nested_aborts),
                    data_iterator=iter(("raw",)),
                    num_microbatches=1,
                    forward_only=False,
                )
        actual.abort(error)
        aborts.append((actual, error))

    def schedule(*, data_iterator, num_microbatches, forward_only):
        token = None
        if failure != "underconsume":
            next(data_iterator)
        if failure == "overconsume":
            next(data_iterator)
        if failure == "before":
            raise primary
        if failure != "missing":
            token = torch.tensor(1.0)
            finalizer("model", token)
        if failure == "duplicate":
            finalizer("model", torch.tensor(2.0))
        if failure == "mutated":
            token.add_(1)
        if failure == "after":
            raise primary
        return "bad"

    expected = RuntimeError if failure in ("before", "finalizer", "after") else MdpStateError
    with pytest.raises(expected) as caught:
        api._run_d4_dynamic_native_schedule(
            schedule,
            owner,
            scheduled_abort,
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    if expected is RuntimeError:
        assert caught.value is primary
    assert aborts == [(owner, caught.value)]
    assert nested_aborts == []
    assert api._ACTIVE_TOKEN_SINK is None
    assert api._TOKEN_SINKS == {}
    assert api._ACTIVE_BOUNDARY is None
    assert api._BOUNDARIES == {}
    assert dynamic_api._ACTIVE_CURSORS == {}
    assert dynamic_api._ACTIVE_COMPLETIONS == {}
    assert _state.graph_releases == [True]
    assert len(_state.restored) == 1
    assert _state.released == list(_state.resources)

    fresh, _fresh_state = _dynamic_owner(monkeypatch, records=("fresh",))
    fresh_finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)

    def retry(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        fresh_finalizer("model", torch.tensor(3.0))
        return "fresh"

    fresh_result = api._run_d4_dynamic_native_schedule(
        retry,
        fresh,
        _aborter([]),
        data_iterator=iter(("raw",)),
        num_microbatches=1,
        forward_only=False,
    )
    assert fresh_result.native_result == "fresh"
    fresh.abort()


def test_dynamic_cursor_registry_substitution_is_preserved_during_trusted_abort(monkeypatch):
    owner, state = _dynamic_owner(monkeypatch, records=("record",))
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)
    foreign = (object(),)
    aborts = []

    def schedule(*, data_iterator, num_microbatches, forward_only):
        cursor = data_iterator
        next(cursor)
        dynamic_api._ACTIVE_CURSORS[id(cursor)] = foreign
        finalizer("model", torch.tensor(1.0))

    def abort(actual, error):
        actual.abort(error)
        aborts.append((actual, error))

    with pytest.raises(MdpStateError, match="cursor exhaustion") as caught:
        api._run_d4_dynamic_native_schedule(
            schedule,
            owner,
            abort,
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )

    assert aborts == [(owner, caught.value)]
    assert dynamic_api._ACTIVE_CURSORS[next(iter(dynamic_api._ACTIVE_CURSORS))] is foreign
    assert state.released == list(state.resources)
    assert state.graph_releases == [True]
    assert len(state.restored) == 1
    dynamic_api._ACTIVE_CURSORS.clear()


def test_dynamic_completion_registry_substitution_aborts_without_foreign_mutation(monkeypatch):
    owner, state = _dynamic_owner(monkeypatch, records=("record",))
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)
    original = dynamic_api._D4DynamicDecoderReplayOwner.prepare_completion
    foreign = (object(),)

    def substitute(self, cursor, schedule_return):
        completion = original(self, cursor, schedule_return)
        dynamic_api._ACTIVE_COMPLETIONS[id(completion)] = foreign
        return completion

    monkeypatch.setattr(dynamic_api._D4DynamicDecoderReplayOwner, "prepare_completion", substitute)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        finalizer("model", torch.tensor(1.0))

    with pytest.raises(MdpStateError, match="exact owner and token"):
        api._run_d4_dynamic_native_schedule(
            schedule,
            owner,
            lambda actual, error: actual.abort(error),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )

    assert dynamic_api._ACTIVE_COMPLETIONS[next(iter(dynamic_api._ACTIVE_COMPLETIONS))] is foreign
    assert state.released == list(state.resources)
    dynamic_api._ACTIVE_COMPLETIONS.clear()


def _real_owner(monkeypatch, *, runtime=None, records=("record",)):
    runtime = runtime or SimpleNamespace(device=torch.device("cpu"))
    released = []
    operations = SimpleNamespace(release=released.append)
    trusted = (
        runtime,
        object(),
        object(),
        records,
        MappingProxyType({}),
        (None, (), operations, None),
        (),
    )
    owner = replay_api._D4FixedDecoderReplayOwner(trusted=trusted, seal=replay_api._OWNER_SEAL)
    reference = weakref.ref(owner)
    replay_api._ACTIVE_OWNERS[id(owner)] = (reference, *trusted, replay_api._OwnerEscrow())
    forward_api._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    monkeypatch.setattr(
        replay_api, "_snapshot_local_authority", lambda binding, authority: authority
    )
    return owner, runtime, released


@pytest.mark.parametrize("container", (lambda value: value, lambda value: [value]))
def test_success_preserves_native_result_iterator_shape_and_exact_token(monkeypatch, container):
    owner, state = _owner(monkeypatch)
    aborts = []
    legacy = []
    finalizer_calls = []

    def native_finalizer(model, token, **kwargs):
        finalizer_calls.append((model, token, kwargs))
        token.mul_(4)
        return "finalizer-result"

    finalizer = api._wrap_d4_native_finalizer(native_finalizer, legacy.append)
    outside = torch.tensor(2.0)
    assert finalizer("legacy", outside, flag=True) == "finalizer-result"
    assert legacy == [outside]
    result = object()
    seen = []

    def schedule(*, data_iterator, num_microbatches, forward_only):
        assert forward_only is False
        assert num_microbatches == 2
        iterator = data_iterator[0] if isinstance(data_iterator, list) else data_iterator
        seen.extend(next(iterator) for _ in range(num_microbatches))
        token = torch.tensor(3.0)
        assert finalizer("model", token) == "finalizer-result"
        assert float(token) == 12.0
        return result

    success = api._run_d4_native_schedule(
        schedule,
        owner,
        _aborter(aborts),
        data_iterator=container(iter(("raw",))),
        num_microbatches=99,
        forward_only=False,
    )

    assert type(success) is api._D4NativeScheduleSuccess
    assert success.native_result is result
    assert success.completion is state.completion
    assert success.completion.globally_reduced_num_tokens is state.token
    assert seen == list(state.records)
    assert aborts == []
    assert api._ACTIVE_TOKEN_SINK is None
    assert api._TOKEN_SINKS == {}
    assert legacy == [outside]


@pytest.mark.parametrize("failure_point", ("before", "after"))
def test_native_failure_preserves_primary_retires_sink_and_fresh_retry(monkeypatch, failure_point):
    owner, _state = _owner(monkeypatch, records=("record",))
    aborts = []
    primary = RuntimeError(f"native {failure_point} finalizer failure")
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        if failure_point == "before":
            raise primary
        finalizer("model", torch.tensor(1.0))
        raise primary

    with pytest.raises(RuntimeError, match=str(primary)) as caught:
        api._run_d4_native_schedule(
            schedule,
            owner,
            _aborter(aborts),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert caught.value is primary
    assert aborts == [(owner, primary)]
    assert api._ACTIVE_TOKEN_SINK is None
    assert api._TOKEN_SINKS == {}

    fresh, _ = _owner(monkeypatch, records=("fresh",))
    fresh_finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)

    def retry(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        fresh_finalizer("model", torch.tensor(1.0))
        return "retry"

    assert (
        api._run_d4_native_schedule(
            retry,
            fresh,
            _aborter([]),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        ).native_result
        == "retry"
    )


@pytest.mark.parametrize("mode", ("missing", "duplicate", "mutated"))
def test_token_capture_is_exactly_once_and_stable(monkeypatch, mode):
    owner, state = _owner(monkeypatch, records=("record",))
    aborts = []
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        if mode != "missing":
            token = torch.tensor(2.0)
            finalizer("model", token)
            if mode == "duplicate":
                finalizer("model", torch.tensor(2.0))
            else:
                token.add_(1)
        return "result"

    message = {
        "missing": "exact post-finalizer token",
        "duplicate": "exactly once",
        "mutated": "exact post-finalizer token",
    }[mode]
    with pytest.raises(MdpStateError, match=message) as caught:
        api._run_d4_native_schedule(
            schedule,
            owner,
            _aborter(aborts),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert aborts == [(owner, caught.value)]
    assert api._ACTIVE_TOKEN_SINK is None
    assert api._TOKEN_SINKS == {}
    assert state.completion is None


def test_reentry_and_sink_substitution_abort_once_without_leak(monkeypatch):
    owner, _state = _owner(monkeypatch, records=("record",))
    aborts = []
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)
    nested = []

    def schedule(*, data_iterator, num_microbatches, forward_only):
        try:
            api._run_d4_native_schedule(
                schedule,
                owner,
                _aborter(nested),
                data_iterator=iter(("raw",)),
                num_microbatches=1,
                forward_only=False,
            )
        except MdpStateError as error:
            assert str(error) == "MDP: D4 native schedule permits one active invocation."
        next(data_iterator)
        finalizer("model", torch.tensor(1.0))
        return "ok"

    assert (
        api._run_d4_native_schedule(
            schedule,
            owner,
            _aborter(aborts),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        ).native_result
        == "ok"
    )
    assert nested == []
    assert aborts == []

    owner, _state = _owner(monkeypatch, records=("record",))

    def substitute(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        api._ACTIVE_TOKEN_SINK.token = torch.tensor(9.0)
        return "bad"

    with pytest.raises(MdpStateError, match="exact state") as caught:
        api._run_d4_native_schedule(
            substitute,
            owner,
            _aborter(aborts),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert aborts[-1] == (owner, caught.value)
    assert api._ACTIVE_TOKEN_SINK is None
    assert api._TOKEN_SINKS == {}


def test_scheduled_abort_reentry_rejects_before_cursor_or_nested_abort(monkeypatch):
    owner, state = _owner(monkeypatch, records=("record",))
    primary = RuntimeError("schedule failed")
    outer_aborts = []
    nested_aborts = []

    def schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        raise primary

    def abort(actual, error):
        with pytest.raises(MdpStateError, match="one active invocation"):
            api._run_d4_native_schedule(
                schedule,
                actual,
                _aborter(nested_aborts),
                data_iterator=iter(("raw",)),
                num_microbatches=1,
                forward_only=False,
            )
        outer_aborts.append((actual, error))

    with pytest.raises(RuntimeError, match="schedule failed") as caught:
        api._run_d4_native_schedule(
            schedule,
            owner,
            abort,
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert caught.value is primary
    assert outer_aborts == [(owner, primary)]
    assert nested_aborts == []
    assert state.events.count("cursor") == 1
    assert api._ACTIVE_BOUNDARY is None
    assert api._BOUNDARIES == {}


@pytest.mark.parametrize("mutation", ("delete", "substitute"))
def test_scheduled_abort_cannot_delete_active_boundary_and_reenter(monkeypatch, mutation):
    owner, state = _owner(monkeypatch, records=("record",))
    primary = RuntimeError("schedule failed")
    outer_aborts = []
    nested_aborts = []
    foreign = object()

    def schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        raise primary

    def abort(actual, error):
        api._ACTIVE_BOUNDARY = None if mutation == "delete" else foreign
        with pytest.raises(MdpStateError, match="one active invocation"):
            api._run_d4_native_schedule(
                schedule,
                actual,
                _aborter(nested_aborts),
                data_iterator=iter(("raw",)),
                num_microbatches=1,
                forward_only=False,
            )
        outer_aborts.append((actual, error))

    with pytest.raises(RuntimeError, match="schedule failed") as caught:
        api._run_d4_native_schedule(
            schedule,
            owner,
            abort,
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert caught.value is primary
    assert outer_aborts == [(owner, primary)]
    assert nested_aborts == []
    assert state.events.count("cursor") == 1
    assert api._BOUNDARIES == {}
    if mutation == "substitute":
        assert api._ACTIVE_BOUNDARY is foreign
        api._ACTIVE_BOUNDARY = None
    else:
        assert api._ACTIVE_BOUNDARY is None

    fresh_owner, _fresh_state = _owner(monkeypatch, records=("record",))
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)

    def fresh_schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        finalizer("model", torch.tensor(1.0))
        return "fresh"

    assert (
        api._run_d4_native_schedule(
            fresh_schedule,
            fresh_owner,
            _aborter([]),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        ).native_result
        == "fresh"
    )


def test_cursor_and_completion_substitution_abort_before_success(monkeypatch):
    owner, state = _owner(monkeypatch, records=("record",))
    aborts = []
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)

    def substitute_cursor(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        api._ACTIVE_TOKEN_SINK.cursor = object.__new__(replay_api._D4FixedDecoderReplayCursor)
        finalizer("model", torch.tensor(1.0))

    with pytest.raises(MdpStateError, match="exact state") as caught:
        api._run_d4_native_schedule(
            substitute_cursor,
            owner,
            _aborter(aborts),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert aborts == [(owner, caught.value)]

    owner, state = _owner(monkeypatch, records=("record",))
    original_prepare = replay_api._D4FixedDecoderReplayOwner.prepare_completion

    def substitute_completion(self, cursor, schedule_return):
        completion = original_prepare(self, cursor, schedule_return)
        return replay_api._D4FixedDecoderCompletion(
            completion.authority,
            completion.globally_reduced_num_tokens,
            self,
            replay_api._COMPLETION_SEAL,
        )

    monkeypatch.setattr(
        replay_api._D4FixedDecoderReplayOwner, "prepare_completion", substitute_completion
    )

    def schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        finalizer("model", torch.tensor(1.0))

    with pytest.raises(MdpStateError, match="completion substituted") as caught:
        api._run_d4_native_schedule(
            schedule,
            owner,
            _aborter(aborts),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert aborts[-1] == (owner, caught.value)
    assert state.completion is not None


def test_native_finalizer_failure_never_captures_or_calls_legacy(monkeypatch):
    owner, state = _owner(monkeypatch, records=("record",))
    aborts = []
    legacy = []
    primary = RuntimeError("native finalizer failed before capture")

    def fail(_model, _token):
        raise primary

    finalizer = api._wrap_d4_native_finalizer(fail, legacy.append)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        finalizer("model", torch.tensor(1.0))

    with pytest.raises(RuntimeError, match="before capture") as caught:
        api._run_d4_native_schedule(
            schedule,
            owner,
            _aborter(aborts),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert caught.value is primary
    assert state.token is None
    assert legacy == []
    assert aborts == [(owner, primary)]


def test_preflight_rejects_vpp_eval_and_double_finalizer_wrap(monkeypatch):
    owner, state = _owner(monkeypatch)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        raise AssertionError("preflight must reject before schedule")

    for kwargs, message in (
        (dict(data_iterator=[iter(()), iter(())], num_microbatches=1, forward_only=False), "VPP1"),
        (dict(data_iterator=iter(()), num_microbatches=1, forward_only=True), "training-only"),
    ):
        with pytest.raises(MdpConfigurationError, match=message):
            api._run_d4_native_schedule(schedule, owner, _aborter([]), **kwargs)
    assert state.events == []

    wrapped = api._wrap_d4_native_finalizer(lambda _model, _token: "native")
    with pytest.raises(MdpConfigurationError, match="wrapped exactly once"):
        api._wrap_d4_native_finalizer(wrapped)


def test_finalizer_fallback_preserves_legacy_result_identity_and_error():
    result = object()
    token = torch.tensor(1.0)
    legacy = []
    calls = []

    def native(model, num_tokens, *args, **kwargs):
        calls.append((model, num_tokens, args, kwargs))
        num_tokens.add_(2)
        return result

    wrapped = api._wrap_d4_native_finalizer(native, legacy.append)
    assert wrapped("model", token, "extra", flag=True) is result
    assert calls == [("model", token, ("extra",), {"flag": True})]
    assert legacy == [token]
    assert float(token) == 3.0

    primary = RuntimeError("legacy finalizer failed")

    def fail(*_args, **_kwargs):
        raise primary

    with pytest.raises(RuntimeError, match="legacy finalizer failed") as caught:
        api._wrap_d4_native_finalizer(fail, legacy.append)("model", token)
    assert caught.value is primary
    assert legacy == [token]


def test_real_replay_owner_success_abort_and_same_runtime_fresh_retry(monkeypatch):
    owner, runtime, released = _real_owner(monkeypatch)
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        assert next(data_iterator) == "record"
        finalizer("model", torch.tensor(4.0))
        return "native"

    success = api._run_d4_native_schedule(
        schedule,
        owner,
        lambda actual, error: actual.abort(error),
        data_iterator=iter(("raw",)),
        num_microbatches=9,
        forward_only=False,
    )
    cursor = replay_api._ACTIVE_OWNERS[id(owner)][-1].cursor
    assert replay_api._ACTIVE_CURSORS[id(cursor)][3] == 1
    assert replay_api._ACTIVE_COMPLETIONS[id(success.completion)][0]() is owner
    assert owner.require_completion(success.completion) is success.completion
    owner.abort()
    assert replay_api._ACTIVE_OWNERS.get(id(owner)) is None
    assert replay_api._ACTIVE_CURSORS.get(id(cursor)) is None
    assert replay_api._ACTIVE_COMPLETIONS.get(id(success.completion)) is None
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(runtime)) is None
    assert released == []

    failed, _, _ = _real_owner(monkeypatch, runtime=runtime)

    def missing(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        return "missing token"

    with pytest.raises(MdpStateError, match="exact post-finalizer token"):
        api._run_d4_native_schedule(
            missing,
            failed,
            lambda actual, error: actual.abort(error),
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert forward_api._ACTIVE_RUNTIME_OWNERS.get(id(runtime)) is None

    fresh, _, _ = _real_owner(monkeypatch, runtime=runtime)
    finalizer = api._wrap_d4_native_finalizer(lambda _model, _token: None)

    def retry(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        finalizer("model", torch.tensor(5.0))
        return "fresh"

    retried = api._run_d4_native_schedule(
        retry,
        fresh,
        lambda actual, error: actual.abort(error),
        data_iterator=iter(("raw",)),
        num_microbatches=1,
        forward_only=False,
    )
    assert retried.native_result == "fresh"
    fresh.abort()


@pytest.mark.parametrize(
    "mutation", ("delete_active", "substitute_active", "delete_registry", "substitute_registry")
)
def test_native_finalizer_callback_cannot_reroute_or_destroy_owned_sink(monkeypatch, mutation):
    owner, _state = _owner(monkeypatch, records=("record",))
    aborts = []
    foreign = object()
    mutated_sinks = []

    def native(_model, _token):
        sink = api._ACTIVE_TOKEN_SINK
        mutated_sinks.append(sink)
        if mutation == "delete_active":
            api._ACTIVE_TOKEN_SINK = None
        elif mutation == "substitute_active":
            api._ACTIVE_TOKEN_SINK = foreign
        elif mutation == "delete_registry":
            del api._TOKEN_SINKS[id(sink)]
        else:
            api._TOKEN_SINKS[id(sink)] = (foreign,)

    finalizer = api._wrap_d4_native_finalizer(native)

    def schedule(*, data_iterator, num_microbatches, forward_only):
        next(data_iterator)
        finalizer("model", torch.tensor(1.0))

    def abort(actual, error):
        aborts.append((actual, error))

    with pytest.raises(MdpStateError) as caught:
        api._run_d4_native_schedule(
            schedule,
            owner,
            abort,
            data_iterator=iter(("raw",)),
            num_microbatches=1,
            forward_only=False,
        )
    assert aborts == [(owner, caught.value)]
    assert owner is not api._ACTIVE_TOKEN_SINK
    assert mutated_sinks[0]._state is api._RETIRED
    assert mutated_sinks[0].owner is None
    assert mutated_sinks[0].cursor is None
    if mutation == "substitute_active":
        assert api._ACTIVE_TOKEN_SINK is foreign
    if mutation == "substitute_registry":
        assert api._TOKEN_SINKS[id(mutated_sinks[0])] == (foreign,)
    api._ACTIVE_TOKEN_SINK = None
    api._TOKEN_SINKS.clear()


def test_absent_d4_route_stays_legacy_when_native_installs_foreign_sink():
    legacy = []
    foreign = object()

    def native(_model, _token):
        api._ACTIVE_TOKEN_SINK = foreign
        return "native"

    wrapped = api._wrap_d4_native_finalizer(native, legacy.append)
    token = torch.tensor(1.0)
    assert wrapped("model", token) == "native"
    assert legacy == [token]
    assert api._ACTIVE_TOKEN_SINK is foreign
    api._ACTIVE_TOKEN_SINK = None
