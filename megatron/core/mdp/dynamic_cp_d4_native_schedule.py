# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private native-schedule boundary for fixed-CP4 repeated-D4 replay."""

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import signature
from typing import Any

from torch import Tensor

from megatron.core.mdp.dynamic_cp_d4_fixed_decoder_replay import (
    _D4FixedDecoderCompletion,
    _D4FixedDecoderReplayCursor,
    _D4FixedDecoderReplayOwner,
    _tensor_descriptor,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

__all__ = ()

_FINALIZER_MARKER = "_mdp_d4_native_finalizer_wrapped"
_SUCCESS_SEAL = object()
_ACTIVE = object()
_RETIRED = object()
_ACTIVE_TOKEN_SINK: "_D4NativeTokenSink | None" = None
_ACTIVE_BOUNDARY: object | None = None
_BOUNDARIES: dict[int, object] = {}
_TOKEN_SINKS: dict[int, tuple[Any, ...]] = {}


def _add_cleanup_note(primary: BaseException, note: str) -> None:
    try:
        primary.add_note(note)
    except BaseException:
        pass


@dataclass(slots=True)
class _TokenSinkEscrow:
    token: Tensor | None = None
    descriptor: tuple[Any, ...] | None = None


class _D4NativeTokenSink:
    """One registered sink visible to a stable preinstalled finalizer wrapper."""

    __slots__ = ("__weakref__", "owner", "cursor", "token", "_state")

    def __init__(
        self, owner: _D4FixedDecoderReplayOwner, cursor: _D4FixedDecoderReplayCursor
    ) -> None:
        self.owner = owner
        self.cursor = cursor
        self.token = None
        self._state = _ACTIVE

    def _entry(self) -> tuple[Any, ...]:
        entry = _TOKEN_SINKS.get(id(self))
        if (
            type(entry) is not tuple
            or len(entry) != 4
            or type(entry[0]) is not weakref.ReferenceType
            or entry[0]() is not self
            or type(entry[3]) is not _TokenSinkEscrow
            or _ACTIVE_TOKEN_SINK is not self
            or self._state is not _ACTIVE
            or self.owner is not entry[1]
            or self.cursor is not entry[2]
            or self.token is not entry[3].token
        ):
            raise MdpStateError("MDP: D4 native schedule token sink retains its exact state.")
        return entry

    def capture(self, token: Tensor) -> None:
        entry = self._entry()
        escrow = entry[3]
        if escrow.token is not None:
            raise MdpStateError("MDP: D4 native schedule captures finalizer tokens exactly once.")
        self.owner.capture_global_num_tokens(token)
        descriptor = _tensor_descriptor(token)
        self.token = token
        escrow.token = token
        escrow.descriptor = descriptor

    def require_captured(self) -> Tensor:
        entry = self._entry()
        token, descriptor = entry[3].token, entry[3].descriptor
        if token is None or _tensor_descriptor(token) != descriptor:
            raise MdpStateError("MDP: D4 native schedule retains the exact post-finalizer token.")
        return token


@dataclass(frozen=True, slots=True)
class _D4NativeScheduleSuccess:
    """Native result plus the existing replay completion authority."""

    native_result: Any = field(compare=False)
    completion: _D4FixedDecoderCompletion = field(compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self) is not _D4NativeScheduleSuccess
            or self._seal is not _SUCCESS_SEAL
            or type(self.completion) is not _D4FixedDecoderCompletion
        ):
            raise MdpConfigurationError("MDP: D4 native schedule success is privately minted.")


def _activate_token_sink(
    owner: _D4FixedDecoderReplayOwner, cursor: _D4FixedDecoderReplayCursor
) -> _D4NativeTokenSink:
    global _ACTIVE_TOKEN_SINK
    if _ACTIVE_TOKEN_SINK is not None:
        raise MdpStateError("MDP: D4 native schedule permits one active token sink.")
    if (
        type(owner) is not _D4FixedDecoderReplayOwner
        or type(cursor) is not _D4FixedDecoderReplayCursor
    ):
        raise MdpStateError("MDP: D4 native schedule activates its exact replay cursor.")
    sink = _D4NativeTokenSink(owner, cursor)
    identity = id(sink)

    def retire(reference: weakref.ReferenceType[Any]) -> None:
        entry = _TOKEN_SINKS.get(identity)
        if entry is not None and entry[0] is reference:
            del _TOKEN_SINKS[identity]

    reference = weakref.ref(sink, retire)
    _TOKEN_SINKS[identity] = (reference, owner, cursor, _TokenSinkEscrow())
    _ACTIVE_TOKEN_SINK = sink
    return sink


def _retire_token_sink(
    sink: _D4NativeTokenSink, trusted_entry: tuple[Any, ...], primary: BaseException | None = None
) -> None:
    global _ACTIVE_TOKEN_SINK
    entry = _TOKEN_SINKS.get(id(sink))
    integrity_error = None
    if (
        entry is not trusted_entry
        or type(entry) is not tuple
        or len(entry) != 4
        or type(entry[0]) is not weakref.ReferenceType
        or entry[0]() is not sink
    ):
        integrity_error = MdpStateError("MDP: D4 native schedule token sink is registered.")
    else:
        try:
            sink._entry()
        except BaseException as error:
            integrity_error = error
    if _TOKEN_SINKS.get(id(sink)) is trusted_entry:
        del _TOKEN_SINKS[id(sink)]
    if _ACTIVE_TOKEN_SINK is sink:
        _ACTIVE_TOKEN_SINK = None
    sink._state = _RETIRED
    sink.owner = None
    sink.cursor = None
    sink.token = None
    if primary is not None and integrity_error is not None:
        _add_cleanup_note(primary, f"suppressed D4 token-sink integrity error: {integrity_error!r}")


def _wrap_d4_native_finalizer(
    native_finalizer: Callable, legacy_capture: Callable[[Tensor | None], Any] | None = None
) -> Callable:
    """Return a stable finalizer wrapper; installation is owned by later integration."""
    if not callable(native_finalizer):
        raise MdpConfigurationError("MDP: D4 native finalizer is callable.")
    if legacy_capture is not None and not callable(legacy_capture):
        raise MdpConfigurationError("MDP: D4 legacy token capture is callable or None.")
    if getattr(native_finalizer, _FINALIZER_MARKER, False):
        raise MdpConfigurationError("MDP: D4 native finalizer is wrapped exactly once.")

    def wrapped(model, num_tokens=None, *args, **kwargs):
        sink = _ACTIVE_TOKEN_SINK
        result = native_finalizer(model, num_tokens, *args, **kwargs)
        if sink is not None:
            if type(sink) is not _D4NativeTokenSink:
                raise MdpStateError("MDP: D4 native finalizer uses the exact active token sink.")
            sink._entry()
            sink.capture(num_tokens)
        elif legacy_capture is not None:
            legacy_capture(num_tokens)
        return result

    setattr(wrapped, _FINALIZER_MARKER, True)
    wrapped._mdp_native = native_finalizer
    wrapped._mdp_legacy_capture = legacy_capture
    return wrapped


def _run_d4_native_schedule(
    forward_backward_func: Callable,
    replay_owner: _D4FixedDecoderReplayOwner,
    scheduled_abort: Callable[[_D4FixedDecoderReplayOwner, BaseException], Any],
    /,
    *args,
    **kwargs,
) -> _D4NativeScheduleSuccess:
    """Run the unchanged VPP1 native schedule over one exact replay cursor."""
    if not callable(forward_backward_func) or not callable(scheduled_abort):
        raise MdpConfigurationError("MDP: D4 native schedule and scheduled abort are callable.")
    if type(replay_owner) is not _D4FixedDecoderReplayOwner:
        raise MdpConfigurationError("MDP: D4 native schedule requires its exact replay owner.")
    schedule_signature = signature(forward_backward_func)
    bound = schedule_signature.bind(*args, **kwargs)
    bound.apply_defaults()
    data_iterator = bound.arguments["data_iterator"]
    if isinstance(data_iterator, (list, tuple)) and len(data_iterator) != 1:
        raise MdpConfigurationError("MDP: D4 native schedule supports VPP1 exactly.")
    if bound.arguments["forward_only"] is not False:
        raise MdpConfigurationError("MDP: D4 native schedule is training-only.")

    global _ACTIVE_BOUNDARY
    replay_owner.require()
    if _ACTIVE_BOUNDARY is not None or _BOUNDARIES:
        raise MdpStateError("MDP: D4 native schedule permits one active invocation.")
    if _ACTIVE_TOKEN_SINK is not None:
        raise MdpStateError("MDP: D4 native schedule permits one active token sink.")
    invocation = object()
    _ACTIVE_BOUNDARY = invocation
    _BOUNDARIES[id(invocation)] = invocation
    sink = None
    sink_entry = None
    try:
        cursor = replay_owner.replay_cursor()
        bound.arguments["data_iterator"] = (
            [cursor] if isinstance(data_iterator, (list, tuple)) else cursor
        )
        bound.arguments["num_microbatches"] = len(replay_owner.records)
        sink = _activate_token_sink(replay_owner, cursor)
        sink_entry = _TOKEN_SINKS[id(sink)]
        native_result = forward_backward_func(*bound.args, **bound.kwargs)
        sink.require_captured()
        _retire_token_sink(sink, sink_entry)
        sink = None
        schedule_return = replay_owner.mark_schedule_returned(cursor)
        completion = replay_owner.prepare_completion(cursor, schedule_return)
        replay_owner.require_completion(completion)
        return _D4NativeScheduleSuccess(native_result, completion, _SUCCESS_SEAL)
    except BaseException as error:
        if sink is not None:
            assert sink_entry is not None
            _retire_token_sink(sink, sink_entry, error)
        try:
            scheduled_abort(replay_owner, error)
        except BaseException as abort_error:
            if abort_error is not error:
                _add_cleanup_note(error, f"suppressed D4 scheduled-abort error: {abort_error!r}")
        raise
    finally:
        if _BOUNDARIES.get(id(invocation)) is invocation:
            del _BOUNDARIES[id(invocation)]
        if _ACTIVE_BOUNDARY is invocation:
            _ACTIVE_BOUNDARY = None
