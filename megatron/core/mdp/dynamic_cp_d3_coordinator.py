# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private one-iteration D3 coordination over existing Dynamic-CP operations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderReadyIteration,
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
    _PreAuthorityDynamicProducer,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

_D3Operation = Callable[..., Any]
_D3AuthorityStatusGate = Callable[[BaseException | None], None]
_D3StatusGate = Callable[[int, BaseException | None, Any], None]


@dataclass(frozen=True)
class _D3CoordinatorBindings:
    """Exact injected D3 operations; this is not a public runtime facade."""

    execution_config_consensus: _D3Operation
    gather_metadata: _D3Operation
    build_authority: _D3Operation
    authority_status_gate: _D3AuthorityStatusGate
    bind_producer: _D3Operation
    prepare_payload: _D3Operation
    execute_payload: _D3Operation
    prepare_embedding: _D3Operation
    execute_embedding: _D3Operation
    prepare_schedule: _D3Operation
    prepare_gradient: _D3Operation
    execute_gradient: _D3Operation
    prepare_encoder_completion: _D3Operation
    execute_encoder_backward: _D3Operation
    execute_encoder_finalize: _D3Operation
    cleanup: _D3Operation
    status_gate: _D3StatusGate

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not callable(getattr(self, name)):
                raise MdpConfigurationError(f"MDP: D3 coordinator binding {name} must be callable.")


@dataclass
class _D3ActiveIteration:
    producer: _PreAuthorityDynamicProducer
    authority: _DynamicIterationAuthority | None = None
    bound_producer: _DynamicProducerCarrier | None = None
    ready: DecoderReadyIteration | None = None
    decoder_complete: bool = False
    cleaned: bool = False
    scheduled_abort_started: bool = False


def _runtime_for_registered_producer(producer: _PreAuthorityDynamicProducer) -> Any:
    owner = producer.owner
    runtime = getattr(owner, "_runtime", None)
    validate = getattr(runtime, "_validate_pre_authority_dynamic_producer", None)
    if not callable(validate):
        raise MdpStateError("MDP: D3 coordinator requires registered producer runtime ownership.")
    validate(owner, producer)
    return runtime


def _validate_consumed_pre_authority_producer(
    producer: _PreAuthorityDynamicProducer, bound: _DynamicProducerCarrier, runtime: Any
) -> None:
    if bound.pre_authority is not producer or bound.owner is not producer.owner:
        raise MdpStateError("MDP: D3 coordinator bound producer preserves exact ownership.")
    if getattr(producer, "_mdp_pre_authority_runtime", None) is not runtime:
        raise MdpStateError("MDP: D3 coordinator producer keeps its exact runtime marker.")
    if getattr(runtime, "_pre_authority_dynamic_producer", None) is not None:
        raise MdpStateError("MDP: D3 coordinator binder consumes the registered producer.")
    retired = getattr(runtime, "_pre_authority_dynamic_producer_is_retired", None)
    if not callable(retired) or not retired(producer):
        raise MdpStateError("MDP: D3 coordinator binder retires the registered producer.")


class _D3Coordinator:
    """Serialize exactly one private D3 lifecycle without retrying collectives."""

    def __init__(self, *, bindings: _D3CoordinatorBindings) -> None:
        if type(bindings) is not _D3CoordinatorBindings:
            raise MdpConfigurationError("MDP: D3 coordinator requires typed private bindings.")
        self._bindings = bindings
        self._active: _D3ActiveIteration | None = None

    @property
    def is_idle(self) -> bool:
        return self._active is None

    def _cleanup(self, state: _D3ActiveIteration) -> BaseException | None:
        if state.cleaned:
            return None
        state.cleaned = True
        try:
            self._bindings.cleanup(
                state.bound_producer if state.bound_producer is not None else state.producer
            )
        except BaseException as error:
            return error
        return None

    def _fail(self, state: _D3ActiveIteration, error: BaseException) -> None:
        cleanup_error = self._cleanup(state)
        self._active = None
        if cleanup_error is not None:
            try:
                error.add_note(f"suppressed D3 coordinator cleanup error: {cleanup_error!r}")
            except Exception:
                pass
        raise error

    def _complete_local_gate(
        self, state: _D3ActiveIteration, *, gate_id: int, local_error: BaseException | None
    ) -> None:
        self._bindings.status_gate(gate_id, local_error, state.authority)
        if local_error is not None:
            error = MdpStateError(
                f"MDP: D3 coordinator gate {gate_id} accepted a local preparation error."
            )
            raise error from local_error

    def _run_precollective(
        self, state: _D3ActiveIteration, *, gate_id: int, operation: Callable[[], Any]
    ) -> Any:
        try:
            value = operation()
            local_error = None
        except BaseException as error:
            value = None
            local_error = error
        try:
            self._complete_local_gate(state, gate_id=gate_id, local_error=local_error)
        except BaseException as error:
            self._fail(state, error)
        return value

    def _run_authority_phase(
        self, state: _D3ActiveIteration, *, metadata: Any, config: Any
    ) -> None:
        try:
            authority = self._bindings.build_authority(metadata, state.producer, config)
            if type(authority) is not _DynamicIterationAuthority:
                raise MdpConfigurationError(
                    "MDP: D3 coordinator authority builder returns typed iteration authority."
                )
            state.authority = authority
            local_error = None
        except BaseException as error:
            local_error = error
        try:
            self._bindings.authority_status_gate(local_error)
        except BaseException as error:
            self._fail(state, error)
        if local_error is not None:
            error = MdpStateError(
                "MDP: D3 coordinator authority status gate accepted a local error."
            )
            raise error from local_error

    def _run_entered_collective(
        self, state: _D3ActiveIteration, operation: Callable[[], Any]
    ) -> Any:
        try:
            return operation()
        except BaseException as error:
            self._fail(state, error)

    def begin_iteration(
        self, *, config: Any, producer: _PreAuthorityDynamicProducer
    ) -> DecoderReadyIteration:
        """Run configuration through gate 2 and return one exact decoder-ready handoff."""
        if self._active is not None:
            raise MdpStateError("MDP: D3 coordinator starts only while idle.")
        if type(producer) is not _PreAuthorityDynamicProducer:
            raise MdpConfigurationError("MDP: D3 coordinator starts from typed producer state.")

        state = _D3ActiveIteration(producer=producer)
        self._active = state
        self._run_entered_collective(
            state, lambda: self._bindings.execution_config_consensus(config)
        )
        metadata = self._run_entered_collective(
            state, lambda: self._bindings.gather_metadata(producer, config)
        )
        try:
            self._run_authority_phase(state, metadata=metadata, config=config)
        except BaseException as error:
            self._fail(state, error)
        assert state.authority is not None

        def bind_and_prepare_payload():
            runtime = _runtime_for_registered_producer(producer)
            bound = self._bindings.bind_producer(state.authority, producer)
            if type(bound) is not _DynamicProducerCarrier:
                raise MdpConfigurationError(
                    "MDP: D3 coordinator producer binder returns typed producer carrier."
                )
            state.bound_producer = bound
            _validate_consumed_pre_authority_producer(producer, bound, runtime)
            return self._bindings.prepare_payload(state.authority, state.bound_producer)

        payload_prepared = self._run_precollective(
            state, gate_id=0, operation=bind_and_prepare_payload
        )
        payload_result = self._run_entered_collective(
            state, lambda: self._bindings.execute_payload(payload_prepared)
        )
        assert state.bound_producer is not None
        embedding_prepared = self._run_precollective(
            state,
            gate_id=1,
            operation=lambda: self._bindings.prepare_embedding(
                state.authority, state.bound_producer, payload_result
            ),
        )
        embedding_result = self._run_entered_collective(
            state, lambda: self._bindings.execute_embedding(embedding_prepared)
        )

        def prepare_schedule():
            ready = self._bindings.prepare_schedule(
                state.authority, state.bound_producer, embedding_result
            )
            if type(ready) is not DecoderReadyIteration:
                raise MdpConfigurationError(
                    "MDP: D3 coordinator schedule preparation returns typed decoder-ready state."
                )
            return ready

        ready = self._run_precollective(state, gate_id=2, operation=prepare_schedule)
        state.ready = ready
        return ready

    def mark_decoder_complete(self, ready: DecoderReadyIteration) -> None:
        """Record exactly one native-schedule completion for the active handoff."""
        state = self._active
        if state is None or state.ready is not ready:
            raise MdpStateError("MDP: D3 coordinator requires its exact decoder-ready handoff.")
        if state.decoder_complete:
            raise MdpStateError("MDP: D3 coordinator records decoder completion exactly once.")
        state.decoder_complete = True

    def abort_scheduled_iteration(
        self, ready: DecoderReadyIteration, primary_error: BaseException
    ) -> None:
        """Retire one failed native-schedule handoff without entering later D3 phases."""
        if not isinstance(primary_error, BaseException):
            raise MdpConfigurationError(
                "MDP: D3 scheduled abort requires a BaseException primary error."
            )
        state = self._active
        if (
            state is None
            or state.ready is not ready
            or state.decoder_complete
            or state.scheduled_abort_started
        ):
            raise MdpStateError(
                "MDP: D3 scheduled abort requires its exact active decoder-ready handoff."
            )
        state.scheduled_abort_started = True

        try:
            self._bindings.status_gate(3, primary_error, state.authority)
        except BaseException as error:
            self._add_secondary_note(primary_error, "scheduled abort gate", error)
        cleanup_error = self._cleanup(state)
        self._active = None
        if cleanup_error is not None:
            self._add_secondary_note(primary_error, "scheduled abort cleanup", cleanup_error)
        raise primary_error

    @staticmethod
    def _add_secondary_note(
        primary_error: BaseException, description: str, secondary_error: BaseException
    ) -> None:
        try:
            primary_error.add_note(f"suppressed D3 {description} error: {secondary_error!r}")
        except BaseException:
            pass

    def end_iteration(self) -> None:
        """Run gates 3--6 and retire the private producer exactly once."""
        state = self._active
        if state is None or state.ready is None or not state.decoder_complete:
            raise MdpStateError("MDP: D3 coordinator end requires decoder completion.")
        assert state.authority is not None and state.bound_producer is not None

        gradient_prepared = self._run_precollective(
            state,
            gate_id=3,
            operation=lambda: self._bindings.prepare_gradient(
                state.authority, state.bound_producer, state.ready
            ),
        )
        gradients = self._run_entered_collective(
            state, lambda: self._bindings.execute_gradient(gradient_prepared)
        )
        completion_prepared = self._run_precollective(
            state,
            gate_id=4,
            operation=lambda: self._bindings.prepare_encoder_completion(
                state.authority, state.bound_producer, gradients
            ),
        )
        finalizer = self._run_precollective(
            state,
            gate_id=5,
            operation=lambda: self._bindings.execute_encoder_backward(completion_prepared),
        )
        self._run_entered_collective(
            state, lambda: self._bindings.execute_encoder_finalize(finalizer)
        )
        cleanup_error = self._cleanup(state)
        try:
            self._complete_local_gate(state, gate_id=6, local_error=cleanup_error)
        except BaseException as error:
            self._fail(state, error)
        self._active = None
