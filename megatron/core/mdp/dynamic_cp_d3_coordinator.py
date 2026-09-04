# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private one-iteration D3 coordination over existing Dynamic-CP operations."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import PreparedDynamicBridgeExchange
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderReadyIteration,
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
    _PreAuthorityDynamicProducer,
)
from megatron.core.mdp.dynamic_cp_transport import PreparedDecoderPayloadBundle
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

_D3Operation = Callable[..., Any]
_D3AuthorityStatusGate = Callable[[BaseException | None], None]


_PENDING_GATE_STATUS_CONTEXT_SEALS: dict[object, tuple[int, ...]] = {}


@dataclass(frozen=True, slots=True)
class _D3GateStatusContext:
    """One exact local snapshot consumed by a future physical status binding."""

    gate_id: int
    authority: _DynamicIterationAuthority
    phase_value: Any = field(compare=False, repr=False)
    ready: DecoderReadyIteration | None = field(compare=False, repr=False)
    _factory_seal: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D3GateStatusContext:
            raise MdpStateError("MDP: D3 gate status context is minted by its factory.")
        if type(self.gate_id) is not int or not 0 <= self.gate_id <= 6:
            raise MdpConfigurationError("MDP: D3 gate status context has a supported gate ID.")
        if type(self.authority) is not _DynamicIterationAuthority:
            raise MdpConfigurationError("MDP: D3 gate status context retains exact authority.")
        if self.gate_id <= 2:
            if self.ready is not None:
                raise MdpConfigurationError(
                    "MDP: D3 gate 0--2 status context has no predecessor ready handoff."
                )
            if (
                self.gate_id == 2
                and self.phase_value is not None
                and type(self.phase_value) is not DecoderReadyIteration
            ):
                raise MdpConfigurationError(
                    "MDP: D3 gate 2 status context carries typed ready preparation."
                )
        elif type(self.ready) is not DecoderReadyIteration:
            raise MdpConfigurationError(
                "MDP: D3 gate 3--6 status context retains exact active ready handoff."
            )
        fingerprint = _PENDING_GATE_STATUS_CONTEXT_SEALS.pop(self._factory_seal, None)
        if fingerprint != _gate_status_context_fingerprint(self):
            raise MdpStateError("MDP: D3 gate status context is minted by its factory.")


def _gate_status_context_fingerprint(context: _D3GateStatusContext) -> tuple[int, ...]:
    return tuple(
        id(value)
        for value in (context.gate_id, context.authority, context.phase_value, context.ready)
    )


def _make_d3_gate_status_context(
    *,
    gate_id: int,
    authority: _DynamicIterationAuthority,
    phase_value: Any,
    ready: DecoderReadyIteration | None,
) -> _D3GateStatusContext:
    """Mint one one-shot status snapshot without deriving a physical wire."""
    token = object()
    kwargs = dict(
        gate_id=gate_id,
        authority=authority,
        phase_value=phase_value,
        ready=ready,
        _factory_seal=token,
    )
    _PENDING_GATE_STATUS_CONTEXT_SEALS[token] = tuple(
        id(value) for name, value in kwargs.items() if name != "_factory_seal"
    )
    try:
        return _D3GateStatusContext(**kwargs)
    except BaseException:
        _PENDING_GATE_STATUS_CONTEXT_SEALS.pop(token, None)
        raise


_D3StatusGate = Callable[[_D3GateStatusContext, BaseException | None], None]


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
    execute_iteration_commit: _D3Operation
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
    payload_prepared: PreparedDecoderPayloadBundle | None = None
    payload_result: Any = None
    embedding_prepared: PreparedDynamicBridgeExchange | None = None
    embedding_result: Any = None
    ready: DecoderReadyIteration | None = None
    commit_ready: Any = None
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

    @staticmethod
    def _make_gate_status_context(
        state: _D3ActiveIteration, *, gate_id: int, phase_value: Any
    ) -> _D3GateStatusContext:
        authority = state.authority
        if type(authority) is not _DynamicIterationAuthority:
            raise MdpStateError("MDP: D3 status context requires exact iteration authority.")
        if gate_id <= 2:
            if state.ready is not None:
                raise MdpStateError("MDP: D3 gate 0--2 status precedes ready publication.")
            ready = None
        else:
            ready = state.ready
            if type(ready) is not DecoderReadyIteration:
                raise MdpStateError("MDP: D3 gate 3--6 status requires exact active ready handoff.")
        context = _make_d3_gate_status_context(
            gate_id=gate_id, authority=authority, phase_value=phase_value, ready=ready
        )
        if (
            type(context) is not _D3GateStatusContext
            or context.gate_id != gate_id
            or context.authority is not authority
            or context.phase_value is not phase_value
            or context.ready is not ready
        ):
            raise MdpStateError("MDP: D3 status context preserves exact active identities.")
        return context

    def _complete_local_gate(
        self,
        state: _D3ActiveIteration,
        *,
        gate_id: int,
        phase_value: Any,
        local_error: BaseException | None,
    ) -> None:
        context = self._make_gate_status_context(state, gate_id=gate_id, phase_value=phase_value)
        self._bindings.status_gate(context, local_error)
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
            self._complete_local_gate(
                state, gate_id=gate_id, phase_value=value, local_error=local_error
            )
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
            prepared = self._bindings.prepare_payload(state.authority, state.bound_producer)
            if type(prepared) is not PreparedDecoderPayloadBundle:
                raise MdpConfigurationError(
                    "MDP: D3 coordinator payload preparation returns typed transport state."
                )
            state.payload_prepared = prepared
            return prepared

        payload_prepared = self._run_precollective(
            state, gate_id=0, operation=bind_and_prepare_payload
        )
        payload_result = self._run_entered_collective(
            state, lambda: self._bindings.execute_payload(payload_prepared)
        )
        if payload_result is not payload_prepared.received_tensors:
            self._fail(
                state,
                MdpStateError(
                    "MDP: D3 coordinator payload execute returns its exact received mapping."
                ),
            )
        state.payload_result = payload_result
        assert state.bound_producer is not None

        def prepare_embedding():
            prepared = self._bindings.prepare_embedding(
                state.authority, state.bound_producer, payload_result
            )
            if type(prepared) is not PreparedDynamicBridgeExchange:
                raise MdpConfigurationError(
                    "MDP: D3 coordinator embedding preparation returns typed transport state."
                )
            if prepared.phase is not BridgePhase.EMBEDDING:
                raise MdpConfigurationError(
                    "MDP: D3 coordinator embedding preparation returns embedding phase state."
                )
            state.embedding_prepared = prepared
            return prepared

        embedding_prepared = self._run_precollective(state, gate_id=1, operation=prepare_embedding)
        embedding_result = self._run_entered_collective(
            state, lambda: self._bindings.execute_embedding(embedding_prepared)
        )
        if embedding_result is not embedding_prepared.received_tensors:
            self._fail(
                state,
                MdpStateError(
                    "MDP: D3 coordinator embedding execute returns its exact received mapping."
                ),
            )
        state.embedding_result = embedding_result

        def prepare_schedule():
            assert state.payload_prepared is not None and state.payload_result is not None
            assert state.embedding_prepared is not None and state.embedding_result is not None
            ready = self._bindings.prepare_schedule(
                state.authority,
                state.bound_producer,
                state.payload_prepared,
                state.payload_result,
                state.embedding_prepared,
                state.embedding_result,
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
            context = self._make_gate_status_context(state, gate_id=3, phase_value=None)
            self._bindings.status_gate(context, primary_error)
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
        commit_ready = self._run_entered_collective(
            state, lambda: self._bindings.execute_encoder_finalize(finalizer)
        )
        state.commit_ready = commit_ready
        cleanup_error = self._cleanup(state)
        try:
            self._complete_local_gate(state, gate_id=6, phase_value=None, local_error=cleanup_error)
        except BaseException as error:
            self._fail(state, error)
        self._run_entered_collective(
            state, lambda: self._bindings.execute_iteration_commit(state.commit_ready)
        )
        state.commit_ready = None
        self._active = None
