# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private ordering and lifecycle for the repeated-D4 decoder prefix."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import PreparedDynamicBridgeExchange
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderGradientReceipt,
    DecoderReadyIteration,
    _DynamicIterationAuthority,
    _validate_decoder_gradient_receipt,
    validate_decoder_ready_iteration,
)
from megatron.core.mdp.dynamic_cp_transport import PreparedDecoderPayloadBundle
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

__all__ = ()


_D4Operation = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class _D4DecoderCoordinatorBindings:
    """One exact set of already-bound repeated-D4 phase callbacks."""

    run_payload: _D4Operation
    run_embedding: _D4Operation
    run_ready: _D4Operation
    run_gradient: _D4Operation
    failure_boundary: _D4Operation
    cleanup: _D4Operation

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not callable(getattr(self, name)):
                raise MdpConfigurationError(
                    f"MDP: D4 decoder coordinator binding {name} must be callable."
                )


@dataclass(slots=True)
class _D4ActiveDecoderIteration:
    authority: _DynamicIterationAuthority
    payload: PreparedDecoderPayloadBundle | None = None
    embedding: PreparedDynamicBridgeExchange | None = None
    ready: DecoderReadyIteration | None = None
    gradient_receipt: DecoderGradientReceipt | None = None
    decoder_complete: bool = False
    gradient_started: bool = False
    cleanup_started: bool = False
    scheduled_abort_started: bool = False


class _D4DecoderCoordinator:
    """Serialize gates 0--3 without entering or retiring a decoder schedule."""

    def __init__(self, *, bindings: _D4DecoderCoordinatorBindings) -> None:
        if type(bindings) is not _D4DecoderCoordinatorBindings:
            raise MdpConfigurationError(
                "MDP: D4 decoder coordinator requires typed private bindings."
            )
        self._bindings = bindings
        self._active: _D4ActiveDecoderIteration | None = None

    @property
    def is_idle(self) -> bool:
        return self._active is None

    @staticmethod
    def _add_secondary_note(
        primary_error: BaseException, description: str, secondary_error: BaseException
    ) -> None:
        try:
            primary_error.add_note(
                f"suppressed D4 decoder coordinator {description} error: {secondary_error!r}"
            )
        except BaseException:
            pass

    def _cleanup(self, state: _D4ActiveDecoderIteration) -> BaseException | None:
        if state.cleanup_started:
            return None
        state.cleanup_started = True
        try:
            self._bindings.cleanup(state.authority)
        except BaseException as error:
            return error
        return None

    def _fail(self, state: _D4ActiveDecoderIteration, error: BaseException) -> None:
        cleanup_error = self._cleanup(state)
        self._active = None
        if cleanup_error is not None:
            self._add_secondary_note(error, "cleanup", cleanup_error)
        raise error

    def _run(self, state: _D4ActiveDecoderIteration, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except BaseException as error:
            self._fail(state, error)

    @staticmethod
    def _validate_ready_lineage(
        state: _D4ActiveDecoderIteration, ready: Any
    ) -> DecoderReadyIteration:
        if type(ready) is not DecoderReadyIteration:
            raise MdpConfigurationError(
                "MDP: D4 decoder coordinator gate 2 returns an exact ready carrier."
            )
        authority = state.authority
        payload = state.payload
        embedding = state.embedding
        assert payload is not None and embedding is not None
        if (
            ready.global_manifest_digest != authority.global_manifest.digest
            or ready.decoder_plan_digest != authority.plan.digest
            or ready.payload_bundle_authority_digest != payload.bundle_authority_digest
            or ready.embedding_route_authority_digest != embedding.route_authority_digest
            or ready.participant_ranks != authority.participant_ranks
        ):
            raise MdpStateError(
                "MDP: D4 decoder coordinator ready carrier retains exact phase lineage."
            )
        validated = validate_decoder_ready_iteration(
            ready,
            global_manifest=authority.global_manifest,
            plan=authority.plan,
            payload_bundle=payload,
            payload_tensors=payload.received_tensors,
            embedding_exchange=embedding,
            embedding_tensors=embedding.received_tensors,
            expected_assignments=ready.assignments,
            authority_digest=ready.authority_digest,
            embedding_width=authority.bridge_width,
            embedding_dtype=authority.bridge_dtype,
            cp_partition_mode=ready.cp_partition_mode,
        )
        if validated is not ready:
            raise MdpStateError(
                "MDP: D4 decoder coordinator retains the exact validated ready carrier."
            )
        return ready

    @staticmethod
    def _validate_gradient_receipt(
        state: _D4ActiveDecoderIteration, receipt: Any
    ) -> DecoderGradientReceipt:
        if type(receipt) is not DecoderGradientReceipt:
            raise MdpConfigurationError(
                "MDP: D4 decoder coordinator gate 3 returns an exact gradient receipt."
            )
        ready = state.ready
        assert ready is not None
        if receipt.prepared.ready is not ready:
            raise MdpStateError(
                "MDP: D4 decoder coordinator gradient receipt retains the active ready handoff."
            )
        authority = state.authority
        validated = _validate_decoder_gradient_receipt(
            receipt,
            global_manifest=authority.global_manifest,
            plan=authority.plan,
            embedding_ledger=authority.embedding_ledger,
            gradient_ledger=authority.gradient_ledger,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            global_rank=ready.global_rank,
            participant_ranks=authority.participant_ranks,
            embedding_width=authority.bridge_width,
            embedding_dtype=authority.bridge_dtype,
            cp_partition_mode=ready.cp_partition_mode,
            iteration_nonce=receipt.iteration_nonce,
        )
        if validated is not receipt:
            raise MdpStateError(
                "MDP: D4 decoder coordinator retains the exact validated gradient receipt."
            )
        return receipt

    def begin_iteration(self, authority: _DynamicIterationAuthority) -> DecoderReadyIteration:
        """Run gates 0--2 once and publish their exact decoder-ready carrier."""
        if self._active is not None:
            raise MdpStateError("MDP: D4 decoder coordinator starts only while idle.")
        if type(authority) is not _DynamicIterationAuthority:
            raise MdpConfigurationError(
                "MDP: D4 decoder coordinator starts from exact iteration authority."
            )
        state = _D4ActiveDecoderIteration(authority=authority)
        self._active = state

        payload = self._run(state, lambda: self._bindings.run_payload(authority))
        if type(payload) is not PreparedDecoderPayloadBundle:
            self._fail(
                state,
                MdpConfigurationError(
                    "MDP: D4 decoder coordinator gate 0 returns an exact payload carrier."
                ),
            )
        state.payload = payload

        embedding = self._run(state, lambda: self._bindings.run_embedding(authority, payload))
        if type(embedding) is not PreparedDynamicBridgeExchange:
            self._fail(
                state,
                MdpConfigurationError(
                    "MDP: D4 decoder coordinator gate 1 returns an exact embedding carrier."
                ),
            )
        if embedding.phase is not BridgePhase.EMBEDDING:
            self._fail(
                state,
                MdpConfigurationError(
                    "MDP: D4 decoder coordinator gate 1 retains embedding phase."
                ),
            )
        state.embedding = embedding

        ready = self._run(state, lambda: self._bindings.run_ready(authority, payload, embedding))
        ready = self._run(state, lambda: self._validate_ready_lineage(state, ready))
        state.ready = ready
        return ready

    def mark_decoder_complete(self, ready: DecoderReadyIteration) -> None:
        """Record exactly one native decoder-schedule completion."""
        state = self._active
        if state is None or state.ready is not ready:
            raise MdpStateError(
                "MDP: D4 decoder coordinator requires its exact active ready handoff."
            )
        if state.scheduled_abort_started or state.cleanup_started:
            raise MdpStateError(
                "MDP: D4 decoder coordinator rejects progress during scheduled abort."
            )
        if state.decoder_complete:
            raise MdpStateError(
                "MDP: D4 decoder coordinator records decoder completion exactly once."
            )
        state.decoder_complete = True

    def end_decoder_phase(self, ready: DecoderReadyIteration) -> DecoderGradientReceipt:
        """Run gate 3 once while retaining active state for future gates 4--6."""
        state = self._active
        if state is None or state.ready is not ready:
            raise MdpStateError(
                "MDP: D4 decoder coordinator requires its exact active ready handoff."
            )
        if state.scheduled_abort_started or state.cleanup_started:
            raise MdpStateError(
                "MDP: D4 decoder coordinator rejects progress during scheduled abort."
            )
        if not state.decoder_complete:
            raise MdpStateError("MDP: D4 decoder coordinator end requires decoder completion.")
        if state.gradient_started:
            raise MdpStateError("MDP: D4 decoder coordinator runs decoder gradient exactly once.")
        state.gradient_started = True

        receipt = self._run(state, lambda: self._bindings.run_gradient(state.authority, ready))
        receipt = self._run(state, lambda: self._validate_gradient_receipt(state, receipt))
        state.gradient_receipt = receipt
        return receipt

    def abort_scheduled_iteration(
        self, ready: DecoderReadyIteration, primary_error: BaseException
    ) -> None:
        """Converge one native-schedule failure, then clean up and re-raise it."""
        if not isinstance(primary_error, BaseException):
            raise MdpConfigurationError(
                "MDP: D4 scheduled abort requires a BaseException primary error."
            )
        state = self._active
        if state is None or state.ready is not ready:
            raise MdpStateError("MDP: D4 scheduled abort requires its exact active ready handoff.")
        if state.decoder_complete or state.gradient_started:
            raise MdpStateError("MDP: D4 scheduled abort runs before decoder completion.")
        if state.scheduled_abort_started:
            raise MdpStateError("MDP: D4 scheduled abort runs exactly once.")
        state.scheduled_abort_started = True

        try:
            self._bindings.failure_boundary(state.authority, ready, primary_error)
        except BaseException as error:
            self._add_secondary_note(primary_error, "failure boundary", error)
        cleanup_error = self._cleanup(state)
        self._active = None
        if cleanup_error is not None:
            self._add_secondary_note(primary_error, "scheduled abort cleanup", cleanup_error)
        raise primary_error


def _make_d4_decoder_coordinator(
    *, bindings: _D4DecoderCoordinatorBindings
) -> _D4DecoderCoordinator:
    """Construct one idle private coordinator from exact callback bindings."""
    return _D4DecoderCoordinator(bindings=bindings)
