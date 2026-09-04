# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private lifecycle facade for the locked, legacy D3 decoder handoff."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_coordinator import _D3Coordinator
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderReadyIteration,
    _DynamicExecutionConfig,
    _PreAuthorityDynamicProducer,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.window import MdpMicrobatchRecord


def _require_callable(name: str, value: Any) -> Callable[[], Any]:
    if not callable(value):
        raise MdpConfigurationError(f"MDP: D3 private facade {name} must be callable.")
    return value


class _D3Vpp1ReplayCursor(Iterator[MdpMicrobatchRecord]):
    """One VPP1 decoder replay cursor over the exact ready-record identities."""

    def __init__(self, records: tuple[MdpMicrobatchRecord, ...]) -> None:
        self._records = records
        self._next = 0

    def __iter__(self) -> "_D3Vpp1ReplayCursor":
        return self

    def __next__(self) -> MdpMicrobatchRecord:
        if self._next >= len(self._records):
            raise MdpStateError(
                f"MDP: D3 replay cursor violates: at most {len(self._records)} records "
                "per cursor per iteration (cursor overrun)."
            )
        record = self._records[self._next]
        self._next += 1
        return record


def _make_d3_vpp1_replay_cursor(
    ready: DecoderReadyIteration, *, virtual_pipeline_parallel_size: int
) -> _D3Vpp1ReplayCursor:
    if type(ready) is not DecoderReadyIteration:
        raise MdpConfigurationError("MDP: D3 replay requires typed decoder-ready state.")
    if ready.role != "decoder":
        raise MdpConfigurationError("MDP: D3 replay handoff is decoder-owned.")
    if not ready.records:
        raise MdpConfigurationError("MDP: D3 replay handoff has non-empty decoder records.")
    if type(ready.records) is not tuple or any(
        type(record) is not MdpMicrobatchRecord for record in ready.records
    ):
        raise MdpConfigurationError(
            "MDP: D3 replay handoff records are exact MdpMicrobatchRecord objects."
        )
    if type(virtual_pipeline_parallel_size) is not int or virtual_pipeline_parallel_size != 1:
        raise MdpConfigurationError("MDP: D3 replay supports the locked VPP1 topology.")
    return _D3Vpp1ReplayCursor(ready.records)


@dataclass(frozen=True)
class _D3FacadeActiveIteration:
    """The one exact handoff owned by this private facade transaction."""

    coordinator: _D3Coordinator
    ready: DecoderReadyIteration


class _D3PrivateFacade:
    """Construct and retire one typed D3 coordinator without public integration."""

    def __init__(
        self,
        *,
        config_factory: Callable[[], Any],
        producer_factory: Callable[[], Any],
        coordinator_factory: Callable[[], Any],
    ) -> None:
        self._config_factory = _require_callable("config_factory", config_factory)
        self._producer_factory = _require_callable("producer_factory", producer_factory)
        self._coordinator_factory = _require_callable("coordinator_factory", coordinator_factory)
        self._active: _D3FacadeActiveIteration | None = None

    @property
    def is_idle(self) -> bool:
        """Return whether this facade currently owns no D3 handoff."""
        return self._active is None

    def begin_iteration(self, *, forward_only: bool = False) -> DecoderReadyIteration:
        """Start one training-only D3 transaction and expose its exact handoff."""
        if type(forward_only) is not bool:
            raise MdpConfigurationError(
                "MDP: D3 private facade forward_only must be an exact bool."
            )
        if forward_only:
            raise MdpConfigurationError("MDP: D3 private facade is training-only.")
        if self._active is not None:
            raise MdpStateError("MDP: D3 private facade begins only while idle.")

        config = self._config_factory()
        if type(config) is not _DynamicExecutionConfig:
            raise MdpConfigurationError(
                "MDP: D3 private facade config_factory returns typed dynamic config."
            )
        coordinator = self._coordinator_factory()
        if type(coordinator) is not _D3Coordinator:
            raise MdpConfigurationError(
                "MDP: D3 private facade coordinator_factory returns typed D3 coordinator."
            )
        if coordinator.is_idle is not True:
            raise MdpStateError(
                "MDP: D3 private facade coordinator_factory returns an idle coordinator."
            )
        producer = self._producer_factory()
        if type(producer) is not _PreAuthorityDynamicProducer:
            raise MdpConfigurationError(
                "MDP: D3 private facade producer_factory returns typed pre-authority producer."
            )

        ready = coordinator.begin_iteration(config=config, producer=producer)
        if type(ready) is not DecoderReadyIteration:
            raise MdpConfigurationError(
                "MDP: D3 private facade coordinator returns typed decoder-ready state."
            )
        self._active = _D3FacadeActiveIteration(coordinator=coordinator, ready=ready)
        return ready

    def _require_active(self, ready: DecoderReadyIteration) -> _D3FacadeActiveIteration:
        active = self._active
        if active is None or active.ready is not ready:
            raise MdpStateError("MDP: D3 private facade requires its exact decoder-ready handoff.")
        return active

    def mark_decoder_complete(self, ready: DecoderReadyIteration) -> None:
        """Delegate one native decoder completion for the exact active handoff."""
        self._require_active(ready).coordinator.mark_decoder_complete(ready)

    def end_iteration(self, ready: DecoderReadyIteration) -> None:
        """Delegate normal completion and retire facade state even when it raises."""
        active = self._require_active(ready)
        try:
            active.coordinator.end_iteration()
        finally:
            if active.coordinator.is_idle is True:
                self._active = None

    def abort_scheduled_iteration(
        self, ready: DecoderReadyIteration, primary_error: BaseException
    ) -> None:
        """Delegate scheduled abort while preserving its exact primary exception."""
        active = self._require_active(ready)
        try:
            active.coordinator.abort_scheduled_iteration(ready, primary_error)
        finally:
            if active.coordinator.is_idle is True:
                self._active = None

    def decoder_replay_cursor(
        self, ready: DecoderReadyIteration, *, virtual_pipeline_parallel_size: int
    ) -> _D3Vpp1ReplayCursor:
        """Expose the one supported VPP1 replay cursor for the active handoff."""
        self._require_active(ready)
        return _make_d3_vpp1_replay_cursor(
            ready, virtual_pipeline_parallel_size=virtual_pipeline_parallel_size
        )
