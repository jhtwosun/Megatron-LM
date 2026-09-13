# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Concrete private composition for repeated-D4 decoder gates 0--3."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_d4_decoder_coordinator import (
    _D4DecoderCoordinator,
    _D4DecoderCoordinatorBindings,
    _make_d4_decoder_coordinator,
)
from megatron.core.mdp.dynamic_cp_d4_embedding_transport import run_repeated_d4_embedding
from megatron.core.mdp.dynamic_cp_d4_gradient_transport import run_repeated_d4_decoder_gradient
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_d4_payload_transport import run_repeated_d4_decoder_payload
from megatron.core.mdp.dynamic_cp_d4_ready_handoff import run_repeated_d4_decoder_ready
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority, _DynamicProducerCarrier
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

__all__ = ()


@dataclass(frozen=True, slots=True)
class _D4DecoderCompositionBindings:
    """One immutable set of per-iteration D4 decoder dependencies."""

    binding: _RepeatedD4GroupBinding
    authority: _DynamicIterationAuthority
    workspace_owner: _D3WorkspaceBindingOwner
    producer: _DynamicProducerCarrier
    cp_partition_mode: str
    decoder_group_getter: Callable[..., Any]
    decoder_group_ranks_getter: Callable[..., Any]
    rebuild_microbatch: Callable[..., Any]
    all_to_all_single: Callable[..., Any]
    byte_generator: Callable[[int], Any] | None
    failure_boundary: Callable[..., Any]
    cleanup: Callable[..., Any]

    def __post_init__(self) -> None:
        typed_dependencies = (
            ("group binding", self.binding, _RepeatedD4GroupBinding),
            ("iteration authority", self.authority, _DynamicIterationAuthority),
            ("workspace owner", self.workspace_owner, _D3WorkspaceBindingOwner),
            ("producer carrier", self.producer, _DynamicProducerCarrier),
        )
        for name, value, expected_type in typed_dependencies:
            if type(value) is not expected_type:
                raise MdpConfigurationError(f"MDP: D4 decoder composition requires exact {name}.")
        if self.producer.authority is not self.authority:
            raise MdpStateError(
                "MDP: D4 decoder composition producer authority is its exact iteration authority."
            )
        self.workspace_owner.require_workspace(self.authority)
        for callback in (
            self.decoder_group_getter,
            self.decoder_group_ranks_getter,
            self.rebuild_microbatch,
            self.all_to_all_single,
            self.failure_boundary,
            self.cleanup,
        ):
            if not callable(callback):
                raise MdpConfigurationError(
                    "MDP: D4 decoder composition callback dependencies are callable."
                )
        if self.byte_generator is not None and not callable(self.byte_generator):
            raise MdpConfigurationError(
                "MDP: D4 decoder composition byte generator is callable or None."
            )


def _make_d4_decoder_composition(
    *, bindings: _D4DecoderCompositionBindings
) -> _D4DecoderCoordinator:
    """Bind one iteration's concrete adapters to the existing D4 coordinator."""
    if type(bindings) is not _D4DecoderCompositionBindings:
        raise MdpConfigurationError("MDP: D4 decoder composition requires typed private bindings.")
    authority = bindings.authority

    def require_authority(actual: _DynamicIterationAuthority) -> None:
        if actual is not authority:
            raise MdpStateError(
                "MDP: D4 decoder composition requires its exact bound iteration authority."
            )

    def run_payload(actual: _DynamicIterationAuthority):
        require_authority(actual)
        workspace = bindings.workspace_owner.require_workspace(actual)
        return run_repeated_d4_decoder_payload(
            bindings.binding,
            actual,
            source_window=bindings.producer.source_window,
            buffers_by_dtype=workspace.payload_transport_buffers,
            all_to_all_single=bindings.all_to_all_single,
            byte_generator=bindings.byte_generator,
        )

    def run_embedding(actual: _DynamicIterationAuthority, _payload):
        require_authority(actual)
        workspace = bindings.workspace_owner.require_workspace(actual)
        return run_repeated_d4_embedding(
            bindings.binding,
            actual,
            item_outputs=bindings.producer.item_outputs,
            send_buffer=workspace.embedding_transport_buffers[0],
            receive_buffer=workspace.embedding_transport_buffers[1],
            all_to_all_single=bindings.all_to_all_single,
            byte_generator=bindings.byte_generator,
        )

    def run_ready(actual: _DynamicIterationAuthority, payload, embedding):
        require_authority(actual)
        return run_repeated_d4_decoder_ready(
            bindings.binding,
            actual,
            workspace_owner=bindings.workspace_owner,
            producer=bindings.producer,
            payload_bundle=payload,
            embedding_exchange=embedding,
            cp_partition_mode=bindings.cp_partition_mode,
            decoder_group_getter=bindings.decoder_group_getter,
            decoder_group_ranks_getter=bindings.decoder_group_ranks_getter,
            rebuild_microbatch=bindings.rebuild_microbatch,
            byte_generator=bindings.byte_generator,
        )

    def run_gradient(actual: _DynamicIterationAuthority, ready):
        require_authority(actual)
        return run_repeated_d4_decoder_gradient(
            bindings.binding,
            actual,
            workspace_owner=bindings.workspace_owner,
            producer=bindings.producer,
            ready=ready,
            cp_partition_mode=bindings.cp_partition_mode,
            all_to_all_single=bindings.all_to_all_single,
            byte_generator=bindings.byte_generator,
        )

    return _make_d4_decoder_coordinator(
        bindings=_D4DecoderCoordinatorBindings(
            run_payload=run_payload,
            run_embedding=run_embedding,
            run_ready=run_ready,
            run_gradient=run_gradient,
            failure_boundary=bindings.failure_boundary,
            cleanup=bindings.cleanup,
        )
    )
