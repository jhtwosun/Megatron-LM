# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 gate-2 decoder-ready composition."""

from collections.abc import Callable
from typing import Any

from megatron.core.mdp.dynamic_cp_bridge_transport import PreparedDynamicBridgeExchange
from megatron.core.mdp.dynamic_cp_d3_ready_handoff import _compose_d3_decoder_ready_handoff
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderReadyIteration,
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
)
from megatron.core.mdp.dynamic_cp_transport import PreparedDecoderPayloadBundle

__all__ = ()


def run_repeated_d4_decoder_ready(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    producer: _DynamicProducerCarrier,
    payload_bundle: PreparedDecoderPayloadBundle,
    embedding_exchange: PreparedDynamicBridgeExchange,
    cp_partition_mode: str,
    decoder_group_getter: Callable[..., Any],
    decoder_group_ranks_getter: Callable[..., Any],
    rebuild_microbatch: Callable[..., Any],
    byte_generator: Callable[[int], Any] | None = None,
) -> DecoderReadyIteration:
    """Compose one decoder-ready carrier behind repeated-D4 gate 2."""

    def prepare() -> DecoderReadyIteration:
        return _compose_d3_decoder_ready_handoff(
            workspace_owner=workspace_owner,
            authority=authority,
            producer=producer,
            payload_bundle=payload_bundle,
            payload_result=payload_bundle.received_tensors,
            embedding_exchange=embedding_exchange,
            embedding_result=embedding_exchange.received_tensors,
            cp_partition_mode=cp_partition_mode,
            decoder_group_getter=decoder_group_getter,
            decoder_group_ranks_getter=decoder_group_ranks_getter,
            rebuild_microbatch=rebuild_microbatch,
        )

    return run_repeated_d4_authority_collective(
        binding,
        authority,
        gate_id=2,
        prepare=prepare,
        domain_collective=lambda ready: ready,
        byte_generator=byte_generator,
    )
