# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 gate-3 decoder-gradient transport."""

from collections.abc import Callable
from typing import Any

import torch.distributed as dist

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    _dynamic_bridge_gate_authority_digest,
    _execute_validated_dynamic_bridge_exchange,
    build_dynamic_bridge_route_authority_digest,
)
from megatron.core.mdp.dynamic_cp_d3_gradient_preparation_binding import (
    _make_d3_gradient_preparation_binding,
)
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _candidate_digest,
    _snapshot_local_authority,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderGradientReceipt,
    DecoderReadyIteration,
    _decoder_gradient_wave_authority_digest,
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
    _make_decoder_gradient_receipt,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError

__all__ = ()


def _candidate_gradient_gate_digest(
    authority: Any, ready: Any, attempt_nonce: bytes
) -> tuple[Any, Any]:
    """Read untrusted gradient authority without skipping a WORLD gate."""
    try:
        route_digest = build_dynamic_bridge_route_authority_digest(
            authority.gradient_ledger,
            authority.embedding_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            width=authority.bridge_width,
            dtype=authority.bridge_dtype,
            participant_ranks=authority.participant_ranks,
        )
        wave_digest = _decoder_gradient_wave_authority_digest(ready, attempt_nonce)
        return route_digest, _dynamic_bridge_gate_authority_digest(
            BridgePhase.GRADIENT, route_digest, wave_digest
        )
    except BaseException:
        return None, None


def run_repeated_d4_decoder_gradient(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    producer: _DynamicProducerCarrier,
    ready: DecoderReadyIteration,
    cp_partition_mode: str,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
    byte_generator: Callable[[int], Any] | None = None,
) -> DecoderGradientReceipt:
    """Prepare, exchange, and seal one reverse-gradient wave behind D4 gate 3."""
    kwargs = {}
    if byte_generator is not None:
        kwargs["byte_generator"] = byte_generator
    runner = binding.begin_attempt(**kwargs)
    route_digest, gate_digest = _candidate_gradient_gate_digest(
        authority, ready, runner.attempt_nonce
    )

    def prepare():
        _snapshot_local_authority(binding, authority)
        if not callable(all_to_all_single):
            raise MdpConfigurationError("MDP: repeated-D4 gradient all_to_all_single is callable.")
        preparation = _make_d3_gradient_preparation_binding(
            workspace_owner=workspace_owner, cp_partition_mode=cp_partition_mode
        )
        prepared = preparation(authority, producer, ready)
        if prepared.exchange.route_authority_digest != route_digest:
            raise MdpBridgeError(
                "MDP: repeated-D4 gradient preparation matches captured route authority."
            )
        return prepared

    def execute(prepared):
        received = _execute_validated_dynamic_bridge_exchange(
            prepared.exchange, group=binding.domain_group, all_to_all_single=all_to_all_single
        )
        return _make_decoder_gradient_receipt(
            prepared, received, iteration_nonce=runner.attempt_nonce
        )

    return runner.run(
        global_manifest_digest=_candidate_digest(authority, "global_manifest"),
        plan_digest=gate_digest,
        gate_id=3,
        prepare=prepare,
        domain_collective=execute,
    )
