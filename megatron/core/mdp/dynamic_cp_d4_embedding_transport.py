# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 embedding transport for one iteration."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    PreparedDynamicBridgeExchange,
    _execute_validated_dynamic_bridge_exchange,
    prepare_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError

__all__ = ()


def run_repeated_d4_embedding(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    item_outputs: Mapping[GlobalVisionItemId, Tensor],
    send_buffer: Tensor,
    receive_buffer: Tensor,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
    byte_generator: Callable[[int], Any] | None = None,
) -> PreparedDynamicBridgeExchange:
    """Prepare and exchange one domain's vision embeddings behind D4 gate 1."""

    def prepare():
        if not callable(all_to_all_single):
            raise MdpConfigurationError("MDP: repeated-D4 embedding all_to_all_single is callable.")
        if not isinstance(item_outputs, Mapping):
            raise MdpConfigurationError("MDP: repeated-D4 embedding item outputs form a mapping.")
        expected_items = {
            item_id
            for item_id, producer_rank in authority.producer_rank_by_item.items()
            if producer_rank == binding.global_rank
        }
        try:
            exact_coverage = set(item_outputs) == expected_items
        except Exception as error:
            raise MdpConfigurationError(
                "MDP: repeated-D4 embedding item output keys are readable."
            ) from error
        if not exact_coverage:
            raise MdpPlanError(
                "MDP: repeated-D4 embedding outputs exactly cover local item authority."
            )
        local_tensors = MappingProxyType(
            {
                entry.key: item_outputs[entry.key.item_id]
                for entry in authority.embedding_ledger.entries
                if entry.src_global_rank == binding.global_rank
            }
        )
        return prepare_dynamic_bridge_exchange(
            authority.embedding_ledger,
            authority.gradient_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            width=authority.bridge_width,
            dtype=authority.bridge_dtype,
            participant_ranks=authority.participant_ranks,
            global_rank=binding.global_rank,
            local_tensors=local_tensors,
            send_buffer=send_buffer,
            receive_buffer=receive_buffer,
        )

    def execute(prepared: PreparedDynamicBridgeExchange) -> PreparedDynamicBridgeExchange:
        _execute_validated_dynamic_bridge_exchange(
            prepared, group=binding.domain_group, all_to_all_single=all_to_all_single
        )
        return prepared

    return run_repeated_d4_authority_collective(
        binding,
        authority,
        gate_id=1,
        prepare=prepare,
        domain_collective=execute,
        byte_generator=byte_generator,
    )
