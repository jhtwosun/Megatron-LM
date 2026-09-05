# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 decoder-payload transport for one iteration."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_execution import DecoderSourceWindow
from megatron.core.mdp.dynamic_cp_routing import attach_local_decoder_payload_tensors
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.dynamic_cp_transport import (
    PreparedDecoderPayloadBundle,
    _execute_validated_decoder_payload_bundle,
    prepare_decoder_payload_bundle,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

__all__ = ()


def run_repeated_d4_decoder_payload(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    source_window: DecoderSourceWindow | None,
    buffers_by_dtype: Mapping[torch.dtype, tuple[Tensor, Tensor]],
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
    byte_generator: Callable[[int], Any] | None = None,
) -> PreparedDecoderPayloadBundle:
    """Prepare and exchange one domain's decoder payload behind D4 gate 0."""

    def prepare():
        if not callable(all_to_all_single):
            raise MdpConfigurationError("MDP: repeated-D4 payload all_to_all_single is callable.")
        is_source = binding.global_rank in authority.source_rank_by_lane.values()
        if is_source != (source_window is not None):
            raise MdpStateError(
                "MDP: repeated-D4 payload source window exists only on the domain source rank."
            )
        local_tensors = (
            attach_local_decoder_payload_tensors(
                authority.payload_ledger,
                plan=authority.plan,
                global_manifest=authority.global_manifest,
                source_rank_by_lane=authority.source_rank_by_lane,
                participant_ranks=authority.participant_ranks,
                source_window=source_window,
                global_rank=binding.global_rank,
            )
            if source_window is not None
            else MappingProxyType({})
        )
        return prepare_decoder_payload_bundle(
            authority.payload_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            source_rank_by_lane=authority.source_rank_by_lane,
            participant_ranks=authority.participant_ranks,
            global_rank=binding.global_rank,
            local_tensors=local_tensors,
            buffers_by_dtype=buffers_by_dtype,
        )

    def execute(prepared: PreparedDecoderPayloadBundle) -> PreparedDecoderPayloadBundle:
        _execute_validated_decoder_payload_bundle(
            prepared, group=binding.domain_group, all_to_all_single=all_to_all_single
        )
        return prepared

    return run_repeated_d4_authority_collective(
        binding,
        authority,
        gate_id=0,
        prepare=prepare,
        domain_collective=execute,
        byte_generator=byte_generator,
    )
