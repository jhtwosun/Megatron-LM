# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private gate-3 local gradient-preparation binding for D3."""

from dataclasses import dataclass, field
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_runtime import (
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
    _prepare_decoder_gradient_exchange,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpStateError

__all__ = ("_D3GradientPreparationBinding", "_make_d3_gradient_preparation_binding")


_PENDING_BINDING_SEALS: dict[object, tuple[int, int]] = {}


@dataclass(frozen=True, slots=True)
class _D3GradientPreparationBinding:
    """Prepare one D3 reverse-gradient exchange without entering gate 3."""

    workspace_owner: _D3WorkspaceBindingOwner = field(compare=False, repr=False)
    cp_partition_mode: str
    _factory_seal: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D3GradientPreparationBinding:
            raise MdpStateError("MDP: D3 gradient preparation binding is minted by its factory.")
        _validate_static_dependencies(
            workspace_owner=self.workspace_owner, cp_partition_mode=self.cp_partition_mode
        )
        fingerprint = _PENDING_BINDING_SEALS.pop(self._factory_seal, None)
        if fingerprint != _binding_fingerprint(self):
            raise MdpStateError("MDP: D3 gradient preparation binding is minted by its factory.")

    def __call__(self, authority: Any, producer: Any, ready: Any, /) -> Any:
        """Prepare one sealed local gradient exchange from retained ready leaves."""
        if type(authority) is not _DynamicIterationAuthority:
            raise MdpConfigurationError(
                "MDP: D3 gradient preparation uses exact iteration authority."
            )
        workspace = self.workspace_owner.require_workspace(authority)
        if workspace.authority is not authority or workspace._released:
            raise MdpStateError("MDP: D3 gradient preparation requires its exact active workspace.")
        if type(producer) is not _DynamicProducerCarrier or producer.authority is not authority:
            raise MdpBridgeError(
                "MDP: D3 gradient preparation retains exact authority-bound producer."
            )
        if (
            producer.gradient_destination_views is not workspace.gradient_views
            or producer.summed_gradient_destination_views is not workspace.summed_gradient_views
        ):
            raise MdpBridgeError(
                "MDP: D3 gradient preparation retains exact workspace gradient views."
            )
        buffers = workspace.gradient_transport_buffers
        if type(buffers) is not tuple or len(buffers) != 2:
            raise MdpConfigurationError(
                "MDP: D3 gradient preparation requires one active gradient transport pair."
            )
        return _prepare_decoder_gradient_exchange(
            ready,
            global_manifest=authority.global_manifest,
            plan=authority.plan,
            embedding_ledger=authority.embedding_ledger,
            gradient_ledger=authority.gradient_ledger,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            embedding_width=authority.bridge_width,
            embedding_dtype=authority.bridge_dtype,
            cp_partition_mode=self.cp_partition_mode,
            global_rank=workspace.rank,
            participant_ranks=authority.participant_ranks,
            send_buffer=buffers[0],
            receive_buffer=buffers[1],
        )


def _binding_fingerprint(binding: _D3GradientPreparationBinding) -> tuple[int, int]:
    return (id(binding.workspace_owner), id(binding.cp_partition_mode))


def _validate_static_dependencies(*, workspace_owner: Any, cp_partition_mode: Any) -> None:
    if type(workspace_owner) is not _D3WorkspaceBindingOwner:
        raise MdpConfigurationError(
            "MDP: D3 gradient preparation binding uses its exact workspace owner."
        )
    if type(cp_partition_mode) is not str or cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError(
            "MDP: D3 gradient preparation binding CP partition mode is supported."
        )


def _make_d3_gradient_preparation_binding(
    *, workspace_owner: _D3WorkspaceBindingOwner, cp_partition_mode: str
) -> _D3GradientPreparationBinding:
    """Mint one immutable local gradient-preparation callback capability."""
    _validate_static_dependencies(
        workspace_owner=workspace_owner, cp_partition_mode=cp_partition_mode
    )
    token = object()
    _PENDING_BINDING_SEALS[token] = (id(workspace_owner), id(cp_partition_mode))
    try:
        return _D3GradientPreparationBinding(
            workspace_owner=workspace_owner,
            cp_partition_mode=cp_partition_mode,
            _factory_seal=token,
        )
    except BaseException:
        _PENDING_BINDING_SEALS.pop(token, None)
        raise
