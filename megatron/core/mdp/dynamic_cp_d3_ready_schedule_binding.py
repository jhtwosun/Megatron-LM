# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private schedule callback binding for the D3 ready handoff."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_ready_handoff import _compose_d3_decoder_ready_handoff
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError

__all__ = ("_D3ReadyScheduleBinding", "_make_d3_ready_schedule_binding")


_PENDING_BINDING_SEALS: dict[object, tuple[int, ...]] = {}


@dataclass(frozen=True, slots=True)
class _D3ReadyScheduleBinding:
    """Forward one completed D3 transport result to the ready-handoff composer."""

    workspace_owner: _D3WorkspaceBindingOwner = field(compare=False, repr=False)
    cp_partition_mode: str
    decoder_group_getter: Callable[..., Any] = field(compare=False, repr=False)
    decoder_group_ranks_getter: Callable[..., Any] = field(compare=False, repr=False)
    rebuild_microbatch: Callable[..., Any] = field(compare=False, repr=False)
    _factory_seal: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D3ReadyScheduleBinding:
            raise MdpStateError("MDP: D3 ready schedule binding is minted by its factory.")
        _validate_static_dependencies(
            workspace_owner=self.workspace_owner,
            cp_partition_mode=self.cp_partition_mode,
            decoder_group_getter=self.decoder_group_getter,
            decoder_group_ranks_getter=self.decoder_group_ranks_getter,
            rebuild_microbatch=self.rebuild_microbatch,
        )
        fingerprint = _PENDING_BINDING_SEALS.pop(self._factory_seal, None)
        if fingerprint != _binding_fingerprint(self):
            raise MdpStateError("MDP: D3 ready schedule binding is minted by its factory.")

    def __call__(
        self,
        authority: Any,
        producer: Any,
        payload_bundle: Any,
        payload_result: Any,
        embedding_exchange: Any,
        embedding_result: Any,
        /,
    ) -> Any:
        """Compose one ready iteration without retaining per-iteration state."""
        return _compose_d3_decoder_ready_handoff(
            workspace_owner=self.workspace_owner,
            authority=authority,
            producer=producer,
            payload_bundle=payload_bundle,
            payload_result=payload_result,
            embedding_exchange=embedding_exchange,
            embedding_result=embedding_result,
            cp_partition_mode=self.cp_partition_mode,
            decoder_group_getter=self.decoder_group_getter,
            decoder_group_ranks_getter=self.decoder_group_ranks_getter,
            rebuild_microbatch=self.rebuild_microbatch,
        )


def _binding_fingerprint(binding: _D3ReadyScheduleBinding) -> tuple[int, ...]:
    return tuple(
        id(value)
        for value in (
            binding.workspace_owner,
            binding.cp_partition_mode,
            binding.decoder_group_getter,
            binding.decoder_group_ranks_getter,
            binding.rebuild_microbatch,
        )
    )


def _validate_static_dependencies(
    *,
    workspace_owner: Any,
    cp_partition_mode: Any,
    decoder_group_getter: Any,
    decoder_group_ranks_getter: Any,
    rebuild_microbatch: Any,
) -> None:
    if type(workspace_owner) is not _D3WorkspaceBindingOwner:
        raise MdpConfigurationError(
            "MDP: D3 ready schedule binding uses its exact workspace owner."
        )
    if type(cp_partition_mode) is not str or cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError(
            "MDP: D3 ready schedule binding CP partition mode is supported."
        )
    for name, callback in (
        ("decoder group getter", decoder_group_getter),
        ("decoder group ranks getter", decoder_group_ranks_getter),
        ("rebuild callback", rebuild_microbatch),
    ):
        if not callable(callback):
            raise MdpConfigurationError(f"MDP: D3 ready schedule binding {name} must be callable.")


def _make_d3_ready_schedule_binding(
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    cp_partition_mode: str,
    decoder_group_getter: Callable[..., Any],
    decoder_group_ranks_getter: Callable[..., Any],
    rebuild_microbatch: Callable[..., Any],
) -> _D3ReadyScheduleBinding:
    """Mint one reusable, immutable D3 schedule callback capability."""
    _validate_static_dependencies(
        workspace_owner=workspace_owner,
        cp_partition_mode=cp_partition_mode,
        decoder_group_getter=decoder_group_getter,
        decoder_group_ranks_getter=decoder_group_ranks_getter,
        rebuild_microbatch=rebuild_microbatch,
    )
    token = object()
    kwargs = dict(
        workspace_owner=workspace_owner,
        cp_partition_mode=cp_partition_mode,
        decoder_group_getter=decoder_group_getter,
        decoder_group_ranks_getter=decoder_group_ranks_getter,
        rebuild_microbatch=rebuild_microbatch,
        _factory_seal=token,
    )
    _PENDING_BINDING_SEALS[token] = tuple(
        id(value) for name, value in kwargs.items() if name != "_factory_seal"
    )
    try:
        return _D3ReadyScheduleBinding(**kwargs)
    except BaseException:
        _PENDING_BINDING_SEALS.pop(token, None)
        raise
