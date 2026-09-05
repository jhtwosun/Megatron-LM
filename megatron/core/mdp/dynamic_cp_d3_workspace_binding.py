# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private one-active-workspace binding for the D3 producer seam."""

from dataclasses import replace

import torch

from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.dynamic_cp_d3_workspace import _DynamicIterationWorkspace
from megatron.core.mdp.dynamic_cp_runtime import (
    _bind_pre_authority_dynamic_producer,
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
    _PreAuthorityDynamicProducer,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.storage import MdpEmbeddingStorage

__all__ = ("_D3WorkspaceBindingOwner",)


def _add_cleanup_note(primary: BaseException, description: str, secondary: BaseException) -> None:
    try:
        primary.add_note(f"suppressed D3 workspace {description} error: {secondary!r}")
    except BaseException:
        pass


class _D3WorkspaceBindingOwner:
    """Own one authority-identity workspace and its bound producer cleanup."""

    def __init__(
        self,
        *,
        rank: int,
        device: torch.device,
        allocator: MdpBufferAllocator,
        storage: MdpEmbeddingStorage,
    ) -> None:
        self._rank = rank
        self._device = device
        self._allocator = allocator
        self._storage = storage
        self._workspace: _DynamicIterationWorkspace | None = None
        self._bound_authority: _DynamicIterationAuthority | None = None
        self._bound_producer: _DynamicProducerCarrier | None = None
        self._bound_cleanup = None

    @property
    def is_idle(self) -> bool:
        return (
            self._workspace is None
            and self._bound_producer is None
            and self._bound_cleanup is None
            and self._bound_authority is None
        )

    def bind(
        self, *, authority: _DynamicIterationAuthority, producer: _PreAuthorityDynamicProducer
    ) -> _DynamicProducerCarrier:
        """Bind one fresh workspace to one exact authority capability."""
        if type(authority) is not _DynamicIterationAuthority:
            raise MdpConfigurationError("MDP: D3 workspace binding uses exact iteration authority.")
        if type(producer) is not _PreAuthorityDynamicProducer:
            raise MdpConfigurationError(
                "MDP: D3 workspace binding uses exact pre-authority producer."
            )
        if not self.is_idle:
            raise MdpStateError("MDP: D3 workspace binding starts from one fresh workspace.")

        workspace = None
        producer_cleanup = None
        try:
            workspace = _DynamicIterationWorkspace(
                authority=authority,
                rank=self._rank,
                device=self._device,
                allocator=self._allocator,
                storage=self._storage,
            )
            self._workspace = workspace
            bound = _bind_pre_authority_dynamic_producer(
                producer=producer,
                authority=authority,
                payload_destination_views=workspace.payload_views,
                embedding_destination_views=workspace.embedding_views,
                gradient_destination_views=workspace.gradient_views,
                summed_gradient_destination_views=workspace.summed_gradient_views,
            )
            if type(bound) is not _DynamicProducerCarrier:
                raise MdpConfigurationError(
                    "MDP: D3 workspace binding returns typed producer carrier."
                )
            producer_cleanup = bound.cleanup
            if (
                bound.authority is not authority
                or bound.pre_authority is not producer
                or bound.payload_destination_views is not workspace.payload_views
                or bound.embedding_destination_views is not workspace.embedding_views
                or bound.gradient_destination_views is not workspace.gradient_views
                or bound.summed_gradient_destination_views is not workspace.summed_gradient_views
            ):
                raise MdpConfigurationError(
                    "MDP: D3 workspace binding preserves exact authority, producer, "
                    "and destination views."
                )
            cleaned = False

            def cleanup_resources() -> None:
                nonlocal cleaned
                if cleaned:
                    return
                cleaned = True
                if self._workspace is workspace:
                    self._workspace = None
                errors = []
                try:
                    workspace.release()
                except BaseException as error:
                    errors.append(("cleanup", error))
                try:
                    producer_cleanup()
                except BaseException as error:
                    errors.append(("producer cleanup", error))
                if errors:
                    _, primary = errors[0]
                    for secondary_description, secondary in errors[1:]:
                        _add_cleanup_note(primary, secondary_description, secondary)
                    raise primary

            returned = None

            def cleanup() -> None:
                if cleaned:
                    return
                if returned is None:
                    raise MdpStateError(
                        "MDP: D3 workspace cleanup requires its exact bound producer."
                    )
                self.cleanup_bound_producer(authority, returned)

            returned = replace(bound, cleanup=cleanup)
            if type(returned) is not _DynamicProducerCarrier or returned.cleanup is not cleanup:
                raise MdpConfigurationError(
                    "MDP: D3 workspace binding retains its exact cleanup capability."
                )
            self._bound_authority = authority
            self._bound_producer = returned
            self._bound_cleanup = cleanup_resources
            return returned
        except BaseException as error:
            self._bound_authority = None
            self._bound_producer = None
            self._bound_cleanup = None
            if workspace is not None:
                if self._workspace is workspace:
                    self._workspace = None
                try:
                    workspace.release()
                except BaseException as cleanup_error:
                    _add_cleanup_note(error, "binding cleanup", cleanup_error)
            if producer_cleanup is not None:
                try:
                    producer_cleanup()
                except BaseException as cleanup_error:
                    _add_cleanup_note(error, "producer cleanup", cleanup_error)
            raise

    def require_bound_producer(
        self, authority: _DynamicIterationAuthority, producer: _DynamicProducerCarrier, /
    ) -> _DynamicProducerCarrier:
        """Return the one exact bound carrier without consuming its cleanup."""
        workspace = self._workspace
        if (
            self._bound_authority is not authority
            or self._bound_producer is not producer
            or self._bound_cleanup is None
            or workspace is None
            or workspace.authority is not authority
            or producer.authority is not authority
        ):
            raise MdpStateError("MDP: D3 workspace binding requires its exact bound producer.")
        return producer

    def cleanup_bound_producer(
        self, authority: _DynamicIterationAuthority, producer: _DynamicProducerCarrier, /
    ) -> None:
        """Consume exact cleanup authority before entering fallible callbacks."""
        if (
            self._bound_authority is not authority
            or self._bound_producer is not producer
            or self._bound_cleanup is None
        ):
            raise MdpStateError("MDP: D3 workspace binding requires its exact bound producer.")
        cleanup = self._bound_cleanup
        self._bound_authority = None
        self._bound_producer = None
        self._bound_cleanup = None
        assert cleanup is not None
        cleanup()

    def require_workspace(
        self, authority: _DynamicIterationAuthority
    ) -> _DynamicIterationWorkspace:
        """Return the one active workspace only for its exact authority object."""
        workspace = self._workspace
        if workspace is None or workspace.authority is not authority:
            raise MdpStateError("MDP: D3 workspace binding requires its exact active workspace.")
        return workspace
