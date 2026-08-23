# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP embedding storage: detached leaves per decoder microbatch on the endpoint.

Text-only microbatches have no storage key; reads return ``None``. Endpoint
storage must return to zero every iteration (``assert_empty``).
"""

from typing import Optional

from torch import Tensor

from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.errors import MdpStateError


class MdpEmbeddingStorage:
    """Holds one detached embedding leaf and its allocator base per microbatch."""

    def __init__(self, allocator: MdpBufferAllocator) -> None:
        self._allocator = allocator
        self._leaves: dict = {}

    def put_leaf(
        self, mb_id: int, leaf: Tensor, *, allocation_base: Optional[Tensor] = None
    ) -> None:
        """Store a leaf and the exact allocator-returned tensor that owns it.

        ``allocation_base`` defaults to ``leaf`` for direct-storage callers;
        runtime allocations always pass the allocator-returned base explicitly.
        """
        if mb_id in self._leaves:
            raise MdpStateError(
                f"MDP: microbatch_id={mb_id} violates: one leaf per microbatch per " "iteration."
            )
        if not leaf.is_leaf or not leaf.requires_grad or leaf.grad_fn is not None:
            raise MdpStateError(
                f"MDP: leaf for microbatch_id={mb_id} violates: is_leaf and "
                "requires_grad with no grad_fn. Only detached leaves cross into the "
                "decoder domain."
            )
        self._leaves[mb_id] = (leaf if allocation_base is None else allocation_base, leaf)

    def get_leaf(self, mb_id: int) -> Optional[Tensor]:
        """Non-destructive read; ``None`` for text-only microbatches."""
        stored = self._leaves.get(mb_id)
        return None if stored is None else stored[1]

    def pop_grad(self, mb_id: int) -> Optional[Tensor]:
        """Take the leaf gradient and release the entry; ``None`` if no leaf."""
        stored = self._leaves.get(mb_id)
        if stored is None:
            return None
        allocation_base, leaf = stored
        if leaf.grad is None:
            raise MdpStateError(
                f"MDP: leaf for microbatch_id={mb_id} violates: a non-empty training "
                "leaf has .grad when pop_grad is called. Decoder backward must have "
                "written the unnormalized numerator gradient into the leaf."
            )
        grad = leaf.grad
        del self._leaves[mb_id]
        self._allocator.release(allocation_base)
        return grad

    def release(self, mb_id: int) -> None:
        """Explicit release for the evaluation path."""
        stored = self._leaves.pop(mb_id, None)
        if stored is not None:
            allocation_base, _ = stored
            self._allocator.release(allocation_base)

    def assert_empty(self) -> None:
        """Lifecycle invariant: storage returns to zero at each iteration boundary."""
        if self._leaves:
            raise MdpStateError(
                f"MDP: embedding storage violates: empty at iteration boundary "
                f"(unconsumed microbatch ids: {sorted(self._leaves)})."
            )
