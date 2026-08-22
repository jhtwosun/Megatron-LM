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
    """Holds ordered detached embedding leaves per vision microbatch."""

    def __init__(self, allocator: MdpBufferAllocator) -> None:
        self._allocator = allocator
        self._leaves: dict = {}

    def put_leaf(self, mb_id: int, leaf: Tensor, *, base: Optional[Tensor] = None) -> None:
        """Store one detached leaf; duplicate insertion for the same id fails."""
        self.put_leaves(mb_id, (leaf,), bases=(leaf if base is None else base,))

    def put_leaves(self, mb_id: int, leaves: tuple, *, bases: Optional[tuple] = None) -> None:
        """Atomically store ordered leaves and their exact allocator bases."""
        if mb_id in self._leaves:
            raise MdpStateError(
                f"MDP: microbatch_id={mb_id} violates: one leaf per microbatch per " "iteration."
            )
        if not isinstance(leaves, tuple) or not leaves:
            raise MdpStateError(
                f"MDP: microbatch_id={mb_id} violates: a non-empty tuple of leaves."
            )
        if bases is None:
            bases = leaves
        if not isinstance(bases, tuple) or len(bases) != len(leaves):
            raise MdpStateError(
                f"MDP: microbatch_id={mb_id} violates: one exact allocator base " "per leaf plane."
            )
        for plane_id, leaf in enumerate(leaves):
            if not leaf.is_leaf or not leaf.requires_grad or leaf.grad_fn is not None:
                raise MdpStateError(
                    f"MDP: leaf plane {plane_id} for microbatch_id={mb_id} violates: "
                    "is_leaf and requires_grad with no grad_fn. Only detached leaves "
                    "cross into the decoder domain."
                )
            base = bases[plane_id]
            if not isinstance(base, Tensor) or (
                base is not leaf and getattr(leaf, "_base", None) is not base
            ):
                raise MdpStateError(
                    f"MDP: leaf plane {plane_id} for microbatch_id={mb_id} violates: "
                    "the exact allocator-returned base owns the stored leaf view."
                )
        self._leaves[mb_id] = (leaves, bases)

    def get_leaf(self, mb_id: int) -> Optional[Tensor]:
        """Non-destructive read; ``None`` for text-only microbatches."""
        entry = self._leaves.get(mb_id)
        if entry is None:
            return None
        leaves, _ = entry
        if len(leaves) != 1:
            raise MdpStateError(
                f"MDP: microbatch_id={mb_id} has {len(leaves)} output planes; " "use get_leaves()."
            )
        return leaves[0]

    def get_leaves(self, mb_id: int) -> Optional[tuple]:
        """Non-destructive ordered read; ``None`` for text-only microbatches."""
        entry = self._leaves.get(mb_id)
        return None if entry is None else entry[0]

    def pop_grad(self, mb_id: int) -> Optional[Tensor]:
        """Take the leaf gradient and release the entry; ``None`` if no leaf."""
        grads = self.pop_grads(mb_id)
        if grads is None:
            return None
        if len(grads) != 1:
            raise MdpStateError(
                f"MDP: microbatch_id={mb_id} has {len(grads)} output planes; " "use pop_grads()."
            )
        return grads[0]

    def pop_grads(self, mb_id: int) -> Optional[tuple]:
        """Take ordered leaf gradients and release all leaf entries."""
        entry = self._leaves.get(mb_id)
        if entry is None:
            return None
        leaves, bases = entry
        for plane_id, leaf in enumerate(leaves):
            if leaf.grad is None:
                raise MdpStateError(
                    f"MDP: leaf plane {plane_id} for microbatch_id={mb_id} violates: "
                    "a non-empty training leaf has .grad when pop_grads is called. "
                    "Decoder backward must have written the unnormalized numerator "
                    "gradient into every leaf."
                )
        grads = tuple(leaf.grad for leaf in leaves)
        del self._leaves[mb_id]
        self._release_bases(bases)
        return grads

    def release(self, mb_id: int) -> None:
        """Explicit release for the evaluation path."""
        entry = self._leaves.pop(mb_id, None)
        if entry is not None:
            self._release_bases(entry[1])

    def _release_bases(self, bases: tuple) -> None:
        first_error = None
        for base in bases:
            try:
                self._allocator.release(base)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def assert_empty(self) -> None:
        """Lifecycle invariant: storage returns to zero at each iteration boundary."""
        if self._leaves:
            raise MdpStateError(
                f"MDP: embedding storage violates: empty at iteration boundary "
                f"(unconsumed microbatch ids: {sorted(self._leaves)})."
            )
