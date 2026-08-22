# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP encoder autograd: forward handle, chunk backward, and encoder THD params.

Activation recompute is native MCore ``TransformerConfig`` checkpointing routed
through the vision config override channel; MDP owns no RNG fork and never
replays the complete vision model in P5.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.plan import EncoderThdSegment, frame_cu_seqlens, frame_lengths
from megatron.core.packed_seq_params import PackedSeqParams


def build_encoder_packed_seq_params(
    segments: Sequence[EncoderThdSegment], *, allocator: MdpBufferAllocator, device
) -> PackedSeqParams:
    """Vision-only ``PackedSeqParams(qkv_format="thd")`` for one chunk sub-layout.

    Frame boundaries are derived from each segment's ``grid_thw``; this metadata
    is unrelated to — and must never be mixed with — decoder sample ``cu_seqlens``.
    """
    cu = frame_cu_seqlens(segments)
    lengths = frame_lengths(segments)
    max_seqlen = max(lengths) if lengths else 0
    cu_tensor = allocator.acquire(
        rows=len(cu), width=0, dtype=torch.int32, device=device, tag="encoder_cu_seqlens"
    )
    cu_tensor.copy_(torch.tensor(cu, dtype=torch.int32))
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_tensor,
        cu_seqlens_kv=cu_tensor,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


@dataclass
class EncoderForwardHandle:
    """The producer's retained P2 state: graph-connected chunk output planes.

    Chunk outputs stay in a list and are never concatenated: ``torch.cat`` would
    allocate a second full copy while all chunk outputs are alive, cancelling
    exactly the memory ``encoder_max_payload_rows`` was meant to save.
    """

    iteration: int
    producer_worker_id: int
    chunk_outputs: tuple
    chunk_layouts: tuple

    def __post_init__(self):
        self.chunk_outputs = tuple(
            (output,) if torch.is_tensor(output) else output for output in self.chunk_outputs
        )
        if len(self.chunk_outputs) != len(self.chunk_layouts):
            raise MdpStateError(
                "MDP: forward handle violates: one layout per chunk output "
                f"({len(self.chunk_outputs)} outputs, {len(self.chunk_layouts)} layouts)."
            )
        plane_count = None
        for index, (outputs, layout) in enumerate(zip(self.chunk_outputs, self.chunk_layouts)):
            if not isinstance(outputs, tuple) or not outputs:
                raise MdpStateError(
                    f"MDP: chunk {index} violates: a non-empty tuple of output planes."
                )
            if plane_count is None:
                plane_count = len(outputs)
            elif len(outputs) != plane_count:
                raise MdpStateError(
                    "MDP: forward handle violates: identical plane count per chunk."
                )
            for plane_id, output in enumerate(outputs):
                if output.shape[0] != layout.total_output_rows:
                    raise MdpStateError(
                        f"MDP: chunk {index} plane {plane_id} violates: output rows "
                        f"== layout.total_output_rows ({output.shape[0]} != "
                        f"{layout.total_output_rows})."
                    )
        self._backward_done = False
        self._released = False

    def detached_outputs(self) -> tuple:
        """Detached views for the EMBEDDING bridge; cross-rank communication
        never joins the autograd graph."""
        detached = tuple(
            tuple(output.detach() for output in outputs) for outputs in self.chunk_outputs
        )
        if detached and all(len(outputs) == 1 for outputs in detached):
            return tuple(outputs[0] for outputs in detached)
        return detached

    def backward(self, chunk_grads: Sequence[Tensor]) -> None:
        """One multi-tensor backward over all chunk outputs."""
        if self._backward_done:
            raise MdpStateError("MDP: forward handle violates: exactly one backward.")
        if len(chunk_grads) != len(self.chunk_outputs):
            raise MdpStateError(
                f"MDP: backward violates: one gradient per chunk output "
                f"({len(chunk_grads)} grads, {len(self.chunk_outputs)} outputs)."
            )
        flat_outputs = []
        flat_grads = []
        for index, (outputs, grads) in enumerate(zip(self.chunk_outputs, chunk_grads)):
            if torch.is_tensor(grads):
                grads = (grads,)
            if not isinstance(grads, tuple) or len(grads) != len(outputs):
                raise MdpStateError(
                    f"MDP: chunk {index} gradient violates: one gradient per output plane."
                )
            for plane_id, (output, grad) in enumerate(zip(outputs, grads)):
                if grad.shape != output.shape or grad.dtype != output.dtype:
                    raise MdpStateError(
                        f"MDP: chunk {index} plane {plane_id} gradient violates: shape "
                        f"and dtype match the output ({tuple(grad.shape)}/{grad.dtype} "
                        f"vs {tuple(output.shape)}/{output.dtype})."
                    )
                if grad.device != output.device:
                    raise MdpStateError(
                        f"MDP: chunk {index} plane {plane_id} gradient violates: "
                        "device matches the output."
                    )
                flat_outputs.append(output)
                flat_grads.append(grad)
        # MCore checkpoint nodes replay selected/full layers here as configured;
        # retain_graph must not be used.
        torch.autograd.backward(tuple(flat_outputs), tuple(flat_grads))
        self._backward_done = True

    def release(self) -> None:
        """Valid only after backward; drops outputs, graphs, and activations."""
        if not self._backward_done:
            raise MdpStateError(
                "MDP: forward handle violates: release only after backward (training)."
            )
        self._release_now()

    def release_forward_only(self) -> None:
        """Evaluation-path release: no graph exists, no backward will come."""
        self._release_now()

    def _release_now(self) -> None:
        self.chunk_outputs = ()
        self.chunk_layouts = ()
        self._released = True

    @property
    def consumed(self) -> bool:
        """Whether this handle has been fully consumed (lifecycle invariant)."""
        return self._released
