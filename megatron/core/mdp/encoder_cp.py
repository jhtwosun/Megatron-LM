# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-agnostic packed-THD helpers for encoder context parallelism."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

_INTEGER_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True)
class EncoderCpPlan:
    """Frozen packed-THD row partition shared by encoder inputs and outputs."""

    cu_seqlens: Tensor
    cu_seqlens_padded: Tensor
    valid_padded_indices: Optional[Tensor]
    rank_major_indices: Optional[Tensor]
    partition_sizes: tuple[int, ...]
    cp_size: int
    cp_rank: int
    max_seqlen: int
    total_rows: int
    total_padded_rows: int

    @property
    def local_indices(self) -> Optional[Tensor]:
        """Native packed-row indices owned by this encoder-CP rank."""
        if self.rank_major_indices is None:
            return None
        start = sum(self.partition_sizes[: self.cp_rank])
        return self.rank_major_indices.narrow(0, start, self.partition_sizes[self.cp_rank])


def _group_geometry(group: dist.ProcessGroup) -> tuple[int, int]:
    """Return group size/rank while retaining the lightweight unit-test seam."""
    if hasattr(group, "size") and hasattr(group, "rank"):
        return int(group.size()), int(group.rank())
    return int(dist.get_world_size(group=group)), int(dist.get_rank(group=group))


def _validate_cu_seqlens(cu_seqlens: Tensor) -> tuple[Tensor, int]:
    if not isinstance(cu_seqlens, Tensor) or cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be a one-dimensional tensor")
    if cu_seqlens.dtype not in _INTEGER_DTYPES:
        raise ValueError("cu_seqlens must have an integer dtype")
    if cu_seqlens.numel() == 0 or int(cu_seqlens[0].item()) != 0:
        raise ValueError("cu_seqlens must start at zero")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if lengths.numel() and bool(torch.any(lengths <= 0).item()):
        raise ValueError("cu_seqlens boundaries must be strictly increasing")
    return lengths, int(cu_seqlens[-1].item())


def _validate_partition_indices(
    indices: Tensor,
    *,
    expected_rows: int,
    total_padded_rows: int,
    device: torch.device,
    cp_rank: int,
) -> Tensor:
    """Validate the cheap invariants of TE's trusted THD partition output."""
    if not isinstance(indices, Tensor) or indices.dim() != 1:
        raise ValueError(f"encoder CP rank {cp_rank} partition indices must be one-dimensional")
    if indices.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"encoder CP rank {cp_rank} partition indices must have an integer dtype")
    if indices.device != device:
        raise ValueError(f"encoder CP rank {cp_rank} partition indices must remain on {device}")
    if indices.numel() != expected_rows:
        raise ValueError(
            f"encoder CP rank {cp_rank} partition size {indices.numel()} != {expected_rows}"
        )
    if indices.numel():
        minimum = int(indices.amin().item())
        maximum = int(indices.amax().item())
        if minimum < 0 or maximum >= total_padded_rows:
            raise ValueError(
                f"encoder CP rank {cp_rank} partition indices must be within "
                f"[0, {total_padded_rows})"
            )
    return indices.to(dtype=torch.int64)


def build_encoder_cp_plan(cu_seqlens: Tensor, cp_group: dist.ProcessGroup) -> EncoderCpPlan:
    """Build packed-THD row ownership, padding each frame to ``2 * ECP``."""
    cp_size, cp_rank = _group_geometry(cp_group)
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank {cp_rank} must be within [0, {cp_size})")
    lengths, total_rows = _validate_cu_seqlens(cu_seqlens)
    max_unpadded = int(lengths.max().item()) if lengths.numel() else 0
    if cp_size == 1:
        return EncoderCpPlan(
            cu_seqlens,
            cu_seqlens,
            None,
            None,
            (total_rows,),
            1,
            0,
            max_unpadded,
            total_rows,
            total_rows,
        )
    if lengths.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=cu_seqlens.device)
        return EncoderCpPlan(
            cu_seqlens, cu_seqlens, empty, empty, (0,) * cp_size, cp_size, cp_rank, 0, 0, 0
        )

    alignment = 2 * cp_size
    lengths_i64 = lengths.to(dtype=torch.int64)
    padded_lengths_i64 = ((lengths_i64 + alignment - 1) // alignment) * alignment
    padded_cumulative_i64 = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=cu_seqlens.device),
            torch.cumsum(padded_lengths_i64, dim=0),
        )
    )
    total_padded_rows = int(padded_cumulative_i64[-1].item())
    if total_padded_rows > torch.iinfo(torch.int32).max:
        raise ValueError("encoder CP padded row count exceeds the TE int32 boundary")
    padded_cumulative = padded_cumulative_i64.to(dtype=cu_seqlens.dtype)
    original_starts = torch.repeat_interleave(cu_seqlens[:-1].to(torch.int64), lengths_i64)
    padded_starts = torch.repeat_interleave(padded_cumulative[:-1].to(torch.int64), lengths_i64)
    original_rows = torch.arange(total_rows, dtype=torch.int64, device=cu_seqlens.device)
    valid_padded_indices = padded_starts + original_rows - original_starts

    te_cu_seqlens = padded_cumulative.to(dtype=torch.int32)
    expected_local_rows = total_padded_rows // cp_size
    rank_indices = []
    for rank in range(cp_size):
        indices = get_thd_partitioned_indices(te_cu_seqlens, total_padded_rows, cp_size, rank)
        rank_indices.append(
            _validate_partition_indices(
                indices,
                expected_rows=expected_local_rows,
                total_padded_rows=total_padded_rows,
                device=cu_seqlens.device,
                cp_rank=rank,
            )
        )
    partition_sizes = tuple(int(indices.numel()) for indices in rank_indices)
    return EncoderCpPlan(
        cu_seqlens=cu_seqlens,
        cu_seqlens_padded=padded_cumulative,
        valid_padded_indices=valid_padded_indices,
        rank_major_indices=torch.cat(rank_indices),
        partition_sizes=partition_sizes,
        cp_size=cp_size,
        cp_rank=cp_rank,
        max_seqlen=int(padded_lengths_i64.max().item()),
        total_rows=total_rows,
        total_padded_rows=total_padded_rows,
    )


def partition_encoder_cp_inputs(
    hidden: Tensor, rotary: Tensor, plan: EncoderCpPlan
) -> tuple[Tensor, Tensor]:
    """Pad and select this rank's native THD rows for hidden states and RoPE."""
    if hidden.dim() == 0 or hidden.size(0) != plan.total_rows:
        rows = hidden.size(0) if hidden.dim() else 0
        raise ValueError(f"hidden rows {rows} != {plan.total_rows}")
    if rotary.dim() == 0 or rotary.size(0) != plan.total_rows:
        rows = rotary.size(0) if rotary.dim() else 0
        raise ValueError(f"rotary rows {rows} != {plan.total_rows}")
    if plan.cp_size == 1 or plan.total_rows == 0:
        return hidden, rotary
    local_indices = plan.local_indices
    assert plan.valid_padded_indices is not None and local_indices is not None
    padded_hidden = hidden.new_zeros((plan.total_padded_rows, *hidden.shape[1:])).index_copy(
        0, plan.valid_padded_indices, hidden
    )
    padded_rotary = rotary.new_zeros((plan.total_padded_rows, *rotary.shape[1:])).index_copy(
        0, plan.valid_padded_indices, rotary
    )
    return padded_hidden.index_select(0, local_indices), padded_rotary.index_select(
        0, local_indices
    )


def restore_encoder_cp_output(
    local_output: Tensor, plan: EncoderCpPlan, cp_group: dist.ProcessGroup
) -> Tensor:
    """Autograd-gather rank-major output rows and remove per-frame padding."""
    if local_output.dim() == 0 or local_output.size(0) != plan.partition_sizes[plan.cp_rank]:
        raise ValueError(
            "local output rows must match the encoder CP partition size "
            f"{plan.partition_sizes[plan.cp_rank]}"
        )
    group_size, group_rank = _group_geometry(cp_group)
    if (group_size, group_rank) != (plan.cp_size, plan.cp_rank):
        raise ValueError("cp_group geometry must match the frozen encoder CP plan")
    if plan.cp_size == 1 or plan.total_padded_rows == 0:
        return local_output
    split_sizes = list(plan.partition_sizes)
    gathered = gather_from_sequence_parallel_region(
        local_output.contiguous(),
        tensor_parallel_output_grad=True,
        group=cp_group,
        output_split_sizes=split_sizes if len(set(split_sizes)) > 1 else None,
    )
    if gathered.size(0) != plan.total_padded_rows:
        raise ValueError(f"gathered output rows {gathered.size(0)} != {plan.total_padded_rows}")
    assert plan.rank_major_indices is not None
    assert plan.valid_padded_indices is not None
    padded_output = local_output.new_zeros(
        (plan.total_padded_rows, *local_output.shape[1:])
    ).index_copy(0, plan.rank_major_indices, gathered)
    return padded_output.index_select(0, plan.valid_padded_indices)
