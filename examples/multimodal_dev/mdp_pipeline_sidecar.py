# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PP x CP replicated-vision setup for the MDP pipeline sidecar."""

import torch

from examples.multimodal_dev.sidecar_prefetch import validate_fused_vision_window
from megatron.core import parallel_state as ps


def pp_cp_replicated_vision_requested(args) -> bool:
    """Return whether the b436 PP x CP vision-replica path is selected."""
    return (
        bool(getattr(args, "mdp_encoder_mode", False))
        and int(getattr(args, "pipeline_model_parallel_size", 1)) > 1
        and getattr(args, "mdp_inner_dp_scope", "cp") == "pp_cp"
        and not bool(getattr(args, "text_only", False))
    )


def cp_fused_vision_requested(args) -> bool:
    """Return whether PP=1 should precompute fused CP vision windows."""
    return (
        bool(getattr(args, "mdp_encoder_mode", False))
        and int(getattr(args, "pipeline_model_parallel_size", 1)) == 1
        and int(getattr(args, "context_parallel_size", 1)) > 1
        and getattr(args, "mdp_inner_dp_scope", "cp") in ("cp", "pp_cp")
        and not bool(getattr(args, "text_only", False))
        and bool(getattr(args, "use_packed_sequence", False))
        and validate_fused_vision_window(
            getattr(args, "mdp_fused_vision_window", False),
            int(
                getattr(args, "mdp_vision_encoder_max_sequence_length", 0)
                or 0
            ),
        )
    )


def broadcast_vision_state(model, group, group_name: str = "PP") -> None:
    """Initialize every replicated vision tower from the first group rank."""
    if group is None or torch.distributed.get_world_size(group=group) <= 1:
        return
    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        raise RuntimeError(
            "PP x CP replicated vision requires vision_model on every pipeline stage"
        )
    source = int(torch.distributed.get_process_group_ranks(group)[0])

    def broadcast_tensor(tensor):
        if tensor.is_cuda:
            torch.distributed.broadcast(tensor, src=source, group=group)
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"replicated vision {group_name} broadcast requires CUDA for CPU staging"
            )
        staged = tensor.to(
            device=torch.device("cuda", torch.cuda.current_device()),
            non_blocking=False,
        )
        torch.distributed.broadcast(staged, src=source, group=group)
        tensor.copy_(staged.cpu())

    with torch.no_grad():
        for tensor in (*vision_model.parameters(), *vision_model.buffers()):
            broadcast_tensor(tensor.data)


def mark_downstream_pp_vision_params_shared(model, pp_rank: int) -> None:
    """Exclude non-PP0 replicas from global parameter-norm accounting."""
    if int(pp_rank) == 0:
        return
    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        return
    for parameter in vision_model.parameters():
        parameter.shared = True


def _build_pp_cp_groups(args):
    """Build the encoder CP gather group (PP0 only) and PP vision grad-sync group.

    The encoder gather group spans only CP ranks at PP stage 0.  Non-PP0 ranks
    run the vision encoder for gradient flow but do not receive gathered
    embeddings.  A separate PP vision sync group allows PP0 to all-reduce its
    vision gradients with PP1+ replicas after backward.
    """
    from examples.multimodal_dev.mdp_parallel_groups import (
        compute_encoder_cp_groups,
        build_local_process_group,
        get_parallel_order,
    )

    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    tp_size = int(getattr(args, "tensor_model_parallel_size", 1))
    cp_size = int(getattr(args, "context_parallel_size", 1))
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1))
    encoder_cp_size = int(getattr(args, "encoder_context_parallel_size", None) or cp_size)

    model_parallel_size = tp_size * cp_size * pp_size
    if model_parallel_size <= 0 or world_size % model_parallel_size:
        raise RuntimeError(
            "PP x CP replicated vision requires world_size divisible by "
            f"TP*CP*PP; got world={world_size}, TP={tp_size}, "
            f"CP={cp_size}, PP={pp_size}"
        )

    order = get_parallel_order(args)
    enc_gather_groups, pp_sync_groups = compute_encoder_cp_groups(
        world_size=world_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        encoder_cp_size=encoder_cp_size,
        order=order,
    )

    from examples.multimodal_dev.mdp_parallel_groups import build_local_process_group as _blpg
    enc_gather_group, _, _ = _blpg(rank_groups=enc_gather_groups, this_rank=rank)
    pp_vision_sync_group, _, _ = _blpg(rank_groups=pp_sync_groups, this_rank=rank)

    if tp_size > 1:
        from examples.multimodal_dev.mdp_parallel_groups import (
            compute_pp_cp_inner_dp_layout,
            _normalize_parallel_order,
        )
        from megatron.core import parallel_state as ps

        rank_generator = ps.RankGenerator(
            tp=tp_size,
            ep=1,
            dp=world_size // model_parallel_size,
            pp=pp_size,
            cp=cp_size,
            order=order,
        )
        tp_source_group, _, _ = _blpg(
            rank_groups=rank_generator.get_ranks("tp"),
            this_rank=rank,
            backend="gloo",
        )
    else:
        tp_source_group = None

    return enc_gather_group, pp_vision_sync_group, tp_source_group


def configure_pp_cp_replicated_vision(model, args) -> bool:
    """Attach encoder CP gather group and PP vision grad-sync group to ``model``.

    The encoder gather group (``_mdp_inner_dp_group``) now spans only CP ranks
    at PP stage 0 instead of the full PP×CP group.  This eliminates the
    all-gather waste where non-PP0 ranks received and discarded the full
    embedding tensor.

    Non-PP0 ranks still run the vision encoder (their parameters receive
    gradients via the PP vision sync group all-reduce after backward) but are
    not in the embedding gather collective.
    """
    if not pp_cp_replicated_vision_requested(args):
        return False
    if not bool(getattr(args, "use_packed_sequence", False)):
        raise RuntimeError("PP x CP replicated vision requires --use-packed-sequence")
    if bool(getattr(args, "use_megatron_fsdp", False)) or bool(
        getattr(args, "use_torch_fsdp2", False)
    ):
        raise RuntimeError("PP x CP replicated vision does not support FSDP")
    if not torch.distributed.is_initialized():
        raise RuntimeError("PP x CP replicated vision setup requires torch.distributed")
    if getattr(model, "vision_model", None) is None:
        raise RuntimeError(
            "PP x CP replicated vision requires vision_model on every pipeline stage"
        )

    enc_gather_group, pp_vision_sync_group, tp_source_group = _build_pp_cp_groups(args)
    pp_group = ps.get_pipeline_model_parallel_group()
    broadcast_vision_state(model, pp_group)
    mark_downstream_pp_vision_params_shared(
        model,
        ps.get_pipeline_model_parallel_rank(),
    )

    pp_rank = ps.get_pipeline_model_parallel_rank()
    attrs = {
        "_mdp_enabled": True,
        # PP0 participates in the encoder CP gather; PP1+ run the encoder
        # locally but do not join the all-gather collective.
        "_mdp_inner_dp_group": enc_gather_group if pp_rank == 0 else None,
        "_mdp_pp_vision_sync_group": pp_vision_sync_group,
        "_mdp_tp_source_group": tp_source_group,
        "_mdp_tp_source_group_device": (
            "cpu" if tp_source_group is not None else None
        ),
        "_mdp_pp_cp_inner": True,
        "_mdp_is_pp0_gather_rank": pp_rank == 0,
        "_mdp_cp_fused_sidecar": False,
        "_pp_cp_batch_sidecar": False,
        "_pipeline_sidecar_enabled": True,
        "_mdp_rank_assignment": None,
        "_mdp_rank_assignment_row_counts": None,
    }
    for name, value in attrs.items():
        setattr(model, name, value)
    return True
