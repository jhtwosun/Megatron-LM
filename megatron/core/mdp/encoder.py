# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP encoder domain: replicated encoder DDP over WORLD, ZeRO-1, and gradient
finalization.

The encoder is fully replicated on every rank and reduced once over WORLD with
prescale 1 (``calculate_per_token_loss=True`` makes the DDP gradient scaling
factor 1.0). The distributed optimizer shards its state over the same WORLD
domain. The encoder never enters the decoder schedule model list.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch

from megatron.core.distributed import DistributedDataParallel
from megatron.core.mdp.config import MdpConfig, apply_vision_config_overrides
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.groups import MdpProcessGroups
from megatron.core.mdp.protocols import MdpModelAdapter
from megatron.core.mdp.rank_mapping import MdpRankMap
from megatron.core.process_groups_config import ProcessGroupCollection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncoderDomain:
    """The assembled encoder side: DDP module, optimizer, effective config."""

    encoder_ddp: Any
    encoder_optimizer: Any
    effective_config: Any


def build_effective_encoder_config(model_config, mdp_config: MdpConfig):
    """Return an immutable sidecar config whose CP dimension is encoder CP."""
    return replace(
        apply_vision_config_overrides(model_config, mdp_config.vision_config_overrides),
        context_parallel_size=mdp_config.encoder_cp,
    )


def build_encoder_pg_collection(
    rank_map: MdpRankMap, *, encoder_cp: int, process_groups: MdpProcessGroups
) -> ProcessGroupCollection:
    """Process groups for the encoder domain.

    ``cp`` is the local logical worker's encoder-CP group. Replicated
    parameters and optimizer state retain the existing WORLD domains through
    ``dp = dp_cp = intra_dp_cp = intra_dist_opt = WORLD``. ``tp/pp/ep/expt_dp``
    reuse the canonical rank-local singleton installed by
    :func:`install_mdp_process_groups`; no rank-dependent group creation occurs
    here. Model/expert composite groups remain unset.
    """
    if encoder_cp != rank_map.spec.encoder_cp:
        raise MdpConfigurationError(
            f"MDP: encoder_cp={encoder_cp} violates: encoder_cp matches the rank-map "
            f"spec ({rank_map.spec.encoder_cp})."
        )
    world = process_groups.world_group
    mine = process_groups.singleton_group

    pgs = ProcessGroupCollection()
    pgs.cp = process_groups.encoder_cp_group
    pgs.dp = world
    pgs.dp_cp = world
    pgs.intra_dp_cp = world
    pgs.intra_dist_opt = world
    pgs.tp = mine
    pgs.pp = mine
    pgs.ep = mine
    pgs.mp = None
    # DDP treats a missing expert-DP group as permission to create a
    # rank-dependent singleton. Reuse the canonical singleton instead.
    pgs.expt_dp = mine
    pgs.tp_ep_pp = None
    pgs.inter_dist_opt = None
    return pgs


def build_encoder_domain(
    *,
    adapter: MdpModelAdapter,
    model_config,
    mdp_config: MdpConfig,
    ddp_config,
    optimizer_config,
    encoder_pgs: ProcessGroupCollection,
    wrap_mixed_precision: bool = True,
) -> EncoderDomain:
    """Assemble the encoder domain (API design 14.2).

    Order: vision config from the override channel; encoder via the adapter's
    shared factory; the same mixed-precision wrapper depth as the decoder;
    DDP over the encoder process groups; DistributedOptimizer from the DDP
    buffers.
    """
    for field_name in ("overlap_grad_reduce", "overlap_param_gather"):
        if getattr(ddp_config, field_name, False):
            raise MdpConfigurationError(
                f"MDP: {field_name}=True violates: encoder gradients only exist after "
                "P5, so an overlapped reduction would fire against an empty buffer "
                "during decoder backward. validate_mdp_config rejects this upstream."
            )
    if getattr(ddp_config, "num_distributed_optimizer_instances", 1) != 1:
        raise MdpConfigurationError(
            "MDP: num_distributed_optimizer_instances != 1 violates: the encoder "
            "shards its optimizer state over WORLD."
        )

    effective_config = build_effective_encoder_config(model_config, mdp_config)
    logger.info(
        "MDP: effective vision config overrides: %s; encoder context parallel size: %s",
        list(mdp_config.vision_config_overrides),
        mdp_config.encoder_cp,
    )
    encoder = adapter.build_encoder(effective_config, pg_collection=encoder_pgs)
    if wrap_mixed_precision and (
        getattr(effective_config, "fp16", False) or getattr(effective_config, "bf16", False)
    ):
        from megatron.core.transformer.module import Float16Module

        encoder = Float16Module(effective_config, encoder.cuda())
    else:
        encoder = encoder.cuda()

    encoder_ddp = DistributedDataParallel(
        config=effective_config,
        ddp_config=ddp_config,
        module=encoder,
        pg_collection=encoder_pgs,
    )
    assert_encoder_prescale_is_one(encoder_ddp)

    from megatron.core.optimizer import get_megatron_optimizer

    encoder_optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[encoder_ddp],
        pg_collection=encoder_pgs,
        # Megatron cannot derive matching Gloo groups for a caller-built
        # collection.
        use_gloo_process_groups=False,
    )
    return EncoderDomain(
        encoder_ddp=encoder_ddp,
        encoder_optimizer=encoder_optimizer,
        effective_config=effective_config,
    )


def assert_encoder_prescale_is_one(encoder_ddp) -> None:
    """Encoder ranks divide one batch's work; they are not data replicas, so
    WORLD reduction must not pre-divide gradients by W."""
    for buffer in list(encoder_ddp.buffers) + list(encoder_ddp.expert_parallel_buffers):
        if buffer.gradient_scaling_factor != 1.0:
            raise MdpConfigurationError(
                f"MDP: encoder gradient buffer prescale "
                f"{buffer.gradient_scaling_factor} violates: prescale == 1. "
                "calculate_per_token_loss=True must be set before DDP construction."
            )


def assert_parameter_disjointness(
    encoder_ddp, decoder_chunks: Sequence, all_trainable_parameters=None
) -> None:
    """Encoder and decoder parameters must be disjoint (and, when the full set
    is provided, together cover every trainable parameter).

    The load-bearing half is the leak check: a shared parameter would be
    reduced by the decoder finalizer in P4, before P5 produces its encoder
    gradient — silently wrong, never an error.
    """
    encoder_ids = {id(p) for p in encoder_ddp.module.parameters()}
    if not encoder_ids:
        raise MdpConfigurationError("MDP: the encoder has no parameters.")
    decoder_ids = set()
    for index, chunk in enumerate(decoder_chunks):
        leaked = [name for name, p in chunk.named_parameters() if id(p) in encoder_ids]
        if leaked:
            raise MdpConfigurationError(
                f"MDP: decoder chunk {index} contains encoder parameters "
                f"{leaked[:5]}; the native schedule would reduce their gradients "
                "before P5 produces them."
            )
        decoder_ids.update(id(p) for p in chunk.parameters())
    if all_trainable_parameters is not None:
        missing = [
            id(p) for p in all_trainable_parameters
            if id(p) not in encoder_ids and id(p) not in decoder_ids
        ]
        if missing:
            raise MdpConfigurationError(
                f"MDP: {len(missing)} trainable parameters belong to neither domain; "
                "encoder and decoder must cover every trainable parameter."
            )


def finalize_encoder_grads(encoder_ddp, *, globally_reduced_num_tokens: torch.Tensor) -> None:
    """WORLD sum-reduce, then scale by ``1/clamp(T_global, min=1)``.

    ``globally_reduced_num_tokens`` must be the same in-place reduced tensor
    the native decoder finalizer produced (captured via
    ``wrap_finalize_model_grads``); recounting tokens on WORLD would count PP
    replicas more than once. When the count is zero, ``clamp(min=1)`` matches
    the native path's no-scaling behavior (masks already zeroed the numerator).
    """
    encoder_ddp.finish_grad_sync()
    denominator = torch.clamp(globally_reduced_num_tokens.float(), min=1.0)
    # Device-side reciprocal: `.item()` here forced a full host sync between
    # the WORLD reduce-scatter and the scale kernels. The double-precision
    # round trip reproduces `float(1.0 / denominator.item())` bit-exactly
    # (fp32 -> f64 is exact, one f64 divide, one rounding back to fp32), and
    # `grad_data *= tensor` broadcasts the 0-dim fp32 scalar exactly like the
    # Python float the kernel would otherwise receive.
    scale = (1.0 / denominator.double()).float().reshape(())
    encoder_ddp.scale_gradients(scale)
