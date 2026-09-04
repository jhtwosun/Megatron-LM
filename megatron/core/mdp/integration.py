# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDP installation seams for the Megatron training loop.

Every function here is a no-op unless ``--mdp-enable`` is set, so the call
sites in ``megatron/training/training.py`` are side-effect free when MDP is
off (a stated acceptance criterion). The model side registers its adapter
builder before ``pretrain()`` runs — core must not import ``examples/``.

Seams:

* the encoder domain is built between ``get_megatron_optimizer`` and the LR
  scheduler, because the composite optimizer must exist before the scheduler
  binds to it;
* ``forward_backward_func`` is wrapped at both the training and the
  evaluation call site (each builds its own callable);
* ``config.finalize_model_grads_func`` is wrapped on first schedule wrap to
  capture the in-place reduced global token count (no MCore change).
"""

import logging
from typing import Callable, Optional

import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import ModalityBridge
from megatron.core.mdp.config import (
    SUPPORTED_RANK_ORDER,
    MdpCompatibilityOptions,
    MdpConfig,
    validate_mdp_config,
)
from megatron.core.mdp.dynamic_cp_d3_composition import _build_d3_runtime_facade
from megatron.core.mdp.encoder import (
    assert_parameter_disjointness,
    build_encoder_domain,
    build_encoder_pg_collection,
)
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime
from megatron.core.mdp.schedule import (
    _wrap_d3_forward_backward,
    wrap_finalize_model_grads,
    wrap_forward_backward,
)
from megatron.core.mdp.storage import MdpEmbeddingStorage

logger = logging.getLogger(__name__)

#: Registered by the model entry point: ``builder(args) -> (adapter, vision_config)``.
_ADAPTER_BUILDER: Optional[Callable] = None

#: The runtime for this process, once built. Module-level because the seams
#: are far apart in the training loop.
_RUNTIME: Optional[MdpRuntime] = None

#: The one reusable training-only D3 facade, built lazily after model config
#: and native Dynamic-CP groups are both available.
_D3_FACADE = None

_D3_STATUS_TIMEOUT_SECONDS = 30.0


def set_adapter_builder(builder: Callable) -> None:
    """Register the model-side adapter builder (call before ``pretrain()``)."""
    global _ADAPTER_BUILDER
    _ADAPTER_BUILDER = builder


def get_runtime() -> Optional[MdpRuntime]:
    """This process's MdpRuntime, or ``None`` when MDP is off."""
    return _RUNTIME


def mdp_enabled(args) -> bool:
    """Whether ``--mdp-enable`` is on for this run."""
    return bool(getattr(args, "mdp_enable", False))


def mdp_config_from_args(args) -> MdpConfig:
    """Build the frozen MdpConfig from the entry point's MDP and encoder flags."""
    modules = getattr(args, "encoder_recompute_modules", None)
    return MdpConfig(
        enable=mdp_enabled(args),
        encoder_cp=getattr(args, "mdp_encoder_cp", 1),
        encoder_max_payload_rows=getattr(args, "mdp_encoder_max_payload_rows", None),
        encoder_recompute_granularity=getattr(
            args, "encoder_recompute_granularity", None
        ),
        encoder_recompute_method=getattr(args, "encoder_recompute_method", None),
        encoder_recompute_num_layers=getattr(
            args, "encoder_recompute_num_layers", None
        ),
        encoder_recompute_modules=tuple(modules) if modules is not None else None,
        locality_slack_permille=getattr(args, "mdp_locality_slack_permille", 10),
        row_alignment=getattr(args, "mdp_row_alignment", 1),
        plan_check_interval=getattr(args, "mdp_plan_check_interval", 1),
        debug_plan_payload_check=getattr(args, "mdp_debug_plan_payload_check", False),
        pixel_locality=getattr(args, "mdp_pixel_locality", False),
        overlap_window_capture=getattr(args, "mdp_overlap_window_capture", False),
    )


def compatibility_options_from_args(args) -> MdpCompatibilityOptions:
    """Snapshot the Megatron args MDP validates against its support matrix."""
    fsdp = bool(
        getattr(args, "use_torch_fsdp2", False)
        or getattr(args, "use_custom_fsdp", False)
        or getattr(args, "use_megatron_fsdp", False)
    )
    cuda_graph = getattr(args, "cuda_graph_impl", "none") not in (None, "none")
    offload = bool(
        getattr(args, "cpu_offloading", False)
        or getattr(args, "fine_grained_activation_offloading", False)
        or getattr(args, "offload_optimizer_states", False)
    )
    # Mirror initialize_model_parallel's order selection (initialize.py):
    # --use-tp-pp-dp-mapping switches to 'tp-cp-ep-pp-dp', which MDP's rank
    # mapping does not support — the snapshot must report the REAL order so
    # validate_mdp_config's rejection can fire instead of building planning
    # groups that no longer match the decoder replicas.
    rank_order = (
        "tp-cp-ep-pp-dp"
        if getattr(args, "use_tp_pp_dp_mapping", False)
        else SUPPORTED_RANK_ORDER
    )
    return MdpCompatibilityOptions(
        world_size=args.world_size,
        tensor_parallel_size=args.tensor_model_parallel_size,
        pipeline_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_parallel_size=getattr(args, "expert_model_parallel_size", 1),
        rank_order=rank_order,
        virtual_pipeline_parallel_size=getattr(
            args, "virtual_pipeline_model_parallel_size", None
        ),
        calculate_per_token_loss=getattr(args, "calculate_per_token_loss", False),
        use_distributed_optimizer=getattr(args, "use_distributed_optimizer", False),
        distributed_optimizer_instances=getattr(
            args, "num_distributed_optimizer_instances", 1
        ),
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        fsdp_enabled=fsdp,
        cuda_graph_enabled=cuda_graph,
        activation_offload_enabled=offload,
        overlap_grad_reduce=getattr(args, "overlap_grad_reduce", False),
        overlap_param_gather=getattr(args, "overlap_param_gather", False),
        overlap_param_gather_with_optimizer_step=bool(
            getattr(args, "overlap_param_gather_with_optimizer_step", False)
        ),
        reuse_grad_buf_for_mxfp8_param_ag=bool(
            getattr(args, "reuse_grad_buf_for_mxfp8_param_ag", False)
        ),
        delay_grad_reduce=bool(getattr(args, "delay_grad_reduce", False)),
        overlap_moe_expert_parallel_comm=bool(
            getattr(args, "overlap_moe_expert_parallel_comm", False)
        ),
        dynamic_context_parallel=bool(getattr(args, "dynamic_context_parallel", False)),
        checkpoint_mode=getattr(args, "ckpt_format", "torch_dist"),
        save_requested=getattr(args, "save", None) is not None,
        load_requested=getattr(args, "load", None) is not None,
    )


def validate_from_args(args) -> None:
    """Run the full support-matrix validation from the parsed args."""
    validate_mdp_config(mdp_config_from_args(args), compatibility_options_from_args(args))


def maybe_build_mdp_domain(
    *,
    args,
    model,
    optimizer,
    optimizer_config,
    ddp_config,
    decoder_pg_collection=None,
):
    """Build the MDP runtime and encoder domain; returns the optimizer.

    Called in ``setup_model_and_optimizer`` after the decoder optimizer is
    built and before the LR scheduler binds. Returns *optimizer* unchanged
    when MDP is off.
    """
    global _RUNTIME
    if not mdp_enabled(args) or optimizer is None:
        return optimizer
    if _ADAPTER_BUILDER is None:
        raise MdpConfigurationError(
            "MDP: --mdp-enable is set but no adapter builder was registered. The "
            "model entry point must call set_adapter_builder() before pretrain(); "
            "core cannot import the model package."
        )

    mdp_config = mdp_config_from_args(args)
    validate_mdp_config(mdp_config, compatibility_options_from_args(args))
    if decoder_pg_collection is not None and hasattr(
        decoder_pg_collection, "get_language_model_collection"
    ):
        decoder_pg_collection = decoder_pg_collection.get_language_model_collection()

    rank_map = build_rank_map(
        MdpRankSpec(
            world_size=args.world_size,
            tp=args.tensor_model_parallel_size,
            pp=args.pipeline_model_parallel_size,
            cp=args.context_parallel_size,
            ep=getattr(args, "expert_model_parallel_size", 1),
            encoder_cp=mdp_config.encoder_cp,
        )
    )
    rank_view = rank_map.view(torch.distributed.get_rank())
    process_groups = install_mdp_process_groups(
        rank_map,
        group_registry=MdpGroupRegistry(),
        decoder_pg_collection=decoder_pg_collection,
    )
    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=mdp_config.encoder_cp, process_groups=process_groups
    )

    adapter, vision_config = _ADAPTER_BUILDER(args)
    embedding_width = getattr(adapter, "embedding_width", args.hidden_size)
    if type(embedding_width) is not int or embedding_width <= 0:
        raise MdpConfigurationError("MDP: model adapter embedding_width must be a positive integer.")
    encoder_domain = build_encoder_domain(
        adapter=adapter,
        model_config=vision_config,
        mdp_config=mdp_config,
        ddp_config=ddp_config,
        optimizer_config=optimizer_config,
        encoder_pgs=encoder_pgs,
    )
    assert_parameter_disjointness(encoder_domain.encoder_ddp, model)

    if mdp_config.encoder_max_payload_rows is not None:
        logger.warning(
            "MDP: encoder_max_payload_rows=%d caps encoder chunks; a single vision "
            "item larger than the cap forms an oversized chunk (check the dataset's "
            "maximum grid).",
            mdp_config.encoder_max_payload_rows,
        )

    if args.bf16:
        params_dtype = torch.bfloat16
    elif args.fp16:
        params_dtype = torch.float16
    else:
        params_dtype = torch.float32
    allocator = DirectBufferAllocator()
    _RUNTIME = MdpRuntime(
        config=mdp_config,
        rank_map=rank_map,
        rank_view=rank_view,
        process_groups=process_groups,
        adapter=adapter,
        encoder_domain=encoder_domain,
        planner=MdpPlanner(
            rank_view,
            locality_slack_permille=mdp_config.locality_slack_permille,
            capacity_policy=RowCapacityPolicy(mdp_config.row_alignment),
            pixel_locality=mdp_config.pixel_locality,
        ),
        bridge=ModalityBridge(allocator),
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=embedding_width,
        params_dtype=params_dtype,
        num_vpp_chunks=len(model),
    )
    logger.info(
        "MDP: runtime installed (outer_dp_rank=%d, worker_id=%s, endpoint=%d, "
        "workers=%d, encoder_recompute_granularity=%s)",
        rank_view.outer_dp_rank,
        rank_view.my_worker_id,
        rank_view.endpoint_rank,
        len(rank_view.worker_ids),
        mdp_config.encoder_recompute_granularity,
    )

    from megatron.core.mdp.optimizer import build_mdp_composite_optimizer

    return build_mdp_composite_optimizer(optimizer, encoder_domain.encoder_optimizer)


def d3_owns_data_schedule(config) -> bool:
    """Whether D3, rather than native ``wrap_data_iterator``, owns this batch."""
    return _RUNTIME is not None and getattr(config, "dynamic_context_parallel", None) is True


def maybe_wrap_forward_backward(
    forward_backward_func: Callable, config=None, *, training: bool = True
) -> Callable:
    """Wrap the schedule with the MDP phases; no-op when MDP is off.

    Also installs the token capture on first use — this is the first point at
    which ``config`` is guaranteed to carry its final grad finalizer.
    """
    if _RUNTIME is None:
        return forward_backward_func
    if type(training) is not bool:
        raise MdpConfigurationError("MDP: schedule purpose must be an exact bool.")
    if config is not None:
        wrap_finalize_model_grads(config, _RUNTIME)
    if d3_owns_data_schedule(config) and training:
        global _D3_FACADE
        if _D3_FACADE is None:
            _D3_FACADE = _build_d3_facade_from_mcore(_RUNTIME, config)
        return _wrap_d3_forward_backward(forward_backward_func, _D3_FACADE)
    return wrap_forward_backward(forward_backward_func, _RUNTIME)


def _build_d3_facade_from_mcore(runtime: MdpRuntime, config):
    """Bind the locked D3 composition to native Dynamic-CP dependencies."""
    if getattr(config, "dynamic_context_parallel", None) is not True:
        raise MdpConfigurationError("MDP: D3 requires native Dynamic-CP groups.")
    if getattr(config, "sequence_packing_scheduler", None) != "default_dynamic_cp":
        raise MdpConfigurationError(
            "MDP: D3 requires the native default_dynamic_cp planning contract."
        )
    max_seqlen = getattr(config, "max_seqlen_per_dp_cp_rank", None)
    minimum_cp = getattr(config, "min_dynamic_context_parallel_size", None)
    if type(max_seqlen) is not int or max_seqlen <= 0:
        raise MdpConfigurationError("MDP: D3 max sequence length must be a positive integer.")
    if type(minimum_cp) is not int or minimum_cp <= 0:
        raise MdpConfigurationError("MDP: D3 minimum CP size must be a positive integer.")

    codec_factory = getattr(runtime.adapter, "build_dynamic_decoder_payload_codec", None)
    if not callable(codec_factory):
        raise MdpConfigurationError("MDP: D3 model adapter must provide its decoder codec.")

    from megatron.core import parallel_state
    from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware

    group = runtime.process_groups.world_group
    ranks = tuple(torch.distributed.get_process_group_ranks(group))
    return _build_d3_runtime_facade(
        producer_runtime=runtime,
        codec=codec_factory(),
        group=group,
        participant_ranks=ranks,
        global_rank=torch.distributed.get_rank(),
        device=runtime.device,
        expected_source_lanes=tuple(range(len(ranks))),
        decoder_solver=next_hdp_group_packing_aware,
        max_seqlen_per_rank=max_seqlen,
        minimum_cp_size=minimum_cp,
        decoder_group_getter=parallel_state.get_dynamic_data_context_parallel_groups,
        decoder_group_ranks_getter=torch.distributed.get_process_group_ranks,
        timeout_seconds=_D3_STATUS_TIMEOUT_SECONDS,
    )


def reset_for_testing() -> None:
    """Drop module state between tests."""
    global _RUNTIME, _ADAPTER_BUILDER, _D3_FACADE
    _RUNTIME = None
    _ADAPTER_BUILDER = None
    _D3_FACADE = None
