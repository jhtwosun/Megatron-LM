# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Concrete private composition for the locked D3 runtime."""

from types import MappingProxyType
from typing import Any, Callable

import torch
import torch.distributed as dist

from megatron.core.mdp.dynamic_cp_bridge_transport import (
    execute_dynamic_bridge_exchange,
    prepare_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_d3_authority_construction import (
    build_d3_iteration_authority,
    derive_decoder_item_authority,
)
from megatron.core.mdp.dynamic_cp_d3_coordinator import (
    _D3Coordinator,
    _D3CoordinatorBindings,
    _D3GateStatusContext,
)
from megatron.core.mdp.dynamic_cp_d3_encoder_backward import _execute_d3_encoder_backward
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding import (
    _make_d3_encoder_completion_gate_binding,
)
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation import (
    _make_d3_encoder_completion_preparation_binding,
)
from megatron.core.mdp.dynamic_cp_d3_encoder_finalize import _make_d3_encoder_finalize_binding
from megatron.core.mdp.dynamic_cp_d3_gradient_gate_binding import _make_d3_gradient_gate_binding
from megatron.core.mdp.dynamic_cp_d3_gradient_preparation_binding import (
    _make_d3_gradient_preparation_binding,
)
from megatron.core.mdp.dynamic_cp_d3_iteration_commit import _execute_d3_iteration_commit
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import (
    DecoderMetadataGatherResult,
    gather_decoder_source_manifests,
)
from megatron.core.mdp.dynamic_cp_d3_private_facade import _D3PrivateFacade
from megatron.core.mdp.dynamic_cp_d3_ready_schedule_binding import _make_d3_ready_schedule_binding
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_execution import (
    _PrecollectiveStatus,
    _run_precollective_consensus,
)
from megatron.core.mdp.dynamic_cp_routing import attach_local_decoder_payload_tensors
from megatron.core.mdp.dynamic_cp_runtime import (
    DYNAMIC_RUNTIME_SCHEMA_VERSION,
    _consensus_dynamic_execution_config,
    _DynamicExecutionConfig,
    _DynamicProducerCarrier,
)
from megatron.core.mdp.dynamic_cp_transport import (
    _execute_validated_decoder_payload_bundle,
    make_precollective_status_gather,
    prepare_decoder_payload_bundle,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)


def _embedding_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 2
    if dtype is torch.float16:
        return 3
    raise MdpConfigurationError("MDP: D3 composition embedding dtype is BF16 or FP16.")


def _run_status(
    context: _D3GateStatusContext,
    local_error: BaseException | None,
    *,
    global_rank: int,
    group_ranks: tuple[int, ...],
    all_gather_status: Callable,
    timeout_seconds: float,
) -> None:
    if type(context) is not _D3GateStatusContext:
        raise MdpConfigurationError("MDP: D3 composition status uses an exact gate context.")
    status = _PrecollectiveStatus(
        global_rank=global_rank,
        global_manifest_digest=context.authority.global_manifest.digest,
        plan_digest=context.authority.plan.digest,
        error_code=int(local_error is not None),
        gate_id=context.gate_id,
    )
    try:
        _run_precollective_consensus(
            status,
            group_ranks=group_ranks,
            all_gather_status=all_gather_status,
            timeout_seconds=timeout_seconds,
        )
    except (MdpBridgeError, MdpPlanError) as error:
        if local_error is not None and error.__cause__ is None:
            raise error from local_error
        raise
    if local_error is not None:
        raise MdpStateError("MDP: D3 status consensus accepted a local error.") from local_error


def _build_d3_runtime_facade(
    *,
    producer_runtime: Any,
    codec: Any,
    group: Any,
    participant_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    expected_source_lanes: tuple[int, ...],
    decoder_solver: Callable,
    max_seqlen_per_rank: int,
    minimum_cp_size: int,
    decoder_group_getter: Callable,
    decoder_group_ranks_getter: Callable,
    timeout_seconds: float,
    group_ranks_getter: Callable = dist.get_process_group_ranks,
    all_to_all_single: Callable = dist.all_to_all_single,
) -> _D3PrivateFacade:
    """Build the training-only TP1/PP1/CP1/ECP1/VPP1 D3 facade."""
    ranks = participant_ranks
    if (
        type(ranks) is not tuple
        or not ranks
        or len(set(ranks)) != len(ranks)
        or type(global_rank) is not int
        or global_rank not in ranks
    ):
        raise MdpConfigurationError("MDP: D3 composition uses exact participant ranks.")
    if tuple(group_ranks_getter(group)) != ranks:
        raise MdpConfigurationError("MDP: D3 composition group order matches participants.")
    if ranks != tuple(range(len(ranks))):
        raise MdpConfigurationError("MDP: D3 composition participants are the ordered WORLD ranks.")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: D3 composition uses an explicit CUDA device.")
    spec = getattr(getattr(producer_runtime, "rank_map", None), "spec", None)
    topology = tuple(getattr(spec, name, None) for name in ("tp", "ep", "pp", "cp", "encoder_cp"))
    if topology != (1, 1, 1, 1, 1) or producer_runtime.num_vpp_chunks != 1:
        raise MdpConfigurationError("MDP: D3 composition is locked to TP1/EP1/PP1/CP1/ECP1/VPP1.")
    if producer_runtime.config.overlap_window_capture is not False:
        raise MdpConfigurationError("MDP: D3 composition requires static encoder CP and capture.")
    if producer_runtime.device != device:
        raise MdpConfigurationError("MDP: D3 composition uses the producer runtime device.")
    for callback in (
        decoder_solver,
        decoder_group_getter,
        decoder_group_ranks_getter,
        group_ranks_getter,
        all_to_all_single,
        getattr(codec, "rebuild_microbatch", None),
        getattr(producer_runtime, "_prepare_dynamic_encoder_producer", None),
    ):
        if not callable(callback):
            raise MdpConfigurationError("MDP: D3 composition dependencies are callable.")
    if (
        type(expected_source_lanes) is not tuple
        or tuple(sorted(expected_source_lanes)) != expected_source_lanes
        or type(max_seqlen_per_rank) is not int
        or max_seqlen_per_rank <= 0
        or type(minimum_cp_size) is not int
        or minimum_cp_size <= 0
    ):
        raise MdpConfigurationError("MDP: D3 composition planning inputs are canonical.")
    dtype = producer_runtime.params_dtype
    dtype_id = _embedding_dtype_id(dtype)
    status_gather = make_precollective_status_gather(
        group=group,
        group_ranks=ranks,
        global_rank=global_rank,
        device=device,
        group_ranks_getter=group_ranks_getter,
    )
    config_slot = {"value": None}

    def config_factory(_iterator, *, num_microbatches, forward_only):
        del num_microbatches
        config = _DynamicExecutionConfig(
            schema_version=DYNAMIC_RUNTIME_SCHEMA_VERSION,
            forward_only=forward_only,
            partition_mode="contiguous",
            embedding_width=producer_runtime.hidden_size,
            embedding_dtype_id=dtype_id,
            participant_ranks=ranks,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            pipeline_parallel_size=1,
            configured_context_parallel_size=1,
            encoder_context_parallel_size=1,
            virtual_pipeline_parallel_size=1,
            expert_group_ranks=None,
            sequence_parallel=False,
            dynamic_encoder_context_parallel=False,
            overlap_window_capture=False,
        )
        config_slot["value"] = config
        return config

    def producer_factory(iterator, *, num_microbatches, forward_only):
        return producer_runtime._prepare_dynamic_encoder_producer(
            iterator, num_microbatches=num_microbatches, forward_only=forward_only, codec=codec
        )

    def coordinator_factory(_iterator, *, num_microbatches, forward_only):
        del num_microbatches
        config = config_slot["value"]
        if type(config) is not _DynamicExecutionConfig or config.forward_only is not forward_only:
            raise MdpStateError("MDP: D3 coordinator consumes its exact pending config.")
        config_slot["value"] = None
        workspace_owner = _D3WorkspaceBindingOwner(
            rank=global_rank,
            device=device,
            allocator=producer_runtime.allocator,
            storage=producer_runtime.storage,
        )
        authority_slot = {"value": None}

        def gather_metadata(producer, selected_config):
            if selected_config is not config:
                raise MdpConfigurationError("MDP: D3 metadata uses the coordinator config.")
            return gather_decoder_source_manifests(
                producer.local_manifest,
                expected_source_lanes=expected_source_lanes,
                group=group,
                group_ranks=ranks,
                global_rank=global_rank,
                device=device,
                timeout_seconds=timeout_seconds,
                local_prepare_error=producer.local_prepare_error,
            )

        def build_authority(metadata, _producer, selected_config):
            if type(metadata) is not DecoderMetadataGatherResult or selected_config is not config:
                raise MdpConfigurationError("MDP: D3 authority uses exact gathered metadata.")
            items = derive_decoder_item_authority(
                metadata, participant_ranks=ranks, decoder_ranks=ranks
            )
            authority = build_d3_iteration_authority(
                items,
                max_seqlen_per_rank=max_seqlen_per_rank,
                minimum_cp_size=minimum_cp_size,
                solver=decoder_solver,
                bridge_width=producer_runtime.hidden_size,
                bridge_dtype=dtype,
            )
            authority_slot["value"] = authority
            return authority

        def authority_status_gate(local_error):
            authority = authority_slot["value"]
            digest = bytes(16) if authority is None else authority.global_manifest.digest
            plan_digest = bytes(16) if authority is None else authority.plan.digest
            status = _PrecollectiveStatus(
                global_rank, digest, plan_digest, int(local_error is not None), 0
            )
            try:
                _run_precollective_consensus(
                    status,
                    group_ranks=ranks,
                    all_gather_status=status_gather,
                    timeout_seconds=timeout_seconds,
                )
            except (MdpBridgeError, MdpPlanError) as error:
                if local_error is not None and error.__cause__ is None:
                    raise error from local_error
                raise
            if local_error is not None:
                raise MdpStateError(
                    "MDP: D3 authority consensus accepted a local error."
                ) from local_error

        def prepare_payload(authority, producer):
            workspace = workspace_owner.require_workspace(authority)
            local_tensors = (
                MappingProxyType({})
                if producer.source_window is None
                else attach_local_decoder_payload_tensors(
                    authority.payload_ledger,
                    plan=authority.plan,
                    global_manifest=authority.global_manifest,
                    source_rank_by_lane=authority.source_rank_by_lane,
                    participant_ranks=ranks,
                    source_window=producer.source_window,
                    global_rank=global_rank,
                )
            )
            return prepare_decoder_payload_bundle(
                authority.payload_ledger,
                plan=authority.plan,
                global_manifest=authority.global_manifest,
                source_rank_by_lane=authority.source_rank_by_lane,
                participant_ranks=ranks,
                global_rank=global_rank,
                local_tensors=local_tensors,
                buffers_by_dtype=workspace.payload_transport_buffers,
            )

        def prepare_embedding(authority, producer, _payload_result):
            workspace = workspace_owner.require_workspace(authority)
            local = MappingProxyType(
                {
                    entry.key: producer.item_outputs[entry.key.item_id]
                    for entry in authority.embedding_ledger.entries
                    if entry.src_global_rank == global_rank
                }
            )
            send, receive = workspace.embedding_transport_buffers
            return prepare_dynamic_bridge_exchange(
                authority.embedding_ledger,
                authority.gradient_ledger,
                plan=authority.plan,
                global_manifest=authority.global_manifest,
                producer_rank_by_item=authority.producer_rank_by_item,
                output_rows_by_item=authority.output_rows_by_item,
                width=authority.bridge_width,
                dtype=authority.bridge_dtype,
                participant_ranks=ranks,
                global_rank=global_rank,
                local_tensors=local,
                send_buffer=send,
                receive_buffer=receive,
            )

        fallback = lambda context, error: _run_status(
            context,
            error,
            global_rank=global_rank,
            group_ranks=ranks,
            all_gather_status=status_gather,
            timeout_seconds=timeout_seconds,
        )
        gradient_preparation = _make_d3_gradient_preparation_binding(
            workspace_owner=workspace_owner, cp_partition_mode="contiguous"
        )
        gradient_gate = _make_d3_gradient_gate_binding(
            workspace_owner=workspace_owner,
            cp_partition_mode="contiguous",
            group=group,
            group_ranks=ranks,
            global_rank=global_rank,
            device=device,
            timeout_seconds=timeout_seconds,
            fallback_status_gate=fallback,
            all_gather_status=status_gather,
            group_ranks_getter=group_ranks_getter,
            all_to_all_single=all_to_all_single,
        )
        completion_preparation = _make_d3_encoder_completion_preparation_binding(
            workspace_owner=workspace_owner, cp_partition_mode="contiguous"
        )
        completion_gate = _make_d3_encoder_completion_gate_binding(
            workspace_owner=workspace_owner,
            cp_partition_mode="contiguous",
            group=group,
            group_ranks=ranks,
            global_rank=global_rank,
            device=device,
            timeout_seconds=timeout_seconds,
            fallback_status_gate=fallback,
            all_gather_status=status_gather,
            group_ranks_getter=group_ranks_getter,
        )
        finalize_gate = _make_d3_encoder_finalize_binding(
            group=group,
            group_ranks=ranks,
            global_rank=global_rank,
            device=device,
            timeout_seconds=timeout_seconds,
            fallback_status_gate=fallback,
            all_gather_status=status_gather,
            group_ranks_getter=group_ranks_getter,
        )
        ready_binding = _make_d3_ready_schedule_binding(
            workspace_owner=workspace_owner,
            cp_partition_mode="contiguous",
            decoder_group_getter=decoder_group_getter,
            decoder_group_ranks_getter=decoder_group_ranks_getter,
            rebuild_microbatch=codec.rebuild_microbatch,
        )

        def status_gate(context, error):
            if context.gate_id == 3:
                return gradient_gate.status_gate(context, error)
            if context.gate_id == 4:
                return completion_gate.status_gate(context, error)
            if context.gate_id == 5:
                return finalize_gate.status_gate(context, error)
            return fallback(context, error)

        def cleanup(value):
            if type(value) is _DynamicProducerCarrier:
                value.cleanup()
            elif getattr(value, "owner", None) is not None:
                value.owner.abort()

        bindings = _D3CoordinatorBindings(
            execution_config_consensus=lambda selected: _consensus_dynamic_execution_config(
                config=selected,
                group_ranks=ranks,
                global_rank=global_rank,
                all_gather_status=status_gather,
                timeout_seconds=timeout_seconds,
            ),
            gather_metadata=gather_metadata,
            build_authority=build_authority,
            authority_status_gate=authority_status_gate,
            bind_producer=lambda authority, producer: workspace_owner.bind(
                authority=authority, producer=producer
            ),
            prepare_payload=prepare_payload,
            execute_payload=lambda prepared: _execute_validated_decoder_payload_bundle(
                prepared, group=group, all_to_all_single=all_to_all_single
            ),
            prepare_embedding=prepare_embedding,
            execute_embedding=lambda prepared: execute_dynamic_bridge_exchange(
                prepared,
                group=group,
                group_ranks_getter=group_ranks_getter,
                all_to_all_single=all_to_all_single,
            ),
            prepare_schedule=ready_binding,
            prepare_gradient=gradient_preparation,
            execute_gradient=gradient_gate.execute_gradient,
            prepare_encoder_completion=completion_preparation,
            execute_encoder_backward=lambda prepared: _execute_d3_encoder_backward(
                completion_gate, prepared
            ),
            execute_encoder_finalize=finalize_gate.finalize,
            execute_iteration_commit=_execute_d3_iteration_commit,
            cleanup=cleanup,
            status_gate=status_gate,
        )
        return _D3Coordinator(bindings=bindings)

    return _D3PrivateFacade(
        config_factory=config_factory,
        producer_factory=producer_factory,
        coordinator_factory=coordinator_factory,
    )
