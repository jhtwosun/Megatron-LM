# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private gate-2 composition from completed D3 transport capabilities."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    PreparedDynamicBridgeExchange,
    validate_prepared_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_d3_local_placement import (
    _place_d3_local_decoder_inputs,
    _validate_embedding_aliases,
    _validate_payload_aliases,
)
from megatron.core.mdp.dynamic_cp_d3_ready_artifacts import (
    _expected_assignments as _canonical_ready_assignments,
)
from megatron.core.mdp.dynamic_cp_d3_ready_artifacts import _materialize_d3_decoder_ready_artifacts
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderReadyIteration,
    _build_decoder_ready_iteration,
    _decoder_ready_authority_digest,
    _dynamic_iteration_plan_digest,
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
    _expected_local_assignments,
    _expected_role,
    _LocalDecoderReadyArtifacts,
    validate_decoder_ready_iteration,
)
from megatron.core.mdp.dynamic_cp_transport import (
    PreparedDecoderPayloadBundle,
    validate_prepared_decoder_payload_bundle,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError

__all__ = ("_compose_d3_decoder_ready_handoff",)


def _validate_inputs(
    *,
    workspace_owner: Any,
    authority: Any,
    producer: Any,
    payload_bundle: Any,
    payload_result: Any,
    embedding_exchange: Any,
    embedding_result: Any,
    cp_partition_mode: Any,
    rebuild_microbatch: Any,
):
    if type(workspace_owner) is not _D3WorkspaceBindingOwner:
        raise MdpConfigurationError("MDP: D3 ready handoff uses its exact workspace owner.")
    if type(authority) is not _DynamicIterationAuthority:
        raise MdpConfigurationError("MDP: D3 ready handoff uses exact iteration authority.")
    workspace = workspace_owner.require_workspace(authority)
    if workspace.authority is not authority:
        raise MdpBridgeError("MDP: D3 ready handoff workspace retains exact authority identity.")
    if type(producer) is not _DynamicProducerCarrier or producer.authority is not authority:
        raise MdpBridgeError("MDP: D3 ready handoff producer retains exact authority identity.")
    if (
        producer.payload_destination_views is not workspace.payload_views
        or producer.embedding_destination_views is not workspace.embedding_views
        or producer.gradient_destination_views is not workspace.gradient_views
        or producer.summed_gradient_destination_views is not workspace.summed_gradient_views
    ):
        raise MdpBridgeError("MDP: D3 ready handoff producer retains exact workspace views.")
    bundle = validate_prepared_decoder_payload_bundle(payload_bundle)
    exchange = validate_prepared_dynamic_bridge_exchange(embedding_exchange)
    if exchange.phase is not BridgePhase.EMBEDDING:
        raise MdpBridgeError("MDP: D3 ready handoff consumes the embedding bridge phase.")
    if payload_result is not bundle.received_tensors:
        raise MdpBridgeError("MDP: D3 ready handoff payload result is the exact gate-0 mapping.")
    if embedding_result is not exchange.received_tensors:
        raise MdpBridgeError("MDP: D3 ready handoff embedding result is the exact gate-1 mapping.")
    _validate_payload_aliases(workspace, bundle)
    _validate_embedding_aliases(workspace, exchange)
    if cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError("MDP: D3 ready handoff CP partition mode is supported.")
    if not callable(rebuild_microbatch):
        raise MdpConfigurationError("MDP: D3 ready handoff rebuild callback is callable.")
    return workspace, bundle, exchange


def _compose_d3_decoder_ready_handoff(
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    authority: _DynamicIterationAuthority,
    producer: _DynamicProducerCarrier,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_result: Mapping,
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_result: Mapping,
    cp_partition_mode: str,
    decoder_group_getter: Callable[..., Any],
    decoder_group_ranks_getter: Callable[..., Any],
    rebuild_microbatch: Callable[..., Any],
) -> DecoderReadyIteration:
    """Compose one local gate-2 carrier without entering a transport or schedule phase."""
    workspace, bundle, exchange = _validate_inputs(
        workspace_owner=workspace_owner,
        authority=authority,
        producer=producer,
        payload_bundle=payload_bundle,
        payload_result=payload_result,
        embedding_exchange=embedding_exchange,
        embedding_result=embedding_result,
        cp_partition_mode=cp_partition_mode,
        rebuild_microbatch=rebuild_microbatch,
    )
    assignments = _expected_local_assignments(
        authority.plan,
        global_rank=workspace.rank,
        decoder_group_getter=decoder_group_getter,
        decoder_group_ranks_getter=decoder_group_ranks_getter,
    )
    if workspace.rank in authority.plan.decoder_ranks:
        placement = _place_d3_local_decoder_inputs(
            workspace=workspace,
            producer=producer,
            payload_bundle=bundle,
            embedding_exchange=exchange,
        )
        assignments = _canonical_ready_assignments(placement, assignments)
        artifacts = _materialize_d3_decoder_ready_artifacts(
            placement=placement,
            assignments=assignments,
            cp_partition_mode=cp_partition_mode,
            rebuild_microbatch=rebuild_microbatch,
        )
    else:
        artifacts = _LocalDecoderReadyArtifacts((), MappingProxyType({}))
    digest = _decoder_ready_authority_digest(
        global_manifest_digest=authority.global_manifest.digest,
        decoder_plan_digest=_dynamic_iteration_plan_digest(authority),
        payload_bundle_authority_digest=bundle.bundle_authority_digest,
        embedding_route_authority_digest=exchange.route_authority_digest,
        participant_ranks=authority.participant_ranks,
        cp_partition_mode=cp_partition_mode,
    )
    ready = _build_decoder_ready_iteration(
        role=_expected_role(plan=authority.plan, global_rank=workspace.rank),
        authority_digest=digest,
        global_manifest=authority.global_manifest,
        plan=authority.plan,
        global_rank=workspace.rank,
        participant_ranks=authority.participant_ranks,
        cp_partition_mode=cp_partition_mode,
        payload_bundle=bundle,
        payload_tensors=payload_result,
        embedding_exchange=exchange,
        embedding_tensors=embedding_result,
        assignments=assignments,
        artifacts=artifacts,
        plan_digest=_dynamic_iteration_plan_digest(authority),
    )
    return validate_decoder_ready_iteration(
        ready,
        global_manifest=authority.global_manifest,
        plan=authority.plan,
        payload_bundle=bundle,
        payload_tensors=payload_result,
        embedding_exchange=exchange,
        embedding_tensors=embedding_result,
        expected_assignments=assignments,
        authority_digest=digest,
        embedding_width=authority.bridge_width,
        embedding_dtype=authority.bridge_dtype,
        cp_partition_mode=cp_partition_mode,
        plan_digest=_dynamic_iteration_plan_digest(authority),
    )
