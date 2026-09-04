# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 local payload validation and embedding placement contracts."""

from dataclasses import replace
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.dynamic_cp_bridge_transport import prepare_dynamic_bridge_exchange
from megatron.core.mdp.dynamic_cp_transport import prepare_decoder_payload_bundle
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpStateError
from megatron.core.mdp.storage import MdpEmbeddingStorage
from tests.unit_tests.mdp.test_dynamic_cp_d3_workspace import (
    _authority,
    _bridge_sources,
    _payload_sources,
)


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_local_placement")


def _runtime():
    return import_module("megatron.core.mdp.dynamic_cp_runtime")


def _workspace(*, rank=5, participant_ranks=(3, 5, 7)):
    workspace_api = import_module("megatron.core.mdp.dynamic_cp_d3_workspace")
    allocator = DirectBufferAllocator()
    workspace = workspace_api._DynamicIterationWorkspace(
        authority=_authority(participant_ranks=participant_ranks),
        rank=rank,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    return workspace


def _prepared(workspace):
    authority = workspace.authority
    device = workspace.device
    payload = prepare_decoder_payload_bundle(
        authority.payload_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        source_rank_by_lane=authority.source_rank_by_lane,
        participant_ranks=authority.participant_ranks,
        global_rank=workspace.rank,
        local_tensors=_payload_sources(authority, workspace.rank, device),
        buffers_by_dtype=workspace.payload_transport_buffers,
    )
    embedding = prepare_dynamic_bridge_exchange(
        authority.embedding_ledger,
        authority.gradient_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        width=authority.bridge_width,
        dtype=authority.bridge_dtype,
        participant_ranks=authority.participant_ranks,
        global_rank=workspace.rank,
        local_tensors=_bridge_sources(
            authority, authority.embedding_ledger, workspace.rank, device
        ),
        send_buffer=workspace.embedding_transport_buffers[0],
        receive_buffer=workspace.embedding_transport_buffers[1],
    )
    return payload, embedding


def _producer(workspace):
    runtime = _runtime()
    owner = object()
    empty = MappingProxyType({})
    pre_authority = runtime._PreAuthorityDynamicProducer(
        rank_view=SimpleNamespace(global_rank=workspace.rank, lane_id=None),
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs=empty,
        sample_location_by_id=empty,
        owner=owner,
        local_prepare_error=None,
        forward_only=False,
    )
    return runtime._DynamicProducerCarrier(
        authority=workspace.authority,
        pre_authority=pre_authority,
        owner=owner,
        rank_view=pre_authority.rank_view,
        local_manifest=pre_authority.local_manifest,
        source_window=pre_authority.source_window,
        static_plan=pre_authority.static_plan,
        native_item_outputs=empty,
        item_outputs=empty,
        payload_destination_views=workspace.payload_views,
        embedding_destination_views=workspace.embedding_views,
        gradient_destination_views=workspace.gradient_views,
        summed_gradient_destination_views=workspace.summed_gradient_views,
        backward=lambda gradients: gradients,
        cleanup=lambda: None,
    )


def _fill_embedding_receive(embedding):
    expected = {}
    for index, (key, tensor) in enumerate(embedding.received_tensors.items(), start=1):
        tensor.fill_(index)
        expected[key] = tensor.clone()
    return expected


def test_zero_copy_payload_and_microbatch_leaf_placement_preserve_exact_inputs():
    workspace = _workspace(participant_ranks=(7, 3, 5))
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)
    expected = _fill_embedding_receive(embedding)

    try:
        result = _api()._place_d3_local_decoder_inputs(
            workspace=workspace,
            producer=producer,
            payload_bundle=payload,
            embedding_exchange=embedding,
        )

        assert type(result) is _api()._D3LocalPlacement
        assert result.workspace is workspace
        assert result.producer is producer
        assert result.payload_bundle is payload
        assert result.embedding_exchange is embedding
        assert result.payload_destination_views is producer.payload_destination_views
        assert result.embedding_destination_views is producer.embedding_destination_views
        assert result.gradient_destination_views is producer.gradient_destination_views
        assert (
            result.summed_gradient_destination_views is producer.summed_gradient_destination_views
        )
        assert result.payload_destination_views is workspace.payload_views
        assert result.embedding_destination_views is workspace.embedding_views
        for key, source in payload.received_tensors.items():
            destination = result.payload_destination_views[key]
            assert source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()
            assert source.storage_offset() == destination.storage_offset()
        for key, destination in result.embedding_destination_views.items():
            assert torch.equal(destination.detach(), expected[key])
            assert destination.requires_grad and not destination.is_leaf
            assert (
                destination.untyped_storage().data_ptr()
                != embedding.received_tensors[key].untyped_storage().data_ptr()
            )
        for microbatch_id in workspace._embedding_bases:
            leaf = workspace.storage.get_leaf(microbatch_id)
            assert leaf is not None and leaf.requires_grad and leaf.is_leaf
        with pytest.raises(MdpStateError, match="fresh"):
            _api()._place_d3_local_decoder_inputs(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=embedding,
            )
    finally:
        workspace.release()


def test_preflight_is_all_or_nothing_and_repeat_or_preactivation_is_rejected():
    workspace = _workspace()
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)
    leaves = tuple(workspace.embedding_views.items())
    first_key, first_leaf = leaves[0]
    last_key, last_leaf = leaves[-1]
    for _, leaf in leaves:
        leaf.zero_()
    malformed = dict(workspace.embedding_views)
    malformed[last_key] = last_leaf[:-1]
    workspace.embedding_views = MappingProxyType(malformed)
    producer = replace(producer, embedding_destination_views=workspace.embedding_views)

    try:
        with pytest.raises(MdpConfigurationError, match="destination"):
            _api()._place_d3_local_decoder_inputs(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=embedding,
            )
        assert first_key != last_key
        assert all(torch.count_nonzero(leaf) == 0 for _, leaf in leaves)
        assert not workspace._embedding_leaves_activated
    finally:
        workspace.release()

    workspace = _workspace()
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)
    workspace.activate_embedding_leaves()
    try:
        with pytest.raises(MdpStateError, match="fresh"):
            _api()._place_d3_local_decoder_inputs(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=embedding,
            )
    finally:
        workspace.release()


@pytest.mark.parametrize(
    "field",
    (
        "authority",
        "payload_destination_views",
        "embedding_destination_views",
        "gradient_destination_views",
        "summed_gradient_destination_views",
    ),
)
def test_rejects_foreign_producer_authority_or_destination_identity(field):
    workspace = _workspace()
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)
    foreign = MappingProxyType({})
    value = object() if field == "authority" else foreign
    if field == "authority":
        object.__setattr__(producer, field, value)
    else:
        producer = replace(producer, **{field: value})

    try:
        with pytest.raises((MdpConfigurationError, MdpBridgeError), match="exact"):
            _api()._place_d3_local_decoder_inputs(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=embedding,
            )
        assert not workspace._embedding_leaves_activated
    finally:
        workspace.release()


def test_rejects_swapped_or_malformed_d2_carriers_and_released_workspace():
    workspace = _workspace()
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)
    swapped = replace(embedding, receive_buffer=workspace.gradient_transport_buffers[1])

    try:
        with pytest.raises((MdpConfigurationError, MdpBridgeError)):
            _api()._place_d3_local_decoder_inputs(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=swapped,
            )
        with pytest.raises((MdpConfigurationError, MdpBridgeError)):
            _api()._place_d3_local_decoder_inputs(
                workspace=workspace,
                producer=producer,
                payload_bundle=replace(payload, participant_ranks=(3, 7, 5)),
                embedding_exchange=embedding,
            )
    finally:
        workspace.release()

    with pytest.raises(MdpStateError, match="released"):
        _api()._place_d3_local_decoder_inputs(
            workspace=workspace,
            producer=producer,
            payload_bundle=payload,
            embedding_exchange=embedding,
        )


def test_rejects_sealed_d2_carriers_from_a_different_workspace():
    workspace = _workspace()
    foreign_workspace = _workspace()
    producer = _producer(workspace)
    payload, embedding = _prepared(foreign_workspace)

    try:
        with pytest.raises(MdpBridgeError, match="workspace"):
            _api()._place_d3_local_decoder_inputs(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=embedding,
            )
        assert not workspace._embedding_leaves_activated
    finally:
        workspace.release()
        foreign_workspace.release()


def test_typed_result_rejects_replaced_bound_capability_mapping():
    workspace = _workspace()
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)
    _api()._place_d3_local_decoder_inputs(
        workspace=workspace, producer=producer, payload_bundle=payload, embedding_exchange=embedding
    )
    foreign = MappingProxyType({})
    producer = replace(
        producer,
        payload_destination_views=foreign,
        embedding_destination_views=foreign,
        gradient_destination_views=foreign,
        summed_gradient_destination_views=foreign,
    )

    try:
        with pytest.raises(MdpBridgeError, match="exact bound capabilities"):
            _api()._D3LocalPlacement(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=embedding,
                payload_destination_views=foreign,
                embedding_destination_views=foreign,
                gradient_destination_views=foreign,
                summed_gradient_destination_views=foreign,
            )
    finally:
        workspace.release()


def test_typed_result_rejects_valid_d2_carriers_from_a_foreign_workspace():
    workspace = _workspace()
    foreign_workspace = _workspace()
    producer = _producer(workspace)
    payload, embedding = _prepared(foreign_workspace)
    workspace.activate_embedding_leaves()

    try:
        with pytest.raises(MdpBridgeError, match="workspace"):
            _api()._D3LocalPlacement(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=embedding,
                payload_destination_views=producer.payload_destination_views,
                embedding_destination_views=producer.embedding_destination_views,
                gradient_destination_views=producer.gradient_destination_views,
                summed_gradient_destination_views=producer.summed_gradient_destination_views,
            )
    finally:
        workspace.release()
        foreign_workspace.release()


def test_typed_result_rejects_fresh_or_released_workspace():
    workspace = _workspace()
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)

    try:
        with pytest.raises(MdpStateError, match="active placed workspace"):
            _api()._D3LocalPlacement(
                workspace=workspace,
                producer=producer,
                payload_bundle=payload,
                embedding_exchange=embedding,
                payload_destination_views=producer.payload_destination_views,
                embedding_destination_views=producer.embedding_destination_views,
                gradient_destination_views=producer.gradient_destination_views,
                summed_gradient_destination_views=producer.summed_gradient_destination_views,
            )
    finally:
        workspace.release()

    with pytest.raises(MdpStateError, match="active placed workspace"):
        _api()._D3LocalPlacement(
            workspace=workspace,
            producer=producer,
            payload_bundle=payload,
            embedding_exchange=embedding,
            payload_destination_views=producer.payload_destination_views,
            embedding_destination_views=producer.embedding_destination_views,
            gradient_destination_views=producer.gradient_destination_views,
            summed_gradient_destination_views=producer.summed_gradient_destination_views,
        )


def test_zero_route_rank_activates_empty_leaf_set_once():
    workspace = _workspace(rank=9, participant_ranks=(3, 5, 7, 9))
    producer = _producer(workspace)
    payload, embedding = _prepared(workspace)

    try:
        result = _api()._place_d3_local_decoder_inputs(
            workspace=workspace,
            producer=producer,
            payload_bundle=payload,
            embedding_exchange=embedding,
        )
        assert not result.embedding_destination_views
        assert workspace._embedding_leaves_activated
    finally:
        workspace.release()
