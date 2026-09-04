# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private gate-4 encoder-completion preparation binding for D3."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from megatron.core.mdp.dynamic_cp_bridge_transport import (
    _capture_authority as _capture_dynamic_bridge_authority,
)
from megatron.core.mdp.dynamic_cp_d3_gradient_gate_binding import _views_match
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderGradientReceipt,
    DecoderGradientReceiptLifecycle,
    _begin_decoder_gradient_receipt_lifecycle,
    _capture_decoder_gradient_receipt_authority,
    _capture_prepared_decoder_gradient_authority,
    _consume_decoder_gradient_receipt,
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
    _retire_decoder_gradient_receipt_lifecycle,
    _validate_decoder_gradient_receipt,
    _validate_decoder_gradient_receipt_lifecycle,
    validate_prepared_decoder_gradient_exchange,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)

__all__ = ()

_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_PENDING_BINDING_SEALS: dict[object, tuple[int, int]] = {}


def _tensor_descriptors(mapping: Mapping) -> tuple:
    if not isinstance(mapping, Mapping):
        raise MdpConfigurationError("MDP: D3 encoder completion tensors form a mapping.")
    if any(not isinstance(tensor, torch.Tensor) for tensor in mapping.values()):
        raise MdpConfigurationError("MDP: D3 encoder completion values are tensors.")
    return tuple(
        (
            id(key),
            id(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
        )
        for key, tensor in mapping.items()
    )


@dataclass(frozen=True)
class _PreparedD3EncoderCompletionAuthority:
    carrier_identity: int
    authority_identity: int
    producer_identity: int
    workspace_identity: int
    receipt_identity: int
    lifecycle_identity: int
    aggregated_mapping_identity: int
    native_completion_identity: int
    iteration_nonce: bytes
    cp_partition_mode: str
    receipt_authority: Any
    prepared_authority: Any
    exchange_authority: Any
    aggregated_descriptors: tuple


@dataclass(frozen=True, slots=True)
class _PreparedD3EncoderCompletion:
    """Sealed local result ready for a later gate-5 backward binding."""

    authority: _DynamicIterationAuthority = field(compare=False, repr=False)
    producer: _DynamicProducerCarrier = field(compare=False, repr=False)
    workspace: Any = field(compare=False, repr=False)
    receipt: DecoderGradientReceipt = field(compare=False, repr=False)
    iteration_nonce: bytes
    cp_partition_mode: str
    lifecycle: DecoderGradientReceiptLifecycle = field(compare=False, repr=False)
    aggregated: Mapping = field(compare=False, repr=False)
    native_completion: Any = field(compare=False, repr=False)
    _authority: _PreparedD3EncoderCompletionAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


def _capture_prepared_authority(
    prepared: _PreparedD3EncoderCompletion,
) -> _PreparedD3EncoderCompletionAuthority:
    return _PreparedD3EncoderCompletionAuthority(
        carrier_identity=id(prepared),
        authority_identity=id(prepared.authority),
        producer_identity=id(prepared.producer),
        workspace_identity=id(prepared.workspace),
        receipt_identity=id(prepared.receipt),
        lifecycle_identity=id(prepared.lifecycle),
        aggregated_mapping_identity=id(prepared.aggregated),
        native_completion_identity=id(prepared.native_completion),
        iteration_nonce=prepared.iteration_nonce,
        cp_partition_mode=prepared.cp_partition_mode,
        receipt_authority=_capture_decoder_gradient_receipt_authority(prepared.receipt),
        prepared_authority=_capture_prepared_decoder_gradient_authority(prepared.receipt.prepared),
        exchange_authority=_capture_dynamic_bridge_authority(prepared.receipt.prepared.exchange),
        aggregated_descriptors=_tensor_descriptors(prepared.aggregated),
    )


def _validate_prepared_d3_encoder_completion(
    prepared: Any,
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    authority: _DynamicIterationAuthority,
    producer: _DynamicProducerCarrier,
    cp_partition_mode: str,
) -> _PreparedD3EncoderCompletion:
    """Validate one exact, still-active gate-4 preparation capability."""
    if type(workspace_owner) is not _D3WorkspaceBindingOwner:
        raise MdpConfigurationError(
            "MDP: D3 encoder completion validation uses its exact workspace owner."
        )
    if (
        type(authority) is not _DynamicIterationAuthority
        or type(producer) is not _DynamicProducerCarrier
    ):
        raise MdpConfigurationError(
            "MDP: D3 encoder completion validation uses exact active inputs."
        )
    if type(prepared) is not _PreparedD3EncoderCompletion:
        raise MdpConfigurationError("MDP: D3 encoder completion has its exact carrier type.")
    if type(prepared._authority) is not _PreparedD3EncoderCompletionAuthority:
        raise MdpBridgeError("MDP: D3 encoder completion has a private authority seal.")
    if (
        type(cp_partition_mode) is not str
        or cp_partition_mode not in ("contiguous", "zigzag")
        or prepared.cp_partition_mode != cp_partition_mode
    ):
        raise MdpConfigurationError("MDP: D3 encoder completion has its exact CP mode.")
    if prepared.authority is not authority or prepared.producer is not producer:
        raise MdpBridgeError("MDP: D3 encoder completion retains exact active identities.")
    workspace = workspace_owner.require_workspace(authority)
    if prepared.workspace is not workspace or workspace._released:
        raise MdpStateError("MDP: D3 encoder completion retains its exact active workspace.")
    _validate_decoder_gradient_receipt_lifecycle(prepared.lifecycle, expected_state="retired")
    gradient_prepared = validate_prepared_decoder_gradient_exchange(
        prepared.receipt.prepared,
        global_manifest=authority.global_manifest,
        plan=authority.plan,
        global_rank=workspace.rank,
        participant_ranks=authority.participant_ranks,
        embedding_width=authority.bridge_width,
        embedding_dtype=authority.bridge_dtype,
        cp_partition_mode=cp_partition_mode,
    )
    receipt_authority = _capture_decoder_gradient_receipt_authority(prepared.receipt)
    prepared_authority = _capture_prepared_decoder_gradient_authority(gradient_prepared)
    exchange_authority = _capture_dynamic_bridge_authority(gradient_prepared.exchange)
    if (
        prepared.receipt.iteration_nonce != prepared.iteration_nonce
        or prepared.receipt._consumed_lifecycle_identity != id(prepared.lifecycle)
        or receipt_authority != prepared.receipt._authority
        or receipt_authority != prepared._authority.receipt_authority
        or prepared_authority != prepared._authority.prepared_authority
        or exchange_authority != prepared._authority.exchange_authority
    ):
        raise MdpBridgeError("MDP: D3 encoder completion matches its private authority seal.")
    if (
        producer.gradient_destination_views is not workspace.gradient_views
        or producer.summed_gradient_destination_views is not workspace.summed_gradient_views
        or tuple(prepared.aggregated) != tuple(workspace.summed_gradient_views)
        or any(
            prepared.aggregated[key] is not workspace.summed_gradient_views[key]
            for key in prepared.aggregated
        )
    ):
        raise MdpBridgeError("MDP: D3 encoder completion retains exact aggregated destinations.")
    if _capture_prepared_authority(prepared) != prepared._authority:
        raise MdpBridgeError("MDP: D3 encoder completion matches its private authority seal.")
    return prepared


def _validate_static_dependencies(*, workspace_owner: Any, cp_partition_mode: Any) -> None:
    if type(workspace_owner) is not _D3WorkspaceBindingOwner:
        raise MdpConfigurationError(
            "MDP: D3 encoder completion binding uses its exact workspace owner."
        )
    if type(cp_partition_mode) is not str or cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError(
            "MDP: D3 encoder completion binding CP partition mode is supported."
        )


def _validate_destinations(authority: Any, workspace: Any) -> tuple:
    destinations = workspace.summed_gradient_views
    expected_keys = tuple(
        item.item_id
        for item in authority.global_manifest.items
        if authority.producer_rank_by_item[item.item_id] == workspace.rank
    )
    if not isinstance(destinations, _MAPPING_PROXY_TYPE) or tuple(destinations) != expected_keys:
        raise MdpPlanError(
            "MDP: D3 encoder completion has immutable ordered producer destinations."
        )
    for key in expected_keys:
        tensor = destinations[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != (authority.output_rows_by_item[key], authority.bridge_width)
            or tensor.dtype != authority.bridge_dtype
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not tensor.is_contiguous()
        ):
            raise MdpConfigurationError(
                "MDP: D3 encoder completion destination matches workspace schema."
            )
    return expected_keys


def _validate_receipt_workspace(
    authority: _DynamicIterationAuthority, workspace: Any, receipt: Any, cp_partition_mode: str
) -> DecoderGradientReceipt:
    if type(receipt) is not DecoderGradientReceipt:
        raise MdpConfigurationError("MDP: D3 encoder completion uses an exact gradient receipt.")
    receipt = _validate_decoder_gradient_receipt(
        receipt,
        global_manifest=authority.global_manifest,
        plan=authority.plan,
        embedding_ledger=authority.embedding_ledger,
        gradient_ledger=authority.gradient_ledger,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        global_rank=workspace.rank,
        participant_ranks=authority.participant_ranks,
        embedding_width=authority.bridge_width,
        embedding_dtype=authority.bridge_dtype,
        cp_partition_mode=cp_partition_mode,
        iteration_nonce=receipt.iteration_nonce,
    )
    buffers = workspace.gradient_transport_buffers
    exchange = receipt.prepared.exchange
    if (
        type(buffers) is not tuple
        or len(buffers) != 2
        or exchange.send_buffer is not buffers[0]
        or exchange.receive_buffer is not buffers[1]
    ):
        raise MdpBridgeError(
            "MDP: D3 encoder completion receipt retains exact workspace gradient buffers."
        )
    if not _views_match(receipt.received_tensors, workspace.gradient_views):
        raise MdpBridgeError(
            "MDP: D3 encoder completion receipt retains exact workspace gradient views."
        )
    return receipt


@dataclass(frozen=True, slots=True)
class _D3EncoderCompletionPreparationBinding:
    """Aggregate one gradient receipt and prepare opaque native completion."""

    workspace_owner: _D3WorkspaceBindingOwner = field(compare=False, repr=False)
    cp_partition_mode: str
    _factory_seal: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _D3EncoderCompletionPreparationBinding:
            raise MdpStateError("MDP: D3 encoder completion binding is minted by its factory.")
        _validate_static_dependencies(
            workspace_owner=self.workspace_owner, cp_partition_mode=self.cp_partition_mode
        )
        fingerprint = _PENDING_BINDING_SEALS.pop(self._factory_seal, None)
        if fingerprint != (id(self.workspace_owner), id(self.cp_partition_mode)):
            raise MdpStateError("MDP: D3 encoder completion binding is minted by its factory.")

    def __call__(self, authority: Any, producer: Any, receipt: Any, /) -> Any:
        if type(authority) is not _DynamicIterationAuthority:
            raise MdpConfigurationError(
                "MDP: D3 encoder completion uses exact iteration authority."
            )
        workspace = self.workspace_owner.require_workspace(authority)
        if workspace.authority is not authority or workspace._released:
            raise MdpStateError("MDP: D3 encoder completion requires its exact active workspace.")
        if (
            type(workspace.rank) is not int
            or workspace.rank != self.workspace_owner._rank
            or workspace.rank not in authority.participant_ranks
            or not isinstance(workspace.device, torch.device)
            or workspace.device.type != "cuda"
            or workspace.device != self.workspace_owner._device
        ):
            raise MdpConfigurationError(
                "MDP: D3 encoder completion workspace has exact rank and CUDA device."
            )
        if type(producer) is not _DynamicProducerCarrier or producer.authority is not authority:
            raise MdpBridgeError(
                "MDP: D3 encoder completion retains exact authority-bound producer."
            )
        if (
            producer.gradient_destination_views is not workspace.gradient_views
            or producer.summed_gradient_destination_views is not workspace.summed_gradient_views
        ):
            raise MdpBridgeError(
                "MDP: D3 encoder completion retains exact workspace gradient views."
            )
        destinations = workspace.summed_gradient_views
        expected_keys = _validate_destinations(authority, workspace)
        receipt = _validate_receipt_workspace(authority, workspace, receipt, self.cp_partition_mode)

        lifecycle = _begin_decoder_gradient_receipt_lifecycle(receipt.iteration_nonce)
        aggregated = _consume_decoder_gradient_receipt(
            lifecycle,
            receipt,
            global_manifest=authority.global_manifest,
            plan=authority.plan,
            embedding_ledger=authority.embedding_ledger,
            gradient_ledger=authority.gradient_ledger,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            global_rank=workspace.rank,
            participant_ranks=authority.participant_ranks,
            embedding_width=authority.bridge_width,
            embedding_dtype=authority.bridge_dtype,
            cp_partition_mode=self.cp_partition_mode,
            destination_tensors=destinations,
        )
        validation_error = None
        try:
            if tuple(aggregated) != expected_keys or any(
                aggregated[key] is not destinations[key] for key in expected_keys
            ):
                raise MdpBridgeError(
                    "MDP: D3 encoder completion preserves exact aggregated destinations."
                )
        except BaseException as error:
            validation_error = error
        try:
            _retire_decoder_gradient_receipt_lifecycle(lifecycle)
        except BaseException as error:
            if validation_error is not None:
                try:
                    validation_error.add_note(
                        f"suppressed D3 encoder completion retirement error: {error!r}"
                    )
                except BaseException:
                    pass
            else:
                raise
        if validation_error is not None:
            raise validation_error

        native_completion = producer.backward(aggregated)
        prepared = _PreparedD3EncoderCompletion(
            authority=authority,
            producer=producer,
            workspace=workspace,
            receipt=receipt,
            iteration_nonce=receipt.iteration_nonce,
            cp_partition_mode=self.cp_partition_mode,
            lifecycle=lifecycle,
            aggregated=aggregated,
            native_completion=native_completion,
        )
        object.__setattr__(prepared, "_authority", _capture_prepared_authority(prepared))
        return prepared


def _make_d3_encoder_completion_preparation_binding(
    *, workspace_owner: _D3WorkspaceBindingOwner, cp_partition_mode: str
) -> _D3EncoderCompletionPreparationBinding:
    """Mint one immutable, reusable local gate-4 preparation callback."""
    _validate_static_dependencies(
        workspace_owner=workspace_owner, cp_partition_mode=cp_partition_mode
    )
    token = object()
    _PENDING_BINDING_SEALS[token] = (id(workspace_owner), id(cp_partition_mode))
    try:
        return _D3EncoderCompletionPreparationBinding(
            workspace_owner=workspace_owner,
            cp_partition_mode=cp_partition_mode,
            _factory_seal=token,
        )
    except BaseException:
        _PENDING_BINDING_SEALS.pop(token, None)
        raise
