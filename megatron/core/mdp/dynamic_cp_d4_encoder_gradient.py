# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private Gate3 encoder-only decoder-gradient route owner."""

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as _replay
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey, dynamic_bridge_split_sizes
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    PreparedDynamicBridgeExchange,
    _execute_validated_dynamic_bridge_exchange,
    prepare_dynamic_bridge_exchange,
    validate_prepared_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_execution import DecoderMicrobatchKey
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord

__all__ = ()

_ACTIVE = object()
_RETIRED = object()
_RECEIPT_SEAL = object()
_OWNER_SEAL = object()
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_RECEIPTS: dict[int, tuple[Any, ...]] = {}


def _add_cleanup_note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


@dataclass(frozen=True, slots=True)
class _D4EncoderOnlyGradientReceipt:
    """Sealed Gate3 result retaining exact completion and received views."""

    authority: _DynamicIterationAuthority = field(compare=False, repr=False)
    completion: _replay._D4FixedDecoderCompletion = field(compare=False, repr=False)
    exchange: PreparedDynamicBridgeExchange = field(compare=False, repr=False)
    received_tensors: Mapping[DynamicBridgeKey, Tensor] = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self._seal is not _RECEIPT_SEAL:
            raise MdpConfigurationError("MDP: encoder-only gradient receipt is privately minted.")


@dataclass(frozen=True, slots=True)
class _GradientCandidate:
    exchange: PreparedDynamicBridgeExchange = field(compare=False, repr=False)
    receipt: _D4EncoderOnlyGradientReceipt = field(compare=False, repr=False)
    transport_buffers: tuple[Tensor, Tensor] = field(compare=False, repr=False)
    group: Any = field(compare=False, repr=False)


def _project_leaf_gradients(
    *,
    records: tuple[MdpMicrobatchRecord, ...],
    embedding_leaves: Mapping[DecoderMicrobatchKey, Tensor],
    authority: _DynamicIterationAuthority,
    global_rank: int,
) -> Mapping[DynamicBridgeKey, Tensor]:
    """Project native replay leaf grads into exact manifest item-route views."""
    if type(records) is not tuple or type(embedding_leaves) is not _MAPPING_PROXY_TYPE:
        raise MdpStateError(
            "MDP: encoder-only gradient route retains exact replay records and leaves."
        )
    item_gradients = {}
    expected_leaf_keys = []
    for index, record in enumerate(records):
        if type(record) is not MdpMicrobatchRecord or record.microbatch_id != index:
            raise MdpStateError("MDP: encoder-only gradient route retains ordered replay records.")
        if type(record.vision_items) is not tuple or any(
            type(item) is not MdpMicrobatchVisionRecord for item in record.vision_items
        ):
            raise MdpStateError(
                "MDP: encoder-only gradient route retains exact replay item records."
            )
        key = DecoderMicrobatchKey(index)
        if not record.vision_items:
            if key in embedding_leaves:
                raise MdpStateError("MDP: text-only replay retains no decoder embedding leaf.")
            continue
        expected_leaf_keys.append(key)
        try:
            leaf = embedding_leaves[key]
        except KeyError as error:
            raise MdpStateError(
                "MDP: encoder-only gradient route covers every vision replay leaf."
            ) from error
        gradient = leaf.grad
        expected_rows = sum(item.output_rows for item in record.vision_items)
        if (
            type(leaf) is not Tensor
            or tuple(leaf.shape) != (expected_rows, authority.bridge_width)
            or leaf.dtype != authority.bridge_dtype
            or not leaf.is_contiguous()
            or not leaf.is_leaf
            or not leaf.requires_grad
            or leaf.grad_fn is not None
            or type(gradient) is not Tensor
            or tuple(gradient.shape) != tuple(leaf.shape)
            or gradient.dtype != authority.bridge_dtype
            or gradient.device != leaf.device
            or not gradient.is_contiguous()
            or gradient.requires_grad
            or gradient.grad_fn is not None
        ):
            raise MdpStateError(
                "MDP: encoder-only native replay produces exact detached leaf gradients."
            )
        offset = 0
        for item in record.vision_items:
            if item.global_item_id in item_gradients:
                raise MdpPlanError(
                    "MDP: encoder-only gradient route visits every manifest item once."
                )
            view = gradient.narrow(0, offset, item.output_rows)
            item_gradients[item.global_item_id] = view
            offset += item.output_rows
        if offset != gradient.shape[0]:
            raise MdpStateError("MDP: encoder-only gradient views cover each replay leaf exactly.")
    if tuple(embedding_leaves) != tuple(expected_leaf_keys):
        raise MdpStateError("MDP: encoder-only gradient route has exact replay leaf coverage.")
    manifest_item_ids = tuple(item.item_id for item in authority.global_manifest.items)
    if tuple(item_gradients) != tuple(
        item_id for item_id in manifest_item_ids if item_id in item_gradients
    ):
        raise MdpPlanError("MDP: encoder-only gradient views preserve manifest item order.")
    expected_entries = tuple(
        entry for entry in authority.gradient_ledger.entries if entry.src_global_rank == global_rank
    )
    try:
        local = MappingProxyType(
            {entry.key: item_gradients[entry.key.item_id] for entry in expected_entries}
        )
    except KeyError as error:
        raise MdpPlanError(
            "MDP: encoder-only gradient views cover exact local route authority."
        ) from error
    if tuple(local) != tuple(entry.key for entry in expected_entries):
        raise MdpPlanError("MDP: encoder-only gradient views follow exact route order.")
    return local


class _D4EncoderGradientRouteOwner:
    """Registered owner of Gate3 receipt and all retained replay resources."""

    __slots__ = (
        "__weakref__",
        "authority",
        "binding",
        "receipt",
        "completion",
        "records",
        "embedding_leaves",
        "text_only",
        "is_selected",
        "_runtime",
        "_trusted",
        "_state",
        "_prepared_reference",
        "_prepared_handoff_reference",
    )

    def __init__(self, trusted: tuple[Any, ...], seal: object) -> None:
        if seal is not _OWNER_SEAL:
            raise MdpConfigurationError(
                "MDP: encoder-only gradient route owner is privately minted."
            )
        (
            self._runtime,
            self.authority,
            self.binding,
            self.receipt,
            self.completion,
            self.records,
            self.embedding_leaves,
        ) = trusted[:7]
        self.text_only, self.is_selected = trusted[12:14]
        self._trusted = trusted
        self._state = _ACTIVE
        self._prepared_reference = None
        self._prepared_handoff_reference = None

    def _prepare_from(self, handoff: _replay._D4FixedDecoderGradientHandoff) -> None:
        entry = _replay._ACTIVE_GRADIENT_HANDOFFS.get(id(handoff))
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(self._runtime))
        if (
            entry is None
            or entry[0]() is not handoff
            or entry[-1] is not True
            or runtime_entry is None
            or runtime_entry[0] is not self._runtime
            or runtime_entry[1]() is not handoff
        ):
            raise MdpStateError("MDP: encoder-only gradient route replaces exact consumed handoff.")
        owner_identity = id(self)
        runtime_identity = id(self._runtime)

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            current = _ACTIVE_OWNERS.get(owner_identity)
            if current is not None and current[0] is reference:
                del _ACTIVE_OWNERS[owner_identity]
            current_runtime = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
            if current_runtime is not None and current_runtime[1] is reference:
                del _replay._forward._ACTIVE_RUNTIME_OWNERS[runtime_identity]

        self._prepared_reference = weakref.ref(self, retire)
        self._prepared_handoff_reference = weakref.ref(handoff)

    def _activate_prepared(self, handoff: _replay._D4FixedDecoderGradientHandoff) -> None:
        reference = self._prepared_reference
        runtime_identity = id(self._runtime)
        _ACTIVE_OWNERS[id(self)] = (reference, *self._trusted)
        _ACTIVE_RECEIPTS[id(self.receipt)] = (
            reference,
            self.receipt,
            self.authority,
            self.completion,
            self.receipt.exchange,
            self.receipt.received_tensors,
        )
        _replay._forward._ACTIVE_RUNTIME_OWNERS[runtime_identity] = (self._runtime, reference)
        completion_entry = _replay._ACTIVE_COMPLETIONS[id(self.completion)]
        _replay._ACTIVE_COMPLETIONS[id(self.completion)] = (reference, *completion_entry[1:])
        _replay._ACTIVE_GRADIENT_HANDOFFS.pop(id(handoff))
        _replay._RETIRED_GRADIENT_HANDOFFS[id(handoff)] = self._prepared_handoff_reference
        handoff._state = _replay._RETIRED
        handoff._consumed = True
        handoff.authority = None
        handoff.binding = None
        handoff.records = ()
        handoff.embedding_leaves = MappingProxyType({})
        handoff.completion = None
        handoff.text_only = None
        handoff.is_selected = None
        handoff._runtime = None
        handoff._trusted = ()
        self._prepared_reference = None
        self._prepared_handoff_reference = None

    def require(self) -> "_D4EncoderGradientRouteOwner":
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_OWNERS.get(id(self))
            if retired is not None and retired() is self:
                raise MdpStateError("MDP: encoder-only gradient route owner is retired.")
            raise MdpStateError("MDP: encoder-only gradient route owner is the exact active owner.")
        current = (
            self._runtime,
            self.authority,
            self.binding,
            self.receipt,
            self.completion,
            self.records,
            self.embedding_leaves,
            *self._trusted[7:12],
            self.text_only,
            self.is_selected,
        )
        if self._state is not _ACTIVE or any(
            actual is not expected for actual, expected in zip(current, entry[1:], strict=True)
        ):
            raise MdpStateError("MDP: encoder-only gradient route owner retains sealed resources.")
        receipt_entry = _ACTIVE_RECEIPTS.get(id(self.receipt))
        completion_entry = _replay._ACTIVE_COMPLETIONS.get(id(self.completion))
        if (
            receipt_entry is None
            or receipt_entry[0]() is not self
            or receipt_entry[1] is not self.receipt
            or self.receipt.authority is not self.authority
            or self.receipt.completion is not self.completion
            or self.receipt.exchange is not receipt_entry[4]
            or self.receipt.received_tensors is not receipt_entry[5]
            or self.receipt.received_tensors is not self.receipt.exchange.received_tensors
            or self.receipt._seal is not _RECEIPT_SEAL
            or completion_entry is None
            or completion_entry[0]() is not self
            or completion_entry[1] is not self.completion
            or completion_entry[2] is not self.authority
            or self.completion._owner is not self._trusted[11]
            or _replay._tensor_descriptor(completion_entry[3]) != completion_entry[4]
        ):
            raise MdpStateError(
                "MDP: encoder-only gradient route retains exact receipt and completion."
            )
        return self

    def abort(self, primary_error: BaseException | None = None) -> None:
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError(
                "MDP: encoder-only gradient route abort error is an exception."
            )
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        trusted = entry[1:]
        integrity_error = None
        try:
            current = (
                object.__getattribute__(self, "_runtime"),
                object.__getattribute__(self, "authority"),
                object.__getattribute__(self, "binding"),
                object.__getattribute__(self, "receipt"),
                object.__getattribute__(self, "completion"),
                object.__getattribute__(self, "records"),
                object.__getattribute__(self, "embedding_leaves"),
                *trusted[7:12],
                object.__getattribute__(self, "text_only"),
                object.__getattribute__(self, "is_selected"),
            )
            if object.__getattribute__(self, "_state") is not _ACTIVE or any(
                actual is not expected for actual, expected in zip(current, trusted, strict=True)
            ):
                integrity_error = MdpStateError(
                    "MDP: encoder-only gradient route owner retains sealed resources."
                )
        except BaseException as error:
            integrity_error = error
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: encoder-only gradient route was aborted.")
        )
        _ACTIVE_OWNERS.pop(id(self))
        _RETIRED_OWNERS[id(self)] = weakref.ref(self)
        runtime = trusted[0]
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            del _replay._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)]
        _ACTIVE_RECEIPTS.pop(id(trusted[3]), None)
        _replay._ACTIVE_COMPLETIONS.pop(id(trusted[4]), None)
        self._state = _RETIRED
        self.authority = None
        self.binding = None
        self.receipt = None
        self.completion = None
        self.records = ()
        self.embedding_leaves = MappingProxyType({})
        self.text_only = None
        self.is_selected = None
        self._runtime = None
        self._trusted = ()
        self._prepared_reference = None
        self._prepared_handoff_reference = None
        if integrity_error is not None:
            _add_cleanup_note(primary, "encoder-only gradient route integrity validation failed.")
        handoff_resources, leaf_bases, transport_buffers, operations = trusted[7:11]
        binding_owner, predecessor_buffers, _predecessor_operations, forward_handle = (
            handoff_resources
        )
        if forward_handle is not None:
            try:
                forward_handle.release_forward_only()
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed encoder graph release error: {error!r}")
        if binding_owner is not None:
            try:
                binding_owner.restore(primary)
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed encoder CP restore error: {error!r}")
        for buffer in (*transport_buffers, *leaf_bases, *predecessor_buffers):
            try:
                operations.release(buffer)
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed encoder gradient buffer release error: {error!r}"
                )


def run_repeated_d4_encoder_gradient(
    handoff: _replay._D4FixedDecoderGradientHandoff,
    authority: _DynamicIterationAuthority,
    completion: _replay._D4FixedDecoderCompletion,
    *,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
    byte_generator: Callable[[int], Any] | None = None,
) -> _D4EncoderGradientRouteOwner:
    """Authorize Gate3, then execute one validated encoder-gradient A2A."""
    if type(handoff) is not _replay._D4FixedDecoderGradientHandoff:
        raise MdpConfigurationError(
            "MDP: encoder-only gradient route uses an exact replay handoff."
        )
    handoff_entry = _replay._ACTIVE_GRADIENT_HANDOFFS.get(id(handoff))
    if handoff_entry is None or handoff_entry[0]() is not handoff:
        handoff.require()
    if authority is not handoff_entry[2] or completion is not handoff_entry[8]:
        raise MdpStateError("MDP: encoder-only gradient route uses exact authority and completion.")
    binding = handoff_entry[3]
    handoff_operations = handoff_entry[6][2]
    candidate = None
    owner = None
    prepare_started = False

    def prepare() -> _GradientCandidate:
        nonlocal candidate, owner, prepare_started
        if prepare_started:
            raise MdpStateError("MDP: encoder-only gradient preparation is one-shot.")
        prepare_started = True
        handoff.consume(authority, completion)
        entry = _replay._ACTIVE_GRADIENT_HANDOFFS[id(handoff)]
        trusted = entry[1:-1]
        runtime = trusted[0]
        handoff_resources = trusted[5]
        operations = handoff_resources[2]
        if not callable(all_to_all_single):
            raise MdpConfigurationError("MDP: encoder-only gradient all_to_all_single is callable.")
        binding_authority = _validate_repeated_d4_group_binding(binding)
        group = binding_authority._domain_group
        if tuple(binding_authority._group_ranks_getter(group)) != binding.domain_ranks:
            raise MdpStateError(
                "MDP: encoder-only gradient route retains native domain group order."
            )
        local_tensors = _project_leaf_gradients(
            records=trusted[3],
            embedding_leaves=trusted[4],
            authority=authority,
            global_rank=binding.global_rank,
        )
        input_splits, output_splits = dynamic_bridge_split_sizes(
            authority.gradient_ledger,
            reverse_ledger=authority.embedding_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            width=authority.bridge_width,
            dtype=authority.bridge_dtype,
            participant_ranks=authority.participant_ranks,
            global_rank=binding.global_rank,
        )
        buffers = []
        try:
            for elements, name in ((sum(input_splits), "send"), (sum(output_splits), "receive")):
                buffer = operations.acquire(
                    rows=elements,
                    width=0,
                    dtype=authority.bridge_dtype,
                    device=runtime.device,
                    tag=f"dynamic_cp_gate3_encoder_gradient_{name}",
                )
                buffers.append(buffer)
            exchange = prepare_dynamic_bridge_exchange(
                authority.gradient_ledger,
                authority.embedding_ledger,
                plan=authority.plan,
                global_manifest=authority.global_manifest,
                producer_rank_by_item=authority.producer_rank_by_item,
                output_rows_by_item=authority.output_rows_by_item,
                width=authority.bridge_width,
                dtype=authority.bridge_dtype,
                participant_ranks=authority.participant_ranks,
                global_rank=binding.global_rank,
                local_tensors=local_tensors,
                send_buffer=buffers[0],
                receive_buffer=buffers[1],
            )
            if type(exchange) is not PreparedDynamicBridgeExchange:
                raise MdpBridgeError(
                    "MDP: encoder-only gradient preparation returns exact exchange."
                )
            validate_prepared_dynamic_bridge_exchange(exchange)
            if (
                exchange.phase is not BridgePhase.GRADIENT
                or exchange.global_rank != binding.global_rank
                or exchange.participant_ranks != binding.domain_ranks
            ):
                raise MdpBridgeError("MDP: encoder-only gradient exchange matches rank authority.")
            receipt = _D4EncoderOnlyGradientReceipt(
                authority, completion, exchange, exchange.received_tensors, _RECEIPT_SEAL
            )
            prepared_candidate = _GradientCandidate(exchange, receipt, tuple(buffers), group)
            owner_trusted = (
                runtime,
                authority,
                binding,
                receipt,
                completion,
                trusted[3],
                trusted[4],
                handoff_resources,
                trusted[6],
                tuple(buffers),
                operations,
                trusted[8],
                trusted[10],
                trusted[11],
            )
            owner = _D4EncoderGradientRouteOwner(owner_trusted, _OWNER_SEAL)
            owner._prepare_from(handoff)
            candidate = prepared_candidate
            return candidate
        except BaseException as error:
            for buffer in reversed(buffers):
                try:
                    operations.release(buffer)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        error,
                        f"suppressed encoder gradient preparation cleanup error: {cleanup_error!r}",
                    )
            raise

    def authorize(prepared: _GradientCandidate) -> _GradientCandidate:
        if prepared is not candidate or type(prepared) is not _GradientCandidate:
            raise MdpTaskFatalError("MDP: encoder-only Gate3 runner returns exact candidate.")
        return prepared

    try:
        result = run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=3,
            prepare=prepare,
            domain_collective=authorize,
            byte_generator=byte_generator,
        )
        if result is not candidate or type(result) is not _GradientCandidate:
            raise MdpTaskFatalError("MDP: encoder-only Gate3 runner returns exact result.")
        try:
            received = _execute_validated_dynamic_bridge_exchange(
                candidate.exchange, group=candidate.group, all_to_all_single=all_to_all_single
            )
            if received is not candidate.receipt.received_tensors:
                raise MdpBridgeError("MDP: encoder-only gradient A2A returns exact received views.")
            owner._activate_prepared(handoff)
        except BaseException as error:
            if type(error) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: physical encoder-gradient route failed after Gate3 final WORLD."
            ) from error
        return owner
    except BaseException as error:
        if candidate is not None:
            for buffer in reversed(candidate.transport_buffers):
                try:
                    handoff_operations.release(buffer)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        error,
                        f"suppressed encoder gradient buffer cleanup error: {cleanup_error!r}",
                    )
        try:
            handoff.abort(error)
        except BaseException as cleanup_error:
            _add_cleanup_note(
                error, f"suppressed encoder gradient handoff cleanup error: {cleanup_error!r}"
            )
        raise
