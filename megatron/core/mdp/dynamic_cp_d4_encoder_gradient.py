# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private Gate3 encoder-only decoder-gradient route owner."""

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp import dynamic_cp_d4_dynamic_decoder_replay as _dynamic_replay
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
_LIFECYCLE_SEAL = object()
_PROVENANCE_SEAL = object()
_COMPLETION_ESCROW_SEAL = object()
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
    completion: Any = field(compare=False, repr=False)
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


@dataclass(frozen=True, slots=True)
class _D4ReplayGradientProvenance:
    mode: object = field(compare=False, repr=False)
    predecessor: Any = field(compare=False, repr=False)
    predecessor_entry: Any = field(compare=False, repr=False)
    completion_lifecycle: Any = field(compare=False, repr=False)
    token: Tensor = field(compare=False, repr=False)
    token_descriptor: tuple[Any, ...] = field(compare=False, repr=False)
    acquire: Callable[..., Tensor] = field(compare=False, repr=False)
    release: Callable[[Tensor], None] = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self._seal is not _PROVENANCE_SEAL:
            raise MdpConfigurationError("MDP: encoder gradient provenance is privately minted.")


class _D4ReplayGradientLifecycle:
    """Sealed mode-neutral view of one exact fixed or dynamic replay handoff."""

    __slots__ = (
        "completion",
        "handoff",
        "kind",
        "provenance",
        "source_entry",
        "source_escrow",
        "_seal",
    )

    def __init__(self, handoff: Any, completion: Any, kind: object, seal: object) -> None:
        if seal is not _LIFECYCLE_SEAL:
            raise MdpConfigurationError(
                "MDP: encoder gradient replay lifecycle is privately minted."
            )
        self.handoff = handoff
        self.completion = completion
        self.kind = kind
        self.source_entry = None
        self.source_escrow = None
        self._seal = _LIFECYCLE_SEAL
        self._refresh()

    def _refresh(self) -> None:
        if self._seal is not _LIFECYCLE_SEAL:
            raise MdpStateError("MDP: encoder gradient replay lifecycle is privately sealed.")
        if self.kind is _replay:
            if type(self.handoff) is not _replay._D4FixedDecoderGradientHandoff:
                raise MdpConfigurationError(
                    "MDP: encoder-only gradient route uses an exact replay handoff."
                )
            entry = _replay._ACTIVE_GRADIENT_HANDOFFS.get(id(self.handoff))
            if (
                type(entry) is not tuple
                or len(entry) != 15
                or type(entry[0]) is not weakref.ReferenceType
                or entry[0]() is not self.handoff
                or entry[8] is not self.completion
                or _replay._ACTIVE_COMPLETIONS.get(id(self.completion)) is not entry[13]
            ):
                self.handoff.require()
                raise MdpStateError(
                    "MDP: encoder-only gradient route uses exact authority and completion."
                )
            self.handoff.require()
            self.source_escrow = None
            trusted = entry[1:-1]
            completion_entry = entry[13]
            operations = trusted[5][2]
            self.provenance = _D4ReplayGradientProvenance(
                _replay,
                trusted[8],
                None,
                None,
                completion_entry[3],
                completion_entry[4],
                operations.acquire,
                operations.release,
                _PROVENANCE_SEAL,
            )
        elif self.kind is _dynamic_replay:
            if type(self.handoff) is not _dynamic_replay._D4DynamicDecoderGradientHandoff:
                raise MdpConfigurationError(
                    "MDP: dynamic encoder gradient route uses an exact replay handoff."
                )
            entry = _dynamic_replay._ACTIVE_GRADIENT_HANDOFFS.get(id(self.handoff))
            escrow = _dynamic_replay._GRADIENT_HANDOFF_ESCROWS.get(id(self.handoff))
            if (
                type(entry) is not tuple
                or len(entry) != 4
                or type(entry[0]) is not weakref.ReferenceType
                or entry[0]() is not self.handoff
                or type(escrow) is not _dynamic_replay._GradientHandoffEscrow
                or escrow.seal is not _dynamic_replay._GRADIENT_HANDOFF_ESCROW_SEAL
                or escrow.reference is not entry[0]
                or escrow.entry is not entry
                or escrow.trusted is not entry[1]
                or type(entry[1]) is not tuple
                or len(entry[1]) != 15
                or entry[1][7] is not self.completion
                or _dynamic_replay._TRUSTED_GRADIENT_HANDOFFS.get(id(self.handoff)) is not escrow
                or _dynamic_replay._ACTIVE_COMPLETIONS.get(id(self.completion))
                is not escrow.completion_entry
            ):
                self.handoff.require()
                raise MdpStateError(
                    "MDP: dynamic encoder gradient route uses exact authority and completion."
                )
            self.handoff.require()
            self.source_escrow = escrow
            trusted = entry[1]
            completion_entry = escrow.completion_entry
            predecessor_entry = trusted[13]
            resources = trusted[5]
            operations = resources[2]
            if (
                type(completion_entry) is not tuple
                or len(completion_entry) != 9
                or completion_entry[0] is not entry[0]
                or completion_entry[1] is not self.completion
                or completion_entry[2] is not trusted[8]
                or completion_entry[3] is not predecessor_entry
                or completion_entry[4] is not self.completion._lifecycle
                or completion_entry[5] is not trusted[1]
                or completion_entry[6] is not self.completion.globally_reduced_num_tokens
                or completion_entry[7] != trusted[9]
                or completion_entry[8] is not _dynamic_replay._GRADIENT_COMPLETION_ENTRY_SEAL
                or self.completion._owner is not trusted[8]
                or self.completion._owner_entry is not predecessor_entry
                or self.completion.authority is not trusted[1]
                or self.completion._seal is not _dynamic_replay._COMPLETION_SEAL
                or type(predecessor_entry) is not tuple
                or len(predecessor_entry) != 12
                or predecessor_entry[1] is not trusted[0]
                or predecessor_entry[2] is not trusted[1]
                or predecessor_entry[3] is not trusted[2]
                or predecessor_entry[5] is not operations.acquire
                or predecessor_entry[6] is not trusted[12]
                or not callable(predecessor_entry[5])
                or not callable(predecessor_entry[6])
                or _dynamic_replay._tensor_descriptor(completion_entry[6]) != completion_entry[7]
            ):
                raise MdpStateError(
                    "MDP: dynamic encoder gradient route retains exact replay provenance."
                )
            self.provenance = _D4ReplayGradientProvenance(
                _dynamic_replay,
                trusted[8],
                predecessor_entry,
                completion_entry[4],
                completion_entry[6],
                completion_entry[7],
                predecessor_entry[5],
                predecessor_entry[6],
                _PROVENANCE_SEAL,
            )
        else:
            raise MdpConfigurationError("MDP: encoder gradient replay lifecycle has exact mode.")
        self.source_entry = entry

    def require(self, authority: _DynamicIterationAuthority) -> tuple[Any, ...]:
        self._refresh()
        self.handoff.require()
        trusted = self.source_entry[1:-1] if self.kind is _replay else self.source_entry[1]
        if authority is not trusted[1]:
            raise MdpStateError(
                "MDP: encoder-only gradient route uses exact authority and completion."
            )
        return trusted

    def consume(self, authority: _DynamicIterationAuthority) -> tuple[Any, ...]:
        self.require(authority)
        self.handoff.consume(authority, self.completion)
        self._refresh()
        return self.source_entry[1:-1] if self.kind is _replay else self.source_entry[1]

    def require_exact(self, authority: _DynamicIterationAuthority) -> tuple[Any, ...]:
        entry = self.source_entry
        trusted = entry[1:-1] if self.kind is _replay else entry[1]
        escrow = self.source_escrow
        if (
            self._seal is not _LIFECYCLE_SEAL
            or self.kind._ACTIVE_GRADIENT_HANDOFFS.get(id(self.handoff)) is not entry
            or authority is not trusted[1]
            or self.completion is not trusted[7]
            or (
                self.kind is _replay
                and self.kind._ACTIVE_COMPLETIONS.get(id(self.completion)) is not entry[13]
            )
            or (
                self.kind is _dynamic_replay
                and (
                    self.kind._GRADIENT_HANDOFF_ESCROWS.get(id(self.handoff)) is not escrow
                    or self.kind._TRUSTED_GRADIENT_HANDOFFS.get(id(self.handoff)) is not escrow
                    or escrow.entry is not entry
                    or escrow.trusted is not trusted
                    or self.kind._ACTIVE_COMPLETIONS.get(id(self.completion))
                    is not escrow.completion_entry
                    or trusted[5][2].acquire is not self.provenance.acquire
                    or trusted[5][2].release is not self.provenance.release
                    or trusted[12] is not self.provenance.release
                    or trusted[13] is not self.provenance.predecessor_entry
                )
            )
        ):
            raise MdpStateError("MDP: encoder gradient replay lifecycle retains exact provenance.")
        self.handoff.require()
        return trusted

    def completion_entry(self) -> tuple[Any, ...]:
        if self.kind is _replay:
            return self.source_entry[13]
        return self.source_escrow.completion_entry

    def successor_completion_escrow(self, reference: weakref.ReferenceType[Any]) -> tuple[Any, ...]:
        source_entry = self.completion_entry()
        successor_entry = (reference, *source_entry[1:])
        return (
            self.kind,
            self.kind._ACTIVE_COMPLETIONS,
            successor_entry,
            self.provenance.release,
            self.provenance,
            _COMPLETION_ESCROW_SEAL,
        )

    def abort(self, primary: BaseException) -> None:
        if self.kind is _replay:
            self.handoff._abort_from_escrow(self.source_entry, primary)
        else:
            self.handoff._abort_from_escrow(self.source_escrow, primary)

    def activate(
        self,
        owner: Any,
        owner_entry: tuple[Any, ...],
        receipt_entry: tuple[Any, ...],
        completion_entry: tuple[Any, ...],
    ) -> None:
        trusted = owner_entry[1:]
        handoff, source_entry = self.handoff, self.source_entry
        runtime, completion = trusted[0], trusted[4]
        reference = owner_entry[0]
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        active = self.kind._ACTIVE_GRADIENT_HANDOFFS
        completions = self.kind._ACTIVE_COMPLETIONS
        if (
            active.get(id(handoff)) is not source_entry
            or completions.get(id(completion)) is not self.completion_entry()
            or runtime_entry is None
            or runtime_entry[0] is not runtime
            or (
                runtime_entry[1]() is not handoff
                if self.kind is _replay
                else runtime_entry[1] is not source_entry[0]
            )
        ):
            raise MdpStateError(
                "MDP: encoder-only gradient activation retains exact predecessor registries."
            )
        _ACTIVE_OWNERS[id(owner)] = owner_entry
        _ACTIVE_RECEIPTS[id(trusted[3])] = receipt_entry
        _replay._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
        completions[id(completion)] = completion_entry
        del active[id(handoff)]
        self.kind._RETIRED_GRADIENT_HANDOFFS[id(handoff)] = weakref.ref(handoff)
        if self.kind is _dynamic_replay:
            escrow = self.source_escrow
            if self.kind._GRADIENT_HANDOFF_ESCROWS.get(id(handoff)) is escrow:
                del self.kind._GRADIENT_HANDOFF_ESCROWS[id(handoff)]
            if self.kind._TRUSTED_GRADIENT_HANDOFFS.get(id(handoff)) is escrow:
                del self.kind._TRUSTED_GRADIENT_HANDOFFS[id(handoff)]
            escrow.reference = None
            escrow.entry = None
            escrow.trusted = None
            escrow.completion_entry = None
            escrow.seal = None
        handoff._state = self.kind._RETIRED
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
        "_prepared_entry",
        "_prepared_handoff_entry",
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
        self._prepared_entry = None
        self._prepared_handoff_entry = None

    def _prepare_from(self, lifecycle: _D4ReplayGradientLifecycle) -> None:
        handoff = lifecycle.handoff
        entry = lifecycle.source_entry
        lifecycle.require_exact(self.authority)
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(self._runtime))
        if (
            entry is None
            or entry[0]() is not handoff
            or (entry[-1] if lifecycle.kind is _replay else entry[2]) is not True
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
        receipt_entry = (
            self._prepared_reference,
            self.receipt,
            self.authority,
            self.completion,
            self.receipt.exchange,
            self.receipt.received_tensors,
        )
        completion_escrow = lifecycle.successor_completion_escrow(self._prepared_reference)
        self._trusted = (*self._trusted, receipt_entry, completion_escrow[2], completion_escrow)
        self._prepared_entry = (self._prepared_reference, *self._trusted)
        self._prepared_handoff_entry = entry

    def _activate_prepared(
        self,
        lifecycle: _D4ReplayGradientLifecycle,
        owner_entry: tuple[Any, ...],
        handoff_entry: tuple[Any, ...],
    ) -> None:
        self._require_prepared(lifecycle, owner_entry, handoff_entry)
        lifecycle.activate(self, owner_entry, owner_entry[15], owner_entry[16])
        self._prepared_reference = None
        self._prepared_handoff_reference = None
        self._prepared_entry = None
        self._prepared_handoff_entry = None

    def _require_prepared(
        self,
        lifecycle: _D4ReplayGradientLifecycle,
        owner_entry: tuple[Any, ...],
        handoff_entry: tuple[Any, ...],
    ) -> None:
        handoff = lifecycle.handoff
        if (
            type(owner_entry) is not tuple
            or len(owner_entry) != 18
            or type(owner_entry[0]) is not weakref.ReferenceType
            or owner_entry[0]() is not self
            or handoff_entry is not lifecycle.source_entry
            or type(handoff_entry[0]) is not weakref.ReferenceType
            or handoff_entry[0]() is not handoff
        ):
            raise MdpStateError(
                "MDP: encoder-only gradient activation uses exact prepared escrows."
            )
        trusted = owner_entry[1:]
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
            *self._trusted[14:17],
        )
        if (
            self._state is not _ACTIVE
            or self._prepared_entry is not owner_entry
            or self._prepared_handoff_entry is not handoff_entry
            or any(
                actual is not expected for actual, expected in zip(current, trusted, strict=True)
            )
        ):
            raise MdpStateError(
                "MDP: encoder-only gradient activation retains sealed prepared resources."
            )
        lifecycle.require_exact(self.authority)
        reference = owner_entry[0]
        runtime, receipt, completion = trusted[0], trusted[3], trusted[4]
        runtime_identity = id(runtime)
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
        if (
            _ACTIVE_OWNERS.get(id(self)) is not None
            or _ACTIVE_RECEIPTS.get(id(receipt)) is not None
            or runtime_entry is None
            or runtime_entry[0] is not runtime
        ):
            raise MdpStateError(
                "MDP: encoder-only gradient activation retains exact predecessor registries."
            )

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
            *self._trusted[14:17],
        )
        if self._state is not _ACTIVE or any(
            actual is not expected for actual, expected in zip(current, entry[1:], strict=True)
        ):
            raise MdpStateError("MDP: encoder-only gradient route owner retains sealed resources.")
        receipt_entry = _ACTIVE_RECEIPTS.get(id(self.receipt))
        completion_escrow = self._trusted[16]
        if (
            type(completion_escrow) is not tuple
            or len(completion_escrow) != 6
            or completion_escrow[5] is not _COMPLETION_ESCROW_SEAL
            or completion_escrow[0] not in (_replay, _dynamic_replay)
            or completion_escrow[1] is not completion_escrow[0]._ACTIVE_COMPLETIONS
            or completion_escrow[2] is not self._trusted[15]
        ):
            raise MdpStateError("MDP: encoder-only gradient route retains exact completion escrow.")
        mode, completion_registry, completion_entry, _release, provenance, _seal = completion_escrow
        if (
            receipt_entry is not self._trusted[14]
            or receipt_entry[0]() is not self
            or receipt_entry[1] is not self.receipt
            or self.receipt.authority is not self.authority
            or self.receipt.completion is not self.completion
            or self.receipt.exchange is not receipt_entry[4]
            or self.receipt.received_tensors is not receipt_entry[5]
            or self.receipt.received_tensors is not self.receipt.exchange.received_tensors
            or self.receipt._seal is not _RECEIPT_SEAL
            or completion_entry is not self._trusted[15]
            or type(completion_entry) is not tuple
            or len(completion_entry) != (5 if mode is _replay else 9)
            or type(completion_entry[0]) is not weakref.ReferenceType
            or completion_entry[0]() is not self
            or completion_entry[1] is not self.completion
            or completion_registry.get(id(self.completion)) is not completion_entry
            or type(provenance) is not _D4ReplayGradientProvenance
            or provenance._seal is not _PROVENANCE_SEAL
            or provenance.mode is not mode
            or completion_escrow[3] is not provenance.release
            or self.completion._owner is not provenance.predecessor
            or self.completion.authority is not self.authority
            or self.completion.globally_reduced_num_tokens is not provenance.token
            or _replay._tensor_descriptor(provenance.token) != provenance.token_descriptor
            or not callable(provenance.acquire)
            or not callable(provenance.release)
            or (
                mode is _dynamic_replay
                and (
                    completion_entry[2] is not provenance.predecessor
                    or completion_entry[3] is not provenance.predecessor_entry
                    or completion_entry[4] is not provenance.completion_lifecycle
                    or completion_entry[5] is not self.authority
                    or completion_entry[6] is not provenance.token
                    or completion_entry[7] != provenance.token_descriptor
                    or completion_entry[8] is not _dynamic_replay._GRADIENT_COMPLETION_ENTRY_SEAL
                    or type(self.completion) is not _dynamic_replay._D4DynamicDecoderCompletion
                    or self.completion._owner_entry is not provenance.predecessor_entry
                    or self.completion._lifecycle is not provenance.completion_lifecycle
                    or self.completion._seal is not _dynamic_replay._COMPLETION_SEAL
                )
            )
            or (
                mode is _replay
                and (
                    completion_entry[2] is not self.authority
                    or completion_entry[3] is not provenance.token
                    or completion_entry[4] != provenance.token_descriptor
                    or type(self.completion) is not _replay._D4FixedDecoderCompletion
                )
            )
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
        self._abort_from_escrow(entry, primary_error)

    def _abort_from_escrow(
        self, entry: tuple[Any, ...], primary_error: BaseException | None = None
    ) -> None:
        """Retire using the exact entry prepared before activation callbacks."""
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError(
                "MDP: encoder-only gradient route abort error is an exception."
            )
        retired = _RETIRED_OWNERS.get(id(self))
        if retired is not None and retired() is self:
            raise MdpStateError("MDP: encoder-only gradient route owner is retired.")
        if (
            type(entry) is not tuple
            or len(entry) != 18
            or type(entry[0]) is not weakref.ReferenceType
            or entry[0]() is not self
        ):
            raise MdpStateError("MDP: encoder-only gradient cleanup uses its exact owner escrow.")
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
                *trusted[14:17],
            )
            completion_escrow = trusted[16]
            provenance = completion_escrow[4]
            if (
                object.__getattribute__(self, "_state") is not _ACTIVE
                or any(
                    actual is not expected
                    for actual, expected in zip(current, trusted, strict=True)
                )
                or type(completion_escrow) is not tuple
                or len(completion_escrow) != 6
                or completion_escrow[5] is not _COMPLETION_ESCROW_SEAL
                or type(provenance) is not _D4ReplayGradientProvenance
                or provenance._seal is not _PROVENANCE_SEAL
                or provenance.mode is not completion_escrow[0]
                or provenance.release is not completion_escrow[3]
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
        if _ACTIVE_OWNERS.get(id(self)) is entry:
            del _ACTIVE_OWNERS[id(self)]
        _RETIRED_OWNERS[id(self)] = weakref.ref(self)
        runtime = trusted[0]
        runtime_entry = _replay._forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            del _replay._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)]
        if _ACTIVE_RECEIPTS.get(id(trusted[3])) is trusted[14]:
            del _ACTIVE_RECEIPTS[id(trusted[3])]
        completion_escrow = trusted[16]
        completion_registry = completion_escrow[1]
        if completion_registry.get(id(trusted[4])) is completion_escrow[2]:
            del completion_registry[id(trusted[4])]
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
        self._prepared_entry = None
        self._prepared_handoff_entry = None
        if integrity_error is not None:
            _add_cleanup_note(primary, "encoder-only gradient route integrity validation failed.")
        handoff_resources, leaf_bases, transport_buffers, operations = trusted[7:11]
        mode, _registry, _entry, trusted_release, provenance, _seal = completion_escrow
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
                (trusted_release if mode is _dynamic_replay else operations.release)(buffer)
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
    lifecycle = _D4ReplayGradientLifecycle(handoff, completion, _replay, _LIFECYCLE_SEAL)
    return _run_repeated_d4_encoder_gradient_lifecycle(
        lifecycle, authority, all_to_all_single=all_to_all_single, byte_generator=byte_generator
    )


def run_repeated_d4_dynamic_encoder_gradient(
    handoff: _dynamic_replay._D4DynamicDecoderGradientHandoff,
    authority: _DynamicIterationAuthority,
    completion: _dynamic_replay._D4DynamicDecoderCompletion,
    *,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
    byte_generator: Callable[[int], Any] | None = None,
) -> _D4EncoderGradientRouteOwner:
    """Authorize Gate3 from one exact completed dynamic decoder replay."""
    if type(handoff) is not _dynamic_replay._D4DynamicDecoderGradientHandoff:
        raise MdpConfigurationError(
            "MDP: dynamic encoder gradient route uses an exact replay handoff."
        )
    try:
        handoff.require()
    except BaseException as primary:
        try:
            handoff.abort(primary)
        except BaseException as cleanup_error:
            _add_cleanup_note(
                primary,
                f"suppressed dynamic encoder gradient handoff cleanup error: {cleanup_error!r}",
            )
        raise
    if (
        type(completion) is not _dynamic_replay._D4DynamicDecoderCompletion
        or completion is not handoff.completion
    ):
        raise MdpStateError(
            "MDP: dynamic encoder gradient route uses exact authority and completion."
        )
    try:
        lifecycle = _D4ReplayGradientLifecycle(
            handoff, completion, _dynamic_replay, _LIFECYCLE_SEAL
        )
    except BaseException as primary:
        if type(handoff) is _dynamic_replay._D4DynamicDecoderGradientHandoff:
            try:
                handoff.abort(primary)
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    primary,
                    f"suppressed dynamic encoder gradient handoff cleanup error: {cleanup_error!r}",
                )
        raise
    return _run_repeated_d4_encoder_gradient_lifecycle(
        lifecycle, authority, all_to_all_single=all_to_all_single, byte_generator=byte_generator
    )


def _run_repeated_d4_encoder_gradient_lifecycle(
    lifecycle: _D4ReplayGradientLifecycle,
    authority: _DynamicIterationAuthority,
    *,
    all_to_all_single: Callable[..., Any],
    byte_generator: Callable[[int], Any] | None,
) -> _D4EncoderGradientRouteOwner:
    if type(lifecycle) is not _D4ReplayGradientLifecycle or lifecycle._seal is not _LIFECYCLE_SEAL:
        raise MdpConfigurationError("MDP: encoder gradient route uses a sealed replay lifecycle.")
    trusted = lifecycle.require(authority)
    handoff, completion = lifecycle.handoff, lifecycle.completion
    binding = trusted[2]
    handoff_operations = trusted[5][2]
    handoff_release = (
        lifecycle.provenance.release
        if lifecycle.kind is _dynamic_replay
        else handoff_operations.release
    )
    candidate = None
    owner = None
    owner_entry = None
    prepare_started = False

    def prepare() -> _GradientCandidate:
        nonlocal candidate, owner, owner_entry, prepare_started
        if prepare_started:
            raise MdpStateError("MDP: encoder-only gradient preparation is one-shot.")
        prepare_started = True
        trusted = lifecycle.consume(authority)
        runtime = trusted[0]
        handoff_resources = trusted[5]
        operations = handoff_resources[2]
        acquire = (
            lifecycle.provenance.acquire
            if lifecycle.kind is _dynamic_replay
            else operations.acquire
        )
        release = (
            lifecycle.provenance.release
            if lifecycle.kind is _dynamic_replay
            else operations.release
        )
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
                buffer = acquire(
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
            owner._prepare_from(lifecycle)
            owner_entry = owner._prepared_entry
            candidate = prepared_candidate
            return candidate
        except BaseException as error:
            for buffer in reversed(buffers):
                try:
                    release(buffer)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        error,
                        f"suppressed encoder gradient preparation cleanup error: {cleanup_error!r}",
                    )
            raise

    def authorize(prepared: _GradientCandidate) -> _GradientCandidate:
        if prepared is not candidate or type(prepared) is not _GradientCandidate:
            raise MdpTaskFatalError("MDP: encoder-only Gate3 runner returns exact candidate.")
        lifecycle.require_exact(authority)
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
            lifecycle.require_exact(authority)
            owner._require_prepared(lifecycle, owner_entry, lifecycle.source_entry)
            received = _execute_validated_dynamic_bridge_exchange(
                candidate.exchange, group=candidate.group, all_to_all_single=all_to_all_single
            )
            if received is not candidate.receipt.received_tensors:
                raise MdpBridgeError("MDP: encoder-only gradient A2A returns exact received views.")
            owner._activate_prepared(lifecycle, owner_entry, lifecycle.source_entry)
        except BaseException as error:
            if type(error) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: physical encoder-gradient route failed after Gate3 final WORLD."
            ) from error
        owner.require()
        return owner
    except BaseException as error:
        current_owner_entry = None if owner is None else _ACTIVE_OWNERS.get(id(owner))
        owner_entry_is_exact = owner_entry is not None and current_owner_entry is owner_entry
        owner_was_activated = owner is not None and (
            owner_entry_is_exact
            or (
                owner_entry is not None
                and (retired := lifecycle.kind._RETIRED_GRADIENT_HANDOFFS.get(id(handoff)))
                is not None
                and retired() is handoff
            )
        )
        if owner_was_activated:
            try:
                owner._abort_from_escrow(owner_entry, error)
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    error, f"suppressed encoder gradient owner cleanup error: {cleanup_error!r}"
                )
            raise
        if candidate is not None:
            for buffer in reversed(candidate.transport_buffers):
                try:
                    handoff_release(buffer)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        error,
                        f"suppressed encoder gradient buffer cleanup error: {cleanup_error!r}",
                    )
        try:
            lifecycle.abort(error)
        except BaseException as cleanup_error:
            _add_cleanup_note(
                error, f"suppressed encoder gradient handoff cleanup error: {cleanup_error!r}"
            )
        raise


def _run_repeated_d4_encoder_gradient_from_replay(
    replay: _replay._D4FixedDecoderReplayOwner,
    authority: _DynamicIterationAuthority,
    completion: _replay._D4FixedDecoderCompletion,
    *,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
    byte_generator: Callable[[int], Any] | None = None,
) -> _D4EncoderGradientRouteOwner:
    """Atomically claim completed replay and return its exact Gate3 successor."""
    if type(replay) is not _replay._D4FixedDecoderReplayOwner:
        raise MdpConfigurationError(
            "MDP: encoder gradient replay route uses an exact fixed decoder owner."
        )
    handoff = None
    try:
        replay.require()
        if authority is not replay.authority:
            raise MdpStateError(
                "MDP: encoder gradient replay route uses its exact iteration authority."
            )
        replay.require_completion(completion)
        handoff = replay._claim_for_gradient(authority, completion)
        return run_repeated_d4_encoder_gradient(
            handoff,
            authority,
            completion,
            all_to_all_single=all_to_all_single,
            byte_generator=byte_generator,
        )
    except BaseException as primary:
        if handoff is None:
            try:
                replay.abort(primary)
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    primary, f"suppressed fixed decoder replay cleanup error: {cleanup_error!r}"
                )
        raise


def _run_repeated_d4_dynamic_encoder_gradient_from_replay(
    replay: _dynamic_replay._D4DynamicDecoderReplayOwner,
    authority: _DynamicIterationAuthority,
    completion: _dynamic_replay._D4DynamicDecoderCompletion,
    *,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
    byte_generator: Callable[[int], Any] | None = None,
) -> _D4EncoderGradientRouteOwner:
    """Atomically claim completed dynamic replay and return its exact Gate3 successor."""
    if type(replay) is not _dynamic_replay._D4DynamicDecoderReplayOwner:
        raise MdpConfigurationError(
            "MDP: encoder gradient replay route uses an exact dynamic decoder owner."
        )
    handoff = None
    try:
        replay.require()
        if authority is not replay.authority:
            raise MdpStateError(
                "MDP: encoder gradient replay route uses its exact iteration authority."
            )
        replay.require_completion(completion)
        handoff = replay._claim_for_gradient(authority, completion)
        return run_repeated_d4_dynamic_encoder_gradient(
            handoff,
            authority,
            completion,
            all_to_all_single=all_to_all_single,
            byte_generator=byte_generator,
        )
    except BaseException as primary:
        if handoff is None:
            try:
                replay.abort(primary)
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    primary, f"suppressed dynamic decoder replay cleanup error: {cleanup_error!r}"
                )
        raise
