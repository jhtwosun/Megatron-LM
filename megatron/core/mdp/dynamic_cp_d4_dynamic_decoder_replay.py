# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private publication-consuming dynamic decoder replay ownership."""

import weakref
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from megatron.core.mdp import dynamic_cp_d4_encoder_forward as _forward
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_bridge_transport import validate_prepared_dynamic_bridge_exchange
from megatron.core.mdp.dynamic_cp_d3_ready_artifacts import (
    _materialize_local_decoder_ready_artifacts,
)
from megatron.core.mdp.dynamic_cp_d3_ready_handoff import _compose_local_decoder_ready_handoff
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _snapshot_local_authority,
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_execution import LocalDecoderAssignment
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderReadyIteration,
    _decoder_ready_authority_digest,
    _dynamic_iteration_plan_digest,
    _DynamicIterationAuthority,
    _expected_local_assignments,
    validate_decoder_ready_iteration,
)
from megatron.core.mdp.dynamic_cp_transport import validate_prepared_decoder_payload_bundle
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)

__all__ = ()

_PREPARED = object()
_ACTIVE = object()
_RETIRED = object()
_OWNER_SEAL = object()
_OWNER_ESCROW_SEAL = object()


class _OwnerEscrow:
    __slots__ = ("entry", "reference", "seal")

    def __init__(self, reference, entry, *, seal):
        if seal is not _OWNER_ESCROW_SEAL:
            raise MdpConfigurationError("MDP: dynamic decoder replay escrow is privately minted.")
        self.reference = reference
        self.entry = entry
        self.seal = seal


_PENDING_OWNERS: dict[int, tuple[Any, ...]] = {}
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_TRUSTED_OWNERS: dict[int, tuple[Any, ...]] = {}
_OWNER_ESCROWS: dict[int, _OwnerEscrow] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}
_RETIRED_ESCROWS: dict[int, weakref.ReferenceType[Any]] = {}


def _add_cleanup_note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


def _storage_interval(tensor: Tensor) -> tuple[int, int, int]:
    start = tensor.storage_offset() * tensor.element_size()
    return (
        tensor.untyped_storage().data_ptr(),
        start,
        start + tensor.numel() * tensor.element_size(),
    )


def _overlaps(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return left[0] == right[0] and left[1] < right[2] and right[1] < left[2]


def _capture_assignment_groups(
    assignments: tuple[LocalDecoderAssignment, ...],
    *,
    global_rank: int,
    group_ranks_getter: Callable[[Any], Any],
) -> tuple[tuple[Any, tuple[int, ...]], ...]:
    if not callable(group_ranks_getter):
        raise MdpConfigurationError("MDP: dynamic decoder group-ranks getter is callable.")
    captured = []
    for local in assignments:
        try:
            size = local.cp_group.size()
            ranks = group_ranks_getter(local.cp_group)
            local_rank = local.cp_group.rank()
        except Exception as error:
            raise MdpConfigurationError(
                "MDP: dynamic decoder native group query succeeds."
            ) from error
        if (
            type(size) is not int
            or size != local.assignment.local_cp_size
            or type(ranks) is not tuple
            or ranks != local.assignment.endpoint_ranks
            or type(local_rank) is not int
            or global_rank not in ranks
            or local_rank != ranks.index(global_rank)
        ):
            raise MdpPlanError("MDP: dynamic decoder assignments retain exact native groups.")
        prior = next((value for group, value in captured if group is local.cp_group), None)
        if prior is not None:
            if prior != ranks:
                raise MdpPlanError("MDP: dynamic decoder assignments retain exact native groups.")
            continue
        captured.append((local.cp_group, ranks))
    return tuple(captured)


def _captured_group_ranks_getter(captured: tuple[tuple[Any, tuple[int, ...]], ...]):
    def ranks(group: Any) -> tuple[int, ...]:
        matches = tuple(value for expected, value in captured if group is expected)
        if len(matches) != 1:
            raise MdpStateError("MDP: dynamic decoder ready uses only captured native groups.")
        return matches[0]

    return ranks


def _retire_consumed_handoff(
    handoff: _forward._D4EncoderReplayHandoff, entry: tuple[Any, ...]
) -> None:
    identity = id(handoff)
    if _forward._ACTIVE_REPLAY_HANDOFFS.get(identity) is entry:
        del _forward._ACTIVE_REPLAY_HANDOFFS[identity]

    def remove(reference: weakref.ReferenceType[Any]) -> None:
        if _forward._RETIRED_REPLAY_HANDOFFS.get(identity) is reference:
            del _forward._RETIRED_REPLAY_HANDOFFS[identity]

    retired = _forward._RETIRED_REPLAY_HANDOFFS.get(identity)
    if retired is None or retired() is None or retired() is handoff:
        _forward._RETIRED_REPLAY_HANDOFFS[identity] = weakref.ref(handoff, remove)
    handoff._state = _forward._RETIRED
    for name in (
        "authority",
        "binding",
        "selected_ranks",
        "membership",
        "layout",
        "payload_bundle",
        "embedding_bundle",
        "output",
        "item_outputs",
        "forward_handle",
        "_runtime",
        "_binding_owner",
        "_buffers",
        "_operations",
    ):
        setattr(handoff, name, None)
    handoff._trusted = ()


def _expected_item_ids(authority: _DynamicIterationAuthority, assignment: LocalDecoderAssignment):
    samples = {sample.sample_id: sample for sample in authority.global_manifest.samples}
    return tuple(
        planned.item_id
        for sample_id in assignment.assignment.sample_ids
        for planned in samples[sample_id].vision_items
    )


def _preflight_embedding_sources(
    authority: _DynamicIterationAuthority,
    *,
    global_rank: int,
    assignments: tuple[LocalDecoderAssignment, ...],
    embedding_exchange: Any,
) -> tuple[tuple[LocalDecoderAssignment, tuple[tuple[Tensor, int], ...], int], ...]:
    exchange = validate_prepared_dynamic_bridge_exchange(embedding_exchange)
    if exchange.phase is not BridgePhase.EMBEDDING:
        raise MdpBridgeError("MDP: dynamic decoder replay consumes the Gate1 embedding phase.")
    expected_keys = tuple(
        entry.key
        for entry in authority.embedding_ledger.entries
        if entry.dst_global_rank == global_rank
    )
    if tuple(exchange.received_tensors) != expected_keys:
        raise MdpBridgeError("MDP: dynamic decoder replay consumes exact embedding routes.")
    items = {item.item_id: item for item in authority.global_manifest.items}
    prepared = []
    for assignment in assignments:
        sources = []
        rows = 0
        for item_id in _expected_item_ids(authority, assignment):
            item_rows = items[item_id].output_rows
            source = exchange.received_tensors[DynamicBridgeKey(item_id, global_rank)]
            if (
                type(source) is not Tensor
                or tuple(source.shape) != (item_rows, authority.bridge_width)
                or source.dtype != authority.bridge_dtype
                or source.device != exchange.receive_buffer.device
                or not source.is_contiguous()
            ):
                raise MdpBridgeError(
                    "MDP: dynamic decoder replay embedding routes match exact item geometry."
                )
            sources.append((source, item_rows))
            rows += item_rows
        prepared.append((assignment, tuple(sources), rows))
    return tuple(prepared)


class _D4DynamicDecoderReplayOwner:
    """Sole post-Gate2 owner of dynamic replay records and E2 resources."""

    __slots__ = (
        "__weakref__",
        "authority",
        "binding",
        "ready",
        "records",
        "embedding_leaves",
        "_runtime",
        "_state",
        "_trusted",
    )

    def __init__(self, trusted: tuple[Any, ...], *, seal: object) -> None:
        if seal is not _OWNER_SEAL:
            raise MdpConfigurationError("MDP: dynamic decoder replay owner is privately minted.")
        self._runtime = trusted[0]
        self.authority = trusted[1]
        self.binding = trusted[2]
        self.ready = None
        self.records = ()
        self.embedding_leaves = MappingProxyType({})
        self._state = _PREPARED
        self._trusted = trusted

    def _claim_handoff(
        self, handoff: _forward._D4EncoderReplayHandoff, handoff_entry: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        handoff.consume(self.authority)
        consumed_entry = _forward._ACTIVE_REPLAY_HANDOFFS.get(id(handoff))
        runtime_entry = _forward._ACTIVE_RUNTIME_OWNERS.get(id(self._runtime))
        if (
            type(consumed_entry) is not tuple
            or len(consumed_entry) != len(handoff_entry)
            or any(
                actual is not expected
                for actual, expected in zip(consumed_entry[:-1], handoff_entry[:-1], strict=True)
            )
            or consumed_entry[-1] is not True
            or consumed_entry[0]() is not handoff
            or runtime_entry is None
            or runtime_entry[0] is not self._runtime
            or runtime_entry[1]() is not handoff
        ):
            raise MdpStateError("MDP: dynamic decoder replay claims its exact consumed handoff.")
        identity = id(self)
        runtime_identity = id(self._runtime)

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            entry = _PENDING_OWNERS.get(identity)
            if entry is not None and entry[0] is reference:
                del _PENDING_OWNERS[identity]
            entry = _ACTIVE_OWNERS.get(identity)
            if entry is not None and entry[0] is reference:
                del _ACTIVE_OWNERS[identity]
            entry = _TRUSTED_OWNERS.get(identity)
            if entry is not None and entry[0] is reference:
                del _TRUSTED_OWNERS[identity]
            escrow = _OWNER_ESCROWS.get(identity)
            if type(escrow) is _OwnerEscrow and escrow.reference is reference:
                del _OWNER_ESCROWS[identity]
            runtime = _forward._ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
            if runtime is not None and runtime[1] is reference:
                del _forward._ACTIVE_RUNTIME_OWNERS[runtime_identity]

        reference = weakref.ref(self, retire)
        entry = (reference, *self._trusted, [], None)
        installed = False
        try:
            _PENDING_OWNERS[identity] = entry
            _TRUSTED_OWNERS[identity] = entry
            _OWNER_ESCROWS[identity] = _OwnerEscrow(reference, entry, seal=_OWNER_ESCROW_SEAL)
            _forward._ACTIVE_RUNTIME_OWNERS[runtime_identity] = (self._runtime, reference)
            self._trusted = entry
            installed = True
            if _forward._ACTIVE_REPLAY_HANDOFFS.get(id(handoff)) is not consumed_entry:
                raise MdpStateError(
                    "MDP: dynamic decoder replay retains its consumed handoff entry."
                )
            _retire_consumed_handoff(handoff, consumed_entry)
            return entry
        except BaseException as primary:
            if installed:
                try:
                    _retire_consumed_handoff(handoff, consumed_entry)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        primary, f"suppressed consumed handoff retirement error: {cleanup_error!r}"
                    )
                try:
                    self._abort_from_entry(entry, primary)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        primary,
                        f"suppressed partial dynamic owner cleanup error: {cleanup_error!r}",
                    )
            raise

    def _seal_ready(
        self, entry: tuple[Any, ...], ready: DecoderReadyIteration, leaf_bases: tuple[Tensor, ...]
    ) -> tuple[Any, ...]:
        if _PENDING_OWNERS.get(id(self)) is not entry or entry[0]() is not self:
            raise MdpStateError("MDP: dynamic decoder replay retains its pending owner.")
        if len(entry[-2]) != len(leaf_bases) or any(
            actual is not expected for actual, expected in zip(entry[-2], leaf_bases, strict=True)
        ):
            raise MdpStateError("MDP: dynamic decoder replay retains exact allocated leaves.")
        self.ready = ready
        self.records = ready.records
        self.embedding_leaves = ready.embedding_leaves
        sealed = (*entry[:-2], tuple(entry[-2]), ready)
        self._trusted = sealed
        _PENDING_OWNERS[id(self)] = sealed
        _TRUSTED_OWNERS[id(self)] = sealed
        _OWNER_ESCROWS[id(self)] = _OwnerEscrow(sealed[0], sealed, seal=_OWNER_ESCROW_SEAL)
        return sealed

    def _require(self, registry: dict[int, tuple[Any, ...]], state: object):
        entry = registry.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_OWNERS.get(id(self))
            escrow = _RETIRED_ESCROWS.get(id(self))
            if (retired is not None and retired() is self) or (
                escrow is not None and escrow() is self
            ):
                raise MdpStateError("MDP: dynamic decoder replay owner is retired.")
            raise MdpStateError("MDP: dynamic decoder replay owner is the exact registered owner.")
        escrow = _OWNER_ESCROWS.get(id(self))
        if (
            _TRUSTED_OWNERS.get(id(self)) is not entry
            or type(escrow) is not _OwnerEscrow
            or escrow.seal is not _OWNER_ESCROW_SEAL
            or escrow.reference is not entry[0]
            or escrow.reference() is not self
            or escrow.entry is not entry
            or self._trusted is not entry
        ):
            raise MdpStateError("MDP: dynamic decoder replay retains its exact trusted entry.")
        runtime_entry = _forward._ACTIVE_RUNTIME_OWNERS.get(id(entry[1]))
        if (
            runtime_entry is None
            or runtime_entry[0] is not entry[1]
            or runtime_entry[1]() is not self
        ):
            raise MdpStateError("MDP: dynamic decoder replay retains sole runtime ownership.")
        current = (
            self._runtime,
            self.authority,
            self.binding,
            self.ready,
            self.records,
            self.embedding_leaves,
        )
        expected = (
            entry[1],
            entry[2],
            entry[3],
            entry[-1],
            entry[-1].records,
            entry[-1].embedding_leaves,
        )
        if self._state is not state or any(
            actual is not wanted for actual, wanted in zip(current, expected, strict=True)
        ):
            raise MdpStateError("MDP: dynamic decoder replay owner retains sealed fields.")
        return entry

    def require_prepared(self) -> "_D4DynamicDecoderReplayOwner":
        self._require(_PENDING_OWNERS, _PREPARED)
        return self

    def _activate(self, entry: tuple[Any, ...]) -> None:
        if self._require(_PENDING_OWNERS, _PREPARED) is not entry:
            raise MdpStateError("MDP: dynamic decoder replay activates its exact prepared owner.")
        del _PENDING_OWNERS[id(self)]
        _ACTIVE_OWNERS[id(self)] = entry
        self._state = _ACTIVE

    def require(self) -> "_D4DynamicDecoderReplayOwner":
        self._require(_ACTIVE_OWNERS, _ACTIVE)
        return self

    def _abort_from_entry(self, entry: tuple[Any, ...], primary: BaseException) -> None:
        identity = id(self)
        cleaned = _RETIRED_ESCROWS.get(identity)
        if cleaned is not None and cleaned() is self:
            raise MdpStateError("MDP: dynamic decoder replay owner is retired.")
        retired = _RETIRED_OWNERS.get(identity)
        if retired is not None and retired() is self:
            raise MdpStateError("MDP: dynamic decoder replay owner is retired.")
        integrity_error = None
        try:
            ready = entry[-1]
            current = (
                object.__getattribute__(self, "_runtime"),
                object.__getattribute__(self, "authority"),
                object.__getattribute__(self, "binding"),
                object.__getattribute__(self, "ready"),
            )
            if (
                _TRUSTED_OWNERS.get(identity) is not entry
                or object.__getattribute__(self, "_trusted") is not entry
                or current[0] is not entry[1]
                or current[1] is not entry[2]
                or current[2] is not entry[3]
                or current[3] is not ready
            ):
                integrity_error = MdpStateError(
                    "MDP: dynamic decoder replay owner retains sealed fields."
                )
        except BaseException as error:
            integrity_error = error
        if _PENDING_OWNERS.get(identity) is entry:
            del _PENDING_OWNERS[identity]
        if _ACTIVE_OWNERS.get(identity) is entry:
            del _ACTIVE_OWNERS[identity]
        if _TRUSTED_OWNERS.get(identity) is entry:
            del _TRUSTED_OWNERS[identity]
        escrow = _OWNER_ESCROWS.get(identity)
        if (
            type(escrow) is _OwnerEscrow
            and escrow.seal is _OWNER_ESCROW_SEAL
            and escrow.reference() is self
            and escrow.entry is entry
        ):
            del _OWNER_ESCROWS[identity]
        runtime = entry[1]
        runtime_entry = _forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            del _forward._ACTIVE_RUNTIME_OWNERS[id(runtime)]

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            cleaned_entry = _RETIRED_ESCROWS.get(identity)
            if cleaned_entry is reference:
                del _RETIRED_ESCROWS[identity]
            retired_entry = _RETIRED_OWNERS.get(identity)
            if retired_entry is reference:
                del _RETIRED_OWNERS[identity]

        reference = weakref.ref(self, remove)
        _RETIRED_ESCROWS[identity] = reference
        retired_entry = _RETIRED_OWNERS.get(identity)
        if retired_entry is None or retired_entry() is self:
            _RETIRED_OWNERS[identity] = reference
        handoff_trusted = entry[4]
        leaf_bases = tuple(entry[-2])
        binding_owner, buffers, _ = handoff_trusted[-3:]
        forward_handle = handoff_trusted[10]
        release = entry[6]
        self._state = _RETIRED
        self.authority = None
        self.binding = None
        self.ready = None
        self.records = ()
        self.embedding_leaves = MappingProxyType({})
        self._runtime = None
        self._trusted = ()
        if integrity_error is not None:
            _add_cleanup_note(primary, "dynamic decoder replay integrity validation failed.")
        if forward_handle is not None:
            try:
                forward_handle.release_forward_only()
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed dynamic replay graph release error: {error!r}"
                )
        if binding_owner is not None:
            try:
                binding_owner.restore(primary)
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed dynamic replay binding restore error: {error!r}"
                )
        for buffer in (*leaf_bases, *buffers):
            try:
                release(buffer)
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed dynamic replay buffer release error: {error!r}"
                )

    def abort(self, primary_error: BaseException | None = None) -> None:
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: dynamic decoder replay abort error is an exception.")
        identity = id(self)
        entry = next(
            (
                candidate
                for candidate in (_ACTIVE_OWNERS.get(identity), _PENDING_OWNERS.get(identity))
                if type(candidate) is tuple
                and len(candidate) >= 2
                and candidate[0]() is self
                and self._trusted is candidate
            ),
            None,
        )
        if entry is None:
            escrow = _OWNER_ESCROWS.get(identity)
            if (
                type(escrow) is not _OwnerEscrow
                or escrow.seal is not _OWNER_ESCROW_SEAL
                or escrow.reference() is not self
                or self._trusted is not escrow.entry
            ):
                self.require()
            entry = escrow.entry
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: dynamic decoder replay was aborted.")
        )
        self._abort_from_entry(entry, primary)


def _validate_ready(
    owner: _D4DynamicDecoderReplayOwner,
    entry: tuple[Any, ...],
    assignments,
    cp_partition_mode,
    *,
    registry: dict[int, tuple[Any, ...]],
    state: object,
):
    if owner._require(registry, state) is not entry:
        raise MdpStateError("MDP: dynamic decoder ready uses its exact owner escrow.")
    authority = entry[2]
    handoff = entry[4]
    ready = entry[-1]
    digest = _decoder_ready_authority_digest(
        global_manifest_digest=authority.global_manifest.digest,
        decoder_plan_digest=_dynamic_iteration_plan_digest(authority),
        payload_bundle_authority_digest=handoff[6].bundle_authority_digest,
        embedding_route_authority_digest=handoff[7].route_authority_digest,
        participant_ranks=authority.participant_ranks,
        cp_partition_mode=cp_partition_mode,
    )
    validate_decoder_ready_iteration(
        ready,
        global_manifest=authority.global_manifest,
        plan=authority.plan,
        payload_bundle=handoff[6],
        payload_tensors=handoff[6].received_tensors,
        embedding_exchange=handoff[7],
        embedding_tensors=handoff[7].received_tensors,
        expected_assignments=assignments,
        authority_digest=digest,
        embedding_width=authority.bridge_width,
        embedding_dtype=authority.bridge_dtype,
        cp_partition_mode=cp_partition_mode,
        plan_digest=_dynamic_iteration_plan_digest(authority),
    )


def run_repeated_d4_dynamic_decoder_replay(
    publication: _forward._D4EncoderPublicationOwner,
    authority: _DynamicIterationAuthority,
    *,
    decoder_group_getter: Callable[..., Any],
    decoder_group_ranks_getter: Callable[[Any], Any],
    rebuild_microbatch: Callable[..., Any],
    cp_partition_mode: str,
    byte_generator: Callable[[int], Any] | None = None,
) -> _D4DynamicDecoderReplayOwner:
    """Claim one E2 publication and authorize dynamic replay behind Gate2."""
    if type(publication) is not _forward._D4EncoderPublicationOwner:
        raise MdpConfigurationError("MDP: dynamic decoder replay requires an exact publication.")
    publication.require()
    publication_entry = _forward._ACTIVE_PUBLICATIONS.get(id(publication))
    if publication_entry is None or publication_entry[0]() is not publication:
        raise MdpStateError("MDP: dynamic decoder replay requires its exact active publication.")
    publication_trusted = publication_entry[1:]
    binding = publication_trusted[2]
    if authority is not publication_trusted[1] or authority is not publication.authority:
        raise MdpStateError("MDP: dynamic decoder replay uses its exact iteration authority.")
    if type(binding) is not _RepeatedD4GroupBinding or publication.binding is not binding:
        raise MdpConfigurationError("MDP: dynamic decoder replay uses an exact D4 binding.")
    handoff = None
    owner = None
    owner_entry = None
    prepare_started = False

    def prepare() -> _D4DynamicDecoderReplayOwner:
        nonlocal handoff, owner, owner_entry, prepare_started
        if prepare_started:
            raise MdpStateError("MDP: dynamic decoder replay preparation is one-shot.")
        prepare_started = True
        publication.require()
        if authority is not publication_trusted[1] or authority is not publication.authority:
            raise MdpStateError("MDP: dynamic decoder replay uses its exact iteration authority.")
        if type(binding) is not _RepeatedD4GroupBinding or publication.binding is not binding:
            raise MdpConfigurationError("MDP: dynamic decoder replay uses an exact D4 binding.")
        _snapshot_local_authority(binding, authority)
        handoff = publication._claim_for_replay(authority)
        handoff_entry = _forward._ACTIVE_REPLAY_HANDOFFS[id(handoff)]
        handoff_trusted = handoff_entry[1:-1]
        operations = handoff_trusted[-1]
        if (
            type(operations) is not _forward._D4EncoderForwardOperations
            or operations._seal is not _forward._OPERATIONS_SEAL
            or operations.device != handoff_trusted[7].receive_buffer.device
            or operations.bridge_width != authority.bridge_width
            or operations.bridge_dtype != authority.bridge_dtype
        ):
            raise MdpStateError("MDP: dynamic decoder replay retains exact forward operations.")
        acquire = operations.acquire
        release = operations.release
        device = operations.device
        if not callable(acquire) or not callable(release):
            raise MdpConfigurationError(
                "MDP: dynamic decoder replay allocator operations are callable."
            )
        trusted = (
            handoff_trusted[0],
            authority,
            binding,
            handoff_trusted,
            acquire,
            release,
            device,
        )
        owner = _D4DynamicDecoderReplayOwner(trusted, seal=_OWNER_SEAL)
        owner_entry = owner._claim_handoff(handoff, handoff_entry)
        assignments = _expected_local_assignments(
            authority.plan,
            global_rank=binding.global_rank,
            decoder_group_getter=decoder_group_getter,
            decoder_group_ranks_getter=decoder_group_ranks_getter,
        )
        if any(assignment.assignment.local_cp_size not in (1, 2, 4) for assignment in assignments):
            raise MdpPlanError("MDP: dynamic decoder replay supports decoder CP1, CP2, or CP4.")
        captured_groups = _capture_assignment_groups(
            assignments,
            global_rank=binding.global_rank,
            group_ranks_getter=decoder_group_ranks_getter,
        )
        _snapshot_local_authority(binding, authority)
        captured_group_ranks = _captured_group_ranks_getter(captured_groups)
        payload = validate_prepared_decoder_payload_bundle(handoff_trusted[6])
        embedding = validate_prepared_dynamic_bridge_exchange(handoff_trusted[7])
        sources = _preflight_embedding_sources(
            authority,
            global_rank=binding.global_rank,
            assignments=assignments,
            embedding_exchange=embedding,
        )
        leaves = {}
        bases = []
        inherited_buffers = (
            tuple(handoff_trusted[-2])
            + tuple(
                buffer
                for child in payload.exchanges
                for buffer in (child.send_buffer, child.receive_buffer)
            )
            + (embedding.send_buffer, embedding.receive_buffer)
        )
        forbidden = tuple(
            _storage_interval(buffer)
            for buffer in inherited_buffers
            if type(buffer) is Tensor and buffer.numel()
        )
        allocated = []
        for assignment, item_sources, rows in sources:
            if not rows:
                continue
            leaf = acquire(
                rows=rows,
                width=authority.bridge_width,
                dtype=authority.bridge_dtype,
                device=device,
                tag=f"d4_dynamic_decoder_leaf_{assignment.key.microbatch_index}",
            )
            bases.append(leaf)
            owner_entry[-2].append(leaf)
            if (
                type(leaf) is not Tensor
                or tuple(leaf.shape) != (rows, authority.bridge_width)
                or leaf.dtype != authority.bridge_dtype
                or leaf.device != device
                or not leaf.is_contiguous()
                or not leaf.is_leaf
                or leaf.requires_grad
                or leaf.grad_fn is not None
            ):
                raise MdpConfigurationError(
                    "MDP: dynamic decoder replay allocator returns exact leaf geometry."
                )
            interval = _storage_interval(leaf)
            if any(_overlaps(interval, other) for other in (*forbidden, *allocated)):
                raise MdpConfigurationError(
                    "MDP: dynamic decoder replay leaves are disjoint from owned buffers."
                )
            allocated.append(interval)
            _snapshot_local_authority(binding, authority)
            cursor = 0
            with torch.no_grad():
                for source, item_rows in item_sources:
                    leaf[cursor : cursor + item_rows].copy_(source)
                    cursor += item_rows
            leaf.requires_grad_(True)
            leaves[assignment.key] = leaf
        leaves = MappingProxyType(leaves)
        artifacts = _materialize_local_decoder_ready_artifacts(
            authority=authority,
            global_rank=binding.global_rank,
            payload_bundle=payload,
            payload_result=payload.received_tensors,
            embedding_exchange=embedding,
            embedding_leaves=leaves,
            assignments=assignments,
            group_ranks_getter=captured_group_ranks,
            cp_partition_mode=cp_partition_mode,
            rebuild_microbatch=rebuild_microbatch,
        )
        ready = _compose_local_decoder_ready_handoff(
            authority=authority,
            global_rank=binding.global_rank,
            payload_bundle=payload,
            payload_result=payload.received_tensors,
            embedding_exchange=embedding,
            embedding_result=embedding.received_tensors,
            assignments=assignments,
            artifacts=artifacts,
            cp_partition_mode=cp_partition_mode,
            decoder_group_ranks_getter=captured_group_ranks,
        )
        owner_entry = owner._seal_ready(owner_entry, ready, tuple(bases))
        _validate_ready(
            owner,
            owner_entry,
            assignments,
            cp_partition_mode,
            registry=_PENDING_OWNERS,
            state=_PREPARED,
        )
        return owner

    def domain_collective(prepared):
        try:
            if prepared is not owner or owner.require_prepared() is not owner:
                raise MdpStateError("MDP: dynamic decoder replay runner retains its owner.")
            _validate_ready(
                owner,
                owner_entry,
                owner_entry[-1].assignments,
                cp_partition_mode,
                registry=_PENDING_OWNERS,
                state=_PREPARED,
            )
            return prepared
        except BaseException as error:
            if isinstance(error, MdpTaskFatalError):
                raise
            raise MdpTaskFatalError(
                "MDP: dynamic decoder replay post-WORLD validation failed."
            ) from error

    try:
        result = run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=2,
            prepare=prepare,
            domain_collective=domain_collective,
            byte_generator=byte_generator,
        )
        if result is not owner or owner_entry is None:
            raise MdpTaskFatalError("MDP: dynamic decoder replay runner returns its exact owner.")
        try:
            _validate_ready(
                owner,
                owner_entry,
                owner_entry[-1].assignments,
                cp_partition_mode,
                registry=_PENDING_OWNERS,
                state=_PREPARED,
            )
            owner._activate(owner_entry)
            owner.require()
        except BaseException as error:
            raise MdpTaskFatalError("MDP: dynamic decoder replay activation failed.") from error
        return owner
    except BaseException as primary:
        if owner is not None and owner_entry is not None:
            try:
                owner._abort_from_entry(owner_entry, primary)
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    primary, f"suppressed dynamic replay cleanup error: {cleanup_error!r}"
                )
        elif owner is not None and (
            (retired := _RETIRED_OWNERS.get(id(owner))) is not None and retired() is owner
        ):
            pass
        elif handoff is not None:
            try:
                handoff.abort(primary)
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    primary, f"suppressed dynamic handoff cleanup error: {cleanup_error!r}"
                )
        else:
            try:
                publication.abort(primary)
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    primary, f"suppressed dynamic publication cleanup error: {cleanup_error!r}"
                )
        raise
