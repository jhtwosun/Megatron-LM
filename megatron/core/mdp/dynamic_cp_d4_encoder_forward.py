# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private Gate0 owner for repeated-D4 selected encoder forward."""

import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp_bridge import dynamic_bridge_split_sizes
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _snapshot_local_authority,
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_d4_embedding_transport import (
    _execute_repeated_d4_embedding,
    _prepare_repeated_d4_embedding,
)
from megatron.core.mdp.dynamic_cp_d4_encoder_execution import _D4EncoderExecutionClaim
from megatron.core.mdp.dynamic_cp_d4_payload_transport import (
    _execute_repeated_d4_decoder_payload,
    _prepare_repeated_d4_decoder_payload,
)
from megatron.core.mdp.dynamic_cp_routing import decoder_payload_split_sizes
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment
from megatron.core.mdp.protocols import DynamicEncoderCpBinding

__all__ = ()

_ACTIVE = object()
_RETIRED = object()
_POISONED = object()
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_ACTIVE_RUNTIME_OWNERS: dict[int, tuple[Any, weakref.ReferenceType[Any]]] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_PUBLICATIONS: dict[int, tuple[Any, ...]] = {}
_RETIRED_PUBLICATIONS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_REPLAY_HANDOFFS: dict[int, tuple[Any, ...]] = {}
_RETIRED_REPLAY_HANDOFFS: dict[int, weakref.ReferenceType[Any]] = {}
_LAYOUT_FIELDS = 11
_OPERATIONS_SEAL = object()


def _add_cleanup_note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


@dataclass(frozen=True, slots=True)
class _D4EncoderForwardOperations:
    allocator: Any = field(repr=False, compare=False)
    acquire: Any = field(repr=False, compare=False)
    release: Any = field(repr=False, compare=False)
    device: torch.device
    params_dtype: torch.dtype
    encoder_domain: Any = field(repr=False, compare=False)
    encoder_ddp: Any = field(repr=False, compare=False)
    raw_encoder: Any = field(repr=False, compare=False)
    zero_grad: Any = field(repr=False, compare=False)
    bind: Any = field(repr=False, compare=False)
    encode: Any = field(repr=False, compare=False)
    payload_width: int
    bridge_width: int
    bridge_dtype: torch.dtype
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self) is not _D4EncoderForwardOperations or self._seal is not _OPERATIONS_SEAL:
            raise MdpConfigurationError(
                "MDP: D4 encoder forward operations are minted by their private snapshot."
            )


def _snapshot_operations(runtime: Any, authority: _DynamicIterationAuthority):
    try:
        allocator = runtime.allocator
        acquire = allocator.acquire
        release = allocator.release
        device = runtime.device
        params_dtype = runtime.params_dtype
        encoder_domain = runtime.encoder_domain
        encoder_ddp = encoder_domain.encoder_ddp
        raw_encoder = encoder_ddp.module
        zero_grad = encoder_ddp.zero_grad_buffer
        adapter = runtime.adapter
        bind = adapter.bind_dynamic_encoder_cp
        encode = adapter.encode
        payload_width = adapter.payload_width
        bridge_width = authority.bridge_width
        bridge_dtype = authority.bridge_dtype
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: D4 encoder forward runtime exposes exact execution operations."
        ) from error
    if (
        not isinstance(device, torch.device)
        or not isinstance(params_dtype, torch.dtype)
        or not callable(acquire)
        or getattr(acquire, "__self__", None) is not allocator
        or not callable(release)
        or getattr(release, "__self__", None) is not allocator
        or not callable(zero_grad)
        or getattr(zero_grad, "__self__", None) is not encoder_ddp
        or not callable(bind)
        or getattr(bind, "__self__", None) is not adapter
        or not callable(encode)
        or getattr(encode, "__self__", None) is not adapter
        or raw_encoder is None
        or type(payload_width) is not int
        or payload_width < 1
        or type(bridge_width) is not int
        or bridge_width < 1
        or not isinstance(bridge_dtype, torch.dtype)
    ):
        raise MdpConfigurationError(
            "MDP: D4 encoder forward runtime exposes sealed DDP and adapter operations."
        )
    return _D4EncoderForwardOperations(
        allocator=allocator,
        acquire=acquire,
        release=release,
        device=device,
        params_dtype=params_dtype,
        encoder_domain=encoder_domain,
        encoder_ddp=encoder_ddp,
        raw_encoder=raw_encoder,
        zero_grad=zero_grad,
        bind=bind,
        encode=encode,
        payload_width=payload_width,
        bridge_width=bridge_width,
        bridge_dtype=bridge_dtype,
        _seal=_OPERATIONS_SEAL,
    )


def _layout_values(layout: EncoderThdLayout) -> tuple[tuple[int, ...], ...]:
    if type(layout) is not EncoderThdLayout:
        raise MdpStateError("MDP: D4 encoder forward layout is exact typed metadata.")
    return tuple(
        (
            segment.global_item_id,
            segment.microbatch_id,
            segment.sample_id,
            segment.image_ordinal,
            segment.payload_row_start,
            segment.payload_rows,
            segment.output_row_start,
            segment.output_rows,
            *segment.grid_thw,
        )
        for segment in layout.segments
    )


def _layout_from_values(values: tuple[tuple[int, ...], ...]) -> EncoderThdLayout:
    if any(
        type(row) is not tuple
        or len(row) != _LAYOUT_FIELDS
        or any(type(value) is not int for value in row)
        for row in values
    ):
        raise MdpStateError("MDP: D4 encoder forward layout wire is exact signed integers.")
    return EncoderThdLayout(
        producer_worker_id=0,
        segments=tuple(
            EncoderThdSegment(
                global_item_id=row[0],
                microbatch_id=row[1],
                sample_id=row[2],
                image_ordinal=row[3],
                payload_row_start=row[4],
                payload_rows=row[5],
                output_row_start=row[6],
                output_rows=row[7],
                grid_thw=(row[8], row[9], row[10]),
            )
            for row in values
        ),
    )


def _validate_layout(
    authority: _DynamicIterationAuthority, layout: EncoderThdLayout
) -> EncoderThdLayout:
    if type(layout) is not EncoderThdLayout or type(layout.segments) is not tuple:
        raise MdpStateError("MDP: D4 encoder forward layout is exact typed metadata.")
    items = authority.global_manifest.items
    if len(layout.segments) != len(items) or any(
        type(segment) is not EncoderThdSegment for segment in layout.segments
    ):
        raise MdpStateError("MDP: D4 encoder forward layout covers exact manifest items.")
    payload_start = 0
    output_start = 0
    for item, segment in zip(items, layout.segments, strict=True):
        payload_rows = item.grid_thw[0] * item.grid_thw[1] * item.grid_thw[2]
        integer_fields = (
            segment.global_item_id,
            segment.microbatch_id,
            segment.sample_id,
            segment.image_ordinal,
            segment.payload_row_start,
            segment.payload_rows,
            segment.output_row_start,
            segment.output_rows,
        )
        if (
            any(type(value) is not int or value < 0 for value in integer_fields)
            or segment.global_item_id != item.item_id.local_item_id
            or segment.image_ordinal != item.image_ordinal
            or segment.grid_thw != item.grid_thw
            or segment.payload_row_start != payload_start
            or segment.payload_rows != payload_rows
            or segment.output_row_start != output_start
            or segment.output_rows != item.output_rows
        ):
            raise MdpStateError("MDP: D4 encoder forward layout matches exact manifest geometry.")
        payload_start += payload_rows
        output_start += item.output_rows
    return layout


class _D4EncoderForwardOwner:
    """One exact owner spanning Gate0 payload exchange and selected forward."""

    __slots__ = (
        "__weakref__",
        "authority",
        "binding",
        "selected_ranks",
        "membership",
        "layout",
        "payload_bundle",
        "output",
        "item_outputs",
        "forward_handle",
        "local_forward_error",
        "text_only",
        "is_selected",
        "is_leader",
        "_runtime",
        "_binding_owner",
        "_trusted_binding_owner",
        "_buffers",
        "_trusted_buffers",
        "_pixels",
        "_source_window",
        "_operations",
        "_state",
        "_prepare_started",
        "_physical_started",
        "_trusted_output",
        "_trusted_item_outputs",
        "_trusted_forward_error",
        "_trusted_forward_handle",
        "_trusted",
    )

    def __init__(self, *, trusted: tuple[Any, ...]) -> None:
        (
            runtime,
            authority,
            binding,
            selected_ranks,
            membership,
            layout,
            text_only,
            is_selected,
            is_leader,
            pixels,
            source_window,
            operations,
        ) = trusted
        self.authority = authority
        self.binding = binding
        self.selected_ranks = selected_ranks
        self.membership = membership
        self.layout = layout
        self.payload_bundle = None
        self.output = None
        self.item_outputs = MappingProxyType({})
        self.local_forward_error = None
        self.forward_handle = None
        self.text_only = text_only
        self.is_selected = is_selected
        self.is_leader = is_leader
        self._runtime = runtime
        self._binding_owner = None
        self._trusted_binding_owner = None
        self._buffers = ()
        self._trusted_buffers = ()
        self._pixels = pixels
        self._source_window = source_window
        self._operations = operations
        self._state = _ACTIVE
        self._prepare_started = False
        self._physical_started = False
        self._trusted_output = None
        self._trusted_item_outputs = self.item_outputs
        self._trusted_forward_error = None
        self._trusted_forward_handle = None
        self._trusted = trusted

    def _activate(self) -> None:
        identity = id(self)
        runtime_identity = id(self._runtime)
        active = _ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
        if active is not None and active[0] is self._runtime and active[1]() is not None:
            raise MdpStateError("MDP: D4 encoder forward runtime already has an active owner.")

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            entry = _ACTIVE_OWNERS.get(identity)
            if entry is not None and entry[0] is reference:
                del _ACTIVE_OWNERS[identity]
            runtime_entry = _ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
            if runtime_entry is not None and runtime_entry[1] is reference:
                del _ACTIVE_RUNTIME_OWNERS[runtime_identity]

        reference = weakref.ref(self, retire)
        _ACTIVE_OWNERS[identity] = (reference, self._trusted)
        _ACTIVE_RUNTIME_OWNERS[runtime_identity] = (self._runtime, reference)

    def require(self) -> "_D4EncoderForwardOwner":
        """Return this exact active owner after sealed-field validation."""
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_OWNERS.get(id(self))
            if retired is not None and retired() is self:
                raise MdpStateError("MDP: D4 encoder forward owner is retired.")
            raise MdpStateError("MDP: D4 encoder forward owner is the exact active owner.")
        trusted = entry[1]
        current = (
            self._runtime,
            self.authority,
            self.binding,
            self.selected_ranks,
            self.membership,
            self.layout,
            self.text_only,
            self.is_selected,
            self.is_leader,
            self._pixels,
            self._source_window,
            self._operations,
        )
        if (
            self._state is not _ACTIVE
            or any(
                actual is not expected for actual, expected in zip(current, trusted, strict=True)
            )
            or self._binding_owner is not self._trusted_binding_owner
            or self._buffers is not self._trusted_buffers
            or self.output is not self._trusted_output
            or self.item_outputs is not self._trusted_item_outputs
            or self.local_forward_error is not self._trusted_forward_error
            or self.forward_handle is not self._trusted_forward_handle
        ):
            raise MdpStateError("MDP: D4 encoder forward owner retains sealed fields.")
        return self

    def _install_operations(self, operations: _D4EncoderForwardOperations) -> None:
        if self._operations is not None or type(operations) is not _D4EncoderForwardOperations:
            raise MdpStateError("MDP: D4 encoder forward operation snapshot installs once.")
        self._operations = operations
        updated = list(self._trusted)
        updated[-1] = operations
        self._trusted = tuple(updated)
        _ACTIVE_OWNERS[id(self)] = (_ACTIVE_OWNERS[id(self)][0], self._trusted)

    def _install_binding(self, binding_owner: DynamicEncoderCpBinding) -> None:
        if self._trusted_binding_owner is not None:
            raise MdpStateError("MDP: D4 encoder forward CP binding installs once.")
        self._binding_owner = self._trusted_binding_owner = binding_owner

    def _install_buffer(self, buffer: torch.Tensor) -> None:
        buffers = (*self._trusted_buffers, buffer)
        self._buffers = self._trusted_buffers = buffers

    def _install_forward_result(
        self, output: Any, item_outputs: MappingProxyType, local_error: BaseException | None
    ) -> None:
        self.output = self._trusted_output = output
        self.item_outputs = self._trusted_item_outputs = item_outputs
        self.local_forward_error = self._trusted_forward_error = local_error
        handle = None
        if local_error is None and self.is_selected:
            iteration = self._runtime._iteration
            if type(iteration) is not int or iteration < 0:
                raise MdpStateError("MDP: D4 encoder forward retains its runtime iteration.")
            handle = EncoderForwardHandle(
                iteration=iteration,
                producer_worker_id=0,
                chunk_outputs=(output,),
                chunk_layouts=(self.layout,),
            )
        self.forward_handle = self._trusted_forward_handle = handle

    def _retire(self, primary: BaseException, *, poisoned: bool) -> None:
        entry = _ACTIVE_OWNERS.pop(id(self), None)
        if entry is None or entry[0]() is not self:
            return
        trusted = entry[1]
        buffers = self._trusted_buffers
        binding_owner = self._trusted_binding_owner
        forward_handle = self._trusted_forward_handle
        pixels = trusted[9]
        operations = trusted[-1]
        identity = id(self)
        runtime = trusted[0]
        runtime_entry = _ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            del _ACTIVE_RUNTIME_OWNERS[id(runtime)]

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            if _RETIRED_OWNERS.get(identity) is reference:
                del _RETIRED_OWNERS[identity]

        _RETIRED_OWNERS[identity] = weakref.ref(self, remove)
        self._state = _POISONED if poisoned else _RETIRED
        self._buffers = None
        self._trusted_buffers = None
        self._binding_owner = None
        self._trusted_binding_owner = None
        self._pixels = None
        self._source_window = None
        self._operations = None
        self.payload_bundle = None
        self.output = None
        self.item_outputs = MappingProxyType({})
        self.local_forward_error = None
        self.forward_handle = None
        self._trusted_output = None
        self._trusted_item_outputs = None
        self._trusted_forward_error = None
        self._trusted_forward_handle = None
        self.authority = None
        self.binding = None
        self.selected_ranks = None
        self.membership = None
        self.layout = None
        self._runtime = None
        self._trusted = ()
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
        for buffer in buffers:
            try:
                operations.release(buffer)
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed D4 forward buffer release error: {error!r}")
        if pixels is not None:
            pixels.clear()

    def abort(self, primary_error: BaseException | None = None) -> None:
        """Retire a prephysical owner and release its exact resources once."""
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: D4 encoder forward abort error is an exception.")
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        try:
            self.require()
        except MdpStateError as integrity_error:
            pass
        else:
            integrity_error = None
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: D4 encoder forward owner was aborted.")
        )
        if integrity_error is not None:
            _add_cleanup_note(primary, f"D4 encoder forward integrity failure: {integrity_error!r}")
        self._retire(primary, poisoned=self._physical_started is True)


class _D4EncoderPublicationOwner:
    """Exact post-Gate1 owner of published embeddings and the retained graph."""

    __slots__ = (
        "__weakref__",
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
        "text_only",
        "is_selected",
        "is_leader",
        "_runtime",
        "_binding_owner",
        "_buffers",
        "_operations",
        "_state",
        "_trusted",
    )

    def __init__(self, trusted: tuple[Any, ...]) -> None:
        (
            runtime,
            authority,
            binding,
            selected_ranks,
            membership,
            layout,
            payload_bundle,
            embedding_bundle,
            output,
            item_outputs,
            forward_handle,
            text_only,
            is_selected,
            is_leader,
            binding_owner,
            buffers,
            operations,
        ) = trusted
        self._runtime = runtime
        self.authority = authority
        self.binding = binding
        self.selected_ranks = selected_ranks
        self.membership = membership
        self.layout = layout
        self.payload_bundle = payload_bundle
        self.embedding_bundle = embedding_bundle
        self.output = output
        self.item_outputs = item_outputs
        self.forward_handle = forward_handle
        self.text_only = text_only
        self.is_selected = is_selected
        self.is_leader = is_leader
        self._binding_owner = binding_owner
        self._buffers = buffers
        self._operations = operations
        self._state = _ACTIVE
        self._trusted = trusted

    def _activate_from(self, predecessor: _D4EncoderForwardOwner) -> None:
        runtime_entry = _ACTIVE_RUNTIME_OWNERS.get(id(self._runtime))
        if (
            runtime_entry is None
            or runtime_entry[0] is not self._runtime
            or runtime_entry[1]() is not predecessor
        ):
            raise MdpStateError("MDP: D4 publication transfers the exact active runtime owner.")
        identity = id(self)
        runtime_identity = id(self._runtime)

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            entry = _ACTIVE_PUBLICATIONS.get(identity)
            if entry is not None and entry[0] is reference:
                del _ACTIVE_PUBLICATIONS[identity]
            runtime_entry = _ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
            if runtime_entry is not None and runtime_entry[1] is reference:
                del _ACTIVE_RUNTIME_OWNERS[runtime_identity]

        reference = weakref.ref(self, retire)
        _ACTIVE_PUBLICATIONS[identity] = (reference, *self._trusted)
        _ACTIVE_RUNTIME_OWNERS[runtime_identity] = (self._runtime, reference)

    def require(self) -> "_D4EncoderPublicationOwner":
        """Return the exact live publication owner after sealed-field validation."""
        entry = _ACTIVE_PUBLICATIONS.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_PUBLICATIONS.get(id(self))
            if retired is not None and retired() is self:
                raise MdpStateError("MDP: D4 encoder publication owner is retired.")
            raise MdpStateError("MDP: D4 encoder publication owner is the exact active owner.")
        current = (
            self._runtime,
            self.authority,
            self.binding,
            self.selected_ranks,
            self.membership,
            self.layout,
            self.payload_bundle,
            self.embedding_bundle,
            self.output,
            self.item_outputs,
            self.forward_handle,
            self.text_only,
            self.is_selected,
            self.is_leader,
            self._binding_owner,
            self._buffers,
            self._operations,
        )
        if self._state is not _ACTIVE or any(
            actual is not expected for actual, expected in zip(current, entry[1:], strict=True)
        ):
            raise MdpStateError("MDP: D4 encoder publication owner retains sealed fields.")
        return self

    def abort(self, primary_error: BaseException | None = None) -> None:
        """Release the retained graph, binding, and transport buffers exactly once."""
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: D4 encoder publication abort error is an exception.")
        entry = _ACTIVE_PUBLICATIONS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        trusted = entry[1:]
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: D4 encoder publication owner was aborted.")
        )
        identity = id(self)
        _ACTIVE_PUBLICATIONS.pop(identity)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            if _RETIRED_PUBLICATIONS.get(identity) is reference:
                del _RETIRED_PUBLICATIONS[identity]

        _RETIRED_PUBLICATIONS[identity] = weakref.ref(self, remove)
        runtime = trusted[0]
        runtime_entry = _ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            del _ACTIVE_RUNTIME_OWNERS[id(runtime)]
        binding_owner, buffers, operations = trusted[-3:]
        forward_handle = trusted[10]
        self._state = _RETIRED
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
            setattr(self, name, None)
        self._trusted = ()
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
        for buffer in buffers:
            try:
                operations.release(buffer)
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed D4 publication buffer release error: {error!r}"
                )

    def _claim_for_replay(
        self, authority: _DynamicIterationAuthority, /
    ) -> "_D4EncoderReplayHandoff":
        """Transfer exact Gate0/1 resources without releasing any of them."""
        self.require()
        if authority is not self.authority:
            raise MdpStateError("MDP: D4 replay handoff requires its exact iteration authority.")
        _snapshot_local_authority(self.binding, authority)
        trusted = _ACTIVE_PUBLICATIONS[id(self)][1:]
        handoff = _D4EncoderReplayHandoff(trusted)
        handoff._activate_from(self)
        return handoff


class _D4EncoderReplayHandoff:
    """Registered one-shot owner of the completed Gate0/1 resources."""

    __slots__ = (
        "__weakref__",
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
        "text_only",
        "is_selected",
        "is_leader",
        "_runtime",
        "_binding_owner",
        "_buffers",
        "_operations",
        "_state",
        "_consumed",
        "_trusted",
    )

    def __init__(self, trusted: tuple[Any, ...]) -> None:
        (
            self._runtime,
            self.authority,
            self.binding,
            self.selected_ranks,
            self.membership,
            self.layout,
            self.payload_bundle,
            self.embedding_bundle,
            self.output,
            self.item_outputs,
            self.forward_handle,
            self.text_only,
            self.is_selected,
            self.is_leader,
            self._binding_owner,
            self._buffers,
            self._operations,
        ) = trusted
        self._state = _ACTIVE
        self._consumed = False
        self._trusted = trusted

    def _activate_from(self, predecessor: _D4EncoderPublicationOwner) -> None:
        runtime = self._runtime
        runtime_entry = _ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        publication_entry = _ACTIVE_PUBLICATIONS.get(id(predecessor))
        if (
            runtime_entry is None
            or runtime_entry[0] is not runtime
            or runtime_entry[1]() is not predecessor
            or publication_entry is None
            or publication_entry[0]() is not predecessor
            or len(publication_entry[1:]) != len(self._trusted)
            or any(
                actual is not expected
                for actual, expected in zip(publication_entry[1:], self._trusted, strict=True)
            )
        ):
            raise MdpStateError("MDP: D4 replay handoff replaces its exact publication owner.")
        identity = id(self)
        runtime_identity = id(runtime)

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            entry = _ACTIVE_REPLAY_HANDOFFS.get(identity)
            if entry is not None and entry[0] is reference:
                del _ACTIVE_REPLAY_HANDOFFS[identity]
            runtime_entry = _ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
            if runtime_entry is not None and runtime_entry[1] is reference:
                del _ACTIVE_RUNTIME_OWNERS[runtime_identity]

        reference = weakref.ref(self, retire)
        _ACTIVE_REPLAY_HANDOFFS[identity] = (reference, *self._trusted, False)
        _ACTIVE_RUNTIME_OWNERS[runtime_identity] = (runtime, reference)
        _ACTIVE_PUBLICATIONS.pop(id(predecessor))
        predecessor_identity = id(predecessor)

        def remove_predecessor(reference: weakref.ReferenceType[Any]) -> None:
            if _RETIRED_PUBLICATIONS.get(predecessor_identity) is reference:
                del _RETIRED_PUBLICATIONS[predecessor_identity]

        _RETIRED_PUBLICATIONS[predecessor_identity] = weakref.ref(predecessor, remove_predecessor)
        predecessor._state = _RETIRED
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
            setattr(predecessor, name, None)
        predecessor._trusted = ()

    def require(self) -> "_D4EncoderReplayHandoff":
        """Validate this exact active handoff without consuming it."""
        entry = _ACTIVE_REPLAY_HANDOFFS.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_REPLAY_HANDOFFS.get(id(self))
            if retired is not None and retired() is self:
                raise MdpStateError("MDP: D4 encoder replay handoff is retired.")
            raise MdpStateError("MDP: D4 encoder replay handoff is the exact active owner.")
        current = (
            self._runtime,
            self.authority,
            self.binding,
            self.selected_ranks,
            self.membership,
            self.layout,
            self.payload_bundle,
            self.embedding_bundle,
            self.output,
            self.item_outputs,
            self.forward_handle,
            self.text_only,
            self.is_selected,
            self.is_leader,
            self._binding_owner,
            self._buffers,
            self._operations,
        )
        if (
            self._state is not _ACTIVE
            or self._consumed is not entry[-1]
            or any(
                actual is not expected
                for actual, expected in zip(current, entry[1:-1], strict=True)
            )
        ):
            raise MdpStateError("MDP: D4 encoder replay handoff retains sealed fields.")
        return self

    def consume(self, authority: _DynamicIterationAuthority, /) -> "_D4EncoderReplayHandoff":
        """Mark the handoff claimed once while retaining runtime ownership."""
        self.require()
        if self._consumed:
            raise MdpStateError("MDP: D4 encoder replay handoff is consumed exactly once.")
        if authority is not self.authority:
            raise MdpStateError("MDP: D4 encoder replay consumes its exact iteration authority.")
        _snapshot_local_authority(self.binding, authority)
        self._consumed = True
        entry = _ACTIVE_REPLAY_HANDOFFS[id(self)]
        _ACTIVE_REPLAY_HANDOFFS[id(self)] = (*entry[:-1], True)
        return self

    def abort(self, primary_error: BaseException | None = None) -> None:
        """Retire once and release every transferred resource exactly once."""
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: D4 encoder replay abort error is an exception.")
        entry = _ACTIVE_REPLAY_HANDOFFS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        integrity_error = None
        try:
            current = (
                object.__getattribute__(self, "_runtime"),
                object.__getattribute__(self, "authority"),
                object.__getattribute__(self, "binding"),
                object.__getattribute__(self, "selected_ranks"),
                object.__getattribute__(self, "membership"),
                object.__getattribute__(self, "layout"),
                object.__getattribute__(self, "payload_bundle"),
                object.__getattribute__(self, "embedding_bundle"),
                object.__getattribute__(self, "output"),
                object.__getattribute__(self, "item_outputs"),
                object.__getattribute__(self, "forward_handle"),
                object.__getattribute__(self, "text_only"),
                object.__getattribute__(self, "is_selected"),
                object.__getattribute__(self, "is_leader"),
                object.__getattribute__(self, "_binding_owner"),
                object.__getattribute__(self, "_buffers"),
                object.__getattribute__(self, "_operations"),
            )
            if (
                object.__getattribute__(self, "_state") is not _ACTIVE
                or object.__getattribute__(self, "_consumed") is not entry[-1]
                or any(
                    actual is not expected
                    for actual, expected in zip(current, entry[1:-1], strict=True)
                )
            ):
                integrity_error = MdpStateError(
                    "MDP: D4 encoder replay handoff retains sealed fields."
                )
        except BaseException as error:
            integrity_error = error
        _ACTIVE_REPLAY_HANDOFFS.pop(id(self))
        trusted = entry[1:-1]
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: D4 encoder replay handoff was aborted.")
        )
        identity = id(self)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            if _RETIRED_REPLAY_HANDOFFS.get(identity) is reference:
                del _RETIRED_REPLAY_HANDOFFS[identity]

        _RETIRED_REPLAY_HANDOFFS[identity] = weakref.ref(self, remove)
        runtime = trusted[0]
        runtime_entry = _ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if runtime_entry is not None and runtime_entry[1]() is self:
            del _ACTIVE_RUNTIME_OWNERS[id(runtime)]
        binding_owner, buffers, operations = trusted[-3:]
        forward_handle = trusted[10]
        self._state = _RETIRED
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
            setattr(self, name, None)
        self._trusted = ()
        if integrity_error is not None:
            _add_cleanup_note(primary, "D4 encoder replay integrity validation failed.")
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
        for buffer in buffers:
            try:
                operations.release(buffer)
            except BaseException as error:
                _add_cleanup_note(primary, f"suppressed D4 replay buffer release error: {error!r}")


def _transfer_to_publication(
    owner: _D4EncoderForwardOwner, embedding_bundle: Any
) -> _D4EncoderPublicationOwner:
    owner.require()
    trusted = (
        owner._runtime,
        owner.authority,
        owner.binding,
        owner.selected_ranks,
        owner.membership,
        owner.layout,
        owner.payload_bundle,
        embedding_bundle,
        owner._trusted_output,
        owner._trusted_item_outputs,
        owner._trusted_forward_handle,
        owner.text_only,
        owner.is_selected,
        owner.is_leader,
        owner._trusted_binding_owner,
        owner._trusted_buffers,
        owner._operations,
    )
    publication = _D4EncoderPublicationOwner(trusted)
    publication._activate_from(owner)
    identity = id(owner)
    _ACTIVE_OWNERS.pop(identity)

    def remove(reference: weakref.ReferenceType[Any]) -> None:
        if _RETIRED_OWNERS.get(identity) is reference:
            del _RETIRED_OWNERS[identity]

    _RETIRED_OWNERS[identity] = weakref.ref(owner, remove)
    pixels = owner._trusted[9]
    owner._state = _RETIRED
    for name in (
        "authority",
        "binding",
        "selected_ranks",
        "membership",
        "layout",
        "payload_bundle",
        "output",
        "item_outputs",
        "local_forward_error",
        "forward_handle",
        "_runtime",
        "_binding_owner",
        "_trusted_binding_owner",
        "_buffers",
        "_trusted_buffers",
        "_pixels",
        "_source_window",
        "_operations",
        "_trusted_output",
        "_trusted_item_outputs",
        "_trusted_forward_error",
        "_trusted_forward_handle",
    ):
        setattr(owner, name, None)
    owner._trusted = ()
    pixels.clear()
    return publication


def _payload_buffers(owner: _D4EncoderForwardOwner) -> MappingProxyType:
    authority = owner.authority
    operations = owner._operations
    buffers = {}
    for dtype in tuple(dict.fromkeys(entry.dtype for entry in authority.payload_ledger.entries)):
        input_splits, output_splits = decoder_payload_split_sizes(
            authority.payload_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            source_rank_by_lane=authority.source_rank_by_lane,
            participant_ranks=authority.participant_ranks,
            dtype=dtype,
            global_rank=owner.binding.global_rank,
        )
        pair = []
        for rows, name in ((sum(input_splits), "send"), (sum(output_splits), "receive")):
            buffer = operations.acquire(
                rows=rows,
                width=0,
                dtype=dtype,
                device=operations.device,
                tag=f"dynamic_cp_gate0_payload_{name}",
            )
            owner._install_buffer(buffer)
            pair.append(buffer)
        buffers[dtype] = tuple(pair)
    return MappingProxyType(buffers)


def _embedding_buffers(owner: _D4EncoderForwardOwner) -> tuple[torch.Tensor, torch.Tensor]:
    authority = owner.authority
    operations = owner._operations
    input_splits, output_splits = dynamic_bridge_split_sizes(
        authority.embedding_ledger,
        reverse_ledger=authority.gradient_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        width=authority.bridge_width,
        dtype=authority.bridge_dtype,
        participant_ranks=authority.participant_ranks,
        global_rank=owner.binding.global_rank,
    )
    buffers = []
    for rows, name in ((sum(input_splits), "send"), (sum(output_splits), "receive")):
        buffer = operations.acquire(
            rows=rows,
            width=0,
            dtype=authority.bridge_dtype,
            device=operations.device,
            tag=f"dynamic_cp_gate1_embedding_{name}",
        )
        owner._install_buffer(buffer)
        buffers.append(buffer)
    return tuple(buffers)


def _prepare_pixels(owner: _D4EncoderForwardOwner) -> torch.Tensor | None:
    if not owner.is_selected:
        return None
    rows = sum(
        item.grid_thw[0] * item.grid_thw[1] * item.grid_thw[2]
        for item in owner.authority.global_manifest.items
    )
    operations = owner._operations
    packed = operations.acquire(
        rows=rows,
        width=operations.payload_width,
        dtype=operations.params_dtype,
        device=operations.device,
        tag="dynamic_cp_gate0_pixels",
    )
    owner._install_buffer(packed)
    if owner.is_leader:
        cursor = 0
        for item in owner.authority.global_manifest.items:
            payload_rows = item.grid_thw[0] * item.grid_thw[1] * item.grid_thw[2]
            source = owner._pixels[item.item_id.local_item_id]
            if (
                type(source) is not torch.Tensor
                or source.shape != (payload_rows, operations.payload_width)
                or source.dtype != operations.params_dtype
                or source.device != operations.device
            ):
                raise MdpStateError("MDP: D4 encoder source pixels match exact manifest geometry.")
            packed[cursor : cursor + payload_rows].copy_(source)
            cursor += payload_rows
    return packed


def run_repeated_d4_encoder_forward(
    runtime: Any,
    claim: _D4EncoderExecutionClaim,
    authority: _DynamicIterationAuthority,
    *,
    all_to_all_single=dist.all_to_all_single,
    broadcast=dist.broadcast,
    byte_generator=None,
) -> _D4EncoderForwardOwner:
    """Own Gate0 and execute decoder payload plus selected encoder forward."""
    if type(claim) is not _D4EncoderExecutionClaim:
        raise MdpConfigurationError("MDP: D4 encoder forward requires an exact execution claim.")
    claim.require()
    if authority is not claim.authority:
        raise MdpStateError("MDP: D4 encoder forward requires its exact joint authority.")
    if runtime is not claim._runtime:
        raise MdpStateError("MDP: D4 encoder forward requires its exact capture runtime.")
    if not callable(all_to_all_single) or not callable(broadcast):
        raise MdpConfigurationError("MDP: D4 encoder forward collective operations are callable.")
    pixels = dict(claim.pixel_sidecar)
    source_window = claim.source_window
    trusted = (
        runtime,
        authority,
        claim.binding,
        claim.selected_ranks,
        claim.membership,
        claim.layout,
        claim.text_only,
        claim.is_selected,
        claim.is_leader,
        pixels,
        source_window,
        None,
    )
    owner = _D4EncoderForwardOwner(trusted=trusted)
    owner._activate()

    try:
        payload_buffers = None
        packed_pixels = None

        def prepare_gate0():
            nonlocal payload_buffers, packed_pixels
            owner.require()
            if owner._prepare_started is not False:
                raise MdpStateError("MDP: D4 encoder forward preparation is one-shot.")
            owner._prepare_started = True
            operations = _snapshot_operations(runtime, authority)
            owner._install_operations(operations)
            claim.abort()
            payload_buffers = _payload_buffers(owner)
            operations.zero_grad()
            if not owner.text_only:
                if owner.is_leader:
                    _validate_layout(authority, owner.layout)
                packed_pixels = _prepare_pixels(owner)
                if owner.is_selected:
                    binding_owner = operations.bind(
                        operations.raw_encoder,
                        membership=owner.membership,
                        global_rank=owner.binding.global_rank,
                    )
                    if type(binding_owner) is not DynamicEncoderCpBinding:
                        raise MdpStateError("MDP: D4 encoder forward receives an exact CP binding.")
                    owner._install_binding(binding_owner)
            owner.payload_bundle = _prepare_repeated_d4_decoder_payload(
                owner.binding,
                authority,
                source_window=source_window,
                buffers_by_dtype=payload_buffers,
                all_to_all_single=all_to_all_single,
            )
            return owner

        def execute_gate0(prepared):
            if prepared is not owner or owner.require() is not owner:
                raise MdpTaskFatalError("MDP: D4 encoder forward runner returned its exact owner.")
            if owner._physical_started is not False:
                raise MdpTaskFatalError("MDP: D4 encoder forward physical callback is one-shot.")
            owner._physical_started = True
            _execute_repeated_d4_decoder_payload(
                owner.binding, owner.payload_bundle, all_to_all_single=all_to_all_single
            )
            if owner.text_only:
                return owner
            if owner.is_selected:
                rows = len(authority.global_manifest.items)
                operations = owner._operations
                layout_wire = operations.acquire(
                    rows=rows,
                    width=_LAYOUT_FIELDS,
                    dtype=torch.int64,
                    device=operations.device,
                    tag="dynamic_cp_gate0_layout",
                )
                owner._install_buffer(layout_wire)
                if owner.is_leader:
                    values = _layout_values(owner.layout)
                    layout_wire.copy_(
                        torch.tensor(values, dtype=torch.int64, device=operations.device)
                    )
                broadcast(layout_wire, src=owner.selected_ranks[0], group=owner.membership.group)
                values = tuple(tuple(int(value) for value in row) for row in layout_wire.tolist())
                received_layout = _validate_layout(authority, _layout_from_values(values))
                if owner.is_leader and received_layout != owner.layout:
                    raise MdpTaskFatalError(
                        "MDP: D4 encoder forward retained its exact source layout."
                    )
                owner.layout = received_layout
                updated = list(owner._trusted)
                updated[5] = received_layout
                owner._trusted = tuple(updated)
                _ACTIVE_OWNERS[id(owner)] = (_ACTIVE_OWNERS[id(owner)][0], owner._trusted)
                broadcast(packed_pixels, src=owner.selected_ranks[0], group=owner.membership.group)
                output = operations.encode(operations.encoder_ddp, packed_pixels, received_layout)
                expected_rows = sum(segment.output_rows for segment in received_layout.segments)
                if (
                    type(output) is not torch.Tensor
                    or output.ndim != 2
                    or output.shape[0] != expected_rows
                    or output.shape[1] != operations.bridge_width
                    or output.dtype != operations.bridge_dtype
                    or output.device != operations.device
                    or not output.is_contiguous()
                    or (expected_rows > 0 and (not output.requires_grad or output.grad_fn is None))
                ):
                    owner._install_forward_result(
                        output,
                        MappingProxyType({}),
                        MdpStateError("MDP: D4 encoder forward output matches exact layout rows."),
                    )
                    return owner
                item_outputs = MappingProxyType({})
                if owner.is_leader:
                    item_outputs = MappingProxyType(
                        {
                            item.item_id: output[
                                segment.output_row_start : segment.output_row_start
                                + segment.output_rows
                            ].detach()
                            for item, segment in zip(
                                authority.global_manifest.items,
                                received_layout.segments,
                                strict=True,
                            )
                        }
                    )
                owner._install_forward_result(output, item_outputs, None)
            return owner

        result = run_repeated_d4_authority_collective(
            owner.binding,
            authority,
            gate_id=0,
            prepare=prepare_gate0,
            domain_collective=execute_gate0,
            byte_generator=byte_generator,
        )
        if result is not owner:
            raise MdpTaskFatalError("MDP: D4 encoder forward runner returned its exact owner.")
        return owner.require()
    except BaseException as error:
        if owner._physical_started is True and not isinstance(error, MdpTaskFatalError):
            fatal = MdpTaskFatalError("MDP: D4 encoder forward physical execution failed.")
            owner._retire(fatal, poisoned=True)
            raise fatal from error
        owner._retire(
            error, poisoned=owner._physical_started is True or isinstance(error, MdpTaskFatalError)
        )
        raise


def run_repeated_d4_encoder_publication(
    owner: _D4EncoderForwardOwner, *, all_to_all_single=dist.all_to_all_single, byte_generator=None
) -> _D4EncoderPublicationOwner:
    """Own Gate1, publish detached embeddings, and transfer the retained graph."""
    if type(owner) is not _D4EncoderForwardOwner:
        raise MdpConfigurationError("MDP: D4 encoder publication requires an exact forward owner.")
    entry = _ACTIVE_OWNERS.get(id(owner))
    if entry is None or entry[0]() is not owner:
        owner.require()
    trusted = entry[1]
    authority, binding = trusted[1], trusted[2]
    prepared_bundle = None
    prepared_outputs = None
    prepared_output = None
    prepare_started = False
    physical_started = False

    try:

        def prepare_gate1():
            nonlocal prepare_started, prepared_bundle, prepared_outputs, prepared_output
            owner.require()
            if prepare_started:
                raise MdpStateError("MDP: D4 encoder publication preparation is one-shot.")
            prepare_started = True
            if owner._trusted_forward_error is not None:
                raise owner._trusted_forward_error
            prepared_outputs = owner._trusted_item_outputs
            prepared_output = owner._trusted_output
            send_buffer, receive_buffer = _embedding_buffers(owner)
            prepared_bundle = _prepare_repeated_d4_embedding(
                binding,
                authority,
                item_outputs=prepared_outputs,
                send_buffer=send_buffer,
                receive_buffer=receive_buffer,
                all_to_all_single=all_to_all_single,
            )
            return owner

        def execute_gate1(prepared):
            nonlocal physical_started
            if prepared is not owner or owner.require() is not owner:
                raise MdpTaskFatalError("MDP: D4 encoder publication runner returned its owner.")
            if (
                owner._trusted_item_outputs is not prepared_outputs
                or owner._trusted_output is not prepared_output
                or prepared_bundle is None
            ):
                raise MdpTaskFatalError("MDP: D4 encoder publication retains prepared outputs.")
            if physical_started:
                raise MdpTaskFatalError(
                    "MDP: D4 encoder publication physical callback is one-shot."
                )
            physical_started = True
            _execute_repeated_d4_embedding(
                binding, prepared_bundle, all_to_all_single=all_to_all_single
            )
            return owner

        result = run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=1,
            prepare=prepare_gate1,
            domain_collective=execute_gate1,
            byte_generator=byte_generator,
        )
        if result is not owner:
            raise MdpTaskFatalError("MDP: D4 encoder publication runner returned its owner.")
        return _transfer_to_publication(owner, prepared_bundle)
    except BaseException as error:
        prepared_output = None
        prepared_outputs = None
        if physical_started and not isinstance(error, MdpTaskFatalError):
            fatal = MdpTaskFatalError("MDP: D4 encoder publication physical execution failed.")
            owner._retire(fatal, poisoned=True)
            raise fatal from error
        owner._retire(error, poisoned=physical_started or isinstance(error, MdpTaskFatalError))
        raise
