# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private retained-P2 D3 owner for a future runtime composition factory."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.plan import EncoderThdLayout
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState

__all__ = ()

_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_PENDING_OWNER_SEALS: dict[object, tuple[int, ...]] = {}
_INT64_MAX = 2**63 - 1


def _descriptor(tensor: torch.Tensor) -> tuple:
    return (
        id(tensor),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tensor.stride(),
    )


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr():
        return False
    left_start = left.storage_offset() * left.element_size()
    right_start = right.storage_offset() * right.element_size()
    left_end = left_start + left.numel() * left.element_size()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


@dataclass(frozen=True, slots=True)
class _PreparedNativeEncoderCompletion:
    """One sealed, unexecuted native encoder completion."""

    owner: Any = field(compare=False, repr=False)
    runtime: MdpRuntime = field(compare=False, repr=False)
    handle: EncoderForwardHandle | None = field(compare=False, repr=False)
    gradient_views: tuple[torch.Tensor, ...] = field(compare=False, repr=False)
    allocation_bases: tuple[torch.Tensor, ...] = field(compare=False, repr=False)
    encoder_cp_follower: bool
    encoder_domain: Any = field(compare=False, repr=False)
    encoder_ddp: Any = field(compare=False, repr=False)
    globally_reduced_num_tokens: torch.Tensor = field(compare=False, repr=False)
    _authority: tuple | None = field(default=None, init=False, compare=False, repr=False)
    _factory_seal: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if (
            type(self) is not _PreparedNativeEncoderCompletion
            or self._factory_seal is None
            or getattr(self.owner, "_mint_token", None) is not self._factory_seal
        ):
            raise MdpStateError("MDP: native encoder completion is minted by its exact factory.")
        self.owner._mint_token = None


def _completion_authority(completion: _PreparedNativeEncoderCompletion) -> tuple:
    return (
        id(completion),
        id(completion.owner),
        id(completion.runtime),
        None if completion.handle is None else id(completion.handle),
        tuple(_descriptor(view) for view in completion.gradient_views),
        tuple(_descriptor(base) for base in completion.allocation_bases),
        completion.encoder_cp_follower,
        id(completion.encoder_domain),
        id(completion.encoder_ddp),
        tuple(id(value) for value in _encoder_ddp_callable_authority(completion.encoder_ddp)),
        _token_descriptor(completion.globally_reduced_num_tokens),
    )


def _mint_completion(
    owner, views, bases, globally_reduced_num_tokens
) -> _PreparedNativeEncoderCompletion:
    token = object()
    owner._mint_token = token
    try:
        completion = _PreparedNativeEncoderCompletion(
            owner,
            owner._runtime,
            owner._handle,
            views,
            bases,
            owner._encoder_cp_follower,
            owner._encoder_domain,
            owner._encoder_ddp,
            globally_reduced_num_tokens,
            _factory_seal=token,
        )
    except BaseException:
        owner._mint_token = None
        raise
    object.__setattr__(completion, "_authority", _completion_authority(completion))
    return completion


class _D3ProducerOwner:
    """Own one retained P2 graph until D3 prepares or aborts it."""

    __slots__ = (
        "_runtime",
        "_owned_runtime",
        "_rank_view",
        "_handle",
        "_outputs",
        "_output_descriptors",
        "_layouts",
        "_geometry",
        "_item_outputs",
        "_item_descriptors",
        "_pixel_bases",
        "_pixel_descriptors",
        "_mint_token",
        "_encoder_cp_follower",
        "_producer",
        "_prepared",
        "_prepared_authority",
        "_state",
        "_producer_input_authority",
        "_encoder_domain",
        "_encoder_ddp",
        "_encoder_ddp_callable_authority",
        "_gradient_bases",
        "_iteration",
        "_finalization_token_authority",
    )

    def __init__(
        self,
        *,
        runtime,
        rank_view,
        handle,
        layouts,
        geometry,
        item_outputs,
        pixel_bases,
        encoder_cp_follower,
        encoder_domain,
        encoder_ddp,
        encoder_ddp_callable_authority,
        _factory_seal=None,
    ) -> None:
        values = (
            runtime,
            rank_view,
            handle,
            layouts,
            geometry,
            item_outputs,
            pixel_bases,
            encoder_cp_follower,
            encoder_domain,
            encoder_ddp,
            encoder_ddp_callable_authority,
        )
        fingerprint = _PENDING_OWNER_SEALS.pop(_factory_seal, None)
        if type(self) is not _D3ProducerOwner or fingerprint != tuple(
            id(value) for value in values
        ):
            raise MdpStateError("MDP: D3 producer owner is minted by its exact factory.")
        self._runtime = self._owned_runtime = runtime
        self._rank_view, self._handle = rank_view, handle
        self._outputs = () if handle is None else tuple(handle.chunk_outputs)
        self._output_descriptors = tuple(_descriptor(value) for value in self._outputs)
        self._layouts, self._geometry, self._item_outputs = layouts, geometry, item_outputs
        self._item_descriptors = tuple(
            (key, _descriptor(value)) for key, value in item_outputs.items()
        )
        self._pixel_bases = pixel_bases
        self._pixel_descriptors = tuple(_descriptor(value) for value in pixel_bases)
        self._encoder_cp_follower = encoder_cp_follower
        self._encoder_domain, self._encoder_ddp = encoder_domain, encoder_ddp
        self._encoder_ddp_callable_authority = encoder_ddp_callable_authority
        self._gradient_bases = ()
        self._iteration = runtime._iteration
        self._finalization_token_authority = None
        self._mint_token = None
        self._producer = self._prepared = self._prepared_authority = None
        self._producer_input_authority = None
        self._state = "registered"
        if (
            type(self._iteration) is not int
            or not 0 <= self._iteration <= _INT64_MAX
            or (handle is not None and handle.iteration != self._iteration)
        ):
            raise MdpStateError("MDP: D3 producer owner captures one exact runtime iteration.")

    @property
    def producer(self):
        if self._producer is None:
            raise MdpStateError("MDP: D3 producer owner no longer retains its producer.")
        return self._producer

    def _mark_pre_authority_dynamic_producer_bound(self, producer, /) -> None:
        """Bind this exact registered owner before its runtime retires the slot."""
        runtime = self._owned_runtime
        if (
            self._state != "registered"
            or self._runtime is not runtime
            or producer is not self._producer
            or runtime._pre_authority_dynamic_producer is not producer
            or producer.owner is not self
        ):
            raise MdpStateError("MDP: D3 producer owner binds its exact registered producer once.")
        self._state = "bound"

    def _validate(self, *, expected_state="bound", backward_done=False) -> MdpRuntime:
        runtime = self._owned_runtime
        if (
            self._state != expected_state
            or type(runtime) is not MdpRuntime
            or self._runtime is not runtime
        ):
            raise MdpStateError("MDP: D3 producer owner is bound exactly once.")
        producer = self.producer
        if (
            runtime._pre_authority_dynamic_producer is not None
            or not runtime._pre_authority_dynamic_producer_is_retired(producer)
        ):
            raise MdpStateError("MDP: D3 producer owner requires its exact consumed producer.")
        if runtime.state is not MdpRuntimeState.EMPTY:
            raise MdpStateError("MDP: D3 producer owner retains the pre-routing P2 state.")
        if (
            type(runtime._iteration) is not int
            or runtime._iteration != self._iteration
            or not 0 <= runtime._iteration <= _INT64_MAX
            or (self._handle is not None and self._handle.iteration != self._iteration)
        ):
            raise MdpStateError("MDP: D3 producer owner retains its exact runtime iteration.")
        _validate_encoder_finalization_domain(
            runtime,
            expected_domain=self._encoder_domain,
            expected_ddp=self._encoder_ddp,
            expected_callable_authority=self._encoder_ddp_callable_authority,
        )
        if runtime.rank_view is not self._rank_view:
            raise MdpStateError("MDP: D3 producer owner retains its exact rank view.")
        if runtime._handle is not self._handle:
            raise MdpStateError("MDP: D3 producer owner retains its exact forward handle.")
        if self._handle is not None and (
            self._handle._backward_done is not backward_done
            or self._handle._released is not False
            or len(self._handle.chunk_outputs) != len(self._outputs)
            or any(a is not b for a, b in zip(self._handle.chunk_outputs, self._outputs))
            or tuple(_descriptor(value) for value in self._outputs) != self._output_descriptors
        ):
            raise MdpStateError("MDP: D3 producer owner retains fresh exact encoder outputs.")
        if (
            type(runtime._chunk_layouts) is not tuple
            or len(runtime._chunk_layouts) != len(self._layouts)
            or any(a is not b for a, b in zip(runtime._chunk_layouts, self._layouts))
            or _capture_geometry(runtime, self._handle, self._layouts) != self._geometry
        ):
            raise MdpStateError("MDP: D3 producer owner retains exact segment geometry.")
        if (
            producer.owner is not self
            or producer.rank_view is not self._rank_view
            or getattr(producer, "_mdp_pre_authority_runtime", None) is not runtime
            or producer.item_outputs is not self._item_outputs
            or _producer_input_authority(producer) != self._producer_input_authority
            or tuple((key, _descriptor(value)) for key, value in self._item_outputs.items())
            != self._item_descriptors
        ):
            raise MdpStateError("MDP: D3 producer owner retains exact producer inputs.")
        if (
            type(runtime._chunk_payload_bases) is not tuple
            or len(runtime._chunk_payload_bases) != len(self._pixel_bases)
            or any(a is not b for a, b in zip(runtime._chunk_payload_bases, self._pixel_bases))
            or tuple(_descriptor(value) for value in self._pixel_bases) != self._pixel_descriptors
        ):
            raise MdpStateError("MDP: D3 producer owner retains exact packed-pixel bases.")
        return runtime

    def _enter_native_encoder_backward(self, completion, /) -> None:
        """Consume the prepared state before any encoder autograd can start."""
        _validate_prepared_native_encoder_completion(completion, owner=self)
        self._state = "backward-entered"

    def _complete_native_encoder_backward(self, completion, /) -> None:
        """Record one returned encoder backward after full retained-state validation."""
        _validate_native_encoder_completion(
            completion, owner=self, expected_state="backward-entered", backward_done=True
        )
        self._state = "backward-complete"

    def _prepare_native_encoder_finalization(self, completion, /) -> tuple:
        """Release every local resource before the WORLD finalization gate."""
        _validate_completed_native_encoder_completion(completion, owner=self)
        runtime, handle = self._owned_runtime, self._handle
        bases, pixels = self._gradient_bases, self._pixel_bases
        token, ddp = completion.globally_reduced_num_tokens, completion.encoder_ddp
        token_authority = _token_descriptor(token)
        self._state = "finalization-preparing"
        errors = []
        if handle is not None:
            runtime._handle = self._handle = None
            try:
                handle.release()
            except BaseException as error:
                errors.append(("encoder forward handle", error))
        for index, base in enumerate(bases):
            self._gradient_bases = bases[index + 1 :]
            try:
                runtime.allocator.release(base)
            except BaseException as error:
                errors.append(("encoder gradient regroup buffer", error))
        for index, base in enumerate(pixels):
            self._pixel_bases = pixels[index + 1 :]
            runtime._chunk_payload_bases = pixels[index + 1 :]
            try:
                runtime.allocator.release(base)
            except BaseException as error:
                errors.append(("packed-pixel buffer", error))
        try:
            if (
                runtime._iteration != self._iteration
                or runtime._captured_num_tokens is not token
                or runtime._token_capture_count != 1
                or runtime._token_consumed is not False
                or _token_descriptor(token) != token_authority
            ):
                raise MdpStateError(
                    "MDP: local cleanup preserves exact iteration and token authority."
                )
            _validate_encoder_finalization_domain(
                runtime,
                expected_domain=self._encoder_domain,
                expected_ddp=ddp,
                expected_callable_authority=self._encoder_ddp_callable_authority,
            )
            if handle is not None and (
                handle._backward_done is not True
                or handle._released is not True
                or handle.chunk_outputs
                or handle.chunk_layouts
            ):
                raise MdpStateError("MDP: local cleanup releases the exact backward handle.")
        except BaseException as error:
            errors.append(("post-release authority validation", error))
        self._scrub_local_completion(completion)
        if errors:
            self._state = "retired"
            primary = errors[0][1]
            for description, error in errors[1:]:
                primary.add_note(f"another cleanup error while releasing {description}: {error!r}")
            try:
                runtime._abort_failed_iteration(primary)
            except BaseException as cleanup_error:
                primary.add_note(f"suppressed D3 finalization reset error: {cleanup_error!r}")
            self._scrub_owner(finalized=False)
            raise primary
        self._state = "finalization-prepared"
        self._finalization_token_authority = token_authority
        self._scrub_owner(finalized=False, preserve_finalization=True)
        return runtime, ddp, token, self._iteration

    def _validate_native_encoder_finalization(self, runtime, ddp, token, iteration, /) -> None:
        if (
            self._state != "finalization-prepared"
            or self._runtime is not runtime
            or self._owned_runtime is not runtime
            or self._encoder_ddp is not ddp
            or self._iteration != iteration
            or runtime._iteration != iteration
            or runtime._captured_num_tokens is not token
            or runtime._token_capture_count != 1
            or runtime._token_consumed is not False
        ):
            raise MdpStateError(
                "MDP: native encoder finalization retains exact prepared authority."
            )
        _validate_encoder_finalization_domain(
            runtime,
            expected_domain=self._encoder_domain,
            expected_ddp=ddp,
            expected_callable_authority=self._encoder_ddp_callable_authority,
        )
        if _token_descriptor(token) != self._finalization_token_authority:
            raise MdpStateError("MDP: native encoder finalization retains its exact token.")

    def _complete_native_encoder_finalization(self, runtime, ddp, token, iteration, /) -> None:
        self._validate_native_encoder_finalization(runtime, ddp, token, iteration)
        runtime._token_consumed = True
        self._state = "retired"
        self._scrub_owner(finalized=True)

    def _scrub_local_completion(self, completion) -> None:
        object.__setattr__(completion, "owner", None)
        object.__setattr__(completion, "handle", None)
        object.__setattr__(completion, "gradient_views", ())
        object.__setattr__(completion, "allocation_bases", ())
        object.__setattr__(completion, "runtime", None)
        object.__setattr__(completion, "encoder_domain", None)
        object.__setattr__(completion, "encoder_ddp", None)
        object.__setattr__(completion, "globally_reduced_num_tokens", None)
        object.__setattr__(completion, "_authority", None)

    def _scrub_owner(self, *, finalized, preserve_finalization=False) -> None:
        if preserve_finalization:
            if self._finalization_token_authority is None:
                raise MdpStateError("MDP: finalization token authority is captured before cleanup.")
        for name, value in (
            ("_rank_view", None),
            ("_outputs", ()),
            ("_output_descriptors", ()),
            ("_layouts", ()),
            ("_geometry", ()),
            ("_item_outputs", MappingProxyType({})),
            ("_item_descriptors", ()),
            ("_pixel_descriptors", ()),
            ("_producer", None),
            ("_prepared", None),
            ("_prepared_authority", None),
            ("_mint_token", None),
            ("_producer_input_authority", None),
            ("_gradient_bases", ()),
        ):
            setattr(self, name, value)
        if finalized or not preserve_finalization:
            self._runtime = self._owned_runtime = None
            self._encoder_domain = self._encoder_ddp = None
            self._encoder_ddp_callable_authority = ()
            self._finalization_token_authority = None

    def prepare_dynamic_completion(self, native_gradients: Mapping, /, *, transport_dtype=None):
        """Regroup exact item gradients without invoking encoder backward."""
        bases = []
        try:
            runtime = self._validate()
            if self._prepared is not None:
                raise MdpStateError("MDP: native encoder completion is prepared exactly once.")
            globally_reduced_num_tokens = _validate_global_token_capture(runtime)
            gradients = _validate_gradients(
                native_gradients, self._item_outputs, transport_dtype=transport_dtype
            )
            views = []
            forbidden = tuple(gradients.values()) + self._outputs + self._pixel_bases
            for output, layout in zip(self._outputs, self._layouts):
                base = runtime.allocator.acquire(
                    rows=layout.total_output_rows,
                    width=output.shape[1],
                    dtype=output.dtype,
                    device=output.device,
                    tag="dynamic_cp_grad_regroup",
                )
                bases.append(base)
                _validate_base(
                    base, output, layout.total_output_rows, forbidden + tuple(bases[:-1])
                )
                view = base[: layout.total_output_rows]
                view.zero_()
                views.append(view)
            by_item = {item: (chunk, start, rows) for item, chunk, _, start, rows in self._geometry}
            for item, gradient in gradients.items():
                chunk, start, rows = by_item[item]
                views[chunk][start : start + rows].copy_(gradient)
            completion = _mint_completion(
                self, tuple(views), tuple(bases), globally_reduced_num_tokens
            )
        except BaseException as error:
            if self._state in ("registered", "bound"):
                self._retire(error, tuple(bases))
            raise
        self._gradient_bases = tuple(bases)
        self._prepared, self._prepared_authority = completion, completion._authority
        return completion

    def abort(self, primary_error: BaseException | None = None, /) -> None:
        if (
            self._state
            not in (
                "registered",
                "bound",
                "backward-entered",
                "backward-complete",
                "finalization-prepared",
            )
            or self._runtime is None
        ):
            raise MdpStateError("MDP: D3 producer owner aborts exactly once.")
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: D3 producer owner primary error is an exception.")
        error = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: D3 producer owner aborted its iteration.")
        )
        self._retire(error, self._gradient_bases)

    def _retire(self, error: BaseException, bases: tuple) -> None:
        runtime, completion = self._owned_runtime, self._prepared
        self._state = "retired"
        try:
            if runtime._handle is not self._handle:
                error.add_note("D3 cleanup ignored a substituted runtime forward handle.")
            if (
                type(runtime._chunk_payload_bases) is not tuple
                or len(runtime._chunk_payload_bases) != len(self._pixel_bases)
                or any(a is not b for a, b in zip(runtime._chunk_payload_bases, self._pixel_bases))
            ):
                error.add_note("D3 cleanup ignored substituted runtime packed-pixel bases.")
            runtime._handle, runtime._chunk_payload_bases = self._handle, self._pixel_bases
            try:
                runtime._abort_failed_iteration(
                    error,
                    cleanup_actions=tuple(
                        (
                            "releasing D3 encoder gradient regroup buffer",
                            lambda base=base: runtime.allocator.release(base),
                        )
                        for base in bases
                    ),
                )
            except BaseException as cleanup_error:
                error.add_note(f"suppressed D3 owner cleanup error: {cleanup_error!r}")
        finally:
            if completion is not None:
                object.__setattr__(completion, "handle", None)
                object.__setattr__(completion, "gradient_views", ())
                object.__setattr__(completion, "allocation_bases", ())
                object.__setattr__(completion, "runtime", None)
                object.__setattr__(completion, "encoder_domain", None)
                object.__setattr__(completion, "encoder_ddp", None)
                object.__setattr__(completion, "globally_reduced_num_tokens", None)
                object.__setattr__(completion, "_authority", None)
            for name, value in (
                ("_runtime", None),
                ("_owned_runtime", None),
                ("_handle", None),
                ("_outputs", ()),
                ("_output_descriptors", ()),
                ("_layouts", ()),
                ("_geometry", ()),
                ("_item_outputs", MappingProxyType({})),
                ("_item_descriptors", ()),
                ("_pixel_bases", ()),
                ("_pixel_descriptors", ()),
                ("_producer", None),
                ("_prepared", None),
                ("_prepared_authority", None),
                ("_mint_token", None),
                ("_producer_input_authority", None),
                ("_encoder_domain", None),
                ("_encoder_ddp", None),
                ("_encoder_ddp_callable_authority", ()),
                ("_gradient_bases", ()),
                ("_finalization_token_authority", None),
            ):
                setattr(self, name, value)


def _capture_geometry(runtime, handle, layouts) -> tuple:
    if handle is None:
        if layouts or runtime._chunk_of_item:
            raise MdpStateError("MDP: empty producer has no retained encoder geometry.")
        return ()
    if type(handle) is not EncoderForwardHandle or handle.consumed:
        raise MdpStateError("MDP: producer owner requires an unconsumed forward handle.")
    if len(layouts) != len(handle.chunk_layouts) or any(
        type(layout) is not EncoderThdLayout or layout is not handle.chunk_layouts[index]
        for index, layout in enumerate(layouts)
    ):
        raise MdpStateError("MDP: producer layouts match the exact forward handle.")
    geometry = []
    for chunk, (layout, output) in enumerate(zip(layouts, handle.chunk_outputs)):
        if output.ndim != 2 or output.shape[0] != layout.total_output_rows:
            raise MdpStateError("MDP: producer chunk output matches its layout rows.")
        cursor = 0
        for segment in layout.segments:
            start, rows = segment.output_row_start, segment.output_rows
            if type(start) is not int or type(rows) is not int or start != cursor or rows <= 0:
                raise MdpStateError(
                    "MDP: producer segment rows are positive, disjoint, and contiguous."
                )
            cursor += rows
            if cursor > output.shape[0]:
                raise MdpStateError("MDP: producer segment rows stay within their chunk output.")
            location = runtime._chunk_of_item.get(segment.global_item_id)
            if (
                type(segment.global_item_id) is not int
                or type(location) is not tuple
                or len(location) != 2
                or location[0] != chunk
                or location[1] is not segment
            ):
                raise MdpStateError("MDP: producer segment map preserves exact plan order.")
            geometry.append((segment.global_item_id, chunk, id(segment), start, rows))
        if cursor != output.shape[0]:
            raise MdpStateError("MDP: producer segments exactly cover their chunk output.")
    if tuple(runtime._chunk_of_item) != tuple(value[0] for value in geometry):
        raise MdpStateError("MDP: producer item map has exact canonical order.")
    return tuple(geometry)


def _validate_outputs(handle, geometry, item_outputs, follower, device) -> None:
    expected = tuple(value[0] for value in geometry)
    if handle is None:
        if item_outputs or follower:
            raise MdpStateError("MDP: empty producer has no retained encoder state.")
        return
    if any(output.device != device for output in handle.chunk_outputs):
        raise MdpStateError("MDP: encoder outputs use the runtime CUDA device.")
    if follower:
        if item_outputs:
            raise MdpStateError("MDP: encoder-CP follower exposes no public item outputs.")
        return
    if tuple(item_outputs) != expected:
        raise MdpStateError("MDP: producer outputs exactly cover canonical local item order.")
    for item, chunk, _segment, start, rows in geometry:
        output = item_outputs[item]
        reference = handle.chunk_outputs[chunk][start : start + rows].detach()
        if (
            not isinstance(output, torch.Tensor)
            or output.requires_grad
            or output.grad_fn is not None
            or not output.is_contiguous()
            or _descriptor(output)[1:] != _descriptor(reference)[1:]
        ):
            raise MdpStateError("MDP: item output is the exact detached chunk storage view.")


def _validate_gradients(gradients, outputs, *, transport_dtype=None):
    if not isinstance(gradients, _MAPPING_PROXY_TYPE):
        raise MdpConfigurationError("MDP: native encoder gradients are an immutable mapping.")
    if transport_dtype not in (None, torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise MdpConfigurationError("MDP: gradient transport dtype is floating-point.")
    if tuple(gradients) != tuple(outputs) or any(type(key) is not int for key in gradients):
        raise MdpStateError("MDP: native gradients have exact canonical item coverage.")
    values = tuple(gradients.values())
    for item, gradient in gradients.items():
        output = outputs[item]
        expected_dtype = output.dtype if transport_dtype is None else transport_dtype
        if (
            not isinstance(gradient, torch.Tensor)
            or gradient.shape != output.shape
            or gradient.dtype != expected_dtype
            or gradient.device != output.device
            or gradient.requires_grad
            or gradient.grad_fn is not None
            or not gradient.is_contiguous()
        ):
            raise MdpStateError("MDP: native gradient is detached and matches its item output.")
    if any(
        _overlaps(value, peer)
        for index, value in enumerate(values)
        for peer in values[index + 1 :] + tuple(outputs.values())
    ):
        raise MdpStateError("MDP: native gradient does not overlap peers or encoder outputs.")
    return gradients


def _validate_base(base, output, rows, forbidden) -> None:
    if (
        not isinstance(base, torch.Tensor)
        or base.ndim != 2
        or tuple(base.shape) != (rows, output.shape[1])
        or base.dtype != output.dtype
        or base.device != output.device
        or base.requires_grad
        or base.grad_fn is not None
        or not base.is_contiguous()
        or any(_overlaps(base, value) for value in forbidden)
    ):
        raise MdpStateError("MDP: encoder gradient allocation is exact and non-aliased.")


def _sample_location_authority(locations) -> tuple:
    if not isinstance(locations, _MAPPING_PROXY_TYPE):
        raise MdpStateError("MDP: producer sample locations are an immutable mapping.")
    canonical_locations = []
    for key, value in locations.items():
        if (
            type(key) is not GlobalSampleId
            or type(value) is not tuple
            or len(value) != 2
            or any(type(component) is not int or component < 0 for component in value)
        ):
            raise MdpStateError("MDP: producer sample locations use canonical integer tuples.")
        canonical_locations.append((key.to_wire_tuple(), value))
    return id(locations), tuple(canonical_locations)


def _producer_input_authority(producer) -> tuple:
    return (
        id(producer.rank_view),
        id(producer.local_manifest),
        id(producer.source_window),
        id(producer.static_plan),
        id(producer.item_outputs),
        *_sample_location_authority(producer.sample_location_by_id),
    )


def _callable_authority(value) -> tuple:
    bound_self = getattr(value, "__self__", None)
    bound_function = getattr(value, "__func__", None)
    return (bound_self, bound_function) if bound_function is not None else (None, value)


def _encoder_ddp_callable_authority(ddp) -> tuple:
    return (
        *_callable_authority(getattr(ddp, "finish_grad_sync", None)),
        *_callable_authority(getattr(ddp, "scale_gradients", None)),
    )


def _validate_encoder_finalization_domain(
    runtime, *, expected_domain=None, expected_ddp=None, expected_callable_authority=None
) -> tuple:
    domain = getattr(runtime, "encoder_domain", None)
    ddp = getattr(domain, "encoder_ddp", None)
    callable_authority = _encoder_ddp_callable_authority(ddp)
    if (
        domain is None
        or ddp is None
        or (expected_domain is not None and domain is not expected_domain)
        or (expected_ddp is not None and ddp is not expected_ddp)
        or not callable(getattr(ddp, "finish_grad_sync", None))
        or not callable(getattr(ddp, "scale_gradients", None))
        or (
            expected_callable_authority is not None
            and (
                len(callable_authority) != len(expected_callable_authority)
                or any(
                    value is not expected
                    for value, expected in zip(callable_authority, expected_callable_authority)
                )
            )
        )
    ):
        raise MdpConfigurationError(
            "MDP: D3 producer owner retains its exact encoder DDP finalization surface."
        )
    return domain, ddp, callable_authority


def _token_descriptor(token: torch.Tensor) -> tuple:
    return (*_descriptor(token), token._version)


def _validate_global_token_capture(runtime) -> torch.Tensor:
    token = getattr(runtime, "_captured_num_tokens", None)
    if (
        type(getattr(runtime, "_token_capture_count", None)) is not int
        or runtime._token_capture_count != 1
        or getattr(runtime, "_token_consumed", None) is not False
        or not isinstance(token, torch.Tensor)
        or token.numel() != 1
        or token.device != runtime.device
        or token.requires_grad
        or token.grad_fn is not None
    ):
        raise MdpStateError(
            "MDP: D3 encoder completion requires one detached unconsumed global token tensor."
        )
    return token


def _validate_prepared_native_encoder_completion(completion, *, owner):
    return _validate_native_encoder_completion(
        completion, owner=owner, expected_state="bound", backward_done=False
    )


def _validate_completed_native_encoder_completion(completion, *, owner):
    return _validate_native_encoder_completion(
        completion, owner=owner, expected_state="backward-complete", backward_done=True
    )


def _validate_native_encoder_completion(completion, *, owner, expected_state, backward_done):
    if (
        type(owner) is not _D3ProducerOwner
        or type(completion) is not _PreparedNativeEncoderCompletion
    ):
        raise MdpConfigurationError("MDP: native completion has exact private types.")
    runtime = owner._validate(expected_state=expected_state, backward_done=backward_done)
    domain, ddp, _ = _validate_encoder_finalization_domain(
        runtime,
        expected_domain=owner._encoder_domain,
        expected_ddp=owner._encoder_ddp,
        expected_callable_authority=owner._encoder_ddp_callable_authority,
    )
    token = _validate_global_token_capture(runtime)
    try:
        completion_authority = _completion_authority(completion)
    except BaseException as error:
        raise MdpStateError("MDP: native completion retains valid exact resources.") from error
    if (
        owner._prepared is not completion
        or completion.owner is not owner
        or completion.runtime is not owner._runtime
        or completion.handle is not owner._handle
        or completion.encoder_domain is not domain
        or completion.encoder_ddp is not ddp
        or completion.globally_reduced_num_tokens is not token
        or completion._authority != owner._prepared_authority
        or completion._authority != completion_authority
        or len(completion.gradient_views) != len(owner._layouts)
        or len(completion.allocation_bases) != len(owner._layouts)
        or len(completion.allocation_bases) != len(owner._gradient_bases)
        or any(
            value is not expected
            for value, expected in zip(completion.allocation_bases, owner._gradient_bases)
        )
    ):
        raise MdpStateError("MDP: native encoder completion matches its exact owner and seal.")
    for view, base, output, layout in zip(
        completion.gradient_views, completion.allocation_bases, owner._outputs, owner._layouts
    ):
        _validate_base(base, output, layout.total_output_rows, owner._outputs + owner._pixel_bases)
        if _descriptor(view)[1:] != _descriptor(base)[1:]:
            raise MdpStateError("MDP: native completion retains exact gradient views.")
    return completion


def _capture_d3_producer_owner(
    *,
    runtime,
    rank_view,
    local_manifest,
    source_window,
    static_plan,
    item_outputs,
    sample_location_by_id,
    forward_only,
    encoder_cp_follower=False,
):
    """Capture a retained-P2 producer for a future D3 composition factory."""
    if type(runtime) is not MdpRuntime or runtime.state is not MdpRuntimeState.EMPTY:
        raise MdpConfigurationError("MDP: D3 owner requires an exact pre-routing P2 MdpRuntime.")
    if rank_view is not runtime.rank_view:
        raise MdpConfigurationError("MDP: D3 owner requires the runtime's exact rank view.")
    if type(forward_only) is not bool or forward_only or type(encoder_cp_follower) is not bool:
        raise MdpConfigurationError("MDP: D3 owner supports exact training roles only.")
    if not isinstance(item_outputs, Mapping) or not isinstance(sample_location_by_id, Mapping):
        raise MdpConfigurationError("MDP: D3 owner inputs are mappings.")
    outputs, locations = MappingProxyType(dict(item_outputs)), MappingProxyType(
        dict(sample_location_by_id)
    )
    _sample_location_authority(locations)
    encoder_domain, encoder_ddp, encoder_ddp_callable_authority = (
        _validate_encoder_finalization_domain(runtime)
    )
    handle, layouts = runtime._handle, tuple(runtime._chunk_layouts)
    geometry = _capture_geometry(runtime, handle, layouts)
    _validate_outputs(handle, geometry, outputs, encoder_cp_follower, runtime.device)
    metadata = (local_manifest, source_window, static_plan)
    contributor = all(value is not None for value in metadata) and bool(locations)
    noncontributor = not any(value is not None for value in metadata) and not locations
    if not (contributor or noncontributor):
        raise MdpConfigurationError(
            "MDP: producer owns complete local source metadata or no source metadata."
        )
    if encoder_cp_follower and contributor:
        raise MdpConfigurationError("MDP: encoder-CP follower owns no source metadata.")
    if outputs and not contributor:
        raise MdpConfigurationError("MDP: encoder outputs require local source metadata.")
    pixel_bases = tuple(runtime._chunk_payload_bases)
    values = (
        runtime,
        rank_view,
        handle,
        layouts,
        geometry,
        outputs,
        pixel_bases,
        encoder_cp_follower,
        encoder_domain,
        encoder_ddp,
        encoder_ddp_callable_authority,
    )
    token = object()
    _PENDING_OWNER_SEALS[token] = tuple(id(value) for value in values)
    try:
        owner = _D3ProducerOwner(
            runtime=runtime,
            rank_view=rank_view,
            handle=handle,
            layouts=layouts,
            geometry=geometry,
            item_outputs=outputs,
            pixel_bases=pixel_bases,
            encoder_cp_follower=encoder_cp_follower,
            encoder_domain=encoder_domain,
            encoder_ddp=encoder_ddp,
            encoder_ddp_callable_authority=encoder_ddp_callable_authority,
            _factory_seal=token,
        )
    except BaseException:
        _PENDING_OWNER_SEALS.pop(token, None)
        raise
    producer = runtime._capture_pre_authority_dynamic_producer(
        owner=owner,
        rank_view=rank_view,
        local_manifest=local_manifest,
        source_window=source_window,
        static_plan=static_plan,
        item_outputs=outputs,
        sample_location_by_id=locations,
        local_prepare_error=None,
        forward_only=False,
    )
    owner._item_outputs = producer.item_outputs
    owner._producer = producer
    owner._producer_input_authority = _producer_input_authority(producer)
    return owner
