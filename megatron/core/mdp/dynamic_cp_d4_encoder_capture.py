# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private capture-only ownership for repeated-D4 encoder preparation."""

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderSourceManifest,
    DecoderSourceWindow,
    validate_decoder_source_manifest,
    validate_decoder_source_window,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.vision_locator import (
    VisionLocatorCatalog,
    build_vision_locator_catalog,
    validate_vision_locator_catalog,
)
from megatron.core.mdp.window import MdpIterationWindow

__all__ = ()

_OPERATIONS_SEAL = object()
_ACTIVE_OWNER = object()
_RETIRED_OWNER = object()
_OPERATION_SNAPSHOTS: dict[int, tuple[Any, ...]] = {}
_PENDING_OWNER_SEALS: dict[object, tuple[int, int]] = {}


def _add_cleanup_note(primary: BaseException, note: str) -> None:
    try:
        primary.add_note(note)
    except BaseException:
        pass


@dataclass(frozen=True)
class _D4EncoderCaptureOperations:
    """Immutable callable snapshot; capture never rereads a live adapter or codec."""

    get_batch: Callable[[Any], Any] = field(repr=False, compare=False)
    estimate_cost: Callable[[Any], int] = field(repr=False, compare=False)
    build_source_window: Callable[..., Any] = field(repr=False, compare=False)
    spatial_merge_size: int
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self) is not _D4EncoderCaptureOperations
            or self._seal is not _OPERATIONS_SEAL
            or not callable(self.get_batch)
            or not callable(self.estimate_cost)
            or not callable(self.build_source_window)
            or type(self.spatial_merge_size) is not int
            or self.spatial_merge_size < 1
        ):
            raise MdpConfigurationError(
                "MDP: D4 encoder capture operations are an exact immutable snapshot."
            )


class _CaptureAdapter:
    __slots__ = ("get_batch", "estimate_cost", "spatial_merge_size")

    def __init__(self, operations: _D4EncoderCaptureOperations) -> None:
        self.get_batch = operations.get_batch
        self.estimate_cost = operations.estimate_cost
        self.spatial_merge_size = operations.spatial_merge_size


def _snapshot_d4_encoder_capture_operations(
    adapter: Any, codec: Any
) -> _D4EncoderCaptureOperations:
    """Snapshot the only adapter/codec operations admitted by capture-only E2a."""
    try:
        get_batch = adapter.get_batch
        estimate_cost = adapter.estimate_cost
        spatial_merge_size = adapter.spatial_merge_size
        build_source_window = codec.build_source_window_with_locations
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: D4 encoder capture adapter and codec expose their required operations."
        ) from error
    if (
        not callable(get_batch)
        or not callable(estimate_cost)
        or not callable(build_source_window)
        or type(spatial_merge_size) is not int
        or spatial_merge_size < 1
    ):
        raise MdpConfigurationError(
            "MDP: D4 encoder capture adapter and codec expose valid immutable operations."
        )
    operations = _D4EncoderCaptureOperations(
        get_batch=get_batch,
        estimate_cost=estimate_cost,
        build_source_window=build_source_window,
        spatial_merge_size=spatial_merge_size,
        _seal=_OPERATIONS_SEAL,
    )
    identity = id(operations)

    def retire(reference: weakref.ReferenceType[Any]) -> None:
        entry = _OPERATION_SNAPSHOTS.get(identity)
        if entry is not None and entry[0] is reference:
            del _OPERATION_SNAPSHOTS[identity]

    reference = weakref.ref(operations, retire)
    _OPERATION_SNAPSHOTS[identity] = (
        reference,
        get_batch,
        estimate_cost,
        build_source_window,
        spatial_merge_size,
    )
    return operations


def _validate_operations(value: Any) -> _D4EncoderCaptureOperations:
    entry = _OPERATION_SNAPSHOTS.get(id(value))
    if (
        type(value) is not _D4EncoderCaptureOperations
        or entry is None
        or entry[0]() is not value
        or value._seal is not _OPERATIONS_SEAL
        or value.get_batch is not entry[1]
        or value.estimate_cost is not entry[2]
        or value.build_source_window is not entry[3]
        or type(value.spatial_merge_size) is not int
        or value.spatial_merge_size != entry[4]
    ):
        raise MdpStateError("MDP: D4 encoder capture uses exact snapshotted operations.")
    return value


class _D4EncoderCaptureOwner:
    """One all-rank, one-shot owner for pre-consensus encoder source capture."""

    __slots__ = (
        "__weakref__",
        "_runtime",
        "_binding",
        "_source_window",
        "_local_manifest",
        "_sample_locations",
        "_pixel_sidecar",
        "_capture_mode",
        "_locator_catalog",
        "_local_prepare_error",
        "_state",
        "_trusted_runtime",
        "_trusted_binding",
        "_trusted_source_window",
        "_trusted_manifest",
        "_trusted_locations",
        "_trusted_pixels",
        "_trusted_pixel_view",
        "_trusted_capture_mode",
        "_trusted_locator_catalog",
        "_trusted_error",
    )

    def __init__(
        self,
        runtime: MdpRuntime,
        binding: _RepeatedD4GroupBinding,
        capture_mode: VisionCaptureMode = VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
        *,
        _factory_seal: object,
    ) -> None:
        if _PENDING_OWNER_SEALS.pop(_factory_seal, None) != (
            id(runtime),
            id(binding),
            capture_mode,
        ):
            raise MdpConfigurationError(
                "MDP: D4 encoder capture owner is minted by its private factory."
            )
        empty_locations = MappingProxyType({})
        empty_pixels = {}
        empty_pixel_view = MappingProxyType(empty_pixels)
        empty_catalog = build_vision_locator_catalog((), ())
        self._runtime = self._trusted_runtime = runtime
        self._binding = self._trusted_binding = binding
        self._source_window = self._trusted_source_window = None
        self._local_manifest = self._trusted_manifest = None
        self._sample_locations = self._trusted_locations = empty_locations
        self._pixel_sidecar = self._trusted_pixel_view = empty_pixel_view
        self._capture_mode = self._trusted_capture_mode = capture_mode
        self._locator_catalog = self._trusted_locator_catalog = empty_catalog
        self._local_prepare_error = self._trusted_error = None
        self._trusted_pixels = empty_pixels
        self._state = _ACTIVE_OWNER

    def _install_source(
        self,
        *,
        source_window: DecoderSourceWindow,
        local_manifest: DecoderSourceManifest,
        sample_locations: Mapping[GlobalSampleId, tuple[int, int]],
        pixels: dict[int, torch.Tensor],
        locator_catalog: VisionLocatorCatalog,
    ) -> None:
        locations = MappingProxyType(dict(sample_locations))
        pixel_view = MappingProxyType(pixels)
        self._source_window = self._trusted_source_window = source_window
        self._local_manifest = self._trusted_manifest = local_manifest
        self._sample_locations = self._trusted_locations = locations
        self._pixel_sidecar = self._trusted_pixel_view = pixel_view
        self._trusted_pixels = pixels
        self._locator_catalog = self._trusted_locator_catalog = locator_catalog

    def _install_error(self, error: Exception) -> None:
        self._local_prepare_error = self._trusted_error = error

    def _integrity_error(self) -> MdpStateError | None:
        if (
            self._state is not _ACTIVE_OWNER
            or self._runtime is not self._trusted_runtime
            or self._binding is not self._trusted_binding
            or self._source_window is not self._trusted_source_window
            or self._local_manifest is not self._trusted_manifest
            or self._sample_locations is not self._trusted_locations
            or self._pixel_sidecar is not self._trusted_pixel_view
            or self._capture_mode is not self._trusted_capture_mode
            or self._locator_catalog is not self._trusted_locator_catalog
            or self._local_prepare_error is not self._trusted_error
        ):
            return MdpStateError("MDP: D4 encoder capture retains its sealed capture fields.")
        return None

    def require(self) -> "_D4EncoderCaptureOwner":
        """Validate without consuming this exact active owner."""
        runtime = self._trusted_runtime
        if runtime is None:
            raise MdpStateError("MDP: D4 encoder capture owner is retired.")
        runtime._require_d4_encoder_capture_owner(self)
        error = self._integrity_error()
        if error is not None:
            raise error
        return self

    @property
    def binding(self) -> _RepeatedD4GroupBinding:
        self.require()
        return self._trusted_binding

    @property
    def source_window(self) -> DecoderSourceWindow | None:
        self.require()
        return self._trusted_source_window

    @property
    def local_manifest(self) -> DecoderSourceManifest | None:
        self.require()
        return self._trusted_manifest

    @property
    def sample_locations(self) -> Mapping[GlobalSampleId, tuple[int, int]]:
        self.require()
        return self._trusted_locations

    @property
    def pixel_sidecar(self) -> Mapping[int, torch.Tensor]:
        self.require()
        return self._trusted_pixel_view

    @property
    def capture_mode(self) -> VisionCaptureMode:
        self.require()
        return self._trusted_capture_mode

    @property
    def locator_catalog(self) -> VisionLocatorCatalog:
        self.require()
        return self._trusted_locator_catalog

    @property
    def local_prepare_error(self) -> Exception | None:
        self.require()
        return self._trusted_error

    def abort(self, primary_error: BaseException | None = None, /) -> None:
        """Retire once, then release only the exact escrowed source pixels."""
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: D4 encoder capture abort error is an exception.")
        runtime = self._trusted_runtime
        if runtime is None:
            raise MdpStateError("MDP: D4 encoder capture owner is retired.")
        try:
            runtime._require_d4_encoder_capture_owner(self)
        except MdpStateError as validation_error:
            if runtime._d4_encoder_capture_trusted_owner is not self:
                raise
            integrity_error = validation_error
        else:
            integrity_error = self._integrity_error()
        error = (
            primary_error
            if primary_error is not None
            else (
                self._trusted_error
                if self._trusted_error is not None
                else MdpStateError("MDP: D4 encoder capture owner aborted its capture.")
            )
        )
        if integrity_error is not None:
            _add_cleanup_note(error, f"capture integrity failure: {integrity_error!r}")
        pixels = self._trusted_pixels
        runtime._retire_d4_encoder_capture_owner(self)
        self._state = _RETIRED_OWNER
        for name, value in (
            ("_runtime", None),
            ("_binding", None),
            ("_source_window", None),
            ("_local_manifest", None),
            ("_sample_locations", None),
            ("_pixel_sidecar", None),
            ("_capture_mode", None),
            ("_locator_catalog", None),
            ("_local_prepare_error", None),
            ("_trusted_runtime", None),
            ("_trusted_binding", None),
            ("_trusted_source_window", None),
            ("_trusted_manifest", None),
            ("_trusted_locations", None),
            ("_trusted_pixels", None),
            ("_trusted_pixel_view", None),
            ("_trusted_capture_mode", None),
            ("_trusted_locator_catalog", None),
            ("_trusted_error", None),
        ):
            setattr(self, name, value)
        try:
            pixels.clear()
        except BaseException as cleanup_error:
            _add_cleanup_note(
                error, f"suppressed D4 encoder pixel release error: {cleanup_error!r}"
            )

    def _claim_for_execution(self, authority: Any, /) -> tuple[Any, ...]:
        """Transfer exact capture resources after joint-authority validation."""
        from megatron.core.mdp.dynamic_cp_runtime import (
            _dynamic_iteration_plan_digest,
            _DynamicIterationAuthority,
        )

        self.require()
        if (
            type(authority) is not _DynamicIterationAuthority
            or authority.encoder_plan is None
            or authority.joint_plan_digest is None
        ):
            raise MdpStateError("MDP: encoder execution claim requires exact joint authority.")
        _dynamic_iteration_plan_digest(authority)
        binding = self._trusted_binding
        if authority.participant_ranks != binding.domain_ranks:
            raise MdpStateError("MDP: encoder execution authority matches its exact D4 domain.")
        if self._trusted_error is not None:
            raise MdpStateError("MDP: failed encoder source capture cannot be claimed.")
        if self._trusted_capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG:
            raise MdpStateError(
                "MDP: stable locator capture cannot enter execution before materialization."
            )
        is_source = binding.global_rank == binding.domain_ranks[0]
        metadata = (self._trusted_source_window, self._trusted_manifest, self._trusted_locations)
        if is_source:
            source_window, local_manifest, locations = metadata
            if (
                type(source_window) is not DecoderSourceWindow
                or type(local_manifest) is not DecoderSourceManifest
            ):
                raise MdpStateError(
                    "MDP: encoder execution source retains exact authority metadata."
                )
            expected_sample_ids = tuple(sample.sample_id for sample in source_window.samples)
            locations_by_microbatch: dict[int, list[int]] = {}
            for microbatch_id, local_sample_id in locations.values():
                locations_by_microbatch.setdefault(microbatch_id, []).append(local_sample_id)
            if (
                source_window.metadata_manifest() != local_manifest
                or authority.global_manifest.samples != local_manifest.samples
                or authority.global_manifest.items != local_manifest.items
                or authority.global_manifest.payloads != local_manifest.payloads
                or tuple(locations) != expected_sample_ids
                or len(set(locations.values())) != len(locations)
                or any(
                    tuple(sorted(local_ids)) != tuple(range(len(local_ids)))
                    for local_ids in locations_by_microbatch.values()
                )
            ):
                raise MdpStateError(
                    "MDP: encoder execution source retains exact authority metadata."
                )
        elif (
            self._trusted_source_window is not None
            or self._trusted_manifest is not None
            or self._trusted_locations
            or self._trusted_pixels
            or self._trusted_locator_catalog.entries
        ):
            raise MdpStateError("MDP: encoder execution non-source capture is exactly empty.")
        runtime = self._trusted_runtime
        transfer = (
            runtime,
            binding,
            self._trusted_source_window,
            self._trusted_manifest,
            self._trusted_locations,
            self._trusted_pixels,
        )
        runtime._retire_d4_encoder_capture_owner(self)
        self._state = _RETIRED_OWNER
        for name in (
            "_runtime",
            "_binding",
            "_source_window",
            "_local_manifest",
            "_sample_locations",
            "_pixel_sidecar",
            "_capture_mode",
            "_locator_catalog",
            "_local_prepare_error",
            "_trusted_runtime",
            "_trusted_binding",
            "_trusted_source_window",
            "_trusted_manifest",
            "_trusted_locations",
            "_trusted_pixels",
            "_trusted_pixel_view",
            "_trusted_capture_mode",
            "_trusted_locator_catalog",
            "_trusted_error",
        ):
            setattr(self, name, None)
        return transfer


def _validate_capture_context(
    runtime: Any, binding: Any, operations: Any, num_microbatches: Any
) -> tuple[Any, int, bool]:
    if type(runtime) is not MdpRuntime:
        raise MdpStateError("MDP: D4 encoder capture uses an exact MdpRuntime.")
    authority = _validate_repeated_d4_group_binding(binding)
    _validate_operations(operations)
    if type(num_microbatches) is not int or num_microbatches < 1:
        raise MdpConfigurationError("MDP: D4 encoder capture num_microbatches is positive int.")
    if (
        runtime._state is not MdpRuntimeState.EMPTY
        or runtime._window is not None
        or runtime._plan is not None
        or type(runtime.num_vpp_chunks) is not int
        or runtime.num_vpp_chunks != 1
    ):
        raise MdpStateError("MDP: D4 encoder capture starts from an idle runtime.")
    spec, view = runtime.rank_map.spec, runtime.rank_view
    if (
        type(spec.world_size) is not int
        or (spec.tp, spec.pp, spec.cp, spec.encoder_cp) != (1, 1, 4, 4)
        or spec.world_size != len(authority.world_ranks)
        or view.global_rank != authority.global_rank
        or tuple(view.planning_group_ranks) != authority.domain_ranks
        or view.endpoint_rank != authority.domain_ranks[0]
        or runtime.device != authority._device
    ):
        raise MdpStateError("MDP: D4 encoder capture matches its runtime rank and domain.")
    source_index = authority.world_ranks.index(authority.domain_ranks[0])
    if source_index % len(authority.domain_ranks):
        raise MdpStateError("MDP: D4 encoder capture domain has a WORLD-derived source lane.")
    source_lane = source_index // len(authority.domain_ranks)
    is_source = authority.global_rank == authority.domain_ranks[0]
    if view.outer_dp_rank != source_lane or view.lane_id != (source_lane if is_source else None):
        raise MdpStateError("MDP: D4 encoder capture runtime agrees with its source lane.")
    return authority, source_lane, is_source


def _normalize_local_error(error: BaseException) -> Exception:
    if isinstance(error, Exception):
        return error
    normalized = MdpStateError("MDP: D4 encoder source capture failed locally.")
    normalized.__cause__ = error
    return normalized


def _capture_d4_encoder_source(
    *,
    runtime: MdpRuntime,
    binding: _RepeatedD4GroupBinding,
    data_iterators: Any,
    num_microbatches: int,
    operations: _D4EncoderCaptureOperations,
    capture_mode: VisionCaptureMode = VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
) -> _D4EncoderCaptureOwner:
    """Capture only the validated domain source and return one owner on every rank."""
    _, source_lane, is_source = _validate_capture_context(
        runtime, binding, operations, num_microbatches
    )
    if type(capture_mode) is not VisionCaptureMode:
        raise MdpConfigurationError("MDP: D4 encoder capture mode is an exact closed enum.")
    token = object()
    _PENDING_OWNER_SEALS[token] = (id(runtime), id(binding), capture_mode)
    try:
        owner = _D4EncoderCaptureOwner(runtime, binding, capture_mode, _factory_seal=token)
    except BaseException:
        _PENDING_OWNER_SEALS.pop(token, None)
        raise
    runtime._register_d4_encoder_capture_owner(owner)
    if not is_source:
        return owner

    window = None
    window_mode = None
    pixels_transferred = False
    try:
        window = MdpIterationWindow.capture(
            data_iterators,
            num_microbatches=num_microbatches,
            adapter=_CaptureAdapter(operations),
            num_vpp_chunks=1,
            lane_id=source_lane,
            my_worker_id=0,
            num_workers=1,
            is_worker_leader=True,
            data_loader_source_worker_ids=(0,),
            capture_error_consensus=None,
        )
        window_mode = window.capture_payload_mode()
        if window_mode is not capture_mode:
            raise MdpStateError("MDP: D4 encoder capture window matches its requested mode.")
        pixels = window.payload_sidecar()
        if type(pixels) is not dict or any(
            type(item_id) is not int or not isinstance(tensor, torch.Tensor)
            for item_id, tensor in pixels.items()
        ):
            raise MdpStateError("MDP: D4 encoder capture owns an exact tensor pixel sidecar.")
        if capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG:
            if pixels:
                raise MdpStateError("MDP: D4 encoder locator capture owns no pixel sidecar.")
            locator_catalog = window.locator_catalog()
            validate_vision_locator_catalog(locator_catalog)
        else:
            locator_catalog = build_vision_locator_catalog((), ())
        owner._trusted_pixels = pixels
        pixel_view = MappingProxyType(pixels)
        owner._pixel_sidecar = owner._trusted_pixel_view = pixel_view
        pixels_transferred = True
        if capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG:
            window.release_capture_payload()
        else:
            window.release_pixels()
        source_window, sample_locations = operations.build_source_window(
            tuple(window.records()), source_dp_lane=source_lane
        )
        if type(source_window) is not DecoderSourceWindow:
            raise MdpStateError("MDP: D4 encoder capture codec returns an exact source window.")
        validate_decoder_source_window(source_window)
        if source_window.source_dp_lane != source_lane:
            raise MdpStateError("MDP: D4 encoder source window belongs to its source lane.")
        local_manifest = source_window.metadata_manifest()
        validate_decoder_source_manifest(local_manifest)
        locator_item_ids = tuple(entry.item_id for entry in locator_catalog.entries)
        manifest_item_ids = tuple(item.item_id for item in local_manifest.items)
        if (
            capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG
            and locator_item_ids != manifest_item_ids
        ) or (capture_mode is VisionCaptureMode.SOURCE_PIXEL_SIDECAR and locator_item_ids):
            raise MdpStateError(
                "MDP: D4 encoder locator catalog exactly matches its source manifest."
            )
        if not isinstance(sample_locations, Mapping):
            raise MdpStateError("MDP: D4 encoder capture sample locations are a mapping.")
        locations = dict(sample_locations)
        if any(
            type(sample_id) is not GlobalSampleId
            or sample_id.source_dp_lane != source_lane
            or type(location) is not tuple
            or len(location) != 2
            or any(type(index) is not int or index < 0 for index in location)
            for sample_id, location in locations.items()
        ):
            raise MdpStateError("MDP: D4 encoder capture sample locations match its source lane.")
        owner._install_source(
            source_window=source_window,
            local_manifest=local_manifest,
            sample_locations=locations,
            pixels=pixels,
            locator_catalog=locator_catalog,
        )
    except BaseException as captured_error:
        error = _normalize_local_error(captured_error)
        if window is not None and not pixels_transferred:
            try:
                if window_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG:
                    window.release_capture_payload()
                else:
                    window.release_pixels()
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    error, f"suppressed D4 encoder capture-window cleanup error: {cleanup_error!r}"
                )
        owner._install_error(error)
    return owner
