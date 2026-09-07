# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exact model-adapter capability for repeated-D4 MDP execution.

The registry in this module is intentionally private. Model packages register
an exact adapter class and explicitly chosen unbound operations; core then
mints one identity-bound capability for a concrete adapter instance. Claiming
the capability returns an immutable operation escrow, so repeated-D4 code does
not rediscover methods or dimensions through mutable instance attributes.
"""

import inspect
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.protocols import VisionCaptureMode

__all__ = ()

_PENDING = object()
_ACTIVE = object()
_RETIRED = object()
_REGISTRATION_SEAL = object()
_OPERATIONS_SEAL = object()
_CAPABILITY_SEAL = object()


@dataclass(frozen=True, slots=True)
class _AdapterRegistration:
    adapter_class: type
    get_batch: Callable[..., Any] = field(repr=False, compare=False)
    estimate_cost: Callable[..., Any] = field(repr=False, compare=False)
    build_dynamic_decoder_payload_codec: Callable[..., Any] = field(repr=False, compare=False)
    estimate_dynamic_encoder_workload: Callable[..., Any] = field(repr=False, compare=False)
    build_encoder: Callable[..., Any] = field(repr=False, compare=False)
    bind_dynamic_encoder_cp: Callable[..., Any] = field(repr=False, compare=False)
    encode: Callable[..., Any] = field(repr=False, compare=False)
    freeze_vision_locator: Callable[..., Any] | None = field(repr=False, compare=False)
    materialize_vision_locator: Callable[..., Any] | None = field(repr=False, compare=False)
    locator_model_arch: str | None
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self) is not _AdapterRegistration or self._seal is not _REGISTRATION_SEAL:
            raise MdpConfigurationError("MDP: core mints exact dynamic adapter registrations.")


@dataclass(slots=True)
class _CapabilityRecord:
    adapter_identity: int
    adapter_reference: weakref.ReferenceType[Any]
    registration: _AdapterRegistration
    state: object
    capability: Any = None
    operations: Any = None


@dataclass(frozen=True, slots=True)
class DynamicEncoderAdapterOperations:
    """Immutable calls and dimensions escrowed from one exact adapter instance."""

    schema_version: ClassVar[int] = 1
    payload_width: int
    embedding_width: int
    spatial_merge_size: int
    _record: _CapabilityRecord = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self) is not DynamicEncoderAdapterOperations or self._seal is not _OPERATIONS_SEAL:
            raise MdpConfigurationError("MDP: core mints dynamic adapter operation escrows.")

    def _adapter(self) -> Any:
        record = self._record
        if type(record) is _CapabilityRecord and record.state is _RETIRED:
            raise MdpStateError("MDP: dynamic adapter operation escrow is retired.")
        if (
            type(record) is not _CapabilityRecord
            or record.operations is not self
            or record.state is not _ACTIVE
            or _ACTIVE_INSTANCES.get(record.adapter_identity) is not record
        ):
            raise MdpStateError("MDP: dynamic adapter operation escrow is inactive or stale.")
        adapter = record.adapter_reference()
        if adapter is None or id(adapter) != record.adapter_identity:
            raise MdpStateError("MDP: dynamic adapter operation escrow lost its exact instance.")
        return adapter

    def get_batch(self, data_iterator: Any) -> Any:
        """Capture one microbatch through the registered unbound operation."""
        record = self._record
        return record.registration.get_batch(self._adapter(), data_iterator)

    def estimate_cost(self, item: Any) -> Any:
        """Estimate one captured item's planning cost."""
        record = self._record
        return record.registration.estimate_cost(self._adapter(), item)

    def build_dynamic_decoder_payload_codec(self) -> Any:
        """Build the registered decoder Dynamic-CP payload codec."""
        record = self._record
        return record.registration.build_dynamic_decoder_payload_codec(self._adapter())

    def estimate_dynamic_encoder_workload(self, items: Any, *, group_size: int) -> Any:
        """Estimate selected encoder work from metadata only."""
        record = self._record
        return record.registration.estimate_dynamic_encoder_workload(
            self._adapter(), items, group_size=group_size
        )

    def build_encoder(self, model_config: Any, *, pg_collection: Any) -> Any:
        """Build the registered model-owned encoder."""
        record = self._record
        return record.registration.build_encoder(
            self._adapter(), model_config, pg_collection=pg_collection
        )

    def bind_dynamic_encoder_cp(self, encoder: Any, *, membership: Any, global_rank: int) -> Any:
        """Bind the encoder to one selected encoder-CP membership."""
        record = self._record
        return record.registration.bind_dynamic_encoder_cp(
            self._adapter(), encoder, membership=membership, global_rank=global_rank
        )

    def encode(self, encoder: Any, payload: Any, layout: Any) -> Any:
        """Encode one planned payload with the registered operation."""
        record = self._record
        return record.registration.encode(self._adapter(), encoder, payload, layout)


@dataclass(frozen=True, slots=True)
class DynamicEncoderLocatorAdapterOperations(DynamicEncoderAdapterOperations):
    """Version-2 operation escrow with exact stable-locator operations."""

    schema_version: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if (
            type(self) is not DynamicEncoderLocatorAdapterOperations
            or self._seal is not _OPERATIONS_SEAL
        ):
            raise MdpConfigurationError("MDP: core mints locator adapter operation escrows.")

    @property
    def locator_model_arch(self) -> str:
        """Return the exact model identity frozen in this active escrow."""
        record = self._record
        self._adapter()
        return record.registration.locator_model_arch

    def get_batch(self, data_iterator: Any) -> Any:
        """Capture through the registered operation with this exact v2 escrow."""
        record = self._record
        return record.registration.get_batch(
            self._adapter(), data_iterator, locator_operations=self
        )

    def freeze_vision_locator(
        self,
        descriptor: Any,
        *,
        dataset_root: str,
        grid_thw: tuple[int, int, int],
        declared_dimensions: tuple[int, int] | None,
    ) -> Any:
        """Freeze one descriptor through the registered exact locator operation."""
        record = self._record
        return record.registration.freeze_vision_locator(
            self._adapter(),
            descriptor,
            dataset_root=dataset_root,
            grid_thw=grid_thw,
            declared_dimensions=declared_dimensions,
        )

    def materialize_vision_locator(self, locator: Any) -> Any:
        """Materialize one locator through the registered exact operation."""
        record = self._record
        return record.registration.materialize_vision_locator(self._adapter(), locator)


class DynamicEncoderAdapterCapability:
    """Opaque, identity-bound, one-shot authority for a dynamic adapter."""

    __slots__ = ("_record", "_seal")

    def __init__(self, record: _CapabilityRecord, *, _seal: object) -> None:
        if type(record) is not _CapabilityRecord or _seal is not _CAPABILITY_SEAL:
            raise MdpConfigurationError("MDP: core mints dynamic adapter capabilities.")
        self._record = record
        self._seal = _seal


_ADAPTER_CLASSES: dict[type, _AdapterRegistration] = {}
_PENDING_INSTANCES: dict[int, _CapabilityRecord] = {}
_ACTIVE_INSTANCES: dict[int, _CapabilityRecord] = {}
_RETIRED_INSTANCES: dict[int, weakref.ReferenceType[Any]] = {}


def _require_unbound_method(adapter_class: type, name: str, operation: Any) -> Callable[..., Any]:
    if not inspect.isfunction(operation) or not any(
        base.__dict__.get(name) is operation for base in adapter_class.__mro__
    ):
        raise MdpConfigurationError(
            f"MDP: {name} is an explicitly supplied unbound method of the registered class."
        )
    return operation


def register_dynamic_encoder_adapter_class(
    adapter_class: type,
    *,
    get_batch: Callable[..., Any],
    estimate_cost: Callable[..., Any],
    build_dynamic_decoder_payload_codec: Callable[..., Any],
    estimate_dynamic_encoder_workload: Callable[..., Any],
    build_encoder: Callable[..., Any],
    bind_dynamic_encoder_cp: Callable[..., Any],
    encode: Callable[..., Any],
    freeze_vision_locator: Callable[..., Any] | None = None,
    materialize_vision_locator: Callable[..., Any] | None = None,
    locator_model_arch: str | None = None,
) -> None:
    """Register one exact adapter class and its explicitly chosen operations.

    A class may be registered only once. Inherited methods are accepted only
    when the caller supplies the exact unbound function from that class's MRO.
    """
    if type(adapter_class) is not type:
        raise MdpConfigurationError("MDP: dynamic adapter registration uses an exact class.")
    if adapter_class in _ADAPTER_CLASSES:
        raise MdpConfigurationError("MDP: dynamic adapter class is already registered.")
    has_freeze = freeze_vision_locator is not None
    has_materialize = materialize_vision_locator is not None
    has_model_arch = locator_model_arch is not None
    if has_freeze != has_materialize or has_freeze != has_model_arch:
        raise MdpConfigurationError(
            "MDP: dynamic adapter registration has complete locator operations and "
            "locator model arch, or none of them."
        )
    if has_model_arch and (type(locator_model_arch) is not str or not locator_model_arch):
        raise MdpConfigurationError("MDP: locator model arch is an exact non-empty string.")
    get_batch = _require_unbound_method(adapter_class, "get_batch", get_batch)
    if has_freeze:
        try:
            parameters = tuple(inspect.signature(get_batch).parameters.values())
        except (TypeError, ValueError) as error:
            raise MdpConfigurationError(
                "MDP: locator adapter get_batch has an inspectable "
                "locator_operations signature."
            ) from error
        locator_parameter = parameters[2] if len(parameters) == 3 else None
        if (
            len(parameters) != 3
            or any(
                parameter.kind
                not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for parameter in parameters[:2]
            )
            or locator_parameter.name != "locator_operations"
            or locator_parameter.kind is not inspect.Parameter.KEYWORD_ONLY
            or locator_parameter.default is not None
        ):
            raise MdpConfigurationError(
                "MDP: locator adapter get_batch requires keyword-only "
                "locator_operations=None."
            )
        freeze_vision_locator = _require_unbound_method(
            adapter_class, "freeze_vision_locator", freeze_vision_locator
        )
        materialize_vision_locator = _require_unbound_method(
            adapter_class, "materialize_vision_locator", materialize_vision_locator
        )
    registration = _AdapterRegistration(
        adapter_class=adapter_class,
        get_batch=get_batch,
        estimate_cost=_require_unbound_method(adapter_class, "estimate_cost", estimate_cost),
        build_dynamic_decoder_payload_codec=_require_unbound_method(
            adapter_class,
            "build_dynamic_decoder_payload_codec",
            build_dynamic_decoder_payload_codec,
        ),
        estimate_dynamic_encoder_workload=_require_unbound_method(
            adapter_class, "estimate_dynamic_encoder_workload", estimate_dynamic_encoder_workload
        ),
        build_encoder=_require_unbound_method(adapter_class, "build_encoder", build_encoder),
        bind_dynamic_encoder_cp=_require_unbound_method(
            adapter_class, "bind_dynamic_encoder_cp", bind_dynamic_encoder_cp
        ),
        encode=_require_unbound_method(adapter_class, "encode", encode),
        freeze_vision_locator=freeze_vision_locator,
        materialize_vision_locator=materialize_vision_locator,
        locator_model_arch=locator_model_arch,
        _seal=_REGISTRATION_SEAL,
    )
    _ADAPTER_CLASSES[adapter_class] = registration


def _existing_record(adapter: Any) -> _CapabilityRecord | None:
    identity = id(adapter)
    for registry in (_PENDING_INSTANCES, _ACTIVE_INSTANCES):
        record = registry.get(identity)
        if record is not None:
            if record.adapter_reference() is adapter:
                return record
            del registry[identity]
    retired = _RETIRED_INSTANCES.get(identity)
    if retired is not None:
        if retired() is adapter:
            raise MdpStateError("MDP: dynamic adapter instance capability is retired.")
        del _RETIRED_INSTANCES[identity]
    return None


def mint_dynamic_encoder_adapter_capability(
    adapter: Any, *, capture_mode: VisionCaptureMode = VisionCaptureMode.SOURCE_PIXEL_SIDECAR
) -> DynamicEncoderAdapterCapability:
    """Mint one pending capability for an exact registered adapter instance."""
    if type(capture_mode) is not VisionCaptureMode:
        raise MdpConfigurationError("MDP: dynamic adapter capture mode is an exact closed enum.")
    registration = _ADAPTER_CLASSES.get(type(adapter))
    if registration is None:
        raise MdpConfigurationError("MDP: exact dynamic adapter class is not registered.")
    if capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG and (
        registration.freeze_vision_locator is None
        or registration.materialize_vision_locator is None
    ):
        raise MdpConfigurationError(
            "MDP: stable locator capture requires complete locator operations."
        )
    if _existing_record(adapter) is not None:
        raise MdpStateError("MDP: dynamic adapter instance already owns a capability.")

    dimensions = []
    for name in ("payload_width", "embedding_width", "spatial_merge_size"):
        value = getattr(adapter, name, None)
        if type(value) is not int or value <= 0:
            raise MdpConfigurationError(
                f"MDP: dynamic adapter {name} is a snapshotted positive integer."
            )
        dimensions.append(value)

    identity = id(adapter)
    minted: list[_CapabilityRecord] = []

    def retire_collected(reference: weakref.ReferenceType[Any]) -> None:
        record = minted[0] if minted else None
        if record is None or record.adapter_reference is not reference:
            return
        if _PENDING_INSTANCES.get(identity) is record:
            del _PENDING_INSTANCES[identity]
        if _ACTIVE_INSTANCES.get(identity) is record:
            del _ACTIVE_INSTANCES[identity]
        record.state = _RETIRED

    try:
        adapter_reference = weakref.ref(adapter, retire_collected)
    except TypeError as error:
        raise MdpConfigurationError(
            "MDP: dynamic adapter instance supports identity-safe weak references."
        ) from error
    record = _CapabilityRecord(identity, adapter_reference, registration, _PENDING)
    capability = DynamicEncoderAdapterCapability(record, _seal=_CAPABILITY_SEAL)
    operations_class = (
        DynamicEncoderLocatorAdapterOperations
        if capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG
        else DynamicEncoderAdapterOperations
    )
    operations = operations_class(
        payload_width=dimensions[0],
        embedding_width=dimensions[1],
        spatial_merge_size=dimensions[2],
        _record=record,
        _seal=_OPERATIONS_SEAL,
    )
    record.capability = capability
    record.operations = operations
    minted.append(record)
    _PENDING_INSTANCES[identity] = record
    return capability


def _capability_record(capability: Any) -> _CapabilityRecord:
    if type(capability) is not DynamicEncoderAdapterCapability:
        raise MdpConfigurationError("MDP: dynamic adapter capability has its exact core type.")
    try:
        record = capability._record
        seal = capability._seal
    except AttributeError as error:
        raise MdpConfigurationError("MDP: dynamic adapter capability is core-minted.") from error
    if (
        seal is not _CAPABILITY_SEAL
        or type(record) is not _CapabilityRecord
        or record.capability is not capability
    ):
        raise MdpConfigurationError("MDP: dynamic adapter capability is core-minted.")
    return record


def claim_dynamic_encoder_adapter_capability(
    adapter: Any, capability: Any
) -> DynamicEncoderAdapterOperations | DynamicEncoderLocatorAdapterOperations:
    """Consume one pending capability and return its immutable operation escrow."""
    record = _capability_record(capability)
    identity = id(adapter)
    if record.state is _ACTIVE:
        raise MdpStateError("MDP: dynamic adapter capability was already claimed.")
    if record.state is _RETIRED:
        raise MdpStateError("MDP: dynamic adapter capability is retired.")
    if (
        record.state is not _PENDING
        or record.adapter_identity != identity
        or record.adapter_reference() is not adapter
        or _PENDING_INSTANCES.get(identity) is not record
        or _ADAPTER_CLASSES.get(type(adapter)) is not record.registration
    ):
        raise MdpConfigurationError(
            "MDP: dynamic adapter capability belongs to its exact registered instance."
        )
    del _PENDING_INSTANCES[identity]
    record.state = _ACTIVE
    _ACTIVE_INSTANCES[identity] = record
    return record.operations


def retire_dynamic_encoder_adapter_capability(
    capability: Any, *, callback: Callable[[], Any] | None = None
) -> None:
    """Retire a pending or active capability after releasing registry ownership.

    The optional callback runs only after pending/active ownership is removed
    and the record is marked retired. A callback failure does not reactivate
    the capability.
    """
    if callback is not None and not callable(callback):
        raise MdpConfigurationError("MDP: dynamic adapter retirement callback is callable.")
    record = _capability_record(capability)
    if record.state is _RETIRED:
        raise MdpStateError("MDP: dynamic adapter capability is already retired.")
    registry = _PENDING_INSTANCES if record.state is _PENDING else _ACTIVE_INSTANCES
    if (
        record.state not in (_PENDING, _ACTIVE)
        or registry.get(record.adapter_identity) is not record
    ):
        raise MdpStateError("MDP: dynamic adapter capability has stale registry ownership.")
    del registry[record.adapter_identity]
    record.state = _RETIRED
    adapter = record.adapter_reference()
    if adapter is not None:
        _RETIRED_INSTANCES[record.adapter_identity] = record.adapter_reference
    if callback is not None:
        callback()


def _reset_dynamic_encoder_adapter_capabilities_for_tests(
    *, registrations: tuple[type, ...] = ()
) -> None:
    if type(registrations) is not tuple or any(
        type(adapter_class) is not type for adapter_class in registrations
    ):
        raise MdpConfigurationError("MDP: test reset registrations are an exact class tuple.")
    records = tuple(
        {
            id(record): record
            for record in (*_PENDING_INSTANCES.values(), *_ACTIVE_INSTANCES.values())
        }.values()
    )
    _PENDING_INSTANCES.clear()
    _ACTIVE_INSTANCES.clear()
    _RETIRED_INSTANCES.clear()
    for adapter_class in registrations:
        _ADAPTER_CLASSES.pop(adapter_class, None)
    for record in records:
        record.state = _RETIRED
