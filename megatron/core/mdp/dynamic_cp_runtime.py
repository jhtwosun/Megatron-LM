# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private gates 0--2 composition and gate-3 gradient transport for Dynamic-CP.

The caller owns buffers, encoder outputs, process groups, and resource
retirement. A trusted rank-local adapter constructs the structural VPP1
records and leaf views. This module serializes the existing all-dtype decoder
payload gate, the existing embedding bridge gate, one decoder-ready status
gate, one reverse-gradient preparation/transport gate, local caller-owned
producer aggregation, and a local one-shot receipt lifecycle. It does not
enter a decoder schedule, create replay cursors, execute backward, clear
caller-owned gradient buffers, retry, or recover from a failure inside an
entered collective.
"""

import hashlib
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import (
    DynamicBridgeKey,
    DynamicBridgeLedger,
    validate_dynamic_bridge_ledger_pair,
)
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    PreparedDynamicBridgeExchange,
    _run_dynamic_bridge_gate,
    build_dynamic_bridge_route_authority_digest,
    prepare_dynamic_bridge_exchange,
    validate_prepared_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    DecoderMicrobatchKey,
    LocalDecoderAssignment,
    _PrecollectiveStatus,
    _run_precollective_consensus,
    bind_local_decoder_assignment,
    validate_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderDynamicPlan, validate_decoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_routing import (
    DecoderPayloadRouteLedger,
    validate_decoder_payload_route_ledger,
)
from megatron.core.mdp.dynamic_cp_transport import (
    PreparedDecoderPayloadBundle,
    _run_decoder_payload_gate,
    _validate_payload_gate_context,
    validate_prepared_decoder_payload_bundle,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord

_INT64_MAX = 2**63 - 1
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_DECODER_READY_AUTHORITY_DOMAIN = b"megatron.mdp.dynamic-cp.decoder-ready"
_DECODER_READY_AUTHORITY_SCHEMA_VERSION = 1
_DECODER_GRADIENT_WAVE_AUTHORITY_DOMAIN = b"megatron.mdp.dynamic-cp.decoder-gradient-wave"
_DECODER_ROLES = ("decoder", "non-decoder")
DYNAMIC_RUNTIME_SCHEMA_VERSION = 5
DYNAMIC_EXECUTION_CONFIG_WIRE_WIDTH = 20
_DYNAMIC_EXECUTION_CONFIG_DOMAIN = b"megatron.mdp.dynamic-cp.runtime-config-v2"
_PARTITION_MODE_IDS = {"contiguous": 1, "zigzag": 2}
_EMBEDDING_DTYPE_IDS = frozenset((2, 3))


@dataclass(frozen=True)
class _LocalDecoderReadyArtifacts:
    """Rank-local records and leaves built without posting collectives."""

    records: tuple[MdpMicrobatchRecord, ...]
    embedding_leaves: Mapping[DecoderMicrobatchKey, Tensor] = field(compare=False, repr=False)


@dataclass(frozen=True)
class _DecoderReadyCarrierAuthority:
    carrier_identity: int
    role: str
    authority_digest: bytes
    global_manifest_digest: bytes
    decoder_plan_digest: bytes
    payload_bundle_authority_digest: bytes
    embedding_route_authority_digest: bytes
    global_rank: int
    participant_ranks: tuple[int, ...]
    cp_partition_mode: str
    payload_bundle_identity: int
    payload_mapping_identity: int
    embedding_exchange_identity: int
    embedding_mapping_identity: int
    assignment_identities: tuple[int, ...]
    record_identities: tuple[int, ...]
    leaf_descriptors: tuple[
        tuple[int, int, tuple[int, ...], torch.dtype, torch.device, int, int], ...
    ]


@dataclass(frozen=True)
class DecoderReadyIteration:
    """One immutable, role-aware handoff accepted by decoder-ready gate 2.

    Tensor contents are caller-owned and are not hashed. The private seal binds
    the exact transport carriers, returned mappings, record objects, and leaf
    views that were validated before gate 2.
    """

    role: str
    authority_digest: bytes
    global_manifest_digest: bytes
    decoder_plan_digest: bytes
    payload_bundle_authority_digest: bytes
    embedding_route_authority_digest: bytes
    global_rank: int
    participant_ranks: tuple[int, ...]
    cp_partition_mode: str
    assignments: tuple[LocalDecoderAssignment, ...] = field(compare=False, repr=False)
    records: tuple[MdpMicrobatchRecord, ...] = field(compare=False, repr=False)
    embedding_leaves: Mapping[DecoderMicrobatchKey, Tensor] = field(compare=False, repr=False)
    _authority: _DecoderReadyCarrierAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


@dataclass(frozen=True)
class _PreparedDecoderGradientAuthority:
    carrier_identity: int
    ready_identity: int
    ready_authority_digest: bytes
    exchange_identity: int
    source_mapping_identity: int
    source_descriptors: tuple[
        tuple[DynamicBridgeKey, int, tuple[int, ...], torch.dtype, torch.device, int, int], ...
    ]


@dataclass(frozen=True)
class PreparedDecoderGradientExchange:
    """One sealed, non-destructive local preparation for the future gate 3."""

    ready: DecoderReadyIteration = field(compare=False, repr=False)
    source_tensors: Mapping[DynamicBridgeKey, Tensor] = field(compare=False, repr=False)
    exchange: PreparedDynamicBridgeExchange = field(compare=False, repr=False)
    _authority: _PreparedDecoderGradientAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


@dataclass(frozen=True)
class _DecoderGradientReceiptAuthority:
    receipt_identity: int
    prepared_identity: int
    iteration_nonce: bytes
    consumed_lifecycle_identity: int | None
    received_mapping_identity: int
    received_descriptors: tuple[
        tuple[DynamicBridgeKey, int, tuple[int, ...], torch.dtype, torch.device, int, int], ...
    ]


@dataclass(frozen=True)
class DecoderGradientReceipt:
    """One sealed gate-3 result, ready for caller-owned producer aggregation."""

    prepared: PreparedDecoderGradientExchange = field(compare=False, repr=False)
    iteration_nonce: bytes
    received_tensors: Mapping[DynamicBridgeKey, Tensor] = field(compare=False, repr=False)
    _consumed_lifecycle_identity: int | None = field(default=None, init=False, repr=False)
    _authority: _DecoderGradientReceiptAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


@dataclass(frozen=True)
class _DecoderGradientReceiptLifecycleAuthority:
    lifecycle_identity: int
    iteration_nonce: bytes
    state: str
    receipt_identity: int | None


@dataclass(frozen=True)
class DecoderGradientReceiptLifecycle:
    """Caller-owned one-shot state for one decoder-gradient receipt."""

    iteration_nonce: bytes
    _receipt_identity: int | None = field(default=None, init=False, repr=False)
    _state: str = field(default="new", init=False, repr=False)
    _authority: _DecoderGradientReceiptLifecycleAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


def _require_integer(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > _INT64_MAX:
        raise MdpConfigurationError(f"MDP: {name} is a non-negative signed-int64 integer.")
    return value


def _require_positive_integer(name: str, value: Any) -> int:
    value = _require_integer(name, value)
    if value == 0:
        raise MdpConfigurationError(f"MDP: {name} is a positive signed-int64 integer.")
    return value


def _require_exact_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise MdpConfigurationError(f"MDP: {name} is an exact bool.")
    return value


def _require_ranks(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise MdpConfigurationError(f"MDP: {name} is a non-empty immutable rank tuple.")
    ranks = tuple(_require_integer(f"{name}[{index}]", rank) for index, rank in enumerate(value))
    if len(set(ranks)) != len(ranks):
        raise MdpConfigurationError(f"MDP: {name} contains unique ranks in authoritative order.")
    return ranks


def _digest_words(payload: bytes) -> tuple[int, int]:
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return struct.unpack("<qq", digest)


@dataclass(frozen=True)
class _DynamicExecutionConfigAuthority:
    """Private seal for one immutable Dynamic-CP topology contract."""

    config_identity: int
    wire: tuple[int, ...]
    digest: bytes


@dataclass(frozen=True)
class _DynamicExecutionConfig:
    """Fixed-wire, digest-consensus topology contract for the staged runtime.

    It deliberately accepts only the legacy singleton encoder domain or a
    contiguous, repeated four-rank joint D4 domain. Later controller slices
    exchange its digest through the existing status wire before any payload
    collective.
    """

    schema_version: int
    forward_only: bool
    partition_mode: str
    embedding_width: int
    embedding_dtype_id: int
    participant_ranks: tuple[int, ...]
    tensor_parallel_size: int
    expert_parallel_size: int
    pipeline_parallel_size: int
    configured_context_parallel_size: int
    encoder_context_parallel_size: int
    virtual_pipeline_parallel_size: int
    expert_group_ranks: tuple[int, ...] | None
    sequence_parallel: bool
    dynamic_encoder_context_parallel: bool
    overlap_window_capture: bool
    digest: bytes = field(init=False, repr=False)
    _wire: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _authority: _DynamicExecutionConfigAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if _require_integer("dynamic runtime schema_version", self.schema_version) != (
            DYNAMIC_RUNTIME_SCHEMA_VERSION
        ):
            raise MdpConfigurationError(
                "MDP: dynamic runtime uses the current configuration schema version."
            )
        forward_only = _require_exact_bool("dynamic runtime forward_only", self.forward_only)
        if forward_only:
            raise MdpConfigurationError("MDP: staged Dynamic-CP runtime is training-only.")
        if self.partition_mode not in _PARTITION_MODE_IDS:
            raise MdpConfigurationError(
                "MDP: dynamic runtime partition_mode is contiguous or zigzag."
            )
        width = _require_positive_integer("dynamic runtime embedding_width", self.embedding_width)
        dtype_id = _require_positive_integer(
            "dynamic runtime embedding_dtype_id", self.embedding_dtype_id
        )
        if dtype_id not in _EMBEDDING_DTYPE_IDS:
            raise MdpConfigurationError(
                "MDP: dynamic runtime embedding dtype id represents BF16 or FP16."
            )
        participants = _require_ranks("dynamic runtime participant ranks", self.participant_ranks)
        topology = tuple(
            _require_positive_integer(f"dynamic runtime {name}", value)
            for name, value in (
                ("tensor_parallel_size", self.tensor_parallel_size),
                ("expert_parallel_size", self.expert_parallel_size),
                ("pipeline_parallel_size", self.pipeline_parallel_size),
                ("configured_context_parallel_size", self.configured_context_parallel_size),
                ("encoder_context_parallel_size", self.encoder_context_parallel_size),
                ("virtual_pipeline_parallel_size", self.virtual_pipeline_parallel_size),
            )
        )
        sequence_parallel = _require_exact_bool(
            "dynamic runtime sequence_parallel", self.sequence_parallel
        )
        dynamic_encoder_cp = _require_exact_bool(
            "dynamic runtime dynamic_encoder_context_parallel",
            self.dynamic_encoder_context_parallel,
        )
        overlap_capture = _require_exact_bool(
            "dynamic runtime overlap_window_capture", self.overlap_window_capture
        )
        tp, ep, pp, cp, encoder_cp, vpp = topology
        legacy_d3 = (tp, pp, cp, encoder_cp, vpp) == (1, 1, 1, 1, 1) and not dynamic_encoder_cp
        repeated_d4_domain = (
            len(participants) == 4
            and participants[0] % 4 == 0
            and participants == tuple(range(participants[0], participants[0] + 4))
        )
        joint_d4 = (
            repeated_d4_domain
            and (tp, pp, cp, encoder_cp, vpp) == (1, 1, 4, 4, 1)
            and ep in (1, 4)
            and dynamic_encoder_cp
        )
        expert_group = self.expert_group_ranks
        if ep == 1:
            if expert_group is not None:
                raise MdpConfigurationError(
                    "MDP: Dynamic-CP EP1 has no multi-rank expert group authority."
                )
            expert_group_size = 0
            expert_group_words = (0, 0)
        else:
            expert_group = _require_ranks("dynamic runtime expert group ranks", expert_group)
            if len(expert_group) != ep:
                raise MdpConfigurationError(
                    "MDP: Dynamic-CP expert group width matches expert parallel size."
                )
            if expert_group != participants:
                raise MdpConfigurationError(
                    "MDP: Dynamic-CP EP4 expert group exactly matches its local D4 domain."
                )
            expert_group_size = len(expert_group)
            expert_group_words = _digest_words(
                struct.pack(f"<{expert_group_size}q", *expert_group)
            )
        if legacy_d3 and self.partition_mode != "contiguous":
            raise MdpConfigurationError(
                "MDP: legacy D3 runtime partition_mode is the locked contiguous layout."
            )
        if sequence_parallel or overlap_capture or not (legacy_d3 or joint_d4):
            raise MdpConfigurationError(
                "MDP: dynamic runtime accepts the legacy D3 topology or the exact joint D4 "
                "size-four CP4/ECP4 domain with EP1 or domain-local EP4 and sequence "
                "parallel and overlap off."
            )
        participant_words = _digest_words(struct.pack(f"<{len(participants)}q", *participants))
        wire = (
            DYNAMIC_RUNTIME_SCHEMA_VERSION,
            int(forward_only),
            _PARTITION_MODE_IDS[self.partition_mode],
            width,
            dtype_id,
            *topology,
            int(sequence_parallel),
            int(dynamic_encoder_cp),
            int(overlap_capture),
            len(participants),
            *participant_words,
            expert_group_size,
            *expert_group_words,
        )
        if len(wire) != DYNAMIC_EXECUTION_CONFIG_WIRE_WIDTH:
            raise AssertionError("dynamic runtime configuration wire width drifted")
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(_DYNAMIC_EXECUTION_CONFIG_DOMAIN)
        hasher.update(struct.pack(f"<{len(wire)}q", *wire))
        object.__setattr__(self, "_wire", wire)
        object.__setattr__(self, "digest", hasher.digest())
        object.__setattr__(
            self,
            "_authority",
            _DynamicExecutionConfigAuthority(
                config_identity=id(self), wire=wire, digest=hasher.digest()
            ),
        )

    def to_wire_tuple(self) -> tuple[int, ...]:
        """Return the fixed-width signed-int64 consensus wire."""
        return self._wire


def _validate_dynamic_execution_config(config: Any) -> _DynamicExecutionConfig:
    """Rebuild and compare one config so post-construction mutation cannot lie."""
    if type(config) is not _DynamicExecutionConfig:
        raise MdpConfigurationError("MDP: runtime config consensus uses its typed carrier.")
    authority = config._authority
    if type(authority) is not _DynamicExecutionConfigAuthority or authority.config_identity != id(config):
        raise MdpStateError("MDP: dynamic runtime config retains its private seal.")
    canonical = _DynamicExecutionConfig(
        schema_version=config.schema_version,
        forward_only=config.forward_only,
        partition_mode=config.partition_mode,
        embedding_width=config.embedding_width,
        embedding_dtype_id=config.embedding_dtype_id,
        participant_ranks=config.participant_ranks,
        tensor_parallel_size=config.tensor_parallel_size,
        expert_parallel_size=config.expert_parallel_size,
        pipeline_parallel_size=config.pipeline_parallel_size,
        configured_context_parallel_size=config.configured_context_parallel_size,
        encoder_context_parallel_size=config.encoder_context_parallel_size,
        virtual_pipeline_parallel_size=config.virtual_pipeline_parallel_size,
        expert_group_ranks=config.expert_group_ranks,
        sequence_parallel=config.sequence_parallel,
        dynamic_encoder_context_parallel=config.dynamic_encoder_context_parallel,
        overlap_window_capture=config.overlap_window_capture,
    )
    if (
        authority.wire != canonical._wire
        or authority.digest != canonical.digest
        or config._wire != canonical._wire
        or config.digest != canonical.digest
    ):
        raise MdpStateError("MDP: dynamic runtime config matches its private seal.")
    return config


def _consensus_dynamic_execution_config(
    *,
    config: _DynamicExecutionConfig,
    group_ranks: tuple[int, ...],
    global_rank: int,
    all_gather_status: Any,
    timeout_seconds: float,
) -> None:
    """Require every runtime participant to advertise the same configuration digest.

    ``group_ranks`` and ``all_gather_status`` are rank-symmetric rendezvous
    context, just as for the lower bridge gates. A local typed-carrier error
    becomes a status error; an asymmetric process-group binding cannot safely
    be discovered because it has no common collective to enter.
    """
    rank = _require_integer("dynamic runtime config global rank", global_rank)
    ranks = _require_ranks("dynamic runtime config group ranks", group_ranks)
    if rank not in ranks:
        raise MdpConfigurationError("MDP: runtime config global rank belongs to its group.")
    if not callable(all_gather_status):
        raise MdpConfigurationError("MDP: runtime config status gather is callable.")
    local_error = None
    digest = bytes(16)
    try:
        config = _validate_dynamic_execution_config(config)
        if ranks != config.participant_ranks:
            raise MdpConfigurationError(
                "MDP: runtime config participants match the exact status group order."
            )
        digest = config.digest
    except Exception as error:
        local_error = error
    status = _PrecollectiveStatus(
        global_rank=rank,
        global_manifest_digest=digest,
        plan_digest=digest,
        error_code=int(local_error is not None),
        gate_id=0,
    )
    try:
        _run_precollective_consensus(
            status,
            group_ranks=ranks,
            all_gather_status=all_gather_status,
            timeout_seconds=timeout_seconds,
        )
    except (MdpBridgeError, MdpPlanError) as error:
        if local_error is not None and error.__cause__ is None:
            raise error from local_error
        raise
    if local_error is not None:
        raise MdpStateError(
            "MDP: runtime config consensus succeeded despite a local error."
        ) from local_error


@dataclass(frozen=True)
class _PreAuthorityDynamicProducer:
    """One local P0--P2 result captured before metadata authority exists."""

    rank_view: Any
    local_manifest: Any
    source_window: Any
    static_plan: Any
    item_outputs: Mapping
    owner: Any
    local_prepare_error: Exception | None
    forward_only: bool

    def __post_init__(self) -> None:
        if _require_exact_bool("pre-authority forward_only", self.forward_only):
            raise MdpConfigurationError(
                "MDP: dynamic producer contracts support training iterations only."
            )
        if not isinstance(self.item_outputs, _MAPPING_PROXY_TYPE):
            raise MdpConfigurationError(
                "MDP: pre-authority producer item_outputs is an immutable local mapping."
            )
        error = self.local_prepare_error
        if error is not None and not isinstance(error, Exception):
            raise MdpConfigurationError(
                "MDP: pre-authority local_prepare_error is an Exception or None."
            )
        local_values = (self.local_manifest, self.source_window, self.static_plan)
        if error is None:
            present = tuple(value is not None for value in local_values)
            contributor = all(present)
            noncontributor = not any(present) and not self.item_outputs
            if self.owner is None or not (contributor or noncontributor):
                raise MdpConfigurationError(
                    "MDP: successful pre-authority producer owns exact contributor P0-P2 "
                    "state or empty noncontributor state."
                )
        elif (
            self.owner is not None
            or any(value is not None for value in local_values)
            or self.item_outputs
        ):
            raise MdpConfigurationError(
                "MDP: failed pre-authority producer carries only its local error and "
                "empty immutable item_outputs."
            )


@dataclass(frozen=True)
class _DynamicIterationAuthority:
    """Global immutable D3 authority consumed by later runtime composition."""

    global_manifest: Any
    plan: Any
    source_rank_by_lane: Mapping
    producer_rank_by_item: Mapping
    output_rows_by_item: Mapping
    payload_ledger: Any
    embedding_ledger: Any
    gradient_ledger: Any
    participant_ranks: tuple[int, ...]
    bridge_width: int
    bridge_dtype: torch.dtype

    def __post_init__(self) -> None:
        for name in ("source_rank_by_lane", "producer_rank_by_item", "output_rows_by_item"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise MdpConfigurationError(
                    f"MDP: dynamic iteration authority {name} is a mapping."
                )
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        typed_fields = (
            ("global_manifest", self.global_manifest, DecoderGlobalManifest),
            ("plan", self.plan, DecoderDynamicPlan),
            ("payload_ledger", self.payload_ledger, DecoderPayloadRouteLedger),
            ("embedding_ledger", self.embedding_ledger, DynamicBridgeLedger),
            ("gradient_ledger", self.gradient_ledger, DynamicBridgeLedger),
        )
        for name, value, expected_type in typed_fields:
            if type(value) is not expected_type:
                raise MdpConfigurationError(
                    f"MDP: dynamic iteration authority {name} has its exact typed carrier."
                )
        if self.embedding_ledger.phase is not BridgePhase.EMBEDDING:
            raise MdpConfigurationError(
                "MDP: dynamic iteration authority embedding ledger has embedding phase."
            )
        if self.gradient_ledger.phase is not BridgePhase.GRADIENT:
            raise MdpConfigurationError(
                "MDP: dynamic iteration authority gradient ledger has gradient phase."
            )
        participants = _require_ranks(
            "dynamic iteration authority participant ranks", self.participant_ranks
        )
        width = _require_positive_integer(
            "dynamic iteration authority bridge width", self.bridge_width
        )
        if not isinstance(self.bridge_dtype, torch.dtype):
            raise MdpConfigurationError(
                "MDP: dynamic iteration authority bridge dtype is a torch dtype."
            )
        object.__setattr__(self, "participant_ranks", participants)
        object.__setattr__(self, "bridge_width", width)
        validate_decoder_global_manifest(self.global_manifest)
        validate_decoder_dynamic_plan(self.plan)
        validate_decoder_payload_route_ledger(
            self.payload_ledger,
            plan=self.plan,
            global_manifest=self.global_manifest,
            source_rank_by_lane=self.source_rank_by_lane,
            participant_ranks=participants,
        )
        validate_dynamic_bridge_ledger_pair(
            self.embedding_ledger,
            self.gradient_ledger,
            plan=self.plan,
            global_manifest=self.global_manifest,
            producer_rank_by_item=self.producer_rank_by_item,
            output_rows_by_item=self.output_rows_by_item,
            width=width,
            dtype=self.bridge_dtype,
            participant_ranks=participants,
        )


@dataclass(frozen=True)
class _DynamicProducerCarrier:
    """Caller-owned producer state bound to one typed global D3 authority.

    This private handoff neither enters a collective nor invokes the callbacks;
    later runtime composition owns both operations.
    """

    authority: _DynamicIterationAuthority
    pre_authority: _PreAuthorityDynamicProducer
    owner: Any
    rank_view: Any
    local_manifest: Any
    source_window: Any
    static_plan: Any
    item_outputs: Mapping
    payload_destination_views: Mapping
    embedding_destination_views: Mapping
    gradient_destination_views: Mapping
    summed_gradient_destination_views: Mapping
    backward: Callable
    cleanup: Callable

    def __post_init__(self) -> None:
        if type(self.authority) is not _DynamicIterationAuthority:
            raise MdpConfigurationError(
                "MDP: dynamic producer carries its exact global iteration authority."
            )
        if type(self.pre_authority) is not _PreAuthorityDynamicProducer:
            raise MdpConfigurationError(
                "MDP: dynamic producer carries its exact pre-authority identity."
            )
        if self.owner is None or self.owner is not self.pre_authority.owner:
            raise MdpConfigurationError(
                "MDP: dynamic producer owner matches its pre-authority identity."
            )
        for name in ("rank_view", "local_manifest", "source_window", "static_plan", "item_outputs"):
            if getattr(self, name) is not getattr(self.pre_authority, name):
                raise MdpConfigurationError(
                    f"MDP: dynamic producer {name} preserves its pre-authority identity."
                )
        for name in (
            "item_outputs",
            "payload_destination_views",
            "embedding_destination_views",
            "gradient_destination_views",
            "summed_gradient_destination_views",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise MdpConfigurationError(
                    f"MDP: dynamic producer {name} is a caller-owned mapping."
                )
        if not callable(self.backward) or not callable(self.cleanup):
            raise MdpConfigurationError(
                "MDP: dynamic producer backward and cleanup callbacks are callable."
            )


def _require_digest(name: str, value: Any) -> bytes:
    if type(value) is not bytes or len(value) != 16:
        raise MdpPlanError(f"MDP: {name} is an exact 16-byte digest.")
    return value


def _require_iteration_nonce(value: Any) -> bytes:
    nonce = _require_digest("decoder gradient iteration nonce", value)
    if nonce == bytes(16):
        raise MdpConfigurationError("MDP: decoder gradient iteration nonce is non-zero.")
    return nonce


def _capture_decoder_gradient_receipt_lifecycle_authority(
    lifecycle: DecoderGradientReceiptLifecycle,
) -> _DecoderGradientReceiptLifecycleAuthority:
    nonce = _require_iteration_nonce(lifecycle.iteration_nonce)
    if lifecycle._state == "new":
        if lifecycle._receipt_identity is not None:
            raise MdpStateError("MDP: new decoder gradient lifecycle has no receipt.")
    elif lifecycle._state == "consumed":
        if not isinstance(lifecycle._receipt_identity, int):
            raise MdpStateError("MDP: consumed decoder gradient lifecycle owns one receipt.")
    elif lifecycle._state == "retired":
        if lifecycle._receipt_identity is not None:
            raise MdpStateError("MDP: retired decoder gradient lifecycle has no receipt.")
    else:
        raise MdpStateError("MDP: decoder gradient lifecycle has a valid state.")
    return _DecoderGradientReceiptLifecycleAuthority(
        lifecycle_identity=id(lifecycle),
        iteration_nonce=nonce,
        state=lifecycle._state,
        receipt_identity=lifecycle._receipt_identity,
    )


def _validate_decoder_gradient_receipt_lifecycle(
    lifecycle: Any, *, expected_state: str
) -> DecoderGradientReceiptLifecycle:
    if type(lifecycle) is not DecoderGradientReceiptLifecycle:
        raise MdpConfigurationError("MDP: decoder gradient lifecycle has its exact owner type.")
    if type(lifecycle._authority) is not _DecoderGradientReceiptLifecycleAuthority:
        raise MdpBridgeError("MDP: decoder gradient lifecycle has a private authority seal.")
    if _capture_decoder_gradient_receipt_lifecycle_authority(lifecycle) != lifecycle._authority:
        raise MdpBridgeError("MDP: decoder gradient lifecycle matches its private authority seal.")
    if lifecycle._state != expected_state:
        raise MdpStateError(f"MDP: decoder gradient lifecycle requires a {expected_state} state.")
    return lifecycle


def _seal_decoder_gradient_receipt_lifecycle(
    lifecycle: DecoderGradientReceiptLifecycle, *, state: str, receipt_identity: int | None
) -> None:
    object.__setattr__(lifecycle, "_state", state)
    object.__setattr__(lifecycle, "_receipt_identity", receipt_identity)
    object.__setattr__(
        lifecycle, "_authority", _capture_decoder_gradient_receipt_lifecycle_authority(lifecycle)
    )


def _digest_integers(hasher: Any, *values: int) -> None:
    checked = tuple(_require_integer("decoder-ready digest field", value) for value in values)
    hasher.update(struct.pack(f"<{len(checked)}q", *checked))


def _digest_bytes(hasher: Any, value: bytes) -> None:
    checked = _require_digest("decoder-ready digest input", value)
    _digest_integers(hasher, len(checked))
    hasher.update(checked)


def _decoder_gradient_wave_authority_digest(ready: Any, iteration_nonce: Any) -> bytes:
    """Return gate-safe wave authority, or neutral authority for a local fault."""
    try:
        if (
            type(ready) is not DecoderReadyIteration
            or type(ready._authority) is not _DecoderReadyCarrierAuthority
        ):
            return bytes(16)
        nonce = _require_iteration_nonce(iteration_nonce)
        ready_digest = _require_digest(
            "decoder gradient ready authority", ready._authority.authority_digest
        )
    except Exception:
        return bytes(16)
    hasher = hashlib.blake2s(digest_size=16)
    hasher.update(_DECODER_GRADIENT_WAVE_AUTHORITY_DOMAIN)
    _digest_bytes(hasher, ready_digest)
    _digest_bytes(hasher, nonce)
    return hasher.digest()


def _digest_text(hasher: Any, value: str) -> None:
    if not isinstance(value, str):
        raise MdpConfigurationError("MDP: decoder-ready digest text is a string.")
    encoded = value.encode("utf-8")
    _digest_integers(hasher, len(encoded))
    hasher.update(encoded)


def _decoder_ready_authority_digest(
    *,
    global_manifest_digest: bytes,
    decoder_plan_digest: bytes,
    payload_bundle_authority_digest: bytes,
    embedding_route_authority_digest: bytes,
    participant_ranks: tuple[int, ...],
    cp_partition_mode: str,
) -> bytes:
    participants = _require_ranks("decoder-ready participant ranks", participant_ranks)
    if cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError("MDP: decoder-ready CP partition mode is contiguous or zigzag.")
    hasher = hashlib.blake2b(digest_size=16)
    _digest_integers(
        hasher, len(_DECODER_READY_AUTHORITY_DOMAIN), _DECODER_READY_AUTHORITY_SCHEMA_VERSION
    )
    hasher.update(_DECODER_READY_AUTHORITY_DOMAIN)
    _digest_bytes(hasher, global_manifest_digest)
    _digest_bytes(hasher, decoder_plan_digest)
    _digest_bytes(hasher, payload_bundle_authority_digest)
    _digest_bytes(hasher, embedding_route_authority_digest)
    _digest_integers(hasher, len(participants), *participants)
    _digest_text(hasher, cp_partition_mode)
    return hasher.digest()


def _storage_pointer(tensor: Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _capture_carrier_authority(
    ready: DecoderReadyIteration,
    *,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_tensors: Mapping[Any, Tensor],
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_tensors: Mapping[Any, Tensor],
) -> _DecoderReadyCarrierAuthority:
    return _DecoderReadyCarrierAuthority(
        carrier_identity=id(ready),
        role=ready.role,
        authority_digest=ready.authority_digest,
        global_manifest_digest=ready.global_manifest_digest,
        decoder_plan_digest=ready.decoder_plan_digest,
        payload_bundle_authority_digest=ready.payload_bundle_authority_digest,
        embedding_route_authority_digest=ready.embedding_route_authority_digest,
        global_rank=ready.global_rank,
        participant_ranks=ready.participant_ranks,
        cp_partition_mode=ready.cp_partition_mode,
        payload_bundle_identity=id(payload_bundle),
        payload_mapping_identity=id(payload_tensors),
        embedding_exchange_identity=id(embedding_exchange),
        embedding_mapping_identity=id(embedding_tensors),
        assignment_identities=tuple(id(value) for value in ready.assignments),
        record_identities=tuple(id(value) for value in ready.records),
        leaf_descriptors=tuple(
            (
                id(key),
                id(tensor),
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
                _storage_pointer(tensor),
                tensor.storage_offset(),
            )
            for key, tensor in ready.embedding_leaves.items()
        ),
    )


def _expected_role(*, plan: DecoderDynamicPlan, global_rank: int) -> str:
    return "decoder" if global_rank in plan.decoder_ranks else "non-decoder"


def _expected_local_assignments(
    plan: DecoderDynamicPlan,
    *,
    global_rank: int,
    decoder_group_getter: Any,
    decoder_group_ranks_getter: Any,
) -> tuple[LocalDecoderAssignment, ...]:
    if global_rank not in plan.decoder_ranks:
        return ()
    if not callable(decoder_group_getter) or not callable(decoder_group_ranks_getter):
        raise MdpConfigurationError(
            "MDP: decoder-ready native Dynamic-CP group getters are callable."
        )
    return tuple(
        bind_local_decoder_assignment(
            plan,
            key=DecoderMicrobatchKey(microbatch.microbatch_index),
            global_rank=global_rank,
            maximum_group_ranks=plan.decoder_ranks,
            group_getter=decoder_group_getter,
            group_ranks_getter=decoder_group_ranks_getter,
        )
        for microbatch in plan.microbatches
    )


def _expected_vision_records(
    manifest: DecoderGlobalManifest, assignment: LocalDecoderAssignment
) -> tuple[tuple[Any, ...], ...]:
    samples = {sample.sample_id: sample for sample in manifest.samples}
    items = {item.item_id: item for item in manifest.items}
    expected = []
    padded_start = 0
    for local_sample_id, sample_id in enumerate(assignment.assignment.sample_ids):
        sample = samples[sample_id]
        for encoder_item in sample.vision_items:
            item = items[encoder_item.item_id]
            expected.append(
                (
                    item.item_id,
                    local_sample_id,
                    item.image_ordinal,
                    item.grid_thw,
                    item.output_rows,
                    tuple(padded_start + offset for offset in item.decoder_offsets),
                )
            )
        padded_start += sample.padded_seqlen
    return tuple(expected)


def _validate_records_and_leaves(
    *,
    records: Any,
    leaves: Any,
    expected_assignments: tuple[LocalDecoderAssignment, ...],
    global_manifest: DecoderGlobalManifest,
    embedding_width: int,
    embedding_dtype: torch.dtype,
    embedding_device: torch.device,
    cp_partition_mode: str,
    forbidden_buffers: tuple[Tensor, ...],
) -> None:
    if not isinstance(records, tuple) or len(records) != len(expected_assignments):
        raise MdpConfigurationError(
            "MDP: decoder-ready records exactly cover local plan microbatches."
        )
    if type(leaves) is not _MAPPING_PROXY_TYPE:
        raise MdpConfigurationError(
            "MDP: decoder-ready embedding leaves form an immutable mapping."
        )
    expected_leaf_keys = []
    rows_by_key = {}
    for assignment, record in zip(expected_assignments, records):
        if type(record) is not MdpMicrobatchRecord:
            raise MdpConfigurationError(
                "MDP: decoder-ready records are exact MdpMicrobatchRecord carriers."
            )
        if type(record.microbatch_id) is not int or record.microbatch_id != (
            assignment.key.microbatch_index
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready record microbatch identity matches its assignment."
            )
        expected_vision = _expected_vision_records(global_manifest, assignment)
        if not isinstance(record.vision_items, tuple) or any(
            type(item) is not MdpMicrobatchVisionRecord for item in record.vision_items
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready vision records form an immutable typed tuple."
            )
        actual_vision = tuple(
            (
                item.global_item_id,
                item.sample_id,
                item.image_ordinal,
                item.grid_thw,
                item.output_rows,
                item.decoder_positions,
            )
            for item in record.vision_items
        )
        if (
            actual_vision != expected_vision
            or type(record.text_only) is not bool
            or record.text_only != (not expected_vision)
        ):
            raise MdpPlanError(
                "MDP: decoder-ready record vision metadata matches manifest authority."
            )
        if type(record.model_payload) is not _MAPPING_PROXY_TYPE:
            raise MdpConfigurationError("MDP: decoder-ready model payload is an immutable mapping.")
        packed = record.decoder_packed_seq_params
        try:
            packed_values = (
                packed.qkv_format,
                packed.total_tokens,
                packed.local_cp_size,
                packed.cp_group,
                packed.cp_partition_mode,
            )
        except Exception as error:
            raise MdpConfigurationError(
                "MDP: decoder-ready record exposes complete packed-THD metadata."
            ) from error
        expected_tokens = sum(
            next(
                sample.padded_seqlen
                for sample in global_manifest.samples
                if sample.sample_id == sample_id
            )
            for sample_id in assignment.assignment.sample_ids
        )
        if (
            packed_values[0] != "thd"
            or type(packed_values[1]) is not int
            or packed_values[1] != expected_tokens
            or type(packed_values[2]) is not int
            or packed_values[2] != assignment.assignment.local_cp_size
            or packed_values[3] is not assignment.cp_group
            or packed_values[4] != cp_partition_mode
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready packed-THD metadata matches its local assignment."
            )
        if expected_vision:
            expected_leaf_keys.append(assignment.key)
            rows_by_key[assignment.key] = sum(value[4] for value in expected_vision)

    actual_leaf_keys = tuple(leaves)
    if len(actual_leaf_keys) != len(expected_leaf_keys) or any(
        actual is not expected for actual, expected in zip(actual_leaf_keys, expected_leaf_keys)
    ):
        raise MdpConfigurationError(
            "MDP: decoder-ready leaves use the exact vision assignment keys in order."
        )
    forbidden_pointers = {
        _storage_pointer(buffer) for buffer in forbidden_buffers if buffer.numel()
    }
    leaf_pointers = []
    for key in expected_leaf_keys:
        leaf = leaves[key]
        if (
            not isinstance(leaf, Tensor)
            or tuple(leaf.shape) != (rows_by_key[key], embedding_width)
            or leaf.dtype != embedding_dtype
            or leaf.device != embedding_device
            or not leaf.is_leaf
            or not leaf.requires_grad
            or leaf.grad_fn is not None
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready leaf has exact geometry and detached-leaf semantics."
            )
        pointer = _storage_pointer(leaf)
        if pointer in forbidden_pointers:
            raise MdpConfigurationError("MDP: decoder-ready leaves do not alias transport buffers.")
        leaf_pointers.append(pointer)
    if len(set(leaf_pointers)) != len(leaf_pointers):
        raise MdpConfigurationError("MDP: decoder-ready leaves use pairwise disjoint storage.")


def validate_decoder_ready_iteration(
    ready: Any,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_tensors: Mapping[Any, Tensor],
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_tensors: Mapping[Any, Tensor],
    expected_assignments: tuple[LocalDecoderAssignment, ...],
    authority_digest: bytes,
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
) -> DecoderReadyIteration:
    """Validate one sealed handoff against exact phase-local authority."""
    if type(ready) is not DecoderReadyIteration:
        raise MdpConfigurationError("MDP: decoder-ready handoff has its exact frozen carrier type.")
    if type(ready._authority) is not _DecoderReadyCarrierAuthority:
        raise MdpBridgeError("MDP: decoder-ready handoff has a private authority seal.")
    validate_decoder_global_manifest(global_manifest)
    validate_decoder_dynamic_plan(plan)
    bundle = validate_prepared_decoder_payload_bundle(payload_bundle)
    exchange = validate_prepared_dynamic_bridge_exchange(embedding_exchange)
    participants = _require_ranks("decoder-ready participant ranks", ready.participant_ranks)
    rank = _require_integer("decoder-ready global rank", ready.global_rank)
    if rank not in participants:
        raise MdpConfigurationError("MDP: decoder-ready global rank belongs to participant ranks.")
    expected_role = _expected_role(plan=plan, global_rank=rank)
    if ready.role not in _DECODER_ROLES or ready.role != expected_role:
        raise MdpConfigurationError("MDP: decoder-ready carrier role matches rank authority.")
    expected_digest = _require_digest("decoder-ready authority digest", authority_digest)
    scalar_fields = (
        (ready.authority_digest, expected_digest),
        (ready.global_manifest_digest, global_manifest.digest),
        (ready.decoder_plan_digest, plan.digest),
        (ready.payload_bundle_authority_digest, bundle.bundle_authority_digest),
        (ready.embedding_route_authority_digest, exchange.route_authority_digest),
        (ready.cp_partition_mode, cp_partition_mode),
    )
    if any(actual != expected for actual, expected in scalar_fields):
        raise MdpBridgeError(
            "MDP: decoder-ready carrier scalar authority matches the active phase."
        )
    if (
        payload_tensors is not bundle.received_tensors
        or embedding_tensors is not exchange.received_tensors
    ):
        raise MdpBridgeError(
            "MDP: decoder-ready carrier retains exact transport carriers and results."
        )
    if bundle.global_rank != rank or exchange.global_rank != rank:
        raise MdpBridgeError("MDP: decoder-ready transport ranks match the local carrier rank.")
    if bundle.participant_ranks != participants or exchange.participant_ranks != participants:
        raise MdpBridgeError("MDP: decoder-ready transports share exact participant order.")
    if (
        not isinstance(expected_assignments, tuple)
        or len(ready.assignments) != len(expected_assignments)
        or any(
            actual is not expected
            for actual, expected in zip(ready.assignments, expected_assignments)
        )
    ):
        raise MdpConfigurationError(
            "MDP: decoder-ready carrier retains exact local assignment identities."
        )
    if expected_role != "decoder" and (
        ready.assignments or ready.records or ready.embedding_leaves
    ):
        raise MdpConfigurationError("MDP: non-decoder decoder-ready carriers are exactly empty.")
    width = _require_integer("decoder-ready embedding width", embedding_width)
    if width == 0 or not isinstance(embedding_dtype, torch.dtype):
        raise MdpConfigurationError(
            "MDP: decoder-ready embedding geometry has positive width and torch dtype."
        )
    forbidden_buffers = tuple(
        buffer for child in bundle.exchanges for buffer in (child.send_buffer, child.receive_buffer)
    ) + (exchange.send_buffer, exchange.receive_buffer)
    _validate_records_and_leaves(
        records=ready.records,
        leaves=ready.embedding_leaves,
        expected_assignments=expected_assignments,
        global_manifest=global_manifest,
        embedding_width=width,
        embedding_dtype=embedding_dtype,
        embedding_device=exchange.receive_buffer.device,
        cp_partition_mode=cp_partition_mode,
        forbidden_buffers=forbidden_buffers,
    )
    if (
        _capture_carrier_authority(
            ready,
            payload_bundle=bundle,
            payload_tensors=payload_tensors,
            embedding_exchange=exchange,
            embedding_tensors=embedding_tensors,
        )
        != ready._authority
    ):
        raise MdpBridgeError(
            "MDP: decoder-ready public geometry matches its private authority seal."
        )
    return ready


def _validate_retained_decoder_ready_iteration(
    ready: Any,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
) -> DecoderReadyIteration:
    """Validate a gate-2 carrier after its gate-0 and gate-1 inputs retired."""
    if type(ready) is not DecoderReadyIteration:
        raise MdpConfigurationError("MDP: retained decoder-ready handoff has its exact type.")
    if type(ready._authority) is not _DecoderReadyCarrierAuthority:
        raise MdpBridgeError("MDP: retained decoder-ready handoff has a private authority seal.")
    validate_decoder_global_manifest(global_manifest)
    validate_decoder_dynamic_plan(plan)
    rank = _require_integer("retained decoder-ready global rank", global_rank)
    participants = _require_ranks("retained decoder-ready participant ranks", participant_ranks)
    ready_participants = _require_ranks(
        "retained decoder-ready carrier participant ranks", ready.participant_ranks
    )
    ready_authority_digest = _require_digest(
        "retained decoder-ready authority digest", ready.authority_digest
    )
    ready_manifest_digest = _require_digest(
        "retained decoder-ready manifest digest", ready.global_manifest_digest
    )
    ready_plan_digest = _require_digest(
        "retained decoder-ready plan digest", ready.decoder_plan_digest
    )
    ready_payload_digest = _require_digest(
        "retained decoder-ready payload digest", ready.payload_bundle_authority_digest
    )
    ready_embedding_digest = _require_digest(
        "retained decoder-ready embedding digest", ready.embedding_route_authority_digest
    )
    if type(ready.cp_partition_mode) is not str:
        raise MdpConfigurationError("MDP: retained decoder-ready CP partition mode has exact type.")
    if rank not in participants:
        raise MdpConfigurationError(
            "MDP: retained decoder-ready global rank belongs to its participants."
        )
    if type(ready.global_rank) is not int or type(ready.role) is not str:
        raise MdpConfigurationError("MDP: retained decoder-ready scalar types are exact.")
    if not isinstance(ready.assignments, tuple):
        raise MdpConfigurationError(
            "MDP: retained decoder-ready assignments form an immutable tuple."
        )
    if (
        ready.global_rank != rank
        or ready_participants != participants
        or ready_manifest_digest != global_manifest.digest
        or ready_plan_digest != plan.digest
        or ready.cp_partition_mode != cp_partition_mode
    ):
        raise MdpBridgeError("MDP: retained decoder-ready scalar authority matches this phase.")
    expected_role = _expected_role(plan=plan, global_rank=rank)
    if ready.role != expected_role:
        raise MdpConfigurationError("MDP: retained decoder-ready role matches plan authority.")
    expected_digest = _decoder_ready_authority_digest(
        global_manifest_digest=global_manifest.digest,
        decoder_plan_digest=plan.digest,
        payload_bundle_authority_digest=ready_payload_digest,
        embedding_route_authority_digest=ready_embedding_digest,
        participant_ranks=participants,
        cp_partition_mode=cp_partition_mode,
    )
    if ready_authority_digest != expected_digest:
        raise MdpBridgeError("MDP: retained decoder-ready authority digest matches its fields.")
    authority = ready._authority
    if (
        authority.carrier_identity != id(ready)
        or authority.role != ready.role
        or authority.authority_digest != ready.authority_digest
        or authority.global_manifest_digest != ready.global_manifest_digest
        or authority.decoder_plan_digest != ready.decoder_plan_digest
        or authority.payload_bundle_authority_digest != ready.payload_bundle_authority_digest
        or authority.embedding_route_authority_digest != ready.embedding_route_authority_digest
        or authority.global_rank != ready.global_rank
        or authority.participant_ranks != ready.participant_ranks
        or authority.cp_partition_mode != ready.cp_partition_mode
        or authority.assignment_identities != tuple(id(value) for value in ready.assignments)
        or authority.record_identities != tuple(id(value) for value in ready.records)
    ):
        raise MdpBridgeError("MDP: retained decoder-ready identities match its private seal.")
    if expected_role != "decoder" and (
        ready.assignments or ready.records or ready.embedding_leaves
    ):
        raise MdpConfigurationError("MDP: retained non-decoder handoff is exactly empty.")
    width = _require_integer("retained decoder-ready embedding width", embedding_width)
    if width == 0 or not isinstance(embedding_dtype, torch.dtype):
        raise MdpConfigurationError(
            "MDP: retained decoder-ready embedding geometry has positive width and torch dtype."
        )
    device = next(
        (value.device for value in ready.embedding_leaves.values() if isinstance(value, Tensor)),
        torch.device("cpu"),
    )
    _validate_records_and_leaves(
        records=ready.records,
        leaves=ready.embedding_leaves,
        expected_assignments=ready.assignments,
        global_manifest=global_manifest,
        embedding_width=width,
        embedding_dtype=embedding_dtype,
        embedding_device=device,
        cp_partition_mode=cp_partition_mode,
        forbidden_buffers=(),
    )
    descriptors = tuple(
        (
            id(key),
            id(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            _storage_pointer(tensor),
            tensor.storage_offset(),
        )
        for key, tensor in ready.embedding_leaves.items()
    )
    if descriptors != authority.leaf_descriptors:
        raise MdpBridgeError("MDP: retained decoder-ready leaves match their private seal.")
    return ready


def _gradient_source_descriptors(
    sources: Mapping[DynamicBridgeKey, Tensor],
) -> tuple[tuple[DynamicBridgeKey, int, tuple[int, ...], torch.dtype, torch.device, int, int], ...]:
    return tuple(
        (
            key,
            id(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            _storage_pointer(tensor),
            tensor.storage_offset(),
        )
        for key, tensor in sources.items()
    )


def _decoder_gradient_sources(
    ready: DecoderReadyIteration,
    *,
    gradient_ledger: DynamicBridgeLedger,
    embedding_dtype: torch.dtype,
) -> Mapping[DynamicBridgeKey, Tensor]:
    """Project detached decoder leaf gradients to canonical reverse-route keys."""
    if (
        type(gradient_ledger) is not DynamicBridgeLedger
        or gradient_ledger.phase is not BridgePhase.GRADIENT
    ):
        raise MdpPlanError("MDP: decoder gradient preparation has an exact gradient ledger.")
    expected_entries = tuple(
        entry for entry in gradient_ledger.entries if entry.src_global_rank == ready.global_rank
    )
    sources: dict[DynamicBridgeKey, Tensor] = {}
    leaf_pointers = {_storage_pointer(leaf) for leaf in ready.embedding_leaves.values()}
    gradient_pointers = set()
    for assignment, record in zip(ready.assignments, ready.records):
        if not record.vision_items:
            continue
        leaf = ready.embedding_leaves[assignment.key]
        gradient = leaf.grad
        if (
            not isinstance(gradient, Tensor)
            or tuple(gradient.shape) != tuple(leaf.shape)
            or gradient.dtype != embedding_dtype
            or gradient.device != leaf.device
            or gradient.requires_grad
            or gradient.grad_fn is not None
        ):
            raise MdpStateError(
                "MDP: decoder gradient preparation requires one detached exact leaf gradient."
            )
        pointer = _storage_pointer(gradient)
        if pointer in leaf_pointers or pointer in gradient_pointers:
            raise MdpBridgeError(
                "MDP: decoder leaf gradients do not alias decoder leaves or each other."
            )
        gradient_pointers.add(pointer)
        cursor = 0
        for item in record.vision_items:
            rows = item.output_rows
            key = DynamicBridgeKey(item.global_item_id, ready.global_rank)
            if key in sources or cursor + rows > gradient.shape[0]:
                raise MdpPlanError("MDP: decoder leaf gradients cover canonical item rows once.")
            sources[key] = gradient.narrow(0, cursor, rows)
            cursor += rows
        if cursor != gradient.shape[0]:
            raise MdpPlanError("MDP: decoder leaf gradient rows exactly cover its vision records.")
    expected_keys = tuple(entry.key for entry in expected_entries)
    if len(sources) != len(expected_keys) or set(sources) != set(expected_keys):
        raise MdpPlanError("MDP: decoder gradient sources exactly cover the reverse route.")
    return MappingProxyType({key: sources[key] for key in expected_keys})


def _capture_prepared_decoder_gradient_authority(
    prepared: PreparedDecoderGradientExchange,
) -> _PreparedDecoderGradientAuthority:
    return _PreparedDecoderGradientAuthority(
        carrier_identity=id(prepared),
        ready_identity=id(prepared.ready),
        ready_authority_digest=prepared.ready.authority_digest,
        exchange_identity=id(prepared.exchange),
        source_mapping_identity=id(prepared.source_tensors),
        source_descriptors=_gradient_source_descriptors(prepared.source_tensors),
    )


def validate_prepared_decoder_gradient_exchange(
    prepared: Any,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
) -> PreparedDecoderGradientExchange:
    """Validate one sealed local gradient preparation without entering gate 3."""
    if type(prepared) is not PreparedDecoderGradientExchange:
        raise MdpConfigurationError("MDP: decoder gradient preparation has its exact carrier type.")
    if type(prepared._authority) is not _PreparedDecoderGradientAuthority:
        raise MdpBridgeError("MDP: decoder gradient preparation has a private authority seal.")
    ready = _validate_retained_decoder_ready_iteration(
        prepared.ready,
        global_manifest=global_manifest,
        plan=plan,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
    )
    exchange = validate_prepared_dynamic_bridge_exchange(prepared.exchange)
    if (
        exchange.phase is not BridgePhase.GRADIENT
        or exchange.global_rank != ready.global_rank
        or exchange.participant_ranks != ready.participant_ranks
    ):
        raise MdpBridgeError("MDP: decoder gradient preparation matches retained rank authority.")
    if _capture_prepared_decoder_gradient_authority(prepared) != prepared._authority:
        raise MdpBridgeError(
            "MDP: decoder gradient preparation matches its private authority seal."
        )
    return prepared


def _capture_decoder_gradient_receipt_authority(
    receipt: DecoderGradientReceipt,
) -> _DecoderGradientReceiptAuthority:
    consumed_lifecycle_identity = receipt._consumed_lifecycle_identity
    if not (
        consumed_lifecycle_identity is None
        or (
            isinstance(consumed_lifecycle_identity, int)
            and not isinstance(consumed_lifecycle_identity, bool)
            and consumed_lifecycle_identity >= 0
        )
    ):
        raise MdpStateError("MDP: decoder gradient receipt has a valid consumption owner.")
    return _DecoderGradientReceiptAuthority(
        receipt_identity=id(receipt),
        prepared_identity=id(receipt.prepared),
        iteration_nonce=_require_iteration_nonce(receipt.iteration_nonce),
        consumed_lifecycle_identity=consumed_lifecycle_identity,
        received_mapping_identity=id(receipt.received_tensors),
        received_descriptors=_gradient_source_descriptors(receipt.received_tensors),
    )


def _validate_decoder_gradient_receipt(
    receipt: Any,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    global_rank: int,
    participant_ranks: tuple[int, ...],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
    iteration_nonce: bytes,
) -> DecoderGradientReceipt:
    """Validate one sealed gate-3 result before local producer aggregation."""
    if type(receipt) is not DecoderGradientReceipt:
        raise MdpConfigurationError("MDP: decoder gradient receipt has its exact carrier type.")
    if type(receipt._authority) is not _DecoderGradientReceiptAuthority:
        raise MdpBridgeError("MDP: decoder gradient receipt has a private authority seal.")
    expected_nonce = _require_iteration_nonce(iteration_nonce)
    prepared = validate_prepared_decoder_gradient_exchange(
        receipt.prepared,
        global_manifest=global_manifest,
        plan=plan,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
    )
    if receipt.received_tensors is not prepared.exchange.received_tensors:
        raise MdpBridgeError("MDP: decoder gradient receipt retains the exact gate-3 mapping.")
    if _capture_decoder_gradient_receipt_authority(receipt) != receipt._authority:
        raise MdpBridgeError("MDP: decoder gradient receipt matches its private authority seal.")
    if receipt.iteration_nonce != expected_nonce:
        raise MdpStateError("MDP: decoder gradient receipt belongs to the active iteration nonce.")
    if receipt._consumed_lifecycle_identity is not None:
        raise MdpStateError("MDP: decoder gradient receipt is consumed exactly once.")
    route_authority_digest = build_dynamic_bridge_route_authority_digest(
        gradient_ledger,
        embedding_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=embedding_width,
        dtype=embedding_dtype,
        participant_ranks=participant_ranks,
    )
    if route_authority_digest != prepared.exchange.route_authority_digest:
        raise MdpBridgeError(
            "MDP: decoder gradient receipt route matches retained transport authority."
        )
    expected_keys = tuple(
        entry.key for entry in gradient_ledger.entries if entry.dst_global_rank == global_rank
    )
    if tuple(receipt.received_tensors) != expected_keys:
        raise MdpPlanError("MDP: decoder gradient receipt covers the exact local producer routes.")
    return receipt


def _make_decoder_gradient_receipt(
    prepared: PreparedDecoderGradientExchange,
    received_tensors: Mapping[DynamicBridgeKey, Tensor],
    *,
    iteration_nonce: bytes,
) -> DecoderGradientReceipt:
    """Seal the exact immutable mapping returned by a successful gate-3 exchange."""
    if received_tensors is not prepared.exchange.received_tensors:
        raise MdpBridgeError("MDP: decoder gradient receipt receives the exact gate-3 mapping.")
    receipt = DecoderGradientReceipt(
        prepared=prepared,
        iteration_nonce=_require_iteration_nonce(iteration_nonce),
        received_tensors=received_tensors,
    )
    object.__setattr__(receipt, "_authority", _capture_decoder_gradient_receipt_authority(receipt))
    return receipt


def _seal_decoder_gradient_receipt_consumption(
    receipt: DecoderGradientReceipt, lifecycle: DecoderGradientReceiptLifecycle
) -> None:
    object.__setattr__(receipt, "_consumed_lifecycle_identity", id(lifecycle))
    object.__setattr__(receipt, "_authority", _capture_decoder_gradient_receipt_authority(receipt))


def _assemble_decoder_gradient_receipt(
    lifecycle: DecoderGradientReceiptLifecycle,
    receipt: DecoderGradientReceipt,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    global_rank: int,
    participant_ranks: tuple[int, ...],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
    destination_tensors: Mapping[GlobalVisionItemId, Tensor],
) -> Mapping[GlobalVisionItemId, Tensor]:
    """Sum endpoint gradients into exact caller-owned producer item buffers.

    This is local, non-destructive receipt consumption only.  It neither runs
    encoder backward nor retires the receipt, decoder leaves, or caller buffers.
    """
    lifecycle = _validate_decoder_gradient_receipt_lifecycle(lifecycle, expected_state="new")
    receipt = _validate_decoder_gradient_receipt(
        receipt,
        global_manifest=global_manifest,
        plan=plan,
        embedding_ledger=embedding_ledger,
        gradient_ledger=gradient_ledger,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
        iteration_nonce=lifecycle.iteration_nonce,
    )
    if not isinstance(destination_tensors, Mapping):
        raise MdpConfigurationError("MDP: decoder gradient destinations are an item mapping.")
    local_items = tuple(
        item.item_id
        for item in global_manifest.items
        if any(
            entry.dst_global_rank == global_rank and entry.key.item_id == item.item_id
            for entry in gradient_ledger.entries
        )
    )
    if set(destination_tensors) != set(local_items):
        raise MdpPlanError("MDP: decoder gradient destinations cover exact local producer items.")
    sources = tuple(receipt.received_tensors.values()) + tuple(
        receipt.prepared.source_tensors.values()
    )
    source_pointers = {_storage_pointer(source) for source in sources if source.numel()}
    source_pointers.update(
        _storage_pointer(buffer)
        for buffer in (
            receipt.prepared.exchange.send_buffer,
            receipt.prepared.exchange.receive_buffer,
        )
        if buffer.numel()
    )
    source_pointers.update(
        _storage_pointer(gradient)
        for leaf in receipt.prepared.ready.embedding_leaves.values()
        if isinstance((gradient := leaf.grad), Tensor) and gradient.numel()
    )
    source_pointers.update(
        _storage_pointer(leaf)
        for leaf in receipt.prepared.ready.embedding_leaves.values()
        if leaf.numel()
    )
    receipt_device = receipt.prepared.exchange.receive_buffer.device
    destinations = {}
    destination_pointers = set()
    for item_id in local_items:
        destination = destination_tensors[item_id]
        rows = next(item.output_rows for item in global_manifest.items if item.item_id == item_id)
        if (
            not isinstance(destination, Tensor)
            or tuple(destination.shape) != (rows, embedding_width)
            or destination.dtype != embedding_dtype
            or destination.device != receipt_device
            or destination.requires_grad
            or destination.grad_fn is not None
            or not destination.is_contiguous()
        ):
            raise MdpConfigurationError(
                "MDP: decoder gradient destination is one detached contiguous item tensor."
            )
        pointer = _storage_pointer(destination)
        if pointer in source_pointers or pointer in destination_pointers:
            raise MdpBridgeError(
                "MDP: decoder gradient destinations do not alias receipt or peers."
            )
        destination_pointers.add(pointer)
        destinations[item_id] = destination
    for destination in destinations.values():
        destination.zero_()
    for entry in gradient_ledger.entries:
        if entry.dst_global_rank == global_rank:
            destinations[entry.key.item_id].add_(receipt.received_tensors[entry.key])
    _seal_decoder_gradient_receipt_consumption(receipt, lifecycle)
    _seal_decoder_gradient_receipt_lifecycle(
        lifecycle, state="consumed", receipt_identity=id(receipt)
    )
    return MappingProxyType({item_id: destinations[item_id] for item_id in local_items})


def _begin_decoder_gradient_receipt_lifecycle(
    iteration_nonce: bytes,
) -> DecoderGradientReceiptLifecycle:
    """Create one local, one-shot owner for a caller-selected backward wave."""
    lifecycle = DecoderGradientReceiptLifecycle(
        iteration_nonce=_require_iteration_nonce(iteration_nonce)
    )
    _seal_decoder_gradient_receipt_lifecycle(lifecycle, state="new", receipt_identity=None)
    return lifecycle


def _consume_decoder_gradient_receipt(
    lifecycle: Any, receipt: DecoderGradientReceipt, **kwargs: Any
) -> Mapping[GlobalVisionItemId, Tensor]:
    """Aggregate one current receipt once, then make the lifecycle consumed."""
    lifecycle = _validate_decoder_gradient_receipt_lifecycle(lifecycle, expected_state="new")
    nonce = lifecycle.iteration_nonce
    if type(receipt) is not DecoderGradientReceipt or receipt.iteration_nonce != nonce:
        raise MdpStateError("MDP: decoder gradient receipt belongs to the active lifecycle nonce.")
    return _assemble_decoder_gradient_receipt(lifecycle, receipt, **kwargs)


def _retire_decoder_gradient_receipt_lifecycle(lifecycle: Any) -> None:
    """Close a consumed local lifecycle without clearing caller-owned buffers."""
    lifecycle = _validate_decoder_gradient_receipt_lifecycle(lifecycle, expected_state="consumed")
    _seal_decoder_gradient_receipt_lifecycle(lifecycle, state="retired", receipt_identity=None)


def _complete_decoder_gradient_phase(
    ready: DecoderReadyIteration,
    *,
    iteration_nonce: bytes,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    group_ranks: tuple[int, ...],
    send_buffer: Tensor,
    receive_buffer: Tensor,
    destination_tensors: Mapping[GlobalVisionItemId, Tensor],
    all_gather_status: Any,
    timeout_seconds: float,
    group: Any,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> Mapping[GlobalVisionItemId, Tensor]:
    """Complete one successful reverse-gradient wave from ready leaves to producers.

    The caller invokes this only after the native decoder schedule populated the
    retained endpoint-leaf gradients. It owns every buffer and the surrounding
    schedule state. This helper seals one local receipt lifecycle; it does not
    make an iteration nonce globally fresh or recover from a failed collective.
    """

    def prepare_gradient() -> PreparedDecoderGradientExchange:
        participants = _require_ranks("decoder gradient participant ranks", participant_ranks)
        if participants != _require_ranks("decoder gradient group ranks", group_ranks):
            raise MdpConfigurationError(
                "MDP: decoder gradient participants and collective group ranks match exactly."
            )
        return _prepare_decoder_gradient_exchange(
            ready,
            global_manifest=global_manifest,
            plan=plan,
            embedding_ledger=embedding_ledger,
            gradient_ledger=gradient_ledger,
            producer_rank_by_item=producer_rank_by_item,
            output_rows_by_item=output_rows_by_item,
            embedding_width=embedding_width,
            embedding_dtype=embedding_dtype,
            cp_partition_mode=cp_partition_mode,
            global_rank=global_rank,
            participant_ranks=participants,
            send_buffer=send_buffer,
            receive_buffer=receive_buffer,
        )

    receipt = _run_decoder_gradient_phase(
        None,
        iteration_nonce=iteration_nonce,
        local_prepare=prepare_gradient,
        predecessor_authority_digest=_decoder_gradient_wave_authority_digest(
            ready, iteration_nonce
        ),
        global_manifest=global_manifest,
        plan=plan,
        embedding_ledger=embedding_ledger,
        gradient_ledger=gradient_ledger,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
        global_rank=global_rank,
        group_ranks=group_ranks,
        all_gather_status=all_gather_status,
        timeout_seconds=timeout_seconds,
        group=group,
        group_ranks_getter=group_ranks_getter,
        all_to_all_single=all_to_all_single,
    )
    lifecycle = _begin_decoder_gradient_receipt_lifecycle(iteration_nonce)
    destinations = _consume_decoder_gradient_receipt(
        lifecycle,
        receipt,
        global_manifest=global_manifest,
        plan=plan,
        embedding_ledger=embedding_ledger,
        gradient_ledger=gradient_ledger,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
        destination_tensors=destination_tensors,
    )
    _retire_decoder_gradient_receipt_lifecycle(lifecycle)
    return destinations


def _prepare_decoder_gradient_exchange(
    ready: DecoderReadyIteration,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    send_buffer: Tensor,
    receive_buffer: Tensor,
) -> PreparedDecoderGradientExchange:
    """Freeze one caller-buffer reverse-gradient exchange without collective work."""
    ready = _validate_retained_decoder_ready_iteration(
        ready,
        global_manifest=global_manifest,
        plan=plan,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
    )
    route_authority_digest = build_dynamic_bridge_route_authority_digest(
        embedding_ledger,
        gradient_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=embedding_width,
        dtype=embedding_dtype,
        participant_ranks=participant_ranks,
    )
    if route_authority_digest != ready.embedding_route_authority_digest:
        raise MdpBridgeError(
            "MDP: decoder gradient preparation route authority matches retained ready authority."
        )
    sources = _decoder_gradient_sources(
        ready, gradient_ledger=gradient_ledger, embedding_dtype=embedding_dtype
    )
    exchange = prepare_dynamic_bridge_exchange(
        gradient_ledger,
        embedding_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=embedding_width,
        dtype=embedding_dtype,
        participant_ranks=participant_ranks,
        global_rank=global_rank,
        local_tensors=sources,
        send_buffer=send_buffer,
        receive_buffer=receive_buffer,
    )
    prepared = PreparedDecoderGradientExchange(
        ready=ready, source_tensors=sources, exchange=exchange
    )
    object.__setattr__(
        prepared, "_authority", _capture_prepared_decoder_gradient_authority(prepared)
    )
    return validate_prepared_decoder_gradient_exchange(
        prepared,
        global_manifest=global_manifest,
        plan=plan,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
    )


def _run_decoder_gradient_gate(
    prepared: Any,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
    global_rank: int,
    group_ranks: tuple[int, ...],
    all_gather_status: Any,
    timeout_seconds: float,
    group: Any,
    local_prepare: Callable[[], PreparedDecoderGradientExchange] | None = None,
    predecessor_authority_digest: bytes | None = None,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> Mapping[DynamicBridgeKey, Tensor]:
    """Run gate 3 for one sealed decoder-gradient preparation.

    Invalid local carriers intentionally enter the status gate with a neutral
    predecessor. The local preparation callback then records the validation
    error, so all ranks converge before the helper can issue gradient A2A.
    """
    if predecessor_authority_digest is None:
        predecessor_authority_digest = bytes(16)
        if (
            type(prepared) is PreparedDecoderGradientExchange
            and type(prepared._authority) is _PreparedDecoderGradientAuthority
        ):
            predecessor_authority_digest = prepared._authority.ready_authority_digest

    def prepare_exchange() -> PreparedDynamicBridgeExchange:
        if local_prepare is None:
            carrier = validate_prepared_decoder_gradient_exchange(
                prepared,
                global_manifest=global_manifest,
                plan=plan,
                global_rank=global_rank,
                participant_ranks=group_ranks,
                embedding_width=embedding_width,
                embedding_dtype=embedding_dtype,
                cp_partition_mode=cp_partition_mode,
            )
        else:
            carrier = local_prepare()
            carrier = validate_prepared_decoder_gradient_exchange(
                carrier,
                global_manifest=global_manifest,
                plan=plan,
                global_rank=global_rank,
                participant_ranks=group_ranks,
                embedding_width=embedding_width,
                embedding_dtype=embedding_dtype,
                cp_partition_mode=cp_partition_mode,
            )
        return carrier.exchange

    return _run_dynamic_bridge_gate(
        phase=BridgePhase.GRADIENT,
        ledger=gradient_ledger,
        reverse_ledger=embedding_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=embedding_width,
        dtype=embedding_dtype,
        global_rank=global_rank,
        group_ranks=group_ranks,
        all_gather_status=all_gather_status,
        local_prepare=prepare_exchange,
        timeout_seconds=timeout_seconds,
        group=group,
        predecessor_authority_digest=predecessor_authority_digest,
        group_ranks_getter=group_ranks_getter,
        all_to_all_single=all_to_all_single,
    )


def _run_decoder_gradient_phase(
    prepared: Any,
    *,
    iteration_nonce: bytes,
    local_prepare: Callable[[], PreparedDecoderGradientExchange] | None = None,
    predecessor_authority_digest: bytes | None = None,
    **kwargs: Any,
) -> DecoderGradientReceipt:
    """Run nonce-bound gate 3 and seal its exact received mapping for aggregation."""

    prepared_slot: dict[str, PreparedDecoderGradientExchange] = {}

    def prepare_phase() -> PreparedDecoderGradientExchange:
        _require_iteration_nonce(iteration_nonce)
        carrier = prepared if local_prepare is None else local_prepare()
        if type(carrier) is PreparedDecoderGradientExchange:
            prepared_slot["value"] = carrier
        return carrier

    if predecessor_authority_digest is None:
        ready = prepared.ready if type(prepared) is PreparedDecoderGradientExchange else None
        predecessor_authority_digest = _decoder_gradient_wave_authority_digest(
            ready, iteration_nonce
        )
    received_tensors = _run_decoder_gradient_gate(
        prepared,
        local_prepare=prepare_phase,
        predecessor_authority_digest=predecessor_authority_digest,
        **kwargs,
    )
    receipt_prepared = prepared_slot.get("value")
    if receipt_prepared is None:
        raise MdpBridgeError("MDP: decoder gradient gate retains its successful prepared exchange.")
    return _make_decoder_gradient_receipt(
        receipt_prepared, received_tensors, iteration_nonce=iteration_nonce
    )


def _build_decoder_ready_iteration(
    *,
    role: str,
    authority_digest: bytes,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    cp_partition_mode: str,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_tensors: Mapping[Any, Tensor],
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_tensors: Mapping[Any, Tensor],
    assignments: tuple[LocalDecoderAssignment, ...],
    artifacts: _LocalDecoderReadyArtifacts,
) -> DecoderReadyIteration:
    ready = DecoderReadyIteration(
        role=role,
        authority_digest=authority_digest,
        global_manifest_digest=global_manifest.digest,
        decoder_plan_digest=plan.digest,
        payload_bundle_authority_digest=payload_bundle.bundle_authority_digest,
        embedding_route_authority_digest=embedding_exchange.route_authority_digest,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        cp_partition_mode=cp_partition_mode,
        assignments=assignments,
        records=artifacts.records,
        embedding_leaves=artifacts.embedding_leaves,
    )
    object.__setattr__(
        ready,
        "_authority",
        _capture_carrier_authority(
            ready,
            payload_bundle=payload_bundle,
            payload_tensors=payload_tensors,
            embedding_exchange=embedding_exchange,
            embedding_tensors=embedding_tensors,
        ),
    )
    return ready


def _run_decoder_ready_phase(
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    payload_ledger: DecoderPayloadRouteLedger,
    source_rank_by_lane: Mapping[int, int],
    payload_local_prepare: Any,
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    embedding_local_prepare: Any,
    local_prepare: Any,
    cp_partition_mode: str,
    decoder_group_getter: Any,
    decoder_group_ranks_getter: Any,
    global_rank: int,
    group_ranks: tuple[int, ...],
    all_gather_status: Any,
    timeout_seconds: float,
    group: Any,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> DecoderReadyIteration:
    """Run real payload and embedding gates, then publish one gate-2 handoff.

    Rendezvous inputs must be rank-symmetric. The payload and embedding
    preparation callbacks run on every participant; the trusted structural
    adapter callback runs only on decoder ranks. Callbacks must remain
    rank-local, must not post collectives, and must not mutate returned
    transport mappings. Failures inside an entered payload or embedding
    collective are task-fatal and are not advanced to a later gate.
    """
    rank, ranks, gather, timeout = _validate_payload_gate_context(
        global_rank=global_rank,
        group_ranks=group_ranks,
        all_gather_status=all_gather_status,
        timeout_seconds=timeout_seconds,
    )

    payload_slot = {}

    def prepare_payload():
        bundle = payload_local_prepare()
        payload_slot["value"] = bundle
        return bundle

    payload_tensors = _run_decoder_payload_gate(
        global_manifest=global_manifest,
        plan=plan,
        ledger=payload_ledger,
        source_rank_by_lane=source_rank_by_lane,
        global_rank=rank,
        group_ranks=ranks,
        all_gather_status=gather,
        local_prepare=prepare_payload,
        timeout_seconds=timeout,
        group=group,
        group_ranks_getter=group_ranks_getter,
        all_to_all_single=all_to_all_single,
    )

    embedding_slot = {}

    def prepare_embedding():
        exchange = embedding_local_prepare()
        embedding_slot["value"] = exchange
        return exchange

    embedding_tensors = _run_dynamic_bridge_gate(
        phase=BridgePhase.EMBEDDING,
        ledger=embedding_ledger,
        reverse_ledger=gradient_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=embedding_width,
        dtype=embedding_dtype,
        global_rank=rank,
        group_ranks=ranks,
        all_gather_status=gather,
        local_prepare=prepare_embedding,
        timeout_seconds=timeout,
        group=group,
        group_ranks_getter=group_ranks_getter,
        all_to_all_single=all_to_all_single,
    )

    manifest_digest = bytes(16)
    ready_digest = bytes(16)
    ready = None
    local_error = None
    try:
        validate_decoder_global_manifest(global_manifest)
        validate_decoder_dynamic_plan(plan)
        if global_manifest.samples != plan.samples:
            raise MdpPlanError("MDP: decoder-ready manifest and plan sample catalogs agree.")
        participants = _require_ranks("decoder-ready participant ranks", ranks)
        payload_bundle = validate_prepared_decoder_payload_bundle(payload_slot["value"])
        embedding_exchange = validate_prepared_dynamic_bridge_exchange(embedding_slot["value"])
        payload_digest = payload_bundle.bundle_authority_digest
        embedding_digest = embedding_exchange.route_authority_digest
        ready_digest = _decoder_ready_authority_digest(
            global_manifest_digest=global_manifest.digest,
            decoder_plan_digest=plan.digest,
            payload_bundle_authority_digest=payload_digest,
            embedding_route_authority_digest=embedding_digest,
            participant_ranks=participants,
            cp_partition_mode=cp_partition_mode,
        )
        manifest_digest = global_manifest.digest
        if payload_tensors is not payload_bundle.received_tensors:
            raise MdpBridgeError("MDP: decoder-ready payload result is the exact gate-0 mapping.")
        if embedding_tensors is not embedding_exchange.received_tensors:
            raise MdpBridgeError("MDP: decoder-ready embedding result is the exact gate-1 mapping.")
        assignments = _expected_local_assignments(
            plan,
            global_rank=rank,
            decoder_group_getter=decoder_group_getter,
            decoder_group_ranks_getter=decoder_group_ranks_getter,
        )
        if rank in plan.decoder_ranks:
            if not callable(local_prepare):
                raise MdpConfigurationError(
                    "MDP: decoder-ready local_prepare is callable on decoder ranks."
                )
            artifacts = local_prepare(payload_tensors, embedding_tensors, assignments)
        else:
            artifacts = _LocalDecoderReadyArtifacts((), MappingProxyType({}))
        if type(artifacts) is not _LocalDecoderReadyArtifacts:
            raise MdpConfigurationError(
                "MDP: decoder-ready local_prepare returns typed local artifacts."
            )
        ready = _build_decoder_ready_iteration(
            role=_expected_role(plan=plan, global_rank=rank),
            authority_digest=ready_digest,
            global_manifest=global_manifest,
            plan=plan,
            global_rank=rank,
            participant_ranks=participants,
            cp_partition_mode=cp_partition_mode,
            payload_bundle=payload_bundle,
            payload_tensors=payload_tensors,
            embedding_exchange=embedding_exchange,
            embedding_tensors=embedding_tensors,
            assignments=assignments,
            artifacts=artifacts,
        )
        validate_decoder_ready_iteration(
            ready,
            global_manifest=global_manifest,
            plan=plan,
            payload_bundle=payload_bundle,
            payload_tensors=payload_tensors,
            embedding_exchange=embedding_exchange,
            embedding_tensors=embedding_tensors,
            expected_assignments=assignments,
            authority_digest=ready_digest,
            embedding_width=embedding_width,
            embedding_dtype=embedding_dtype,
            cp_partition_mode=cp_partition_mode,
        )
    except Exception as error:
        local_error = error

    status = _PrecollectiveStatus(
        global_rank=rank,
        global_manifest_digest=manifest_digest,
        plan_digest=ready_digest,
        error_code=int(local_error is not None),
        gate_id=2,
    )
    try:
        _run_precollective_consensus(
            status, group_ranks=ranks, all_gather_status=gather, timeout_seconds=timeout
        )
    except (MdpBridgeError, MdpPlanError) as error:
        if local_error is not None and error.__cause__ is None:
            raise error from local_error
        raise
    if local_error is not None:
        raise MdpStateError(
            "MDP: decoder-ready gate consensus succeeded despite a local error."
        ) from local_error
    return ready
