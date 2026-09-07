# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL model adapter for MDP.

The adapter is everything model-specific MDP core needs: native batch
collation into a :class:`CapturedMicrobatch`, an integer LPT cost, the shared
vision-encoder factory, and a chunk-oblivious ``encode``. It lives in
``examples/multimodal_dev`` because ``megatron/core/mdp`` must not import
model packages.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Iterator, Optional

import torch

from examples.multimodal_dev.models.qwen35_vl.configuration import VISION_KWARGS
from examples.multimodal_dev.models.qwen35_vl.specs import (
    get_qwen35_vl_encoder_cp_vision_spec,
    get_qwen35_vl_vision_spec,
)
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.mdp.dynamic_cp import (
    DynamicCpGroupMembership,
    GlobalSampleId,
    GlobalVisionItemId,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderGlobalManifest,
    DecoderMicrobatchKey,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderSourceWindow,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
    finalize_decoder_source_window,
    validate_decoder_global_manifest,
    validate_decoder_payload_packet,
    validate_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderCpAssignment,
    DecoderSampleMetadata,
    EncoderVisionItemMetadata,
    EncoderWorkEstimate,
)
from megatron.core.mdp.dynamic_encoder_adapter_capability import (
    register_dynamic_encoder_adapter_class,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError
from megatron.core.mdp.protocols import (
    CapturedMicrobatch,
    CapturedVisionItem,
    DynamicEncoderCpBinding,
    VisionCaptureMode,
)
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams

_DECODER_ROUTED_FIELD_ORDER = ("input_ids", "labels", "loss_mask", "padding_mask", "position_ids")
_DECODER_REQUIRED_TENSOR_FIELDS = _DECODER_ROUTED_FIELD_ORDER[:-1]
_DECODER_ALLOWED_PAYLOAD_FIELDS = frozenset(
    (*_DECODER_ROUTED_FIELD_ORDER, "attention_mask", "image_grid_thw")
)
_INTEGER_DTYPES = frozenset((torch.int32, torch.int64))
_DYNAMIC_ENCODER_CP_BINDING_SLOT = "_mdp_dynamic_encoder_cp_binding"


def _add_restore_note(primary_error: BaseException, restore_error: BaseException) -> None:
    try:
        primary_error.add_note(
            f"Suppressed dynamic encoder CP restoration failure: {restore_error!r}"
        )
    except BaseException:
        pass


def _bind_dynamic_encoder_cp(
    encoder, model, *, membership, global_rank, group_attribute, size_attribute
):
    """Install a validated subgroup and return its one-shot restoration owner."""
    if type(membership) is not DynamicCpGroupMembership:
        raise MdpConfigurationError("MDP: dynamic encoder CP membership is typed.")
    if membership.group_size not in (1, 2, 4):
        raise MdpConfigurationError("MDP: dynamic encoder CP membership is E1, E2, or E4.")
    if type(global_rank) is not int or global_rank not in membership.ranks:
        raise MdpConfigurationError("MDP: dynamic encoder CP rank belongs to its membership.")
    group = membership.group
    size = getattr(group, "size", None)
    rank = getattr(group, "rank", None)
    if not callable(size) or not callable(rank):
        raise MdpConfigurationError("MDP: dynamic encoder CP group exposes size and local rank.")
    actual_size = size()
    actual_rank = rank()
    if type(actual_size) is not int or actual_size != membership.group_size:
        raise MdpConfigurationError("MDP: dynamic encoder CP group size matches membership.")
    if type(actual_rank) is not int or actual_rank != membership.ranks.index(global_rank):
        raise MdpConfigurationError("MDP: dynamic encoder CP local rank matches global rank.")
    if getattr(encoder, _DYNAMIC_ENCODER_CP_BINDING_SLOT, None) is not None:
        raise MdpStateError("MDP: encoder already has an active dynamic encoder CP binding.")
    if not hasattr(model, "config") or not hasattr(model.config, "context_parallel_size"):
        raise MdpConfigurationError("MDP: dynamic encoder CP model has configured CP authority.")
    if not hasattr(model, group_attribute):
        raise MdpConfigurationError("MDP: dynamic encoder CP model has a CP group authority.")
    if size_attribute is not None and not hasattr(model, size_attribute):
        raise MdpConfigurationError("MDP: dynamic encoder CP model has a CP size authority.")

    attentions = tuple(
        module for module in model.modules() if isinstance(module, TEDotProductAttention)
    )
    if not attentions:
        raise MdpConfigurationError("MDP: dynamic encoder CP model has TE attention modules.")
    for attention in attentions:
        if not callable(getattr(attention, "set_context_parallel_group", None)):
            raise MdpConfigurationError("MDP: TE attention exposes context parallel binding.")
        if not hasattr(attention, "config") or not hasattr(
            attention.config, "context_parallel_size"
        ):
            raise MdpConfigurationError("MDP: TE attention has configured CP authority.")

    model_cp_size = model.config.context_parallel_size
    model_group = getattr(model, group_attribute)
    model_size = None if size_attribute is None else getattr(model, size_attribute)
    attention_state = tuple(
        (
            attention,
            attention.cp_group,
            attention.cp_global_ranks,
            attention.config.context_parallel_size,
            attention.cp_stream,
            attention.cp_comm_type,
        )
        for attention in attentions
    )
    target_group = None if membership.group_size == 1 else group
    target_ranks = None if membership.group_size == 1 else membership.ranks
    binding = None

    def is_current():
        return getattr(encoder, _DYNAMIC_ENCODER_CP_BINDING_SLOT, None) is binding

    def restore(primary_error):
        restore_primary = primary_error

        def attempt(action):
            nonlocal restore_primary
            try:
                action()
            except BaseException as restore_error:
                if restore_primary is None:
                    restore_primary = restore_error
                else:
                    _add_restore_note(restore_primary, restore_error)

        for attention, old_group, old_ranks, old_size, stream, comm_type in reversed(
            attention_state
        ):
            attempt(
                lambda attention=attention, old_group=old_group, old_ranks=old_ranks, stream=stream, comm_type=comm_type: attention.set_context_parallel_group(
                    old_group, old_ranks, None if old_group is None else stream, comm_type
                )
            )
            attempt(
                lambda attention=attention, old_size=old_size: setattr(
                    attention.config, "context_parallel_size", old_size
                )
            )
            attempt(
                lambda attention=attention, old_group=old_group: setattr(
                    attention, "cp_group", old_group
                )
            )
            attempt(
                lambda attention=attention, old_ranks=old_ranks: setattr(
                    attention, "cp_global_ranks", old_ranks
                )
            )
            attempt(
                lambda attention=attention, stream=stream: setattr(attention, "cp_stream", stream)
            )
            attempt(
                lambda attention=attention, comm_type=comm_type: setattr(
                    attention, "cp_comm_type", comm_type
                )
            )
        attempt(lambda: setattr(model.config, "context_parallel_size", model_cp_size))
        attempt(lambda: setattr(model, group_attribute, model_group))
        if size_attribute is not None:
            attempt(lambda: setattr(model, size_attribute, model_size))
        try:
            setattr(encoder, _DYNAMIC_ENCODER_CP_BINDING_SLOT, None)
        except BaseException as restore_error:
            if restore_primary is None:
                restore_primary = restore_error
            else:
                _add_restore_note(restore_primary, restore_error)
        if primary_error is None and restore_primary is not None:
            raise restore_primary

    binding = DynamicEncoderCpBinding(membership, is_current=is_current, restore=restore)
    setattr(encoder, _DYNAMIC_ENCODER_CP_BINDING_SLOT, binding)
    try:
        model.config.context_parallel_size = membership.group_size
        setattr(model, group_attribute, target_group)
        if size_attribute is not None:
            setattr(model, size_attribute, membership.group_size)
        for attention, _group, _ranks, _size, stream, comm_type in attention_state:
            attention.config.context_parallel_size = membership.group_size
            attention.set_context_parallel_group(
                target_group, target_ranks, None if target_group is None else stream, comm_type
            )
    except BaseException as primary_error:
        binding.restore(primary_error)
        raise
    return binding


class MultimodalDecoderPayloadCodec:
    """Typed, pixel-free source codec for MDP decoder Dynamic-CP.

    Source tensors remain on their original device and storage. The codec
    creates per-sample physical THD views and delegates canonical catalog and
    digest validation to :mod:`megatron.core.mdp.dynamic_cp_execution`.
    """

    @staticmethod
    def _require_nonnegative_integer(name: str, value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise MdpConfigurationError(
                f"MDP: {name}={value!r} violates: {name} is a non-negative integer."
            )
        return value

    @classmethod
    def _cu_values(cls, name: str, value: Any) -> tuple[int, ...]:
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise MdpConfigurationError(f"MDP: decoder {name} is a one-dimensional tensor.")
        if value.dtype not in _INTEGER_DTYPES:
            raise MdpConfigurationError(f"MDP: decoder {name} uses int32 or int64 metadata.")
        values = tuple(int(component) for component in value.tolist())
        if len(values) < 2 or values[0] != 0:
            raise MdpConfigurationError(f"MDP: decoder {name} starts at zero and names samples.")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise MdpConfigurationError(
                f"MDP: decoder {name} has strictly increasing sample boundaries."
            )
        return values

    @classmethod
    def _packed_sample_lengths(cls, packed: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not isinstance(packed, PackedSeqParams) or packed.qkv_format != "thd":
            raise MdpConfigurationError(
                "MDP: decoder Dynamic-CP source records require THD PackedSeqParams."
            )
        valid_q = cls._cu_values("cu_seqlens_q", packed.cu_seqlens_q)
        valid_kv = cls._cu_values("cu_seqlens_kv", packed.cu_seqlens_kv)
        padded_q = cls._cu_values("cu_seqlens_q_padded", packed.cu_seqlens_q_padded)
        padded_kv = cls._cu_values("cu_seqlens_kv_padded", packed.cu_seqlens_kv_padded)
        if valid_q != valid_kv or padded_q != padded_kv:
            raise MdpConfigurationError(
                "MDP: decoder compact and padded THD q/kv cumulative lengths agree."
            )
        if len(valid_q) != len(padded_q):
            raise MdpConfigurationError(
                "MDP: decoder compact and padded THD metadata name the same samples."
            )
        total_tokens = cls._require_nonnegative_integer("total_tokens", packed.total_tokens)
        if padded_q[-1] != total_tokens:
            raise MdpConfigurationError("MDP: decoder padded THD endpoint and total_tokens agree.")
        valid_lengths = tuple(right - left for left, right in zip(valid_q, valid_q[1:]))
        padded_lengths = tuple(right - left for left, right in zip(padded_q, padded_q[1:]))
        if any(valid > padded for valid, padded in zip(valid_lengths, padded_lengths)):
            raise MdpConfigurationError(
                "MDP: each decoder valid THD length fits its physical padded interval."
            )
        return valid_lengths, padded_lengths

    @staticmethod
    def _grid_rows(value: Any) -> tuple[tuple[int, int, int], ...]:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape[1] != 3
            or value.dtype not in _INTEGER_DTYPES
        ):
            raise MdpConfigurationError(
                "MDP: image_grid_thw is an integer tensor with shape [num_items, 3]."
            )
        rows = tuple(tuple(int(component) for component in row) for row in value.tolist())
        if any(component <= 0 for row in rows for component in row):
            raise MdpConfigurationError("MDP: image_grid_thw dimensions are positive.")
        return rows

    @classmethod
    def _validate_model_payload(
        cls, payload: Any, *, total_tokens: int
    ) -> tuple[tuple[int, int, int], ...]:
        if not isinstance(payload, Mapping):
            raise MdpConfigurationError("MDP: decoder model_payload is a mapping.")
        unexpected = tuple(name for name in payload if name not in _DECODER_ALLOWED_PAYLOAD_FIELDS)
        if unexpected:
            raise MdpConfigurationError(
                f"MDP: decoder payload allowlist rejects fields {unexpected!r}."
            )
        missing = tuple(name for name in _DECODER_REQUIRED_TENSOR_FIELDS if name not in payload)
        if missing:
            raise MdpConfigurationError(
                f"MDP: decoder payload allowlist requires fields {missing!r}."
            )
        if "image_grid_thw" not in payload:
            raise MdpConfigurationError(
                "MDP: decoder payload allowlist requires image_grid_thw metadata."
            )

        for name in _DECODER_ROUTED_FIELD_ORDER:
            value = payload.get(name)
            if name == "position_ids" and value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise MdpConfigurationError(
                    f"MDP: decoder field {name} is a tensor or None; nested image payload "
                    "objects are unsupported."
                )
            if value.ndim == 0 or value.shape[-1] != total_tokens:
                raise MdpConfigurationError(
                    f"MDP: decoder field {name} token extent matches the full physical "
                    f"record ({total_tokens})."
                )
            if name in _DECODER_REQUIRED_TENSOR_FIELDS and (value.ndim != 2 or value.shape[0] != 1):
                raise MdpConfigurationError(f"MDP: decoder field {name} has Qwen THD shape [1, T].")
            if name == "position_ids" and not (
                (value.ndim == 2 and value.shape[0] == 1)
                or (value.ndim == 3 and tuple(value.shape[:2]) == (3, 1))
            ):
                raise MdpConfigurationError(
                    "MDP: position_ids has Qwen THD shape [1, T] or [3, 1, T]."
                )

        if payload.get("attention_mask") is not None:
            raise MdpConfigurationError(
                "MDP: attention_mask accepts None only; tensor or nested image payload "
                "objects have no authorized layout-specific slicing rule."
            )
        return cls._grid_rows(payload["image_grid_thw"])

    @staticmethod
    def _packet_presence_signature(
        packet: DecoderPayloadPacket,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (tuple(spec.name for spec in packet.field_specs), packet.none_fields)

    @staticmethod
    def _packet_schema_signature(
        packet: DecoderPayloadPacket,
    ) -> tuple[tuple[str, torch.dtype, str, tuple[int, ...]], ...]:
        return tuple(
            (spec.name, spec.dtype, spec.device_type, spec.shape[:-1])
            for spec in packet.field_specs
        )

    @classmethod
    def _validate_packet_collection(cls, packets: Sequence[DecoderPayloadPacket]) -> None:
        if not packets:
            raise MdpPlanError("MDP: decoder payload packet collection is non-empty.")
        for packet in packets:
            validate_decoder_payload_packet(packet)
        presence = {cls._packet_presence_signature(packet) for packet in packets}
        if len(presence) != 1:
            raise MdpConfigurationError(
                "MDP: decoder optional tensor/None field presence is consistent across packets."
            )
        schemas = {cls._packet_schema_signature(packet) for packet in packets}
        if len(schemas) != 1:
            raise MdpConfigurationError(
                "MDP: decoder tensor dtype, device, rank, and leading shape agree across packets."
            )

    @classmethod
    def _build_packet(
        cls,
        payload: Mapping[str, Any],
        *,
        sample_id: GlobalSampleId,
        valid_seqlen: int,
        padded_start: int,
        padded_seqlen: int,
    ) -> DecoderPayloadPacket:
        tensor_fields = {}
        none_fields = []
        padded_end = padded_start + padded_seqlen
        for name in _DECODER_ROUTED_FIELD_ORDER:
            value = payload.get(name)
            if value is None:
                none_fields.append(name)
            else:
                tensor_fields[name] = value[..., padded_start:padded_end]
        none_fields.append("attention_mask")

        field_specs = tuple(
            DecoderTensorFieldSpec(
                name=name,
                dtype=tensor.dtype,
                shape=tuple(tensor.shape),
                device_type=tensor.device.type,
            )
            for name, tensor in tensor_fields.items()
        )
        position_ids = payload.get("position_ids")
        header = DecoderPayloadHeaderV1(
            schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
            source_dp_lane=sample_id.source_dp_lane,
            local_sample_order=sample_id.local_sample_order,
            valid_seqlen=valid_seqlen,
            padded_seqlen=padded_seqlen,
            tensor_field_count=len(field_specs),
            none_field_count=len(none_fields),
            position_components_or_minus_one=(
                -1 if position_ids is None else int(position_ids.shape[0])
            ),
        )
        return DecoderPayloadPacket(
            schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
            sample_id=sample_id,
            valid_seqlen=valid_seqlen,
            padded_seqlen=padded_seqlen,
            header=header.to_wire_tuple(),
            field_specs=field_specs,
            tensor_fields=MappingProxyType(tensor_fields),
            none_fields=tuple(none_fields),
        )

    @classmethod
    def _record_items(
        cls,
        record: MdpMicrobatchRecord,
        *,
        source_dp_lane: int,
        sample_base: int,
        item_base: int,
        valid_lengths: tuple[int, ...],
        padded_lengths: tuple[int, ...],
        grid_rows: tuple[tuple[int, int, int], ...],
    ) -> tuple[
        tuple[tuple[EncoderVisionItemMetadata, ...], ...], tuple[DecoderVisionItemMetadata, ...]
    ]:
        if not isinstance(record.vision_items, tuple):
            raise MdpConfigurationError("MDP: decoder vision_items is an immutable tuple.")
        if len(record.vision_items) != len(grid_rows):
            raise MdpConfigurationError(
                "MDP: decoder vision-item order and image_grid_thw rows agree."
            )
        if not isinstance(record.text_only, bool) or record.text_only != (
            len(record.vision_items) == 0
        ):
            raise MdpConfigurationError(
                "MDP: decoder text_only state agrees with the absence of vision items."
            )

        padded_starts = [0]
        for length in padded_lengths:
            padded_starts.append(padded_starts[-1] + length)
        encoder_items = [[] for _ in valid_lengths]
        decoder_items = []
        for index, (item, grid_row) in enumerate(zip(record.vision_items, grid_rows)):
            if not isinstance(item, MdpMicrobatchVisionRecord):
                raise MdpConfigurationError(
                    "MDP: decoder source vision records use MdpMicrobatchVisionRecord."
                )
            local_sample = cls._require_nonnegative_integer("vision item sample_id", item.sample_id)
            if local_sample >= len(valid_lengths):
                raise MdpConfigurationError(
                    "MDP: decoder vision item names a sample in its source microbatch."
                )
            image_ordinal = cls._require_nonnegative_integer(
                "vision item image_ordinal", item.image_ordinal
            )
            if image_ordinal != len(encoder_items[local_sample]):
                raise MdpConfigurationError(
                    "MDP: decoder vision-item ordinals are contiguous in source order."
                )
            if not isinstance(item.grid_thw, tuple) or tuple(item.grid_thw) != grid_row:
                raise MdpConfigurationError(
                    "MDP: decoder vision-item grid order matches image_grid_thw metadata."
                )
            output_rows = cls._require_nonnegative_integer(
                "vision item output_rows", item.output_rows
            )
            if output_rows == 0:
                raise MdpConfigurationError("MDP: decoder vision item output_rows is positive.")
            if (
                not isinstance(item.decoder_positions, tuple)
                or len(item.decoder_positions) != output_rows
            ):
                raise MdpConfigurationError(
                    "MDP: decoder vision item decoder_positions match output_rows."
                )
            decoder_positions = tuple(
                cls._require_nonnegative_integer("decoder vision slot", position)
                for position in item.decoder_positions
            )
            if len(set(decoder_positions)) != len(decoder_positions):
                raise MdpConfigurationError("MDP: decoder vision item slots are unique.")
            valid_start = padded_starts[local_sample]
            valid_end = valid_start + valid_lengths[local_sample]
            if any(
                position < valid_start or position >= valid_end for position in decoder_positions
            ):
                raise MdpConfigurationError(
                    "MDP: decoder vision item slot lies inside its owning sample valid interval."
                )

            sample_id = GlobalSampleId(source_dp_lane, sample_base + local_sample)
            item_id = GlobalVisionItemId(source_dp_lane, item_base + index)
            encoder_item = EncoderVisionItemMetadata(
                item_id=item_id, sample_id=sample_id, image_ordinal=image_ordinal
            )
            encoder_items[local_sample].append(encoder_item)
            decoder_items.append(
                DecoderVisionItemMetadata(
                    item_id=item_id,
                    sample_id=sample_id,
                    image_ordinal=image_ordinal,
                    grid_thw=grid_row,
                    output_rows=output_rows,
                    decoder_offsets=tuple(position - valid_start for position in decoder_positions),
                )
            )
        return tuple(tuple(items) for items in encoder_items), tuple(decoder_items)

    def build_source_window(
        self, records: Sequence[MdpMicrobatchRecord], *, source_dp_lane: int
    ) -> DecoderSourceWindow:
        """Catalog source-local THD views without copying decoder tensors."""
        window, _ = self.build_source_window_with_locations(records, source_dp_lane=source_dp_lane)
        return window

    def build_source_window_with_locations(
        self, records: Sequence[MdpMicrobatchRecord], *, source_dp_lane: int
    ) -> tuple[DecoderSourceWindow, Mapping[GlobalSampleId, tuple[int, int]]]:
        """Catalog a source window and its microbatch-local sample locations."""
        lane = self._require_nonnegative_integer("source_dp_lane", source_dp_lane)
        if (
            not isinstance(records, Sequence)
            or isinstance(records, (str, bytes, bytearray))
            or not records
        ):
            raise MdpConfigurationError(
                "MDP: decoder source codec requires a non-empty ordered record sequence."
            )

        samples = []
        items = []
        packets = []
        sample_locations = {}
        next_sample_order = 0
        next_item_id = 0
        for record in records:
            if not isinstance(record, MdpMicrobatchRecord):
                raise MdpConfigurationError(
                    "MDP: decoder source codec consumes MdpMicrobatchRecord values."
                )
            valid_lengths, padded_lengths = self._packed_sample_lengths(
                record.decoder_packed_seq_params
            )
            total_tokens = sum(padded_lengths)
            grid_rows = self._validate_model_payload(
                record.model_payload, total_tokens=total_tokens
            )
            record_encoder_items, record_decoder_items = self._record_items(
                record,
                source_dp_lane=lane,
                sample_base=next_sample_order,
                item_base=next_item_id,
                valid_lengths=valid_lengths,
                padded_lengths=padded_lengths,
                grid_rows=grid_rows,
            )
            items.extend(record_decoder_items)

            padded_start = 0
            for local_sample, (valid_seqlen, padded_seqlen) in enumerate(
                zip(valid_lengths, padded_lengths)
            ):
                sample_id = GlobalSampleId(lane, next_sample_order + local_sample)
                sample_locations[sample_id] = (record.microbatch_id, local_sample)
                samples.append(
                    DecoderSampleMetadata(
                        sample_id=sample_id,
                        valid_seqlen=valid_seqlen,
                        padded_seqlen=padded_seqlen,
                        vision_items=record_encoder_items[local_sample],
                    )
                )
                packets.append(
                    self._build_packet(
                        record.model_payload,
                        sample_id=sample_id,
                        valid_seqlen=valid_seqlen,
                        padded_start=padded_start,
                        padded_seqlen=padded_seqlen,
                    )
                )
                padded_start += padded_seqlen
            next_sample_order += len(valid_lengths)
            next_item_id += len(record_decoder_items)

        window = finalize_decoder_source_window(
            source_dp_lane=lane, samples=samples, items=items, packets=packets
        )
        validate_decoder_source_window(window)
        self._validate_packet_collection(window.packets)
        return window, MappingProxyType(sample_locations)

    def validate_packet(self, packet: DecoderPayloadPacket) -> None:
        """Validate one destination packet against the fixed Qwen decoder schema."""
        validate_decoder_payload_packet(packet)
        tensor_names = tuple(spec.name for spec in packet.field_specs)
        supported_none_fields = (*_DECODER_ROUTED_FIELD_ORDER[-1:], "attention_mask")
        if any(name not in supported_none_fields for name in packet.none_fields):
            raise MdpConfigurationError(
                "MDP: decoder payload packet None fields use the supported optional schema."
            )
        canonical_none_fields = tuple(
            name for name in supported_none_fields if name in packet.none_fields
        )
        if packet.none_fields != canonical_none_fields:
            raise MdpConfigurationError(
                "MDP: decoder payload packet optional None fields use canonical order."
            )
        expected_tensor_names = tuple(
            name for name in _DECODER_ROUTED_FIELD_ORDER if name not in packet.none_fields
        )
        if tensor_names != expected_tensor_names:
            raise MdpConfigurationError(
                "MDP: decoder payload packet tensor fields use the fixed routed-field order."
            )
        for spec in packet.field_specs:
            tensor = packet.tensor_fields[spec.name]
            if spec.name in _DECODER_REQUIRED_TENSOR_FIELDS and (
                tensor.ndim != 2 or tensor.shape[0] != 1
            ):
                raise MdpConfigurationError(
                    f"MDP: decoder packet field {spec.name} has Qwen THD shape [1, T]."
                )
            if spec.name == "position_ids" and not (
                (tensor.ndim == 2 and tensor.shape[0] == 1)
                or (tensor.ndim == 3 and tuple(tensor.shape[:2]) == (3, 1))
            ):
                raise MdpConfigurationError(
                    "MDP: decoder packet position_ids has Qwen THD shape [1, T] or [3, 1, T]."
                )

    @classmethod
    def _validate_assignment(cls, assignment: Any) -> DecoderCpAssignment:
        if not isinstance(assignment, DecoderCpAssignment):
            raise MdpPlanError("MDP: decoder rebuild requires a DecoderCpAssignment.")
        if not isinstance(assignment.sample_ids, tuple) or not assignment.sample_ids:
            raise MdpPlanError("MDP: decoder assignment names an immutable non-empty sample tuple.")
        if any(not isinstance(sample_id, GlobalSampleId) for sample_id in assignment.sample_ids):
            raise MdpPlanError("MDP: decoder assignment sample IDs use GlobalSampleId.")
        if len(set(assignment.sample_ids)) != len(assignment.sample_ids):
            raise MdpPlanError("MDP: decoder assignment sample IDs are unique.")
        if not isinstance(assignment.endpoint_ranks, tuple) or not assignment.endpoint_ranks:
            raise MdpPlanError("MDP: decoder assignment has immutable non-empty endpoint ranks.")
        try:
            endpoint_ranks = tuple(
                cls._require_nonnegative_integer(
                    f"decoder assignment endpoint_ranks[{index}]", rank
                )
                for index, rank in enumerate(assignment.endpoint_ranks)
            )
        except MdpConfigurationError as error:
            raise MdpPlanError(
                "MDP: decoder assignment endpoint ranks are non-negative integers."
            ) from error
        if len(set(endpoint_ranks)) != len(endpoint_ranks):
            raise MdpPlanError("MDP: decoder assignment endpoint ranks are unique.")
        return assignment

    def _select_rebuild_inputs(
        self, global_manifest: DecoderGlobalManifest, assignment: DecoderCpAssignment, packets: Any
    ) -> tuple[
        tuple[DecoderSampleMetadata, ...],
        tuple[DecoderVisionItemMetadata, ...],
        tuple[DecoderPayloadPacket, ...],
    ]:
        validate_decoder_global_manifest(global_manifest)
        assignment = self._validate_assignment(assignment)
        if not isinstance(packets, tuple):
            raise MdpConfigurationError(
                "MDP: global decoder manifest rebuild requires an ordered packet tuple."
            )
        if any(not isinstance(packet, DecoderPayloadPacket) for packet in packets):
            raise MdpConfigurationError("MDP: global decoder rebuild uses typed packet members.")
        packet_ids = tuple(packet.sample_id for packet in packets)
        if packet_ids != assignment.sample_ids:
            raise MdpPlanError(
                "MDP: decoder destination packets match the exact assignment sample order."
            )
        for packet in packets:
            self.validate_packet(packet)
        self._validate_packet_collection(packets)

        sample_by_id = {sample.sample_id: sample for sample in global_manifest.samples}
        payload_by_id = {payload.sample_id: payload for payload in global_manifest.payloads}
        try:
            selected_samples = tuple(sample_by_id[sample_id] for sample_id in assignment.sample_ids)
            expected_payloads = tuple(
                payload_by_id[sample_id] for sample_id in assignment.sample_ids
            )
        except KeyError as error:
            raise MdpPlanError(
                "MDP: decoder assignment names a sample in the global manifest."
            ) from error
        if tuple(packet.metadata() for packet in packets) != expected_payloads:
            raise MdpPlanError(
                "MDP: decoder packets exactly match global manifest payload metadata."
            )
        return selected_samples, global_manifest.items, packets

    @staticmethod
    def _cumulative_lengths(lengths: Sequence[int]) -> tuple[int, ...]:
        values = [0]
        for length in lengths:
            values.append(values[-1] + length)
        return tuple(values)

    def rebuild_microbatch(
        self,
        global_manifest: DecoderGlobalManifest,
        assignment: DecoderCpAssignment,
        *,
        packets: tuple[DecoderPayloadPacket, ...],
        key: DecoderMicrobatchKey,
        cp_group: Any,
        cp_partition_mode: str,
    ) -> MdpMicrobatchRecord:
        """Rebuild one destination assignment in physical THD sample order."""
        if not isinstance(key, DecoderMicrobatchKey):
            raise MdpConfigurationError("MDP: decoder rebuild key is a DecoderMicrobatchKey.")
        if cp_partition_mode not in ("zigzag", "contiguous"):
            raise MdpConfigurationError(
                "MDP: decoder rebuild CP partition mode is zigzag or contiguous."
            )
        selected_samples, catalog_items, selected_packets = self._select_rebuild_inputs(
            global_manifest, assignment, packets
        )
        try:
            group_size = getattr(cp_group, "size", None)
        except Exception as error:
            raise MdpConfigurationError("MDP: decoder rebuild CP group query failed.") from error
        if not callable(group_size):
            raise MdpConfigurationError("MDP: decoder rebuild CP group exposes a callable size.")
        try:
            actual_group_size = group_size()
        except Exception as error:
            raise MdpConfigurationError("MDP: decoder rebuild CP group query failed.") from error
        if (
            not isinstance(actual_group_size, int)
            or isinstance(actual_group_size, bool)
            or actual_group_size != assignment.local_cp_size
        ):
            raise MdpConfigurationError(
                "MDP: decoder rebuild CP group size matches the assignment endpoints."
            )

        tensor_names = tuple(spec.name for spec in selected_packets[0].field_specs)
        none_fields = selected_packets[0].none_fields
        rebuilt_payload = {}
        try:
            for name in _DECODER_ROUTED_FIELD_ORDER:
                if name in tensor_names:
                    rebuilt_payload[name] = torch.cat(
                        tuple(packet.tensor_fields[name] for packet in selected_packets), dim=-1
                    )
                elif name in none_fields:
                    rebuilt_payload[name] = None
        except Exception as error:
            raise MdpConfigurationError(
                "MDP: decoder destination packet tensors concatenate on physical T."
            ) from error
        if "attention_mask" in none_fields:
            rebuilt_payload["attention_mask"] = None

        padded_lengths = [sample.padded_seqlen for sample in selected_samples]
        total_padded = sum(padded_lengths)
        if cp_partition_mode == "contiguous":
            aligned_total = ((total_padded + actual_group_size - 1) // actual_group_size) * (
                actual_group_size
            )
            tail_padding = aligned_total - total_padded
            if tail_padding:
                pad_values = {
                    "input_ids": 0,
                    "labels": -100,
                    "loss_mask": 0,
                    "padding_mask": True,
                    "position_ids": 0,
                }
                for name, value in tuple(rebuilt_payload.items()):
                    if value is not None and name in pad_values:
                        rebuilt_payload[name] = torch.nn.functional.pad(
                            value, (0, tail_padding), value=pad_values[name]
                        )
                padded_lengths[-1] += tail_padding

        item_by_id = {item.item_id: item for item in catalog_items}
        vision_records = []
        grids = []
        padded_start = 0
        for destination_sample, sample in enumerate(selected_samples):
            for encoder_item in sample.vision_items:
                try:
                    item = item_by_id[encoder_item.item_id]
                except KeyError as error:
                    raise MdpPlanError(
                        "MDP: decoder rebuild manifest contains every assigned vision item."
                    ) from error
                if item.sample_id != sample.sample_id:
                    raise MdpPlanError(
                        "MDP: decoder rebuild vision item remains attached to its source sample."
                    )
                vision_records.append(
                    MdpMicrobatchVisionRecord(
                        global_item_id=item.item_id,
                        sample_id=destination_sample,
                        image_ordinal=item.image_ordinal,
                        grid_thw=item.grid_thw,
                        output_rows=item.output_rows,
                        decoder_positions=tuple(
                            padded_start + offset for offset in item.decoder_offsets
                        ),
                    )
                )
                grids.append(item.grid_thw)
            padded_start += sample.padded_seqlen

        rebuilt_payload["image_grid_thw"] = (
            torch.tensor(grids, dtype=torch.int64, device="cpu")
            if grids
            else torch.empty((0, 3), dtype=torch.int64, device="cpu")
        )
        valid_cu_values = self._cumulative_lengths(
            tuple(sample.valid_seqlen for sample in selected_samples)
        )
        padded_cu_values = self._cumulative_lengths(
            padded_lengths
        )
        metadata_device = selected_packets[0].tensor_fields[tensor_names[0]].device
        valid_cu = torch.tensor(valid_cu_values, dtype=torch.int32, device=metadata_device)
        padded_cu = torch.tensor(padded_cu_values, dtype=torch.int32, device=metadata_device)
        max_padded_seqlen = max(padded_lengths)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=valid_cu,
            cu_seqlens_kv=valid_cu.clone(),
            cu_seqlens_q_padded=padded_cu,
            cu_seqlens_kv_padded=padded_cu.clone(),
            max_seqlen_q=max_padded_seqlen,
            max_seqlen_kv=max_padded_seqlen,
            local_cp_size=assignment.local_cp_size,
            cp_group=cp_group,
            total_tokens=padded_cu_values[-1],
            cp_partition_mode=cp_partition_mode,
        )
        return MdpMicrobatchRecord(
            microbatch_id=key.microbatch_index,
            text_only=not vision_records,
            vision_items=tuple(vision_records),
            decoder_packed_seq_params=packed,
            model_payload=MappingProxyType(rebuilt_payload),
        )


class Qwen35VLMdpAdapter:
    """MdpModelAdapter implementation for Qwen3.5-VL.

    Args:
        out_hidden_size: Language decoder hidden size (patch-merger output).
        vision_kwargs: Optional override of the Qwen3.5-VL vision kwargs.
    """

    def __init__(self, out_hidden_size: int, vision_kwargs: Optional[dict] = None):
        self._vision_kwargs = dict(vision_kwargs or VISION_KWARGS)
        self._vision_kwargs["out_hidden_size"] = out_hidden_size
        self.embedding_width = out_hidden_size
        self.spatial_merge_size = self._vision_kwargs["spatial_merge_size"]
        self.payload_width = (
            self._vision_kwargs["in_channels"]
            * self._vision_kwargs["temporal_patch_size"]
            * self._vision_kwargs["patch_size"] ** 2
        )

    def build_dynamic_decoder_payload_codec(self) -> MultimodalDecoderPayloadCodec:
        """Build one iteration-independent decoder Dynamic-CP source codec."""
        return MultimodalDecoderPayloadCodec()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture_batch(
        self,
        data_iterator: Iterator,
        *,
        locator_operations,
        expected_locator_arch: str,
        raw_batch_validator=None,
    ) -> Optional[CapturedMicrobatch]:
        """Shared one-microbatch capture with model-owned pre-pack validation.

        Requires the vision sidecar (``--mdp-enable`` makes the collator emit
        it), from which the per-item records and decoder positions are cut.
        The pixel payload and the sidecar keys are removed from the replayed
        model payload: pixels never enter the decoder.
        """
        from examples.multimodal_dev.forward_step import get_batch

        capture_iterator = data_iterator
        if raw_batch_validator is not None:

            def validated_iterator():
                try:
                    raw_batch = next(data_iterator)
                except StopIteration:
                    return
                raw_batch_validator(raw_batch)
                yield raw_batch

            capture_iterator = validated_iterator()
        if locator_operations is None:
            batch = get_batch(capture_iterator)
            capture_mode = VisionCaptureMode.SOURCE_PIXEL_SIDECAR
        else:
            batch = get_batch(
                capture_iterator,
                locator_operations=locator_operations,
                expected_locator_arch=expected_locator_arch,
            )
            capture_mode = VisionCaptureMode.STABLE_LOCATOR_CATALOG
        if batch is None:
            return None
        if "vision_item_meta" not in batch:
            raise RuntimeError(
                "MDP adapter needs the vision sidecar; the collator must run with "
                "with_vision_sidecar=True (set by --mdp-enable)."
            )
        meta = batch.pop("vision_item_meta")
        positions = batch.pop("vision_decoder_positions")
        pixels = batch.pop("pixel_values", None)
        locators = batch.pop("vision_locators", ())
        merge = self.spatial_merge_size

        items = []
        position_cursor = 0
        # One D2H transfer for the whole positions tensor; slicing the CPU
        # copy per item avoids a device sync per vision item.
        positions_cpu = positions.cpu().tolist()
        for row in meta.cpu().tolist():
            sample_id, ordinal, t, h, w, payload_row_start = (int(v) for v in row)
            output_rows = t * (h // merge) * (w // merge)
            decoder_positions = tuple(
                positions_cpu[position_cursor : position_cursor + output_rows]
            )
            position_cursor += output_rows
            items.append(
                CapturedVisionItem(
                    sample_id=sample_id,
                    image_ordinal=ordinal,
                    grid_thw=(t, h, w),
                    payload_row_start=payload_row_start,
                    payload_rows=t * h * w,
                    decoder_positions=decoder_positions,
                )
            )
        if position_cursor != positions.numel():
            raise RuntimeError(
                f"vision sidecar mismatch: consumed {position_cursor} decoder positions "
                f"of {positions.numel()}"
            )

        packed_seq_params = batch.pop("packed_seq_params", None)
        if pixels is not None and pixels.shape[0] == 0:
            pixels = None
        if pixels is not None and pixels.dtype == torch.float32:
            pixels = pixels.bfloat16()
        return CapturedMicrobatch(
            decoder_packed_seq_params=packed_seq_params,
            vision_items=tuple(items),
            flat_pixel_payload=pixels,
            model_payload=MappingProxyType(batch),
            vision_capture_mode=capture_mode,
            vision_locators=locators,
        )

    def get_batch(
        self, data_iterator: Iterator, *, locator_operations=None
    ) -> Optional[CapturedMicrobatch]:
        """Capture Qwen through the shared native THD sidecar path."""
        return self._capture_batch(
            data_iterator,
            locator_operations=locator_operations,
            expected_locator_arch="qwen35_vl",
        )

    def freeze_vision_locator(
        self,
        descriptor: Any,
        *,
        dataset_root: str,
        grid_thw: tuple[int, int, int],
        declared_dimensions: tuple[int, int] | None,
    ):
        """Freeze one Energon descriptor into stable no-byte metadata."""
        from examples.multimodal_dev.data.energon.materializer import freeze_descriptor_locator

        return freeze_descriptor_locator(
            descriptor,
            dataset_root=dataset_root,
            grid_thw=grid_thw,
            declared_dimensions=declared_dimensions,
        )

    def materialize_vision_locator(self, locator: Any) -> bytes:
        """Materialize one escrowed locator through the generic Energon contract."""
        from examples.multimodal_dev.data.energon.materializer import vision_locator_image_bytes

        return vision_locator_image_bytes(locator)

    def prepare_materialized_vision_payloads(self, locators, encoded_payloads):
        """Decode and patchify exact in-memory payloads with Qwen's materializer."""
        from examples.multimodal_dev.models.qwen35_vl.energon import build_image_materializer

        descriptors = []
        for locator, payload in zip(locators, encoded_payloads, strict=True):
            descriptor = {
                "kind": "image_bytes",
                "encoded_image": payload,
                "grid_thw": locator.grid_thw,
            }
            if locator.declared_dimensions is not None:
                descriptor["height"], descriptor["width"] = locator.declared_dimensions
            descriptors.append(descriptor)
        grids = torch.tensor(
            [locator.grid_thw for locator in locators], dtype=torch.int64, device="cpu"
        ).reshape(-1, 3)
        return build_image_materializer(args=None)(tuple(descriptors), grids)

    # ------------------------------------------------------------------
    # Planning cost
    # ------------------------------------------------------------------

    def estimate_cost(self, item: CapturedVisionItem) -> int:
        """Patch rows as the LPT ordering cost; never sizes any buffer."""
        return item.payload_rows

    def estimate_dynamic_encoder_workload(
        self, items: tuple[DecoderVisionItemMetadata, ...], *, group_size: int
    ) -> EncoderWorkEstimate:
        """Estimate native per-rank Qwen THD rows using metadata only."""
        if type(self) is not Qwen35VLMdpAdapter:
            raise MdpConfigurationError(
                "MDP: dynamic encoder workload requires the exact Qwen3.5-VL adapter."
            )
        if not isinstance(items, tuple) or not items:
            raise MdpConfigurationError(
                "MDP: Qwen dynamic encoder items are an immutable non-empty tuple."
            )
        if type(group_size) is not int or group_size not in (1, 2, 4):
            raise MdpConfigurationError("MDP: Qwen dynamic encoder group_size is E1, E2, or E4.")

        merge = self.spatial_merge_size
        seen = set()
        effective_rows = 0
        cost_units = 0
        alignment = 2 * group_size
        for item in items:
            if not isinstance(item, DecoderVisionItemMetadata):
                raise MdpConfigurationError(
                    "MDP: Qwen dynamic encoder items use typed decoder metadata."
                )
            if item.item_id in seen:
                raise MdpConfigurationError(
                    "MDP: Qwen dynamic encoder items have unique item identities."
                )
            seen.add(item.item_id)
            grid = item.grid_thw
            if (
                not isinstance(grid, tuple)
                or len(grid) != 3
                or any(type(value) is not int or value <= 0 for value in grid)
            ):
                raise MdpConfigurationError(
                    "MDP: Qwen dynamic encoder grid_thw has three positive integers."
                )
            temporal, height, width = grid
            if height % merge or width % merge:
                raise MdpConfigurationError(
                    "MDP: Qwen dynamic encoder grid_thw is spatial-merge aligned."
                )
            if type(item.output_rows) is not int or item.output_rows != temporal * (
                height // merge
            ) * (width // merge):
                raise MdpConfigurationError(
                    "MDP: Qwen dynamic encoder output_rows matches grid_thw."
                )
            frame_rows = height * width
            valid_rows = temporal * frame_rows
            padded_frame_rows = (
                frame_rows
                if group_size == 1
                else ((frame_rows + alignment - 1) // alignment) * alignment
            )
            effective_rows += temporal * padded_frame_rows // group_size
            cost_units += valid_rows
        return EncoderWorkEstimate(effective_rows, cost_units)

    # ------------------------------------------------------------------
    # Encoder factory and forward
    # ------------------------------------------------------------------

    def build_encoder(self, model_config, *, pg_collection) -> torch.nn.Module:
        """Same factory as the non-MDP path (models/qwen35_vl/model.py)."""
        kwargs = self._vision_kwargs
        encoder_cp = model_config.context_parallel_size > 1
        return Qwen35VLVisionEncoder(
            config=model_config,
            transformer_layer_spec=(
                get_qwen35_vl_encoder_cp_vision_spec()
                if encoder_cp
                else get_qwen35_vl_vision_spec()
            ),
            encoder_context_parallel=encoder_cp,
            pg_collection=pg_collection,
            in_channels=kwargs["in_channels"],
            patch_size=kwargs["patch_size"],
            temporal_patch_size=kwargs["temporal_patch_size"],
            spatial_merge_size=kwargs["spatial_merge_size"],
            out_hidden_size=kwargs["out_hidden_size"],
            max_num_positions=kwargs["max_num_positions"],
        )

    def bind_dynamic_encoder_cp(self, encoder, *, membership, global_rank):
        """Temporarily bind Qwen vision attention to one encoder subgroup."""
        if type(self) is not Qwen35VLMdpAdapter:
            raise MdpConfigurationError(
                "MDP: dynamic encoder CP binding requires the exact Qwen3.5-VL adapter."
            )
        if not isinstance(encoder, Qwen35VLVisionEncoder):
            raise MdpConfigurationError(
                "MDP: Qwen dynamic encoder CP requires its typed vision encoder."
            )
        return _bind_dynamic_encoder_cp(
            encoder,
            encoder,
            membership=membership,
            global_rank=global_rank,
            group_attribute="_encoder_cp_group",
            size_attribute="_encoder_cp_size",
        )

    def encode(self, encoder: torch.nn.Module, payload: torch.Tensor, layout) -> torch.Tensor:
        """Encoder forward for one (already rebased) chunk sub-layout.

        The encoder builds its vision-only THD ``PackedSeqParams`` internally
        from ``grid_thw`` (one sub-sequence per temporal frame); the decoder
        ``PackedSeqParams`` is never read here.
        """
        # A CPU tensor, deliberately: with the grid cache enabled the encoder
        # consumes grid_thw exclusively as Python lists (tolist), so a device
        # tensor here cost a blocking pageable H2D on the busy compute stream
        # (~2.4 ms/iter measured) followed by D2H readbacks inside the
        # encoder. The encoder moves it to the device itself on the uncached
        # (QWEN35_VL_GRID_CACHE=0) fallback paths that do tensor math on it.
        grid_thw = torch.tensor([segment.grid_thw for segment in layout.segments], dtype=torch.long)
        return encoder(payload, grid_thw)


register_dynamic_encoder_adapter_class(
    Qwen35VLMdpAdapter,
    get_batch=Qwen35VLMdpAdapter.get_batch,
    estimate_cost=Qwen35VLMdpAdapter.estimate_cost,
    build_dynamic_decoder_payload_codec=Qwen35VLMdpAdapter.build_dynamic_decoder_payload_codec,
    estimate_dynamic_encoder_workload=Qwen35VLMdpAdapter.estimate_dynamic_encoder_workload,
    build_encoder=Qwen35VLMdpAdapter.build_encoder,
    bind_dynamic_encoder_cp=Qwen35VLMdpAdapter.bind_dynamic_encoder_cp,
    encode=Qwen35VLMdpAdapter.encode,
    freeze_vision_locator=Qwen35VLMdpAdapter.freeze_vision_locator,
    materialize_vision_locator=Qwen35VLMdpAdapter.materialize_vision_locator,
    prepare_materialized_vision_payloads=Qwen35VLMdpAdapter.prepare_materialized_vision_payloads,
    locator_model_arch="qwen35_vl",
)


def build_mdp_adapter(args, language_config) -> Qwen35VLMdpAdapter:
    """Adapter factory used by the pretrain entry point."""
    return Qwen35VLMdpAdapter(out_hidden_size=language_config.hidden_size)
