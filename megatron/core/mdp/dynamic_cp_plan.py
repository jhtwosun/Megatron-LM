# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure decoder scheduling and encoder-wave plans for MDP Dynamic-CP.

The module deliberately owns no process-group handles and imports no tensor or
distributed package. Decoder packing is delegated to an injected native-shaped
solver; encoder geometry and cost are delegated to an injected adapter query.
"""

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from megatron.core.mdp.dynamic_cp import (
    DynamicCpGroupSpec,
    GlobalSampleId,
    GlobalVisionItemId,
    dynamic_cp_group_sizes,
    is_power_of_two,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError

DECODER_DYNAMIC_PLAN_SCHEMA_VERSION = 1
ENCODER_DYNAMIC_PLAN_SCHEMA_VERSION = 1

_DECODER_DIGEST_DOMAIN = b"megatron.mdp.dynamic-cp.decoder-plan"
_ENCODER_DIGEST_DOMAIN = b"megatron.mdp.dynamic-cp.encoder-plan"
_INT64_MAX = 2**63 - 1


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_integer(name: str, value: Any, *, minimum: int = 0, positive: bool = False) -> int:
    lower_bound = 1 if positive else minimum
    if not _is_integer(value) or value < lower_bound:
        qualifier = "a positive integer" if positive else f"an integer >= {minimum}"
        raise MdpPlanError(f"MDP: {name}={value!r} violates: {name} is {qualifier}.")
    if value > _INT64_MAX:
        raise MdpPlanError(f"MDP: {name}={value!r} violates: {name} fits in signed int64.")
    return value


def _require_sample_id(name: str, value: Any) -> GlobalSampleId:
    if not isinstance(value, GlobalSampleId):
        raise MdpPlanError(f"MDP: {name}={value!r} violates: a GlobalSampleId.")
    _require_integer(f"{name}.source_dp_lane", value.source_dp_lane)
    _require_integer(f"{name}.local_sample_order", value.local_sample_order)
    return value


def _require_item_id(name: str, value: Any) -> GlobalVisionItemId:
    if not isinstance(value, GlobalVisionItemId):
        raise MdpPlanError(f"MDP: {name}={value!r} violates: a GlobalVisionItemId.")
    _require_integer(f"{name}.source_dp_lane", value.source_dp_lane)
    _require_integer(f"{name}.local_item_id", value.local_item_id)
    return value


def _require_sequence(name: str, value: Any) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MdpPlanError(f"MDP: {name}={value!r} violates: {name} is an ordered sequence.")
    return value


def _digest_ints(hasher: Any, *values: int) -> None:
    for value in values:
        _require_integer("digest field", value)
    hasher.update(struct.pack(f"<{len(values)}q", *values))


def _new_digest(domain: bytes, schema_version: int) -> Any:
    """Start a length-framed, domain-separated, fixed-width plan digest."""
    hasher = hashlib.blake2b(digest_size=16)
    _digest_ints(hasher, len(domain), schema_version)
    hasher.update(domain)
    return hasher


@dataclass(frozen=True)
class EncoderVisionItemMetadata:
    """Stable item-to-sample catalog entry shared by both pure plans."""

    item_id: GlobalVisionItemId
    sample_id: GlobalSampleId
    image_ordinal: int


@dataclass(frozen=True)
class DecoderSampleMetadata:
    """Metadata gathered for one decoder sample; token payload stays elsewhere."""

    sample_id: GlobalSampleId
    valid_seqlen: int
    padded_seqlen: int
    vision_items: tuple[EncoderVisionItemMetadata, ...]


@dataclass(frozen=True)
class DecoderCpAssignment:
    """One logical decoder pack and the ordered endpoint ranks executing it."""

    sample_ids: tuple[GlobalSampleId, ...]
    endpoint_ranks: tuple[int, ...]

    @property
    def local_cp_size(self) -> int:
        """Effective decoder CP size for this logical assignment."""
        return len(self.endpoint_ranks)


@dataclass(frozen=True)
class DecoderMicrobatchPlan:
    """All disjoint Dynamic-CP assignments executed in one decoder microbatch."""

    microbatch_index: int
    assignments: tuple[DecoderCpAssignment, ...]


@dataclass(frozen=True)
class DecoderDynamicPlan:
    """Deterministic decoder scheduling result for one captured MDP window."""

    samples: tuple[DecoderSampleMetadata, ...]
    decoder_ranks: tuple[int, ...]
    max_seqlen_per_rank: int
    minimum_cp_size: int
    microbatches: tuple[DecoderMicrobatchPlan, ...]
    digest: bytes

    @property
    def effective_num_microbatches(self) -> int:
        """Number of decoder microbatches produced by dynamic packing."""
        return len(self.microbatches)

    @property
    def sample_ids(self) -> tuple[GlobalSampleId, ...]:
        """Canonical source sample IDs covered by the plan."""
        return tuple(sample.sample_id for sample in self.samples)

    @property
    def items(self) -> tuple[EncoderVisionItemMetadata, ...]:
        """Canonical vision-item catalog derived from source samples."""
        return tuple(item for sample in self.samples for item in sample.vision_items)

    @property
    def item_ids(self) -> tuple[GlobalVisionItemId, ...]:
        """Canonical vision-item IDs covered by the plan."""
        return tuple(item.item_id for item in self.items)


class DecoderPackingSolver(Protocol):
    """The pure call surface exposed by native ``next_hdp_group_packing_aware``."""

    def __call__(
        self,
        sample_seqlens: list[tuple[int, int]],
        total_gpus: int,
        max_seq_len_per_rank: int,
        min_cp_size: int = 1,
    ) -> tuple[Any, Any, Any, Any]:
        """Return native per-rank lengths, leftovers, costs, and sample IDs."""
        ...


@dataclass(frozen=True)
class EncoderWorkUnit:
    """One ordered item pack scheduled as an indivisible encoder execution."""

    item_ids: tuple[GlobalVisionItemId, ...]


@dataclass(frozen=True)
class EncoderWorkEstimate:
    """Adapter-reported rows and integer scheduling cost for one whole pack."""

    effective_rows_per_rank: int
    cost_units: int


class EncoderWorkloadQuery(Protocol):
    """Model-adapter seam for candidate-size encoder geometry and cost."""

    def __call__(
        self, item_ids: tuple[GlobalVisionItemId, ...], group_size: int
    ) -> EncoderWorkEstimate:
        """Estimate one ordered work unit at one candidate encoder CP size."""
        ...


@dataclass(frozen=True)
class EncoderExecution:
    """One encoder work unit assigned to a nested subgroup by logical rank slots."""

    group_size: int
    group_index: int
    rank_slots: tuple[int, ...]
    item_ids: tuple[GlobalVisionItemId, ...]
    effective_rows_per_rank: int
    cost_units: int


@dataclass(frozen=True)
class EncoderExecutionWave:
    """Concurrent encoder executions whose rank slots are pairwise disjoint."""

    wave_index: int
    executions: tuple[EncoderExecution, ...]


@dataclass(frozen=True)
class EncoderDynamicPlan:
    """Deterministic encoder execution waves for one maximum-ECP source pool."""

    source_samples: tuple[DecoderSampleMetadata, ...]
    pool_ranks: tuple[int, ...]
    max_seqlen_per_rank: int
    waves: tuple[EncoderExecutionWave, ...]
    digest: bytes

    @property
    def sample_ids(self) -> tuple[GlobalSampleId, ...]:
        """Canonical source sample IDs joined to encoder work."""
        return tuple(sample.sample_id for sample in self.source_samples)

    @property
    def items(self) -> tuple[EncoderVisionItemMetadata, ...]:
        """Canonical vision-item catalog derived from source samples."""
        return tuple(item for sample in self.source_samples for item in sample.vision_items)

    @property
    def item_ids(self) -> tuple[GlobalVisionItemId, ...]:
        """Canonical vision-item IDs executed by the encoder plan."""
        return tuple(item.item_id for item in self.items)


def validate_dynamic_plan_catalog(
    decoder_plan: DecoderDynamicPlan, encoder_plan: EncoderDynamicPlan
) -> None:
    """Require decoder and encoder plans to use one exact canonical source catalog."""
    if not isinstance(decoder_plan, DecoderDynamicPlan) or not isinstance(
        encoder_plan, EncoderDynamicPlan
    ):
        raise MdpPlanError("MDP: dynamic plan catalog join requires decoder/encoder plans.")
    validate_decoder_dynamic_plan(decoder_plan)
    validate_encoder_dynamic_plan(encoder_plan)
    if decoder_plan.samples != encoder_plan.source_samples:
        raise MdpPlanError(
            "MDP: dynamic decoder/encoder plans require the exact same canonical source "
            "sample and vision-item catalog."
        )


def _canonical_decoder_samples(
    samples: Sequence[DecoderSampleMetadata],
) -> tuple[DecoderSampleMetadata, ...]:
    _require_sequence("samples", samples)
    canonical = []
    seen_samples = set()
    seen_items = set()
    for sample in samples:
        if not isinstance(sample, DecoderSampleMetadata):
            raise MdpPlanError("MDP: decoder samples contain only DecoderSampleMetadata.")
        sample_id = _require_sample_id("sample_id", sample.sample_id)
        if sample_id in seen_samples:
            raise MdpPlanError("MDP: decoder sample IDs are unique within one plan.")
        seen_samples.add(sample_id)
        valid = _require_integer("valid_seqlen", sample.valid_seqlen, positive=True)
        padded = _require_integer("padded_seqlen", sample.padded_seqlen, positive=True)
        if valid > padded:
            raise MdpPlanError("MDP: decoder sample violates: valid_seqlen <= padded_seqlen.")
        if not isinstance(sample.vision_items, tuple):
            raise MdpPlanError("MDP: decoder vision_items is an immutable ordered tuple.")
        for expected_ordinal, item in enumerate(sample.vision_items):
            if not isinstance(item, EncoderVisionItemMetadata):
                raise MdpPlanError(
                    "MDP: decoder vision_items contain only EncoderVisionItemMetadata."
                )
            item_id = _require_item_id("item_id", item.item_id)
            _require_sample_id("item.sample_id", item.sample_id)
            ordinal = _require_integer("image_ordinal", item.image_ordinal)
            if item.sample_id != sample_id or item_id.source_dp_lane != sample_id.source_dp_lane:
                raise MdpPlanError(
                    f"MDP: vision item {item_id} violates: item metadata names its owning sample."
                )
            if ordinal != expected_ordinal:
                raise MdpPlanError(
                    f"MDP: sample {sample_id} violates: vision items follow image_ordinal order "
                    "0..N-1."
                )
            if item_id in seen_items:
                raise MdpPlanError("MDP: decoder item IDs are unique within one plan.")
            seen_items.add(item_id)
        canonical.append(sample)
    if not canonical:
        raise MdpPlanError("MDP: decoder plan requires at least one source sample.")
    return tuple(sorted(canonical, key=lambda sample: sample.sample_id))


def _canonical_decoder_ranks(decoder_ranks: Sequence[int]) -> tuple[int, ...]:
    _require_sequence("decoder_ranks", decoder_ranks)
    ranks = tuple(
        _require_integer(f"decoder_ranks[{index}]", rank)
        for index, rank in enumerate(decoder_ranks)
    )
    if len(ranks) <= 1 or not is_power_of_two(len(ranks)):
        raise MdpPlanError(
            "MDP: decoder_ranks violates: Dynamic-CP has a power-of-two size greater than one."
        )
    if len(set(ranks)) != len(ranks):
        raise MdpPlanError("MDP: decoder_ranks violates: ranks are unique and ordered.")
    return ranks


def _solver_integer(name: str, value: Any) -> int:
    try:
        return _require_integer(name, value)
    except MdpPlanError as error:
        raise MdpPlanError(
            f"MDP decoder solver output violates: {name} is a non-negative signed-int64 " "integer."
        ) from error


def _solver_rows(name: str, rows: Any, expected_count: int) -> tuple[tuple[int, ...], ...]:
    try:
        rows = _require_sequence(name, rows)
    except MdpPlanError as error:
        raise MdpPlanError(f"MDP decoder solver output violates: {name} are rank rows.") from error
    if len(rows) != expected_count:
        raise MdpPlanError(
            f"MDP decoder solver output violates: {name} has {expected_count} rank rows."
        )
    converted = []
    for rank_slot, row in enumerate(rows):
        try:
            row = _require_sequence(f"{name}[{rank_slot}]", row)
        except MdpPlanError as error:
            raise MdpPlanError(
                f"MDP decoder solver output violates: {name}[{rank_slot}] is an ordered row."
            ) from error
        converted.append(
            tuple(_solver_integer(f"{name}[{rank_slot}] value", value) for value in row)
        )
    return tuple(converted)


def _parse_solver_assignments(
    rank_lengths: tuple[tuple[int, ...], ...],
    rank_sample_ids: tuple[tuple[int, ...], ...],
    *,
    pending: dict[int, int],
    decoder_ranks: tuple[int, ...],
    max_seqlen_per_rank: int,
    minimum_cp_size: int,
    dense_to_global: dict[int, GlobalSampleId],
) -> tuple[tuple[DecoderCpAssignment, ...], set[int]]:
    if all(not sample_ids for sample_ids in rank_sample_ids):
        raise MdpPlanError(
            "MDP decoder solver output violates: each call makes progress; all rank rows "
            "are empty."
        )
    for rank_slot, (sample_ids, lengths) in enumerate(zip(rank_sample_ids, rank_lengths)):
        if not sample_ids:
            raise MdpPlanError(
                f"MDP decoder solver output violates: empty rank {rank_slot} is not a "
                "complete native DCP microbatch."
            )
        if len(sample_ids) != len(lengths):
            raise MdpPlanError(
                "MDP decoder solver output violates: sample ID and length rows have equal size."
            )
        if len(set(sample_ids)) != len(sample_ids):
            raise MdpPlanError(
                "MDP decoder solver output violates: duplicate sample ID inside one rank row."
            )
        for sample_id in sample_ids:
            if sample_id not in pending:
                raise MdpPlanError(
                    f"MDP decoder solver output violates: unknown sample ID {sample_id}."
                )

    assignments = []
    logically_assigned = set()
    rank_slot = 0
    total_ranks = len(decoder_ranks)
    while rank_slot < total_ranks:
        sample_ids = rank_sample_ids[rank_slot]
        group_end = rank_slot + 1
        while group_end < total_ranks and rank_sample_ids[group_end] == sample_ids:
            group_end += 1
        group_size = group_end - rank_slot
        if not is_power_of_two(group_size) or group_size < minimum_cp_size:
            raise MdpPlanError(
                f"MDP decoder solver output violates: physical replica group size "
                f"{group_size} is a scheduled power of two."
            )
        if rank_slot % group_size:
            raise MdpPlanError(
                f"MDP decoder solver output violates: size-{group_size} subgroup at rank "
                f"slot {rank_slot} is aligned in the ordered decoder pool."
            )
        lengths = rank_lengths[rank_slot]
        if any(rank_lengths[member] != lengths for member in range(rank_slot, group_end)):
            raise MdpPlanError(
                "MDP decoder solver output violates: subgroup members have identical ordered "
                "IDs and lengths."
            )
        for sample_id, length in zip(sample_ids, lengths):
            if length != pending[sample_id]:
                raise MdpPlanError(
                    f"MDP decoder solver output violates: length {length} for sample "
                    f"{sample_id} equals its padded length {pending[sample_id]}."
                )
        packed_rows = sum(lengths)
        _require_integer("solver packed length sum", packed_rows)
        group_capacity = group_size * max_seqlen_per_rank
        _require_integer("solver subgroup capacity", group_capacity)
        if packed_rows > group_capacity:
            raise MdpPlanError(
                f"MDP decoder solver output violates: packed rows {packed_rows} fit subgroup "
                f"capacity {group_capacity}."
            )
        duplicate = logically_assigned.intersection(sample_ids)
        if duplicate:
            raise MdpPlanError(
                f"MDP decoder solver output violates: logical sample IDs {sorted(duplicate)} "
                "appear in two disjoint subgroups."
            )
        logically_assigned.update(sample_ids)
        assignments.append(
            DecoderCpAssignment(
                sample_ids=tuple(dense_to_global[sample_id] for sample_id in sample_ids),
                endpoint_ranks=decoder_ranks[rank_slot:group_end],
            )
        )
        rank_slot = group_end
    return tuple(assignments), logically_assigned


def _parse_solver_leftovers(
    leftovers: Any, *, pending: dict[int, int]
) -> tuple[list[tuple[int, int]], set[int]]:
    try:
        leftovers = _require_sequence("solver leftovers", leftovers)
    except MdpPlanError as error:
        raise MdpPlanError("MDP decoder solver output violates: leftovers are pairs.") from error
    parsed = []
    seen = set()
    for entry in leftovers:
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 2:
            raise MdpPlanError("MDP decoder solver output violates: leftovers are ID/length pairs.")
        sample_id = _solver_integer("leftover sample ID", entry[0])
        length = _solver_integer("leftover padded length", entry[1])
        if sample_id not in pending:
            raise MdpPlanError(
                f"MDP decoder solver output violates: unknown leftover sample ID {sample_id}."
            )
        if length != pending[sample_id]:
            raise MdpPlanError(
                "MDP decoder solver output violates: leftover length equals its padded length."
            )
        if sample_id in seen:
            raise MdpPlanError("MDP decoder solver output violates: leftover IDs are unique.")
        seen.add(sample_id)
        parsed.append((sample_id, length))
    return parsed, seen


def build_decoder_dynamic_plan(
    samples: Sequence[DecoderSampleMetadata],
    *,
    decoder_ranks: Sequence[int],
    max_seqlen_per_rank: int,
    minimum_cp_size: int,
    solver: DecoderPackingSolver,
) -> DecoderDynamicPlan:
    """Schedule canonical sample metadata through an injected native DCP solver."""
    canonical_samples = _canonical_decoder_samples(samples)
    ranks = _canonical_decoder_ranks(decoder_ranks)
    capacity = _require_integer("max_seqlen_per_rank", max_seqlen_per_rank, positive=True)
    try:
        dynamic_cp_group_sizes(minimum_cp_size, len(ranks))
    except MdpConfigurationError as error:
        raise MdpPlanError("MDP: decoder minimum CP size is valid for decoder_ranks.") from error
    total_capacity = len(ranks) * capacity
    _require_integer("decoder total capacity", total_capacity, positive=True)
    if not callable(solver):
        raise MdpPlanError("MDP: decoder solver is callable.")

    dense_to_global = {
        dense_id: sample.sample_id for dense_id, sample in enumerate(canonical_samples)
    }
    pending = [
        (dense_id, sample.padded_seqlen) for dense_id, sample in enumerate(canonical_samples)
    ]
    for sample_id, padded_length in pending:
        if padded_length > total_capacity:
            raise MdpPlanError(
                f"MDP: sample {dense_to_global[sample_id]} padded length {padded_length} "
                f"exceeds decoder capacity {total_capacity}."
            )

    microbatches = []
    all_assigned = set()
    while pending:
        pending_map = dict(pending)
        try:
            result = solver(
                list(pending),
                total_gpus=len(ranks),
                max_seq_len_per_rank=capacity,
                min_cp_size=minimum_cp_size,
            )
        except MdpPlanError:
            raise
        except Exception as error:
            raise MdpPlanError("MDP decoder solver call failed.") from error
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise MdpPlanError("MDP decoder solver output violates: a four-field result.")
        raw_lengths, raw_leftovers, _, raw_sample_ids = result
        rank_lengths = _solver_rows("length rows", raw_lengths, len(ranks))
        rank_sample_ids = _solver_rows("sample ID rows", raw_sample_ids, len(ranks))
        assignments, assigned = _parse_solver_assignments(
            rank_lengths,
            rank_sample_ids,
            pending=pending_map,
            decoder_ranks=ranks,
            max_seqlen_per_rank=capacity,
            minimum_cp_size=minimum_cp_size,
            dense_to_global=dense_to_global,
        )
        leftovers, leftover_ids = _parse_solver_leftovers(raw_leftovers, pending=pending_map)
        overlap = assigned.intersection(leftover_ids)
        if overlap:
            raise MdpPlanError(
                f"MDP decoder solver output violates: assigned/leftover overlap {sorted(overlap)}."
            )
        if assigned.union(leftover_ids) != set(pending_map):
            raise MdpPlanError("MDP decoder solver output violates: exact pending coverage.")
        if not assigned:
            raise MdpPlanError("MDP decoder solver output violates: each call makes progress.")
        duplicate = all_assigned.intersection(assigned)
        if duplicate:
            raise MdpPlanError(
                f"MDP decoder solver output violates: samples assigned once ({sorted(duplicate)})."
            )
        all_assigned.update(assigned)
        microbatches.append(DecoderMicrobatchPlan(len(microbatches), assignments))
        pending = leftovers

    expected = set(range(len(canonical_samples)))
    if all_assigned != expected:
        raise MdpPlanError("MDP: decoder plan violates: every sample is scheduled exactly once.")
    frozen_microbatches = tuple(microbatches)
    digest = _decoder_digest(
        canonical_samples, ranks, capacity, minimum_cp_size, frozen_microbatches
    )
    plan = DecoderDynamicPlan(
        samples=canonical_samples,
        decoder_ranks=ranks,
        max_seqlen_per_rank=capacity,
        minimum_cp_size=minimum_cp_size,
        microbatches=frozen_microbatches,
        digest=digest,
    )
    return validate_decoder_dynamic_plan(plan)


def _decoder_digest(
    samples: tuple[DecoderSampleMetadata, ...],
    ranks: tuple[int, ...],
    capacity: int,
    minimum_cp_size: int,
    microbatches: tuple[DecoderMicrobatchPlan, ...],
) -> bytes:
    hasher = _new_digest(_DECODER_DIGEST_DOMAIN, DECODER_DYNAMIC_PLAN_SCHEMA_VERSION)
    _digest_ints(hasher, capacity, minimum_cp_size, len(ranks), *ranks, len(samples))
    for sample in samples:
        _digest_ints(
            hasher,
            *sample.sample_id.to_wire_tuple(),
            sample.valid_seqlen,
            sample.padded_seqlen,
            len(sample.vision_items),
        )
        for item in sample.vision_items:
            _digest_ints(
                hasher,
                *item.item_id.to_wire_tuple(),
                *item.sample_id.to_wire_tuple(),
                item.image_ordinal,
            )
    _digest_ints(hasher, len(microbatches))
    for microbatch in microbatches:
        _digest_ints(hasher, microbatch.microbatch_index, len(microbatch.assignments))
        for assignment in microbatch.assignments:
            _digest_ints(hasher, len(assignment.sample_ids))
            for sample_id in assignment.sample_ids:
                _digest_ints(hasher, *sample_id.to_wire_tuple())
            _digest_ints(
                hasher,
                assignment.local_cp_size,
                len(assignment.endpoint_ranks),
                *assignment.endpoint_ranks,
            )
    return hasher.digest()


def validate_decoder_dynamic_plan(plan: DecoderDynamicPlan) -> DecoderDynamicPlan:
    """Validate every invariant and digest owned by the decoder plan builder."""
    if not isinstance(plan, DecoderDynamicPlan):
        raise MdpPlanError("MDP: decoder dynamic plan is a DecoderDynamicPlan.")
    if not isinstance(plan.samples, tuple):
        raise MdpPlanError("MDP: decoder dynamic plan samples are an immutable tuple.")
    samples = _canonical_decoder_samples(plan.samples)
    if samples != plan.samples:
        raise MdpPlanError("MDP: decoder dynamic plan samples are in canonical ID order.")
    if not isinstance(plan.decoder_ranks, tuple):
        raise MdpPlanError("MDP: decoder dynamic plan ranks are an immutable tuple.")
    ranks = _canonical_decoder_ranks(plan.decoder_ranks)
    capacity = _require_integer(
        "decoder plan max_seqlen_per_rank", plan.max_seqlen_per_rank, positive=True
    )
    minimum_cp_size = _require_integer(
        "decoder plan minimum_cp_size", plan.minimum_cp_size, positive=True
    )
    try:
        dynamic_cp_group_sizes(minimum_cp_size, len(ranks))
    except MdpConfigurationError as error:
        raise MdpPlanError(
            "MDP: decoder plan minimum CP size is valid for decoder_ranks."
        ) from error
    total_capacity = _require_integer(
        "decoder plan total capacity", len(ranks) * capacity, positive=True
    )
    if any(sample.padded_seqlen > total_capacity for sample in samples):
        raise MdpPlanError("MDP: every decoder plan sample fits the maximum decoder pool.")
    if not isinstance(plan.microbatches, tuple) or not plan.microbatches:
        raise MdpPlanError("MDP: decoder dynamic plan has immutable non-empty microbatches.")

    sample_by_id = {sample.sample_id: sample for sample in samples}
    scheduled = set()
    for expected_index, microbatch in enumerate(plan.microbatches):
        if not isinstance(microbatch, DecoderMicrobatchPlan):
            raise MdpPlanError("MDP: decoder plan contains DecoderMicrobatchPlan carriers.")
        index = _require_integer("decoder plan microbatch index", microbatch.microbatch_index)
        if index != expected_index:
            raise MdpPlanError("MDP: decoder plan microbatch indices are contiguous in order.")
        if not isinstance(microbatch.assignments, tuple) or not microbatch.assignments:
            raise MdpPlanError("MDP: decoder microbatch has immutable non-empty assignments.")
        rank_slot = 0
        for assignment in microbatch.assignments:
            if not isinstance(assignment, DecoderCpAssignment):
                raise MdpPlanError("MDP: decoder plan contains DecoderCpAssignment carriers.")
            if not isinstance(assignment.endpoint_ranks, tuple):
                raise MdpPlanError("MDP: decoder assignment endpoints are an immutable tuple.")
            endpoints = tuple(
                _require_integer(f"decoder assignment endpoint rank {slot}", rank)
                for slot, rank in enumerate(assignment.endpoint_ranks)
            )
            group_size = len(endpoints)
            if (
                not is_power_of_two(group_size)
                or group_size < minimum_cp_size
                or rank_slot % group_size
            ):
                raise MdpPlanError(
                    "MDP: decoder assignment endpoint size is an aligned scheduled power of two."
                )
            if endpoints != ranks[rank_slot : rank_slot + group_size]:
                raise MdpPlanError(
                    "MDP: decoder assignments are contiguous slices of the ordered rank pool."
                )
            rank_slot += group_size
            if not isinstance(assignment.sample_ids, tuple) or not assignment.sample_ids:
                raise MdpPlanError("MDP: decoder assignment has immutable non-empty sample IDs.")
            assignment_ids = tuple(
                _require_sample_id("decoder assignment sample", sample_id)
                for sample_id in assignment.sample_ids
            )
            if len(set(assignment_ids)) != len(assignment_ids):
                raise MdpPlanError("MDP: decoder assignment sample IDs are unique.")
            if any(sample_id not in sample_by_id for sample_id in assignment_ids):
                raise MdpPlanError("MDP: decoder assignment samples belong to the plan catalog.")
            if scheduled.intersection(assignment_ids):
                raise MdpPlanError("MDP: decoder plan schedules every sample exactly once.")
            scheduled.update(assignment_ids)
            packed_rows = 0
            for sample_id in assignment_ids:
                packed_rows = _require_integer(
                    "decoder assignment padded row sum",
                    packed_rows + sample_by_id[sample_id].padded_seqlen,
                )
            group_capacity = _require_integer(
                "decoder assignment capacity", group_size * capacity, positive=True
            )
            if packed_rows > group_capacity:
                raise MdpPlanError("MDP: decoder assignment packed rows fit subgroup capacity.")
        if rank_slot != len(ranks):
            raise MdpPlanError(
                "MDP: decoder microbatch assignments partition the ordered decoder ranks."
            )
    if scheduled != set(sample_by_id):
        raise MdpPlanError("MDP: decoder plan schedules every source sample exactly once.")
    if not isinstance(plan.digest, bytes) or len(plan.digest) != 16:
        raise MdpPlanError("MDP: decoder dynamic plan digest is exactly 16 bytes.")
    expected_digest = _decoder_digest(samples, ranks, capacity, minimum_cp_size, plan.microbatches)
    if plan.digest != expected_digest:
        raise MdpPlanError("MDP: decoder dynamic plan digest matches its canonical contents.")
    return plan


def _canonical_encoder_groups(
    group_specs: Sequence[DynamicCpGroupSpec],
) -> tuple[tuple[DynamicCpGroupSpec, ...], tuple[int, ...]]:
    _require_sequence("group_specs", group_specs)
    if not group_specs:
        raise MdpPlanError("MDP: encoder Dynamic-CP requires nested group specs.")
    if any(not isinstance(spec, DynamicCpGroupSpec) for spec in group_specs):
        raise MdpPlanError("MDP: encoder group_specs contain only DynamicCpGroupSpec.")
    specs = tuple(sorted(group_specs, key=lambda spec: (spec.group_size, spec.group_index)))
    keys = tuple((spec.group_size, spec.group_index) for spec in specs)
    if len(set(keys)) != len(keys):
        raise MdpPlanError("MDP: encoder group specs have unique size/index keys.")
    maximum_size = max(spec.group_size for spec in specs)
    maximum_specs = tuple(spec for spec in specs if spec.group_size == maximum_size)
    if len(maximum_specs) != 1 or maximum_specs[0].group_index != 0:
        raise MdpPlanError("MDP: encoder group specs contain one maximum-size pool.")
    pool_ranks = maximum_specs[0].ranks
    for index, rank in enumerate(pool_ranks):
        _require_integer(f"pool_ranks[{index}]", rank)
    sizes = tuple(sorted({spec.group_size for spec in specs}))
    try:
        expected_sizes = dynamic_cp_group_sizes(sizes[0], maximum_size)
    except MdpConfigurationError as error:
        raise MdpPlanError("MDP: encoder group spec sizes form a Dynamic-CP range.") from error
    if sizes != expected_sizes:
        raise MdpPlanError("MDP: encoder group specs contain every scheduled size.")
    for size in sizes:
        size_specs = tuple(spec for spec in specs if spec.group_size == size)
        expected_count = maximum_size // size
        if tuple(spec.group_index for spec in size_specs) != tuple(range(expected_count)):
            raise MdpPlanError("MDP: encoder group indices partition the maximum pool.")
        for spec in size_specs:
            start = spec.group_index * size
            if spec.ranks != pool_ranks[start : start + size]:
                raise MdpPlanError("MDP: encoder group ranks follow ordered nested pool slices.")
    return specs, pool_ranks


def _canonical_work_units(
    work_units: Sequence[EncoderWorkUnit], *, items: tuple[EncoderVisionItemMetadata, ...]
) -> tuple[EncoderWorkUnit, ...]:
    _require_sequence("work_units", work_units)
    catalog_positions = {item.item_id: index for index, item in enumerate(items)}
    expected = set(catalog_positions)
    covered = set()
    canonical = []
    for unit in work_units:
        if not isinstance(unit, EncoderWorkUnit) or not isinstance(unit.item_ids, tuple):
            raise MdpPlanError("MDP: encoder work units contain immutable EncoderWorkUnit packs.")
        if not unit.item_ids:
            raise MdpPlanError("MDP: encoder work unit contains at least one item.")
        ids = tuple(
            _require_item_id(f"work unit item {index}", item_id)
            for index, item_id in enumerate(unit.item_ids)
        )
        if len(set(ids)) != len(ids):
            raise MdpPlanError("MDP: encoder work-unit item IDs are unique.")
        if any(item_id not in expected for item_id in ids):
            raise MdpPlanError("MDP: encoder work-unit items belong to the item catalog.")
        positions = tuple(catalog_positions[item_id] for item_id in ids)
        if positions != tuple(sorted(positions)):
            raise MdpPlanError("MDP: encoder work-unit items follow catalog order.")
        if covered.intersection(ids):
            raise MdpPlanError("MDP: encoder work units cover catalog items exactly once.")
        covered.update(ids)
        canonical.append(unit)
    if covered != expected:
        raise MdpPlanError("MDP: encoder work units cover catalog items exactly once.")
    canonical.sort(key=lambda unit: tuple(catalog_positions[item_id] for item_id in unit.item_ids))
    return tuple(canonical)


@dataclass(frozen=True)
class _PreparedEncoderWork:
    unit: EncoderWorkUnit
    group_size: int
    effective_rows_per_rank: int
    cost_units: int


def _prepare_encoder_work(
    work_units: tuple[EncoderWorkUnit, ...],
    *,
    sizes: tuple[int, ...],
    capacity: int,
    workload_query: EncoderWorkloadQuery,
) -> tuple[_PreparedEncoderWork, ...]:
    prepared = []
    for unit in work_units:
        selected = None
        for group_size in sizes:
            try:
                estimate = workload_query(unit.item_ids, group_size)
            except MdpPlanError:
                raise
            except Exception as error:
                raise MdpPlanError(
                    f"MDP: encoder adapter workload query failed for {unit.item_ids} at "
                    f"group_size={group_size}."
                ) from error
            if not isinstance(estimate, EncoderWorkEstimate):
                raise MdpPlanError("MDP: encoder adapter query returns EncoderWorkEstimate.")
            try:
                rows = _require_integer(
                    "adapter effective_rows_per_rank",
                    estimate.effective_rows_per_rank,
                    positive=True,
                )
                cost = _require_integer("adapter cost_units", estimate.cost_units)
            except MdpPlanError as error:
                raise MdpPlanError(
                    "MDP: encoder adapter estimate uses positive integer rows and "
                    "non-negative integer cost in signed int64."
                ) from error
            if rows <= capacity:
                selected = _PreparedEncoderWork(unit, group_size, rows, cost)
                break
        if selected is None:
            raise MdpPlanError(
                f"MDP: encoder work unit {unit.item_ids} exceeds per-rank capacity "
                f"{capacity} at every candidate group size {sizes}."
            )
        prepared.append(selected)
    return tuple(prepared)


def _encoder_waves(
    prepared: tuple[_PreparedEncoderWork, ...], *, specs: tuple[DynamicCpGroupSpec, ...]
) -> tuple[EncoderExecutionWave, ...]:
    """Place LPT work by projected physical slot load, then stable group index."""
    specs_by_size = {
        size: tuple(spec for spec in specs if spec.group_size == size)
        for size in sorted({spec.group_size for spec in specs})
    }
    maximum_size = max(spec.group_size for spec in specs)
    slot_loads = [0] * maximum_size
    mutable_waves: list[list[tuple[DynamicCpGroupSpec, _PreparedEncoderWork]]] = []

    def placement_key(spec: DynamicCpGroupSpec, work: _PreparedEncoderWork) -> tuple:
        """Rank a subgroup by projected slot loads, then stable group index."""
        start = spec.group_index * spec.group_size
        projected = tuple(
            sorted(
                (
                    slot_loads[slot] + work.cost_units
                    for slot in range(start, start + spec.group_size)
                ),
                reverse=True,
            )
        )
        return projected, spec.group_index, work.unit.item_ids

    ordered = sorted(prepared, key=lambda work: (-work.cost_units, work.unit.item_ids))
    for work in ordered:
        selected_wave = None
        selected_spec = None
        for wave_index, wave in enumerate(mutable_waves):
            used_slots = {
                slot
                for assigned_spec, _ in wave
                for slot in range(
                    assigned_spec.group_index * assigned_spec.group_size,
                    (assigned_spec.group_index + 1) * assigned_spec.group_size,
                )
            }
            available = []
            for spec in specs_by_size[work.group_size]:
                start = spec.group_index * spec.group_size
                slots = set(range(start, start + spec.group_size))
                if used_slots.isdisjoint(slots):
                    available.append(spec)
            if available:
                selected_wave = wave_index
                selected_spec = min(available, key=lambda spec: placement_key(spec, work))
                break
        if selected_spec is None:
            selected_wave = len(mutable_waves)
            mutable_waves.append([])
            selected_spec = min(
                specs_by_size[work.group_size], key=lambda spec: placement_key(spec, work)
            )
        start = selected_spec.group_index * selected_spec.group_size
        for slot in range(start, start + selected_spec.group_size):
            new_load = slot_loads[slot] + work.cost_units
            _require_integer("encoder rank-slot cumulative cost", new_load)
            slot_loads[slot] = new_load
        mutable_waves[selected_wave].append((selected_spec, work))

    waves = []
    for wave_index, wave in enumerate(mutable_waves):
        executions = []
        for spec, work in wave:
            start = spec.group_index * spec.group_size
            executions.append(
                EncoderExecution(
                    group_size=spec.group_size,
                    group_index=spec.group_index,
                    rank_slots=tuple(range(start, start + spec.group_size)),
                    item_ids=work.unit.item_ids,
                    effective_rows_per_rank=work.effective_rows_per_rank,
                    cost_units=work.cost_units,
                )
            )
        executions.sort(
            key=lambda execution: (
                execution.rank_slots[0],
                execution.group_size,
                execution.item_ids,
            )
        )
        waves.append(EncoderExecutionWave(wave_index, tuple(executions)))
    return tuple(waves)


def build_encoder_dynamic_plan(
    source_samples: Sequence[DecoderSampleMetadata],
    work_units: Sequence[EncoderWorkUnit],
    *,
    group_specs: Sequence[DynamicCpGroupSpec],
    max_seqlen_per_rank: int,
    workload_query: EncoderWorkloadQuery,
) -> EncoderDynamicPlan:
    """Join source samples to encoder work and form deterministic execution waves."""
    canonical_samples = _canonical_decoder_samples(source_samples)
    canonical_items = tuple(item for sample in canonical_samples for item in sample.vision_items)
    specs, pool_ranks = _canonical_encoder_groups(group_specs)
    capacity = _require_integer("max_seqlen_per_rank", max_seqlen_per_rank, positive=True)
    if not callable(workload_query):
        raise MdpPlanError("MDP: encoder workload_query is callable.")
    canonical_units = _canonical_work_units(work_units, items=canonical_items)
    sizes = tuple(sorted({spec.group_size for spec in specs}))
    prepared = _prepare_encoder_work(
        canonical_units, sizes=sizes, capacity=capacity, workload_query=workload_query
    )
    waves = _encoder_waves(prepared, specs=specs)
    digest = _encoder_digest(canonical_samples, pool_ranks, capacity, waves)
    plan = EncoderDynamicPlan(canonical_samples, pool_ranks, capacity, waves, digest)
    return validate_encoder_dynamic_plan(plan)


def _encoder_digest(
    source_samples: tuple[DecoderSampleMetadata, ...],
    pool_ranks: tuple[int, ...],
    capacity: int,
    waves: tuple[EncoderExecutionWave, ...],
) -> bytes:
    hasher = _new_digest(_ENCODER_DIGEST_DOMAIN, ENCODER_DYNAMIC_PLAN_SCHEMA_VERSION)
    _digest_ints(hasher, capacity, len(pool_ranks), *pool_ranks, len(source_samples))
    for sample in source_samples:
        _digest_ints(
            hasher,
            *sample.sample_id.to_wire_tuple(),
            sample.valid_seqlen,
            sample.padded_seqlen,
            len(sample.vision_items),
        )
        for item in sample.vision_items:
            _digest_ints(
                hasher,
                *item.item_id.to_wire_tuple(),
                *item.sample_id.to_wire_tuple(),
                item.image_ordinal,
            )
    _digest_ints(hasher, len(waves))
    for wave in waves:
        _digest_ints(hasher, wave.wave_index, len(wave.executions))
        for execution in wave.executions:
            _digest_ints(
                hasher,
                execution.group_size,
                execution.group_index,
                len(execution.rank_slots),
                *execution.rank_slots,
                execution.effective_rows_per_rank,
                execution.cost_units,
                len(execution.item_ids),
            )
            for item_id in execution.item_ids:
                _digest_ints(hasher, *item_id.to_wire_tuple())
    return hasher.digest()


def validate_encoder_dynamic_plan(plan: EncoderDynamicPlan) -> EncoderDynamicPlan:
    """Validate every invariant and digest owned by the encoder plan builder."""
    if not isinstance(plan, EncoderDynamicPlan):
        raise MdpPlanError("MDP: encoder dynamic plan is an EncoderDynamicPlan.")
    if not isinstance(plan.source_samples, tuple):
        raise MdpPlanError("MDP: encoder dynamic plan source samples are an immutable tuple.")
    source_samples = _canonical_decoder_samples(plan.source_samples)
    if source_samples != plan.source_samples:
        raise MdpPlanError("MDP: encoder dynamic plan source samples are in canonical ID order.")
    if not isinstance(plan.pool_ranks, tuple):
        raise MdpPlanError("MDP: encoder dynamic plan pool ranks are an immutable tuple.")
    pool_ranks = tuple(
        _require_integer(f"encoder plan pool rank {index}", rank)
        for index, rank in enumerate(plan.pool_ranks)
    )
    if not is_power_of_two(len(pool_ranks)):
        raise MdpPlanError("MDP: encoder dynamic plan pool has a positive power-of-two size.")
    if len(set(pool_ranks)) != len(pool_ranks):
        raise MdpPlanError("MDP: encoder dynamic plan pool ranks are unique and ordered.")
    capacity = _require_integer(
        "encoder plan max_seqlen_per_rank", plan.max_seqlen_per_rank, positive=True
    )
    if not isinstance(plan.waves, tuple):
        raise MdpPlanError("MDP: encoder dynamic plan waves are an immutable tuple.")

    expected_item_ids = tuple(
        item.item_id for sample in source_samples for item in sample.vision_items
    )
    if bool(expected_item_ids) != bool(plan.waves):
        raise MdpPlanError(
            "MDP: encoder dynamic plan waves are empty exactly for text-only catalogs."
        )
    catalog_positions = {item_id: position for position, item_id in enumerate(expected_item_ids)}
    expected_items = set(expected_item_ids)
    covered = set()
    slot_costs = [0] * len(pool_ranks)

    for expected_wave_index, wave in enumerate(plan.waves):
        if not isinstance(wave, EncoderExecutionWave):
            raise MdpPlanError("MDP: encoder plan contains EncoderExecutionWave carriers.")
        wave_index = _require_integer("encoder plan wave index", wave.wave_index)
        if wave_index != expected_wave_index:
            raise MdpPlanError("MDP: encoder plan wave indices are contiguous in order.")
        if not isinstance(wave.executions, tuple) or not wave.executions:
            raise MdpPlanError("MDP: encoder wave has immutable non-empty executions.")

        used_slots = set()
        execution_keys = []
        for execution in wave.executions:
            if not isinstance(execution, EncoderExecution):
                raise MdpPlanError("MDP: encoder plan contains EncoderExecution carriers.")
            group_size = _require_integer(
                "encoder execution group_size", execution.group_size, positive=True
            )
            if not is_power_of_two(group_size) or group_size > len(pool_ranks):
                raise MdpPlanError(
                    "MDP: encoder execution group size is a scheduled power of two within "
                    "the pool."
                )
            group_index = _require_integer("encoder execution group_index", execution.group_index)
            if group_index >= len(pool_ranks) // group_size:
                raise MdpPlanError(
                    "MDP: encoder execution rank slots name an exact nested subgroup."
                )
            if not isinstance(execution.rank_slots, tuple):
                raise MdpPlanError("MDP: encoder execution rank slots are an immutable tuple.")
            rank_slots = tuple(
                _require_integer(f"encoder execution rank slot {index}", slot)
                for index, slot in enumerate(execution.rank_slots)
            )
            start = group_index * group_size
            expected_slots = tuple(range(start, start + group_size))
            if rank_slots != expected_slots:
                raise MdpPlanError(
                    "MDP: encoder execution rank slots name an exact nested subgroup."
                )
            if used_slots.intersection(rank_slots):
                raise MdpPlanError("MDP: encoder wave rank slots are pairwise disjoint.")
            used_slots.update(rank_slots)

            if not isinstance(execution.item_ids, tuple) or not execution.item_ids:
                raise MdpPlanError(
                    "MDP: encoder execution has immutable non-empty vision-item IDs."
                )
            item_ids = tuple(
                _require_item_id(f"encoder execution item {index}", item_id)
                for index, item_id in enumerate(execution.item_ids)
            )
            if len(set(item_ids)) != len(item_ids):
                raise MdpPlanError("MDP: encoder execution item IDs are unique.")
            if any(item_id not in expected_items for item_id in item_ids):
                raise MdpPlanError("MDP: encoder execution items belong to the plan catalog.")
            positions = tuple(catalog_positions[item_id] for item_id in item_ids)
            if positions != tuple(sorted(positions)):
                raise MdpPlanError("MDP: encoder execution items follow catalog order.")
            if covered.intersection(item_ids):
                raise MdpPlanError("MDP: encoder plan covers every catalog item exactly once.")
            covered.update(item_ids)

            rows = _require_integer(
                "encoder execution effective_rows_per_rank",
                execution.effective_rows_per_rank,
                positive=True,
            )
            if rows > capacity:
                raise MdpPlanError("MDP: encoder execution rows fit per-rank capacity.")
            cost = _require_integer("encoder execution cost_units", execution.cost_units)
            for slot in rank_slots:
                slot_costs[slot] = _require_integer(
                    "encoder rank-slot cumulative cost", slot_costs[slot] + cost
                )
            execution_keys.append((rank_slots[0], group_size, item_ids))

        if tuple(execution_keys) != tuple(sorted(execution_keys)):
            raise MdpPlanError("MDP: encoder wave executions are in canonical order.")

    if covered != expected_items:
        raise MdpPlanError("MDP: encoder plan covers every catalog item exactly once.")
    if not isinstance(plan.digest, bytes) or len(plan.digest) != 16:
        raise MdpPlanError("MDP: encoder dynamic plan digest is exactly 16 bytes.")
    expected_digest = _encoder_digest(source_samples, pool_ranks, capacity, plan.waves)
    if plan.digest != expected_digest:
        raise MdpPlanError("MDP: encoder dynamic plan digest matches its canonical contents.")
    return plan
