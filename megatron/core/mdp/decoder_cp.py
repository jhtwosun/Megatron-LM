# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Immutable decoder context-parallel slice planning.

The base MDP plan contains only fixed-width vision descriptors. This
supplemental plan is built from explicit decoder metadata before any bridge
transport. It maps encoder-output rows to the exact native CP-local decoder
positions without inspecting opaque model payloads.
"""

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from megatron.core.context_parallel_layout import get_thd_context_parallel_rank_indices
from megatron.core.mdp.errors import MdpPlanError

DECODER_CP_SLICE_SCHEMA_VERSION = 1
_ITERATION_DIGEST_DOMAIN = b"megatron.mdp.decoder_cp.iteration.v1"


@dataclass(frozen=True)
class DecoderCpItemSlice:
    """One item's rows owned by one decoder-CP endpoint.

    ``source_row_ids`` index the item's complete encoder output. Matching
    ``local_decoder_positions`` index flattened rank-local decoder input.
    Empty tuples remain explicit for zero-row ownership.
    """

    global_item_id: int
    output_rows: int
    source_row_ids: tuple
    local_decoder_positions: tuple
    leaf_row_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_row_ids, tuple) or not isinstance(
            self.local_decoder_positions, tuple
        ):
            raise MdpPlanError("MDP: decoder CP item slice row mappings must be tuples.")
        if self.output_rows < 0:
            raise MdpPlanError(
                f"MDP: decoder CP item {self.global_item_id} output_rows must be " "non-negative."
            )
        if self.leaf_row_start < 0:
            raise MdpPlanError(
                f"MDP: decoder CP item {self.global_item_id} leaf_row_start must be "
                "non-negative."
            )
        if len(self.source_row_ids) != len(self.local_decoder_positions):
            raise MdpPlanError(
                f"MDP: decoder CP item {self.global_item_id} violates: source rows "
                "and local decoder positions have equal length."
            )
        if len(set(self.source_row_ids)) != len(self.source_row_ids) or any(
            type(row) is not int or row < 0 or row >= self.output_rows
            for row in self.source_row_ids
        ):
            raise MdpPlanError(
                f"MDP: decoder CP item {self.global_item_id} source rows must be "
                f"unique integer indices inside output_rows={self.output_rows}."
            )
        if len(set(self.local_decoder_positions)) != len(self.local_decoder_positions) or any(
            type(position) is not int or position < 0 for position in self.local_decoder_positions
        ):
            raise MdpPlanError(
                f"MDP: decoder CP item {self.global_item_id} local decoder positions "
                "must be unique non-negative integer indices."
            )

    @property
    def row_count(self) -> int:
        """Number of compact rows owned by this endpoint."""
        return len(self.source_row_ids)


@dataclass(frozen=True)
class DecoderCpMicrobatchSlice:
    """One microbatch's compact leaf layout on one decoder-CP endpoint."""

    microbatch_id: int
    slice_id: int
    endpoint_rank: int
    decoder_input_shape: tuple
    local_decoder_input_shape: tuple
    packed_cu_seqlens_q_padded: tuple
    items: tuple

    def __post_init__(self) -> None:
        _validate_decoder_input_shape(self.decoder_input_shape)
        _validate_decoder_input_shape(self.local_decoder_input_shape)
        if not isinstance(self.packed_cu_seqlens_q_padded, tuple):
            raise MdpPlanError("MDP: packed_cu_seqlens_q_padded must be an immutable tuple.")
        if self.packed_cu_seqlens_q_padded:
            _validate_cu_seqlens(self.packed_cu_seqlens_q_padded)
            if self.packed_cu_seqlens_q_padded[-1] != self.decoder_input_shape[1]:
                raise MdpPlanError(
                    "MDP: packed_cu_seqlens_q_padded last boundary must equal the "
                    "decoder sequence length."
                )
        if not isinstance(self.items, tuple):
            raise MdpPlanError("MDP: decoder CP microbatch slice items must be a tuple.")

        expected_leaf_start = 0
        seen_items = set()
        seen_local_positions = set()
        local_tokens = self.local_decoder_input_shape[0] * self.local_decoder_input_shape[1]
        for item in self.items:
            if item.global_item_id in seen_items:
                raise MdpPlanError(
                    f"MDP: decoder CP microbatch {self.microbatch_id} slice "
                    f"{self.slice_id} contains duplicate item {item.global_item_id}."
                )
            seen_items.add(item.global_item_id)
            if item.leaf_row_start != expected_leaf_start:
                raise MdpPlanError(
                    f"MDP: decoder CP microbatch {self.microbatch_id} slice "
                    f"{self.slice_id} leaf_row_start values must be contiguous in "
                    "item order."
                )
            expected_leaf_start += item.row_count
            for position in item.local_decoder_positions:
                if position >= local_tokens:
                    raise MdpPlanError(
                        f"MDP: local decoder position {position} lies outside local "
                        f"shape {self.local_decoder_input_shape}."
                    )
                if position in seen_local_positions:
                    raise MdpPlanError(
                        f"MDP: decoder CP microbatch {self.microbatch_id} slice "
                        f"{self.slice_id} requires unique decoder positions across "
                        "vision items."
                    )
                seen_local_positions.add(position)

    @property
    def total_leaf_rows(self) -> int:
        """Valid compact leaf rows on this endpoint."""
        return sum(item.row_count for item in self.items)


@dataclass(frozen=True)
class DecoderCpSlicePlan:
    """Tuple-only supplemental decoder-CP plan for one MDP iteration."""

    schema_version: int
    iteration: int
    cp_size: int
    microbatch_slices: tuple
    digest: bytes
    _microbatch_slice_by_key: dict = field(init=False, repr=False, compare=False)
    _item_slice_by_key: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != DECODER_CP_SLICE_SCHEMA_VERSION:
            raise MdpPlanError(
                f"MDP: decoder CP slice schema {self.schema_version} != "
                f"{DECODER_CP_SLICE_SCHEMA_VERSION}."
            )
        if self.cp_size < 1:
            raise MdpPlanError(f"MDP: decoder CP size must be positive, got {self.cp_size}.")
        if not isinstance(self.microbatch_slices, tuple):
            raise MdpPlanError("MDP: decoder CP microbatch_slices must be a tuple.")
        if not isinstance(self.digest, bytes) or len(self.digest) != 16:
            raise MdpPlanError("MDP: decoder CP slice digest must be exactly 16 bytes.")

        keys = tuple((entry.microbatch_id, entry.slice_id) for entry in self.microbatch_slices)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise MdpPlanError(
                "MDP: decoder CP microbatch slices must have unique sorted "
                "(microbatch_id, slice_id) keys."
            )

        slices_by_microbatch = {}
        endpoint_by_slice = {}
        item_microbatch = {}
        microbatch_slice_by_key = {}
        item_slice_by_key = {}
        for entry in self.microbatch_slices:
            if not 0 <= entry.slice_id < self.cp_size:
                raise MdpPlanError(
                    f"MDP: decoder CP slice_id={entry.slice_id} lies outside "
                    f"[0, {self.cp_size})."
                )
            prior_endpoint = endpoint_by_slice.setdefault(entry.slice_id, entry.endpoint_rank)
            if prior_endpoint != entry.endpoint_rank:
                raise MdpPlanError(
                    f"MDP: decoder CP slice {entry.slice_id} must use one stable " "endpoint rank."
                )
            slices_by_microbatch.setdefault(entry.microbatch_id, []).append(entry)
            microbatch_slice_by_key[(entry.microbatch_id, entry.slice_id)] = entry
            for item in entry.items:
                prior_microbatch = item_microbatch.setdefault(
                    item.global_item_id, entry.microbatch_id
                )
                if prior_microbatch != entry.microbatch_id:
                    raise MdpPlanError(
                        f"MDP: decoder CP item {item.global_item_id} appears in "
                        "multiple microbatches."
                    )
                item_slice_by_key[(item.global_item_id, entry.slice_id)] = item

        for microbatch_id, entries in slices_by_microbatch.items():
            if tuple(entry.slice_id for entry in entries) != tuple(range(self.cp_size)):
                raise MdpPlanError(
                    f"MDP: decoder CP microbatch {microbatch_id} must contain every "
                    f"slice 0..{self.cp_size - 1}."
                )
            first = entries[0]
            expected_item_ids = tuple(item.global_item_id for item in first.items)
            expected_output_rows = tuple(item.output_rows for item in first.items)
            coverage = {item_id: [] for item_id in expected_item_ids}
            for entry in entries:
                if (
                    entry.decoder_input_shape != first.decoder_input_shape
                    or entry.local_decoder_input_shape != first.local_decoder_input_shape
                    or entry.packed_cu_seqlens_q_padded != first.packed_cu_seqlens_q_padded
                ):
                    raise MdpPlanError(
                        f"MDP: decoder CP microbatch {microbatch_id} shapes must "
                        "agree across endpoint slices."
                    )
                if (
                    tuple(item.global_item_id for item in entry.items) != expected_item_ids
                    or tuple(item.output_rows for item in entry.items) != expected_output_rows
                ):
                    raise MdpPlanError(
                        f"MDP: decoder CP microbatch {microbatch_id} item layout "
                        "must agree across endpoint slices."
                    )
                for item in entry.items:
                    coverage[item.global_item_id].extend(item.source_row_ids)
            for item_id, output_rows in zip(expected_item_ids, expected_output_rows):
                rows = coverage[item_id]
                if len(rows) != len(set(rows)) or sorted(rows) != list(range(output_rows)):
                    raise MdpPlanError(
                        f"MDP: decoder CP item {item_id} source rows must form one "
                        f"disjoint, complete 0..{output_rows - 1} union across "
                        "endpoint slices."
                    )

        object.__setattr__(self, "_microbatch_slice_by_key", microbatch_slice_by_key)
        object.__setattr__(self, "_item_slice_by_key", item_slice_by_key)
        expected_digest = compute_decoder_cp_slice_digest(
            self.schema_version, self.iteration, self.cp_size, self.microbatch_slices
        )
        if self.digest != expected_digest:
            raise MdpPlanError("MDP: decoder CP slice digest does not match its immutable data.")

    def microbatch_slice(self, microbatch_id: int, slice_id: int) -> DecoderCpMicrobatchSlice:
        """Return one endpoint microbatch slice."""
        try:
            return self._microbatch_slice_by_key[(microbatch_id, slice_id)]
        except KeyError:
            raise MdpPlanError(
                f"MDP: decoder CP slice ({microbatch_id}, {slice_id}) is not in " "this plan."
            ) from None

    def item_slice(self, global_item_id: int, slice_id: int) -> DecoderCpItemSlice:
        """Return one item's slice on one endpoint."""
        try:
            return self._item_slice_by_key[(global_item_id, slice_id)]
        except KeyError:
            raise MdpPlanError(
                f"MDP: decoder CP item slice ({global_item_id}, {slice_id}) is not " "in this plan."
            ) from None


def decoder_cp_rank_global_indices(
    *, decoder_input_shape: tuple, cp_size: int, packed_cu_seqlens: Optional[Sequence[int]]
) -> tuple:
    """Global flattened token indices in each decoder-CP rank's local order.

    ``packed_cu_seqlens=None`` selects BSHD. Packed THD and BSHD both use the
    generic native zigzag layout helper; BSHD applies its single-sequence order
    independently to each batch row.
    """
    batch, sequence = _validate_decoder_input_shape(decoder_input_shape)
    if cp_size < 1:
        raise MdpPlanError(f"MDP: decoder CP size must be positive, got {cp_size}.")

    if packed_cu_seqlens is None:
        if cp_size == 1:
            return (tuple(range(batch * sequence)),)
        divisor = 2 * cp_size
        if sequence % divisor:
            raise MdpPlanError(
                f"MDP: BSHD sequence length {sequence} must be divisible by "
                f"2 * decoder CP size ({divisor})."
            )
        single_sequence = _rank_indices_from_layout((0, sequence), cp_size)
        return tuple(
            tuple(
                batch_index * sequence + sequence_index
                for batch_index in range(batch)
                for sequence_index in single_sequence[cp_rank]
            )
            for cp_rank in range(cp_size)
        )

    if batch != 1:
        raise MdpPlanError(
            f"MDP: THD decoder_input_shape must have B=1, got " f"{decoder_input_shape}."
        )
    cu = _strict_int_tuple(packed_cu_seqlens, "decoder THD cu_seqlens")
    if not cu or cu[-1] != sequence:
        raise MdpPlanError(
            "MDP: THD cu_seqlens last boundary must equal decoder sequence "
            f"length {sequence}, got {cu[-1] if cu else None}."
        )
    if cp_size == 1:
        _validate_cu_seqlens(cu)
        return (tuple(range(sequence)),)
    return _rank_indices_from_layout(cu, cp_size)


def build_decoder_cp_slice_plan(
    plan,
    records: Sequence,
    *,
    decoder_endpoint_ranks: Sequence[int],
    cp_partition_mode: str = "zigzag",
) -> DecoderCpSlicePlan:
    """Build and validate the supplemental plan from explicit window metadata."""
    if cp_partition_mode != "zigzag":
        raise MdpPlanError(
            "MDP: decoder CP local routing requires cp_partition_mode='zigzag', "
            f"got {cp_partition_mode!r}."
        )
    endpoints = _strict_int_tuple(decoder_endpoint_ranks, "decoder_endpoint_ranks")
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise MdpPlanError(
            "MDP: decoder_endpoint_ranks must be a non-empty tuple of unique " "integer ranks."
        )
    cp_size = len(endpoints)
    _validate_route_product(plan, endpoints)

    records_by_microbatch = {}
    for record in records:
        if record.microbatch_id in records_by_microbatch:
            raise MdpPlanError(
                f"MDP: duplicate decoder metadata for microbatch " f"{record.microbatch_id}."
            )
        records_by_microbatch[record.microbatch_id] = record
    expected_microbatches = tuple(layout.microbatch_id for layout in plan.layouts)
    if set(records_by_microbatch) != set(expected_microbatches):
        raise MdpPlanError("MDP: decoder metadata must cover exactly the base plan microbatches.")

    microbatch_slices = []
    seen_global_item_ids = set()
    for layout in plan.layouts:
        record = records_by_microbatch[layout.microbatch_id]
        shape = record.decoder_input_shape
        params = record.decoder_packed_seq_params
        packed_cu = None
        if params is not None:
            if getattr(params, "qkv_format", None) != "thd":
                raise MdpPlanError(
                    f"MDP: microbatch {layout.microbatch_id} decoder packed format "
                    "must be 'thd' when packed metadata is present."
                )
            packed_mode = getattr(params, "cp_partition_mode", "zigzag")
            if packed_mode != "zigzag":
                raise MdpPlanError(
                    f"MDP: microbatch {layout.microbatch_id} decoder packed "
                    "metadata requires cp_partition_mode='zigzag', got "
                    f"{packed_mode!r}."
                )
            raw_cu = getattr(params, "cu_seqlens_q_padded", None)
            if raw_cu is None:
                raise MdpPlanError(
                    f"MDP: microbatch {layout.microbatch_id} THD metadata requires "
                    "cu_seqlens_q_padded."
                )
            raw_values = (
                raw_cu.detach().cpu().tolist() if isinstance(raw_cu, torch.Tensor) else raw_cu
            )
            packed_cu = _strict_int_tuple(raw_values, "packed_cu_seqlens_q_padded")

        rank_indices = decoder_cp_rank_global_indices(
            decoder_input_shape=shape, cp_size=cp_size, packed_cu_seqlens=packed_cu
        )
        owner_by_global_position = {}
        for slice_id, indices in enumerate(rank_indices):
            for local_position, global_position in enumerate(indices):
                if global_position in owner_by_global_position:
                    raise MdpPlanError(
                        f"MDP: decoder token {global_position} has multiple CP " "owners."
                    )
                owner_by_global_position[global_position] = (slice_id, local_position)
        total_tokens = shape[0] * shape[1]
        if set(owner_by_global_position) != set(range(total_tokens)):
            raise MdpPlanError(
                f"MDP: decoder CP rank indices must cover all {total_tokens} " "tokens once."
            )

        record_items = tuple(record.vision_items)
        base_item_ids = tuple(segment.global_item_id for segment in layout.segments)
        record_item_ids = tuple(item.global_item_id for item in record_items)
        if record_item_ids != base_item_ids:
            raise MdpPlanError(
                f"MDP: microbatch {layout.microbatch_id} vision item order must "
                "match the base plan layout."
            )
        item_mappings = []
        microbatch_positions = set()
        for segment, item in zip(layout.segments, record_items):
            if (
                item.output_rows != segment.output_rows
                or len(item.decoder_positions) != segment.output_rows
            ):
                raise MdpPlanError(
                    f"MDP: decoder CP item {item.global_item_id} output_rows must "
                    "match both the base plan and decoder_positions length."
                )
            positions = _strict_int_tuple(item.decoder_positions, "decoder_positions")
            if any(position < 0 or position >= total_tokens for position in positions):
                raise MdpPlanError(
                    f"MDP: decoder CP item {item.global_item_id} positions must lie "
                    f"inside decoder_input_shape={shape}."
                )
            if len(set(positions)) != len(positions) or microbatch_positions.intersection(
                positions
            ):
                raise MdpPlanError(
                    f"MDP: microbatch {layout.microbatch_id} requires unique "
                    "decoder positions across vision items."
                )
            microbatch_positions.update(positions)
            if item.global_item_id in seen_global_item_ids:
                raise MdpPlanError(
                    f"MDP: decoder CP item {item.global_item_id} appears more than " "once."
                )
            seen_global_item_ids.add(item.global_item_id)
            item_mappings.append((segment, positions))

        local_shape = (shape[0], shape[1] // cp_size)
        for slice_id, endpoint_rank in enumerate(endpoints):
            item_slices = []
            leaf_row_start = 0
            for segment, positions in item_mappings:
                source_rows = []
                local_positions = []
                for source_row, global_position in enumerate(positions):
                    owner_slice, local_position = owner_by_global_position[global_position]
                    if owner_slice == slice_id:
                        source_rows.append(source_row)
                        local_positions.append(local_position)
                item_slice = DecoderCpItemSlice(
                    global_item_id=segment.global_item_id,
                    output_rows=segment.output_rows,
                    source_row_ids=tuple(source_rows),
                    local_decoder_positions=tuple(local_positions),
                    leaf_row_start=leaf_row_start,
                )
                item_slices.append(item_slice)
                leaf_row_start += item_slice.row_count
            microbatch_slices.append(
                DecoderCpMicrobatchSlice(
                    microbatch_id=layout.microbatch_id,
                    slice_id=slice_id,
                    endpoint_rank=endpoint_rank,
                    decoder_input_shape=shape,
                    local_decoder_input_shape=local_shape,
                    packed_cu_seqlens_q_padded=packed_cu or (),
                    items=tuple(item_slices),
                )
            )

    microbatch_slices = tuple(microbatch_slices)
    digest = compute_decoder_cp_slice_digest(
        DECODER_CP_SLICE_SCHEMA_VERSION, plan.iteration, cp_size, microbatch_slices
    )
    return DecoderCpSlicePlan(
        schema_version=DECODER_CP_SLICE_SCHEMA_VERSION,
        iteration=plan.iteration,
        cp_size=cp_size,
        microbatch_slices=microbatch_slices,
        digest=digest,
    )


def compute_decoder_cp_slice_digest(
    schema_version: int,
    iteration: int,
    cp_size: int,
    microbatch_slices: Sequence[DecoderCpMicrobatchSlice],
) -> bytes:
    """Return a deterministic length-delimited digest of every slice field."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"megatron.mdp.decoder_cp.slice_plan\x00")
    _digest_ints(hasher, (schema_version, iteration, cp_size))
    _digest_length(hasher, len(microbatch_slices))
    for entry in microbatch_slices:
        _digest_ints(hasher, (entry.microbatch_id, entry.slice_id, entry.endpoint_rank))
        _digest_ints(hasher, entry.decoder_input_shape)
        _digest_ints(hasher, entry.local_decoder_input_shape)
        _digest_ints(hasher, entry.packed_cu_seqlens_q_padded)
        _digest_length(hasher, len(entry.items))
        for item in entry.items:
            _digest_ints(hasher, (item.global_item_id, item.output_rows, item.leaf_row_start))
            _digest_ints(hasher, item.source_row_ids)
            _digest_ints(hasher, item.local_decoder_positions)
    return hasher.digest()


def compute_decoder_cp_iteration_digest(base_digest: bytes, slice_digest: bytes) -> bytes:
    """Combine base and supplemental plans under a distinct digest domain."""
    for name, digest in (("base", base_digest), ("slice", slice_digest)):
        if not isinstance(digest, bytes) or len(digest) != 16:
            raise MdpPlanError(f"MDP: decoder CP {name} plan digest must be exactly 16 bytes.")
    hasher = hashlib.blake2b(digest_size=16)
    _digest_length(hasher, len(_ITERATION_DIGEST_DOMAIN))
    hasher.update(_ITERATION_DIGEST_DOMAIN)
    for digest in (base_digest, slice_digest):
        _digest_length(hasher, len(digest))
        hasher.update(digest)
    return hasher.digest()


def assert_consistent_decoder_cp_iteration(plan, slice_plan, *, planning_group) -> None:
    """All-gather the combined digest on every compact-routing iteration."""
    import torch.distributed as dist

    if plan.iteration != slice_plan.iteration:
        raise MdpPlanError(
            f"MDP: base iteration {plan.iteration} != decoder CP slice iteration "
            f"{slice_plan.iteration}."
        )
    digest = compute_decoder_cp_iteration_digest(plan.digest, slice_plan.digest)
    local = torch.tensor(list(digest), dtype=torch.uint8, device="cuda")
    group_size = dist.get_world_size(group=planning_group)
    gathered = [torch.empty_like(local) for _ in range(group_size)]
    dist.all_gather(gathered, local, group=planning_group)
    digests = tuple(bytes(value.tolist()) for value in gathered)
    if any(other != digest for other in digests):
        raise MdpPlanError(
            f"MDP: combined decoder CP plan digest mismatch at iteration "
            f"{plan.iteration} in outer_dp_rank={plan.outer_dp_rank}: "
            f"{[value.hex() for value in digests]}."
        )


def _rank_indices_from_layout(cu_seqlens: tuple, cp_size: int) -> tuple:
    _validate_cu_seqlens(cu_seqlens)
    cu = torch.tensor(cu_seqlens, dtype=torch.long)
    try:
        return tuple(
            tuple(
                int(index)
                for index in get_thd_context_parallel_rank_indices(
                    cu, cp_size, cp_rank, "zigzag"
                ).tolist()
            )
            for cp_rank in range(cp_size)
        )
    except ValueError as error:
        raise MdpPlanError(f"MDP: invalid decoder CP zigzag metadata: {error}") from error


def _validate_cu_seqlens(cu_seqlens: tuple) -> None:
    if not cu_seqlens or cu_seqlens[0] != 0:
        raise MdpPlanError("MDP: decoder THD cu_seqlens must start at zero.")
    if any(current < previous for previous, current in zip(cu_seqlens, cu_seqlens[1:])):
        raise MdpPlanError("MDP: decoder THD cu_seqlens must be nondecreasing.")


def _validate_decoder_input_shape(shape: tuple) -> tuple:
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise MdpPlanError(
            "MDP: decoder_input_shape must be an explicit (B, S) tuple, got " f"{shape!r}."
        )
    batch, sequence = shape
    if type(batch) is not int or type(sequence) is not int or batch < 1 or sequence < 1:
        raise MdpPlanError(
            "MDP: decoder_input_shape must contain positive integers, got " f"{shape!r}."
        )
    return batch, sequence


def _validate_route_product(plan, endpoints: tuple) -> None:
    item_ids = tuple(
        segment.global_item_id for layout in plan.layouts for segment in layout.segments
    )
    expected = {
        (item_id, slice_id, endpoint)
        for item_id in item_ids
        for slice_id, endpoint in enumerate(endpoints)
    }
    actual = {(route.global_item_id, route.slice_id, route.endpoint_rank) for route in plan.routes}
    if len(actual) != len(plan.routes) or actual != expected:
        raise MdpPlanError(
            "MDP: base route product must contain exactly one route for every "
            "(vision item, decoder CP endpoint) pair."
        )


def _strict_int_tuple(values: Sequence[int], name: str) -> tuple:
    values = tuple(values)
    if any(type(value) is not int for value in values):
        raise MdpPlanError(f"MDP: {name} must contain only integers.")
    return values


def _digest_length(hasher, length: int) -> None:
    hasher.update(struct.pack("<Q", length))


def _digest_ints(hasher, values: Sequence[int]) -> None:
    values = tuple(int(value) for value in values)
    _digest_length(hasher, len(values))
    if values:
        hasher.update(struct.pack(f"<{len(values)}q", *values))
