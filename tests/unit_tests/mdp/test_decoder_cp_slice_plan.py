# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-CP compact slice-plan mapping and digest tests."""

import dataclasses
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.models.base import _cp_split_tensor
from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.mdp import decoder_cp
from megatron.core.mdp.decoder_cp import (
    DECODER_CP_SLICE_SCHEMA_VERSION,
    build_decoder_cp_slice_plan,
    decoder_cp_rank_global_indices,
)
from megatron.core.mdp.errors import MdpPlanError
from megatron.core.mdp.plan import (
    PLAN_SCHEMA_VERSION,
    EncoderThdLayout,
    EncoderThdSegment,
    LayoutSegment,
    MdpBatchPlan,
    MicrobatchLayout,
    RouteSlice,
    RowCapacityPolicy,
)
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord


def _base_plan(*, item_rows=(4, 2), endpoints=(10, 11)):
    segments = []
    layout_segments = []
    routes = []
    output_start = 0
    for item_id, output_rows in enumerate(item_rows):
        segments.append(
            EncoderThdSegment(
                global_item_id=item_id,
                microbatch_id=0,
                sample_id=0,
                image_ordinal=item_id,
                payload_row_start=item_id,
                payload_rows=1,
                output_row_start=output_start,
                output_rows=output_rows,
                grid_thw=(1, 1, 1),
            )
        )
        layout_segments.append(
            LayoutSegment(
                global_item_id=item_id, leaf_row_start=output_start, output_rows=output_rows
            )
        )
        routes.extend(
            RouteSlice(
                global_item_id=item_id,
                producer_worker_id=0,
                endpoint_rank=endpoint,
                slice_id=slice_id,
            )
            for slice_id, endpoint in enumerate(endpoints)
        )
        output_start += output_rows
    return MdpBatchPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        iteration=7,
        outer_dp_rank=0,
        capacity_policy=RowCapacityPolicy(),
        routes=tuple(routes),
        layouts=(
            MicrobatchLayout(
                microbatch_id=0,
                text_only=not item_rows,
                total_output_rows=output_start,
                segments=tuple(layout_segments),
            ),
        ),
        encoder_layouts=(EncoderThdLayout(producer_worker_id=0, segments=tuple(segments)),),
        digest=b"\x00" * 16,
    )


def _record(
    *, positions=((0, 4, 8, 12), (1, 2)), shape=(1, 16), cu=None, cp_partition_mode="zigzag"
):
    params = None
    if cu is not None:
        params = SimpleNamespace(
            qkv_format="thd",
            cp_partition_mode=cp_partition_mode,
            cu_seqlens_q_padded=torch.tensor(cu, dtype=torch.int32),
        )
    items = tuple(
        MdpMicrobatchVisionRecord(
            global_item_id=item_id,
            sample_id=0,
            image_ordinal=item_id,
            grid_thw=(1, 1, 1),
            output_rows=len(item_positions),
            decoder_positions=tuple(item_positions),
        )
        for item_id, item_positions in enumerate(positions)
    )
    return MdpMicrobatchRecord(
        microbatch_id=0,
        text_only=not items,
        vision_items=items,
        decoder_input_shape=shape,
        decoder_packed_seq_params=params,
        model_payload=MappingProxyType({}),
    )


@pytest.mark.parametrize("cp_size", (1, 2, 4))
def test_bshd_rank_indices_match_native_zigzag_order(cp_size):
    batch, sequence = 3, 16
    actual = decoder_cp_rank_global_indices(
        decoder_input_shape=(batch, sequence), cp_size=cp_size, packed_cu_seqlens=None
    )
    sentinel = torch.arange(batch * sequence).view(batch, sequence)
    for cp_rank in range(cp_size):
        expected = _cp_split_tensor(sentinel, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank).reshape(
            -1
        )
        assert actual[cp_rank] == tuple(expected.tolist())


@pytest.mark.parametrize("cp_size,cu_seqlens", ((2, (0, 8, 8, 24)), (4, (0, 16, 16, 48))))
def test_thd_rank_indices_match_installed_te_with_unequal_and_duplicate_boundaries(
    cp_size, cu_seqlens
):
    actual = decoder_cp_rank_global_indices(
        decoder_input_shape=(1, cu_seqlens[-1]), cp_size=cp_size, packed_cu_seqlens=cu_seqlens
    )
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    for cp_rank in range(cp_size):
        expected = get_thd_partitioned_indices(cu, cu_seqlens[-1], cp_size, cp_rank).long()
        assert actual[cp_rank] == tuple(expected.cpu().tolist())


def test_slice_plan_preserves_zero_and_noncontiguous_item_slices():
    plan = build_decoder_cp_slice_plan(_base_plan(), (_record(),), decoder_endpoint_ranks=(10, 11))
    assert plan.schema_version == DECODER_CP_SLICE_SCHEMA_VERSION
    assert len(plan.digest) == 16
    assert plan.microbatch_slice(0, 0).total_leaf_rows == 4
    assert plan.microbatch_slice(0, 1).total_leaf_rows == 2

    rank0_item0 = plan.item_slice(0, 0)
    rank1_item0 = plan.item_slice(0, 1)
    assert rank0_item0.source_row_ids == (0, 3)
    assert rank0_item0.local_decoder_positions == (0, 4)
    assert rank1_item0.source_row_ids == (1, 2)
    assert rank1_item0.local_decoder_positions == (0, 4)

    rank1_item1 = plan.item_slice(1, 1)
    assert rank1_item1.source_row_ids == ()
    assert rank1_item1.local_decoder_positions == ()
    assert rank1_item1.leaf_row_start == 2

    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.cp_size = 4


def test_slice_digest_is_deterministic_length_delimited_and_covers_positions():
    base = build_decoder_cp_slice_plan(_base_plan(), (_record(),), decoder_endpoint_ranks=(10, 11))
    same = build_decoder_cp_slice_plan(_base_plan(), (_record(),), decoder_endpoint_ranks=(10, 11))
    moved = build_decoder_cp_slice_plan(
        _base_plan(), (_record(positions=((1, 5, 9, 13), (2, 3))),), decoder_endpoint_ranks=(10, 11)
    )
    assert base == same
    assert base.digest == same.digest
    assert moved.digest != base.digest


def test_iteration_digest_is_domain_separated_and_covers_both_plans():
    compute = decoder_cp.compute_decoder_cp_iteration_digest
    base_digest = bytes(range(16))
    slice_digest = bytes(range(16, 32))

    combined = compute(base_digest, slice_digest)
    assert len(combined) == 16
    assert combined == compute(base_digest, slice_digest)
    assert combined != compute(bytes(reversed(base_digest)), slice_digest)
    assert combined != compute(base_digest, bytes(reversed(slice_digest)))
    assert combined not in (base_digest, slice_digest)


def test_slice_digest_preserves_duplicate_packed_boundaries_when_mapping_is_equal():
    compact = build_decoder_cp_slice_plan(
        _base_plan(), (_record(cu=(0, 8, 16)),), decoder_endpoint_ranks=(10, 11)
    )
    duplicated = build_decoder_cp_slice_plan(
        _base_plan(), (_record(cu=(0, 8, 8, 16)),), decoder_endpoint_ranks=(10, 11)
    )
    assert compact.item_slice(0, 0).source_row_ids == duplicated.item_slice(0, 0).source_row_ids
    assert compact.microbatch_slice(0, 0).packed_cu_seqlens_q_padded == (0, 8, 16)
    assert duplicated.microbatch_slice(0, 0).packed_cu_seqlens_q_padded == (0, 8, 8, 16)
    assert compact.digest != duplicated.digest


@pytest.mark.parametrize(
    "builder_mode,packed_mode", (("contiguous", "zigzag"), ("zigzag", "contiguous"))
)
def test_slice_plan_rejects_non_zigzag_partition_mode(builder_mode, packed_mode):
    with pytest.raises(MdpPlanError, match="cp_partition_mode='zigzag'"):
        build_decoder_cp_slice_plan(
            _base_plan(),
            (_record(cu=(0, 8, 16), cp_partition_mode=packed_mode),),
            decoder_endpoint_ranks=(10, 11),
            cp_partition_mode=builder_mode,
        )


@pytest.mark.parametrize(
    "plan,record,endpoints,match",
    (
        (_base_plan(), _record(positions=((0, 4, 8), (1, 2))), (10, 11), "output_rows"),
        (
            _base_plan(),
            _record(positions=((0, 4, 8, 12), (0, 2))),
            (10, 11),
            "unique decoder positions",
        ),
        (
            _base_plan(),
            _record(positions=((0, 4, 8, 16), (1, 2))),
            (10, 11),
            "inside decoder_input_shape",
        ),
        (
            dataclasses.replace(_base_plan(), routes=_base_plan().routes[:-1]),
            _record(),
            (10, 11),
            "route product",
        ),
    ),
)
def test_slice_plan_rejects_incomplete_or_ambiguous_coverage(plan, record, endpoints, match):
    with pytest.raises(MdpPlanError, match=match):
        build_decoder_cp_slice_plan(plan, (record,), decoder_endpoint_ranks=endpoints)


@pytest.mark.parametrize(
    "shape,cu,match",
    (
        ((2, 15), None, "divisible"),
        ((2, 16), (0, 8, 16), "THD decoder_input_shape"),
        ((1, 16), (0, 8, 12), "last boundary"),
    ),
)
def test_rank_index_mapping_rejects_invalid_shape_or_packed_metadata(shape, cu, match):
    with pytest.raises(MdpPlanError, match=match):
        decoder_cp_rank_global_indices(decoder_input_shape=shape, cp_size=2, packed_cu_seqlens=cu)
