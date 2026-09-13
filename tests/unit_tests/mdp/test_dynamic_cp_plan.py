# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure tests for independent decoder and encoder Dynamic-CP plans."""

from dataclasses import replace
from importlib import import_module

import pytest

from megatron.core.mdp.dynamic_cp import (
    DynamicCpGroupSpec,
    GlobalSampleId,
    GlobalVisionItemId,
    nested_dynamic_cp_group_specs,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderCpAssignment,
    DecoderSampleMetadata,
    EncoderVisionItemMetadata,
    EncoderWorkEstimate,
    EncoderWorkUnit,
    build_decoder_dynamic_plan,
    build_encoder_dynamic_plan,
    validate_dynamic_plan_catalog,
    validate_encoder_dynamic_plan,
)
from megatron.core.mdp.errors import MdpPlanError

_INT64_MAX = 2**63 - 1


def _validate_decoder_plan(plan):
    return import_module("megatron.core.mdp.dynamic_cp_plan").validate_decoder_dynamic_plan(plan)


def _sample(lane, order, valid, padded, *item_ids):
    sample_id = GlobalSampleId(lane, order)
    return DecoderSampleMetadata(
        sample_id=sample_id,
        valid_seqlen=valid,
        padded_seqlen=padded,
        vision_items=tuple(
            EncoderVisionItemMetadata(
                item_id=GlobalVisionItemId(lane, item_id),
                sample_id=sample_id,
                image_ordinal=image_ordinal,
            )
            for image_ordinal, item_id in enumerate(item_ids)
        ),
    )


class _TwoMicrobatchSolver:
    """Native-shaped solver fixture with one full and one split DPxCP wave."""

    def __init__(self):
        self.calls = []

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        self.calls.append((tuple(sample_seqlens), total_gpus, max_seq_len_per_rank, min_cp_size))
        if len(self.calls) == 1:
            assert sample_seqlens == [(0, 16), (1, 2), (2, 8)]
            return (
                [[16], [16], [16], [16]],
                [(1, 2), (2, 8)],
                [object()] * 4,
                [[0], [0], [0], [0]],
            )
        assert sample_seqlens == [(1, 2), (2, 8)]
        return ([[8], [8], [2], [2]], [], None, [[2], [2], [1], [1]])


def _full_group_solver(sample_seqlens, total_gpus, **kwargs):
    del kwargs
    sample_ids = [sample_id for sample_id, _ in sample_seqlens]
    lengths = [length for _, length in sample_seqlens]
    return (
        [list(lengths) for _ in range(total_gpus)],
        [],
        None,
        [list(sample_ids) for _ in range(total_gpus)],
    )


def _four_singleton_plan():
    samples = tuple(_sample(0, index, 4, 4) for index in range(4))

    def solver(sample_seqlens, total_gpus, **kwargs):
        del kwargs
        assert sample_seqlens == [(0, 4), (1, 4), (2, 4), (3, 4)]
        assert total_gpus == 4
        return ([[4], [4], [4], [4]], [], None, [[0], [1], [2], [3]])

    return build_decoder_dynamic_plan(
        samples,
        decoder_ranks=(9, 3, 12, 6),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=solver,
    )


def test_decoder_plan_maps_native_solver_output_to_stable_global_ids():
    samples = (_sample(1, 0, 7, 8), _sample(0, 1, 2, 2, 2), _sample(0, 0, 13, 16, 0, 1))
    solver = _TwoMicrobatchSolver()

    plan = build_decoder_dynamic_plan(
        samples,
        decoder_ranks=(17, 3, 21, 8),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=solver,
    )

    assert solver.calls == [(((0, 16), (1, 2), (2, 8)), 4, 4, 1), (((1, 2), (2, 8)), 4, 4, 1)]
    assert plan.effective_num_microbatches == 2
    assert plan.sample_ids == (GlobalSampleId(0, 0), GlobalSampleId(0, 1), GlobalSampleId(1, 0))
    assert plan.item_ids == (
        GlobalVisionItemId(0, 0),
        GlobalVisionItemId(0, 1),
        GlobalVisionItemId(0, 2),
    )
    assert tuple(
        (assignment.sample_ids, assignment.endpoint_ranks)
        for assignment in plan.microbatches[0].assignments
    ) == (((GlobalSampleId(0, 0),), (17, 3, 21, 8)),)
    assert tuple(
        (assignment.sample_ids, assignment.endpoint_ranks, assignment.local_cp_size)
        for assignment in plan.microbatches[1].assignments
    ) == (((GlobalSampleId(1, 0),), (17, 3), 2), ((GlobalSampleId(0, 1),), (21, 8), 2))


def test_decoder_plan_digest_and_records_ignore_input_order():
    samples = (_sample(0, 0, 13, 16, 0, 1), _sample(0, 1, 2, 2, 2), _sample(1, 0, 7, 8))
    first = build_decoder_dynamic_plan(
        samples,
        decoder_ranks=(17, 3, 21, 8),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=_TwoMicrobatchSolver(),
    )
    second = build_decoder_dynamic_plan(
        tuple(reversed(samples)),
        decoder_ranks=(17, 3, 21, 8),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=_TwoMicrobatchSolver(),
    )

    assert first.samples == second.samples
    assert first.microbatches == second.microbatches
    assert first.digest == second.digest
    assert len(first.digest) == 16


@pytest.mark.parametrize(
    "padded_lengths,expected_cp_sizes",
    [((4, 4, 4, 4), (1, 1, 1, 1)), ((8, 8), (2, 2)), ((16,), (4,))],
)
def test_decoder_plan_uses_actual_native_packing_solver(padded_lengths, expected_cp_sizes):
    from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware

    plan = build_decoder_dynamic_plan(
        tuple(
            _sample(source_lane, 0, padded_length, padded_length)
            for source_lane, padded_length in enumerate(padded_lengths)
        ),
        decoder_ranks=(9, 3, 12, 6),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=next_hdp_group_packing_aware,
    )

    assert plan.effective_num_microbatches == 1
    assert (
        tuple(assignment.local_cp_size for assignment in plan.microbatches[0].assignments)
        == expected_cp_sizes
    )
    assert {
        sample_id
        for assignment in plan.microbatches[0].assignments
        for sample_id in assignment.sample_ids
    } == {GlobalSampleId(source_lane, 0) for source_lane in range(len(padded_lengths))}


def test_decoder_digest_changes_with_capacity_and_ordered_rank_semantics():
    sample = _sample(0, 0, 4, 4, 0)
    baseline = build_decoder_dynamic_plan(
        (sample,),
        decoder_ranks=(9, 3, 12, 6),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=_full_group_solver,
    )
    changed_capacity = build_decoder_dynamic_plan(
        (sample,),
        decoder_ranks=(9, 3, 12, 6),
        max_seqlen_per_rank=5,
        minimum_cp_size=1,
        solver=_full_group_solver,
    )
    changed_rank_order = build_decoder_dynamic_plan(
        (sample,),
        decoder_ranks=(3, 9, 12, 6),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=_full_group_solver,
    )

    assert baseline.digest != changed_capacity.digest
    assert baseline.digest != changed_rank_order.digest


def test_public_decoder_plan_validator_accepts_builder_output_and_rejects_stale_digest():
    plan = _four_singleton_plan()

    assert _validate_decoder_plan(plan) is plan
    with pytest.raises(MdpPlanError):
        _validate_decoder_plan(replace(plan, digest=b"stale-plan-hash!"))


def test_public_decoder_plan_validator_rejects_content_reassignment_with_stale_digest():
    plan = _four_singleton_plan()
    microbatch = plan.microbatches[0]
    assignments = list(microbatch.assignments)
    assignments[0] = replace(assignments[0], sample_ids=assignments[1].sample_ids)
    assignments[1] = replace(assignments[1], sample_ids=microbatch.assignments[0].sample_ids)
    corrupted = replace(plan, microbatches=(replace(microbatch, assignments=tuple(assignments)),))

    with pytest.raises(MdpPlanError):
        _validate_decoder_plan(corrupted)


def test_public_decoder_plan_validator_rejects_non_power_of_two_three_plus_one_slices():
    plan = _four_singleton_plan()
    sample_ids = plan.sample_ids
    corrupted_microbatch = replace(
        plan.microbatches[0],
        assignments=(
            DecoderCpAssignment(sample_ids[:3], plan.decoder_ranks[:3]),
            DecoderCpAssignment(sample_ids[3:], plan.decoder_ranks[3:]),
        ),
    )

    with pytest.raises(MdpPlanError):
        _validate_decoder_plan(replace(plan, microbatches=(corrupted_microbatch,)))


@pytest.mark.parametrize("mutation", ("capacity", "minimum-cp"))
def test_public_decoder_plan_validator_rechecks_capacity_and_minimum_cp(mutation):
    plan = _four_singleton_plan()
    if mutation == "capacity":
        oversized = replace(plan.samples[0], valid_seqlen=5, padded_seqlen=5)
        corrupted = replace(plan, samples=(oversized, *plan.samples[1:]))
    else:
        corrupted = replace(plan, minimum_cp_size=2)

    with pytest.raises(MdpPlanError):
        _validate_decoder_plan(corrupted)


def test_decoder_plan_rejects_sample_over_capacity_before_solver_call():
    calls = []

    with pytest.raises(MdpPlanError, match="capacity"):
        build_decoder_dynamic_plan(
            (_sample(0, 0, 17, 17),),
            decoder_ranks=(0, 1, 2, 3),
            max_seqlen_per_rank=4,
            minimum_cp_size=1,
            solver=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


@pytest.mark.parametrize("error_type", [AssertionError, TypeError])
def test_decoder_plan_normalizes_solver_call_failures(error_type):
    def failing_solver(*args, **kwargs):
        del args, kwargs
        raise error_type("native solver failure")

    with pytest.raises(MdpPlanError, match="solver call failed"):
        build_decoder_dynamic_plan(
            (_sample(0, 0, 2, 2),),
            decoder_ranks=(0, 1),
            max_seqlen_per_rank=4,
            minimum_cp_size=1,
            solver=failing_solver,
        )


@pytest.mark.parametrize(
    "samples,match",
    [
        ((_sample(0, 0, 2, 2), _sample(0, 0, 2, 2)), "sample IDs are unique"),
        ((_sample(0, 0, 2, 2, 0), _sample(0, 1, 2, 2, 0)), "item IDs are unique"),
        ((_sample(0, 0, 3, 2),), "valid_seqlen <= padded_seqlen"),
        ((_sample(0, 0, True, 2),), "positive integer"),
        ((_sample(0, 0, 1, _INT64_MAX + 1),), "signed int64"),
    ],
)
def test_decoder_plan_rejects_invalid_source_metadata(samples, match):
    with pytest.raises(MdpPlanError, match=match):
        build_decoder_dynamic_plan(
            samples,
            decoder_ranks=(0, 1),
            max_seqlen_per_rank=4,
            minimum_cp_size=1,
            solver=lambda *args, **kwargs: None,
        )


def test_decoder_plan_rejects_inconsistent_item_catalog_order():
    sample_id = GlobalSampleId(0, 0)
    items = (
        EncoderVisionItemMetadata(GlobalVisionItemId(0, 1), sample_id, 1),
        EncoderVisionItemMetadata(GlobalVisionItemId(0, 0), sample_id, 0),
    )
    sample = DecoderSampleMetadata(sample_id, 2, 2, items)

    with pytest.raises(MdpPlanError, match="image_ordinal order"):
        build_decoder_dynamic_plan(
            (sample,),
            decoder_ranks=(0, 1),
            max_seqlen_per_rank=4,
            minimum_cp_size=1,
            solver=lambda *args, **kwargs: None,
        )


def test_decoder_plan_rejects_item_catalog_sample_mismatch():
    sample = DecoderSampleMetadata(
        GlobalSampleId(0, 0),
        2,
        2,
        (EncoderVisionItemMetadata(GlobalVisionItemId(1, 0), GlobalSampleId(1, 0), 0),),
    )

    with pytest.raises(MdpPlanError, match="owning sample"):
        build_decoder_dynamic_plan(
            (sample,),
            decoder_ranks=(0, 1),
            max_seqlen_per_rank=4,
            minimum_cp_size=1,
            solver=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize(
    "failure,match",
    [
        ("duplicate", "solver.*duplicate"),
        ("lost", "solver.*coverage"),
        ("no_progress", "solver.*progress"),
        ("bad_lengths", "solver.*length"),
        ("unknown", "solver.*unknown"),
        ("bool_id", "solver.*integer"),
        ("float_length", "solver.*integer"),
        ("overlap", "solver.*overlap"),
        ("wrong_rank_rows", "solver.*rank rows"),
        ("empty_rank", "solver.*empty rank"),
        ("disjoint_duplicate", "solver.*logical sample"),
        ("packed_over_capacity", "solver.*capacity"),
    ],
)
def test_decoder_plan_rejects_malformed_solver_coverage(failure, match):
    def solver(sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        del max_seq_len_per_rank, min_cp_size
        if failure == "duplicate":
            return [[2, 2]] * total_gpus, [(1, 2)], None, [[0, 0]] * total_gpus
        if failure == "lost":
            return [[2]] * total_gpus, [], None, [[0]] * total_gpus
        if failure == "no_progress":
            return (
                [[] for _ in range(total_gpus)],
                sample_seqlens,
                None,
                [[] for _ in range(total_gpus)],
            )
        if failure == "bad_lengths":
            return [[99]] * total_gpus, [(1, 2)], None, [[0]] * total_gpus
        if failure == "unknown":
            return [[2]] * total_gpus, sample_seqlens, None, [[99]] * total_gpus
        if failure == "bool_id":
            return [[2]] * total_gpus, sample_seqlens, None, [[True]] * total_gpus
        if failure == "float_length":
            return [[2.0]] * total_gpus, sample_seqlens, None, [[0]] * total_gpus
        if failure == "overlap":
            return [[2]] * total_gpus, sample_seqlens, None, [[0]] * total_gpus
        if failure == "wrong_rank_rows":
            return [[2]], [(1, 2)], None, [[0]] * total_gpus
        if failure == "empty_rank":
            return [[2], []], [(1, 2)], None, [[0], []]
        if failure == "disjoint_duplicate":
            return [[2], [2], [2], [2]], [(2, 2)], None, [[0], [1], [0], [1]]
        return [[5, 5]] * total_gpus, [], None, [[0, 1]] * total_gpus

    if failure == "disjoint_duplicate":
        samples = tuple(_sample(0, index, 2, 2) for index in range(3))
        ranks = (0, 1, 2, 3)
    elif failure == "packed_over_capacity":
        samples = (_sample(0, 0, 5, 5), _sample(0, 1, 5, 5))
        ranks = (0, 1)
    else:
        samples = (_sample(0, 0, 2, 2), _sample(0, 1, 2, 2))
        ranks = (0, 1)
    with pytest.raises(MdpPlanError, match=match):
        build_decoder_dynamic_plan(
            samples, decoder_ranks=ranks, max_seqlen_per_rank=4, minimum_cp_size=1, solver=solver
        )


@pytest.mark.parametrize(
    "failure,match",
    [
        ("unaligned", "aligned"),
        ("cp3", "power of two"),
        ("lengths", "identical ordered IDs and lengths"),
    ],
)
def test_decoder_plan_rejects_invalid_physical_replica_groups(failure, match):
    def solver(*args, **kwargs):
        del args, kwargs
        if failure == "unaligned":
            rank_ids = [[1], [0], [0], [2]]
            rank_lengths = [[2], [2], [2], [2]]
        elif failure == "cp3":
            rank_ids = [[0], [0], [0], [1]]
            rank_lengths = [[2], [2], [2], [2]]
        else:
            rank_ids = [[0], [0], [1], [2]]
            rank_lengths = [[2], [3], [2], [2]]
        return rank_lengths, [], None, rank_ids

    samples = tuple(_sample(0, index, 2, 2) for index in range(3))
    with pytest.raises(MdpPlanError, match=match):
        build_decoder_dynamic_plan(
            samples,
            decoder_ranks=(0, 1, 2, 3),
            max_seqlen_per_rank=4,
            minimum_cp_size=1,
            solver=solver,
        )


def _workload_query(item_ids, group_size):
    key = tuple(item_id.to_wire_tuple() for item_id in item_ids)
    estimates = {
        (((0, 0), (0, 1)), 1): EncoderWorkEstimate(5, 9),
        (((0, 0), (0, 1)), 2): EncoderWorkEstimate(3, 4),
        (((0, 2),), 1): EncoderWorkEstimate(2, 7),
        (((0, 2),), 2): EncoderWorkEstimate(1, 3),
    }
    return estimates[(key, group_size)]


def _encoder_validation_plan():
    samples = (_sample(0, 0, 1, 1, 0), _sample(0, 1, 1, 1, 1))
    item_ids = tuple(item.item_id for sample in samples for item in sample.vision_items)
    return build_encoder_dynamic_plan(
        samples,
        tuple(EncoderWorkUnit((item_id,)) for item_id in item_ids),
        group_specs=nested_dynamic_cp_group_specs((9, 3), minimum_size=1),
        max_seqlen_per_rank=1,
        workload_query=lambda ids, size: EncoderWorkEstimate(1, 1),
    )


@pytest.mark.parametrize(
    "invalid_digest,match",
    [
        (b"stale-plan-hash!", "digest matches"),
        (b"short", "exactly 16 bytes"),
        (bytearray(16), "exactly 16 bytes"),
    ],
)
def test_public_encoder_plan_validator_accepts_builder_output_and_rejects_invalid_digest(
    invalid_digest, match
):
    plan = _encoder_validation_plan()

    assert validate_encoder_dynamic_plan(plan) is plan
    with pytest.raises(MdpPlanError, match=match):
        validate_encoder_dynamic_plan(replace(plan, digest=invalid_digest))


@pytest.mark.parametrize(
    "failure,match",
    [
        ("source_tuple", "immutable tuple"),
        ("source_order", "canonical ID order"),
        ("pool_tuple", "immutable tuple"),
        ("pool_size", "power-of-two"),
        ("duplicate_pool", "unique"),
        ("wave_tuple", "immutable tuple"),
        ("wave_index", "indices are contiguous"),
        ("empty_executions", "non-empty executions"),
        ("execution_carrier", "EncoderExecution carriers"),
        ("group_size", "scheduled power of two"),
        ("group_index", "exact nested subgroup"),
        ("rank_slots", "exact nested subgroup"),
        ("execution_order", "canonical order"),
        ("item_order", "catalog order"),
        ("missing_coverage", "exactly once"),
        ("duplicate_coverage", "exactly once"),
        ("rows_capacity", "per-rank capacity"),
        ("negative_cost", "integer >= 0"),
        ("capacity", "positive integer"),
        ("pool_rank_int64", "signed int64"),
        ("row_int64", "signed int64"),
        ("cost_int64", "signed int64"),
        ("cumulative_cost", "signed int64"),
    ],
)
def test_public_encoder_plan_validator_rejects_malformed_structure(failure, match):
    plan = _encoder_validation_plan()
    wave = plan.waves[0]
    first, second = wave.executions

    if failure == "source_tuple":
        corrupted = replace(plan, source_samples=list(plan.source_samples))
    elif failure == "source_order":
        corrupted = replace(plan, source_samples=tuple(reversed(plan.source_samples)))
    elif failure == "pool_tuple":
        corrupted = replace(plan, pool_ranks=list(plan.pool_ranks))
    elif failure == "pool_size":
        corrupted = replace(plan, pool_ranks=(*plan.pool_ranks, 5))
    elif failure == "duplicate_pool":
        corrupted = replace(plan, pool_ranks=(9, 9))
    elif failure == "wave_tuple":
        corrupted = replace(plan, waves=list(plan.waves))
    elif failure == "wave_index":
        corrupted = replace(plan, waves=(replace(wave, wave_index=1),))
    elif failure == "empty_executions":
        corrupted = replace(plan, waves=(replace(wave, executions=()),))
    elif failure == "execution_carrier":
        corrupted = replace(plan, waves=(replace(wave, executions=(object(),)),))
    elif failure == "group_size":
        corrupted = replace(
            plan, waves=(replace(wave, executions=(replace(first, group_size=3), second)),)
        )
    elif failure == "group_index":
        corrupted = replace(
            plan, waves=(replace(wave, executions=(replace(first, group_index=1), second)),)
        )
    elif failure == "rank_slots":
        corrupted = replace(
            plan, waves=(replace(wave, executions=(replace(first, rank_slots=(1,)), second)),)
        )
    elif failure == "execution_order":
        corrupted = replace(
            plan, waves=(replace(wave, executions=tuple(reversed(wave.executions))),)
        )
    elif failure == "item_order":
        combined = replace(
            first, group_size=2, rank_slots=(0, 1), item_ids=tuple(reversed(plan.item_ids))
        )
        corrupted = replace(plan, waves=(replace(wave, executions=(combined,)),))
    elif failure == "missing_coverage":
        corrupted = replace(plan, waves=(replace(wave, executions=(first,)),))
    elif failure == "duplicate_coverage":
        corrupted = replace(
            plan,
            waves=(replace(wave, executions=(first, replace(second, item_ids=first.item_ids))),),
        )
    elif failure == "rows_capacity":
        corrupted = replace(
            plan,
            waves=(replace(wave, executions=(replace(first, effective_rows_per_rank=2), second)),),
        )
    elif failure == "negative_cost":
        corrupted = replace(
            plan, waves=(replace(wave, executions=(replace(first, cost_units=-1), second)),)
        )
    elif failure == "capacity":
        corrupted = replace(plan, max_seqlen_per_rank=0)
    elif failure == "pool_rank_int64":
        corrupted = replace(plan, pool_ranks=(_INT64_MAX + 1, plan.pool_ranks[1]))
    elif failure == "row_int64":
        corrupted = replace(
            plan,
            waves=(
                replace(
                    wave,
                    executions=(replace(first, effective_rows_per_rank=_INT64_MAX + 1), second),
                ),
            ),
        )
    elif failure == "cost_int64":
        corrupted = replace(
            plan,
            waves=(replace(wave, executions=(replace(first, cost_units=_INT64_MAX + 1), second)),),
        )
    else:
        first_wave = replace(wave, executions=(replace(first, cost_units=_INT64_MAX),))
        second_wave = replace(
            wave,
            wave_index=1,
            executions=(replace(second, group_index=0, rank_slots=(0,), cost_units=1),),
        )
        corrupted = replace(plan, waves=(first_wave, second_wave))

    with pytest.raises(MdpPlanError, match=match):
        validate_encoder_dynamic_plan(corrupted)


def test_encoder_plan_queries_whole_pack_and_builds_disjoint_execution_waves():
    decoder_samples = (_sample(0, 0, 13, 16, 0, 1), _sample(0, 1, 2, 2, 2), _sample(1, 0, 7, 8))
    catalog = tuple(item for sample in decoder_samples for item in sample.vision_items)
    item_ids = tuple(item.item_id for item in catalog)
    specs = nested_dynamic_cp_group_specs((9, 3), minimum_size=1)
    calls = []

    def query(ids, group_size):
        calls.append((ids, group_size))
        return _workload_query(ids, group_size)

    plan = build_encoder_dynamic_plan(
        decoder_samples,
        (EncoderWorkUnit((item_ids[0], item_ids[1])), EncoderWorkUnit((item_ids[2],))),
        group_specs=specs,
        max_seqlen_per_rank=4,
        workload_query=query,
    )

    assert calls == [
        ((item_ids[0], item_ids[1]), 1),
        ((item_ids[0], item_ids[1]), 2),
        ((item_ids[2],), 1),
    ]
    assert plan.items == catalog
    assert plan.sample_ids == tuple(
        sample.sample_id for sample in sorted(decoder_samples, key=lambda value: value.sample_id)
    )
    assert len(plan.waves) == 2
    assert tuple(
        (
            execution.item_ids,
            execution.group_size,
            execution.group_index,
            execution.rank_slots,
            execution.effective_rows_per_rank,
            execution.cost_units,
        )
        for wave in plan.waves
        for execution in wave.executions
    ) == (((item_ids[2],), 1, 0, (0,), 2, 7), ((item_ids[0], item_ids[1]), 2, 0, (0, 1), 3, 4))
    assert len(plan.digest) == 16


def test_encoder_lpt_fills_available_subgroups_in_one_wave_by_pool_order():
    sample = _sample(0, 0, 4, 4, 0, 1, 2, 3)
    catalog = sample.vision_items
    item_ids = tuple(item.item_id for item in catalog)
    specs = nested_dynamic_cp_group_specs((11, 4, 19, 2), minimum_size=1)

    def query(ids, group_size):
        assert group_size == 1
        return EncoderWorkEstimate(1, 10 - ids[0].local_item_id)

    plan = build_encoder_dynamic_plan(
        (sample,),
        tuple(EncoderWorkUnit((item_id,)) for item_id in reversed(item_ids)),
        group_specs=specs,
        max_seqlen_per_rank=1,
        workload_query=query,
    )

    assert len(plan.waves) == 1
    assert tuple(execution.rank_slots for execution in plan.waves[0].executions) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )
    assert tuple(execution.item_ids[0] for execution in plan.waves[0].executions) == item_ids


def test_encoder_wave_can_mix_nonoverlapping_group_sizes():
    sample = _sample(0, 0, 2, 2, 0, 1)
    catalog = sample.vision_items

    def query(item_ids, group_size):
        if item_ids == (catalog[0].item_id,):
            return EncoderWorkEstimate(2 if group_size == 1 else 1, 10)
        return EncoderWorkEstimate(1, 5)

    plan = build_encoder_dynamic_plan(
        (sample,),
        tuple(EncoderWorkUnit((item.item_id,)) for item in catalog),
        group_specs=nested_dynamic_cp_group_specs((11, 4, 19, 2), minimum_size=1),
        max_seqlen_per_rank=1,
        workload_query=query,
    )

    assert len(plan.waves) == 1
    assert tuple(
        (execution.group_size, execution.rank_slots) for execution in plan.waves[0].executions
    ) == ((2, (0, 1)), (1, (2,)))


def test_encoder_nested_groups_balance_physical_slot_load_across_waves():
    sample = _sample(0, 0, 3, 3, 0, 1, 2)
    catalog = sample.vision_items
    costs = {catalog[0].item_id: 100, catalog[1].item_id: 90, catalog[2].item_id: 10}

    def query(item_ids, group_size):
        item_id = item_ids[0]
        if item_id != catalog[2].item_id:
            return EncoderWorkEstimate(2 if group_size == 1 else 1, costs[item_id])
        return EncoderWorkEstimate(1, costs[item_id])

    plan = build_encoder_dynamic_plan(
        (sample,),
        tuple(EncoderWorkUnit((item.item_id,)) for item in catalog),
        group_specs=nested_dynamic_cp_group_specs((11, 4, 19, 2), minimum_size=1),
        max_seqlen_per_rank=1,
        workload_query=query,
    )

    assert tuple(
        (execution.item_ids, execution.rank_slots) for execution in plan.waves[0].executions
    ) == (((catalog[0].item_id,), (0, 1)), ((catalog[1].item_id,), (2, 3)))
    assert tuple(
        (execution.item_ids, execution.group_index, execution.rank_slots)
        for execution in plan.waves[1].executions
    ) == (
        ((catalog[2].item_id,), 2, (2,)),
    )


def test_encoder_equal_cost_ties_use_global_item_id_then_group_index():
    sample = _sample(0, 0, 2, 2, 0, 1)
    catalog = sample.vision_items
    units = tuple(EncoderWorkUnit((item.item_id,)) for item in reversed(catalog))
    specs = nested_dynamic_cp_group_specs((9, 3), minimum_size=1)

    first = build_encoder_dynamic_plan(
        (sample,),
        units,
        group_specs=specs,
        max_seqlen_per_rank=1,
        workload_query=lambda item_ids, size: EncoderWorkEstimate(1, 7),
    )
    second = build_encoder_dynamic_plan(
        (sample,),
        tuple(reversed(units)),
        group_specs=tuple(reversed(specs)),
        max_seqlen_per_rank=1,
        workload_query=lambda item_ids, size: EncoderWorkEstimate(1, 7),
    )

    expected = (((catalog[0].item_id,), 0), ((catalog[1].item_id,), 1))
    assert (
        tuple(
            (execution.item_ids, execution.group_index) for execution in first.waves[0].executions
        )
        == expected
    )
    assert first.waves == second.waves
    assert first.digest == second.digest


def test_reverse_unequal_decoder_cp2_encoder_cp4():
    sample = _sample(0, 0, 2, 2, 0)
    decoder = build_decoder_dynamic_plan(
        (sample,),
        decoder_ranks=(17, 3),
        max_seqlen_per_rank=2,
        minimum_cp_size=1,
        solver=_full_group_solver,
    )

    def query(item_ids, group_size):
        return EncoderWorkEstimate(1 if group_size == 4 else 2, 1)

    encoder = build_encoder_dynamic_plan(
        decoder.samples,
        (EncoderWorkUnit((sample.vision_items[0].item_id,)),),
        group_specs=nested_dynamic_cp_group_specs((9, 3, 12, 6), minimum_size=1),
        max_seqlen_per_rank=1,
        workload_query=query,
    )

    validate_dynamic_plan_catalog(decoder, encoder)
    assert decoder.microbatches[0].assignments[0].local_cp_size == 2
    assert encoder.waves[0].executions[0].group_size == 4


def _encoder_plan_for_source_samples(source_samples):
    items = tuple(item for sample in source_samples for item in sample.vision_items)
    return build_encoder_dynamic_plan(
        source_samples,
        tuple(EncoderWorkUnit((item.item_id,)) for item in items),
        group_specs=nested_dynamic_cp_group_specs((9, 3), minimum_size=1),
        max_seqlen_per_rank=2,
        workload_query=lambda item_ids, size: EncoderWorkEstimate(1, 1),
    )


def test_dynamic_decoder_encoder_catalog_join_accepts_exact_source_catalog():
    samples = (_sample(0, 0, 1, 1, 0), _sample(0, 1, 1, 1))
    decoder = build_decoder_dynamic_plan(
        samples,
        decoder_ranks=(17, 3),
        max_seqlen_per_rank=2,
        minimum_cp_size=1,
        solver=_full_group_solver,
    )
    encoder = _encoder_plan_for_source_samples(tuple(reversed(samples)))

    assert validate_dynamic_plan_catalog(decoder, encoder) is None
    assert decoder.samples == encoder.source_samples


@pytest.mark.parametrize("invalid_plan", ["decoder", "encoder"])
def test_dynamic_catalog_join_validates_each_plan_before_comparing_catalogs(invalid_plan):
    encoder = _encoder_validation_plan()
    decoder = build_decoder_dynamic_plan(
        encoder.source_samples,
        decoder_ranks=(17, 3),
        max_seqlen_per_rank=1,
        minimum_cp_size=1,
        solver=_full_group_solver,
    )

    if invalid_plan == "decoder":
        decoder = replace(decoder, digest=b"stale-plan-hash!")
    else:
        encoder = replace(encoder, digest=b"stale-plan-hash!")

    with pytest.raises(MdpPlanError, match=f"{invalid_plan} dynamic plan digest matches"):
        validate_dynamic_plan_catalog(decoder, encoder)


@pytest.mark.parametrize(
    "mismatch", ["missing", "extra", "substituted", "remapped", "unknown_sample"]
)
def test_dynamic_decoder_encoder_catalog_join_rejects_mismatch(mismatch):
    decoder_samples = (_sample(0, 0, 1, 1, 0), _sample(0, 1, 1, 1))
    if mismatch == "missing":
        encoder_samples = (_sample(0, 0, 1, 1), _sample(0, 1, 1, 1))
    elif mismatch == "extra":
        encoder_samples = (_sample(0, 0, 1, 1, 0, 1), _sample(0, 1, 1, 1))
    elif mismatch == "substituted":
        encoder_samples = (_sample(0, 0, 1, 1, 1), _sample(0, 1, 1, 1))
    elif mismatch == "remapped":
        encoder_samples = (_sample(0, 0, 1, 1), _sample(0, 1, 1, 1, 0))
    else:
        encoder_samples = (*decoder_samples, _sample(0, 2, 1, 1))

    decoder = build_decoder_dynamic_plan(
        decoder_samples,
        decoder_ranks=(17, 3),
        max_seqlen_per_rank=2,
        minimum_cp_size=1,
        solver=_full_group_solver,
    )
    encoder = _encoder_plan_for_source_samples(encoder_samples)

    with pytest.raises(MdpPlanError, match="exact same canonical source"):
        validate_dynamic_plan_catalog(decoder, encoder)


def test_encoder_plan_digest_is_stable_with_unequal_decoder_encoder_maxima():
    samples = (_sample(0, 0, 13, 16, 0, 1), _sample(0, 1, 2, 2, 2), _sample(1, 0, 7, 8))
    decoder = build_decoder_dynamic_plan(
        samples,
        decoder_ranks=(17, 3, 21, 8),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=_TwoMicrobatchSolver(),
    )
    specs = nested_dynamic_cp_group_specs((9, 3), minimum_size=1)
    units = (
        EncoderWorkUnit((GlobalVisionItemId(0, 0), GlobalVisionItemId(0, 1))),
        EncoderWorkUnit((GlobalVisionItemId(0, 2),)),
    )

    first = build_encoder_dynamic_plan(
        decoder.samples,
        units,
        group_specs=specs,
        max_seqlen_per_rank=4,
        workload_query=_workload_query,
    )
    second = build_encoder_dynamic_plan(
        tuple(reversed(decoder.samples)),
        tuple(reversed(units)),
        group_specs=tuple(reversed(specs)),
        max_seqlen_per_rank=4,
        workload_query=_workload_query,
    )

    assert len(decoder.microbatches[0].assignments[0].endpoint_ranks) == 4
    assert max(execution.group_size for wave in first.waves for execution in wave.executions) == 2
    assert first.waves == second.waves
    assert first.digest == second.digest


def test_encoder_digest_changes_with_adapter_rows():
    sample = _sample(0, 0, 1, 1, 0)
    catalog = sample.vision_items
    unit = (EncoderWorkUnit((catalog[0].item_id,)),)
    specs = nested_dynamic_cp_group_specs((9, 3), minimum_size=1)
    first = build_encoder_dynamic_plan(
        (sample,),
        unit,
        group_specs=specs,
        max_seqlen_per_rank=4,
        workload_query=lambda ids, size: EncoderWorkEstimate(1, 1),
    )
    second = build_encoder_dynamic_plan(
        (sample,),
        unit,
        group_specs=specs,
        max_seqlen_per_rank=4,
        workload_query=lambda ids, size: EncoderWorkEstimate(2, 1),
    )

    assert first.digest != second.digest


def test_text_only_encoder_plan_is_empty_and_does_not_query_adapter():
    calls = []
    sample = _sample(0, 0, 1, 1)
    plan = build_encoder_dynamic_plan(
        (sample,),
        (),
        group_specs=nested_dynamic_cp_group_specs((5, 1), minimum_size=1),
        max_seqlen_per_rank=4,
        workload_query=lambda *args: calls.append(args),
    )

    assert plan.items == ()
    assert plan.waves == ()
    assert calls == []
    assert validate_encoder_dynamic_plan(plan) is plan


@pytest.mark.parametrize(
    "source_samples,units,match",
    [
        (
            (_sample(0, 0, 1, 1, 0),),
            (EncoderWorkUnit((GlobalVisionItemId(0, 0), GlobalVisionItemId(0, 0))),),
            "unique",
        ),
        ((_sample(0, 0, 1, 1, 0, 1),), (EncoderWorkUnit((GlobalVisionItemId(0, 0),)),), "exactly"),
        (
            (_sample(0, 0, 1, 1, 0),),
            (
                EncoderWorkUnit((GlobalVisionItemId(0, 0),)),
                EncoderWorkUnit((GlobalVisionItemId(0, 0),)),
            ),
            "exactly",
        ),
    ],
)
def test_encoder_plan_rejects_duplicate_or_lost_items(source_samples, units, match):
    with pytest.raises(MdpPlanError, match=match):
        build_encoder_dynamic_plan(
            source_samples,
            units,
            group_specs=nested_dynamic_cp_group_specs((0, 1), minimum_size=1),
            max_seqlen_per_rank=4,
            workload_query=lambda ids, size: EncoderWorkEstimate(1, 1),
        )


def test_encoder_plan_rejects_noncanonical_item_order_inside_work_unit():
    sample = _sample(0, 0, 1, 1, 0, 1)
    catalog = sample.vision_items
    with pytest.raises(MdpPlanError, match="catalog order"):
        build_encoder_dynamic_plan(
            (sample,),
            (EncoderWorkUnit((catalog[1].item_id, catalog[0].item_id)),),
            group_specs=nested_dynamic_cp_group_specs((0, 1), minimum_size=1),
            max_seqlen_per_rank=4,
            workload_query=lambda ids, size: EncoderWorkEstimate(1, 1),
        )


def test_encoder_plan_rejects_item_owned_by_unknown_source_sample():
    source_sample_id = GlobalSampleId(0, 0)
    item_id = GlobalVisionItemId(0, 0)
    source_sample = DecoderSampleMetadata(
        source_sample_id, 1, 1, (EncoderVisionItemMetadata(item_id, GlobalSampleId(0, 1), 0),)
    )

    with pytest.raises(MdpPlanError, match="owning sample"):
        build_encoder_dynamic_plan(
            (source_sample,),
            (EncoderWorkUnit((item_id,)),),
            group_specs=nested_dynamic_cp_group_specs((0, 1), minimum_size=1),
            max_seqlen_per_rank=4,
            workload_query=lambda item_ids, size: EncoderWorkEstimate(1, 1),
        )


def test_encoder_plan_rejects_work_that_exceeds_maximum_group_capacity():
    sample = _sample(0, 0, 1, 1, 0)
    catalog = sample.vision_items

    with pytest.raises(MdpPlanError, match="capacity"):
        build_encoder_dynamic_plan(
            (sample,),
            (EncoderWorkUnit((catalog[0].item_id,)),),
            group_specs=nested_dynamic_cp_group_specs((9, 3), minimum_size=1),
            max_seqlen_per_rank=4,
            workload_query=lambda ids, size: EncoderWorkEstimate(5, 1),
        )


@pytest.mark.parametrize(
    "estimate",
    [
        EncoderWorkEstimate(0, 1),
        EncoderWorkEstimate(True, 1),
        EncoderWorkEstimate(1.0, 1),
        EncoderWorkEstimate(1, -1),
        EncoderWorkEstimate(1, False),
        EncoderWorkEstimate(1, 1.0),
        EncoderWorkEstimate(_INT64_MAX + 1, 1),
    ],
)
def test_encoder_plan_rejects_invalid_adapter_estimate(estimate):
    sample = _sample(0, 0, 1, 1, 0)
    catalog = sample.vision_items

    with pytest.raises(MdpPlanError, match="adapter"):
        build_encoder_dynamic_plan(
            (sample,),
            (EncoderWorkUnit((catalog[0].item_id,)),),
            group_specs=nested_dynamic_cp_group_specs((0,), minimum_size=1),
            max_seqlen_per_rank=4,
            workload_query=lambda ids, size: estimate,
        )


@pytest.mark.parametrize("error_type", [AssertionError, TypeError])
def test_encoder_plan_normalizes_workload_query_failures(error_type):
    sample = _sample(0, 0, 1, 1, 0)

    def failing_query(*args):
        del args
        raise error_type("adapter failure")

    with pytest.raises(MdpPlanError, match="workload query failed") as captured:
        build_encoder_dynamic_plan(
            (sample,),
            (EncoderWorkUnit((sample.vision_items[0].item_id,)),),
            group_specs=nested_dynamic_cp_group_specs((0, 1), minimum_size=1),
            max_seqlen_per_rank=1,
            workload_query=failing_query,
        )

    assert isinstance(captured.value.__cause__, error_type)


def test_encoder_plan_rejects_rank_that_cannot_enter_fixed_width_digest():
    sample = _sample(0, 0, 1, 1, 0)
    catalog = sample.vision_items

    with pytest.raises(MdpPlanError, match="signed int64"):
        build_encoder_dynamic_plan(
            (sample,),
            (EncoderWorkUnit((catalog[0].item_id,)),),
            group_specs=(DynamicCpGroupSpec(1, 0, (_INT64_MAX + 1,)),),
            max_seqlen_per_rank=4,
            workload_query=lambda ids, size: EncoderWorkEstimate(1, 1),
        )
