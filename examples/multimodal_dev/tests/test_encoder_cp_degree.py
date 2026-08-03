# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Owner mapping and gather coverage for --encoder-context-parallel-size < CP."""

import pytest

from examples.multimodal_dev.data.energon_vision_balance import (
    assign_images_lpt,
    image_costs_from_grid,
)
from examples.multimodal_dev.mdp_parallel_groups import (
    encoder_owner_layout,
    mdp_prepartition_layout,
)

# (context_parallel_size, encoder_context_parallel_size)
CP_ENCODER_CP = [(2, 1), (4, 2), (4, 1), (8, 2)]

# Three packed items with 1..3 images each, deliberately unequal in cost.
GRIDS_BY_ITEM = [
    [(1, 16, 16), (1, 8, 8)],
    [(1, 32, 32)],
    [(1, 8, 16), (1, 16, 8), (1, 24, 24)],
]


def _loader_assignment(cp_size, encoder_cp_size):
    """Mirror the loader: LPT over the encoder owners only."""
    _rank, world, _is_owner = encoder_owner_layout(0, cp_size, encoder_cp_size)
    costs_by_item = [
        image_costs_from_grid(grids, hidden_size=1280) for grids in GRIDS_BY_ITEM
    ]
    return assign_images_lpt(costs_by_item, world, across_items=False)


def _gathered_images(assignment):
    """Mirror the gather: rank-major concatenation, canonical reorder."""
    gather_order = [
        (item_idx, image_idx)
        for rank in sorted(assignment)
        for item_idx, image_idx in assignment[rank]
    ]
    return sorted(gather_order)


@pytest.mark.parametrize("cp_size,encoder_cp_size", CP_ENCODER_CP)
def test_only_the_first_encoder_cp_ranks_are_owners(cp_size, encoder_cp_size):
    for cp_rank in range(cp_size):
        owner_rank, world, is_owner = encoder_owner_layout(
            cp_rank, cp_size, encoder_cp_size
        )
        assert world == encoder_cp_size
        assert is_owner == (cp_rank < encoder_cp_size)
        assert owner_rank == (cp_rank if is_owner else -1)


@pytest.mark.parametrize("cp_size,encoder_cp_size", CP_ENCODER_CP)
def test_every_image_has_exactly_one_owner(cp_size, encoder_cp_size):
    assignment = _loader_assignment(cp_size, encoder_cp_size)
    assert sorted(assignment) == list(range(encoder_cp_size))

    owners = {}
    for rank, items in assignment.items():
        for item_idx, image_idx in items:
            assert (item_idx, image_idx) not in owners
            owners[(item_idx, image_idx)] = rank
            assert 0 <= rank < encoder_cp_size

    all_images = [
        (item_idx, image_idx)
        for item_idx, grids in enumerate(GRIDS_BY_ITEM)
        for image_idx in range(len(grids))
    ]
    assert sorted(owners) == all_images


@pytest.mark.parametrize("cp_size,encoder_cp_size", CP_ENCODER_CP)
def test_every_decoder_cp_rank_receives_every_image(cp_size, encoder_cp_size):
    all_images = [
        (item_idx, image_idx)
        for item_idx, grids in enumerate(GRIDS_BY_ITEM)
        for image_idx in range(len(grids))
    ]
    for cp_rank in range(cp_size):
        # Every CP rank plans against the same owner world, so the gather
        # metadata is identical on owners and non-owners alike.
        owner_rank, world, is_owner = encoder_owner_layout(
            cp_rank, cp_size, encoder_cp_size
        )
        assignment = _loader_assignment(cp_size, encoder_cp_size)
        assert world == encoder_cp_size
        # Non-owners materialize nothing: the sentinel owner index is absent
        # from the assignment, so their local image shard is empty.
        if is_owner:
            assert assignment[owner_rank]
        else:
            assert assignment.get(owner_rank, []) == []
        assert _gathered_images(assignment) == all_images


def test_encoder_cp_size_must_divide_cp_size():
    with pytest.raises(ValueError):
        encoder_owner_layout(0, 4, 3)


# ---------------------------------------------------------------------------
# Loader-layout parity.  Both loaders derive (prepartition_rank,
# prepartition_world, prepartition_encoder_stage) from mdp_prepartition_layout,
# and the layout no longer depends on --mdp-inner-dp-scope: the legacy
# all-PP-stages-encode branch was unreachable under encoder_cp_size | cp_size.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_encoder_cp_equal_to_cp_keeps_the_pre_feature_layout(cp_size, pp_size):
    for pp_rank in range(pp_size):
        for cp_rank in range(cp_size):
            layout = mdp_prepartition_layout(
                cp_rank=cp_rank,
                cp_size=cp_size,
                pp_rank=pp_rank,
                pp_size=pp_size,
                encoder_cp_size=cp_size,
            )
            # PP=1 CP-only: (cp_rank, C, True).
            # PP>1 pp_cp PP0-only gather: (cp_rank, C, pp_rank == 0).
            assert layout == (cp_rank, cp_size, pp_rank == 0 or pp_size == 1)


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("cp_size,encoder_cp_size", CP_ENCODER_CP)
def test_smaller_encoder_cp_only_moves_ownership(cp_size, encoder_cp_size, pp_size):
    for pp_rank in range(pp_size):
        for cp_rank in range(cp_size):
            owner_rank, world, encoder_stage = mdp_prepartition_layout(
                cp_rank=cp_rank,
                cp_size=cp_size,
                pp_rank=pp_rank,
                pp_size=pp_size,
                encoder_cp_size=encoder_cp_size,
            )
            assert world == encoder_cp_size
            assert owner_rank == (cp_rank if cp_rank < encoder_cp_size else -1)
            # The encoder stage stays a PP property: E < C moves ownership
            # within a stage, it does not move which stage encodes.
            assert encoder_stage == (pp_rank == 0 or pp_size == 1)
