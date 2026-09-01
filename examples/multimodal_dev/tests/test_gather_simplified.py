# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for simplified gather_to_inner_dp_zero (metadata-only path)."""

import inspect

import pytest

# modality_bridge imports torch at module level; skip all gather tests
# when running on a CPU-only login node without torch installed.
torch = pytest.importorskip("torch", reason="torch not available on this node")

from examples.multimodal_dev.modality_bridge import (  # noqa: E402
    gather_order_row_counts_from_assignment,
    gather_to_inner_dp_zero,
    per_rank_row_counts_from_assignment,
)


def _fake_cp2_group():
    """Return None to signal single-rank (no-op) tests."""
    return None


class TestGatherSimplified:
    """gather_to_inner_dp_zero no longer accepts return_zero_dependency_only."""

    def test_signature_has_no_return_zero_dependency_only(self):
        sig = inspect.signature(gather_to_inner_dp_zero)
        assert "return_zero_dependency_only" not in sig.parameters, (
            "return_zero_dependency_only must be removed from gather_to_inner_dp_zero"
        )

    def test_global_per_image_row_counts_required(self):
        """The function requires global_per_image_row_counts (metadata path)."""
        sig = inspect.signature(gather_to_inner_dp_zero)
        param = sig.parameters.get("global_per_image_row_counts")
        assert param is not None
        assert param.default is inspect.Parameter.empty, (
            "global_per_image_row_counts must be required (no default)"
        )


class TestPerRankRowCounts:
    def test_single_rank(self):
        assignment = {0: [(0, 0), (0, 1)]}
        row_counts = [3, 5]
        result = per_rank_row_counts_from_assignment(assignment, row_counts, world_size=1)
        assert result == [8]

    def test_two_ranks_balanced(self):
        assignment = {0: [(0, 0)], 1: [(0, 1)]}
        row_counts = [4, 4]
        result = per_rank_row_counts_from_assignment(assignment, row_counts, world_size=2)
        assert result == [4, 4]

    def test_two_ranks_imbalanced(self):
        assignment = {0: [(0, 0), (0, 2)], 1: [(0, 1)]}
        row_counts = [3, 7, 2]
        result = per_rank_row_counts_from_assignment(assignment, row_counts, world_size=2)
        assert result == [5, 7]


class TestGatherOrderRowCounts:
    def test_gather_order_matches_rank_order(self):
        assignment = {0: [(0, 0)], 1: [(0, 1)]}
        row_counts = [3, 5]
        result = gather_order_row_counts_from_assignment(assignment, row_counts)
        assert result == [3, 5]

    def test_reversed_assignment_still_rank_major(self):
        assignment = {1: [(0, 1)], 0: [(0, 0)]}
        row_counts = [3, 5]
        result = gather_order_row_counts_from_assignment(assignment, row_counts)
        # gather order is rank-major: rank 0 first, rank 1 second.
        assert result == [3, 5]
