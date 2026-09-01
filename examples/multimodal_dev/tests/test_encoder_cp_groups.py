# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for encoder CP group construction (PP0-only gather groups)."""

import pytest

from examples.multimodal_dev.mdp_parallel_groups import compute_encoder_cp_groups


def _all_ranks_covered(groups, world_size):
    all_ranks = sorted(r for g in groups for r in g)
    return all_ranks == list(range(world_size))


class TestEncoderCPGroups:
    def test_pp1_cp2_gather_equals_cp_group(self):
        """With PP=1 the encoder gather group is the same as the CP group."""
        enc_groups, pp_sync_groups = compute_encoder_cp_groups(
            world_size=4, tp_size=1, cp_size=2, pp_size=1, encoder_cp_size=2
        )
        assert len(enc_groups) == 2
        assert all(len(g) == 2 for g in enc_groups)
        assert _all_ranks_covered(enc_groups, 4)

    def test_pp2_cp2_gather_restricted_to_pp0(self):
        """With PP=2, the gather group contains only PP0 ranks."""
        enc_groups, pp_sync_groups = compute_encoder_cp_groups(
            world_size=8, tp_size=1, cp_size=2, pp_size=2, encoder_cp_size=2,
            order="tp-cp-ep-dp-pp",
        )
        # PP0 ranks with CP=2, DP=2: there are 2 gather groups, each of size 2.
        assert all(len(g) == 2 for g in enc_groups)
        # PP stride = TP*CP*DP = 1*2*2 = 4; PP0 ranks are 0-3.
        for g in enc_groups:
            for r in g:
                assert r < 4, f"Non-PP0 rank {r} in encoder gather group {g}"

    def test_pp2_cp2_sync_groups_span_all_pp(self):
        """PP vision sync groups cover all PP stages."""
        _, pp_sync_groups = compute_encoder_cp_groups(
            world_size=8, tp_size=1, cp_size=2, pp_size=2, encoder_cp_size=2,
            order="tp-cp-ep-dp-pp",
        )
        # Each sync group has one rank per PP stage: size 2.
        assert all(len(g) == 2 for g in pp_sync_groups)
        assert _all_ranks_covered(pp_sync_groups, 8)

    def test_encoder_cp_size_must_divide_cp_size(self):
        with pytest.raises(ValueError, match="must divide"):
            compute_encoder_cp_groups(
                world_size=8, tp_size=1, cp_size=3, pp_size=2, encoder_cp_size=2
            )

    def test_world_size_must_divide_tp_cp_pp(self):
        with pytest.raises(ValueError, match="divisible"):
            compute_encoder_cp_groups(
                world_size=7, tp_size=1, cp_size=2, pp_size=2, encoder_cp_size=2
            )

    def test_tp2_cp2_pp2(self):
        """TP=2 slice is correctly handled."""
        enc_groups, pp_sync_groups = compute_encoder_cp_groups(
            world_size=16, tp_size=2, cp_size=2, pp_size=2, encoder_cp_size=2,
            order="tp-cp-ep-dp-pp",
        )
        # PP0 occupies ranks 0–7 (pp_stride = TP*CP*DP = 2*2*2 = 8).
        for g in enc_groups:
            for r in g:
                assert r < 8, f"Non-PP0 rank {r} in encoder gather group"

    def test_gather_groups_disjoint_and_cover_pp0(self):
        """Encoder gather groups partition exactly the PP0 ranks."""
        enc_groups, _ = compute_encoder_cp_groups(
            world_size=8, tp_size=1, cp_size=2, pp_size=2, encoder_cp_size=2,
            order="tp-cp-ep-dp-pp",
        )
        pp0_ranks = list(range(4))
        covered = sorted(r for g in enc_groups for r in g)
        assert covered == pp0_ranks
