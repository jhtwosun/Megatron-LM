# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Iteration-window tests with a stub adapter (CPU only)."""

from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.window import (
    MdpIterationWindow,
    pixel_capture_owner_state,
    pixel_capture_suppressed,
)

MERGE = 2


def _item(sample_id, ordinal, grid, payload_row_start, positions=None):
    t, h, w = grid
    output_rows = t * (h // MERGE) * (w // MERGE)
    return CapturedVisionItem(
        sample_id=sample_id,
        image_ordinal=ordinal,
        grid_thw=grid,
        payload_row_start=payload_row_start,
        payload_rows=t * h * w,
        decoder_positions=(
            tuple(range(100, 100 + output_rows)) if positions is None else positions
        ),
    )


def _microbatch(items, total_rows, sentinel_base=0.0):
    pixels = None
    if items:
        pixels = torch.zeros(total_rows, 4)
        for index, item in enumerate(items):
            pixels[
                item.payload_row_start : item.payload_row_start + item.payload_rows
            ] = sentinel_base + index + 1
    return CapturedMicrobatch(
        decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
        vision_items=tuple(items),
        flat_pixel_payload=pixels,
        model_payload=MappingProxyType({"input_ids": torch.zeros(1, 8)}),
    )


class _StubAdapter:
    payload_width = 4
    spatial_merge_size = MERGE

    def __init__(self, microbatches):
        self._microbatches = list(microbatches)

    def get_batch(self, iterator):
        next(iterator)  # consume exactly one sampler element per microbatch
        if not self._microbatches:
            return None
        return self._microbatches.pop(0)

    def estimate_cost(self, item):
        return item.payload_rows


class _OwnershipAdapter(_StubAdapter):
    """Mirror collate: retain metadata but materialize pixels only for the owner."""

    def __init__(self, microbatches):
        super().__init__(microbatches)
        self.capture_states = []

    def get_batch(self, iterator):
        captured = super().get_batch(iterator)
        suppressed = pixel_capture_suppressed()
        self.capture_states.append((suppressed, pixel_capture_owner_state()))
        if not suppressed:
            return captured
        return CapturedMicrobatch(
            decoder_packed_seq_params=captured.decoder_packed_seq_params,
            vision_items=captured.vision_items,
            flat_pixel_payload=None,
            model_payload=captured.model_payload,
        )


def _default_microbatches():
    mb0 = _microbatch(
        [_item(0, 0, (1, 4, 4), 0), _item(0, 1, (1, 4, 8), 16), _item(2, 0, (2, 4, 4), 48)],
        total_rows=80,
    )
    mb1 = _microbatch([], total_rows=0)  # text-only
    return [mb0, mb1]


def test_capture_builds_descriptors_and_sidecar_on_endpoint():
    sampler = iter(range(10))
    window = MdpIterationWindow.capture(
        sampler,
        num_microbatches=2,
        adapter=_StubAdapter(_default_microbatches()),
        num_vpp_chunks=2,
        lane_id=3,
        my_worker_id=0,
        num_workers=1,
    )
    descriptors = window.descriptors()
    assert [d.global_item_id for d in descriptors] == [0, 1, 2]
    assert all(d.owner_dp_lane == 3 for d in descriptors)
    assert [d.microbatch_id for d in descriptors] == [0, 0, 0]
    assert descriptors[1].payload_rows == 32
    assert descriptors[1].output_rows == 8
    sidecar = window.payload_sidecar()
    assert set(sidecar) == {0, 1, 2}
    assert (sidecar[1] == 2.0).all() and sidecar[1].shape == (32, 4)
    # Exactly two sampler elements consumed: one per microbatch.
    assert next(sampler) == 2

    records = window.records()
    assert not records[0].text_only and records[1].text_only
    assert records[0].vision_items[2].global_item_id == 2
    assert records[0].vision_items[2].output_rows == 8


def test_non_endpoint_member_holds_records_and_owned_pixels():
    window = MdpIterationWindow.capture(
        iter(range(10)),
        num_microbatches=2,
        adapter=_StubAdapter(_default_microbatches()),
        num_vpp_chunks=1,
        lane_id=None,
        my_worker_id=0,
        num_workers=1,
    )
    assert window.descriptors() == ()
    assert set(window.payload_sidecar()) == {0, 1, 2}
    assert len(window.records()) == 2


@pytest.mark.parametrize(
    ("global_rank", "is_worker_leader", "expected_state", "expected_sidecar"),
    (
        (0, True, (False, True), {0, 1, 2}),
        (1, False, (True, False), set()),
        (2, True, (True, False), set()),
    ),
    ids=("owner-leader", "owner-follower", "non-owner-leader"),
)
def test_encoder_cp_pixel_capture_is_worker_leader_only(
    global_rank, is_worker_leader, expected_state, expected_sidecar
):
    """TP2/ECP2: only the selected logical worker's leader decodes pixels."""
    rank_map = build_rank_map(
        MdpRankSpec(world_size=4, tp=2, pp=2, cp=1, ep=1, encoder_cp=2)
    )
    view = rank_map.view(global_rank)
    adapter = _OwnershipAdapter(_default_microbatches()[:1])

    window = MdpIterationWindow.capture(
        iter(range(10)),
        num_microbatches=1,
        adapter=adapter,
        num_vpp_chunks=1,
        lane_id=view.lane_id,
        my_worker_id=view.my_worker_id,
        num_workers=len(view.worker_ids),
        data_loader_source_worker_ids=rank_map.data_loader_source_worker_ids(0),
        is_worker_leader=is_worker_leader,
    )

    assert adapter.capture_states == [expected_state]
    assert set(window.payload_sidecar()) == expected_sidecar


def test_tp2_ecp3_misalignment_never_selects_a_tp1_worker_leader():
    rank_map = build_rank_map(
        MdpRankSpec(world_size=6, tp=2, pp=3, cp=1, ep=1, encoder_cp=3)
    )
    source_worker_ids = rank_map.data_loader_source_worker_ids(0)
    assert source_worker_ids == (0,)

    observations = []
    for global_rank in (0, 2, 3, 4):
        view = rank_map.view(global_rank)
        is_worker_leader = global_rank == rank_map.worker_leader_rank(
            view.outer_dp_rank, view.my_worker_id
        )
        adapter = _OwnershipAdapter(_default_microbatches()[:1])
        window = MdpIterationWindow.capture(
            iter(range(10)),
            num_microbatches=1,
            adapter=adapter,
            num_vpp_chunks=1,
            lane_id=view.lane_id,
            my_worker_id=view.my_worker_id,
            num_workers=len(view.worker_ids),
            data_loader_source_worker_ids=source_worker_ids,
            is_worker_leader=is_worker_leader,
        )
        observations.append(
            (
                global_rank,
                view.my_worker_id,
                is_worker_leader,
                adapter.capture_states[0],
                set(window.payload_sidecar()),
            )
        )

    assert observations == [
        (0, 0, True, (False, True), {0, 1, 2}),
        (2, 0, False, (True, False), set()),
        (3, 1, True, (True, False), set()),
        (4, 1, False, (True, False), set()),
    ]


def test_vpp_replay_contract():
    window = MdpIterationWindow.capture(
        iter(range(10)),
        num_microbatches=2,
        adapter=_StubAdapter(_default_microbatches()),
        num_vpp_chunks=3,
        lane_id=0,
        my_worker_id=0,
        num_workers=1,
    )
    cursors = window.replay_iterators()
    assert len(cursors) == 3
    # Every cursor replays the same records, independently.
    first = [next(cursors[0]) for _ in range(2)]
    second = [next(cursors[1]) for _ in range(2)]
    assert first == second == list(window.records())
    with pytest.raises(MdpStateError, match="overrun"):
        next(cursors[0])
    with pytest.raises(MdpStateError, match="once per capture"):
        window.replay_iterators()


def test_release_pixels_clears_the_sidecar_only():
    window = MdpIterationWindow.capture(
        iter(range(10)),
        num_microbatches=2,
        adapter=_StubAdapter(_default_microbatches()),
        num_vpp_chunks=1,
        lane_id=0,
        my_worker_id=0,
        num_workers=1,
    )
    window.release_pixels()
    assert window.payload_sidecar() == {}
    assert len(window.records()) == 2  # replay records unaffected


@pytest.mark.parametrize(
    "mutate, match",
    [
        # decoder_positions shorter than output_rows
        (
            lambda: [_microbatch([_item(0, 0, (1, 4, 4), 0, positions=(1, 2))], 16)],
            "decoder_positions",
        ),
        # items without pixels
        (
            lambda: [
                CapturedMicrobatch(
                    decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
                    vision_items=(_item(0, 0, (1, 4, 4), 0),),
                    flat_pixel_payload=None,
                    model_payload=MappingProxyType({}),
                )
            ],
            "both exist or both",
        ),
        # grid not divisible by merge size
        (lambda: [_microbatch([_item(0, 0, (1, 3, 4), 0)], 12)], "divisible"),
        # duplicate (sample, ordinal)
        (
            lambda: [
                _microbatch([_item(0, 0, (1, 4, 4), 0), _item(0, 0, (1, 4, 4), 16)], 32)
            ],
            "without duplicates",
        ),
        # payload interval out of bounds
        (lambda: [_microbatch([_item(0, 0, (1, 4, 4), 8)], 16)], "inside"),
        # wrong qkv_format
        (
            lambda: [
                CapturedMicrobatch(
                    decoder_packed_seq_params=SimpleNamespace(qkv_format="bshd"),
                    vision_items=(),
                    flat_pixel_payload=None,
                    model_payload=MappingProxyType({}),
                )
            ],
            "thd",
        ),
    ],
)
def test_capture_validation(mutate, match):
    with pytest.raises(MdpConfigurationError, match=match):
        MdpIterationWindow.capture(
            iter(range(10)),
            num_microbatches=1,
            adapter=_StubAdapter(mutate()),
            num_vpp_chunks=1,
            lane_id=0,
            my_worker_id=0,
            num_workers=1,
        )


def test_exhausted_iterator_raises():
    with pytest.raises(MdpStateError, match="exhausted"):
        MdpIterationWindow.capture(
            iter(range(10)),
            num_microbatches=3,
            adapter=_StubAdapter(_default_microbatches()),
            num_vpp_chunks=1,
            lane_id=0,
            my_worker_id=0,
            num_workers=1,
        )
