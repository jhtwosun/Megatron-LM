# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure contracts for MDP decoder and encoder Dynamic-CP."""

import pytest

from megatron.core.mdp.dynamic_cp import (
    DynamicCpGroupMembership,
    DynamicCpGroupSpec,
    GlobalSampleId,
    GlobalVisionItemId,
    dynamic_cp_group_sizes,
    lookup_decoder_dynamic_cp_group,
    member_dynamic_cp_group_specs,
    nested_dynamic_cp_group_specs,
    select_dynamic_cp_group,
)
from megatron.core.mdp.errors import MdpConfigurationError


def test_global_ids_are_stable_and_source_lane_qualified():
    assert GlobalSampleId(source_dp_lane=2, local_sample_order=7) == GlobalSampleId(2, 7)
    assert GlobalSampleId(2, 7) != GlobalSampleId(3, 7)
    assert GlobalVisionItemId(source_dp_lane=2, local_item_id=11) == GlobalVisionItemId(2, 11)
    assert GlobalVisionItemId(2, 11) != GlobalVisionItemId(3, 11)
    assert GlobalSampleId(2, 7).to_wire_tuple() == (2, 7)
    assert GlobalVisionItemId(2, 11).to_wire_tuple() == (2, 11)
    assert GlobalSampleId.from_wire_tuple((2, 7)) == GlobalSampleId(2, 7)
    assert GlobalVisionItemId.from_wire_tuple((2, 11)) == GlobalVisionItemId(2, 11)


@pytest.mark.parametrize(
    "identity_type,components",
    [
        (GlobalSampleId, (-1, 0)),
        (GlobalSampleId, (0, -1)),
        (GlobalVisionItemId, (-1, 0)),
        (GlobalVisionItemId, (0, -1)),
        (GlobalSampleId, (True, 0)),
        (GlobalSampleId, (0, 1.5)),
        (GlobalVisionItemId, (False, 0)),
        (GlobalVisionItemId, (0, "1")),
    ],
)
def test_global_ids_reject_negative_components(identity_type, components):
    with pytest.raises(MdpConfigurationError, match="non-negative integer"):
        identity_type(*components)


@pytest.mark.parametrize(
    "identity_type,wire_value",
    [
        (GlobalSampleId, [0, 1]),
        (GlobalSampleId, (0,)),
        (GlobalSampleId, (0, 1, 2)),
        (GlobalSampleId, (0, True)),
        (GlobalVisionItemId, [0, 1]),
        (GlobalVisionItemId, (0,)),
        (GlobalVisionItemId, (0, 1, 2)),
        (GlobalVisionItemId, (0, 1.0)),
    ],
)
def test_global_id_wire_tuple_rejects_invalid_shape_and_types(identity_type, wire_value):
    with pytest.raises(MdpConfigurationError, match="2-int tuple"):
        identity_type.from_wire_tuple(wire_value)


@pytest.mark.parametrize(
    "minimum,maximum,expected",
    [(1, 1, (1,)), (1, 2, (1, 2)), (1, 4, (1, 2, 4)), (2, 4, (2, 4)), (4, 4, (4,))],
)
def test_dynamic_cp_group_sizes(minimum, maximum, expected):
    assert dynamic_cp_group_sizes(minimum, maximum) == expected


@pytest.mark.parametrize(
    "minimum,maximum",
    [(0, 1), (1, 0), (3, 4), (1, 3), (4, 2), (True, 1), (1, False), (1.0, 2), (1, 2.0)],
)
def test_dynamic_cp_group_sizes_reject_invalid_ranges(minimum, maximum):
    with pytest.raises(MdpConfigurationError):
        dynamic_cp_group_sizes(minimum, maximum)


def test_nested_group_specs_follow_pool_order_without_contiguous_rank_assumptions():
    specs = nested_dynamic_cp_group_specs((9, 3, 12, 6), minimum_size=1)

    assert tuple((spec.group_size, spec.group_index, spec.ranks) for spec in specs) == (
        (1, 0, (9,)),
        (1, 1, (3,)),
        (1, 2, (12,)),
        (1, 3, (6,)),
        (2, 0, (9, 3)),
        (2, 1, (12, 6)),
        (4, 0, (9, 3, 12, 6)),
    )
    assert tuple(
        (spec.group_size, spec.ranks)
        for spec in member_dynamic_cp_group_specs(specs, global_rank=12)
    ) == ((1, (12,)), (2, (12, 6)), (4, (9, 3, 12, 6)))


@pytest.mark.parametrize(
    "spec_kwargs",
    [
        dict(group_size=3, group_index=0, ranks=(0, 1, 2)),
        dict(group_size=True, group_index=0, ranks=(0,)),
        dict(group_size=1, group_index=-1, ranks=(0,)),
        dict(group_size=1, group_index=True, ranks=(0,)),
        dict(group_size=2, group_index=0, ranks=(0,)),
        dict(group_size=2, group_index=0, ranks=(0, 0)),
        dict(group_size=2, group_index=0, ranks=(0, -1)),
        dict(group_size=2, group_index=0, ranks=(0, True)),
    ],
)
def test_dynamic_group_spec_rejects_invalid_fields(spec_kwargs):
    with pytest.raises(MdpConfigurationError):
        DynamicCpGroupSpec(**spec_kwargs)


def test_decoder_lookup_is_independent_of_unequal_encoder_group_sizes():
    encoder_sizes = dynamic_cp_group_sizes(1, 2)
    decoder_groups = {1: object(), 2: object(), 4: object()}
    calls = []

    def get_native_group(*, group_size):
        calls.append(group_size)
        return decoder_groups[group_size]

    selected = lookup_decoder_dynamic_cp_group(
        4, minimum_size=1, maximum_size=4, group_getter=get_native_group
    )

    assert encoder_sizes == (1, 2)
    assert selected is decoder_groups[4]
    assert calls == [4]


def test_decoder_lookup_rejects_unscheduled_size_before_native_lookup():
    calls = []

    with pytest.raises(MdpConfigurationError, match="scheduled group size"):
        lookup_decoder_dynamic_cp_group(
            3,
            minimum_size=1,
            maximum_size=4,
            group_getter=lambda *, group_size: calls.append(group_size),
        )

    assert calls == []


def test_decoder_lookup_rejects_trivial_maximum_before_native_lookup():
    calls = []

    with pytest.raises(MdpConfigurationError, match="maximum_size > 1"):
        lookup_decoder_dynamic_cp_group(
            1,
            minimum_size=1,
            maximum_size=1,
            group_getter=lambda *, group_size: calls.append(group_size),
        )

    assert calls == []


def test_decoder_lookup_wraps_missing_native_group():
    def missing_group(*, group_size):
        assert group_size == 2
        raise KeyError("missing")

    with pytest.raises(MdpConfigurationError, match="native Dynamic-CP group"):
        lookup_decoder_dynamic_cp_group(
            2, minimum_size=1, maximum_size=4, group_getter=missing_group
        )


def test_rank_local_group_lookup_is_exact_by_size():
    group_1 = object()
    group_2 = object()
    memberships = (
        DynamicCpGroupMembership(group_size=1, ranks=(5,), group=group_1),
        DynamicCpGroupMembership(group_size=2, ranks=(5, 1), group=group_2),
    )

    assert select_dynamic_cp_group(memberships, 1).group is group_1
    assert select_dynamic_cp_group(memberships, 2).ranks == (5, 1)
    with pytest.raises(MdpConfigurationError, match="available sizes"):
        select_dynamic_cp_group(memberships, 4)


@pytest.mark.parametrize(
    "membership_kwargs",
    [
        dict(group_size=3, ranks=(0, 1, 2), group=object()),
        dict(group_size=True, ranks=(0,), group=object()),
        dict(group_size=2, ranks=(0,), group=object()),
        dict(group_size=2, ranks=(0, 0), group=object()),
        dict(group_size=2, ranks=(0, -1), group=object()),
        dict(group_size=2, ranks=(0, False), group=object()),
        dict(group_size=1, ranks=(0,), group=None),
    ],
)
def test_dynamic_group_membership_rejects_invalid_fields(membership_kwargs):
    with pytest.raises(MdpConfigurationError):
        DynamicCpGroupMembership(**membership_kwargs)
