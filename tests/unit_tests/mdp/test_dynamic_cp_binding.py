# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Rank-local native-group binding tests for decoder Dynamic-CP plans."""

from dataclasses import FrozenInstanceError, replace

import pytest

import megatron.core.mdp.dynamic_cp_execution as execution
from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.dynamic_cp_execution import DecoderMicrobatchKey
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata, build_decoder_dynamic_plan
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError

_INT64_MAX = 2**63 - 1
_RANKS = (17, 3, 21, 8)


def _sample(order, padded_seqlen):
    return DecoderSampleMetadata(
        sample_id=GlobalSampleId(0, order),
        valid_seqlen=padded_seqlen,
        padded_seqlen=padded_seqlen,
        vision_items=(),
    )


class _TwoWaveSolver:
    """Emit CP4 first, then CP2 + CP1 + CP1 in the same native pool."""

    def __init__(self):
        self.calls = 0

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        assert total_gpus == 4
        assert max_seq_len_per_rank == 4
        assert min_cp_size == 1
        self.calls += 1
        if self.calls == 1:
            assert sample_seqlens == [(0, 16), (1, 8), (2, 2), (3, 3)]
            return ([[16], [16], [16], [16]], [(1, 8), (2, 2), (3, 3)], None, [[0], [0], [0], [0]])
        assert sample_seqlens == [(1, 8), (2, 2), (3, 3)]
        return ([[8], [8], [2], [3]], [], None, [[1], [1], [2], [3]])


def _plan():
    return build_decoder_dynamic_plan(
        tuple(_sample(index, padded) for index, padded in enumerate((16, 8, 2, 3))),
        decoder_ranks=_RANKS,
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=_TwoWaveSolver(),
    )


class _OpaqueGroup:
    def __init__(
        self,
        ranks,
        global_rank,
        *,
        size_value=None,
        local_rank=None,
        size_error=None,
        rank_error=None,
    ):
        self.ranks = tuple(ranks)
        self.global_rank = global_rank
        self.size_value = len(self.ranks) if size_value is None else size_value
        self.local_rank = self.ranks.index(global_rank) if local_rank is None else local_rank
        self.size_error = size_error
        self.rank_error = rank_error

    def size(self):
        if self.size_error is not None:
            raise self.size_error
        return self.size_value

    def rank(self):
        if self.rank_error is not None:
            raise self.rank_error
        return self.local_rank

    def __eq__(self, other):
        raise AssertionError("opaque process-group equality must not be observed")

    def __repr__(self):
        raise AssertionError("opaque process-group repr must not be observed")


class _BrokenSizeAttribute:
    def __init__(self, error):
        self.error = error

    @property
    def size(self):
        raise self.error


class _BrokenRankAttribute(_OpaqueGroup):
    def __init__(self, ranks, global_rank, *, error):
        super().__init__(ranks, global_rank)
        self.error = error

    @property
    def rank(self):
        raise self.error


def _assignment_for(plan, key, global_rank):
    return next(
        assignment
        for assignment in plan.microbatches[key.microbatch_index].assignments
        if global_rank in assignment.endpoint_ranks
    )


def _bind(*args, **kwargs):
    return execution.bind_local_decoder_assignment(*args, **kwargs)


@pytest.mark.parametrize(
    "microbatch_index,global_rank,expected_size",
    ((0, 3, 4), (0, 8, 4), (1, 3, 2), (1, 21, 1)),
    ids=("cp4-middle", "cp4-tail", "cp2", "cp1"),
)
def test_local_assignment_binds_exact_native_group(microbatch_index, global_rank, expected_size):
    plan = _plan()
    key = DecoderMicrobatchKey(microbatch_index)
    assignment = _assignment_for(plan, key, global_rank)
    group = _OpaqueGroup(assignment.endpoint_ranks, global_rank)
    group_calls = []
    rank_calls = []

    def group_getter(*, group_size):
        group_calls.append(group_size)
        return group

    def group_ranks_getter(value):
        rank_calls.append(value)
        return list(value.ranks)

    local = _bind(
        plan,
        key=key,
        global_rank=global_rank,
        maximum_group_ranks=_RANKS,
        group_getter=group_getter,
        group_ranks_getter=group_ranks_getter,
    )

    assert isinstance(local, execution.LocalDecoderAssignment)
    assert local.key is key
    assert local.assignment is assignment
    assert local.cp_group is group
    assert assignment.local_cp_size == expected_size
    assert group_calls == [expected_size]
    assert rank_calls == [group]
    assert "cp_group" not in repr(local)
    assert replace(local, cp_group=object()) == local
    with pytest.raises(FrozenInstanceError):
        local.cp_group = object()


@pytest.mark.parametrize("invalid_plan", (object(), "stale"), ids=("carrier", "digest"))
def test_binding_validates_plan_before_injected_getters(invalid_plan):
    if invalid_plan == "stale":
        invalid_plan = replace(_plan(), digest=b"stale-plan-hash!")
    calls = []

    def group_getter(*, group_size):
        calls.append(("group", group_size))

    def group_ranks_getter(group):
        calls.append(("ranks", group))

    with pytest.raises(MdpPlanError):
        _bind(
            invalid_plan,
            key=DecoderMicrobatchKey(0),
            global_rank=17,
            maximum_group_ranks=_RANKS,
            group_getter=group_getter,
            group_ranks_getter=group_ranks_getter,
        )
    assert calls == []


@pytest.mark.parametrize(
    "key,match",
    (
        (0, "DecoderMicrobatchKey"),
        (object(), "DecoderMicrobatchKey"),
        (DecoderMicrobatchKey(2), "outside"),
    ),
    ids=("integer", "carrier", "out-of-range"),
)
def test_binding_rejects_invalid_key_before_lookup(key, match):
    calls = []

    with pytest.raises(MdpConfigurationError, match=match):
        _bind(
            _plan(),
            key=key,
            global_rank=17,
            maximum_group_ranks=_RANKS,
            group_getter=lambda **kwargs: calls.append(kwargs),
            group_ranks_getter=lambda group: calls.append(group),
        )
    assert calls == []


@pytest.mark.parametrize(
    "global_rank,match",
    (
        (True, "signed-int64"),
        (3.0, "signed-int64"),
        (-1, "signed-int64"),
        (_INT64_MAX + 1, "signed-int64"),
        (99, "one local assignment"),
    ),
)
def test_binding_rejects_invalid_or_nonlocal_rank_before_lookup(global_rank, match):
    calls = []

    with pytest.raises(MdpConfigurationError, match=match):
        _bind(
            _plan(),
            key=DecoderMicrobatchKey(0),
            global_rank=global_rank,
            maximum_group_ranks=_RANKS,
            group_getter=lambda **kwargs: calls.append(kwargs),
            group_ranks_getter=lambda group: calls.append(group),
        )
    assert calls == []


@pytest.mark.parametrize(
    "maximum_group_ranks,match",
    (
        (list(_RANKS), "immutable tuple"),
        ((), "non-empty"),
        ((_RANKS[1], _RANKS[0], *_RANKS[2:]), "exact ordered"),
        ((_RANKS[0], _RANKS[0], *_RANKS[2:]), "unique"),
        ((_RANKS[0], True, *_RANKS[2:]), "signed-int64"),
    ),
)
def test_binding_rejects_invalid_maximum_rank_authority_before_lookup(maximum_group_ranks, match):
    calls = []

    with pytest.raises(MdpConfigurationError, match=match):
        _bind(
            _plan(),
            key=DecoderMicrobatchKey(0),
            global_rank=17,
            maximum_group_ranks=maximum_group_ranks,
            group_getter=lambda **kwargs: calls.append(kwargs),
            group_ranks_getter=lambda group: calls.append(group),
        )
    assert calls == []


@pytest.mark.parametrize(
    "native_ranks,match",
    (
        ((3, 17), "exact ordered"),
        ((17, 21), "exact ordered"),
        ((17, 17), "unique"),
        ((17, True), "signed-int64"),
        ((17, 3.0), "signed-int64"),
    ),
)
def test_binding_rejects_invalid_native_ordered_ranks(native_ranks, match):
    plan = _plan()
    key = DecoderMicrobatchKey(1)
    group = _OpaqueGroup((17, 3), global_rank=3)

    with pytest.raises(MdpConfigurationError, match=match):
        _bind(
            plan,
            key=key,
            global_rank=3,
            maximum_group_ranks=_RANKS,
            group_getter=lambda **kwargs: group,
            group_ranks_getter=lambda value: list(native_ranks),
        )


@pytest.mark.parametrize(
    "size_value,local_rank,match",
    (
        (1, 1, "group size"),
        (True, 1, "group size"),
        (2.0, 1, "group size"),
        (2, 0, "local rank"),
        (2, True, "signed-int64"),
        (2, 1.0, "signed-int64"),
        (2, -1, "signed-int64"),
        (2, 2, "local rank"),
    ),
)
def test_binding_rejects_native_group_geometry(size_value, local_rank, match):
    group = _OpaqueGroup((17, 3), 3, size_value=size_value, local_rank=local_rank)

    with pytest.raises(MdpConfigurationError, match=match):
        _bind(
            _plan(),
            key=DecoderMicrobatchKey(1),
            global_rank=3,
            maximum_group_ranks=_RANKS,
            group_getter=lambda **kwargs: group,
            group_ranks_getter=lambda value: list(value.ranks),
        )


@pytest.mark.parametrize(
    "seam",
    ("group-getter", "size-getattr", "size-call", "ranks-getter", "rank-getattr", "rank-call"),
)
def test_binding_normalizes_ordinary_native_query_failures(seam):
    group = _OpaqueGroup((17, 3), 3)
    group_error = RuntimeError(seam)
    group_getter = lambda **kwargs: group
    group_ranks_getter = lambda value: list(value.ranks)
    if seam == "group-getter":
        group_getter = lambda **kwargs: (_ for _ in ()).throw(group_error)
    elif seam == "size-getattr":
        group = _BrokenSizeAttribute(group_error)
    elif seam == "size-call":
        group = _OpaqueGroup((17, 3), 3, size_error=group_error)
    elif seam == "ranks-getter":
        group_ranks_getter = lambda value: (_ for _ in ()).throw(group_error)
    elif seam == "rank-getattr":
        group = _BrokenRankAttribute((17, 3), 3, error=group_error)
    else:
        group = _OpaqueGroup((17, 3), 3, rank_error=group_error)

    with pytest.raises(MdpConfigurationError, match="query failed") as caught:
        _bind(
            _plan(),
            key=DecoderMicrobatchKey(1),
            global_rank=3,
            maximum_group_ranks=_RANKS,
            group_getter=group_getter,
            group_ranks_getter=group_ranks_getter,
        )
    assert caught.value.__cause__ is group_error


@pytest.mark.parametrize("missing", ("group", "ranks"))
def test_binding_rejects_noncallable_getters_before_query(missing):
    group_getter = None if missing == "group" else lambda **kwargs: object()
    ranks_getter = None if missing == "ranks" else lambda group: ()

    with pytest.raises(MdpConfigurationError, match="callable"):
        _bind(
            _plan(),
            key=DecoderMicrobatchKey(0),
            global_rank=17,
            maximum_group_ranks=_RANKS,
            group_getter=group_getter,
            group_ranks_getter=ranks_getter,
        )


def test_binding_does_not_intercept_base_exceptions():
    interrupt = KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt) as caught:
        _bind(
            _plan(),
            key=DecoderMicrobatchKey(0),
            global_rank=17,
            maximum_group_ranks=_RANKS,
            group_getter=lambda **kwargs: (_ for _ in ()).throw(interrupt),
            group_ranks_getter=lambda group: (),
        )
    assert caught.value is interrupt


def test_binding_rejects_missing_native_group_handle():
    with pytest.raises(MdpConfigurationError, match="native Dynamic-CP group"):
        _bind(
            _plan(),
            key=DecoderMicrobatchKey(1),
            global_rank=3,
            maximum_group_ranks=_RANKS,
            group_getter=lambda **kwargs: None,
            group_ranks_getter=lambda group: (),
        )
