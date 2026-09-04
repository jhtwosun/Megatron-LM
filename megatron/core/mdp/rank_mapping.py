# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP rank mapping: outer-DP planning groups and logical encoder workers.

Pure-compute module. :func:`build_rank_map` derives every set strictly from
Megatron-LM ``RankGenerator`` coordinates: it must not call ``torch.distributed``,
create process groups, query local devices, read rank-local environment
variables, or mutate global MPU state. Ranks belonging to a group or a pipeline
stage must never be assumed contiguous.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from megatron.core.mdp.config import SUPPORTED_RANK_ORDER
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.parallel_state import RankGenerator


@dataclass(frozen=True)
class MdpRankSpec:
    """Parallel dimensions the rank map is derived from."""

    world_size: int
    tp: int
    pp: int
    cp: int
    ep: int
    encoder_cp: int
    rank_order: str = SUPPORTED_RANK_ORDER


@dataclass(frozen=True)
class MdpRankView:
    """The slice of the rank map one rank needs; no global rank lists are copied."""

    global_rank: int
    outer_dp_rank: int
    lane_id: Optional[int]
    my_worker_id: Optional[int]
    endpoint_rank: int
    planning_group_ranks: tuple
    worker_ids: tuple
    decoder_endpoint_id: Optional[int] = None


class MdpRankMap:
    """Outer-DP planning groups and logical-worker resolution, built once and shared.

    A planning group is one outer-DP group: all model-parallel ranks with one fixed
    decoder ``dp_rank``. Inside a group, the ``TP x CP x PP`` encoder ranks form
    ``inner_dp / encoder_cp`` logical workers with stable ids ``0..num_workers-1``;
    ``worker_ranks()`` is the only resolution point from logical workers to
    physical ranks.
    """

    def __init__(
        self,
        spec: MdpRankSpec,
        planning_groups: Sequence[tuple],
        tp_groups: Sequence[tuple],
        decoder_endpoint_groups: Sequence[tuple],
    ):
        self._spec = spec
        self._groups = tuple(planning_groups)
        self._decoder_endpoints = tuple(decoder_endpoint_groups)
        if len(self._decoder_endpoints) != len(self._groups):
            raise MdpConfigurationError(
                "MDP: decoder endpoint groups must map one-to-one to planning groups."
            )
        for planning_group, endpoints in zip(self._groups, self._decoder_endpoints):
            if len(endpoints) != spec.cp or not set(endpoints).issubset(planning_group):
                raise MdpConfigurationError(
                    f"MDP: decoder endpoints {endpoints} violate: exactly cp={spec.cp} "
                    f"PP0/TP0 ranks inside planning group {planning_group}."
                )
        self._tp_group_by_rank = {}
        for group in tp_groups:
            group = tuple(group)
            if len(group) != spec.tp:
                raise MdpConfigurationError(
                    f"MDP: TP group {group} violates: group size == tp={spec.tp}."
                )
            for rank in group:
                if rank in self._tp_group_by_rank:
                    raise MdpConfigurationError(
                        f"MDP: rank {rank} appears in more than one TP group."
                    )
                self._tp_group_by_rank[rank] = group

        self._rank_to_coord = {}
        for outer_dp_rank, group in enumerate(self._groups):
            for index_in_group, rank in enumerate(group):
                if rank in self._rank_to_coord:
                    raise MdpConfigurationError(
                        f"MDP: rank {rank} appears in more than one planning group; "
                        "outer-DP groups must form a disjoint partition of WORLD."
                    )
                self._rank_to_coord[rank] = (outer_dp_rank, index_in_group)
        covered = len(self._rank_to_coord)
        if covered != spec.world_size:
            raise MdpConfigurationError(
                f"MDP: planning groups cover {covered} ranks but world_size is "
                f"{spec.world_size}; every rank must belong to exactly one group."
            )
        if len(self._tp_group_by_rank) != spec.world_size:
            raise MdpConfigurationError(
                f"MDP: TP groups cover {len(self._tp_group_by_rank)} ranks but "
                f"world_size is {spec.world_size}."
            )

    @property
    def spec(self) -> MdpRankSpec:
        """The spec this map was built from."""
        return self._spec

    @property
    def num_workers_per_group(self) -> int:
        """Logical encoder workers per planning group."""
        inner_dp = self._spec.tp * self._spec.cp * self._spec.pp
        return inner_dp // self._spec.encoder_cp

    def planning_groups(self) -> Sequence[tuple]:
        """All outer-DP planning groups in ascending ``outer_dp_rank`` order."""
        return self._groups

    def endpoint_rank(self, outer_dp_rank: int) -> int:
        """The canonical PP0/CP0 endpoint rank of one planning group."""
        return self._decoder_endpoints[outer_dp_rank][0]

    def decoder_endpoint_ranks(self, outer_dp_rank: int) -> tuple:
        """The PP0/TP0 ranks, ordered by decoder context-parallel rank."""
        return self._decoder_endpoints[outer_dp_rank]

    def tp_group_ranks(self, global_rank: int) -> tuple:
        """The native decoder TP group containing ``global_rank``."""
        try:
            return self._tp_group_by_rank[global_rank]
        except KeyError as error:
            raise MdpConfigurationError(
                f"MDP: global_rank={global_rank} violates: rank belongs to a TP "
                f"group of world_size={self._spec.world_size}."
            ) from error

    def data_loader_source_worker_ids(self, outer_dp_rank: int) -> tuple:
        """Logical workers containing a native TP0 DataLoader source rank."""
        sources = []
        for worker_id in range(self.num_workers_per_group):
            worker_ranks = self.worker_ranks(outer_dp_rank, worker_id)
            if any(rank == self.tp_group_ranks(rank)[0] for rank in worker_ranks):
                sources.append(worker_id)
        return tuple(sources)

    def worker_ranks(self, outer_dp_rank: int, worker_id: int) -> tuple:
        """Resolve one logical worker to its physical ranks.

        With ``encoder_cp=1`` this always returns a one-element tuple. With
        ``encoder_cp>1`` one logical worker maps to ``encoder_cp`` ranks; the plan
        is unchanged and only the bridge's physical expansion differs.
        """
        num_workers = self.num_workers_per_group
        if not 0 <= worker_id < num_workers:
            raise MdpConfigurationError(
                f"MDP: worker_id={worker_id} violates: 0 <= worker_id < {num_workers}."
            )
        group = self._groups[outer_dp_rank]
        encoder_cp = self._spec.encoder_cp
        return tuple(group[worker_id * encoder_cp : (worker_id + 1) * encoder_cp])

    def view(self, global_rank: int) -> MdpRankView:
        """The local view for one rank; stores only what that rank needs."""
        if global_rank not in self._rank_to_coord:
            raise MdpConfigurationError(
                f"MDP: global_rank={global_rank} violates: rank belongs to a "
                f"planning group of world_size={self._spec.world_size}."
            )
        outer_dp_rank, index_in_group = self._rank_to_coord[global_rank]
        group = self._groups[outer_dp_rank]
        endpoints = self.decoder_endpoint_ranks(outer_dp_rank)
        endpoint = endpoints[0]
        decoder_endpoint_id = endpoints.index(global_rank) if global_rank in endpoints else None
        return MdpRankView(
            global_rank=global_rank,
            outer_dp_rank=outer_dp_rank,
            lane_id=outer_dp_rank if global_rank == endpoint else None,
            my_worker_id=index_in_group // self._spec.encoder_cp,
            endpoint_rank=endpoint,
            planning_group_ranks=group,
            worker_ids=tuple(range(self.num_workers_per_group)),
            decoder_endpoint_id=decoder_endpoint_id,
        )


def endpoint_worker_id(view: MdpRankView) -> int:
    """The logical worker hosting the group's endpoint rank (worker 0 today).

    Derived purely from the view, mirroring the planner's own derivation:
    workers partition the planning-group ranks in fixed-width blocks.
    """
    ranks_per_worker = len(view.planning_group_ranks) // len(view.worker_ids)
    return view.planning_group_ranks.index(view.endpoint_rank) // ranks_per_worker


def build_rank_map(spec: MdpRankSpec) -> MdpRankMap:
    """Build the rank map from ``RankGenerator`` coordinates. Pure compute.

    Planning groups come from ``RankGenerator.get_ranks('tp-cp-pp')``. Decoder
    endpoints are derived independently as the ordered CP group whose members are
    both native TP-primary ranks and native PP-primary ranks; no planning-group
    positional stride is treated as topology.
    """
    if spec.rank_order != SUPPORTED_RANK_ORDER:
        raise MdpConfigurationError(
            f"MDP: rank_order={spec.rank_order!r} violates: rank_order == "
            f"'{SUPPORTED_RANK_ORDER}'. Other orders are not validated."
        )
    for name in ("world_size", "tp", "pp", "cp", "ep", "encoder_cp"):
        value = getattr(spec, name)
        if value < 1:
            raise MdpConfigurationError(
                f"MDP: {name}={value} violates: {name} >= 1."
            )
    model_parallel = spec.tp * spec.pp * spec.cp
    if spec.world_size % model_parallel != 0:
        raise MdpConfigurationError(
            f"MDP: world_size={spec.world_size} violates: world_size % "
            f"(TP * PP * CP) == 0 with TP * PP * CP = {model_parallel}."
        )
    inner_dp = spec.tp * spec.cp * spec.pp
    if inner_dp % spec.encoder_cp != 0:
        raise MdpConfigurationError(
            f"MDP: encoder_cp={spec.encoder_cp} violates: encoder_cp divides "
            f"physical encoder ranks per planning group = TP * CP * PP = {inner_dp}."
        )

    decoder_dp = spec.world_size // model_parallel
    # The default RankGenerator carries ep=1; EP lives only in the expert generator
    # and does not participate in the decoder data-parallel decomposition.
    generator = RankGenerator(
        tp=spec.tp, ep=1, dp=decoder_dp, pp=spec.pp, cp=spec.cp, order=spec.rank_order
    )
    planning_groups = [tuple(group) for group in generator.get_ranks("tp-cp-pp")]
    if len(planning_groups) != decoder_dp:
        raise MdpConfigurationError(
            f"MDP: RankGenerator produced {len(planning_groups)} outer-DP groups; "
            f"expected decoder_dp={decoder_dp}."
        )
    tp_groups = [tuple(group) for group in generator.get_ranks("tp")]
    tp_primary_ranks = {group[0] for group in tp_groups}
    pp_primary_ranks = {group[0] for group in generator.get_ranks("pp")}
    cp_groups = [tuple(group) for group in generator.get_ranks("cp")]
    decoder_endpoint_groups = []
    for planning_group in planning_groups:
        endpoint_set = set(planning_group) & tp_primary_ranks & pp_primary_ranks
        matches = [group for group in cp_groups if set(group) == endpoint_set]
        if len(matches) != 1:
            raise MdpConfigurationError(
                f"MDP: planning group {planning_group} has {len(matches)} native "
                "PP0/TP0 decoder endpoint CP groups; expected exactly one."
            )
        decoder_endpoint_groups.append(matches[0])
    return MdpRankMap(spec, planning_groups, tp_groups, decoder_endpoint_groups)
