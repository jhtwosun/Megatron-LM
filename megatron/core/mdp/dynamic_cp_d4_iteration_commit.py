# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 Gate-7 cleanup and iteration commit."""

import hashlib
import struct
from collections.abc import Callable
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_iteration_commit import (
    _D3IterationCommitReady,
    _execute_d3_iteration_commit,
)
from megatron.core.mdp.dynamic_cp_d3_iteration_commit import (
    _validate_ready as _validate_d3_iteration_commit_ready,
)
from megatron.core.mdp.dynamic_cp_d3_workspace import _DynamicIterationWorkspace
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _candidate_digest,
    _snapshot_local_authority,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_execution import _require_digest
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority, _DynamicProducerCarrier
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError, MdpTaskFatalError

__all__ = ()

_TERMINAL_DIGEST_PERSON = b"mcore-mdp-d4-t"
_TERMINAL_DIGEST_VERSION = 1
_INT64_MAX = 2**63 - 1


def _terminal_commit_digest(
    *,
    global_manifest_digest: bytes,
    plan_digest: bytes,
    iteration: int,
    world_ranks: tuple[int, ...],
) -> bytes:
    """Bind terminal cleanup to deterministic iteration and WORLD authority."""
    manifest = _require_digest("terminal global manifest digest", global_manifest_digest)
    plan = _require_digest("terminal plan digest", plan_digest)
    if type(iteration) is not int or not 0 <= iteration <= _INT64_MAX:
        raise MdpConfigurationError("MDP: terminal commit iteration is a signed-int64 integer.")
    if (
        type(world_ranks) is not tuple
        or not world_ranks
        or any(type(rank) is not int or not 0 <= rank <= _INT64_MAX for rank in world_ranks)
        or len(set(world_ranks)) != len(world_ranks)
    ):
        raise MdpConfigurationError(
            "MDP: terminal commit WORLD ranks are a non-empty unique integer tuple."
        )
    digest = hashlib.blake2b(digest_size=16, person=_TERMINAL_DIGEST_PERSON)
    digest.update(struct.pack("<3q", _TERMINAL_DIGEST_VERSION, iteration, len(world_ranks)))
    digest.update(struct.pack(f"<{len(world_ranks)}q", *world_ranks))
    digest.update(manifest)
    digest.update(plan)
    return digest.digest()


def _candidate_terminal_digest(binding: Any, authority: Any, commit_ready: Any) -> bytes | None:
    """Read untrusted terminal lineage without letting one rank skip WORLD."""
    try:
        return _terminal_commit_digest(
            global_manifest_digest=authority.global_manifest.digest,
            plan_digest=authority.plan.digest,
            iteration=commit_ready.iteration,
            world_ranks=binding.world_ranks,
        )
    except BaseException:
        return None


def _add_cleanup_note(primary_error: BaseException, secondary_error: BaseException) -> None:
    try:
        primary_error.add_note(
            f"suppressed repeated-D4 terminal cleanup error: {secondary_error!r}"
        )
    except BaseException:
        pass


def run_repeated_d4_iteration_commit(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    producer: _DynamicProducerCarrier,
    commit_ready: _D3IterationCommitReady,
    byte_generator: Callable[[int], Any] | None = None,
) -> None:
    """Clean one exact workspace and commit only after repeated-D4 Gate 7."""
    kwargs = {}
    if byte_generator is not None:
        kwargs["byte_generator"] = byte_generator
    prepare_started = False
    cleanup_started = False
    commit_started = False
    retained_commit: _D3IterationCommitReady | None = None
    commit_result = object()

    def cleanup_once() -> None:
        nonlocal cleanup_started
        if cleanup_started:
            raise MdpStateError("MDP: repeated-D4 Gate 7 cleans its producer exactly once.")
        cleanup_started = True
        if (
            type(authority) is not _DynamicIterationAuthority
            or type(workspace_owner) is not _D3WorkspaceBindingOwner
            or type(producer) is not _DynamicProducerCarrier
        ):
            raise MdpConfigurationError(
                "MDP: repeated-D4 Gate 7 cleanup uses exact private inputs."
            )
        workspace_owner.cleanup_bound_producer(authority, producer)
        if workspace_owner.is_idle is not True:
            raise MdpStateError("MDP: repeated-D4 Gate 7 cleanup leaves its workspace idle.")

    def cleanup_after_failure(primary_error: BaseException) -> None:
        if cleanup_started:
            return
        try:
            cleanup_once()
        except BaseException as cleanup_error:
            _add_cleanup_note(primary_error, cleanup_error)

    def prepare(manifest_digest: Any, terminal_digest: Any) -> _D3IterationCommitReady:
        nonlocal prepare_started, retained_commit
        if prepare_started:
            raise MdpStateError("MDP: repeated-D4 Gate 7 prepares terminal cleanup exactly once.")
        prepare_started = True
        snapshot = _snapshot_local_authority(binding, authority)
        if (
            type(binding) is not _RepeatedD4GroupBinding
            or type(authority) is not _DynamicIterationAuthority
            or type(workspace_owner) is not _D3WorkspaceBindingOwner
            or type(producer) is not _DynamicProducerCarrier
            or type(commit_ready) is not _D3IterationCommitReady
        ):
            raise MdpConfigurationError("MDP: repeated-D4 Gate 7 uses exact private inputs.")
        exact_producer = workspace_owner.require_bound_producer(authority, producer)
        workspace = workspace_owner.require_workspace(authority)
        if (
            exact_producer is not producer
            or type(workspace) is not _DynamicIterationWorkspace
            or workspace.authority is not authority
            or workspace.rank != binding.global_rank
            or producer.rank_view.global_rank != binding.global_rank
            or producer.payload_destination_views is not workspace.payload_views
            or producer.embedding_destination_views is not workspace.embedding_views
            or producer.gradient_destination_views is not workspace.gradient_views
            or producer.summed_gradient_destination_views is not workspace.summed_gradient_views
        ):
            raise MdpStateError("MDP: repeated-D4 Gate 7 retains its exact workspace and producer.")
        ready = _validate_d3_iteration_commit_ready(commit_ready)
        if ready is not commit_ready:
            raise MdpStateError("MDP: repeated-D4 Gate 7 retains its exact commit capability.")
        expected_digest = _terminal_commit_digest(
            global_manifest_digest=snapshot.global_manifest.digest,
            plan_digest=snapshot.plan.digest,
            iteration=ready.iteration,
            world_ranks=binding.world_ranks,
        )
        if manifest_digest != snapshot.global_manifest.digest or terminal_digest != expected_digest:
            raise MdpStateError("MDP: repeated-D4 Gate 7 retains exact terminal authority.")
        retained_commit = ready
        cleanup_once()
        return ready

    def commit(value: Any) -> object:
        nonlocal commit_started
        if commit_started:
            raise MdpTaskFatalError("MDP: repeated-D4 Gate 7 enters commit exactly once.")
        commit_started = True
        if (
            type(value) is not _D3IterationCommitReady
            or value is not retained_commit
            or value is not commit_ready
        ):
            raise MdpTaskFatalError(
                "MDP: repeated-D4 Gate 7 executes the exact retained commit capability."
            )
        try:
            _execute_d3_iteration_commit(value)
        except BaseException as error:
            if type(error) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: repeated-D4 iteration commit failed after repeated-D4 status."
            ) from error
        return commit_result

    try:
        runner = binding.begin_attempt(**kwargs)
        manifest_digest = _candidate_digest(authority, "global_manifest")
        terminal_digest = _candidate_terminal_digest(binding, authority, commit_ready)
        result = runner.run(
            global_manifest_digest=manifest_digest,
            plan_digest=terminal_digest,
            gate_id=7,
            prepare=lambda: prepare(manifest_digest, terminal_digest),
            domain_collective=commit,
        )
    except BaseException as error:
        cleanup_after_failure(error)
        raise
    if result is not commit_result or not commit_started:
        error = MdpTaskFatalError(
            "MDP: repeated-D4 Gate 7 retains the exact post-WORLD commit result."
        )
        cleanup_after_failure(error)
        raise error
