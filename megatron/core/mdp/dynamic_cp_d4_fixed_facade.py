# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private fixed-CP4 repeated-D4 iteration composition."""

from collections.abc import Callable, Mapping
from typing import Any

import torch.distributed as dist

from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as _gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_execution as _execution
from megatron.core.mdp import dynamic_cp_d4_encoder_forward as _forward
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as _gradient
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient_finalize as _gate6
from megatron.core.mdp import dynamic_cp_d4_encoder_iteration_commit as _gate7
from megatron.core.mdp import dynamic_cp_d4_encoder_selected_backward as _gate5
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as _replay
from megatron.core.mdp import dynamic_cp_d4_native_schedule as _native
from megatron.core.mdp import dynamic_cp_d4_transaction as _transaction
from megatron.core.mdp.dynamic_cp_d4_authority_construction import (
    build_repeated_d4_joint_iteration_authority,
)
from megatron.core.mdp.dynamic_cp_d4_source_catalog import _gather_d4_source_catalog
from megatron.core.mdp.errors import MdpConfigurationError

__all__ = ()


def _begin(owner: _transaction._D4TransactionOwner, begin: Callable[[], Any]):
    try:
        return begin()
    except BaseException as error:
        owner.abort(error)
        raise


def _advance(owner, begin, adapter):
    lease = _begin(owner, begin)
    try:
        successor = adapter()
    except BaseException as error:
        lease.fail(error)
        raise
    return lease.adopt(successor)


def _scheduled_abort(owner: _replay._D4FixedDecoderReplayOwner, error: BaseException) -> None:
    owner.abort(error)


def _run_repeated_d4_fixed_iteration(
    runtime: Any,
    capture_owner: Any,
    *,
    decoder_max_seqlen_per_rank: int,
    decoder_solver: Any,
    encoder_max_seqlen_per_rank: int,
    encoder_minimum_cp_size: int,
    encoder_workload_query: Callable[..., Any],
    bridge_width: int,
    bridge_dtype: Any,
    rebuild_microbatch: Callable[..., Any],
    cp_partition_mode: str,
    forward_backward_func: Callable[..., Any],
    native_schedule_args: tuple[Any, ...] = (),
    native_schedule_kwargs: Mapping[str, Any] | None = None,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
    broadcast: Callable[..., Any] = dist.broadcast,
    byte_generator: Callable[[int], Any] | None = None,
) -> None:
    """Run one joint-authority iteration with a fixed native decoder CP4 schedule."""
    if type(native_schedule_args) is not tuple or (
        native_schedule_kwargs is not None and not isinstance(native_schedule_kwargs, Mapping)
    ):
        raise MdpConfigurationError("MDP: fixed facade native arguments are immutable and mapped.")
    schedule_kwargs = dict(native_schedule_kwargs or {})
    transaction = _transaction._begin_d4_transaction(capture_owner)
    binding = transaction.binding
    try:
        projection = _gather_d4_source_catalog(
            capture_owner, binding, expected_dynamic_decoder_cp=False
        )
        transaction.attach_source_catalog(projection)
        authority = build_repeated_d4_joint_iteration_authority(
            binding,
            projection.metadata,
            decoder_max_seqlen_per_rank=decoder_max_seqlen_per_rank,
            decoder_minimum_cp_size=4,
            decoder_solver=decoder_solver,
            encoder_max_seqlen_per_rank=encoder_max_seqlen_per_rank,
            encoder_minimum_cp_size=encoder_minimum_cp_size,
            encoder_workload_query=encoder_workload_query,
            bridge_width=bridge_width,
            bridge_dtype=bridge_dtype,
            locator_catalog_digest=projection.local_locator_digest,
        )
        transaction.attach_authority(authority)
        _replay._fixed_assignments(authority, binding)
    except BaseException as error:
        transaction.abort(error)
        raise

    if projection.local_locator_catalog is None:
        execution_adapter = lambda: _execution.claim_d4_encoder_execution(capture_owner, authority)
    else:
        execution_adapter = lambda: _execution.claim_d4_locator_encoder_execution(
            capture_owner, authority, projection.local_locator_catalog
        )
    execution = _advance(transaction, transaction.begin_execution, execution_adapter)
    forward = _advance(
        transaction,
        transaction.begin_forward,
        lambda: _forward.run_repeated_d4_encoder_forward(
            runtime,
            execution,
            authority,
            all_to_all_single=all_to_all_single,
            broadcast=broadcast,
            byte_generator=byte_generator,
        ),
    )
    publication = _advance(
        transaction,
        transaction.begin_publication,
        lambda: _forward.run_repeated_d4_encoder_publication(
            forward, all_to_all_single=all_to_all_single, byte_generator=byte_generator
        ),
    )
    replay = _advance(
        transaction,
        transaction.begin_replay,
        lambda: _replay._run_repeated_d4_fixed_decoder_replay_from_publication(
            publication,
            authority,
            rebuild_microbatch=rebuild_microbatch,
            cp_partition_mode=cp_partition_mode,
            byte_generator=byte_generator,
        ),
    )
    native = _advance(
        transaction,
        transaction.begin_native_schedule,
        lambda: _native._run_d4_native_schedule(
            forward_backward_func,
            replay,
            _scheduled_abort,
            *native_schedule_args,
            **schedule_kwargs,
        ),
    )
    gradient = _advance(
        transaction,
        transaction.begin_gradient,
        lambda: _gradient._run_repeated_d4_encoder_gradient_from_replay(
            replay,
            authority,
            native.completion,
            all_to_all_single=all_to_all_single,
            byte_generator=byte_generator,
        ),
    )
    authorized = _advance(
        transaction,
        transaction.begin_backward_authorization,
        lambda: _gate4.run_repeated_d4_encoder_backward_authorization(
            gradient, authority, native.completion, byte_generator=byte_generator
        ),
    )
    backward = _advance(
        transaction,
        transaction.begin_backward,
        lambda: _gate5.run_repeated_d4_encoder_selected_backward(
            authorized, authority, native.completion, byte_generator=byte_generator
        ),
    )
    ready = _advance(
        transaction,
        transaction.begin_finalize,
        lambda: _gate6.run_repeated_d4_encoder_gradient_finalize(
            backward, authority, native.completion, byte_generator=byte_generator
        ),
    )

    lease = _begin(transaction, transaction.begin_commit)
    try:
        arguments = lease.arguments()
        result = _gate7.run_repeated_d4_encoder_iteration_commit(
            *arguments, byte_generator=byte_generator
        )
    except BaseException as error:
        lease.fail(error)
        raise
    lease.complete(result)
    return None
