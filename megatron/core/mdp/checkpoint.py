# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP checkpoint facade: synchronous global torch_dist save/load.

Logical keys: ``language_model.*`` stays with the decoder chunks (PP/VPP
shards, decoder DP-CP replica metadata, produced by the native checkpoint
path); ``vision_model.*`` comes from the encoder DDP with **encoder WORLD**
replica metadata — one logical copy replicated on every rank. Plans, leaves,
forward handles, autograd graphs, and communication handles are never
persisted.

For static MDP and decoder-only Dynamic-CP, optimizer, LR-scheduler, and RNG
state round-trip through the native paths, so a resume is exact rather than
weight-only; the composite optimizer keeps the two sharding domains apart with
a fixed encoder key (see :mod:`megatron.core.mdp.optimizer`). Repeated-D4
dynamic encoder CP is provisionally weight-only and therefore requires all
native optimizer and RNG save/load paths to be disabled explicitly. What
remains rejected for every MDP checkpoint are the *execution modes* that cannot
work here: the fully-parallel save/load wrappers (they reshard every child over
a single DP-CP group, which is wrong for the encoder's WORLD domain),
asynchronous and non-persistent saves, and constant-structure caching (MDP
rebuilds its plan-derived structures every iteration).
"""

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from megatron.core.mdp import integration
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
from megatron.core.mdp.errors import MdpCheckpointError
from megatron.core.mdp.runtime import MdpRuntimeState

#: The state-dict key the encoder state travels under. It sits next to the
#: native ``model``/``model<N>`` keys so the sharded save/load skeleton is
#: symmetric between save and load.
ENCODER_STATE_KEY = "mdp_vision_model"

#: The logical prefix the encoder weights are published under, matching the
#: keys a native (non-MDP) multimodal checkpoint carries.
ENCODER_STATE_PREFIX = "vision_model."

#: ``DistributedDataParallel`` contributes one ``module.`` level below the
#: logical prefix, and ``DDP.load_state_dict`` delegates straight to that
#: child — so both levels come off before the state is handed back.
_DDP_CHILD_PREFIX = "module."

#: TransformerEngine's per-module opaque state. It appears in ``state_dict()``
#: but not necessarily in the sharded state dict, and holds no weights. This
#: mirrors the producing side's ``extra_state_suffix`` default in
#: :func:`megatron.core.transformer.utils.make_sharded_tensors_for_checkpoint`;
#: keep the two in step if core ever renames it.
_EXTRA_STATE_SUFFIX = "_extra_state"

_CHECKPOINT_STATUS_VERSION = 1
_CHECKPOINT_STATUS_WIDTH = 7
_OPERATION_CODES = {"save": 1, "load": 2}
_PHASE_CODES = {"pre-state": 1, "pre-io": 2, "post-decode": 3}
_LEGAL_BOUNDARIES = {("save", "pre-state"), ("load", "pre-io"), ("load", "post-decode")}
_SIGNED_INT64_MAX = (1 << 63) - 1
_CHECKPOINT_TOPOLOGY_FIELDS = (
    "world_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "mdp_encoder_cp",
)


def _checkpoint_topology_digest(binding: _RepeatedD4GroupBinding) -> bytes:
    authority = _validate_repeated_d4_group_binding(binding)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"megatron.mdp.repeated_d4.checkpoint.topology.v1")
    hasher.update(struct.pack("<q", len(authority.world_ranks)))
    hasher.update(struct.pack(f"<{len(authority.world_ranks)}q", *authority.world_ranks))
    hasher.update(struct.pack("<qq", 4, authority.expert_parallel_size))
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class _RepeatedD4CheckpointStatus:
    global_rank: int
    operation: str
    phase: str
    topology_digest: bytes
    lifecycle_generation: int
    iteration: int
    error_code: int

    def __post_init__(self) -> None:
        if (
            type(self.global_rank) is not int
            or self.global_rank < 0
            or type(self.operation) is not str
            or type(self.phase) is not str
            or (self.operation, self.phase) not in _LEGAL_BOUNDARIES
            or type(self.topology_digest) is not bytes
            or len(self.topology_digest) != 16
            or type(self.lifecycle_generation) is not int
            or not 0 <= self.lifecycle_generation <= _SIGNED_INT64_MAX
            or type(self.iteration) is not int
            or not 0 <= self.iteration <= _SIGNED_INT64_MAX
            or type(self.error_code) is not int
            or self.error_code not in (0, 1)
        ):
            raise MdpCheckpointError("MDP: checkpoint WORLD status has exact bounded fields.")

    def to_wire_tuple(self) -> tuple[int, ...]:
        operation = _OPERATION_CODES[self.operation]
        phase = _PHASE_CODES[self.phase]
        control = (_CHECKPOINT_STATUS_VERSION << 16) | (operation << 8) | phase
        digest_words = struct.unpack("<qq", self.topology_digest)
        return (
            self.global_rank,
            *digest_words,
            self.lifecycle_generation,
            self.iteration,
            self.error_code,
            control,
        )

    @classmethod
    def from_wire_tuple(cls, value: Any) -> "_RepeatedD4CheckpointStatus":
        if type(value) is not tuple or len(value) != _CHECKPOINT_STATUS_WIDTH:
            raise MdpCheckpointError(
                f"MDP: checkpoint WORLD status wire has width {_CHECKPOINT_STATUS_WIDTH}."
            )
        if any(type(word) is not int for word in value):
            raise MdpCheckpointError("MDP: checkpoint WORLD status wire uses exact integers.")
        rank, digest_0, digest_1, generation, iteration, error_code, control = value
        version = (control >> 16) & 0xFF
        operation_code = (control >> 8) & 0xFF
        phase_code = control & 0xFF
        operations = {code: name for name, code in _OPERATION_CODES.items()}
        phases = {code: name for name, code in _PHASE_CODES.items()}
        if (
            control < 0
            or version != _CHECKPOINT_STATUS_VERSION
            or operation_code not in operations
            or phase_code not in phases
            or control != (_CHECKPOINT_STATUS_VERSION << 16) | (operation_code << 8) | phase_code
        ):
            raise MdpCheckpointError("MDP: checkpoint WORLD status has a canonical control word.")
        return cls(
            global_rank=rank,
            operation=operations[operation_code],
            phase=phases[phase_code],
            topology_digest=struct.pack("<qq", digest_0, digest_1),
            lifecycle_generation=generation,
            iteration=iteration,
            error_code=error_code,
        )


def _converge_repeated_d4_checkpoint_boundary(
    runtime,
    snapshot,
    *,
    operation: str,
    phase: str,
    iteration: int,
    local_error: BaseException | None,
    boundary_digest: bytes | None = None,
) -> None:
    """Converge one checkpoint boundary on the lifecycle-bound WORLD group."""
    binding = getattr(runtime, "dynamic_group_binding", None)
    authority = _validate_repeated_d4_group_binding(binding)
    topology_digest = _checkpoint_topology_digest(binding)
    gather = authority._status_gather_factory(
        group=authority._world_group,
        group_ranks=authority.world_ranks,
        global_rank=authority.global_rank,
        device=authority._device,
        group_ranks_getter=authority._group_ranks_getter,
    )
    effective_error = local_error
    try:
        if local_error is not None and not isinstance(local_error, BaseException):
            raise MdpCheckpointError("MDP: checkpoint local error is an exception or None.")
        if type(iteration) is not int or not 0 <= iteration <= _SIGNED_INT64_MAX:
            raise MdpCheckpointError("MDP: checkpoint iteration is an exact signed-int64 value.")
        if boundary_digest is not None:
            if type(boundary_digest) is not bytes or len(boundary_digest) != 16:
                raise MdpCheckpointError("MDP: checkpoint boundary digest is exactly 16 bytes.")
            topology_digest = boundary_digest
        required_snapshot = snapshot.require()
        status = _RepeatedD4CheckpointStatus(
            authority.global_rank,
            operation,
            phase,
            topology_digest,
            required_snapshot.generation,
            iteration,
            int(local_error is not None),
        )
    except BaseException as error:
        effective_error = error
        status = _RepeatedD4CheckpointStatus(
            authority.global_rank, "save", "pre-state", topology_digest, 0, 0, 1
        )
    try:
        gathered = gather(status.to_wire_tuple(), timeout_seconds=authority._timeout_seconds)
    except BaseException as error:
        raise MdpCheckpointError("MDP: checkpoint WORLD status gather failed.") from error
    if type(gathered) is not tuple or len(gathered) != len(authority.world_ranks):
        raise MdpCheckpointError("MDP: checkpoint WORLD returns one ordered row per rank.")
    parsed = []
    for expected_rank, wire in zip(authority.world_ranks, gathered):
        try:
            row = _RepeatedD4CheckpointStatus.from_wire_tuple(wire)
        except MdpCheckpointError as error:
            raise MdpCheckpointError(
                f"MDP: checkpoint WORLD row for rank {expected_rank} is malformed."
            ) from error
        if row.global_rank != expected_rank:
            raise MdpCheckpointError(
                f"MDP: checkpoint WORLD expected rank {expected_rank}, got {row.global_rank}."
            )
        parsed.append(row)
    local_index = authority.world_ranks.index(authority.global_rank)
    if parsed[local_index] != status:
        raise MdpCheckpointError("MDP: checkpoint WORLD preserves the exact local row.")
    reference = parsed[0]
    for row in parsed:
        if row.error_code:
            raise MdpCheckpointError(
                f"MDP: checkpoint WORLD rejected rank {row.global_rank}."
            ) from effective_error
        if (
            row.operation != reference.operation
            or row.phase != reference.phase
            or row.topology_digest != reference.topology_digest
            or row.lifecycle_generation != reference.lifecycle_generation
            or row.iteration != reference.iteration
        ):
            raise MdpCheckpointError(
                f"MDP: checkpoint WORLD boundary mismatch at rank {row.global_rank}."
            )
    if effective_error is not None:
        raise MdpCheckpointError(
            "MDP: checkpoint WORLD accepted a local error."
        ) from effective_error


def _checkpoint_boundary_state(args, *, operation: str, phase: str, iteration: int):
    runtime = integration.get_runtime()
    snapshot = None
    local_error = None
    try:
        snapshot = integration.get_d4_checkpoint_lifecycle_snapshot()
        if runtime is None or snapshot is None:
            raise MdpCheckpointError("MDP: repeated-D4 checkpoint requires its runtime lifecycle.")
        if runtime.state is not MdpRuntimeState.EMPTY:
            raise MdpCheckpointError("MDP: repeated-D4 checkpoint runtime must be EMPTY.")
        if operation == "save" and snapshot.may_save is not True:
            raise MdpCheckpointError("MDP: save pre-state requires a committed idle iteration.")
        if operation == "load" and snapshot.may_load is not True:
            raise MdpCheckpointError("MDP: load pre-io requires a fresh initial idle process.")
    except BaseException as error:
        local_error = error
    _converge_repeated_d4_checkpoint_boundary(
        runtime,
        snapshot,
        operation=operation,
        phase=phase,
        iteration=iteration,
        local_error=local_error,
    )
    return runtime, snapshot


def prepare_repeated_d4_checkpoint_save(args, *, iteration: int) -> None:
    if getattr(args, "mdp_dynamic_encoder_cp", None) is not True:
        return
    _checkpoint_boundary_state(args, operation="save", phase="pre-state", iteration=iteration)


def prepare_repeated_d4_checkpoint_load(args) -> None:
    if getattr(args, "mdp_dynamic_encoder_cp", None) is not True:
        return
    _checkpoint_boundary_state(args, operation="load", phase="pre-io", iteration=0)


def _decoded_checkpoint_boundary_digest(state_dict, runtime) -> bytes:
    binding = getattr(runtime, "dynamic_group_binding", None)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"megatron.mdp.repeated_d4.checkpoint.decoded_header.v1")
    hasher.update(_checkpoint_topology_digest(binding))
    if state_dict is None:
        hasher.update(b"\0")
        return hasher.digest()
    hasher.update(b"\1")
    version = state_dict["checkpoint_version"]
    if type(version) is int:
        encoded_version = str(version).encode("ascii")
        hasher.update(b"i")
        hasher.update(struct.pack("<q", len(encoded_version)))
        hasher.update(encoded_version)
    else:
        hasher.update(b"f")
        hasher.update(struct.pack("<d", version))
    checkpoint_args = state_dict["args"]
    for name in _CHECKPOINT_TOPOLOGY_FIELDS:
        encoded = str(getattr(checkpoint_args, name)).encode("ascii")
        hasher.update(struct.pack("<q", len(encoded)))
        hasher.update(encoded)
    return hasher.digest()


def _validate_native_checkpoint_arguments(args, checkpoint_args, version) -> None:
    """Mirror the comparisons in training.checkpointing.check_checkpoint_args."""

    def compare(name, *, checkpoint_name=None, default=None):
        source_name = name if checkpoint_name is None else checkpoint_name
        try:
            expected = getattr(args, name)
            actual = (
                getattr(checkpoint_args, source_name, default)
                if default is not None
                else getattr(checkpoint_args, source_name)
            )
            matches = actual == expected
        except BaseException as error:
            raise MdpCheckpointError(
                f"MDP: decoded checkpoint argument {name!r} is present and comparable."
            ) from error
        if type(matches) is not bool or not matches:
            raise MdpCheckpointError(f"MDP: decoded checkpoint argument {name!r} matches exactly.")

    compare("num_layers")
    compare("hidden_size")
    compare("num_attention_heads")
    compare("add_position_embedding", default=True)
    if getattr(args, "vocab_file"):
        compare("max_position_embeddings")
        compare("make_vocab_size_divisible_by")
        if not getattr(args, "use_dist_ckpt"):
            compare("padded_vocab_size")
        compare("tokenizer_type")
    if getattr(args, "data_parallel_random_init"):
        compare("data_parallel_random_init")
    if getattr(args, "phase_transition_iterations"):
        compare("global_batch_size")
    if version < 3.0:
        compare("tensor_model_parallel_size", checkpoint_name="model_parallel_size")
    if version >= 3.0 and not getattr(args, "use_dist_ckpt"):
        compare("tensor_model_parallel_size")
        compare("pipeline_model_parallel_size")


def _validate_repeated_d4_decoded_checkpoint_content(
    args, state_dict, runtime
) -> tuple[int, bytes]:
    if state_dict is None:
        return 0, _decoded_checkpoint_boundary_digest(state_dict, runtime)
    if not isinstance(state_dict, Mapping):
        raise MdpCheckpointError("MDP: repeated-D4 load requires decoded checkpoint content.")
    iteration = state_dict.get("iteration")
    if type(iteration) is not int or not 0 <= iteration <= _SIGNED_INT64_MAX:
        raise MdpCheckpointError(
            "MDP: decoded checkpoint iteration is an exact signed-int64 value."
        )
    version = state_dict.get("checkpoint_version")
    if type(version) not in (int, float) or (type(version) is float and not math.isfinite(version)):
        raise MdpCheckpointError("MDP: decoded checkpoint version is an exact finite number.")
    if version not in (0, 1.0) and version < 2.0:
        raise MdpCheckpointError("MDP: decoded checkpoint version is legacy 0/1.0 or at least 2.0.")
    checkpoint_args = state_dict.get("args")
    if checkpoint_args is None:
        raise MdpCheckpointError("MDP: decoded checkpoint contains its native args header.")
    for name in _CHECKPOINT_TOPOLOGY_FIELDS:
        expected = getattr(args, name, None)
        actual = getattr(checkpoint_args, name, None)
        if type(expected) is not int or type(actual) is not int or actual != expected:
            raise MdpCheckpointError(
                f"MDP: decoded checkpoint topology field {name!r} matches exactly."
            )
    _validate_native_checkpoint_arguments(args, checkpoint_args, version)
    model_keys = [
        key for key in state_dict if key == "model" or key.startswith("model") and key[5:].isdigit()
    ]
    if not model_keys or ("model" in model_keys and len(model_keys) != 1):
        raise MdpCheckpointError("MDP: decoded checkpoint has canonical model/modelN keys.")
    has_language_weight = False
    for key in model_keys:
        value = state_dict[key]
        if not isinstance(value, Mapping):
            raise MdpCheckpointError("MDP: decoded checkpoint model shard is a mapping.")
        for parameter_name in value:
            if type(parameter_name) is not str or not parameter_name.startswith("language_model."):
                raise MdpCheckpointError(
                    "MDP: decoded checkpoint model keys retain language_model.*."
                )
            has_language_weight = True
    if not has_language_weight:
        raise MdpCheckpointError("MDP: decoded checkpoint contains language_model weights.")
    encoder_state = state_dict.get(ENCODER_STATE_KEY)
    if not isinstance(encoder_state, Mapping):
        raise MdpCheckpointError("MDP: decoded checkpoint contains mapped vision weights.")
    prefix = ENCODER_STATE_PREFIX + _DDP_CHILD_PREFIX
    if any(type(key) is not str or not key.startswith(prefix) for key in encoder_state):
        raise MdpCheckpointError("MDP: decoded checkpoint vision keys retain vision_model.*.")
    expected_encoder = {
        prefix + key
        for key in runtime.encoder_domain.encoder_ddp.state_dict()
        if not key.endswith(_EXTRA_STATE_SUFFIX)
    }
    missing = expected_encoder - set(encoder_state)
    if missing:
        raise MdpCheckpointError(
            f"MDP: decoded checkpoint is missing {len(missing)} vision weight(s)."
        )
    return iteration, _decoded_checkpoint_boundary_digest(state_dict, runtime)


def validate_repeated_d4_decoded_checkpoint(args, state_dict) -> None:
    if getattr(args, "mdp_dynamic_encoder_cp", None) is not True:
        return
    runtime = integration.get_runtime()
    snapshot = None
    local_error = None
    iteration = 0
    boundary_digest = None
    try:
        snapshot = integration.get_d4_checkpoint_lifecycle_snapshot()
        if runtime is None or snapshot is None or snapshot.may_load is not True:
            raise MdpCheckpointError("MDP: post-decode load requires its initial lifecycle.")
        if runtime.state is not MdpRuntimeState.EMPTY:
            raise MdpCheckpointError("MDP: post-decode load runtime must remain EMPTY.")
        iteration, boundary_digest = _validate_repeated_d4_decoded_checkpoint_content(
            args, state_dict, runtime
        )
    except BaseException as error:
        local_error = error
    _converge_repeated_d4_checkpoint_boundary(
        runtime,
        snapshot,
        operation="load",
        phase="post-decode",
        iteration=iteration,
        local_error=local_error,
        boundary_digest=boundary_digest,
    )


def encoder_sharded_state_dict(encoder_ddp) -> Mapping:
    """The encoder's sharded model-weight state with WORLD replica metadata.

    The encoder is fully replicated: its replica domain is WORLD, not the
    decoder's DP-CP group — reusing the decoder metadata here would make
    every PP stage claim a distinct (wrong) replica coordinate.
    """
    return encoder_ddp.sharded_state_dict(
        prefix=ENCODER_STATE_PREFIX, metadata={"dp_cp_group": torch.distributed.group.WORLD}
    )


def add_encoder_state(state_dict: dict, encoder_ddp) -> dict:
    """Add the encoder weights to a torch_dist checkpoint state dict."""
    if ENCODER_STATE_KEY in state_dict:
        raise MdpCheckpointError(
            f"MDP: state dict already contains {ENCODER_STATE_KEY!r}; the encoder "
            "state must be contributed exactly once."
        )
    state_dict[ENCODER_STATE_KEY] = encoder_sharded_state_dict(encoder_ddp)
    return state_dict


def load_encoder_state(state_dict: Mapping, encoder_ddp, *, strict: bool = True) -> None:
    """Restore the encoder weights from a loaded torch_dist checkpoint.

    ``generate_state_dict`` builds both the save state and the load skeleton,
    so ``dist_checkpointing.load`` has already read the encoder tensors back by
    the time this runs. Nothing else copies them into the encoder module — the
    encoder lives outside the decoder model-chunk list that
    ``load_checkpoint`` iterates — so this is the missing half of the round
    trip.
    """
    if ENCODER_STATE_KEY not in state_dict:
        raise MdpCheckpointError(
            f"MDP: the checkpoint has no {ENCODER_STATE_KEY!r} entry; it was not "
            "written by an MDP run and carries no vision-encoder weights."
        )
    prefix = ENCODER_STATE_PREFIX + _DDP_CHILD_PREFIX
    inner = {}
    for key, value in state_dict[ENCODER_STATE_KEY].items():
        if not key.startswith(prefix):
            raise MdpCheckpointError(
                f"MDP: encoder state key {key!r} does not start with {prefix!r}; the "
                "encoder save and load skeletons have drifted apart."
            )
        inner[key[len(prefix) :]] = value

    # ``strict`` arrives from ``load_checkpoint``'s own parameter, and
    # ``load_state_dict(strict=False)`` merely reports missing keys -- and
    # ``_BaseDataParallel.load_state_dict`` drops even that report, returning
    # ``None``. The encoder is replicated and always written whole, so there is
    # no "empty stage" case to tolerate here the way there is for a decoder
    # chunk: an absent key means the state did not round-trip, and skipping it
    # would resume from the random initialization. Check before delegating.
    #
    # TransformerEngine's ``_extra_state`` entries are exempt: they are present
    # in ``state_dict()`` but the sharded state dict legitimately omits the ones
    # whose modules contribute no persistent extra state, and they carry no
    # weights, so their absence cannot leave a randomly initialized encoder.
    expected = {key for key in encoder_ddp.state_dict() if not key.endswith(_EXTRA_STATE_SUFFIX)}
    missing = sorted(expected - set(inner))
    if missing:
        raise MdpCheckpointError(
            f"MDP: the checkpoint is missing {len(missing)} encoder tensor(s), "
            f"first {missing[:5]}; the encoder would stay randomly initialized. "
            "A non-strict --dist-ckpt-strictness drops the keys the checkpoint "
            "cannot supply, which is how they get here."
        )

    # The check above already enforces the property that matters, so a strict
    # failure below can only come from an ``_extra_state`` or unexpected-key
    # mismatch -- never from an absent weight. Retry non-strictly for those, the
    # way ``load_model_state_dict`` does for every decoder chunk in
    # ``megatron.training.checkpointing``: TransformerEngine changes which
    # ``_extra_state`` entries it publishes between versions, and the encoder
    # must not be stricter about that than the decoder it trains beside.
    try:
        encoder_ddp.load_state_dict(inner, strict=strict)
    except RuntimeError:
        if not strict:
            raise
        encoder_ddp.load_state_dict(inner, strict=False)


def assert_supported_checkpoint_config(args) -> None:
    """Reject checkpoint configurations MDP cannot honor, at startup.

    Static MDP and decoder-only Dynamic-CP retain full-state resume. Repeated-D4
    dynamic encoder CP currently accepts weight-only checkpoints, in addition
    to the execution-mode restrictions shared by every MDP checkpoint (see the
    module docstring). The caller has already validated and enabled MDP.
    """
    problems = []
    save_or_load = (
        getattr(args, "save", None) is not None or getattr(args, "load", None) is not None
    )
    if save_or_load:
        # Design doc section 12: only the synchronous, persistent, global
        # torch_dist mode is supported. Asynchronous, non-persistent, and
        # constant-structure caching modes are rejected at startup (scoped to
        # save/load so checkpoint-free runs are unaffected by defaults).
        if getattr(args, "async_save", False):
            problems.append("no --async-save (asynchronous save is unsupported)")
        if getattr(args, "non_persistent_ckpt_type", None) is not None:
            problems.append(
                "no --non-persistent-ckpt-type (non-persistent checkpoints are " "unsupported)"
            )
        if getattr(args, "ckpt_assume_constant_structure", False):
            problems.append(
                "no --ckpt-assume-constant-structure (MDP's plan-derived "
                "structures change per iteration; a cached structure goes stale)"
            )
        if getattr(args, "mdp_dynamic_encoder_cp", None) is True:
            required_weight_only_flags = (
                ("no_save_optim", "--no-save-optim"),
                ("no_load_optim", "--no-load-optim"),
                ("no_save_rng", "--no-save-rng"),
                ("no_load_rng", "--no-load-rng"),
            )
            missing = [
                flag
                for attribute, flag in required_weight_only_flags
                if getattr(args, attribute, None) is not True
            ]
            if missing:
                problems.append(
                    "repeated-D4 dynamic encoder CP is provisionally weight-only; require "
                    + " ".join(missing)
                )
    if getattr(args, "save", None) is not None:
        # Megatron defaults ckpt_fully_parallel_save=True; the fully-parallel
        # path shards across one DP-CP group for every child, which is wrong
        # for the encoder's WORLD replica domain. Scoped to save/load so runs
        # that never touch a checkpoint are not rejected by the default.
        if getattr(args, "ckpt_fully_parallel_save", False):
            problems.append("--no-ckpt-fully-parallel-save")
    if getattr(args, "load", None) is not None:
        if getattr(args, "ckpt_fully_parallel_load", False):
            problems.append("--no-ckpt-fully-parallel-load (or omit --ckpt-fully-parallel-load)")
    if problems:
        raise MdpCheckpointError(
            "MDP: unsupported checkpoint configuration; run with " + " ".join(problems) + "."
        )
