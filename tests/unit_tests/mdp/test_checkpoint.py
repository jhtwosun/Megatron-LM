# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint facade tests: torch_dist round trip of the encoder state with
WORLD replica metadata.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_checkpoint.py
"""

import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.mdp import checkpoint as checkpoint_api
from megatron.core.mdp import dynamic_cp_d4_group_binding as binding_api
from megatron.core.mdp import dynamic_cp_d4_transaction as transaction_api
from megatron.core.mdp import integration
from megatron.core.mdp.checkpoint import (
    ENCODER_STATE_KEY,
    add_encoder_state,
    assert_supported_checkpoint_config,
    load_encoder_state,
)
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.errors import MdpCheckpointError
from megatron.core.mdp.runtime import MdpRuntimeState

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_DISTRIBUTED = _WORLD_SIZE > 1

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


def test_exact_resume_flags_are_accepted():
    """Optimizer, LR-scheduler and RNG state round-trip, so the native
    `--no-*-optim`/`--no-*-rng` flags must no longer be demanded."""
    exact_resume = SimpleNamespace(
        save="/tmp/x",
        load="/tmp/x",
        no_save_optim=False,
        no_save_rng=False,
        no_load_optim=False,
        no_load_rng=False,
        ckpt_fully_parallel_save=False,
        ckpt_fully_parallel_load=False,
    )
    assert_supported_checkpoint_config(exact_resume)
    no_ckpt = SimpleNamespace(save=None, load=None)
    assert_supported_checkpoint_config(no_ckpt)


def _checkpoint_args(*, save="/tmp/x", load="/tmp/x", **overrides):
    values = dict(
        save=save,
        load=load,
        mdp_enable=True,
        mdp_dynamic_encoder_cp=True,
        dynamic_context_parallel=False,
        no_save_optim=True,
        no_load_optim=True,
        no_save_rng=True,
        no_load_rng=True,
        ckpt_fully_parallel_save=False,
        ckpt_fully_parallel_load=False,
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        add_position_embedding=True,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
        max_position_embeddings=16,
        make_vocab_size_divisible_by=8,
        padded_vocab_size=32,
        tokenizer_type="test-tokenizer",
        global_batch_size=4,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("save", "load"),
    (("/tmp/x", None), (None, "/tmp/x"), ("/tmp/x", "/tmp/x")),
    ids=("save-only", "load-only", "save-and-load"),
)
@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
def test_repeated_d4_accepts_only_explicit_weight_only_checkpoint_policy(
    save, load, dynamic_decoder
):
    args = _checkpoint_args(save=save, load=load, dynamic_context_parallel=dynamic_decoder)
    before = vars(args).copy()
    assert_supported_checkpoint_config(args)
    assert vars(args) == before


@pytest.mark.parametrize(
    ("field", "flag"),
    (
        ("no_save_optim", "--no-save-optim"),
        ("no_load_optim", "--no-load-optim"),
        ("no_save_rng", "--no-save-rng"),
        ("no_load_rng", "--no-load-rng"),
    ),
)
@pytest.mark.parametrize("value", (None, False, 0, 1, "yes"), ids=repr)
@pytest.mark.parametrize(
    ("save", "load"),
    (("/tmp/x", None), (None, "/tmp/x"), ("/tmp/x", "/tmp/x")),
    ids=("save-only", "load-only", "save-and-load"),
)
@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
def test_repeated_d4_rejects_non_exact_weight_only_flags(
    field, flag, value, save, load, dynamic_decoder
):
    args = _checkpoint_args(
        save=save, load=load, dynamic_context_parallel=dynamic_decoder, **{field: value}
    )
    before = vars(args).copy()
    with pytest.raises(MdpCheckpointError, match=flag):
        assert_supported_checkpoint_config(args)
    assert vars(args) == before


@pytest.mark.parametrize(
    ("field", "flag"),
    (
        ("no_save_optim", "--no-save-optim"),
        ("no_load_optim", "--no-load-optim"),
        ("no_save_rng", "--no-save-rng"),
        ("no_load_rng", "--no-load-rng"),
    ),
)
@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
def test_repeated_d4_rejects_absent_weight_only_flags(field, flag, dynamic_decoder):
    args = _checkpoint_args(dynamic_context_parallel=dynamic_decoder)
    delattr(args, field)
    with pytest.raises(MdpCheckpointError, match=flag):
        assert_supported_checkpoint_config(args)


@pytest.mark.parametrize("predicate", (None, False, 0, 1, "yes"), ids=repr)
def test_non_repeated_d4_checkpoint_modes_keep_exact_resume_compatibility(predicate):
    args = SimpleNamespace(
        save="/tmp/x",
        load="/tmp/x",
        mdp_enable=True,
        mdp_dynamic_encoder_cp=predicate,
        no_save_optim=False,
        no_load_optim=False,
        no_save_rng=False,
        no_load_rng=False,
        ckpt_fully_parallel_save=False,
        ckpt_fully_parallel_load=False,
    )
    assert_supported_checkpoint_config(args)


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("static-mdp", "decoder-only-d3"))
def test_static_mdp_and_decoder_only_d3_keep_full_state_resume(dynamic_decoder):
    # This helper is called only after _setup_mdp validates mdp_enable. The
    # policy discriminator is deliberately encoder D4, never decoder DCP.
    args = SimpleNamespace(
        save="/tmp/x",
        load="/tmp/x",
        mdp_enable=True,
        mdp_dynamic_encoder_cp=False,
        dynamic_context_parallel=dynamic_decoder,
        no_save_optim=False,
        no_load_optim=False,
        no_save_rng=False,
        no_load_rng=False,
        ckpt_fully_parallel_save=False,
        ckpt_fully_parallel_load=False,
    )
    assert_supported_checkpoint_config(args)


def test_checkpoint_free_repeated_d4_does_not_force_or_validate_weight_only_flags():
    args = SimpleNamespace(save=None, load=None, mdp_enable=True, mdp_dynamic_encoder_cp=True)
    before = vars(args).copy()
    assert_supported_checkpoint_config(args)
    assert vars(args) == before


def test_fully_parallel_modes_are_rejected():
    # Megatron defaults ckpt_fully_parallel_save=True: it must be rejected
    # when saving (the fully-parallel path shards over one DP-CP group for
    # every child, which is wrong for the encoder's WORLD replica domain).
    fully_parallel_save = SimpleNamespace(save="/tmp/x", load=None, ckpt_fully_parallel_save=True)
    with pytest.raises(MdpCheckpointError, match="fully-parallel-save"):
        assert_supported_checkpoint_config(fully_parallel_save)
    fully_parallel_load = SimpleNamespace(save=None, load="/tmp/x", ckpt_fully_parallel_load=True)
    with pytest.raises(MdpCheckpointError, match="fully-parallel-load"):
        assert_supported_checkpoint_config(fully_parallel_load)


def test_unsupported_checkpoint_execution_modes_rejected():
    # Design doc section 12: asynchronous, non-persistent, and constant-
    # structure caching modes must fail at startup when a save/load is
    # requested; checkpoint-free runs are unaffected.
    base = dict(save="/tmp/x", load=None, ckpt_fully_parallel_save=False)
    for field, match in (
        ("async_save", "async-save"),
        ("ckpt_assume_constant_structure", "constant-structure"),
    ):
        args = SimpleNamespace(**base)
        setattr(args, field, True)
        with pytest.raises(MdpCheckpointError, match=match):
            assert_supported_checkpoint_config(args)
    args = SimpleNamespace(**base, non_persistent_ckpt_type="global")
    with pytest.raises(MdpCheckpointError, match="non-persistent"):
        assert_supported_checkpoint_config(args)
    # The same flags are ignored when no checkpoint is requested.
    quiet = SimpleNamespace(
        save=None, load=None, async_save=True, ckpt_assume_constant_structure=True
    )
    assert_supported_checkpoint_config(quiet)


class _CheckpointRuntime:
    def __init__(self, binding):
        self.config = MdpConfig(enable=True, dynamic_encoder_cp=True)
        self.dynamic_group_binding = binding
        self.state = MdpRuntimeState.EMPTY
        self.encoder_domain = SimpleNamespace(
            encoder_ddp=SimpleNamespace(
                state_dict=lambda: {"layers.0.weight": object(), "layers.1.weight": object()}
            )
        )
        if _DISTRIBUTED:
            self.process_groups = SimpleNamespace(world_group=torch.distributed.group.WORLD)
            self.rank_view = SimpleNamespace(global_rank=torch.distributed.get_rank())
            self.device = torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def repeated_d4_checkpoint_runtime(monkeypatch):
    integration.reset_for_testing()
    binding = object()
    runtime = _CheckpointRuntime(binding)
    authority = SimpleNamespace(world_ranks=tuple(range(8)), expert_parallel_size=1)
    monkeypatch.setattr(
        transaction_api, "_validate_repeated_d4_group_binding", lambda _value: authority
    )
    monkeypatch.setattr(
        checkpoint_api, "_validate_repeated_d4_group_binding", lambda _value: authority
    )
    lifecycle = transaction_api._bind_d4_checkpoint_lifecycle(runtime, binding)
    monkeypatch.setattr(integration, "_RUNTIME", runtime)
    monkeypatch.setattr(integration, "_D4_CHECKPOINT_LIFECYCLE", lifecycle)
    yield runtime, binding, lifecycle
    integration.reset_for_testing()


def _checkpoint_world_gate(monkeypatch, calls):
    def converge(
        runtime, snapshot, *, operation, phase, iteration, local_error, **_boundary_metadata
    ):
        calls.append((runtime, snapshot, operation, phase, iteration, local_error))
        if local_error is not None:
            raise MdpCheckpointError(
                f"MDP: repeated-D4 WORLD rejected {operation} {phase}"
            ) from local_error

    monkeypatch.setattr(checkpoint_api, "_converge_repeated_d4_checkpoint_boundary", converge)


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
def test_repeated_d4_save_and_fresh_load_use_exact_lifecycle_boundary(
    monkeypatch, repeated_d4_checkpoint_runtime, dynamic_decoder
):
    runtime, binding, lifecycle = repeated_d4_checkpoint_runtime
    args = _checkpoint_args(dynamic_context_parallel=dynamic_decoder)
    calls = []
    _checkpoint_world_gate(monkeypatch, calls)

    checkpoint_api.prepare_repeated_d4_checkpoint_load(args)
    assert calls[-1][2:] == ("load", "pre-io", 0, None)
    with pytest.raises(MdpCheckpointError, match="save.*pre-state"):
        checkpoint_api.prepare_repeated_d4_checkpoint_save(args, iteration=11)
    assert isinstance(calls[-1][-1], BaseException)

    lifecycle.begin_iteration(runtime, binding)
    lifecycle.commit_iteration(runtime, binding)
    checkpoint_api.prepare_repeated_d4_checkpoint_save(args, iteration=11)
    assert calls[-1][2:] == ("save", "pre-state", 11, None)
    with pytest.raises(MdpCheckpointError, match="load.*pre-io"):
        checkpoint_api.prepare_repeated_d4_checkpoint_load(args)
    assert isinstance(calls[-1][-1], BaseException)


@pytest.mark.parametrize("operation", ("save", "load"))
@pytest.mark.parametrize("invalid_state", ("active", "poisoned"))
def test_checkpoint_boundary_converges_local_idle_validation_errors(
    monkeypatch, repeated_d4_checkpoint_runtime, invalid_state, operation
):
    runtime, binding, lifecycle = repeated_d4_checkpoint_runtime
    args = _checkpoint_args()
    lifecycle.begin_iteration(runtime, binding)
    if invalid_state == "poisoned":
        lifecycle.poison(RuntimeError("iteration failed"), runtime, binding)
    calls = []
    _checkpoint_world_gate(monkeypatch, calls)

    with pytest.raises(MdpCheckpointError, match="WORLD rejected"):
        if operation == "save":
            checkpoint_api.prepare_repeated_d4_checkpoint_save(args, iteration=11)
        else:
            checkpoint_api.prepare_repeated_d4_checkpoint_load(args)
    assert len(calls) == 1 and isinstance(calls[0][-1], BaseException)


def test_checkpoint_boundary_requires_empty_runtime_after_committed_iteration(
    monkeypatch, repeated_d4_checkpoint_runtime
):
    runtime, binding, lifecycle = repeated_d4_checkpoint_runtime
    lifecycle.begin_iteration(runtime, binding)
    lifecycle.commit_iteration(runtime, binding)
    runtime.state = MdpRuntimeState.DECODER_READY
    calls = []
    _checkpoint_world_gate(monkeypatch, calls)

    with pytest.raises(MdpCheckpointError, match="WORLD rejected"):
        checkpoint_api.prepare_repeated_d4_checkpoint_save(_checkpoint_args(), iteration=11)
    assert isinstance(calls[0][-1], BaseException)
    assert "EMPTY" in str(calls[0][-1])


@pytest.mark.parametrize("predicate", (None, False, 0, 1, "yes"), ids=repr)
def test_checkpoint_boundaries_are_exact_literal_true_only(monkeypatch, predicate):
    args = SimpleNamespace(mdp_dynamic_encoder_cp=predicate)
    monkeypatch.setattr(
        integration,
        "get_d4_checkpoint_lifecycle_snapshot",
        lambda: pytest.fail("non-D4 checkpoint queried lifecycle"),
    )
    monkeypatch.setattr(
        checkpoint_api,
        "_converge_repeated_d4_checkpoint_boundary",
        lambda *_args, **_kwargs: pytest.fail("non-D4 checkpoint entered WORLD"),
        raising=False,
    )

    assert checkpoint_api.prepare_repeated_d4_checkpoint_save(args, iteration=1) is None
    assert checkpoint_api.prepare_repeated_d4_checkpoint_load(args) is None
    assert checkpoint_api.validate_repeated_d4_decoded_checkpoint(args, None) is None


def _decoded_state(args):
    checkpoint_args = SimpleNamespace(
        world_size=args.world_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        mdp_encoder_cp=args.mdp_encoder_cp,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        add_position_embedding=args.add_position_embedding,
        max_position_embeddings=args.max_position_embeddings,
        make_vocab_size_divisible_by=args.make_vocab_size_divisible_by,
        padded_vocab_size=args.padded_vocab_size,
        tokenizer_type=args.tokenizer_type,
        data_parallel_random_init=args.data_parallel_random_init,
        global_batch_size=args.global_batch_size,
        model_parallel_size=args.tensor_model_parallel_size,
    )
    return {
        "args": checkpoint_args,
        "iteration": 11,
        "checkpoint_version": 3.0,
        "model": {"language_model.layers.0.weight": "decoder"},
        ENCODER_STATE_KEY: {
            "vision_model.module.layers.0.weight": "vision-0",
            "vision_model.module.layers.1.weight": "vision-1",
        },
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-encoder",
        "missing-encoder-weight",
        "bad-encoder-key",
        "missing-language",
        "topology",
        "missing-args",
        "missing-iteration",
        "missing-version",
        "nonexact-version",
        "unsupported-version",
        "negative-version",
        "native-num-layers",
        "native-hidden-size",
    ),
)
def test_decoded_checkpoint_validation_is_pure_and_world_consensed(
    monkeypatch, repeated_d4_checkpoint_runtime, mutation
):
    args = _checkpoint_args()
    args.world_size = 8
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.context_parallel_size = 4
    args.expert_model_parallel_size = 1
    args.mdp_encoder_cp = 4
    state = _decoded_state(args)
    if mutation == "missing-encoder":
        state.pop(ENCODER_STATE_KEY)
    elif mutation == "missing-encoder-weight":
        state[ENCODER_STATE_KEY].pop("vision_model.module.layers.1.weight")
    elif mutation == "bad-encoder-key":
        state[ENCODER_STATE_KEY] = {"decoder.weight": "bad-prefix"}
    elif mutation == "missing-language":
        state["model"] = {"vision_model.weight": "wrong-model-domain"}
    elif mutation == "topology":
        state["args"].context_parallel_size = 2
    elif mutation == "missing-args":
        state.pop("args")
    elif mutation == "missing-iteration":
        state.pop("iteration")
    elif mutation == "missing-version":
        state.pop("checkpoint_version")
    elif mutation == "nonexact-version":
        state["checkpoint_version"] = "3.0"
    elif mutation == "unsupported-version":
        state["checkpoint_version"] = 1.5
    elif mutation == "negative-version":
        state["checkpoint_version"] = -1
    elif mutation == "native-num-layers":
        state["args"].num_layers += 1
    else:
        state["args"].hidden_size += 1
    before = deepcopy(state)
    nested_before = {key: value.copy() for key, value in state.items() if type(value) is dict}
    args_before = vars(state["args"]).copy() if "args" in state else None
    calls = []
    _checkpoint_world_gate(monkeypatch, calls)

    with pytest.raises(MdpCheckpointError, match="load.*post-decode"):
        checkpoint_api.validate_repeated_d4_decoded_checkpoint(args, state)
    assert state == before
    assert all(state[key] == value for key, value in nested_before.items())
    if args_before is not None:
        assert vars(state["args"]) == args_before
    assert isinstance(calls[-1][-1], BaseException)


@pytest.mark.parametrize(
    ("field", "mode"),
    (
        ("num_attention_heads", "always"),
        ("add_position_embedding", "always"),
        ("max_position_embeddings", "vocab"),
        ("make_vocab_size_divisible_by", "vocab"),
        ("padded_vocab_size", "vocab"),
        ("tokenizer_type", "vocab"),
        ("data_parallel_random_init", "data-parallel-random"),
        ("global_batch_size", "phase-transition"),
        ("model_parallel_size", "legacy-version"),
    ),
)
def test_decoded_checkpoint_prevalidates_native_argument_comparisons(
    monkeypatch, repeated_d4_checkpoint_runtime, field, mode
):
    args = _checkpoint_args()
    for name, value in (
        ("world_size", 8),
        ("tensor_model_parallel_size", 1),
        ("pipeline_model_parallel_size", 1),
        ("context_parallel_size", 4),
        ("expert_model_parallel_size", 1),
        ("mdp_encoder_cp", 4),
    ):
        setattr(args, name, value)
    if mode == "vocab":
        args.vocab_file = "tokenizer.model"
        args.use_dist_ckpt = False
    elif mode == "data-parallel-random":
        args.data_parallel_random_init = True
    elif mode == "phase-transition":
        args.phase_transition_iterations = [1]
    state = _decoded_state(args)
    if mode == "legacy-version":
        state["checkpoint_version"] = 1.0
    setattr(state["args"], field, f"wrong-{field}")
    calls = []
    _checkpoint_world_gate(monkeypatch, calls)

    with pytest.raises(MdpCheckpointError, match="load.*post-decode"):
        checkpoint_api.validate_repeated_d4_decoded_checkpoint(args, state)
    assert len(calls) == 1
    assert isinstance(calls[0][-1], BaseException)


def test_valid_decoded_checkpoint_preserves_logical_keys_and_metadata(
    monkeypatch, repeated_d4_checkpoint_runtime
):
    args = _checkpoint_args()
    args.world_size = 8
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.context_parallel_size = 4
    args.expert_model_parallel_size = 1
    args.mdp_encoder_cp = 4
    state = _decoded_state(args)
    calls = []
    _checkpoint_world_gate(monkeypatch, calls)

    checkpoint_api.validate_repeated_d4_decoded_checkpoint(args, state)

    assert calls[-1][2:] == ("load", "post-decode", 11, None)
    assert tuple(state["model"]) == ("language_model.layers.0.weight",)
    assert tuple(state[ENCODER_STATE_KEY]) == (
        "vision_model.module.layers.0.weight",
        "vision_model.module.layers.1.weight",
    )


@pytest.mark.parametrize("version", (2.5, 4.0))
def test_decoded_checkpoint_accepts_native_supported_noncanonical_versions(
    monkeypatch, repeated_d4_checkpoint_runtime, version
):
    args = _checkpoint_args()
    for name, value in (
        ("world_size", 8),
        ("tensor_model_parallel_size", 1),
        ("pipeline_model_parallel_size", 1),
        ("context_parallel_size", 4),
        ("expert_model_parallel_size", 1),
        ("mdp_encoder_cp", 4),
    ):
        setattr(args, name, value)
    state = _decoded_state(args)
    state["checkpoint_version"] = version
    calls = []
    _checkpoint_world_gate(monkeypatch, calls)

    checkpoint_api.validate_repeated_d4_decoded_checkpoint(args, state)

    assert calls[-1][2:] == ("load", "post-decode", 11, None)


@pytest.mark.parametrize(
    "decoder_state",
    (
        {"model": {"language_model.layers.0.weight": "decoder"}},
        {"model0": {"language_model.layers.0.weight": "decoder"}, "model1": {}},
    ),
    ids=("pp-single", "vpp-with-native-empty-stage"),
)
def test_decoded_checkpoint_accepts_native_pp_vpp_model_key_grammar(
    monkeypatch, repeated_d4_checkpoint_runtime, decoder_state
):
    args = _checkpoint_args()
    for name, value in (
        ("world_size", 8),
        ("tensor_model_parallel_size", 1),
        ("pipeline_model_parallel_size", 1),
        ("context_parallel_size", 4),
        ("expert_model_parallel_size", 1),
        ("mdp_encoder_cp", 4),
    ):
        setattr(args, name, value)
    state = _decoded_state(args)
    state.pop("model")
    state.update(decoder_state)
    calls = []
    _checkpoint_world_gate(monkeypatch, calls)

    checkpoint_api.validate_repeated_d4_decoded_checkpoint(args, state)
    assert calls[-1][2:] == ("load", "post-decode", 11, None)


def test_checkpoint_status_generation_and_iteration_use_signed_int64_bounds():
    maximum = (1 << 63) - 1
    status = checkpoint_api._RepeatedD4CheckpointStatus(
        global_rank=0,
        operation="save",
        phase="pre-state",
        topology_digest=b"t" * 16,
        lifecycle_generation=maximum,
        iteration=maximum,
        error_code=0,
    )
    assert (
        checkpoint_api._RepeatedD4CheckpointStatus.from_wire_tuple(status.to_wire_tuple()) == status
    )
    for field in ("lifecycle_generation", "iteration"):
        values = dict(
            global_rank=0,
            operation="save",
            phase="pre-state",
            topology_digest=b"t" * 16,
            lifecycle_generation=0,
            iteration=0,
            error_code=0,
        )
        values[field] = maximum + 1
        with pytest.raises(MdpCheckpointError, match="int64|bounded"):
            checkpoint_api._RepeatedD4CheckpointStatus(**values)
        wire = list(status.to_wire_tuple())
        wire[3 if field == "lifecycle_generation" else 4] = maximum + 1
        with pytest.raises(MdpCheckpointError, match="int64|bounded"):
            checkpoint_api._RepeatedD4CheckpointStatus.from_wire_tuple(tuple(wire))


@pytest.mark.skipif(_WORLD_SIZE != 8, reason="requires the exact repeated-D4 world8")
def test_checkpoint_boundary_uses_one_real_lifecycle_bound_world_consensus():
    rank = torch.distributed.get_rank()
    world = torch.distributed.group.WORLD
    domains = tuple(
        torch.distributed.new_group(ranks=ranks) for ranks in (tuple(range(4)), tuple(range(4, 8)))
    )
    factory_calls = []
    gather_calls = []

    def status_factory(**kwargs):
        factory_calls.append((tuple(kwargs["group_ranks"]), kwargs["global_rank"]))
        gather = checkpoint_api.make_precollective_status_gather(**kwargs)

        def counted(value, *, timeout_seconds):
            gather_calls.append((tuple(kwargs["group_ranks"]), value))
            return gather(value, timeout_seconds=timeout_seconds)

        return counted

    binding = binding_api._make_repeated_d4_group_binding(
        world_group=world,
        domain_group=domains[rank // 4],
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=torch.device("cuda", torch.cuda.current_device()),
        timeout_seconds=30.0,
        status_gather_factory=status_factory,
    )
    factory_calls.clear()
    gather_calls.clear()

    try:
        for case in (
            "success",
            "one-rank-error",
            "generation-skew",
            "iteration-skew",
            "iteration-overflow",
            "bad-phase",
            "bad-order",
        ):
            runtime = _CheckpointRuntime(binding)
            lifecycle = transaction_api._bind_d4_checkpoint_lifecycle(runtime, binding)
            try:
                lifecycle.begin_iteration(runtime, binding)
                lifecycle.commit_iteration(runtime, binding)
                if case == "generation-skew" and rank >= 4:
                    lifecycle.begin_iteration(runtime, binding)
                    lifecycle.commit_iteration(runtime, binding)
                snapshot = lifecycle.snapshot(runtime, binding)
                if case == "iteration-overflow":
                    iteration = 1 << 63 if rank == 0 else (1 << 63) - 1
                else:
                    iteration = 12 if case == "iteration-skew" and rank >= 4 else 11
                phase = "unknown" if case == "bad-phase" else "pre-state"
                operation = "load" if case == "bad-order" else "save"
                local_error = (
                    RuntimeError("rank zero failed")
                    if case == "one-rank-error" and rank == 0
                    else None
                )
                before_factory = len(factory_calls)
                before_gather = len(gather_calls)
                if case == "success":
                    checkpoint_api._converge_repeated_d4_checkpoint_boundary(
                        runtime,
                        snapshot,
                        operation=operation,
                        phase=phase,
                        iteration=iteration,
                        local_error=local_error,
                    )
                else:
                    with pytest.raises(MdpCheckpointError, match="WORLD|checkpoint"):
                        checkpoint_api._converge_repeated_d4_checkpoint_boundary(
                            runtime,
                            snapshot,
                            operation=operation,
                            phase=phase,
                            iteration=iteration,
                            local_error=local_error,
                        )
                assert factory_calls[before_factory:] == [(tuple(range(8)), rank)]
                assert len(gather_calls) == before_gather + 1
                assert gather_calls[-1][0] == tuple(range(8))
                if case not in ("bad-phase", "bad-order", "iteration-overflow"):
                    status = checkpoint_api._RepeatedD4CheckpointStatus.from_wire_tuple(
                        gather_calls[-1][1]
                    )
                    assert status.global_rank == rank
                    assert status.operation == operation
                    assert status.phase == phase
                    assert status.lifecycle_generation == snapshot.generation
                    assert status.iteration == iteration
                    assert status.topology_digest == checkpoint_api._checkpoint_topology_digest(
                        binding
                    )
            finally:
                lifecycle.retire(runtime, binding)

        for case in (
            "all-none",
            "content-success",
            "mixed-none",
            "version-skew",
            "topology-skew",
            "num-layers-skew",
            "hidden-size-skew",
            "invalid-legacy-version",
        ):
            args = _checkpoint_args()
            for name, value in (
                ("world_size", 8),
                ("tensor_model_parallel_size", 1),
                ("pipeline_model_parallel_size", 1),
                ("context_parallel_size", 4),
                ("expert_model_parallel_size", 1),
                ("mdp_encoder_cp", 4),
            ):
                setattr(args, name, value)
            runtime = _CheckpointRuntime(binding)
            lifecycle = transaction_api._bind_d4_checkpoint_lifecycle(runtime, binding)
            integration._RUNTIME = runtime
            integration._D4_CHECKPOINT_LIFECYCLE = lifecycle
            try:
                state = None if case in ("all-none", "mixed-none") else _decoded_state(args)
                if case == "mixed-none" and rank != 0:
                    state = _decoded_state(args)
                if state is not None:
                    state["model"]["language_model.layers.0.weight"] = f"decoder-rank-{rank}"
                    state[ENCODER_STATE_KEY][
                        "vision_model.module.layers.0.weight"
                    ] = f"vision-rank-{rank}"
                if case == "version-skew" and rank >= 4:
                    state["checkpoint_version"] = 2.0
                if case == "topology-skew" and rank >= 4:
                    args.pipeline_model_parallel_size = 2
                    state["args"].pipeline_model_parallel_size = 2
                if case == "num-layers-skew" and rank >= 4:
                    state["args"].num_layers += 1
                if case == "hidden-size-skew" and rank >= 4:
                    state["args"].hidden_size += 1
                if case == "invalid-legacy-version" and rank >= 4:
                    state["checkpoint_version"] = -1
                before_factory = len(factory_calls)
                before_gather = len(gather_calls)
                if case in ("all-none", "content-success"):
                    checkpoint_api.validate_repeated_d4_decoded_checkpoint(args, state)
                else:
                    with pytest.raises(MdpCheckpointError, match="WORLD|checkpoint"):
                        checkpoint_api.validate_repeated_d4_decoded_checkpoint(args, state)
                assert factory_calls[before_factory:] == [(tuple(range(8)), rank)]
                assert len(gather_calls) == before_gather + 1
                status = checkpoint_api._RepeatedD4CheckpointStatus.from_wire_tuple(
                    gather_calls[-1][1]
                )
                assert status.operation == "load"
                assert status.phase == "post-decode"
            finally:
                integration._D4_CHECKPOINT_LIFECYCLE = None
                integration._RUNTIME = None
                lifecycle.retire(runtime, binding)
    finally:
        torch.distributed.destroy_process_group(domains[rank // 4])


def test_add_encoder_state_rejects_duplicates():
    class _FakeDdp:
        def sharded_state_dict(self, prefix="", metadata=None):
            return {"marker": prefix}

    state = add_encoder_state({}, _FakeDdp())
    assert state[ENCODER_STATE_KEY] == {"marker": "vision_model."}
    with pytest.raises(MdpCheckpointError, match="exactly once"):
        add_encoder_state(state, _FakeDdp())


def test_load_encoder_state_strips_both_prefix_levels():
    class _FakeDdp:
        def __init__(self, keys=("proj.weight",)):
            self.loaded = None
            self.strict = None
            self._keys = keys

        def state_dict(self):
            # `load_encoder_state` compares the checkpoint's keys against the
            # module's own before delegating, so the double must expose them.
            return {key: None for key in self._keys}

        def load_state_dict(self, state_dict, strict=True):
            self.loaded = state_dict
            self.strict = strict

    encoder = _FakeDdp()
    load_encoder_state(
        {ENCODER_STATE_KEY: {"vision_model.module.proj.weight": "w"}}, encoder, strict=False
    )
    assert encoder.loaded == {"proj.weight": "w"}
    assert encoder.strict is False

    with pytest.raises(MdpCheckpointError, match=ENCODER_STATE_KEY):
        load_encoder_state({"model": {}}, _FakeDdp())
    with pytest.raises(MdpCheckpointError, match="drifted apart"):
        load_encoder_state({ENCODER_STATE_KEY: {"proj.weight": "w"}}, _FakeDdp())


class _RealDdp(torch.nn.Module):
    """Stand-in with the same ``load_state_dict`` contract as the encoder DDP.

    ``_BaseDataParallel.load_state_dict`` forwards to the wrapped module and
    returns ``None``, so a caller cannot learn which keys were missing from its
    return value -- which is why the guard below checks the keys up front.
    """

    def __init__(self):
        super().__init__()
        self.module = torch.nn.Linear(4, 4, bias=False)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


def test_load_encoder_state_rejects_missing_keys_even_when_not_strict():
    """A relaxed load must not leave the encoder randomly initialized.

    ``strict`` reaches :func:`load_encoder_state` from ``load_checkpoint``'s own
    parameter, and ``torch.nn.Module.load_state_dict(strict=False)`` reports
    missing keys instead of raising. The encoder is fully replicated and is
    always written whole, so a key that is absent here means the state did not
    round-trip -- there is no "empty stage" case to tolerate, unlike a decoder
    chunk. Silently skipping it would resume training from the random
    initialization, which is exactly the failure ``load_encoder_state`` exists
    to prevent.
    """
    encoder = _RealDdp()
    complete = {
        "vision_model.module." + key: value for key, value in encoder.module.state_dict().items()
    }
    load_encoder_state({ENCODER_STATE_KEY: complete}, encoder, strict=False)

    with pytest.raises(MdpCheckpointError, match="missing"):
        load_encoder_state({ENCODER_STATE_KEY: {}}, _RealDdp(), strict=False)
    with pytest.raises(MdpCheckpointError, match="missing"):
        load_encoder_state({ENCODER_STATE_KEY: {}}, _RealDdp(), strict=True)


class _ExtraStateLinear(torch.nn.Linear):
    """A layer with TransformerEngine's extra-state contract.

    Overriding ``get_extra_state``/``set_extra_state`` is what makes
    ``torch.nn.Module`` publish an ``_extra_state`` key in ``state_dict()`` and,
    under ``strict=True``, demand it back on load -- the same contract TE's
    layers carry.
    """

    def get_extra_state(self):
        return {"fp8_meta": None}

    def set_extra_state(self, state):
        pass


def test_load_encoder_state_tolerates_absent_te_extra_state():
    """TE's ``_extra_state`` entries hold no weights and may be absent.

    A real TE encoder publishes ``..._extra_state`` keys in ``state_dict()``
    that the sharded state dict does not always carry, so the weight check must
    exempt them -- and so must the delegated load, which would otherwise reject
    the very same keys one line later. ``megatron.training.checkpointing``'s
    ``load_model_state_dict`` gives every decoder chunk that tolerance already;
    the encoder must not be stricter than the decoder it trains beside.

    The double deliberately does *not* override ``load_state_dict``:
    ``_BaseDataParallel.load_state_dict`` forwards ``strict`` verbatim, so a
    double that dropped it would hide the mismatch this test exists to pin down.
    """

    class _ExtraStateDdp(_RealDdp):
        def __init__(self):
            super().__init__()
            self.module = _ExtraStateLinear(4, 4, bias=False)

    encoder = _ExtraStateDdp()
    assert "_extra_state" in encoder.state_dict()
    weights = {"vision_model.module.weight": torch.full((4, 4), 3.0)}

    load_encoder_state({ENCODER_STATE_KEY: weights}, encoder, strict=True)
    assert torch.equal(encoder.module.weight, torch.full((4, 4), 3.0))

    # The weight guard still fires for the same module: exempting extra state
    # must not have relaxed the check that keeps the encoder off its random
    # initialization.
    with pytest.raises(MdpCheckpointError, match="missing"):
        load_encoder_state({ENCODER_STATE_KEY: {}}, _ExtraStateDdp(), strict=True)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_encoder_state_round_trips_strictly(tmp_path_factory):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.mdp.encoder import build_encoder_pg_collection
    from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
    from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
    from megatron.core.transformer.transformer_config import TransformerConfig

    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1))
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)

    class _Enc(torch.nn.Module):
        def __init__(self, config, seed):
            super().__init__()
            self.config = config
            torch.manual_seed(seed)
            self.proj = torch.nn.Linear(8, 8, bias=False)
            self.head = torch.nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.head(self.proj(x))

    def _build(seed):
        model_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            calculate_per_token_loss=True,
            use_cpu_initialization=True,
        )
        return DistributedDataParallel(
            config=model_config,
            ddp_config=DistributedDataParallelConfig(
                use_distributed_optimizer=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            ),
            module=_Enc(model_config, seed).cuda(),
            pg_collection=encoder_pgs,
        )

    source = _build(seed=7)
    target = _build(seed=99)
    probe = torch.full((2, 8), 0.25, device="cuda")
    with torch.no_grad():
        source_out = source(probe).clone()
        assert not torch.equal(source(probe), target(probe))

    # Every rank must agree on the directory; rank 0 broadcasts its tmp dir.
    if torch.distributed.get_rank() == 0:
        directory = str(tmp_path_factory.mktemp("mdp_ckpt"))
    else:
        directory = None
    holder = [directory]
    torch.distributed.broadcast_object_list(holder, src=0)
    directory = holder[0]

    state = add_encoder_state({}, source)
    dist_checkpointing.save(state[ENCODER_STATE_KEY], directory)
    torch.distributed.barrier()

    load_skeleton = add_encoder_state({}, target)
    loaded = dist_checkpointing.load(load_skeleton[ENCODER_STATE_KEY], directory)
    load_encoder_state({ENCODER_STATE_KEY: loaded}, target, strict=True)

    with torch.no_grad():
        for source_param, target_param in zip(
            source.module.parameters(), target.module.parameters()
        ):
            assert torch.equal(source_param, target_param)
        assert torch.equal(target(probe), source_out)
