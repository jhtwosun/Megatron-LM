# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Note: --ckpt-format torch_dist has tests in tests/unit_tests/dist_checkpointing.
import os
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import pytest
import torch
import torch.distributed.checkpoint

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.mdp import checkpoint as mdp_checkpoint_api
from megatron.core.mdp import integration as mdp_integration
from megatron.core.mdp.runtime import MdpRuntimeState
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training import checkpointing as checkpointing_api
from megatron.training.checkpointing import (
    CheckpointType,
    _build_sharded_state_dict_metadata,
    _load_base_checkpoint,
    get_checkpoint_tracker_filename,
    load_checkpoint,
    read_metadata,
    save_checkpoint,
)
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class MockModel(MegatronModule):
    """Dummy megatron model."""

    def __init__(self, config):
        super().__init__(config=config)
        self.l = torch.nn.Linear(1, 2)
        torch.nn.init.ones_(self.l.weight)
        torch.nn.init.zeros_(self.l.bias)
        self._called_metadata = []

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        self._called_metadata.append(metadata)
        return self.state_dict()


class MockState:
    def __init__(self, state_dict):
        self._state_dict = state_dict
        self.is_stub_optimizer = False
        self._called_metadata = []

        # Optimizers are expected to have this attribute for checkpointing.
        self.param_groups = []

    def state_dict(self, is_loading=False):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def save_parameter_state(self, *args, **kwargs):
        pass

    def load_parameter_state(self, *args, **kwargs):
        pass

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        self._called_metadata.append(metadata)
        return self.state_dict()


class _CheckpointBoundaryReached(RuntimeError):
    pass


def test_repeated_d4_save_boundary_precedes_native_callbacks_rng_and_state(monkeypatch):
    args = SimpleNamespace(mdp_dynamic_encoder_cp=True, async_save=False)
    set_args(args)
    events = []
    monkeypatch.setattr(
        checkpointing_api,
        "prepare_repeated_d4_checkpoint_save",
        lambda selected, *, iteration: events.append(("save", iteration)),
        raising=False,
    )
    monkeypatch.setattr(
        checkpointing_api,
        "on_save_checkpoint_start",
        lambda *_args: (_ for _ in ()).throw(_CheckpointBoundaryReached("native callback")),
    )
    monkeypatch.setattr(
        checkpointing_api,
        "get_rng_state",
        lambda *_args, **_kwargs: pytest.fail("RNG state generated before D4 boundary"),
    )
    monkeypatch.setattr(
        checkpointing_api,
        "generate_state_dict",
        lambda *_args, **_kwargs: pytest.fail("state generated before D4 boundary"),
    )

    with pytest.raises(_CheckpointBoundaryReached, match="native callback"):
        save_checkpoint(1, [], None, None, 0)
    assert events == [("save", 1)]


def test_repeated_d4_load_pre_boundary_precedes_any_checkpoint_io(monkeypatch):
    args = SimpleNamespace(
        mdp_dynamic_encoder_cp=True, load="/does/not/matter", pretrained_checkpoint="/also/not/read"
    )
    set_args(args)
    monkeypatch.setattr(
        checkpointing_api,
        "prepare_repeated_d4_checkpoint_load",
        lambda selected: (_ for _ in ()).throw(_CheckpointBoundaryReached("load pre-io")),
        raising=False,
    )
    monkeypatch.setattr(
        checkpointing_api,
        "checkpoint_exists",
        lambda *_args: pytest.fail("checkpoint path read before D4 load boundary"),
    )
    monkeypatch.setattr(
        checkpointing_api,
        "_load_base_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("native checkpoint I/O started before boundary"),
    )

    with pytest.raises(_CheckpointBoundaryReached, match="load pre-io"):
        load_checkpoint([], None, None)


def _minimal_load_args():
    # Supported native training setup invokes load_checkpoint once in a fresh
    # process. K1b intentionally does not add LOADED_IDLE or repeated direct-load support.
    return SimpleNamespace(
        mdp_dynamic_encoder_cp=True,
        load="/does/not/matter",
        pretrained_checkpoint=None,
        ckpt_format="torch",
        auto_detect_ckpt_format=False,
        world_size=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=4,
        expert_model_parallel_size=1,
        mdp_encoder_cp=4,
    )


def test_repeated_d4_post_decode_boundary_precedes_all_state_mutation(monkeypatch):
    args = _minimal_load_args()
    set_args(args)
    state = {"args": object(), "model": {}, "mdp_vision_model": {}}
    events = []
    monkeypatch.setattr(
        checkpointing_api,
        "prepare_repeated_d4_checkpoint_load",
        lambda selected: events.append("pre"),
        raising=False,
    )
    monkeypatch.setattr(
        checkpointing_api,
        "validate_repeated_d4_decoded_checkpoint",
        lambda selected, decoded: (
            events.append(("post", decoded))
            or (_ for _ in ()).throw(_CheckpointBoundaryReached("post-decode"))
        ),
        raising=False,
    )
    monkeypatch.setattr(checkpointing_api, "unwrap_model", lambda value: value)
    monkeypatch.setattr(
        checkpointing_api,
        "_load_base_checkpoint",
        lambda *_args, **_kwargs: (state, "checkpoint", False, CheckpointType.LEGACY),
    )
    for name in ("set_checkpoint_version", "check_checkpoint_args", "update_num_microbatches"):
        monkeypatch.setattr(
            checkpointing_api,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"{_name} mutated state before post-decode boundary"
            ),
        )

    with pytest.raises(_CheckpointBoundaryReached, match="post-decode"):
        load_checkpoint([object()], None, None)
    assert events == ["pre", ("post", state)]


@pytest.mark.parametrize("preliminary", (False, True), ids=("main", "preliminary-rank0"))
def test_native_load_io_failure_is_task_fatal_without_post_decode_consensus(
    monkeypatch, preliminary
):
    args = _minimal_load_args()
    args.auto_detect_ckpt_format = preliminary
    set_args(args)
    events = []
    monkeypatch.setattr(
        checkpointing_api,
        "prepare_repeated_d4_checkpoint_load",
        lambda selected: events.append("pre"),
        raising=False,
    )
    monkeypatch.setattr(
        checkpointing_api,
        "validate_repeated_d4_decoded_checkpoint",
        lambda *_args: pytest.fail("post-decode consensus entered after native I/O failure"),
        raising=False,
    )
    monkeypatch.setattr(checkpointing_api, "unwrap_model", lambda value: value)
    native_error = OSError("native read failed")
    monkeypatch.setattr(
        checkpointing_api,
        "_load_base_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(native_error),
    )

    with pytest.raises(OSError, match="native read failed") as raised:
        load_checkpoint([object()], None, None)
    assert raised.value is native_error
    assert events == ["pre"]


def test_missing_checkpoint_still_runs_post_decode_consensus_before_clean_return(monkeypatch):
    args = _minimal_load_args()
    set_args(args)
    events = []
    monkeypatch.setattr(
        checkpointing_api,
        "prepare_repeated_d4_checkpoint_load",
        lambda selected: events.append("pre"),
        raising=False,
    )
    binding = object()
    runtime = SimpleNamespace(state=MdpRuntimeState.EMPTY, dynamic_group_binding=binding)
    snapshot = SimpleNamespace(may_load=True)
    authority = SimpleNamespace(world_ranks=tuple(range(8)), expert_parallel_size=1)
    monkeypatch.setattr(
        mdp_checkpoint_api,
        "_validate_repeated_d4_group_binding",
        lambda selected: authority if selected is binding else pytest.fail("wrong binding"),
    )
    monkeypatch.setattr(mdp_integration, "get_runtime", lambda: runtime)
    monkeypatch.setattr(mdp_integration, "get_d4_checkpoint_lifecycle_snapshot", lambda: snapshot)

    def converge(
        selected_runtime,
        selected_snapshot,
        *,
        operation,
        phase,
        iteration,
        local_error,
        **_boundary_metadata,
    ):
        assert selected_runtime is runtime
        assert selected_snapshot is snapshot
        assert (operation, phase, iteration, local_error) == ("load", "post-decode", 0, None)
        events.append(("post", None))

    monkeypatch.setattr(mdp_checkpoint_api, "_converge_repeated_d4_checkpoint_boundary", converge)
    monkeypatch.setattr(
        checkpointing_api,
        "validate_repeated_d4_decoded_checkpoint",
        mdp_checkpoint_api.validate_repeated_d4_decoded_checkpoint,
        raising=False,
    )
    monkeypatch.setattr(checkpointing_api, "unwrap_model", lambda value: value)
    monkeypatch.setattr(
        checkpointing_api,
        "_load_base_checkpoint",
        lambda *_args, **_kwargs: (None, "checkpoint", False, None),
    )

    assert load_checkpoint([object()], None, None) == (0, 0)
    assert events == ["pre", ("post", None)]


def create_checkpoint(load_path, ckpt_format):
    """Setup a dummy checkpoint directory."""
    iteration = 123
    ckpt_dir = load_path / "iter_{:07d}".format(iteration)
    tracker_path = get_checkpoint_tracker_filename(load_path)
    with open(tracker_path, "w") as f:
        f.write(str(iteration))

    state_dict = {"args": "dummy", "iteration": iteration}

    if ckpt_format == "torch":
        # Torch checkpoints use a specific directory structure.
        pt_dir = ckpt_dir / "mp_rank_00"
        pt_dir.mkdir(parents=True)
        torch.save(state_dict, pt_dir / "model_optim_rng.pt")
    elif ckpt_format == "torch_dcp" and is_torch_min_version("2.4.0"):
        torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)


@pytest.fixture
def create_args():
    """Setup dummy args."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.async_strategy = "mcore"
    args.data_parallel_random_init = False
    args.no_save_optim = False
    args.no_save_rng = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.dist_ckpt_optim_fully_reshardable = False
    args.distrib_optim_fully_reshardable_mem_efficient = False
    args.auto_detect_ckpt_format = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None
    args.swiglu = True
    args.num_experts = 1
    args.verify_integrity = False

    yield args


@pytest.fixture
def create_ckpt_load_args(create_args):
    """Setup dummy args allowing checkpoint load."""
    args = create_args
    args.auto_detect_ckpt_format = False
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.num_layers = 1
    args.hidden_size = 2
    args.num_attention_heads = 1
    args.add_position_embedding = False
    args.vocab_file = None
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.ckpt_assume_constant_structure = False
    args.ckpt_fully_parallel_save = False
    args.ckpt_fully_parallel_load = False
    args.ckpt_load_validate_sharding_integrity = True
    args.dist_ckpt_strictness = 'assume_ok_unexpected'
    args.use_megatron_fsdp = False
    args.strict_fsdp_dtensor_load = True
    args.phase_transition_iterations = None

    yield args


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


@pytest.mark.parametrize("ckpt_format", ["torch_dcp"])
def test_load_base_checkpoint(
    init_model_parallel, create_ckpt_load_args, ckpt_format, tmp_path_dist_ckpt
):
    """Test _load_base_checkpoint."""

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    # TempNamedDir uses the same directory for all ranks in a multi-GPU setup. Cleanup is handled.
    with TempNamedDir(tmp_path_dist_ckpt / "test_load_base_checkpoint", sync=True) as load_dir:
        create_checkpoint(load_dir, ckpt_format)
        args = create_ckpt_load_args
        args.ckpt_format = ckpt_format

        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir, args, rank0=True
        )

    assert state_dict["args"] == "dummy"
    assert state_dict["iteration"] == 123

    expected_ckpt_path = None
    if ckpt_format == "torch":
        expected_ckpt_path = str(load_dir / "iter_0000123" / "mp_rank_00" / "model_optim_rng.pt")
    elif ckpt_format == "torch_dcp":
        expected_ckpt_path = str(load_dir / "iter_0000123")

    assert checkpoint_name == expected_ckpt_path
    assert not release

    expected_ckpt_type = None
    if ckpt_format == "torch":
        expected_ckpt_type = CheckpointType.LEGACY
    elif ckpt_format == "torch_dcp":
        expected_ckpt_type = CheckpointType.TORCH_DCP

    assert ckpt_type == expected_ckpt_type


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dcp", "fsdp_dtensor"])
def test_save_checkpoint(init_model_parallel, create_args, tmp_path_dist_ckpt, ckpt_format):
    """Test save_checkpoint."""
    args = create_args
    args.ckpt_format = ckpt_format

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    iteration = 123
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    optimizer = MockState({"optimizer": "optimizer_state"})
    if ckpt_format == "fsdp_dtensor":
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_distributed_optimizer=True, use_megatron_fsdp=True
            ),
            module=model,
        )
        optimizer = MockState({"state": {}})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
    num_floating_point_operations_so_far = 456

    with TempNamedDir(tmp_path_dist_ckpt / "test_save_checkpoint", sync=True) as save_dir:
        args.save = save_dir
        set_args(args)

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        with open(args.save / "latest_checkpointed_iteration.txt", "r") as f:
            assert iteration == int(f.read())

        ckpt_dir = args.save / "iter_0000123"

        expected_ckpt_path = None
        if ckpt_format == "torch":
            expected_ckpt_path = ckpt_dir / "mp_rank_00" / "model_optim_rng.pt"
        elif ckpt_format in ["torch_dcp", "fsdp_dtensor"]:
            expected_ckpt_path = ckpt_dir / ".metadata"

        assert os.path.exists(expected_ckpt_path)


@pytest.mark.parametrize("ckpt_format", ["torch"])
def test_load_checkpoint(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, ckpt_format
):
    """Test load_checkpoint."""
    args = create_ckpt_load_args
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    with TempNamedDir(tmp_path_dist_ckpt / "test_load_checkpoint", sync=True) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_floating_point_operations_so_far = 456

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        # Create new model, optimizer, and scheduler instances to load into.
        new_model = MockModel(config)
        new_optimizer = MockState({"optimizer": "dummy1"})
        new_opt_param_scheduler = MockState({"opt_param_scheduler": "dummy2"})

        # Load checkpoint
        loaded_iter, loaded_flops = load_checkpoint(
            [new_model], new_optimizer, new_opt_param_scheduler, strict=True
        )

        assert loaded_iter == iteration
        assert loaded_flops == num_floating_point_operations_so_far

        for k in model.state_dict():
            assert torch.equal(model.state_dict()[k], new_model.state_dict()[k])

        assert new_optimizer.state_dict() == optimizer.state_dict()
        assert new_opt_param_scheduler.state_dict() == opt_param_scheduler.state_dict()


def test_dist_checkpoint_versioning(init_model_parallel, tmp_path_dist_ckpt, create_ckpt_load_args):
    """Test distributed checkpoint versioning."""
    args = create_ckpt_load_args
    args.ckpt_format = 'torch_dist'
    args.use_distributed_optimizer = True
    args.use_dist_ckpt = True

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_dist_checkpoint_versioning", sync=True
    ) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_fp_ops = 456

        base_metadata = _build_sharded_state_dict_metadata(args)
        first_job_mock_metadata = {**base_metadata, 'metadata_A': 42, 'metadata_B_soon_removed': 43}
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=first_job_mock_metadata,
        ):
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)

        second_job_mock_metadata = {
            **base_metadata,
            'metadata_A': 'changed_default_value',
            'metadata_C_new': {'nested': 'val'},
        }
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=second_job_mock_metadata,
        ):
            # Load checkpoint (into the same model, we don't check load correctness here)
            load_checkpoint([model], optimizer, opt_param_scheduler, strict=True)
            assert optimizer._called_metadata[-1] == first_job_mock_metadata

            # Save the checkpoint again to check if the content metadata for the new checkpoint will be new
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)
            assert optimizer._called_metadata[-1] == second_job_mock_metadata

        assert optimizer._called_metadata == model._called_metadata
        assert optimizer._called_metadata == [
            first_job_mock_metadata,
            first_job_mock_metadata,
            second_job_mock_metadata,
        ]


@pytest.mark.parametrize(
    "metadata_content,expected_iter,expected_release",
    [
        ("456", 456, False),  # Normal iteration
        ("release", 0, True),  # Release checkpoint should return iteration=1
        ("123", 123, False),  # Another normal iteration
    ],
)
def test_read_metadata_non_distributed(tmp_path, metadata_content, expected_iter, expected_release):
    """Test read_metadata without torch.distributed initialized."""
    test_dir = tmp_path / "test_read_metadata_non_distributed"
    test_dir.mkdir(parents=True, exist_ok=True)
    tracker_file = test_dir / "latest_checkpointed_iteration.txt"

    with open(tracker_file, "w") as f:
        f.write(metadata_content)

    with mock.patch('torch.distributed.is_initialized', return_value=False):
        max_iter, release = read_metadata(str(tracker_file))

    assert max_iter == expected_iter, f"Expected iteration {expected_iter}, got {max_iter}"
    assert release == expected_release, f"Expected release={expected_release}, got {release}"


def _make_metadata_args(
    use_distributed_optimizer=False,
    use_layer_wise_distributed_optimizer=False,
    ckpt_format='torch_dist',
    dist_ckpt_optim_fully_reshardable=False,
    distrib_optim_fully_reshardable_mem_efficient=False,
):
    args = SimpleNamespace()
    args.use_distributed_optimizer = use_distributed_optimizer
    args.use_layer_wise_distributed_optimizer = use_layer_wise_distributed_optimizer
    args.ckpt_format = ckpt_format
    args.dist_ckpt_optim_fully_reshardable = dist_ckpt_optim_fully_reshardable
    args.distrib_optim_fully_reshardable_mem_efficient = (
        distrib_optim_fully_reshardable_mem_efficient
    )
    return args


class TestBuildShardedStateDictMetadata:
    """``_build_sharded_state_dict_metadata`` must set ``distrib_optim_sharding_type``
    whenever a real :class:`DistributedOptimizer` instance will be used at save
    time -- otherwise the DistOpt path falls through to the deprecated
    ``fully_sharded_model_space`` default whose ``flattened_range`` usage is
    rejected by ``ShardedTensor.validate_metadata_integrity`` post commit
    5ab481cb45.
    """

    DUMMY_GROUP = object()

    def test_distributed_optimizer_sets_dp_reshardable_default(self):
        args = _make_metadata_args(use_distributed_optimizer=True)
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'dp_reshardable'

    def test_distributed_optimizer_fully_reshardable_flag(self):
        args = _make_metadata_args(
            use_distributed_optimizer=True, dist_ckpt_optim_fully_reshardable=True
        )
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fully_reshardable'
        assert metadata['distrib_optim_fully_reshardable_mem_efficient'] is False

    def test_distributed_optimizer_fsdp_dtensor(self):
        args = _make_metadata_args(use_distributed_optimizer=True, ckpt_format='fsdp_dtensor')
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fsdp_dtensor'

    def test_layer_wise_only_still_sets_sharding_type(self):
        # Arg parser flips ``use_distributed_optimizer`` off when Muon is in
        # use, but the LayerWise + DistOpt split path still has a DistOpt
        # sub-optimizer for non-Muon params, so the metadata is required.
        args = _make_metadata_args(use_layer_wise_distributed_optimizer=True)
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'dp_reshardable'

    def test_layer_wise_with_fully_reshardable(self):
        args = _make_metadata_args(
            use_layer_wise_distributed_optimizer=True, dist_ckpt_optim_fully_reshardable=True
        )
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert metadata['distrib_optim_sharding_type'] == 'fully_reshardable'

    def test_no_distributed_optimizer_no_sharding_type(self):
        args = _make_metadata_args()
        metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=self.DUMMY_GROUP)
        assert 'distrib_optim_sharding_type' not in metadata
