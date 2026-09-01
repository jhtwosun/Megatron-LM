# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for the optional, model-independent Energon 7 provider."""

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROVIDER_MODULE = "examples.multimodal_dev.data.energon.provider"
_PROVIDER_PATH = f"{_PROVIDER_MODULE}.train_valid_test_datasets_provider"


def _provider():
    return importlib.import_module(_PROVIDER_MODULE)


def _task_encoder_type():
    class TaskEncoder:
        def preencode_sample(self, sample):
            return sample

        def encode_sample(self, sample):
            return sample

        def batch(self, samples):
            return samples

        def encode_batch(self, batch):
            return batch

    return TaskEncoder


def _fake_energon(version="7.3.2"):
    task_encoder_type = _task_encoder_type()

    def get_train_dataset(path, *, worker_config, **kwargs):
        return path, worker_config, kwargs

    def get_val_datasets(path, *, worker_config, **kwargs):
        return path, worker_config, kwargs

    def get_loader(dataset, *, worker_config=None, prefetch_factor=2):
        return dataset, worker_config, prefetch_factor

    def get_savable_loader(dataset, *, worker_config=None, prefetch_factor=2):
        return dataset, worker_config, prefetch_factor

    return SimpleNamespace(
        __version__=version,
        WorkerConfig=type("WorkerConfig", (), {}),
        TaskEncoder=task_encoder_type,
        MetadatasetV2=type("MetadatasetV2", (), {}),
        get_train_dataset=get_train_dataset,
        get_val_datasets=get_val_datasets,
        get_loader=get_loader,
        get_savable_loader=get_savable_loader,
    )


def _args(**overrides):
    values = {
        "model_arch": "qwen35_vl",
        "dataset_provider": "energon",
        "dataloader_type": "external",
        "energon_path": "/dataset",
        "energon_split": "train",
        "energon_val_split": "val",
        "energon_packing_buffer_size": 32,
        "energon_max_samples_per_sequence": 8,
        "energon_shuffle_buffer_size": 100,
        "energon_prefetch_factor": 3,
        "num_workers": 2,
        "micro_batch_size": 3,
        "eval_iters": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_selects_one_lazy_energon_provider_without_changing_mock():
    sys.modules.pop(_PROVIDER_MODULE, None)
    energon_was_loaded = "megatron.energon" in sys.modules
    models = importlib.reload(importlib.import_module("examples.multimodal_dev.models"))
    providers = models.MODEL_REGISTRY["qwen35_vl"]["dataset_providers"]

    assert providers["energon"] == _PROVIDER_PATH
    assert providers["mock"] == ("examples.multimodal_dev.data.mock.train_valid_test_datasets_provider")
    assert _PROVIDER_MODULE not in sys.modules
    assert ("megatron.energon" in sys.modules) is energon_was_loaded


def test_importing_generic_provider_does_not_import_optional_or_qwen_policy():
    sys.modules.pop(_PROVIDER_MODULE, None)
    sys.modules.pop("megatron.energon", None)
    provider = _provider()

    assert "megatron.energon" not in sys.modules
    source = Path(provider.__file__).read_text()
    assert "qwen" not in source.lower()
    assert "token_id" not in source.lower()


def test_multimodal_args_expose_only_generic_energon_controls():
    from examples.multimodal_dev.arguments import add_multimodal_args

    parser = argparse.ArgumentParser()
    add_multimodal_args(parser)
    args = parser.parse_args(
        ["--dataset-provider", "energon", "--energon-path", "/data", "--energon-prefetch-factor", "5"]
    )

    assert args.energon_path == "/data"
    assert args.energon_split == "train"
    assert args.energon_val_split == "val"
    assert args.energon_packing_buffer_size > 0
    assert args.energon_max_samples_per_sequence > 0
    assert args.energon_shuffle_buffer_size > 0
    assert args.energon_prefetch_factor == 5


def test_missing_optional_energon_is_actionable_and_dependency_errors_survive(monkeypatch):
    provider = _provider()

    def missing_energon(_name):
        raise ModuleNotFoundError("missing energon", name="megatron.energon")

    monkeypatch.setattr(provider, "import_module", missing_energon)
    with pytest.raises(
        provider.EnergonCompatibilityError, match=r"requires megatron-energon>=7\.0\.0.*Install or upgrade"
    ):
        provider.load_energon7_api()

    def missing_dependency(_name):
        raise ModuleNotFoundError("missing dependency", name="dependency")

    monkeypatch.setattr(provider, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="dependency"):
        provider.load_energon7_api()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda module: setattr(module, "__version__", "6.9.0"), "found 6.9.0"),
        (lambda module: delattr(module, "MetadatasetV2"), "MetadatasetV2"),
        (lambda module: setattr(module, "TaskEncoder", type("TaskEncoder", (), {})), "TaskEncoder.preencode_sample"),
        (
            lambda module: setattr(module, "get_train_dataset", lambda path, **kwargs: (path, kwargs)),
            "get_train_dataset.*keyword-only worker_config",
        ),
    ],
)
def test_energon7_guard_rejects_version_symbol_and_signature_drift(monkeypatch, mutation, message):
    provider = _provider()
    module = _fake_energon()
    mutation(module)
    monkeypatch.setattr(provider, "import_module", lambda _name: module)

    with pytest.raises(provider.EnergonCompatibilityError, match=message):
        provider.load_energon7_api()


@pytest.mark.parametrize("loader_name", ["get_loader", "get_savable_loader"])
@pytest.mark.parametrize(
    ("loader", "message"),
    [
        (lambda dataset, *, worker_config=object(), prefetch_factor=2: dataset, "default to None"),
        (lambda dataset, *, worker_config, prefetch_factor=2: dataset, "default to None"),
        (lambda dataset, *, worker_config=None: dataset, "prefetch_factor"),
        (lambda *, dataset, worker_config=None, prefetch_factor=2: dataset, "positional dataset"),
    ],
)
def test_energon7_guard_rejects_loader_signature_drift(monkeypatch, loader_name, loader, message):
    provider = _provider()
    module = _fake_energon()
    setattr(module, loader_name, loader)
    monkeypatch.setattr(provider, "import_module", lambda _name: module)

    with pytest.raises(provider.EnergonCompatibilityError, match=message):
        provider.load_energon7_api()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("dataloader_type", "single", "--dataloader-type external"),
        ("energon_path", None, "--energon-path"),
        ("num_workers", -1, "--num-workers"),
        ("energon_packing_buffer_size", 0, "--energon-packing-buffer-size"),
        ("energon_prefetch_factor", True, "--energon-prefetch-factor"),
    ],
)
def test_provider_arguments_fail_before_loader_construction(name, value, message):
    provider = _provider()
    with pytest.raises(ValueError, match=message):
        provider._validate_provider_args(_args(**{name: value}))


def test_provider_uses_dataset_owned_worker_config_and_exact_split_contract(monkeypatch):
    provider = _provider()
    events = []
    task_encoder_type = _task_encoder_type()

    class WorkerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("worker", kwargs))

    def train(path, *, worker_config, **kwargs):
        events.append(("train", path, worker_config, kwargs))
        return "train-dataset"

    def valid(path, *, worker_config, **kwargs):
        events.append(("valid", path, worker_config, kwargs))
        return [("valid-a", object()), ("valid-b", object())]

    def loader(dataset, *, worker_config=None, prefetch_factor=2):
        events.append(("loader", dataset, worker_config, prefetch_factor))
        return iter((dataset,))

    api = provider.Energon7Api(
        version="7.3.2",
        major_version=7,
        worker_config_type=WorkerConfig,
        task_encoder_type=task_encoder_type,
        metadataset_v2_type=type("MetadatasetV2", (), {}),
        get_train_dataset=train,
        get_val_datasets=valid,
        get_loader=loader,
        get_savable_loader=loader,
    )
    args = _args()
    monkeypatch.setattr(provider, "load_energon7_api", lambda: api)
    monkeypatch.setattr(provider, "get_args", lambda: args)
    monkeypatch.setattr(provider.parallel_state, "get_data_parallel_rank", lambda: 1)
    monkeypatch.setattr(provider.parallel_state, "get_data_parallel_world_size", lambda: 4)
    monkeypatch.setattr(provider.parallel_state, "get_data_parallel_group", lambda: "dp")
    monkeypatch.setattr(provider, "_build_task_encoder", lambda _api, _args: task_encoder_type())

    train_iterator, valid_iterators, test_iterator = provider.train_valid_test_datasets_provider(None)

    assert next(train_iterator) == "train-dataset"
    assert [next(iterator) for iterator in valid_iterators] == ["valid-a", "valid-b"]
    assert test_iterator is None
    worker = next(event[1] for event in events if event[0] == "worker")
    assert worker == {"rank": 1, "world_size": 4, "num_workers": 2, "data_parallel_group": "dp"}
    assert all(event[2] is None for event in events if event[0] == "loader")
    train_event = next(event for event in events if event[0] == "train")
    assert train_event[1] == "/dataset"
    assert train_event[3]["split_part"] == "train"
    assert train_event[3]["batch_size"] == 3
    valid_event = next(event for event in events if event[0] == "valid")
    assert valid_event[3]["split_part"] == "val"


def test_eval_off_never_constructs_validation(monkeypatch):
    provider = _provider()
    api = _fake_energon()
    api.get_train_dataset = lambda path, *, worker_config, **kwargs: (path,)
    api.get_savable_loader = lambda dataset, *, worker_config=None, prefetch_factor=2: dataset
    api.get_val_datasets = lambda *args, **kwargs: pytest.fail("validation must stay lazy")
    monkeypatch.setattr(
        provider,
        "load_energon7_api",
        lambda: provider.Energon7Api(
            version="7.3.2",
            major_version=7,
            worker_config_type=lambda **kwargs: kwargs,
            task_encoder_type=api.TaskEncoder,
            metadataset_v2_type=api.MetadatasetV2,
            get_train_dataset=api.get_train_dataset,
            get_val_datasets=api.get_val_datasets,
            get_loader=api.get_loader,
            get_savable_loader=api.get_savable_loader,
        ),
    )
    monkeypatch.setattr(provider, "get_args", lambda: _args(eval_iters=0))
    monkeypatch.setattr(provider.parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(provider.parallel_state, "get_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(provider.parallel_state, "get_data_parallel_group", lambda: "dp")
    monkeypatch.setattr(provider, "_build_task_encoder", lambda api, args: api.task_encoder_type())

    _, valid, test = provider.train_valid_test_datasets_provider(None)
    assert valid is None
    assert test is None
