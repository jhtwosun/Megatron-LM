# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest

_PROVIDER_MODULE = "examples.multimodal_dev.data.qwen35_energon.provider"
_PROVIDER_PATH = f"{_PROVIDER_MODULE}.train_valid_test_datasets_provider"


def _fake_energon(version="7.3.2"):
    class TaskEncoder:
        def preencode_sample(self, sample):
            return sample

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
        TaskEncoder=TaskEncoder,
        get_train_dataset=get_train_dataset,
        get_val_datasets=get_val_datasets,
        get_loader=get_loader,
        get_savable_loader=get_savable_loader,
    )


def test_qwen35_registry_keeps_energon_provider_lazy():
    sys.modules.pop(_PROVIDER_MODULE, None)
    models = importlib.reload(importlib.import_module("examples.multimodal_dev.models"))

    assert models.MODEL_REGISTRY["qwen35_vl"]["dataset_providers"]["energon"] == (_PROVIDER_PATH)
    assert _PROVIDER_MODULE not in sys.modules


def test_mock_provider_resolves_without_importing_energon(monkeypatch):
    pretrain = importlib.import_module("examples.multimodal_dev.pretrain_multimodal")
    models = importlib.import_module("examples.multimodal_dev.models")
    real_import_module = pretrain.importlib.import_module
    imported = []

    def guarded_import(module_name, package=None):
        imported.append(module_name)
        if module_name in (_PROVIDER_MODULE, "megatron.energon"):
            raise AssertionError("mock provider imported optional Energon support")
        return real_import_module(module_name, package)

    monkeypatch.setattr(pretrain.importlib, "import_module", guarded_import)
    provider = pretrain._resolve_provider_fn(
        models.MODEL_REGISTRY["qwen35_vl"]["dataset_providers"]["mock"]
    )

    assert callable(provider)
    assert _PROVIDER_MODULE not in imported
    assert "megatron.energon" not in imported


def test_selected_provider_reports_missing_energon_actionably(monkeypatch):
    pretrain = importlib.import_module("examples.multimodal_dev.pretrain_multimodal")
    provider_fn = pretrain._resolve_provider_fn(_PROVIDER_PATH)
    provider_module = importlib.import_module(_PROVIDER_MODULE)

    def missing_energon(_module_name):
        raise ModuleNotFoundError("No module named 'megatron.energon'", name="megatron.energon")

    monkeypatch.setattr(provider_module, "import_module", missing_energon)
    with pytest.raises(
        provider_module.EnergonCompatibilityError,
        match=r"(?i)requires megatron-energon>=7\.0\.0.*install or upgrade",
    ):
        provider_fn(None)


def test_selected_provider_rejects_older_energon(monkeypatch):
    provider = importlib.import_module(_PROVIDER_MODULE)
    monkeypatch.setattr(provider, "import_module", lambda _name: _fake_energon("6.9.0"))

    with pytest.raises(
        provider.EnergonCompatibilityError,
        match=r"requires megatron-energon>=7\.0\.0.*found 6\.9\.0",
    ):
        provider.load_energon7_api()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda module: setattr(module, "TaskEncoder", type("TaskEncoder", (), {})),
            "TaskEncoder.preencode_sample",
        ),
        (
            lambda module: setattr(
                module, "get_train_dataset", lambda path, **kwargs: (path, kwargs)
            ),
            "get_train_dataset.*worker_config",
        ),
    ],
)
def test_energon7_guard_rejects_old_hook_and_worker_ownership_contracts(
    monkeypatch, mutate, message
):
    provider = importlib.import_module(_PROVIDER_MODULE)
    module = _fake_energon()
    mutate(module)
    monkeypatch.setattr(provider, "import_module", lambda _name: module)

    with pytest.raises(provider.EnergonCompatibilityError, match=message):
        provider.load_energon7_api()


def test_energon7_loader_boundary_does_not_repass_worker_config(monkeypatch):
    provider = importlib.import_module(_PROVIDER_MODULE)
    module = _fake_energon()
    calls = []

    def get_loader(dataset, **kwargs):
        calls.append(("plain", dataset, kwargs))
        return "plain-loader"

    def get_savable_loader(dataset, **kwargs):
        calls.append(("savable", dataset, kwargs))
        return "savable-loader"

    module.get_loader = get_loader
    module.get_savable_loader = get_savable_loader
    monkeypatch.setattr(provider, "import_module", lambda _name: module)
    api = provider.load_energon7_api()
    dataset = object()

    assert provider.build_loader(api, dataset, savable=False, prefetch_factor=3) == ("plain-loader")
    assert provider.build_loader(api, dataset, savable=True, prefetch_factor=5) == (
        "savable-loader"
    )
    assert calls == [
        ("plain", dataset, {"prefetch_factor": 3}),
        ("savable", dataset, {"prefetch_factor": 5}),
    ]


def test_installed_energon_satisfies_g1_contract():
    provider = importlib.import_module(_PROVIDER_MODULE)
    api = provider.load_energon7_api()

    assert api.major_version >= 7
    assert hasattr(api.task_encoder_type, "preencode_sample")
    assert "worker_config" in inspect.signature(api.get_train_dataset).parameters
    assert "worker_config" in inspect.signature(api.get_val_datasets).parameters
