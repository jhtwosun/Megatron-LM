# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generic Energon 7 dataset-provider boundary."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from typing import Any

from megatron.core import parallel_state
from megatron.training import get_args

_MINIMUM_ENERGON_VERSION = "7.0.0"
_TASK_ENCODER_HOOKS = ("preencode_sample", "encode_sample", "batch", "encode_batch")


class EnergonCompatibilityError(RuntimeError):
    """The selected Energon installation does not satisfy the provider contract."""


@dataclass(frozen=True)
class Energon7Api:
    """Validated Energon symbols used by the generic provider."""

    version: str
    major_version: int
    worker_config_type: type
    task_encoder_type: type
    metadataset_v2_type: type
    get_train_dataset: Callable[..., Any]
    get_val_datasets: Callable[..., Any]
    get_loader: Callable[..., Any]
    get_savable_loader: Callable[..., Any]


def _compatibility_error(detail: str) -> EnergonCompatibilityError:
    return EnergonCompatibilityError(
        "Dataset provider 'energon' requires "
        f"megatron-energon>={_MINIMUM_ENERGON_VERSION}; {detail}. "
        "Install or upgrade Megatron Energon before selecting this provider"
    )


def _require_keyword_only_parameter(function: Callable[..., Any], owner: str, name: str) -> None:
    try:
        parameter = signature(function).parameters.get(name)
    except (TypeError, ValueError) as exc:
        raise _compatibility_error(f"cannot inspect {owner}") from exc
    if (
        parameter is None
        or parameter.kind is not Parameter.KEYWORD_ONLY
        or parameter.default is not Parameter.empty
    ):
        raise _compatibility_error(f"{owner} must require keyword-only {name}")


def _require_loader_signature(function: Callable[..., Any], owner: str) -> None:
    try:
        loader_signature = signature(function)
    except (TypeError, ValueError) as exc:
        raise _compatibility_error(f"cannot inspect {owner}") from exc
    parameters = tuple(loader_signature.parameters.values())
    if (
        not parameters
        or parameters[0].kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        or parameters[0].default is not Parameter.empty
    ):
        raise _compatibility_error(f"{owner} first argument must be a required positional dataset")
    by_name = {parameter.name: parameter for parameter in parameters}
    worker_config = by_name.get("worker_config")
    if worker_config is None or worker_config.default is not None:
        raise _compatibility_error(f"{owner} worker_config must be omittable and default to None")
    prefetch_factor = by_name.get("prefetch_factor")
    if prefetch_factor is None or prefetch_factor.kind not in (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    ):
        raise _compatibility_error(f"{owner} prefetch_factor must accept a keyword argument")
    try:
        loader_signature.bind(object(), prefetch_factor=1)
    except TypeError as exc:
        raise _compatibility_error(
            f"{owner} cannot be called with a positional dataset and keyword prefetch_factor"
        ) from exc


def load_energon7_api() -> Energon7Api:
    """Import and validate Energon only after its provider is selected."""
    try:
        energon = import_module("megatron.energon")
    except ModuleNotFoundError as exc:
        if exc.name not in ("megatron", "megatron.energon"):
            raise
        raise _compatibility_error("the optional 'megatron.energon' module is unavailable") from exc

    version = str(getattr(energon, "__version__", ""))
    try:
        major_version = int(version.split(".", 1)[0])
    except ValueError as exc:
        raise _compatibility_error(
            f"the installed module does not expose a valid version (found {version!r})"
        ) from exc
    if major_version < 7:
        raise _compatibility_error(f"found {version}")

    required_names = (
        "WorkerConfig",
        "TaskEncoder",
        "MetadatasetV2",
        "get_train_dataset",
        "get_val_datasets",
        "get_loader",
        "get_savable_loader",
    )
    missing = tuple(name for name in required_names if not hasattr(energon, name))
    if missing:
        raise _compatibility_error(f"missing required API symbols {missing}")
    for name in ("WorkerConfig", "TaskEncoder", "MetadatasetV2"):
        if not isinstance(getattr(energon, name), type):
            raise _compatibility_error(f"{name} must be a class")
    for name in ("get_train_dataset", "get_val_datasets", "get_loader", "get_savable_loader"):
        if not callable(getattr(energon, name)):
            raise _compatibility_error(f"{name} must be callable")
    for hook in _TASK_ENCODER_HOOKS:
        if not callable(getattr(energon.TaskEncoder, hook, None)):
            raise _compatibility_error(f"TaskEncoder.{hook} is unavailable")

    _require_keyword_only_parameter(energon.get_train_dataset, "get_train_dataset", "worker_config")
    _require_keyword_only_parameter(energon.get_val_datasets, "get_val_datasets", "worker_config")
    _require_loader_signature(energon.get_loader, "get_loader")
    _require_loader_signature(energon.get_savable_loader, "get_savable_loader")
    return Energon7Api(
        version=version,
        major_version=major_version,
        worker_config_type=energon.WorkerConfig,
        task_encoder_type=energon.TaskEncoder,
        metadataset_v2_type=energon.MetadatasetV2,
        get_train_dataset=energon.get_train_dataset,
        get_val_datasets=energon.get_val_datasets,
        get_loader=energon.get_loader,
        get_savable_loader=energon.get_savable_loader,
    )


def _require_integer(args: Any, name: str, *, positive: bool) -> int:
    value = getattr(args, name, None)
    option = "--" + name.replace("_", "-")
    if type(value) is not int or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{option} must be a {qualifier} integer")
    return value


def _validate_provider_args(args: Any) -> None:
    if getattr(args, "dataloader_type", None) != "external":
        raise ValueError("--dataloader-type external is required with --dataset-provider energon")
    try:
        path = os.fspath(getattr(args, "energon_path", None))
    except TypeError as exc:
        raise ValueError("--energon-path is required with --dataset-provider energon") from exc
    if not isinstance(path, str) or not path.strip():
        raise ValueError("--energon-path is required with --dataset-provider energon")
    for name in ("energon_split", "energon_val_split"):
        value = getattr(args, name, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"--{name.replace('_', '-')} must be a non-empty string")
    _require_integer(args, "num_workers", positive=False)
    for name in (
        "energon_packing_buffer_size",
        "energon_max_samples_per_sequence",
        "energon_shuffle_buffer_size",
        "energon_prefetch_factor",
    ):
        _require_integer(args, name, positive=True)


def _resolve_callable(spec: Any, *, owner: str) -> Callable[..., Any]:
    if callable(spec):
        return spec
    if not isinstance(spec, str) or "." not in spec:
        raise TypeError(f"{owner} must be a callable or dotted import path")
    module_name, attribute = spec.rsplit(".", 1)
    resolved = getattr(import_module(module_name), attribute)
    if not callable(resolved):
        raise TypeError(f"{owner} must resolve to a callable")
    return resolved


def _build_task_encoder(api: Energon7Api, args: Any) -> Any:
    from examples.multimodal_dev.models import MODEL_REGISTRY

    model_arch = getattr(args, "model_arch", None)
    registry = MODEL_REGISTRY.get(model_arch)
    factory_spec = None if registry is None else registry.get("energon_task_encoder_factory")
    if factory_spec is None:
        raise NotImplementedError(
            f"Model {model_arch!r} does not define energon_task_encoder_factory"
        )
    factory = _resolve_callable(factory_spec, owner=f"{model_arch!r} energon_task_encoder_factory")
    task_encoder = factory(args=args, energon_api=api)
    if not isinstance(task_encoder, api.task_encoder_type):
        raise TypeError(
            f"Model {model_arch!r} energon_task_encoder_factory must return "
            "a megatron.energon.TaskEncoder"
        )
    return task_encoder


def _worker_config(api: Energon7Api, args: Any) -> Any:
    return api.worker_config_type(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=args.num_workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
    )


def _build_loader(api: Energon7Api, dataset: Any, *, savable: bool, prefetch_factor: int) -> Any:
    """Build a loader without re-passing the dataset-owned WorkerConfig."""
    loader_fn = api.get_savable_loader if savable else api.get_loader
    return loader_fn(dataset, prefetch_factor=prefetch_factor)


def _validation_datasets(entries: Any) -> list[Any]:
    if not entries:
        raise ValueError("Energon returned no validation datasets")
    datasets = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError(
                "Each Energon validation dataset entry must be a (dataset, factory) pair"
            )
        datasets.append(entry[0])
    return datasets


def train_valid_test_datasets_provider(_train_val_test_num_samples: Any):
    """Return Energon-owned external iterators for the selected model."""
    api = load_energon7_api()
    args = get_args()
    _validate_provider_args(args)
    worker_config = _worker_config(api, args)
    task_encoder = _build_task_encoder(api, args)
    common = {
        "worker_config": worker_config,
        "task_encoder": task_encoder,
        "batch_size": args.micro_batch_size,
        "packing_buffer_size": args.energon_packing_buffer_size,
    }
    train_dataset = api.get_train_dataset(
        args.energon_path,
        split_part=args.energon_split,
        shuffle_buffer_size=args.energon_shuffle_buffer_size,
        max_samples_per_sequence=args.energon_max_samples_per_sequence,
        **common,
    )
    train_iterator = iter(
        _build_loader(
            api, train_dataset, savable=True, prefetch_factor=args.energon_prefetch_factor
        )
    )

    valid_iterators = None
    if int(getattr(args, "eval_iters", 0)) > 0:
        entries = api.get_val_datasets(
            args.energon_path, split_part=args.energon_val_split, **common
        )
        valid_iterators = [
            iter(
                _build_loader(
                    api, dataset, savable=False, prefetch_factor=args.energon_prefetch_factor
                )
            )
            for dataset in _validation_datasets(entries)
        ]
    return train_iterator, valid_iterators, None
