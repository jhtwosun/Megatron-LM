# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Energon 7 API boundary for the Qwen3.5-VL dataset provider."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from typing import Any

from examples.multimodal_dev.models.qwen35_vl.configuration import QWEN35_VL_IMAGE_TOKEN_ID
from megatron.core import parallel_state
from megatron.training import get_args

_MINIMUM_ENERGON_VERSION = "7.0.0"


class EnergonCompatibilityError(RuntimeError):
    """The selected Energon installation does not satisfy the provider contract."""


@dataclass(frozen=True)
class Energon7Api:
    """Validated Energon symbols used by the Qwen3.5-VL provider."""

    version: str
    major_version: int
    worker_config_type: type
    task_encoder_type: type
    get_train_dataset: Callable[..., Any]
    get_val_datasets: Callable[..., Any]
    get_loader: Callable[..., Any]
    get_savable_loader: Callable[..., Any]


def _compatibility_error(detail: str) -> EnergonCompatibilityError:
    return EnergonCompatibilityError(
        "Qwen3.5-VL dataset provider 'energon' requires "
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
        "get_train_dataset",
        "get_val_datasets",
        "get_loader",
        "get_savable_loader",
    )
    missing = tuple(name for name in required_names if not hasattr(energon, name))
    if missing:
        raise _compatibility_error(f"missing required API symbols {missing}")
    for name in ("WorkerConfig", "TaskEncoder"):
        if not isinstance(getattr(energon, name), type):
            raise _compatibility_error(f"{name} must be a class")
    for name in ("get_train_dataset", "get_val_datasets", "get_loader", "get_savable_loader"):
        if not callable(getattr(energon, name)):
            raise _compatibility_error(f"{name} must be callable")
    if not callable(getattr(energon.TaskEncoder, "preencode_sample", None)):
        raise _compatibility_error("TaskEncoder.preencode_sample is unavailable")

    _require_keyword_only_parameter(energon.get_train_dataset, "get_train_dataset", "worker_config")
    _require_keyword_only_parameter(energon.get_val_datasets, "get_val_datasets", "worker_config")
    return Energon7Api(
        version=version,
        major_version=major_version,
        worker_config_type=energon.WorkerConfig,
        task_encoder_type=energon.TaskEncoder,
        get_train_dataset=energon.get_train_dataset,
        get_val_datasets=energon.get_val_datasets,
        get_loader=energon.get_loader,
        get_savable_loader=energon.get_savable_loader,
    )


def build_loader(api: Energon7Api, dataset: Any, *, savable: bool, prefetch_factor: int) -> Any:
    """Build a loader without re-passing the dataset-owned WorkerConfig."""
    loader_fn = api.get_savable_loader if savable else api.get_loader
    return loader_fn(dataset, prefetch_factor=prefetch_factor)


def _validate_provider_args(args) -> None:
    if getattr(args, "dataloader_type", "single") != "external":
        raise ValueError("--dataloader-type external is required with --dataset-provider energon")
    if int(getattr(args, "micro_batch_size", 1)) != 1:
        raise ValueError("Qwen3.5-VL Energon packing requires --micro-batch-size 1")
    if not bool(getattr(args, "use_packed_sequence", False)):
        raise ValueError("Qwen3.5-VL Energon packing requires --use-packed-sequence")
    if not getattr(args, "energon_path", None):
        raise ValueError("--energon-path is required with --dataset-provider energon")
    if parallel_state.model_parallel_is_initialized():
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
    else:
        tp_size = int(getattr(args, "tensor_model_parallel_size", 1))
    if tp_size != 1:
        raise ValueError("Qwen3.5-VL Energon packing does not yet support tensor parallelism")
    for name in (
        "energon_packing_buffer_size",
        "energon_max_samples_per_sequence",
        "energon_shuffle_buffer_size",
        "energon_prefetch_factor",
    ):
        if int(getattr(args, name, 0)) <= 0:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be positive")


def _worker_config(api: Energon7Api, args):
    workers = int(getattr(args, "num_workers", 2))
    if workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if parallel_state.model_parallel_is_initialized():
        return api.worker_config_type(
            rank=parallel_state.get_data_parallel_rank(),
            world_size=parallel_state.get_data_parallel_world_size(),
            num_workers=workers,
            data_parallel_group=parallel_state.get_data_parallel_group(),
        )
    return api.worker_config_type(rank=0, world_size=1, num_workers=workers)


def _tokenizer(args):
    tokenizer_path = getattr(args, "tokenizer_model", None)
    if not tokenizer_path:
        raise ValueError(
            "--tokenizer-model must name a Qwen3.5-VL tokenizer with " "--dataset-provider energon"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if int(image_token_id) != QWEN35_VL_IMAGE_TOKEN_ID:
        raise ValueError(
            "--tokenizer-model is not a Qwen3.5-VL tokenizer: "
            f"<|image_pad|>={image_token_id}, expected {QWEN35_VL_IMAGE_TOKEN_ID}"
        )
    return tokenizer


def _build_task_encoder(api: Energon7Api, args):
    from .task_encoder import Qwen35EnergonTaskEncoder

    if not issubclass(Qwen35EnergonTaskEncoder, api.task_encoder_type):
        raise _compatibility_error("Qwen35EnergonTaskEncoder does not inherit Energon TaskEncoder")
    if parallel_state.model_parallel_is_initialized():
        cp_size = parallel_state.get_context_parallel_world_size()
    else:
        cp_size = int(getattr(args, "context_parallel_size", 1))
    return Qwen35EnergonTaskEncoder(
        tokenizer=_tokenizer(args),
        seq_length=int(args.seq_length),
        image_token_id=int(getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID)),
        patch_size=int(getattr(args, "vision_patch_size", 16)),
        temporal_patch_size=int(getattr(args, "vision_temporal_patch_size", 2)),
        spatial_merge_size=int(getattr(args, "vision_spatial_merge_size", 2)),
        context_parallel_size=int(cp_size),
    )


def train_valid_test_datasets_provider(_train_val_test_num_samples):
    """Return Energon-owned external loaders for Qwen3.5-VL packing."""
    api = load_energon7_api()
    args = get_args()
    _validate_provider_args(args)
    worker_config = _worker_config(api, args)
    task_encoder = _build_task_encoder(api, args)
    common = {
        "worker_config": worker_config,
        "task_encoder": task_encoder,
        "batch_size": 1,
        "packing_buffer_size": int(args.energon_packing_buffer_size),
        "max_samples_per_sequence": int(args.energon_max_samples_per_sequence),
    }
    train_dataset = api.get_train_dataset(
        args.energon_path,
        split_part=args.energon_split,
        shuffle_buffer_size=int(args.energon_shuffle_buffer_size),
        **common,
    )
    train_loader = build_loader(
        api, train_dataset, savable=True, prefetch_factor=int(args.energon_prefetch_factor)
    )

    valid_loader = None
    if int(getattr(args, "eval_iters", 0)) > 0:
        valid_datasets = api.get_val_datasets(
            args.energon_path, split_part=args.energon_val_split, **common
        )
        if valid_datasets:
            valid_loader = build_loader(
                api,
                valid_datasets[0][0],
                savable=False,
                prefetch_factor=int(args.energon_prefetch_factor),
            )
    return train_loader, valid_loader, None
