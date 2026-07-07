# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron dataset provider for packed Qwen3.5-VL Energon data."""

from __future__ import annotations

from megatron.core import parallel_state
from megatron.energon import (
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)
from megatron.training import get_args

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
)

from .task_encoder import Qwen35EnergonTaskEncoder


def _worker_config(args) -> WorkerConfig:
    workers = int(getattr(args, "num_workers", 2))
    if not parallel_state.is_initialized():
        return WorkerConfig.default_worker_config(workers)
    return WorkerConfig(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
    )


def _tokenizer(args):
    tokenizer_path = getattr(args, "tokenizer_model", None)
    if not tokenizer_path:
        raise ValueError(
            "--tokenizer-model must point to a Qwen3.5-VL HuggingFace "
            "tokenizer when --dataset-provider energon is used"
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


def _task_encoder(args, tokenizer) -> Qwen35EnergonTaskEncoder:
    cp_size = (
        parallel_state.get_context_parallel_world_size()
        if parallel_state.model_parallel_is_initialized()
        else int(getattr(args, "context_parallel_size", 1))
    )
    return Qwen35EnergonTaskEncoder(
        tokenizer=tokenizer,
        seq_length=int(getattr(args, "total_seq_length", args.seq_length)),
        image_token_id=int(getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID)),
        patch_size=int(getattr(args, "vision_patch_size", 16)),
        temporal_patch_size=int(getattr(args, "vision_temporal_patch_size", 2)),
        spatial_merge_size=int(getattr(args, "vision_spatial_merge_size", 2)),
        image_min_pixels=int(getattr(args, "image_min_pixels", 0)),
        image_max_pixels=int(getattr(args, "image_max_pixels", 0)),
        context_parallel_size=int(cp_size),
    )


def train_valid_test_datasets_provider(_train_val_test_num_samples):
    """Return external Energon loaders for Megatron pretraining."""
    args = get_args()
    data_path = getattr(args, "energon_path", None)
    if not data_path:
        raise ValueError("--energon-path is required with --dataset-provider energon")
    if getattr(args, "dataloader_type", "single") != "external":
        raise ValueError(
            "--dataloader-type external is required with --dataset-provider energon"
        )
    if int(getattr(args, "micro_batch_size", 1)) != 1:
        raise ValueError("Qwen3.5-VL Energon packing requires --micro-batch-size 1")
    if int(args.energon_packing_buffer_size) < 1:
        raise ValueError("--energon-packing-buffer-size must be positive")
    if int(args.energon_max_samples_per_sequence) < 1:
        raise ValueError("--energon-max-samples-per-sequence must be positive")
    if int(args.energon_prefetch_factor) < 1:
        raise ValueError("--energon-prefetch-factor must be positive")
    if (
        parallel_state.model_parallel_is_initialized()
        and parallel_state.get_tensor_model_parallel_rank() != 0
    ):
        return None, None, None

    worker_config = _worker_config(args)
    tokenizer = _tokenizer(args)
    common = {
        "batch_size": 1,
        "worker_config": worker_config,
        "packing_buffer_size": int(args.energon_packing_buffer_size),
        "max_samples_per_sequence": int(args.energon_max_samples_per_sequence),
    }
    train_dataset = get_train_dataset(
        data_path,
        task_encoder=_task_encoder(args, tokenizer),
        split_part=args.energon_split,
        shuffle_buffer_size=int(args.energon_shuffle_buffer_size),
        **common,
    )
    train_loader = iter(
        get_savable_loader(
            train_dataset,
            prefetch_factor=int(args.energon_prefetch_factor),
        )
    )

    validation_datasets = get_val_datasets(
        data_path, task_encoder=_task_encoder(args, tokenizer), **common
    )
    validation_loader = (
        iter(get_loader(validation_datasets[0][0]))
        if validation_datasets
        else None
    )
    return train_loader, validation_loader, None
