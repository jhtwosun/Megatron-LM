# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3-VL-owned wrappers around the shared synthetic datasets."""

from examples.multimodal_dev.models.qwen3_vl.configuration import (
    IMAGE_TOKEN_ID,
    VIDEO_TOKEN_ID,
    VISION_START_TOKEN_ID,
)


def _validate_vocab(args) -> int:
    vocab_size = int(getattr(args, "padded_vocab_size", 151936))
    if max(IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, VISION_START_TOKEN_ID) >= vocab_size:
        raise ValueError(
            "Qwen3-VL mock data requires padded_vocab_size greater than all canonical "
            f"special-token IDs; got {vocab_size}."
        )
    return vocab_size


def mock_dataset_provider(train_val_test_num_samples):
    """Build shared mock samples with all Qwen3-VL token IDs overridden."""
    from examples.multimodal_dev.data.mock import MockQwen35VLDataset
    from megatron.training import get_args

    args = get_args()
    kwargs = dict(
        seq_length=getattr(args, "total_seq_length", 1024),
        image_seq_length=getattr(args, "image_seq_length", 256),
        vocab_size=_validate_vocab(args),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        image_size=getattr(args, "image_size", 224),
    )
    return tuple(
        MockQwen35VLDataset(num_samples=num_samples, **kwargs)
        for num_samples in train_val_test_num_samples
    )


def mdp_mock_dataset_provider(train_val_test_num_samples):
    """Build shared MDP mock samples with canonical Qwen3-VL IDs."""
    from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset
    from megatron.training import get_args

    args = get_args()
    kwargs = dict(
        vocab_size=_validate_vocab(args),
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
    )
    return tuple(
        MdpThdMockDataset(num_samples=num_samples, seed=1234 + split, **kwargs)
        for split, num_samples in enumerate(train_val_test_num_samples)
    )
