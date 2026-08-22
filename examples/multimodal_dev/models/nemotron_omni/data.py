# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Image-only mock providers for canonical expanded Nemotron Omni batches."""

import torch

from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset
from examples.multimodal_dev.data.mock import MockQwen35VLDataset
from examples.multimodal_dev.models.nemotron_omni.configuration import IMAGE_TOKEN_ID

_VISION_START_TOKEN_ID = 19
_VIDEO_TOKEN_ID = 17


class _ExpandedNemotronMockDataset(MockQwen35VLDataset):
    """Reuse deterministic patch payload generation with ordinary 1-D RoPE positions."""

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["position_ids"] = torch.arange(sample["input_ids"].numel(), dtype=torch.long)
        return sample


def mock_dataset_provider(train_val_test_num_samples):
    """Provide native image-only expanded-sequence mock datasets."""
    from megatron.training import get_args

    args = get_args()
    kwargs = dict(
        seq_length=getattr(args, "total_seq_length", 1024),
        image_seq_length=2**31 - 1,
        vocab_size=getattr(args, "padded_vocab_size", 131072),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        image_size=getattr(args, "image_size", 224),
        patch_size=16,
        temporal_patch_size=1,
        spatial_merge_size=2,
    )
    return tuple(
        _ExpandedNemotronMockDataset(num_samples=count, **kwargs)
        for count in train_val_test_num_samples
    )


def mdp_mock_dataset_provider(train_val_test_num_samples):
    """Provide THD image-only mocks with 768-wide RADIO patch rows."""
    from megatron.training import get_args

    args = get_args()
    scenarios = (
        (((1, 8, 8),), (16, 16)),
        (((1, 4, 8), (1, 6, 6)), (8, 12, 8)),
        ((), (48,)),
        (((1, 4, 4), (1, 4, 6)), (10, 6, 10)),
    )
    kwargs = dict(
        vocab_size=getattr(args, "padded_vocab_size", 131072),
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        temporal_patch_size=1,
        spatial_merge_size=2,
        scenarios=scenarios,
    )
    return tuple(
        MdpThdMockDataset(num_samples=count, seed=1234 + split, **kwargs)
        for split, count in enumerate(train_val_test_num_samples)
    )
