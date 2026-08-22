# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""G4 real-data functional gate for the Qwen3.5-VL Energon boundary.

Run this module alone with ``RUN_QWEN35_REAL_BLEND_G4=1``.  It intentionally
requires the real prepared shards, tokenizer assets, and staged external image
containers named by the environment.  Ordinary unit suites skip the module.
"""

import hashlib
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from examples.multimodal_dev import forward_step
from examples.multimodal_dev.data.qwen35_energon import provider
from examples.multimodal_dev.data.qwen35_energon.materializer import materialize_image_descriptors
from examples.multimodal_dev.models.qwen35_vl.configuration import MROPE_SECTION
from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_ENABLED = os.environ.get("RUN_QWEN35_REAL_BLEND_G4") == "1"
pytestmark = pytest.mark.skipif(not _ENABLED, reason="explicit G4 real-data gate")

_BLEND_SHA = "8feb037cc21651e82a03945c02b4124c913a16ae45e61842db3f0edc1122c6d8"
_SHARDS = {
    "mantis": (
        "078e0699544183c2050a8a7cb3c20290b245018288c1646c7aa5e51273d94452",
        "sample_000000000.json",
        "ff5b0eb1b0915ea0a9027e6fcadab39ab64e00fc7e73b37df273ef25027bde07",
    ),
    "m4": (
        "2ab66da4f6acb40b30240f3ca4eff452032d94ed19858b923b75b40c8ed6a3f8",
        "sample_000000002.json",
        "41748376218f3cfea90353f5b2e241e0473849c6b542e2d26443b47debd6a0b3",
    ),
    "pixmo": (
        "b4432a224f15dd2383ba8aaa68600d270ef56df58b1e9e052f4b555180959884",
        "sample_000000000.json",
        "13cbc6c1585bbe74476243371184fc05b4cbc247aa9e9613c8d60484d2f889d4",
    ),
}


def _required_env(name):
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required for the G4 real-data gate")
    path = Path(value)
    assert path.exists(), (name, path)
    return path


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample(dataset_root, dataset):
    shard = dataset_root / f"qwen35-energon-lazy-{dataset}" / "shards/shard-000000.tar"
    shard_sha, member, member_sha = _SHARDS[dataset]
    assert _sha256_file(shard) == shard_sha
    with tarfile.open(shard) as archive:
        payload_bytes = archive.extractfile(member).read()
    assert _sha256_bytes(payload_bytes) == member_sha
    return json.loads(payload_bytes)


def _provider_args(path, tokenizer_path, *, seq_length=2048):
    return SimpleNamespace(
        dataloader_type="external",
        micro_batch_size=1,
        use_packed_sequence=True,
        energon_path=str(path),
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        num_workers=0,
        energon_packing_buffer_size=1,
        energon_max_samples_per_sequence=1,
        energon_shuffle_buffer_size=1,
        energon_prefetch_factor=1,
        tokenizer_model=str(tokenizer_path),
        seq_length=seq_length,
        image_token_id=248056,
        vision_patch_size=16,
        vision_temporal_patch_size=2,
        vision_spatial_merge_size=2,
        energon_split="train",
        energon_val_split="val",
        eval_iters=0,
        mdp_enable=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    assert int(os.environ.get("WORLD_SIZE", "0")) == 1, "run the G4 gate with torchrun world1"
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield
    Utils.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    assert not torch.distributed.is_initialized()


def test_real_blend_yaml_shards_and_external_descriptor_kinds_are_exact(tmp_path, monkeypatch):
    blend = _required_env("QWEN35_REAL_BLEND_YAML")
    dataset_root = _required_env("QWEN35_REAL_DATASET_ROOT")
    m4_zip = _required_env("QWEN35_G4_STAGED_M4_ZIP")
    pixmo_parquet = _required_env("QWEN35_G4_STAGED_PIXMO_PARQUET")
    assert _sha256_file(blend) == _BLEND_SHA
    canonical = yaml.safe_load(blend.read_text())
    train = canonical["splits"]["train"]["blend"]
    assert [entry["weight"] for entry in train] == [1, 1, 1]
    assert [entry["subflavors"]["source_dataset"] for entry in train] == ["mantis", "m4", "pixmo"]

    mantis = _sample(dataset_root, "mantis")
    m4 = _sample(dataset_root, "m4")
    pixmo = _sample(dataset_root, "pixmo")
    assert mantis["image_descriptors"][1]["grid_thw"] == [1, 28, 40]
    assert len(m4["image_descriptors"]) == 12
    assert {tuple(item["grid_thw"]) for item in m4["image_descriptors"]} == {(1, 10, 10)}
    assert pixmo["image_descriptors"][0]["grid_thw"] == [1, 90, 62]

    m4_descriptor = dict(m4["image_descriptors"][0])
    assert m4_descriptor["kind"] == "zip_image"
    assert m4_descriptor["zip_path"] == "/mnt/datasets/M4-Instruct-Data/RAVEN_train_images.zip"
    m4_descriptor["zip_path"] = str(m4_zip)
    m4_pixels = materialize_image_descriptors(
        [m4_descriptor], [(1, 10, 10)], patch_size=16, temporal_patch_size=2, spatial_merge_size=2
    )
    assert tuple(m4_pixels.shape) == (100, 1536)

    pixmo_descriptor = dict(pixmo["image_descriptors"][0])
    assert pixmo_descriptor["kind"] == "parquet_column_image"
    assert pixmo_descriptor["row_idx"] == 1289 and pixmo_descriptor["column"] == "image"
    pixmo_descriptor["parquet_path"] = str(pixmo_parquet)
    pixmo_pixels = materialize_image_descriptors(
        [pixmo_descriptor],
        [(1, 90, 62)],
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
    )
    assert tuple(pixmo_pixels.shape) == (5580, 1536)
    assert torch.isfinite(m4_pixels).all() and torch.isfinite(pixmo_pixels).all()

    # Preserve the untouched canonical YAML; this derivative removes only the
    # inaccessible val/test paths and is an explicit runtime artifact.
    derivative = tmp_path / "train-only-metadataset.yaml"
    derivative.write_text(
        yaml.safe_dump({**canonical, "splits": {"train": canonical["splits"]["train"]}})
    )
    assert _sha256_file(blend) == _BLEND_SHA
    assert yaml.safe_load(derivative.read_text())["splits"]["train"] == canonical["splits"]["train"]

    tokenizer_path = _required_env("QWEN35_ACTUAL_TOKENIZER_PATH")
    args = _provider_args(derivative, tokenizer_path, seq_length=16384)
    monkeypatch.setattr(provider, "get_args", lambda: args)
    blended_train, valid, test = provider.train_valid_test_datasets_provider(None)
    assert valid is None and test is None
    blended_sample = next(iter(blended_train))
    assert int(blended_sample["qwen35_energon_prepacked"].reshape(-1)[0]) == 1
    assert len(blended_sample["image_descriptors"]) > 0


def _vision_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=64,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        apply_rope_fusion=False,
    )


def _language_config():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        apply_rope_fusion=False,
    )
    config.mrope_section = list(MROPE_SECTION)
    config.mrope_interleaved = True
    return config


def test_real_mantis_provider_get_batch_and_tiny_qwen35_forward_backward(monkeypatch):
    dataset_root = _required_env("QWEN35_REAL_DATASET_ROOT")
    tokenizer_path = _required_env("QWEN35_ACTUAL_TOKENIZER_PATH")
    args = _provider_args(dataset_root / "qwen35-energon-lazy-mantis", tokenizer_path)
    monkeypatch.setattr(provider, "get_args", lambda: args)
    monkeypatch.setattr(forward_step, "get_args", lambda: args)

    train, valid, test = provider.train_valid_test_datasets_provider(None)
    assert valid is None and test is None
    packed = next(iter(train))
    assert int(packed["qwen35_energon_prepacked"].reshape(-1)[0]) == 1
    assert len(packed["image_descriptors"]) > 0

    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    model = (
        Qwen35VLModel(
            language_config=_language_config(),
            language_spec=get_gpt_layer_with_transformer_engine_spec(),
            vision_config=_vision_config(),
            vocab_size=248320,
            max_sequence_length=args.seq_length,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
        )
        .bfloat16()
        .cuda()
    )
    output, loss_fn = forward_step.forward_step(iter([packed]), model)
    loss, tokens, report = loss_fn(output)
    assert int(tokens) > 0
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    assert torch.isfinite(report["lm loss"]).all()
    loss.backward()

    decoder_grads = [
        param.grad for param in model.language_model.parameters() if param.requires_grad
    ]
    vision_grads = [param.grad for param in model.vision_model.parameters() if param.requires_grad]
    assert decoder_grads and vision_grads
    assert all(grad is not None and bool(torch.isfinite(grad).all()) for grad in decoder_grads)
    assert all(grad is not None and bool(torch.isfinite(grad).all()) for grad in vision_grads)
    assert sum(float(grad.float().abs().sum()) for grad in decoder_grads) > 0
    assert sum(float(grad.float().abs().sum()) for grad in vision_grads) > 0
