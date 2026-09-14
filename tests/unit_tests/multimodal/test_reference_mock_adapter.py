"""Allocated-only reference5885 content and legacy descriptor integration checks."""

import importlib.util
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.data import reference_mock
from examples.multimodal_dev.data.blend_dataset import Qwen35VLDataset
from examples.multimodal_dev.data.dataset_utils import RawSample
from examples.multimodal_dev.data.mock_mdp import MockBackend

CONFIG = {"mode": "distribution", "type": "lognormal", "format": "thd",
          "min_seq_len": 512, "max_seq_len": 4096, "mean_seq_len": 2048,
          "lognormal_sigma": 1.1}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(monkeypatch):
    # Caller mounts the immutable reference clone read-only; no network or model download.
    root = Path(os.environ["PR7_REFERENCE_SOURCE"])
    data = root / "examples/multimodal_dev/data"
    scenarios = load_module("reference_oracle_scenarios", data / "mdp_scenarios.py")
    monkeypatch.setitem(__import__("sys").modules,
                        "examples.multimodal_dev.data.mdp_scenarios", scenarios)
    oracle = load_module("reference_oracle_dataset", data / "mdp_mock.py")
    monkeypatch.setattr(reference_mock, "get_args", lambda: SimpleNamespace(
        padded_vocab_size=248448, image_token_id=248056))
    adapter = reference_mock.ReferenceMock(json.dumps(CONFIG), "train")
    expected_scenarios = scenarios.build_scenarios(length_config=CONFIG)
    assert adapter.scenarios == expected_scenarios
    return adapter, oracle.MdpThdMockDataset(
        num_samples=64, vocab_size=248448, scenarios=expected_scenarios)


def dataset(metadata=False):
    ds = Qwen35VLDataset.__new__(Qwen35VLDataset)
    ds.seq_length = 16384
    ds.image_token_id, ds.video_token_id, ds.vision_start_token_id = 248056, 248057, 248053
    ds.spatial_merge_size, ds.patch_size, ds.temporal_patch_size = 2, 16, 2
    ds.image_size_max = ds.image_max_pixels = ds.image_min_pixels = 0
    ds._pixel_dim = 1536
    ds.metadata_only_batch = metadata
    ds.cp_size, ds.emit_cu_seqlens, ds.align = 2, True, 64
    ds.mdp_loader_prepartition = False
    return ds


def test_all_reference_scenarios_exact(source):
    adapter, oracle = source
    ds = dataset()
    saw_empty = saw_temporal = saw_multi = False
    for idx in range(64):
        rng = torch.get_rng_state().clone()
        raw, expected = adapter[idx], oracle[idx]
        assert torch.equal(rng, torch.get_rng_state())
        doc = ds._build_doc(raw)
        for key, actual in (("input_ids", doc["input_ids"]),
                            ("labels", doc["reference_labels"]),
                            ("loss_mask", doc["reference_loss_mask"]),
                            ("image_grid_thw", doc["image_grid_thw"]),
                            ("pixel_values", doc["pixel_values"])):
            assert torch.equal(actual, expected[key]), (idx, key)
        grids = expected["image_grid_thw"]
        saw_empty |= grids.shape[0] == 0
        saw_multi |= grids.shape[0] > 1
        saw_temporal |= bool((grids[:, 0] > 1).any())
        lazy = dataset(metadata=True)._build_doc(raw)
        assert lazy["pixel_values"].numel() == 0
        assert lazy["_mdp_image_descriptors"] == doc["_mdp_image_descriptors"]
    assert saw_empty and saw_temporal and saw_multi


def test_packed_content_masks_preserve_video_and_boundary():
    ds = dataset()
    ids = torch.tensor([3, ds.video_token_id, 4])
    raw = RawSample([], "", [], ids, torch.tensor([ds.video_token_id, 4, 0]),
                    torch.tensor([1., 1., 0.]))
    doc = ds._build_doc(raw)
    packed_ids = torch.cat([ids, torch.zeros(61, dtype=torch.long), ids,
                            torch.zeros(61, dtype=torch.long), torch.zeros(64, dtype=torch.long)])
    labels, mask = ds._shifted_labels_and_loss_mask(
        packed_ids, [64, 64], [3, 3], reference_docs=[doc, doc])
    for start in (0, 64):
        assert torch.equal(labels[start:start + 3], raw.reference_labels)
        assert torch.equal(mask[start:start + 3], raw.reference_loss_mask)
        assert mask[start + 3:start + 64].sum() == 0
    assert mask[128:].sum() == 0
    legacy_labels, legacy_mask = ds._shifted_labels_and_loss_mask(packed_ids, [64, 64], [3, 3])
    assert legacy_mask[1] == 0 and legacy_labels[1] == -100


def test_default_backend_and_missing_fields(monkeypatch):
    monkeypatch.delenv("PR7_REFERENCE_MOCK_CONFIG", raising=False)
    backend = MockBackend(max_samples=10)
    raw = backend[0]
    assert raw.reference_input_ids is None and raw.text
    rng = random.Random(1729)
    count = rng.randint(1, 6)
    sizes = [rng.randint(224, 512) for _ in range(count)]
    words = rng.randint(192, 768)
    expected_ids = [100, 200 + count] + [300 + rng.randrange(10000) for _ in range(words)]
    assert raw.text == " ".join(map(str, expected_ids))
    assert [d["width"] for d in raw.image_descriptors] == sizes
    assert [d["seed"] for d in raw.image_descriptors] == [1729 * 1000003 + i for i in range(count)]
    raw.reference_input_ids = torch.tensor([1])
    with pytest.raises(ValueError, match="together"):
        dataset()._build_doc(raw)


def test_reject_truncation_bad_grid_and_placeholder(source):
    adapter, _ = source
    raw = adapter[0]
    ds = dataset()
    ds.seq_length = 1
    with pytest.raises(ValueError, match="truncation"):
        ds._build_doc(raw)
    raw.image_descriptors[0]["grid_thw"][0] = 0
    with pytest.raises(ValueError, match="grid"):
        dataset()._build_doc(raw)
    raw = adapter[0]
    raw.reference_input_ids[raw.reference_input_ids == 248056] = 1
    with pytest.raises(ValueError, match="placeholder"):
        dataset()._build_doc(raw)


def test_actual_finalizers_preserve_content_fields(source):
    adapter, _ = source
    ds = dataset()
    docs = [ds._build_doc(adapter[i]) for i in (0, 4)]
    single = ds._finalize_container(docs[0])
    n = docs[0]["content_len"]
    assert torch.equal(single["labels"][:n], docs[0]["reference_labels"])
    assert torch.equal(single["loss_mask"][:n], docs[0]["reference_loss_mask"])
    assert single["loss_mask"][n:].sum() == 0
    packed = ds._finalize_packed_container(docs)
    offset = 0
    for doc in docs:
        n = doc["content_len"]
        assert torch.equal(packed["labels"][offset:offset + n], doc["reference_labels"])
        assert torch.equal(packed["loss_mask"][offset:offset + n], doc["reference_loss_mask"])
        padded = ((n + 63) // 64) * 64
        assert packed["loss_mask"][offset + n:offset + padded].sum() == 0
        offset += padded
    assert packed["loss_mask"][offset:].sum() == 0
