# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static native indexed capture: no pre-plan image reads, typed model handoff."""

from dataclasses import replace
import io
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.data.energon import materializer as api
from examples.multimodal_dev.tests.test_qwen35_energon_contract import _encoder, _sample
from examples.multimodal_dev.tests.test_static_energon_locators import indexed_dataset
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.vision_locator import VisionLocatorKind


def _native_sample(root, key):
    from megatron.energon.epathlib import EPath
    from megatron.energon.flavors.webdataset.itar_reader import SqliteITarEntryReader

    return SqliteITarEntryReader(EPath(root), part_filter=lambda part: part == "json")[key]


@pytest.mark.parametrize(
    "key,count,expected",
    [("one", 1, (b"first-jpeg",)), ("two", 2, (b"second-jpeg", b"third-jpeg"))],
)
def test_native_json_filter_freezes_indexed_images_without_byte_reads(
    indexed_dataset, monkeypatch, key, count, expected
):
    import tarfile
    from megatron.energon.cache.file_store import WebdatasetFileStore

    root, roots = indexed_dataset
    extract = tarfile.TarFile.extractfile
    reads = []

    def json_only(archive, member):
        name = member.name if hasattr(member, "name") else member
        reads.append(name)
        assert name.endswith(".json"), "pre-plan native reader extracted image bytes"
        return extract(archive, member)

    with monkeypatch.context() as capture:
        capture.setattr(tarfile.TarFile, "extractfile", json_only)
        capture.setattr(WebdatasetFileStore, "__getitem__", lambda *_: pytest.fail("pixel read"))
        sample = _native_sample(root, key)
        assert "jpg" not in sample and "jpgs" not in sample
        descriptors = tuple(
            {
                "kind": "raw_bytes",
                "key": key,
                "image_idx": index,
                "tar_path": str(root / "shard.tar"),
                "grid_thw": (1, 4, 4),
            }
            for index in range(count)
        )
        bound = api.bind_webdataset_image_descriptors(sample, descriptors, storage_roots=roots)
        locators = tuple(
            api.prepare_static_vision_locator(
                descriptor, storage_roots=roots, grid_thw=(1, 4, 4), declared_dimensions=None
            )
            for descriptor in bound
        )
    assert reads == [key + ".json"]
    assert all(locator.kind is VisionLocatorKind.WEBDATASET_ENTRY for locator in locators)
    assert tuple(api.vision_locator_image_bytes(locator) for locator in locators) == expected


@pytest.mark.parametrize("corruption", ["key", "tar", "source", "outside", "files", "bool_index"])
def test_native_descriptor_identity_is_bound_to_authoritative_source(indexed_dataset, corruption):
    root, roots = indexed_dataset
    sample = _native_sample(root, "one")
    descriptor = {"kind": "raw_bytes", "key": "one", "image_idx": 0}
    if corruption == "key":
        descriptor["key"] = "two"
    elif corruption == "tar":
        descriptor["tar_path"] = str(root / "other.tar")
    elif corruption == "source":
        sample["__sources__"] = (replace(sample["__sources__"][0], shard_name="other.tar"),)
    elif corruption == "files":
        sample["__sources__"] = (replace(sample["__sources__"][0], file_names=()),)
    elif corruption == "bool_index":
        descriptor["image_idx"] = False
    else:
        roots = api.validate_locator_storage_roots([str(root / ".nv-meta")])
    with pytest.raises(ValueError):
        api.bind_webdataset_image_descriptors(sample, (descriptor,), storage_roots=roots)


def test_native_index_rejects_ambiguous_jpg_and_jpgs(indexed_dataset, monkeypatch):
    from megatron.energon.cache.file_store import WebdatasetFileStore

    root, roots = indexed_dataset
    sample = _native_sample(root, "one")
    monkeypatch.setattr(
        WebdatasetFileStore,
        "list_sample_parts",
        lambda *_a, **_k: iter((("jpg", 10, 0), ("jpgs", 20, 0))),
    )
    with pytest.raises(ValueError, match="exactly one"):
        api.bind_webdataset_image_descriptors(sample, ({"kind": "raw_bytes"},), storage_roots=roots)


def test_static_carrier_cannot_expand_trusted_root_authority(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    lazy = _encoder(static_locator_roots=[str(tmp_path)])
    document = lazy.preencode_sample(
        _sample(grids=((1, 2, 2),), carriers=(str(tmp_path / "outside.jpg"),))
    )
    with pytest.raises(ValueError):
        api.validate_static_energon_batch(
            [document], storage_roots=api.validate_locator_storage_roots([str(root)])
        )


def test_task_encoder_preserves_text_and_applies_grid_policy_before_freeze(tmp_path, monkeypatch):
    roots = api.validate_locator_storage_roots([str(tmp_path)])
    eager = _encoder()
    lazy = _encoder(static_locator_roots=roots)
    sample = _sample(grids=((1, 2, 2),), carriers=(str(tmp_path / "image.jpg"),))
    # This is a policy ordering test, not validation of the real resize policy.
    monkeypatch.setattr(type(eager), "_descriptor_grid", lambda *_: (1, 4, 4))
    ordinary, captured = eager.preencode_sample(sample), lazy.preencode_sample(sample)
    for key in ("input_ids", "labels", "loss_mask", "image_grid_thw"):
        assert torch.equal(ordinary[key], captured[key])
    assert captured["vision_locators"][0].grid_thw == (1, 4, 4)
    assert captured["pixel_values"].numel() == 0
    text = lazy.preencode_sample(_sample())
    assert text["vision_locators"] == ()
    batch = lazy.batch([lazy.pack_selected_samples([captured, text])])
    assert api.validate_static_energon_batch(batch, storage_roots=roots) is batch


@pytest.mark.parametrize("corruption", ["marker", "pixels", "grid", "untyped"])
def test_static_document_rejects_unvalidated_or_materialized_carriers(tmp_path, corruption):
    lazy = _encoder(static_locator_roots=[str(tmp_path)])
    document = lazy.preencode_sample(
        _sample(grids=((1, 2, 2),), carriers=(str(tmp_path / "image.jpg"),))
    )
    if corruption == "marker":
        document.pop("vision_capture_mode")
    elif corruption == "pixels":
        document["pixel_values"] = torch.zeros(4, lazy.payload_width)
    elif corruption == "grid":
        document["image_grid_thw"] = torch.tensor([[1, 4, 4]])
    else:
        document["vision_locators"] = ({},)
    with pytest.raises(ValueError):
        api.validate_static_energon_batch([document], storage_roots=lazy.static_locator_roots)


def test_static_batch_bypasses_eager_decode_only_for_explicit_qwen_mode(tmp_path, monkeypatch):
    from examples.multimodal_dev.forward_step import _prepare_energon_batch

    lazy = _encoder(static_locator_roots=[str(tmp_path)])
    batch = [lazy.preencode_sample(_sample())]
    args = SimpleNamespace(
        dataset_provider="energon",
        model_arch="qwen35_vl",
        mdp_vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        energon_vision_storage_roots=lazy.static_locator_roots,
    )
    monkeypatch.setattr(api, "prepare_energon_batch", lambda *_a, **_k: pytest.fail("eager decode"))
    assert _prepare_energon_batch(batch, args) is batch


def test_model_static_real_fill_uses_existing_cpu_materializer(tmp_path, monkeypatch):
    from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter

    locator = api.prepare_static_vision_locator(
        {"kind": "image_path", "path": str(tmp_path / "image.jpg")},
        storage_roots=api.validate_locator_storage_roots([str(tmp_path)]),
        grid_thw=(1, 2, 2),
        declared_dimensions=None,
    )
    adapter = Qwen35VLMdpAdapter(16)
    pixels = torch.arange(4 * adapter.payload_width, dtype=torch.float32).reshape(4, -1)
    monkeypatch.setattr(
        adapter,
        "materialize_vision_locator",
        lambda value: b"encoded" if value == locator else None,
    )

    def prepare(locators, payloads):
        assert locators == (locator,) and payloads == (b"encoded",)
        return pixels

    monkeypatch.setattr(adapter, "prepare_materialized_vision_payloads", prepare)
    destination = torch.empty_like(pixels, dtype=torch.bfloat16)
    adapter.fill_vision_payload(locator, destination)
    assert torch.equal(destination, pixels.to(torch.bfloat16))


def test_real_image_eager_and_static_producer_patch_values_match_exactly(tmp_path):
    from PIL import Image
    from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
    from examples.multimodal_dev.models.qwen35_vl.energon import build_image_materializer

    encoded = io.BytesIO()
    Image.new("RGB", (256, 256), (41, 97, 173)).save(encoded, format="JPEG")
    path = tmp_path / "image.jpg"
    path.write_bytes(encoded.getvalue())
    grid = (1, 16, 16)
    descriptor = {
        "kind": "image_path",
        "path": str(path),
        "grid_thw": grid,
        "height": 256,
        "width": 256,
    }
    locator = api.prepare_static_vision_locator(
        descriptor,
        storage_roots=api.validate_locator_storage_roots([str(tmp_path)]),
        grid_thw=grid,
        declared_dimensions=(256, 256),
    )
    eager = build_image_materializer(args=None)((descriptor,), torch.tensor([grid]))
    actual = torch.empty_like(eager, dtype=torch.bfloat16)
    Qwen35VLMdpAdapter(16).fill_vision_payload(locator, actual)
    assert torch.equal(actual, eager.to(torch.bfloat16))


def test_core_requires_adapter_opt_in_for_real_locator_kind():
    from megatron.core.mdp.errors import MdpConfigurationError
    from megatron.core.mdp.static_vision import bind_static_vision_catalog
    from megatron.core.mdp.vision_locator import VisionDataLocator, build_vision_locator_catalog
    from tests.unit_tests.mdp.test_static_vision import _fixture

    _, view, plan, catalog = _fixture()
    entries = tuple(
        replace(
            entry,
            locator=VisionDataLocator(
                VisionLocatorKind.SHARED_FILE,
                f"/trusted/{index}.jpg",
                None,
                None,
                api.VisionLocatorIndexSentinel.UNUSED,
                entry.locator.grid_thw,
            ),
        )
        for index, entry in enumerate(catalog.entries)
    )
    real = build_vision_locator_catalog(tuple(entry.item_id for entry in entries), entries)
    with pytest.raises(MdpConfigurationError):
        bind_static_vision_catalog(real, plan, view.worker_ids)
    locators, digest = bind_static_vision_catalog(
        real, plan, view.worker_ids, allowed_kinds=(VisionLocatorKind.SHARED_FILE,)
    )
    assert len(locators) == len(entries) and len(digest) == 16


@pytest.mark.parametrize("kinds", [None, (), [VisionLocatorKind.SHARED_FILE], (1,)])
def test_core_rejects_malformed_static_kind_capability(kinds):
    from megatron.core.mdp.errors import MdpConfigurationError
    from megatron.core.mdp.static_vision import bind_static_vision_catalog
    from tests.unit_tests.mdp.test_static_vision import _fixture

    _, view, plan, catalog = _fixture()
    with pytest.raises(MdpConfigurationError):
        bind_static_vision_catalog(catalog, plan, view.worker_ids, allowed_kinds=kinds)


@pytest.mark.parametrize("static", [False, True])
def test_generic_provider_forwards_json_filter_only_for_root_bound_static_encoder(
    tmp_path, monkeypatch, static
):
    from examples.multimodal_dev.data.energon import provider
    from examples.multimodal_dev.tests.test_energon_provider_contract import _args

    calls = []
    roots = api.validate_locator_storage_roots([str(tmp_path)])
    encoder = SimpleNamespace(static_locator_roots=roots if static else None)
    args = _args(
        energon_vision_storage_roots=roots,
        mdp_enable=True,
        mdp_vision_capture_mode=(
            VisionCaptureMode.STABLE_LOCATOR_CATALOG
            if static
            else VisionCaptureMode.SOURCE_PIXEL_SIDECAR
        ),
    )

    def train(*_args, **kwargs):
        calls.append(kwargs)
        return ()

    def valid(*_args, **kwargs):
        calls.append(kwargs)
        return [((), None)]

    native = SimpleNamespace(get_train_dataset=train, get_val_datasets=valid)
    monkeypatch.setattr(provider, "load_energon7_api", lambda: native)
    monkeypatch.setattr(provider, "get_args", lambda: args)
    monkeypatch.setattr(provider, "_worker_config", lambda *_: object())
    monkeypatch.setattr(provider, "_build_task_encoder", lambda *_: encoder)
    monkeypatch.setattr(provider, "_build_loader", lambda _api, dataset, **_: iter(dataset))
    provider.train_valid_test_datasets_provider(None)
    assert len(calls) == 2
    for kwargs in calls:
        assert kwargs.get("part_filter") == (["json"] if static else None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_arch", "other"),
        ("energon_vision_storage_roots", None),
        ("tensor_model_parallel_size", 2),
        ("use_packed_sequence", False),
    ],
)
def test_static_real_admission_keeps_unsupported_launches_rejected(field, value):
    from megatron.core.mdp.config import MdpConfig
    from megatron.core.mdp.errors import MdpConfigurationError
    from megatron.core.mdp.integration import _resolve_vision_capture_mode

    args = SimpleNamespace(
        mdp_enable=True,
        dataset_provider="energon",
        model_arch="qwen35_vl",
        energon_vision_storage_roots=("/trusted",),
        tensor_model_parallel_size=1,
        use_packed_sequence=True,
        mdp_vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
    )
    assert (
        _resolve_vision_capture_mode(args, MdpConfig()) is VisionCaptureMode.STABLE_LOCATOR_CATALOG
    )
    setattr(args, field, value)
    with pytest.raises(MdpConfigurationError):
        _resolve_vision_capture_mode(args, MdpConfig())
