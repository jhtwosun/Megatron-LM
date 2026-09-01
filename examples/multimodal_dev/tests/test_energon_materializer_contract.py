# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Descriptor-I/O and owner-only Energon materialization contracts."""

import io
import os
import pickle
import zipfile
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.multimodal_dev.forward_step import build_vision_sidecar
from examples.multimodal_dev.models.qwen35_vl.configuration import QWEN35_VL_IMAGE_TOKEN_ID

_GENERIC = "examples.multimodal_dev.data.energon.materializer"
_QWEN = "examples.multimodal_dev.models.qwen35_vl.energon"
_QWEN_FACTORY = f"{_QWEN}.build_image_materializer"
_QWEN_VALIDATOR = f"{_QWEN}.validate_image_metadata"
_PIXEL_WIDTH = 1536
_TP2 = int(os.environ.get("WORLD_SIZE", "1")) == 2


def _jpeg_bytes(color=(41, 97, 173), size=(256, 256)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG", quality=100, subsampling=0)
    return output.getvalue()


def _document(*, descriptors=(), grids=(), pixels=None):
    if pixels is None:
        pixels = torch.empty(0, _PIXEL_WIDTH, dtype=torch.float32)
    return {
        "input_ids": torch.tensor([11, QWEN35_VL_IMAGE_TOKEN_ID, 12], dtype=torch.long),
        "labels": torch.tensor([-100, -100, 12], dtype=torch.long),
        "loss_mask": torch.tensor([0.0, 0.0, 1.0]),
        "pixel_values": pixels,
        "image_grid_thw": torch.tensor(grids, dtype=torch.long).reshape(-1, 3),
        "image_descriptors": tuple(descriptors),
    }


def _args(**overrides):
    values = {"dataset_provider": "energon", "model_arch": "qwen35_vl"}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registry_keeps_qwen_materializer_lazy_and_model_owned():
    from examples.multimodal_dev.models import MODEL_REGISTRY

    assert MODEL_REGISTRY["qwen35_vl"]["energon_image_materializer_factory"] == _QWEN_FACTORY
    assert MODEL_REGISTRY["qwen35_vl"]["energon_image_metadata_validator"] == _QWEN_VALIDATOR


def test_non_owner_performs_zero_factory_or_descriptor_io(monkeypatch):
    generic = __import__(_GENERIC, fromlist=["prepare_energon_batch"])
    from examples.multimodal_dev.models import MODEL_REGISTRY

    calls = []

    def forbidden_factory(*, args):
        calls.append(args)
        raise AssertionError("non-owner resolved a materializer")

    monkeypatch.setitem(MODEL_REGISTRY["qwen35_vl"], "energon_image_materializer_factory", forbidden_factory)
    descriptor = {"kind": "image_path", "path": "/must/not/be/opened.jpg", "grid_thw": (1, 2, 2)}
    document = _document(descriptors=(descriptor,), grids=((1, 2, 2),))

    prepared = generic.prepare_energon_batch([document], args=_args(), materialize_pixels=False)

    assert prepared[0] is document
    assert prepared[0]["image_descriptors"][0] is descriptor
    assert prepared[0]["pixel_values"].shape == (0, _PIXEL_WIDTH)
    assert calls == []


def test_mock_batch_is_byte_for_byte_untouched(monkeypatch):
    generic = __import__(_GENERIC, fromlist=["prepare_energon_batch"])
    document = _document()
    assert (
        generic.prepare_energon_batch([document], args=_args(dataset_provider="mock"), materialize_pixels=True)[0]
        is document
    )


def test_metadata_only_sidecar_preserves_image_positions_without_pixels():
    document = _document(descriptors=({"grid_thw": (1, 2, 2)},), grids=((1, 2, 2),))
    sidecar = build_vision_sidecar(
        [document], [0, 3], image_token_id=QWEN35_VL_IMAGE_TOKEN_ID, spatial_merge_size=2, expect_pixels=False
    )

    assert sidecar["vision_item_meta"].tolist() == [[0, 0, 1, 2, 2, 0]]
    assert sidecar["vision_decoder_positions"].tolist() == [1]
    with pytest.raises(ValueError, match="suppressed pixel capture"):
        build_vision_sidecar(
            [{**document, "pixel_values": torch.ones(4, _PIXEL_WIDTH)}],
            [0, 3],
            image_token_id=QWEN35_VL_IMAGE_TOKEN_ID,
            spatial_merge_size=2,
            expect_pixels=False,
        )


def test_bytes_path_jpgs_and_zip_preserve_exact_encoded_identity(tmp_path):
    generic = __import__(_GENERIC, fromlist=["descriptor_image_bytes"])
    first = _jpeg_bytes(color=(1, 2, 3))
    second = _jpeg_bytes(color=(4, 5, 6))

    direct = {"kind": "image_bytes", "encoded_image": first}
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(second)
    bundle_path = tmp_path / "images.jpgs"
    bundle_path.write_bytes(pickle.dumps([first, second], protocol=4))
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nested/image.jpg", first)

    assert generic.descriptor_image_bytes(direct) == first
    assert generic.descriptor_image_bytes({"kind": "raw_bytes", "bytes": second}) == second
    assert generic.descriptor_image_bytes({"kind": "image_path", "path": image_path}) == second
    assert (
        generic.descriptor_image_bytes(
            {
                "kind": "raw_bytes",
                "encoded_images": pickle.dumps([first, second], protocol=4),
                "encoded_image_index": 0,
            }
        )
        == first
    )
    assert generic.descriptor_image_bytes({"kind": "jpgs", "path": bundle_path, "encoded_image_index": 1}) == second
    assert (
        generic.descriptor_image_bytes(
            {"kind": "zip_image", "zip_path": zip_path, "candidates": ("missing.jpg", "nested/image.jpg")}
        )
        == first
    )


def test_jpgs_rejects_pickle_globals():
    generic = __import__(_GENERIC, fromlist=["descriptor_image_bytes"])
    descriptor = {
        "kind": "jpgs",
        "encoded_images": pickle.dumps(ValueError("unsafe"), protocol=4),
        "encoded_image_index": 0,
    }
    with pytest.raises(ValueError, match="global objects are not allowed"):
        generic.descriptor_image_bytes(descriptor)


def test_parquet_descriptor_preserves_exact_encoded_identity(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    generic = __import__(_GENERIC, fromlist=["descriptor_image_bytes"])
    first = _jpeg_bytes(color=(7, 8, 9))
    second = _jpeg_bytes(color=(10, 11, 12))
    path = tmp_path / "images.parquet"
    pq.write_table(pa.table({"image": [first, second]}), path, row_group_size=1)

    assert (
        generic.descriptor_image_bytes(
            {"kind": "parquet_column_image", "parquet_path": path, "column": "image", "row_idx": 1}
        )
        == second
    )


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        ({"kind": "unknown", "encoded_image": b"x"}, "unsupported"),
        ({"kind": "image_bytes"}, "ambiguous or missing"),
        ({"kind": "image_bytes", "encoded_image": b"x", "path": "also.jpg"}, "ambiguous or missing"),
        ({"kind": "zip_image", "zip_path": "images.zip", "candidate": "../escape.jpg"}, "safe"),
        (
            {"kind": "zip_image", "zip_path": "images.zip", "candidates": ("valid.jpg", 7)},
            "member paths must be strings",
        ),
        (
            {"kind": "parquet_column_image", "parquet_path": "images.parquet", "column": "image", "row_idx": -1},
            "non-negative",
        ),
    ],
)
def test_malformed_descriptors_fail_before_any_io(monkeypatch, descriptor, message):
    generic = __import__(_GENERIC, fromlist=["descriptor_image_bytes"])
    io_calls = []
    monkeypatch.setattr(generic, "_read_bytes", lambda *args: io_calls.append(args))
    monkeypatch.setattr(generic.zipfile, "ZipFile", lambda *args: io_calls.append(args))

    with pytest.raises(ValueError, match=message):
        generic.descriptor_image_bytes(descriptor)
    assert io_calls == []


def test_later_malformed_descriptor_prevents_all_prior_io(monkeypatch):
    generic = __import__(_GENERIC, fromlist=["prepare_energon_batch"])
    from examples.multimodal_dev.models import MODEL_REGISTRY

    calls = []

    def forbidden_factory(*, args):
        calls.append(("factory", args))
        raise AssertionError("materializer resolved before every descriptor was validated")

    monkeypatch.setitem(MODEL_REGISTRY["qwen35_vl"], "energon_image_materializer_factory", forbidden_factory)
    monkeypatch.setattr(generic, "_read_bytes", lambda *args: calls.append(("io", args)))
    valid = {"kind": "image_path", "path": "/must/not/be-opened.jpg"}
    malformed = {"kind": "unknown", "encoded_image": b"invalid"}
    document = _document(descriptors=(valid, malformed), grids=((1, 2, 2), (1, 2, 2)))

    with pytest.raises(ValueError, match="unsupported image descriptor kind"):
        generic.prepare_energon_batch([document], args=_args(), materialize_pixels=True)
    assert calls == []


def test_later_grid_mismatch_prevents_factory_io_and_decode(monkeypatch):
    generic = __import__(_GENERIC, fromlist=["prepare_energon_batch"])
    from examples.multimodal_dev.models import MODEL_REGISTRY

    calls = []

    def forbidden_factory(*, args):
        calls.append(("factory", args))
        raise AssertionError("factory resolved before every descriptor grid was validated")

    monkeypatch.setitem(MODEL_REGISTRY["qwen35_vl"], "energon_image_materializer_factory", forbidden_factory)
    monkeypatch.setattr(generic, "_read_bytes", lambda *args: calls.append(("io", args)))
    monkeypatch.setattr(generic, "load_descriptor_image", lambda *args: calls.append(("decode", args)))
    first = {"kind": "image_path", "path": "/must/not/open-first.jpg", "grid_thw": (1, 2, 2)}
    later_mismatch = {"kind": "image_path", "path": "/must/not/open-second.jpg", "grid_thw": (1, 4, 2)}
    document = _document(descriptors=(first, later_mismatch), grids=((1, 2, 2), (1, 2, 2)))

    with pytest.raises(ValueError, match="descriptor 1 grid_thw does not match"):
        generic.prepare_energon_batch([document], args=_args(), materialize_pixels=True)
    assert calls == []


def test_qwen_materializer_prevalidates_later_grid_before_processor_or_decode(monkeypatch):
    generic = __import__(_GENERIC, fromlist=["load_descriptor_image"])
    qwen = __import__(_QWEN, fromlist=["_materialize_images"])
    calls = []
    descriptors = (
        {"kind": "image_path", "path": "/must/not/open-first.jpg", "grid_thw": (1, 2, 2)},
        {"kind": "image_path", "path": "/must/not/open-second.jpg", "grid_thw": (1, 4, 2)},
    )
    grids = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
    monkeypatch.setattr(qwen, "_image_processor", lambda: calls.append("processor"))
    monkeypatch.setattr(generic, "load_descriptor_image", lambda *args: calls.append(("decode", args)))

    with pytest.raises(ValueError, match="descriptor 1 grid_thw does not match"):
        qwen._materialize_images(descriptors, grids)
    assert calls == []


def test_owner_qwen_materialization_decodes_once_and_stays_on_cpu(monkeypatch):
    generic = __import__(_GENERIC, fromlist=["prepare_energon_batch"])
    qwen = __import__(_QWEN, fromlist=["build_image_materializer"])
    encoded = _jpeg_bytes()
    grid = qwen.derive_image_grid_thw(width=256, height=256)
    descriptor = {"kind": "image_bytes", "encoded_image": encoded, "grid_thw": grid, "width": 256, "height": 256}
    document = _document(descriptors=(descriptor,), grids=(grid,))
    decode_calls = []
    original = generic.load_descriptor_image

    def counted(value):
        decode_calls.append(value)
        return original(value)

    monkeypatch.setattr(generic, "load_descriptor_image", counted)
    prepared = generic.prepare_energon_batch([document], args=_args(), materialize_pixels=True)
    pixels = prepared[0]["pixel_values"]

    assert decode_calls == [descriptor]
    assert pixels.device.type == "cpu"
    assert pixels.dtype == torch.float32
    assert pixels.shape == (grid[0] * grid[1] * grid[2], _PIXEL_WIDTH)
    assert torch.isfinite(pixels).all()


@pytest.fixture
def _tp2_group():
    if not _TP2:
        pytest.skip("requires torchrun world2")
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    yield
    Utils.destroy_model_parallel()


def test_tp2_owner_materialization_failure_converges_before_pack(_tp2_group, monkeypatch):
    from examples.multimodal_dev import forward_step

    rank = torch.distributed.get_rank()
    primary = RuntimeError("owner materialization failed")
    iterator = iter(([{"sample": 1}],)) if rank == 0 else iter(())
    pack_calls = []

    def prepare(data, args):
        assert rank == 0
        raise primary

    monkeypatch.setattr(
        forward_step,
        "get_args",
        lambda: SimpleNamespace(dataset_provider="energon", use_packed_sequence=True, seq_length=32, mdp_enable=False),
    )
    monkeypatch.setattr(forward_step, "_prepare_energon_batch", prepare)
    monkeypatch.setattr(forward_step, "pack_or_pad_batch", lambda *args, **kwargs: pack_calls.append((args, kwargs)))

    caught = None
    try:
        forward_step.get_batch(iterator)
    except BaseException as exc:
        caught = exc
    observation = {
        "type": type(caught).__name__,
        "message": str(caught),
        "primary": caught is primary,
        "pack_calls": len(pack_calls),
    }
    gathered = [None, None]
    torch.distributed.all_gather_object(gathered, observation)

    assert gathered[0] == {
        "type": "RuntimeError",
        "message": "owner materialization failed",
        "primary": True,
        "pack_calls": 0,
    }
    assert gathered[1]["type"] == "RuntimeError"
    assert "TP source" in gathered[1]["message"]
    assert gathered[1]["primary"] is False
    assert gathered[1]["pack_calls"] == 0


@pytest.mark.parametrize("use_packed_sequence", [True, False], ids=["thd", "bshd"])
def test_tp2_owner_h2d_failure_converges_after_metadata_broadcast(_tp2_group, monkeypatch, use_packed_sequence):
    from examples.multimodal_dev import forward_step

    rank = torch.distributed.get_rank()
    primary = RuntimeError("owner pixel H2D failed")
    args = SimpleNamespace(
        dataset_provider="energon",
        model_arch="qwen35_vl",
        use_packed_sequence=use_packed_sequence,
        seq_length=8,
        mdp_enable=False,
        sequence_parallel=False,
    )
    document = _document(
        descriptors=({"kind": "image_path", "path": "/synthetic/image.jpg"},),
        grids=((1, 2, 2),),
        pixels=torch.ones(4, 2),
    )
    iterator = iter(([document],)) if rank == 0 else iter(())
    broadcast_keys = []

    class _AssembledPixels:
        @staticmethod
        def is_pinned():
            return False

        @staticmethod
        def to(*args, **kwargs):
            raise primary

    original_broadcast = forward_step.broadcast_data_batch

    def observed_broadcast(data, device="cuda"):
        broadcast_keys.append(tuple(sorted(data)))
        return original_broadcast(data, device=device)

    original_move = forward_step._move_tp_owner_pixels

    def failing_move(pixel_chunks, *, device):
        original_concat = forward_step.torch.concat
        if rank == 0:
            forward_step.torch.concat = lambda _chunks: _AssembledPixels()
        try:
            return original_move(pixel_chunks, device=device)
        finally:
            forward_step.torch.concat = original_concat

    monkeypatch.setattr(forward_step, "get_args", lambda: args)
    monkeypatch.setattr(forward_step, "_prepare_energon_batch", lambda data, args: data)
    monkeypatch.setattr(forward_step, "broadcast_data_batch", observed_broadcast)
    monkeypatch.setattr(forward_step, "_move_tp_owner_pixels", failing_move)
    caught = None
    try:
        forward_step.get_batch(iterator)
    except BaseException as exc:
        caught = exc
    observation = {
        "type": type(caught).__name__,
        "message": str(caught),
        "primary": caught is primary,
        "broadcast_keys": broadcast_keys,
    }
    gathered = [None, None]
    torch.distributed.all_gather_object(gathered, observation)

    assert gathered[0]["type"] == "RuntimeError"
    assert gathered[0]["message"] == "owner pixel H2D failed"
    assert gathered[0]["primary"] is True
    assert gathered[1]["type"] == "RuntimeError"
    assert "pixel assembly or H2D failed" in gathered[1]["message"]
    assert gathered[1]["primary"] is False
    assert all(len(item["broadcast_keys"]) == 1 for item in gathered)
    assert all("pixel_values" not in item["broadcast_keys"][0] for item in gathered)


@pytest.mark.parametrize("use_packed_sequence", [True, False], ids=["thd", "bshd"])
def test_tp2_success_keeps_pixels_and_h2d_on_source_only(_tp2_group, monkeypatch, use_packed_sequence):
    from examples.multimodal_dev import forward_step
    from examples.multimodal_dev.models import MODEL_REGISTRY

    generic = __import__(_GENERIC, fromlist=["descriptor_image_bytes"])
    rank = torch.distributed.get_rank()
    args = SimpleNamespace(
        dataset_provider="energon",
        model_arch="qwen35_vl",
        use_packed_sequence=use_packed_sequence,
        seq_length=8,
        mdp_enable=False,
        sequence_parallel=False,
    )
    source_pixels = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    descriptor = {"kind": "image_path", "path": "/synthetic/image.jpg"}
    document = _document(descriptors=(descriptor,), grids=((1, 2, 2),))
    iterator = iter(([document],)) if rank == 0 else iter(())
    factory_calls = []
    materialize_calls = []
    descriptor_io = []
    broadcast_keys = []
    owner_move_rows = []

    def read_bytes(value, owner):
        descriptor_io.append((value, owner))
        return b"encoded-image"

    def factory(*, args):
        factory_calls.append(args)

        def materialize(descriptors, grids):
            materialize_calls.append((descriptors, grids.clone()))
            assert [generic.descriptor_image_bytes(value) for value in descriptors] == [b"encoded-image"]
            return source_pixels.clone()

        return materialize

    original_broadcast = forward_step.broadcast_data_batch

    def observed_broadcast(data, device="cuda"):
        broadcast_keys.append(tuple(sorted(data)))
        return original_broadcast(data, device=device)

    original_move = forward_step._move_tp_owner_pixels

    def observed_move(pixels, *, device):
        owner_move_rows.append(0 if pixels is None else sum(int(value.shape[0]) for value in pixels))
        return original_move(pixels, device=device)

    monkeypatch.setattr(forward_step, "get_args", lambda: args)
    monkeypatch.setattr(generic, "_read_bytes", read_bytes)
    monkeypatch.setitem(MODEL_REGISTRY["qwen35_vl"], "energon_image_materializer_factory", factory)
    monkeypatch.setattr(forward_step, "broadcast_data_batch", observed_broadcast)
    monkeypatch.setattr(forward_step, "_move_tp_owner_pixels", observed_move)

    batch = forward_step.get_batch(iterator)
    observation = {
        "factory_calls": len(factory_calls),
        "materialize_calls": len(materialize_calls),
        "descriptor_io": len(descriptor_io),
        "broadcast_keys": broadcast_keys,
        "owner_move_rows": owner_move_rows,
        "pixel_present": "pixel_values" in batch,
        "pixel_device": (batch["pixel_values"].device.type if "pixel_values" in batch else None),
        "pixel_values": (batch["pixel_values"].cpu().tolist() if "pixel_values" in batch else None),
        "nonpixel_tensors": {
            key: {"shape": tuple(value.shape), "dtype": str(value.dtype), "values": value.cpu().tolist()}
            for key, value in batch.items()
            if torch.is_tensor(value) and key != "pixel_values"
        },
    }
    gathered = [None, None]
    torch.distributed.all_gather_object(gathered, observation)

    assert gathered[0]["factory_calls"] == 1
    assert gathered[0]["materialize_calls"] == 1
    assert gathered[0]["descriptor_io"] == 1
    assert gathered[1]["factory_calls"] == 0
    assert gathered[1]["materialize_calls"] == 0
    assert gathered[1]["descriptor_io"] == 0
    assert all("pixel_values" not in item["broadcast_keys"][0] for item in gathered)
    assert gathered[0]["owner_move_rows"] == [4]
    assert gathered[1]["owner_move_rows"] == [0]
    assert gathered[0]["pixel_present"] is True
    assert gathered[0]["pixel_device"] == "cuda"
    assert gathered[0]["pixel_values"] == source_pixels.tolist()
    assert gathered[1]["pixel_present"] is False
    assert gathered[1]["pixel_device"] is None
    assert gathered[1]["pixel_values"] is None
    assert gathered[0]["nonpixel_tensors"] == gathered[1]["nonpixel_tensors"]


@pytest.mark.parametrize("use_packed_sequence", [True, False], ids=["thd", "bshd"])
def test_tp2_text_only_energon_skips_pixel_concat_and_h2d(_tp2_group, monkeypatch, use_packed_sequence):
    from examples.multimodal_dev import forward_step
    from examples.multimodal_dev.models import MODEL_REGISTRY

    rank = torch.distributed.get_rank()
    args = SimpleNamespace(
        dataset_provider="energon",
        model_arch="qwen35_vl",
        use_packed_sequence=use_packed_sequence,
        seq_length=8,
        mdp_enable=False,
        sequence_parallel=False,
    )
    document = _document()
    document["input_ids"] = torch.tensor([11, 12, 13], dtype=torch.long)
    iterator = iter(([document],)) if rank == 0 else iter(())
    factory_calls = []
    broadcast_keys = []
    owner_move_inputs = []

    def forbidden_factory(*, args):
        factory_calls.append(args)
        raise AssertionError("text-only Energon batch resolved an image materializer")

    original_broadcast = forward_step.broadcast_data_batch

    def observed_broadcast(data, device="cuda"):
        broadcast_keys.append(tuple(sorted(data)))
        return original_broadcast(data, device=device)

    original_move = forward_step._move_tp_owner_pixels

    def observed_move(pixel_chunks, *, device):
        owner_move_inputs.append(pixel_chunks is None)
        return original_move(pixel_chunks, device=device)

    monkeypatch.setattr(forward_step, "get_args", lambda: args)
    monkeypatch.setitem(MODEL_REGISTRY["qwen35_vl"], "energon_image_materializer_factory", forbidden_factory)
    monkeypatch.setattr(forward_step, "broadcast_data_batch", observed_broadcast)
    monkeypatch.setattr(forward_step, "_move_tp_owner_pixels", observed_move)

    batch = forward_step.get_batch(iterator)
    observation = {
        "factory_calls": len(factory_calls),
        "broadcast_keys": broadcast_keys,
        "owner_move_inputs": owner_move_inputs,
        "pixel_present": "pixel_values" in batch,
        "nonpixel_tensors": {
            key: value.cpu().tolist()
            for key, value in batch.items()
            if torch.is_tensor(value) and key != "pixel_values"
        },
    }
    gathered = [None, None]
    torch.distributed.all_gather_object(gathered, observation)

    assert all(item["factory_calls"] == 0 for item in gathered)
    assert all(item["owner_move_inputs"] == [True] for item in gathered)
    assert all(item["pixel_present"] is False for item in gathered)
    assert all("pixel_values" not in item["broadcast_keys"][0] for item in gathered)
    assert gathered[0]["nonpixel_tensors"] == gathered[1]["nonpixel_tensors"]
