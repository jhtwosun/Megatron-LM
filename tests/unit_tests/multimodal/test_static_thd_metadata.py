"""Allocated static-capacity metadata tests; not an attention graph correctness verdict."""

from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.data.blend_dataset import Qwen35VLDataset
from examples.multimodal_dev.data.energon_mdp import _doc_index_for_image
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer


@pytest.mark.parametrize("lengths", [[1088], [512, 1088, 2048], [8192, 8192], [512] * 31])
def test_static_capacity_and_real_image_ownership(lengths):
    ds = Qwen35VLDataset.__new__(Qwen35VLDataset)
    ds.thd_static_packing, ds.thd_max_packed_sequences = True, 32
    ds.seq_length, ds.cp_size = 16384, 2
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    image_cu = [0] + [i // 2 for i in range(1, len(cu))]
    out = {"cu_seqlens": torch.tensor(cu, dtype=torch.int32),
           "image_cu_seqlens": torch.tensor(image_cu, dtype=torch.int32),
           "pixel_cu_seqlens": torch.tensor(image_cu, dtype=torch.int32),
           "loss_mask": torch.zeros(16384)}
    out["loss_mask"][:cu[-1]] = 1
    out = ds._static_thd_metadata(out)
    assert out["cu_seqlens"].shape == (33,)
    assert out["cu_seqlens"].tolist()[:len(cu)] == cu
    assert out["cu_seqlens"][-1] == 16384
    assert torch.equal(out["cu_seqlens"], out["cu_seqlens_padded"])
    for image in range(image_cu[-1]):
        assert _doc_index_for_image(image_cu, image) == _doc_index_for_image(
            out["image_cu_seqlens"].tolist(), image)
    assert out["loss_mask"][cu[-1]:].sum() == 0


def test_reject_static_overflow():
    ds = Qwen35VLDataset.__new__(Qwen35VLDataset)
    ds.thd_static_packing, ds.thd_max_packed_sequences = True, 32
    ds.seq_length, ds.cp_size = 16384, 2
    with pytest.raises(ValueError, match="capacity"):
        ds._static_thd_metadata({"cu_seqlens": torch.arange(33, dtype=torch.int32) * 512})


def test_four_tensor_reconstruction_variable_layout():
    owner = SimpleNamespace(config=SimpleNamespace(thd_static_packing=True,
        thd_max_packed_sequences=32, max_seqlen_per_dp_cp_rank=8192, context_parallel_size=2))
    for prefix in ([0, 512, 16384], [0, 1088, 2176, 16384]):
        cu = torch.tensor(prefix + [16384] * (33 - len(prefix)), dtype=torch.int32)
        fields = {key: cu.clone() for key in (
            "cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")}
        packed = PackedSeqParams(qkv_format="thd", **fields)
        kwargs = {"packed_seq_params": packed, "attention_mask": None}
        TransformerLayer._decompose_static_thd(owner, kwargs)
        assert set(kwargs) == set(fields)
        TransformerLayer._reconstruct_static_thd(owner, kwargs)
        rebuilt = kwargs["packed_seq_params"]
        assert kwargs["attention_mask"] is None
        assert rebuilt.max_seqlen_q == rebuilt.max_seqlen_kv == 16384
        for key in fields:
            assert getattr(rebuilt, key) is fields[key]
        with pytest.raises(ValueError, match="explicit mask"):
            TransformerLayer._decompose_static_thd(owner, {
                "packed_seq_params": packed, "attention_mask": torch.ones(1, dtype=torch.bool)})
    owner.config.thd_static_packing = False
    with pytest.raises(ValueError, match="opt-in"):
        TransformerLayer._decompose_static_thd(owner, {"packed_seq_params": packed})
