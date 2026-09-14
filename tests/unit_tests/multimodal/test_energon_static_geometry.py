from bisect import bisect_right

import pytest
import torch

from examples.multimodal_dev.data.qwen35_energon.task_encoder import Qwen35EnergonTaskEncoder


def encoder(**kwargs):
    return Qwen35EnergonTaskEncoder(
        tokenizer=object(), seq_length=256, cp_size=2,
        mdp_loader_prepartition=True, mdp_loader_prepartition_materialize=False,
        **kwargs,
    )


def docs(enc, images):
    ids = torch.tensor([11, 12, 13, 14, 15])
    text = dict(input_ids=ids, _assistant_loss_mask=torch.tensor([0., 0., 1., 1., 1.]),
                content_len=5, real_len=5, image_grid_thw=torch.zeros(0, 3, dtype=torch.long),
                num_images=0, num_patches=0, _mdp_image_descriptors=[])
    if not images:
        return [text]
    ids = torch.cat([enc._image_block(8), torch.tensor([21, 22, 23])])
    image = dict(input_ids=ids, _assistant_loss_mask=torch.ones(len(ids)),
                 content_len=len(ids), real_len=len(ids),
                 image_grid_thw=torch.tensor([[2, 4, 4]]), num_images=1,
                 num_patches=32, _mdp_image_descriptors=[{"grid_thw": [2, 4, 4]}])
    return [text, image]


@pytest.mark.parametrize("images", [False, True])
def test_real_finalizer_preserves_content_and_owner_metadata(images):
    old, new = encoder(), encoder(thd_static_packing=True, report_workload_geometry=True)
    samples = docs(old, images)
    before, after = old.pack_selected_samples(samples), new.pack_selected_samples(samples)
    for key in ("input_ids", "tokens", "labels", "loss_mask", "position_ids",
                "pixel_values", "image_grid_thw"):
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    assert after["_mdp_image_descriptors_json"] == before["_mdp_image_descriptors_json"]
    assert after["_mdp_image_descriptors"] == before["_mdp_image_descriptors"]
    assert "_reference_flops_geometry" not in before
    lengths = [d["content_len"] for d in samples]
    assert after["_reference_flops_geometry"].tolist() == [
        sum(lengths), sum(n*n for n in lengths), 32 if images else 0, 512 if images else 0]
    cu = after["cu_seqlens"].tolist()
    assert cu[:len(samples)+1] == before["cu_seqlens"].tolist()
    assert len(cu) == 33 and cu[-1] == 256
    assert all((b-a) % 4 == 0 for a, b in zip(cu, cu[1:]))
    assert not after["loss_mask"][before["cu_seqlens"][-1]:].any()
    for key in ("image_cu_seqlens", "pixel_cu_seqlens"):
        assert len(after[key]) == 33
        for index in range(int(before[key][-1])):
            assert bisect_right(before[key].tolist(), index) == bisect_right(after[key].tolist(), index)
    assert new.encode_batch(new.batch([after])) is after


def test_static_capacity_rejected_without_changing_default():
    old = encoder()
    sample = docs(old, False)[0]
    assert old.pack_selected_samples([sample, sample])["cu_seqlens"].numel() == 3
    new = encoder(thd_static_packing=True, thd_max_packed_sequences=2)
    with pytest.raises(ValueError, match="reserved capacity"):
        new.pack_selected_samples([sample, sample])


def test_static_tail_rejects_supervision():
    new = encoder(thd_static_packing=True)
    out = encoder().pack_selected_samples(docs(new, False))
    out["loss_mask"][-1] = 1
    with pytest.raises(ValueError, match="zero loss"):
        new._static_thd_metadata(out)


@pytest.mark.parametrize("count,budget", [(64, 16384), (128, 16384), (128, 8192)])
def test_real_buffer_capacity_preserves_more_than_31_documents(count, budget):
    old = Qwen35EnergonTaskEncoder(tokenizer=object(), seq_length=budget, cp_size=2)
    new = Qwen35EnergonTaskEncoder(
        tokenizer=object(), seq_length=budget, cp_size=2,
        thd_static_packing=True, thd_max_packed_sequences=129,
        report_workload_geometry=True,
    )
    samples = [docs(old, False)[0] for _ in range(count)]
    before = old.pack_selected_samples(samples)
    after = new.pack_selected_samples(samples)
    for key in ("input_ids", "labels", "loss_mask", "position_ids"):
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    assert after["cu_seqlens"].numel() == 130
    assert after["cu_seqlens"][:count+1].tolist() == before["cu_seqlens"].tolist()
    assert after["_reference_flops_geometry"].tolist() == [count*5, count*25, 0, 0]
    assert int(after["loss_mask"].sum()) == int(before["loss_mask"].sum()) > 0
