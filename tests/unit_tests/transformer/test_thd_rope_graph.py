# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compare graph-safe THD indexing with the pre-vectorization implementation."""

import pytest
import torch

from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_thd,
    _get_thd_freqs_on_this_cp_rank,
)


class CPGroup:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def oracle(t, cu, freqs, group, interleaved=False):
    boundaries = cu.cpu().tolist()
    lengths = [(b - a) // group.size() for a, b in zip(boundaries, boundaries[1:])]
    splits = torch.split(t, lengths)
    exact = freqs.shape[0] == boundaries[-1]
    packed = torch.cat(
        [
            _get_thd_freqs_on_this_cp_rank(
                group.rank(), group.size(), part, freqs, boundaries[i] if exact else 0
            )
            for i, part in enumerate(splits)
        ]
    )
    return _apply_rotary_pos_emb_bshd(
        t.unsqueeze(1), packed, rotary_interleaved=interleaved
    ).squeeze(1)


@pytest.mark.parametrize("cp", [1, 2, 4])
@pytest.mark.parametrize("full_frequencies", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_thd_mapping_and_gradients(cp, full_frequencies, interleaved, dtype):
    torch.manual_seed(1234)
    cu = torch.tensor([0, 16, 16, 48, 64, 64], device="cuda", dtype=torch.int32)
    for rank in range(cp):
        group = CPGroup(cp, rank)
        t = torch.randn(64 // cp, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        freqs = torch.randn(
            64 if full_frequencies else 32, 1, 1, 8, device="cuda", requires_grad=True
        )
        expected = oracle(t, cu, freqs, group, interleaved)
        actual = _apply_rotary_pos_emb_thd(
            t, cu, freqs, cp_group=group, rotary_interleaved=interleaved
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.randn_like(actual)
        expected_grads = torch.autograd.grad(expected, (t, freqs), grad, retain_graph=True)
        actual_grads = torch.autograd.grad(actual, (t, freqs), grad)
        for got, want in zip(actual_grads, expected_grads):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cp", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_graph_replays_different_layouts(cp, dtype):
    for rank in range(cp):
        group = CPGroup(cp, rank)
        t = torch.randn(64 // cp, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        freqs = torch.randn(64, 1, 1, 8, device="cuda")
        cu = torch.tensor([0, 16, 48, 64, 64], device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                out = _apply_rotary_pos_emb_thd(t, cu, freqs, cp_group=group)
                out.sum().backward()
                t.grad = None
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _apply_rotary_pos_emb_thd(t, cu, freqs, cp_group=group)
            out.sum().backward()
        for boundaries in ([0, 16, 48, 64, 64], [0, 8, 32, 48, 64]):
            cu.copy_(torch.tensor(boundaries, dtype=torch.int32, device="cuda"))
            t.grad.zero_()
            graph.replay()
            expected = oracle(t, cu, freqs, group)
            expected_grad = torch.autograd.grad(expected.sum(), t)[0]
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            torch.testing.assert_close(t.grad, expected_grad, rtol=1e-5, atol=1e-6)
