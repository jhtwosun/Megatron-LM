"""Attention-only compatibility guards; no global packed metadata mutation."""
import pytest
import torch
from types import SimpleNamespace
from megatron.core.extensions import transformer_engine as extension
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.extensions.transformer_engine import _causal_cp_attention_kwargs


def fixture():
    real = torch.tensor([0, 11, 24], dtype=torch.int32)
    padded = torch.tensor([0, 16, 32], dtype=torch.int32)
    packed = dict(qkv_format="thd", pad_between_seqs=True,
                  cu_seqlens_q=real, cu_seqlens_kv=real,
                  cu_seqlens_q_padded=padded, cu_seqlens_kv_padded=padded)
    guards = dict(cp_size=4, attention_type="self", attn_mask_type=AttnMaskType.causal,
                  attention_dropout=0.0, window_size=None, attention_bias=None)
    return packed, guards


def test_attention_copy_preserves_true_metadata():
    packed, guards = fixture()
    result = _causal_cp_attention_kwargs(packed, **guards)
    assert result is not packed
    assert result["pad_between_seqs"] is False
    assert result["cu_seqlens_q"] is packed["cu_seqlens_q_padded"]
    assert packed["pad_between_seqs"] is True
    assert packed["cu_seqlens_q"].tolist() == [0, 11, 24]


def test_native_inferred_padding_params_remain_unchanged():
    packed, guards = fixture()
    packed["pad_between_seqs"] = None
    params = PackedSeqParams(**packed)
    kwargs = {name: getattr(params, name) for name in packed}
    result = _causal_cp_attention_kwargs(kwargs, **guards)
    assert result["pad_between_seqs"] is False
    assert params.pad_between_seqs is None
    assert params.cu_seqlens_q.tolist() == [0, 11, 24]


def test_no_internal_padding_retains_native_inference():
    packed, guards = fixture()
    packed["pad_between_seqs"] = None
    packed["cu_seqlens_q_padded"] = packed["cu_seqlens_kv_padded"] = torch.tensor([0,11,32])
    assert _causal_cp_attention_kwargs(packed, **guards) is packed


def test_effective_maxima_cover_padded_documents():
    packed, guards = fixture()
    packed.update(max_seqlen_q=13, max_seqlen_kv=13)
    result = _causal_cp_attention_kwargs(packed, **guards)
    assert result["max_seqlen_q"] == result["max_seqlen_kv"] == 16
    assert packed["max_seqlen_q"] == 13


@pytest.mark.parametrize("boundaries", [[1, 12, 25], [0, 15, 14], [[0,11,24]]])
def test_malformed_real_boundaries_unchanged(boundaries):
    packed, guards = fixture()
    packed["cu_seqlens_q"] = packed["cu_seqlens_kv"] = torch.tensor(boundaries)
    assert _causal_cp_attention_kwargs(packed, **guards) is packed


@pytest.mark.parametrize("name,value", [
    ("cp_size", 1), ("attention_type", "cross"),
    ("attn_mask_type", AttnMaskType.no_mask), ("attention_dropout", 0.1),
    ("window_size", (16, 0)), ("attention_bias", object()),
])
def test_excluded_attention_guards(name, value):
    packed, guards = fixture()
    guards[name] = value
    assert _causal_cp_attention_kwargs(packed, **guards) is packed


@pytest.mark.parametrize("name,value", [
    ("qkv_format", "sbhd"), ("pad_between_seqs", False),
    ("cu_seqlens_q_padded", None),
    ("cu_seqlens_kv", torch.tensor([0, 12, 24], dtype=torch.int32)),
    ("cu_seqlens_kv_padded", torch.tensor([0, 8, 32], dtype=torch.int32)),
])
def test_excluded_metadata(name, value):
    packed, guards = fixture()
    packed[name] = value
    assert _causal_cp_attention_kwargs(packed, **guards) is packed


def test_padding_must_not_shorten_any_document():
    packed, guards = fixture()
    packed["cu_seqlens_q_padded"] = packed["cu_seqlens_kv_padded"] = torch.tensor([0,8,32])
    assert _causal_cp_attention_kwargs(packed, **guards) is packed


def test_cuda_capture_retains_existing_path(monkeypatch):
    packed, guards = fixture()
    packed = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in packed.items()}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _causal_cp_attention_kwargs(packed, **guards) is packed


@pytest.mark.parametrize("effective_dropout", [0.0, 0.1])
@pytest.mark.parametrize("excluded", [None, "qk_clip", "log_max_attention_logit",
    "fp8", "fp4", "quant_recipe", "autocast", "fp16"])
def test_wrapper_inferred_padding_and_effective_dropout(monkeypatch, effective_dropout, excluded):
    packed, _ = fixture()
    packed["pad_between_seqs"] = None
    params = PackedSeqParams(**packed)
    attention = extension.TEDotProductAttention.__new__(extension.TEDotProductAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(context_parallel_size=4, window_size=None,
        attention_dropout=0.0, qk_clip=False, log_max_attention_logit=False)
    if excluded in ("qk_clip", "log_max_attention_logit", "fp8", "fp4", "quant_recipe"):
        setattr(attention.config,excluded,True)
    monkeypatch.setattr(extension.FP8GlobalStateManager,"is_fp8_enabled",
                        lambda: excluded == "autocast")
    attention._mcore_attention_type = "self"
    attention._mcore_attention_dropout = effective_dropout
    attention.cp_group = attention.cp_global_ranks = attention.cp_stream = None
    attention.num_splits = None
    attention.kept_packed_seq_params = set(packed)
    attention.qkv_format = "thd"
    attention.te_forward_mask_type = True
    attention.current_max_attn_logits = None
    monkeypatch.setattr(extension.te.pytorch.DotProductAttention, "forward",
                        lambda self, *args, **kwargs: (kwargs,torch.tensor(0.0))
                        if excluded in ("qk_clip","log_max_attention_logit") else kwargs)
    tensor = torch.zeros(8,1,8,dtype=torch.float16 if excluded == "fp16" else torch.bfloat16)
    result = attention.forward(tensor, tensor, tensor, None, AttnMaskType.causal,
                               packed_seq_params=params)
    assert result["pad_between_seqs"] is (False if effective_dropout == 0 and excluded is None else None)
    assert params.pad_between_seqs is None
    assert params.cu_seqlens_q.tolist() == [0,11,24]
