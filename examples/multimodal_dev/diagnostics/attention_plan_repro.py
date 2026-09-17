"""Single-GPU TE packed self-attention cold/reused-shape diagnostic.

No Megatron, distributed initialization, model weights or dataset required.
Run in a fresh process. See attention_plan_repro.md for scope and interpretation.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import threading
import time
from itertools import accumulate


def shape_cases():
    """A/B keep total tokens and max length fixed, changing sequence count only."""
    return {
        "warmup": [256],
        "A": [4096] + [2048] * 8,
        "B": [4096] + [2048] * 7 + [1024, 1024],
        "C": [4096] * 9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    # Imports are deferred so --help and shape tests do not initialize Torch.
    import torch
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend,
        fused_attn_fwd,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; no CPU fallback")
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    print(json.dumps({
        "kind": "environment", "pid": os.getpid(), "tid": threading.get_native_id(),
        "python": platform.python_version(), "torch": torch.__version__,
        "te": importlib.metadata.version("transformer_engine"),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "backend": "explicit NVTE_F16_arbitrary_seqlen (no fallback)",
        "dtype": "bfloat16", "heads": 16, "head_dim": 128,
        "training": True, "dropout": 0.0, "mask": "padding (noncausal)",
        "layout": "thd_thd_thd", "vision_or_decoder": "vision attention geometry",
    }), flush=True)

    # All random generation, H2D and input allocation precede measurements.
    cases = {}
    for name, lengths in shape_cases().items():
        qkv = [torch.randn(sum(lengths), 16, 128, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
        cu = torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)
        cases[name] = (lengths, qkv, cu)
    torch.cuda.synchronize()
    reference = {}
    visits = {}
    schedule = ["warmup"] * 2 + ["A"] * args.repeats + ["B"] * args.repeats
    schedule += ["A"] + ["C"] * args.repeats + ["A", "B", "C"]
    for index, name in enumerate(schedule):
        lengths, (q, k, v), cu = cases[name]
        visit = visits.get(name, 0)
        label = f"packed_attention/{name}/visit={visit}"
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(label)
        begin = time.perf_counter_ns()
        output, auxiliary = fused_attn_fwd(
            is_training=True, max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths),
            cu_seqlens_q=cu, cu_seqlens_kv=cu, q=q, k=k, v=v,
            fake_dtype=torch.bfloat16,
            fused_attention_backend=FusedAttnBackend["F16_arbitrary_seqlen"],
            qkv_layout="thd_thd_thd", o_format="thd",
            cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu,
            attn_mask_type="padding", dropout=0.0,
        )
        returned = time.perf_counter_ns()
        torch.cuda.synchronize()
        complete = time.perf_counter_ns()
        torch.cuda.nvtx.range_pop()
        # Sanity checks are deliberately outside the measured range.
        if not torch.isfinite(output).all().item():
            raise RuntimeError(f"Non-finite output: {label}")
        if name in reference:
            torch.testing.assert_close(output, reference[name], rtol=0, atol=0)
        else:
            reference[name] = output.clone()
        visits[name] = visit + 1
        print(json.dumps({
            "kind": "call", "index": index, "case": name, "visit": visit,
            "lengths": lengths, "sequences": len(lengths), "tokens": sum(lengths),
            "max_seqlen": max(lengths),
            "host_call_ms": (returned - begin) / 1e6,
            "completed_wall_ms": (complete - begin) / 1e6,
            "finite": True, "repeat_exact": visit > 0,
        }), flush=True)
        del output, auxiliary
    print(json.dumps({"kind": "complete", "calls": len(schedule)}), flush=True)


if __name__ == "__main__":
    main()
