"""PR7 useful matmul FLOPs from global-step content geometry, not a hardware counter.

FMA=2; training=3*forward; causal attention uses half-square convention.
The denominator is whole multimodal optimizer-step wall time, including loading
and communication. Decoder and vision counts remain separately reportable.
"""


def training_flops(tokens, squared_lengths, patch_rows, squared_frame_patches):
    """T=sum L, U=sum L², R=sum t*h*w, A=sum t*(h*w)² before any padding."""
    if min(tokens, squared_lengths, patch_rows, squared_frame_patches) < 0:
        raise ValueError("geometry counts cannot be negative")
    layers, hidden, query, kv, topk, ffn, vocab = 48, 2048, 4096, 512, 8, 768, 248448
    decoder_forward = {
        "qkv": 2 * layers * tokens * hidden * (query + 2 * kv),
        "causal_attention": 2 * layers * squared_lengths * query,
        "attention_projection": 2 * layers * tokens * query * hidden,
        "experts": 6 * layers * tokens * topk * hidden * ffn,
        "logits": 2 * tokens * hidden * vocab,
    }
    vision_hidden, vision_layers, vision_ffn = 1152, 27, 4304
    merged_hidden = 4 * vision_hidden
    vision_forward = {
        "patch_projection": 2 * patch_rows * 1536 * vision_hidden,
        "attention_projections": vision_layers * 8 * patch_rows * vision_hidden**2,
        "bidirectional_attention": vision_layers * 4 * squared_frame_patches * vision_hidden,
        "mlp": vision_layers * 4 * patch_rows * vision_hidden * vision_ffn,
        "merger": 2 * (patch_rows / 4) * (merged_hidden**2 + merged_hidden * hidden),
    }
    decoder = 3 * sum(decoder_forward.values())
    vision = 3 * sum(vision_forward.values())
    return {
        "decoder_training_flops": decoder,
        "vision_training_flops": vision,
        "total_training_flops": decoder + vision,
        "decoder_forward_components": decoder_forward,
        "vision_forward_components": vision_forward,
    }


def tflops_per_gpu(flops, world_size, step_ms):
    if world_size <= 0 or step_ms <= 0:
        raise ValueError("world size and optimizer step duration must be positive")
    return flops / (world_size * step_ms * 1e9)


if __name__ == "__main__":
    # Reconcile the prior independent fixed-shape PR7 decoder numerator exactly.
    fixed = training_flops(256 * 8192, 256 * 8192**2, 0, 0)
    assert fixed["decoder_training_flops"] == 60_867_864_202_051_584
    assert fixed["vision_training_flops"] == 0
    short = training_flops(512, 512**2, 256, 256**2)
    double = training_flops(1024, 2 * 512**2, 512, 2 * 256**2)
    for key in ("decoder_training_flops", "vision_training_flops", "total_training_flops"):
        assert double[key] == 2 * short[key]
    assert tflops_per_gpu(16e12, 16, 1000) == 1
    print("workload numerator reconciliation, additivity and units PASS")
