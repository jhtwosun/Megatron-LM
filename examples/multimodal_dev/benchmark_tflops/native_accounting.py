# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Evaluate trusted, pinned native FLOPs code without importing the training stack.

This is not a sandbox for untrusted Python. All source files must be reviewed
and pinned. Missing attended-boundary evidence stays unavailable, not estimated.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


def read_pinned(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Source hash mismatch: {path}")
    return raw.decode("utf-8")


def extract_function(text, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one function named {name}")
    return nodes[0]


def load_native(training_path, training_sha256, utils_path, utils_sha256,
                variants_path, variants_sha256):
    training = read_pinned(training_path, training_sha256)
    utils = read_pinned(utils_path, utils_sha256)
    function = extract_function(training, "num_floating_point_operations")
    helper = extract_function(utils, "is_hybrid_model")
    variants = read_pinned(variants_path, variants_sha256)
    linear_helper = extract_function(variants, "is_linear_attention_variant")
    scope = {"Optional": Optional}
    exec(compile(ast.Module(body=[helper, linear_helper, function], type_ignores=[]), str(training_path), "exec"), scope)
    fingerprint = hashlib.sha256(ast.get_source_segment(training, function).encode()).hexdigest()
    return scope[function.name], fingerprint


def segment_geometry(logical_cu, physical_cu):
    """Only equal final logical/storage boundaries are unambiguous here.

    Nonstatic paths can use different valid lengths and storage offsets. Those
    require a separate backend audit; never square storage gaps as attention.
    Repeated endpoints contribute zero. A nonzero dummy segment is included.
    """
    for cu in (logical_cu, physical_cu):
        if len(cu) < 2 or cu[0] != 0 or any(type(v) is not int or v < 0 for v in cu):
            raise ValueError("Invalid cumulative boundaries")
        if any(b < a for a, b in zip(cu, cu[1:])):
            raise ValueError("Nonmonotonic cumulative boundaries")
    if logical_cu != physical_cu:
        return {"Tpad": None, "Upad": None, "status": "requires_backend_valid_length_audit"}
    lengths = [b - a for a, b in zip(logical_cu, logical_cu[1:])]
    return {"Tpad": sum(lengths), "Upad": sum(v * v for v in lengths),
            "status": "equal_final_boundaries", "lengths": lengths}


def validate_world(resolved_args, world_size):
    keys = ("tensor_model_parallel_size", "pipeline_model_parallel_size",
            "context_parallel_size", "data_parallel_size")
    values = [resolved_args.get(k) for k in ("world_size", *keys)]
    if type(world_size) is not int or world_size <= 0 or any(
            type(v) is not int or v <= 0 for v in values):
        raise ValueError("Complete positive integer world/parallelism arguments required")
    if values[0] != world_size or math.prod(values[1:]) != world_size:
        raise ValueError("World/parallelism mismatch")


def aggregate(native, resolved_args, rows, world_size, start, end):
    """Rows contain original step_ms and independently proven global Tpad/Upad."""
    if type(world_size) is not int or world_size <= 0 or not 1 <= start <= end:
        raise ValueError("Invalid world or window")
    validate_world(resolved_args, world_size)
    # This first adapter supports the audited standard-attention path only.
    required = ("hybrid_layer_pattern", "hybrid_override_pattern", "experimental_attention_variant",
                "linear_attention_freq", "multi_latent_attention", "padded_vocab_size")
    if any(k not in resolved_args for k in required):
        raise ValueError("Missing resolved branch or vocabulary arguments")
    if any(resolved_args[k] is not None for k in required[:4]) or resolved_args["multi_latent_attention"]:
        raise ValueError("Only the audited standard-attention branch is supported")
    if type(resolved_args["padded_vocab_size"]) is not int or resolved_args["padded_vocab_size"] <= 0:
        raise ValueError("Resolved padded vocabulary is required")
    if [r["iteration"] for r in rows] != list(range(1, len(rows) + 1)) or end > len(rows):
        raise ValueError("Original timing rows must be complete and ordered")
    selected = rows[start - 1:end]
    for r in rows:
        if not math.isfinite(r["step_ms"]) or r["step_ms"] <= 0:
            raise ValueError("Invalid original step time")
    if any(r.get(k) is None for r in selected for k in ("Tpad", "Upad")):
        return {"status": "unavailable_boundary_evidence", "window": [start, end],
                "mean_tflops_gpu": None, "pooled_tflops_gpu": None}
    evaluated = []
    for r in selected:
        if any(type(r[k]) is not int or r[k] < 0 for k in ("Tpad", "Upad")):
            raise ValueError("Global boundary moments must be nonnegative integers")
        t, u = r["Tpad"], r["Upad"]
        if (t == 0 and u != 0) or (t > 0 and not t <= u <= t * t):
            raise ValueError("Impossible global boundary moments")
        # Native code can mutate args, e.g. the non-GQA query-group field.
        flops = native(SimpleNamespace(**copy.deepcopy(resolved_args)), r["Tpad"], r["Upad"])
        if not math.isfinite(flops) or flops <= 0:
            raise ValueError("Invalid native FLOPs")
        evaluated.append({**r, "native_decoder_flops": flops,
                          "native_decoder_tflops_gpu": flops / (world_size * r["step_ms"] * 1e9)})
    return {"status": "accounting_only_requires_external_provenance_gate", "window": [start, end],
            "samples": len(evaluated), "world_size": world_size,
            "mean_tflops_gpu": statistics.mean(r["native_decoder_tflops_gpu"] for r in evaluated),
            "pooled_tflops_gpu": sum(r["native_decoder_flops"] for r in evaluated)
            / (world_size * sum(r["step_ms"] for r in evaluated) * 1e9),
            "median_step_ms": statistics.median(r["step_ms"] for r in evaluated), "rows": evaluated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--utils-source", type=Path, required=True)
    parser.add_argument("--utils-sha256", required=True)
    parser.add_argument("--variants-source", type=Path, required=True)
    parser.add_argument("--variants-sha256", required=True)
    parser.add_argument("--resolved-args", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    native, fingerprint = load_native(args.training_source, args.training_sha256,
                                      args.utils_source, args.utils_sha256,
                                      args.variants_source, args.variants_sha256)
    result = aggregate(native, json.loads(args.resolved_args.read_text()),
                       json.loads(args.rows.read_text()), args.world_size, args.start, args.end)
    result["provenance"] = {"training_sha256": args.training_sha256, "utils_sha256": args.utils_sha256,
                            "variants_sha256": args.variants_sha256,
                            "function_sha256": fingerprint,
                            "args_sha256": hashlib.sha256(args.resolved_args.read_bytes()).hexdigest(),
                            "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest()}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
