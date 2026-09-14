"""Collect the explicit reference window; not a substitute for sacct/verifier gates."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from workload_flops import tflops_per_gpu, training_flops


def collect(log, world_size=16, gbs=64, start=4, end=20, total=20):
    if world_size <= 0 or gbs <= 0 or not 1 <= start <= end <= total:
        raise ValueError("invalid topology or measurement window")
    rows = {}
    for line in log.splitlines():
        match = re.search(r"iteration\s+(\d+)\s*/\s*(\d+)\s*\|", line)
        if not match:
            continue
        iteration, logged_total = map(int, match.groups())
        if logged_total != total or iteration in rows:
            raise ValueError("expected unique iteration records for one complete run")
        if not 1 <= iteration <= total:
            raise ValueError("unexpected iteration")

        def field(label):
            value = re.search(re.escape(label) + r":\s*([^|]+)", line)
            if value is None:
                raise ValueError(f"missing {label} at iteration {iteration}")
            number = float(value[1].strip())
            if not math.isfinite(number):
                raise ValueError(f"non-finite {label}")
            return number

        ms = field("elapsed time per iteration (ms)")
        legacy = field("throughput per GPU (TFLOP/s/GPU)")
        loss = field("lm loss")
        grad_norm = field("grad norm")
        if field("global batch size") != gbs:
            raise ValueError("GBS changed")
        for label in ("number of skipped iterations", "number of nan iterations"):
            if field(label) != 0:
                raise ValueError(f"invalid measurement: {label}")
        # Counts are native means per logical packed bin, not per loss token.
        geometry = [field("ref_geom_" + key) * gbs for key in ("T", "U", "R", "A")]
        counts = training_flops(*geometry)
        rows[iteration] = {
            "iteration": iteration, "step_ms": ms, "legacy_tflops_gpu": legacy,
            "lm_loss": loss, "grad_norm": grad_norm,
            "global_geometry_T_U_R_A": geometry,
            **{key.replace("training_flops", "modeled_tflops_gpu"):
               tflops_per_gpu(counts[key], world_size, ms)
               for key in ("decoder_training_flops", "vision_training_flops", "total_training_flops")},
        }
    if sorted(rows) != list(range(1, total + 1)):
        raise ValueError("full run completion is required")
    measured = [rows[i] for i in range(start, end + 1)]
    metrics = ("step_ms", "legacy_tflops_gpu", "decoder_modeled_tflops_gpu",
               "vision_modeled_tflops_gpu", "total_modeled_tflops_gpu")
    return {
        "status": "log_aggregation_only_requires_job_and_artifact_verification",
        "world_size": world_size, "gbs": gbs, "window": [start, end],
        "samples": len(measured), "expected_total": total,
        "mean": {key: statistics.mean(row[key] for row in measured) for key in metrics},
        "median": {key: statistics.median(row[key] for row in measured) for key in metrics},
        "rows": measured,
        "definition": "Useful content matmul model, FMA2, training3xforward; whole VLM step denominator",
        "excluded": ["padding", "router", "norm/RoPE/softmax", "optimizer FLOPs", "kernel internal recomputation"],
        "geometry_precision": "12E logged per-bin means multiplied by GBS; rounded, not an exact integer archive",
        "model": "PR7 Qwen3-VL only; constants in workload_flops.py",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--gbs", type=int, required=True)
    parser.add_argument("--start", type=int, default=4)
    parser.add_argument("--end", type=int, default=20)
    parser.add_argument("--total", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(collect(args.log.read_text(), args.world_size, args.gbs,
                             args.start, args.end, args.total), indent=2))
