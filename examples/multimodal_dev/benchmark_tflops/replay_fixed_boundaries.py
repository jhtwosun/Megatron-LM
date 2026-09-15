# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU-only native fixed-reference boundary replay; run inside an allocation.

No model is constructed. This emits normalization-boundary evidence, not a claim
of historical token identity or an executed Transformer Engine attention call.
Use a reviewed JSON manifest containing full resolved_args, dataset_kwargs,
source_seal, reference_config, world_size, dp_size, gbs, total_steps,
sampler_total_samples and consumed_samples. No model defaults are fabricated.
"""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from native_accounting import extract_function, segment_geometry, validate_world


def check_seal(root, seal, expected_sha256):
    raw = seal.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Unexpected source seal digest")
    lines = raw.decode().splitlines()
    if not lines:
        raise ValueError("Empty source seal")
    seen = set()
    for line in lines:
        digest, relative = line.split(maxsplit=1)
        path = (root / relative.lstrip("*")).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Seal path escapes source root")
        if path in seen:
            raise ValueError("Duplicate source seal path")
        seen.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source seal mismatch: {relative}")


def validate_manifest(manifest):
    args, kwargs = manifest["resolved_args"], manifest["dataset_kwargs"]
    validate_world(args, manifest["world_size"])
    keys = ("dp_size", "gbs", "total_steps", "sampler_total_samples", "consumed_samples")
    if any(type(manifest[k]) is not int or manifest[k] < (0 if k == "consumed_samples" else 1)
           for k in keys):
        raise ValueError("Invalid integer sampler configuration")
    dp, gbs, steps, total, consumed = (manifest[k] for k in keys)
    expected = {"data_parallel_size": dp, "global_batch_size": gbs,
                "train_iters": steps, "consumed_train_samples": consumed,
                "micro_batch_size": 1, "num_workers": 0, "dataloader_type": "single",
                "skipped_train_samples": 0, "rampup_batch_size": None}
    if any(k not in args or args[k] != value for k, value in expected.items()):
        raise ValueError("Manifest differs from original resolved sampler arguments")
    if consumed != 0:
        raise ValueError("Resumed training requires a separate replay audit")
    if gbs % dp or total != gbs * steps:
        raise ValueError("Sampler horizon or batch alignment mismatch")
    if kwargs.get("max_samples") != total or kwargs.get("cp_size") != args["context_parallel_size"]:
        raise ValueError("Dataset/sampler bounds mismatch")
    if type(args.get("seq_length")) is not int or args["seq_length"] <= 0 or kwargs.get("seq_length") != args["seq_length"]:
        raise ValueError("Dataset sequence budget differs from resolved arguments")
    if args.get("train_samples") is not None:
        raise ValueError("Sample-based training requires a separate replay audit")
    return dp, gbs, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    cli = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Torch replay requires an approved allocation")
    manifest = json.loads(cli.manifest.read_text())
    source = cli.source.resolve()
    seal = Path(manifest["source_seal"])
    check_seal(source, seal, manifest["source_seal_sha256"])
    dp, gbs, steps = validate_manifest(manifest)
    config = manifest["reference_config"]
    if config.get("mode") != "fixed_grid":
        raise ValueError("This first replay supports fixed-reference grids only")
    kwargs = dict(manifest["dataset_kwargs"])
    if not kwargs["thd_static_packing"]:
        raise ValueError("Require the reviewed static native path")
    if kwargs["backend"] != "mock" or not kwargs["dataloader_sequence_packing"]:
        raise ValueError("Require the native fixed mock packed dataset")
    args = manifest["resolved_args"]
    if args["num_workers"] != 0 or args["micro_batch_size"] != 1:
        raise ValueError("Only the original workers0/MBS1 path is supported")
    os.environ["PR7_REFERENCE_MOCK_CONFIG"] = json.dumps(config)
    sys.path.insert(0, str(source))
    import torch

    import megatron.training.global_vars as global_vars
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.training.datasets.data_samplers import MegatronPretrainingSampler

    global_vars._GLOBAL_ARGS = SimpleNamespace(**args)
    from examples.multimodal_dev.data.blend_dataset import Qwen35VLDataset

    # Fixed-reference records carry native precomputed IDs; no text tokenizer
    # should be invoked. Fail instead of substituting a lexical tokenizer.
    class ReferenceOnlyTokenizer:
        def tokenize(self, text):
            raise RuntimeError("Unexpected text tokenization in fixed-reference replay")

    forward_text = (source / "examples/multimodal_dev/forward_step.py").read_text()
    normalize_node = extract_function(forward_text, "_prepare_prepacked_batch")
    scope = {"torch": torch, "PackedSeqParams": PackedSeqParams, "Dict": dict, "Any": object}
    exec(compile(ast.Module(body=[normalize_node], type_ignores=[]), "native_forward_step", "exec"), scope)
    normalize = scope["_prepare_prepacked_batch"]
    records = []
    totals = [{"iteration": i + 1, "Tpad": 0, "Upad": 0, "content_geometry": [0] * 4}
              for i in range(steps)]
    for rank in range(dp):
        dataset = Qwen35VLDataset(**{**kwargs, "dataloader_dp_rank": rank,
                                    "tokenizer": ReferenceOnlyTokenizer()})
        sampler = iter(MegatronPretrainingSampler(
            total_samples=manifest["sampler_total_samples"],
            consumed_samples=manifest["consumed_samples"], micro_batch_size=1,
            data_parallel_rank=rank, data_parallel_size=dp))
        for iteration in range(steps):
            for microbatch in range(gbs // dp):
                index, = next(sampler)
                expected_index = manifest["consumed_samples"] + rank + (iteration * (gbs // dp) + microbatch) * dp
                if index != expected_index:
                    raise ValueError("Native sampler order differs from reviewed consumption")
                batch = dataset[index]
                hashes = {}
                for key in ("input_ids", "labels", "loss_mask", "position_ids", "image_grid_thw",
                            "cu_seqlens", "cu_seqlens_padded", "_mdp_image_descriptors_json"):
                    value = batch.get(key)
                    if value is None:
                        hashes[key] = None
                    else:
                        payload = value.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(value) else json.dumps(value, sort_keys=True).encode()
                        hashes[key] = hashlib.sha256(payload).hexdigest()
                content = batch["_reference_flops_geometry"].tolist()
                params = normalize(batch)["packed_seq_params"]
                logical = params.cu_seqlens_q.tolist()
                physical = params.cu_seqlens_q_padded.tolist()
                geometry = segment_geometry(logical, physical)
                if geometry["Tpad"] is None:
                    raise ValueError("Static final logical/storage boundaries unexpectedly differ")
                records.append({"dp_rank": rank, "iteration": iteration + 1, "microbatch": microbatch,
                                "sample_index": index, "logical_cu": logical, "physical_cu": physical,
                                "hashes": hashes, "geometry": geometry})
                row = totals[iteration]
                row["Tpad"] += geometry["Tpad"]
                row["Upad"] += geometry["Upad"]
                row["content_geometry"] = [a + b for a, b in zip(row["content_geometry"], content)]
    if len(records) != steps * gbs:
        raise ValueError("Incomplete sampler consumption")
    check_seal(source, seal, manifest["source_seal_sha256"])
    print(json.dumps({"status": "candidate_native_normalization_replay_requires_log_and_backend_review",
                      "manifest_sha256": hashlib.sha256(cli.manifest.read_bytes()).hexdigest(),
                      "source_seal_sha256": hashlib.sha256(seal.read_bytes()).hexdigest(),
                      "normalizer_sha256": hashlib.sha256(ast.get_source_segment(forward_text, normalize_node).encode()).hexdigest(),
                      "historical_input_identity": None, "executed_TE_boundary_proof": None,
                      "steps": totals, "records": records}, allow_nan=False))


if __name__ == "__main__":
    main()
