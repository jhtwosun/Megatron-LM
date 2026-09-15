# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Standard-library tests; no Torch or training-stack imports."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from native_accounting import aggregate, load_native, segment_geometry
from replay_fixed_boundaries import check_seal, validate_manifest


class AccountingTest(unittest.TestCase):
    def setUp(self):
        self.args = dict(hybrid_layer_pattern=None, hybrid_override_pattern=None,
                         experimental_attention_variant=None, linear_attention_freq=None,
                         multi_latent_attention=False, padded_vocab_size=248448,
                         world_size=16, tensor_model_parallel_size=1,
                         pipeline_model_parallel_size=2, context_parallel_size=2,
                         data_parallel_size=4)

    def test_world_and_impossible_moments(self):
        row = dict(iteration=1, step_ms=1, Tpad=10, Upad=100)
        for args in ({**self.args, "world_size": 64},
                     {**self.args, "data_parallel_size": 16},
                     {k: v for k, v in self.args.items() if k != "world_size"}):
            with self.assertRaises(ValueError):
                aggregate(lambda *a: 1, args, [row], 16, 1, 1)
        for t, u in ((10, 1), (10, 101), (0, 1), (10, 0)):
            with self.assertRaises(ValueError):
                aggregate(lambda *a: 1, self.args, [{**row, "Tpad": t, "Upad": u}], 16, 1, 1)

    def test_seal_identity_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("content")
            seal = root / "seal"
            line = hashlib.sha256(b"content").hexdigest() + "  source.py\n"
            for contents in ("", line + line, line.replace("source.py", "../outside.py")):
                seal.write_text(contents)
                with self.assertRaises(ValueError):
                    check_seal(root, seal, hashlib.sha256(contents.encode()).hexdigest())
            seal.write_text(line)
            check_seal(root, seal, hashlib.sha256(line.encode()).hexdigest())
            with self.assertRaises(ValueError):
                check_seal(root, seal, "0" * 64)

    def test_manifest_sampler_binding(self):
        manifest = dict(world_size=16, dp_size=4, gbs=64, total_steps=20,
                        sampler_total_samples=1280, consumed_samples=0,
                        dataset_kwargs=dict(max_samples=1280, cp_size=2, seq_length=16384),
                        resolved_args={**self.args, "global_batch_size": 64, "train_iters": 20,
                                       "consumed_train_samples": 0, "micro_batch_size": 1,
                                       "num_workers": 0, "dataloader_type": "single",
                                       "skipped_train_samples": 0, "rampup_batch_size": None,
                                       "train_samples": None, "seq_length": 16384})
        self.assertEqual(validate_manifest(manifest), (4, 64, 20))
        for section, key, value in ((None, "gbs", 32), (None, "total_steps", 19),
                                    (None, "consumed_samples", 1),
                                    (None, "sampler_total_samples", 1284),
                                    ("dataset_kwargs", "max_samples", 2560),
                                    ("dataset_kwargs", "seq_length", 8192),
                                    ("resolved_args", "num_workers", 2),
                                    ("resolved_args", "dataloader_type", "cyclic")):
            bad = copy.deepcopy(manifest)
            (bad if section is None else bad[section])[key] = value
            with self.assertRaises(ValueError):
                validate_manifest(bad)
        resumed = copy.deepcopy(manifest)
        resumed["consumed_samples"] = resumed["resolved_args"]["consumed_train_samples"] = 64
        resumed["sampler_total_samples"] = resumed["dataset_kwargs"]["max_samples"] = 1344
        with self.assertRaises(ValueError):
            validate_manifest(resumed)

    def test_dummy_zero_and_ambiguous_storage(self):
        self.assertEqual(segment_geometry([0, 4, 6, 6], [0, 4, 6, 6])["Upad"], 20)
        self.assertEqual(segment_geometry([0, 4, 4], [0, 4, 4])["Tpad"], 4)
        self.assertIsNone(segment_geometry([0, 4], [0, 8])["Upad"])
        for cu in ([1, 2], [0, 2, 1], [0, 1.5], [0]):
            with self.assertRaises(ValueError):
                segment_geometry(cu, cu)

    def test_mean_and_pooled_are_different(self):
        rows = [dict(iteration=1, step_ms=1000, Tpad=1, Upad=1),
                dict(iteration=2, step_ms=2000, Tpad=1, Upad=1)]
        result = aggregate(lambda a, t, u: 16e12, self.args, rows, 16, 1, 2)
        self.assertEqual(result["mean_tflops_gpu"], 0.75)
        self.assertAlmostEqual(result["pooled_tflops_gpu"], 2 / 3)
        self.assertEqual(result["median_step_ms"], 1500)
        rows[0]["Upad"] = None
        self.assertIsNone(aggregate(lambda *a: 0, self.args, rows, 16, 1, 2)["mean_tflops_gpu"])

    def test_fail_closed(self):
        row = dict(iteration=1, step_ms=1000, Tpad=1, Upad=1)
        for rows in ([{**row, "step_ms": float("nan")}], [row, row],
                     [{**row, "Tpad": -1}], [{**row, "iteration": 2}]):
            with self.assertRaises(ValueError):
                aggregate(lambda *a: 1, self.args, rows, 16, 1, 1)
        with self.assertRaises(ValueError):
            aggregate(lambda *a: 1, {}, [row], 16, 1, 1)

    def test_hash_checked_ast_not_module_import(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            text = "raise RuntimeError('module must not execute')\ndef num_floating_point_operations(args, t, u):\n    return t + u\n"
            helper = "def is_hybrid_model(args):\n    return False\n"
            variants = "def is_linear_attention_variant(value):\n    return False\n"
            (p / "training.py").write_text(text)
            (p / "utils.py").write_text(helper)
            (p / "variants.py").write_text(variants)
            sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
            fn, _ = load_native(p / "training.py", sha(text), p / "utils.py", sha(helper),
                                p / "variants.py", sha(variants))
            self.assertEqual(fn(None, 2, 3), 5)
            with self.assertRaises(ValueError):
                load_native(p / "training.py", "0" * 64, p / "utils.py", sha(helper),
                            p / "variants.py", sha(variants))


if __name__ == "__main__":
    unittest.main()
