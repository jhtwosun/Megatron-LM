"""CPU-only collector tests; no Torch import or GPU work."""
import unittest

from collect_workload import collect
from workload_flops import training_flops


def log_rows(total=20, gbs=64):
    fields = {
        "elapsed time per iteration (ms)": 1000,
        "throughput per GPU (TFLOP/s/GPU)": 123,
        "lm loss": 2,
        "grad norm": 1,
        "global batch size": gbs,
        "number of skipped iterations": 0,
        "number of nan iterations": 0,
        "ref_geom_T": 1024,
        "ref_geom_U": 2 * 512**2,
        "ref_geom_R": 256,
        "ref_geom_A": 256**2,
    }
    values = " | ".join(f"{key}: {value}" for key, value in fields.items())
    return "\n".join(f"iteration {i}/{total} | {values} |" for i in range(1, total + 1))


class CollectorTest(unittest.TestCase):
    def test_defaults_and_window(self):
        result = collect(log_rows())
        self.assertEqual(result["samples"], 17)
        self.assertEqual(result["window"], [4, 20])
        self.assertEqual(collect(log_rows(), start=10)["samples"], 11)
        numerator = training_flops(64 * 1024, 64 * 2 * 512**2,
                                   64 * 256, 64 * 256**2)
        self.assertEqual(result["mean"]["decoder_modeled_tflops_gpu"],
                         numerator["decoder_training_flops"] / 16e12)

    def test_explicit_world_and_batch(self):
        a = collect(log_rows(), world_size=8)
        b = collect(log_rows(), world_size=16)
        self.assertEqual(a["mean"]["total_modeled_tflops_gpu"],
                         2 * b["mean"]["total_modeled_tflops_gpu"])
        c = collect(log_rows(gbs=128), gbs=128)
        self.assertEqual(c["mean"]["total_modeled_tflops_gpu"],
                         2 * b["mean"]["total_modeled_tflops_gpu"])

    def test_invalid_logs(self):
        log = log_rows()
        bad = [
            log.rsplit("\n", 1)[0],
            log + "\n" + log.splitlines()[0],
            log.replace("grad norm: 1", "grad norm: nan"),
            log.replace("global batch size: 64", "global batch size: 32"),
            log.replace("number of skipped iterations: 0", "number of skipped iterations: 1"),
            log.replace("elapsed time per iteration (ms): 1000", "elapsed time per iteration (ms): 0"),
            log.replace("ref_geom_R: 256", "ref_geom_R: -1"),
        ]
        for invalid in bad:
            with self.subTest(invalid=invalid[:80]), self.assertRaises(ValueError):
                collect(invalid)
        with self.assertRaises(ValueError):
            collect(log, start=21)


if __name__ == "__main__":
    unittest.main()
