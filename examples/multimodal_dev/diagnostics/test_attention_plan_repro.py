"""Shape-contract tests; standard library only, safe without CUDA/Torch."""

import runpy
import unittest
from pathlib import Path


class ShapeTests(unittest.TestCase):
    def setUp(self):
        module = runpy.run_path(str(Path(__file__).with_name("attention_plan_repro.py")))
        self.cases = module["shape_cases"]()

    def test_sequence_count_axis(self):
        a, b = self.cases["A"], self.cases["B"]
        self.assertEqual((sum(a), max(a)), (sum(b), max(b)))
        self.assertEqual((len(a), len(b)), (9, 10))

    def test_token_bucket_axis(self):
        a, c = self.cases["A"], self.cases["C"]
        self.assertEqual((len(a), max(a)), (len(c), max(c)))
        self.assertLess(sum(a), 32768)
        self.assertGreater(sum(c), 32768)

    def test_lengths(self):
        for lengths in self.cases.values():
            self.assertTrue(all(isinstance(n, int) and n > 0 and n % 64 == 0
                                for n in lengths))


if __name__ == "__main__":
    unittest.main()
