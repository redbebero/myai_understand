import unittest

from exact_reduction import rank


class ExactReductionTest(unittest.TestCase):
    def test_rank(self):
        self.assertEqual(rank([[1.0, 2.0], [2.0, 4.0]]), 1)
        self.assertEqual(rank([[1.0, 0.0], [0.0, 1.0]]), 2)


if __name__ == "__main__":
    unittest.main()
