import unittest

import numpy as np

from .adam_decomposition_experiment import _w1_norm


class AdamDecompositionTest(unittest.TestCase):
    def test_w1_norm_is_euclidean(self):
        self.assertAlmostEqual(_w1_norm(np.array([[3.0, 4.0]])), 5.0)


if __name__ == "__main__":
    unittest.main()
