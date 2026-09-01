import unittest

import numpy as np

from .gradient_vs_inverse_experiment import _cosine


class GradientVsInverseTest(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(_cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 1.0)
        self.assertAlmostEqual(_cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
