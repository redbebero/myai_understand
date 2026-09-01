import unittest

import numpy as np

from .gradient_averaging_controls import _cosine


class GradientAveragingControlsTest(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(_cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
