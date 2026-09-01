import unittest

import numpy as np

from .variance_scaling_experiment import _pearson


class VarianceScalingTest(unittest.TestCase):
    def test_pearson_detects_opposite_coordinates(self):
        self.assertAlmostEqual(_pearson(np.array([1.0, 2.0]), np.array([2.0, 1.0])), -1.0)


if __name__ == "__main__":
    unittest.main()
