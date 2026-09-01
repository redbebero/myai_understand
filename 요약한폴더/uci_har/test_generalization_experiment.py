import unittest

import numpy as np

from .generalization_experiment import _parameter_norm


class GeneralizationTest(unittest.TestCase):
    def test_parameter_norm_uses_weight_and_bias(self):
        delta = {"w0": np.ones((2, 2)), "b0": np.ones(2)}
        self.assertAlmostEqual(_parameter_norm(delta, 0), np.sqrt(6))


if __name__ == "__main__":
    unittest.main()
