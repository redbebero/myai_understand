import unittest

import numpy as np

from .gradient_averaging_experiment import _gradient_decomposition


class GradientAveragingTest(unittest.TestCase):
    def test_residuals_cancel_in_mean(self):
        result = _gradient_decomposition(np.asarray([[1.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(result["mean_norm"], np.sqrt(0.5))
        self.assertAlmostEqual(result["residual_cancellation"], 1.0 - np.sqrt(0.5))
        self.assertTrue(np.allclose(result["residual"].mean(axis=0), 0.0))


if __name__ == "__main__":
    unittest.main()
