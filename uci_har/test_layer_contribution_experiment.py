import unittest

import numpy as np

from .layer_contribution_experiment import _scale_group


class LayerContributionTest(unittest.TestCase):
    def test_group_scaling_reaches_common_norm(self):
        delta = {"w1": np.ones((2, 2)), "b1": np.ones(2), "w2": np.zeros((2, 2)), "b2": np.zeros(2)}
        scaled = _scale_group(delta, "W1", 2.0)
        self.assertAlmostEqual(np.sqrt(np.sum(scaled["w1"] ** 2) + np.sum(scaled["b1"] ** 2)), 2.0)


if __name__ == "__main__":
    unittest.main()
