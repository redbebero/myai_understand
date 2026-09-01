import unittest

import numpy as np

from .concentration_experiment import _information


class ConcentrationTest(unittest.TestCase):
    def test_information_share_sums_to_one(self):
        values = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
        targets = np.array([0, 0, 1, 1])
        result = _information(values, targets)
        self.assertAlmostEqual(result["share"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
