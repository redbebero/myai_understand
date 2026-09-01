import unittest

import numpy as np

from .distribution_inverse_experiment import _gaussian_pdf


class DistributionInverseTest(unittest.TestCase):
    def test_gaussian_pdf_symmetry(self):
        values = _gaussian_pdf(np.array([-1.0, 1.0]), 0.0, 1.0)
        self.assertAlmostEqual(values[0], values[1])


if __name__ == "__main__":
    unittest.main()
