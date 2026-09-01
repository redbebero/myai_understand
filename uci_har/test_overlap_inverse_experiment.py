import unittest

import numpy as np

from .overlap_inverse_experiment import _gaussian_pdf


class OverlapInverseTest(unittest.TestCase):
    def test_gaussian_pdf_is_positive(self):
        self.assertGreater(_gaussian_pdf(np.array([0.0]), 0.0, 1.0)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
