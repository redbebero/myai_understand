import unittest

import numpy as np

from .input_geometry_experiment import _alignment


class InputGeometryTest(unittest.TestCase):
    def test_alignment_is_one_inside_basis(self):
        basis = np.eye(2)
        matrix = np.eye(2)
        self.assertAlmostEqual(_alignment(matrix, basis), 1.0)


if __name__ == "__main__":
    unittest.main()
