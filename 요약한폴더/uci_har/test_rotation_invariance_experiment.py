import unittest

import numpy as np

from .rotation_invariance_experiment import _orthogonal


class RotationInvarianceTest(unittest.TestCase):
    def test_orthogonal_basis(self):
        matrix = _orthogonal(7, 4)
        self.assertTrue(np.allclose(matrix.T @ matrix, np.eye(4)))


if __name__ == "__main__":
    unittest.main()
