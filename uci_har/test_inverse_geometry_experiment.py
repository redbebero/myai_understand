import unittest

import numpy as np

from .inverse_geometry_experiment import _target_geometry


class InverseGeometryTest(unittest.TestCase):
    def test_target_increases_centered_pair_distances(self):
        centroids = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        target = _target_geometry(centroids)
        self.assertGreater(np.linalg.norm(target[0] - target[1]), np.linalg.norm(centroids[0] - centroids[1]))


if __name__ == "__main__":
    unittest.main()
