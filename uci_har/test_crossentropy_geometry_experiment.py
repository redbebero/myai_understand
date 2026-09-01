import unittest

import numpy as np

from .crossentropy_geometry_experiment import _pairwise_cosines


class CrossEntropyGeometryTest(unittest.TestCase):
    def test_pairwise_cosines_has_one_value_per_pair(self):
        result = _pairwise_cosines(np.eye(3))
        self.assertEqual(len(result), 3)
        self.assertTrue(np.allclose(result, 0.0))


if __name__ == "__main__":
    unittest.main()
