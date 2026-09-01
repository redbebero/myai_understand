import unittest

import numpy as np

from .class_pair_inverse_experiment import _normalize


class ClassPairInverseTest(unittest.TestCase):
    def test_normalize_has_unit_norm(self):
        self.assertAlmostEqual(np.linalg.norm(_normalize(np.array([3.0, 4.0]))), 1.0)


if __name__ == "__main__":
    unittest.main()
