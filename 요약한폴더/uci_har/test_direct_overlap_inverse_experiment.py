import unittest

import numpy as np

from .direct_overlap_inverse_experiment import _sigmoid


class DirectOverlapInverseTest(unittest.TestCase):
    def test_sigmoid_is_bounded(self):
        values = _sigmoid(np.array([-100.0, 0.0, 100.0]))
        self.assertTrue(np.all((values > 0.0) & (values <= 1.0)))


if __name__ == "__main__":
    unittest.main()
