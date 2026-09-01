import unittest

import numpy as np

from .validation_selective_inverse_experiment import _partition


class ValidationSelectiveInverseTest(unittest.TestCase):
    def test_partition_is_disjoint_and_complete(self):
        class Model:
            pass
        model = {"w0": np.eye(2), "b0": np.zeros(2), "w1": np.eye(2), "b1": np.zeros(2), "w2": np.eye(2), "b2": np.zeros(2)}
        inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
        targets = np.array([0, 1])
        result = _partition(model, inputs, targets)
        self.assertEqual(set(result["misclassified"]) | set(result["vulnerable"]) | set(result["safe"]), {0, 1})
        self.assertEqual(set(result["vulnerable"]) & set(result["safe"]), set())


if __name__ == "__main__":
    unittest.main()
