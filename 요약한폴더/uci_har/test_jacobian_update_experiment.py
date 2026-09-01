import unittest

import numpy as np

from .jacobian_update_experiment import jacobian_delta_parts


class JacobianUpdateTest(unittest.TestCase):
    def test_bias_jacobian_part_has_batch_shape(self):
        model = {"w1": np.ones((2, 2)), "b1": np.zeros(2), "w2": np.ones((2, 2)), "b2": np.zeros(2), "w3": np.ones((2, 2)), "b3": np.zeros(2)}
        parts, _, _ = jacobian_delta_parts(model, np.ones((3, 2)), {name: np.zeros_like(value) for name, value in model.items()})
        self.assertEqual(parts["b2"].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
