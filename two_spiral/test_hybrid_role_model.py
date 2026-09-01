import unittest

from hybrid_role_model import role_features, train


class HybridRoleModelTest(unittest.TestCase):
    def test_only_final_combination_is_trainable(self):
        rows = [{"inputs": [0.2, 0.3], "target": 1}, {"inputs": [-0.2, -0.3], "target": 0}]
        model = train(rows, epochs=1)
        self.assertEqual(model["trainable_parameter_count"], len(role_features(rows[0]["inputs"])))
        self.assertEqual(model["fixed_spiral_constant"], 4 * __import__("math").pi)


if __name__ == "__main__":
    unittest.main()
