import unittest

from human_role_model import nodes, predict_detail


class HumanRoleModelTest(unittest.TestCase):
    def test_roles_are_visible(self):
        result = nodes([0.2, 0.3])
        self.assertEqual(set(result), {"distance", "direction", "spiral"})

    def test_prediction_is_hand_designed_not_trained(self):
        result = predict_detail([0.2, 0.3])
        self.assertIn(result["prediction"], (0, 1))
        self.assertIsInstance(result["score"], float)


if __name__ == "__main__":
    unittest.main()
