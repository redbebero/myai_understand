import unittest

import numpy as np

from .direction_experiment import (
    activity_direction,
    correlation_control_selection,
    fit_r2,
    remove_direction_projection,
)


class DirectionExperimentTest(unittest.TestCase):
    def test_activity_direction_is_normalized_class_contrast(self):
        values = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        targets = np.array([0, 0, 1, 1])
        direction = activity_direction(values, targets, label=0)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)
        self.assertGreater(direction[0], 0.0)

    def test_r2_increases_for_a_joint_feature_combination(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        y = np.array([1.0, 1.0, 2.0, 0.0])
        self.assertGreater(fit_r2(x, y), fit_r2(x[:, :1], y))

    def test_correlation_control_keeps_representatives_not_duplicates(self):
        x = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0], [3.0, 3.0, 0.0], [4.0, 4.0, 1.0]])
        y = np.array([0.0, 1.0, 0.0, 1.0])
        selected = correlation_control_selection(x, y, names=("a", "a-copy", "b"), max_features=2, correlation_limit=0.9)
        self.assertEqual(len(selected), 2)
        self.assertNotEqual({row["name"] for row in selected}, {"a", "a-copy"})

    def test_remove_direction_projection_removes_target_component(self):
        values = np.array([[1.0, 0.0], [2.0, 0.0]])
        direction = np.array([1.0, 0.0])
        changed = remove_direction_projection(values, direction, strength=1.0)
        self.assertTrue(np.allclose(changed[:, 0], 0.0))


if __name__ == "__main__":
    unittest.main()
