import unittest

import numpy as np

from .geometry_principle_experiment import geometry_metrics


class GeometryPrincipleTest(unittest.TestCase):
    def test_separated_classes_have_larger_between_structure(self):
        values = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
        targets = np.array([0, 0, 1, 1])
        result = geometry_metrics(values, targets)
        self.assertGreater(result["between_variance"], result["within_variance"])
        self.assertGreater(result["distance_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
