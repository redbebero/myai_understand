import unittest

import numpy as np

from .margin_inverse_experiment import _margin_metrics


class MarginInverseTest(unittest.TestCase):
    def test_margin_metrics_reports_accuracy_and_quantiles(self):
        model = {
            "w0": np.eye(2), "b0": np.zeros(2),
            "w1": np.eye(2), "b1": np.zeros(2),
            "w2": np.array([[2.0, 0.0], [0.0, 2.0]]), "b2": np.zeros(2),
        }
        metrics = _margin_metrics(model, np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["mean_margin"], 0.0)
        self.assertEqual(sum(map(sum, metrics["confusion"])), 2)


if __name__ == "__main__":
    unittest.main()
