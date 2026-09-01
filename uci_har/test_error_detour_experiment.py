import unittest

import numpy as np

from .error_detour_experiment import _loss, _run_condition


class ErrorDetourExperimentTest(unittest.TestCase):
    def test_ascent_then_descent_records_expected_phases(self):
        rng = np.random.default_rng(3)
        inputs = rng.normal(size=(24, 4))
        targets = np.arange(24) % 2
        data = {
            "train_x": inputs,
            "train_y": targets,
            "val_x": inputs,
            "val_y": targets,
            "test_x": inputs,
            "test_y": targets,
        }
        result = _run_condition(data, seed=3, updates=4, ascent_updates=2, batch_size=24, learning_rate=0.01)
        self.assertEqual([row["phase"] for row in result["records"]], ["start", "ascent", "ascent", "descent", "descent"])
        self.assertGreater(result["records"][2]["train_loss"], result["records"][0]["train_loss"])
        self.assertLess(result["records"][-1]["train_loss"], result["records"][2]["train_loss"])

    def test_loss_is_finite(self):
        model = {"w0": np.zeros((2, 2)), "b0": np.zeros(2)}
        inputs = np.ones((2, 2))
        targets = np.array([0, 1])
        self.assertAlmostEqual(_loss(model, inputs, targets, 0), np.log(2))


if __name__ == "__main__":
    unittest.main()
