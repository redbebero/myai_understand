import unittest

import numpy as np

from .uci_har_experiment import (
    INPUTS,
    HIDDEN,
    CLASSES,
    baseline_forward,
    count_parameters,
    role_features,
    redesign_predict,
    rank_features,
)


class UciHarExperimentTest(unittest.TestCase):
    def test_model_dimensions_and_parameter_count(self):
        self.assertEqual(INPUTS, 561)
        self.assertEqual(HIDDEN, (64, 32))
        self.assertEqual(CLASSES, 6)
        self.assertEqual(count_parameters(), 38246)

    def test_forward_exposes_hidden_activations(self):
        model = {
            "w1": np.zeros((INPUTS, 64)),
            "b1": np.zeros(64),
            "w2": np.zeros((64, 32)),
            "b2": np.zeros(32),
            "w3": np.zeros((32, CLASSES)),
            "b3": np.zeros(CLASSES),
        }
        hidden1, hidden2, probabilities = baseline_forward(model, np.zeros(INPUTS))
        self.assertEqual(hidden1.shape, (64,))
        self.assertEqual(hidden2.shape, (32,))
        self.assertEqual(probabilities.shape, (CLASSES,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_role_features_are_fixed_and_named(self):
        features = role_features(np.arange(INPUTS, dtype=float))
        self.assertEqual(len(features), 8)
        self.assertTrue(np.isfinite(features).all())

    def test_redesign_predict_returns_one_of_six_classes(self):
        model = {"weights": np.zeros((8, CLASSES))}
        prediction = redesign_predict(model, np.zeros(INPUTS))
        self.assertIn(prediction, range(CLASSES))

    def test_feature_ranking_has_one_score_per_input(self):
        model = {
            "w1": np.ones((INPUTS, 64)),
            "w2": np.ones((64, 32)),
            "w3": np.ones((32, CLASSES)),
        }
        self.assertEqual(rank_features(model).shape, (INPUTS,))


if __name__ == "__main__":
    unittest.main()
