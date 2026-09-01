import unittest

import numpy as np

from .interaction_experiment import (
    ablate_hidden_pair,
    class_change_summary,
    evaluate_outputs,
    feature_condition_summary,
    joint_activation_summary,
    pair_interactions,
    pair_interaction_score,
)
from .uci_har_experiment import baseline_forward


def toy_model():
    return {
        "w1": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b1": np.zeros(2),
        "w2": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b2": np.zeros(2),
        "w3": np.array([[2.0, -2.0], [-2.0, 2.0]]),
        "b3": np.zeros(2),
    }


class InteractionExperimentTest(unittest.TestCase):
    def test_pair_ablation_zeroes_both_hidden_paths_without_mutating_source(self):
        model = toy_model()
        ablated = ablate_hidden_pair(model, layer=2, first=0, second=1)
        self.assertTrue(np.allclose(ablated["w2"][:, :], 0.0))
        self.assertTrue(np.allclose(ablated["w3"][:, :], 0.0))
        self.assertFalse(np.allclose(model["w3"], 0.0))

    def test_pair_interaction_score_is_joint_drop_minus_single_drops(self):
        score = pair_interaction_score(0.90, 0.80, 0.75, 0.50)
        self.assertAlmostEqual(score, 0.15)

    def test_evaluation_reports_accuracy_loss_probabilities_and_class_changes(self):
        model = toy_model()
        inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
        targets = np.array([0, 1])
        metrics = evaluate_outputs(model, inputs, targets)
        self.assertEqual(metrics["predictions"].shape, (2,))
        self.assertEqual(metrics["probabilities"].shape, (2, 2))
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertGreaterEqual(metrics["cross_entropy"], 0.0)
        changed = class_change_summary(model, ablate_hidden_pair(model, 1, 0, 1), inputs, targets)
        self.assertEqual(changed["prediction_flips"], 1)

    def test_joint_activation_summary_counts_conditions_by_target(self):
        model = toy_model()
        inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        targets = np.array([0, 1, 0])
        summary = joint_activation_summary(model, inputs, targets, layer=1, first=0, second=1)
        self.assertEqual(summary["joint_active"], 1)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(set(summary["by_class"]), {"0", "1"})

    def test_pair_interactions_returns_joint_activation_and_class_change_evidence(self):
        model = toy_model()
        inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        targets = np.array([0, 1, 0])
        pairs = pair_interactions(model, inputs, targets, layer=2, candidates=[0, 1])
        self.assertEqual(len(pairs), 1)
        self.assertIn("interaction_loss", pairs[0])
        self.assertIn("joint_activation", pairs[0])
        self.assertIn("class_changes", pairs[0])

    def test_feature_condition_summary_names_features_and_tests_condition_intervention(self):
        model = toy_model()
        train_x = np.array([[2.0, 2.0], [1.5, 1.5], [2.0, 0.0], [0.0, 2.0]])
        train_y = np.array([0, 0, 1, 1])
        test_x = train_x.copy()
        test_y = train_y.copy()
        names = ("sensor-x", "sensor-y")
        summary = feature_condition_summary(
            model, train_x, train_y, test_x, test_y, names, layer=1, first=0, second=1, top_features=1
        )
        self.assertEqual(summary["features"][0]["name"], "sensor-x")
        self.assertIn("rule", summary)
        self.assertIn("test_intervention", summary)
        self.assertIn("dynamic_judgment", summary["rule"])
        self.assertIn("judgment_precision", summary["test_rule"])
        self.assertIn("matched_count", summary["rule"])


if __name__ == "__main__":
    unittest.main()
