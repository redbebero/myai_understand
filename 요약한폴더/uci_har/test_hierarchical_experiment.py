import unittest

import numpy as np

from .hierarchical_experiment import (
    ablation_effect_by_class,
    classwise_metrics,
    hierarchy_relation,
)
from .interaction_experiment import ablate_hidden_pair


def toy_model():
    return {
        "w1": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b1": np.zeros(2),
        "w2": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b2": np.zeros(2),
        "w3": np.array([[2.0, -2.0], [-2.0, 2.0]]),
        "b3": np.zeros(2),
    }


class HierarchicalExperimentTest(unittest.TestCase):
    def test_classwise_metrics_reports_counts_accuracy_and_confusion(self):
        model = toy_model()
        inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        targets = np.array([0, 1, 0])
        result = classwise_metrics(model, inputs, targets)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["by_class"]["0"]["count"], 2)
        self.assertEqual(sum(map(sum, result["confusion"])), 3)

    def test_pair_ablation_reports_per_class_effect(self):
        model = toy_model()
        inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        targets = np.array([0, 1, 0])
        ablated = ablate_hidden_pair(model, layer=2, first=0, second=1)
        result = ablation_effect_by_class(model, ablated, inputs, targets)
        self.assertIn("baseline", result)
        self.assertIn("ablated", result)
        self.assertIn("accuracy_delta", result["by_class"]["0"])

    def test_hierarchy_relation_distinguishes_downstream_and_feature_overlap(self):
        result = hierarchy_relation(
            {"layer": 1, "first": 0, "second": 1, "feature_names": ["a", "b"]},
            {"layer": 2, "first": 2, "second": 3, "feature_names": ["b", "c"]},
        )
        self.assertTrue(result["downstream_layer_order"])
        self.assertEqual(result["feature_overlap"], ["b"])


if __name__ == "__main__":
    unittest.main()
