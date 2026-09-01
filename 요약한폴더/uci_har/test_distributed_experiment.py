import unittest

import numpy as np

from .distributed_experiment import (
    activation_concentration,
    class_activation_profile,
    pattern_similarity,
    select_discriminative_units,
)


class DistributedExperimentTest(unittest.TestCase):
    def test_pattern_similarity_is_invariant_to_unit_permutation(self):
        values = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0]])
        permuted = values[:, [2, 0, 1]]
        self.assertAlmostEqual(pattern_similarity(values, permuted), 1.0, places=6)

    def test_class_activation_profile_reports_three_class_distances(self):
        values = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 3.0]])
        targets = np.array([0, 0, 1, 1, 2, 2])
        profile = class_activation_profile(values, targets, labels=(0, 1, 2))
        self.assertEqual(profile["sample_count"], 6)
        self.assertEqual(set(profile["centroids"]), {"0", "1", "2"})
        self.assertEqual(len(profile["centroid_distances"]), 3)

    def test_concentration_and_selection_measure_distributed_contrast(self):
        values = np.array([[0.0, 1.0, 0.2], [1.0, 0.0, 0.2], [0.0, 2.0, 0.4], [2.0, 0.0, 0.4]])
        targets = np.array([0, 0, 1, 1])
        selected = select_discriminative_units(values, targets, top_k=2, labels=(0, 1))
        concentration = activation_concentration(values, targets, labels=(0, 1))
        self.assertEqual(len(selected["units"]), 2)
        self.assertGreaterEqual(concentration["top_2_fraction"], concentration["top_1_fraction"])


if __name__ == "__main__":
    unittest.main()
