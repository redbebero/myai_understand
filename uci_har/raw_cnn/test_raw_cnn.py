import unittest

import numpy as np

from .raw_cnn_experiment import (
    CHANNELS,
    STEPS,
    FILTERS,
    KERNEL,
    CLASSES,
    count_parameters,
    cnn_forward,
    raw_role_features,
    expanded_role_features,
    expanded_role_names,
    temporal_role_features,
    temporal_role_names,
    quantized_model,
    quantized_accuracy,
)


class RawCnnExperimentTest(unittest.TestCase):
    def test_raw_shape_and_parameter_count(self):
        self.assertEqual((CHANNELS, STEPS), (9, 128))
        self.assertEqual((FILTERS, KERNEL), (12, 9))
        self.assertEqual(count_parameters(), 1062)

    def test_quantized_model_keeps_shape_and_returns_accuracy(self):
        model = {
            "kernels": np.ones((FILTERS, CHANNELS, KERNEL)),
            "bias": np.zeros(FILTERS),
            "output": np.ones((FILTERS, CLASSES)),
            "output_bias": np.zeros(CLASSES),
        }
        quantized = quantized_model(model, 8)
        self.assertEqual(quantized["kernels"].shape, model["kernels"].shape)
        self.assertIsInstance(quantized_accuracy(quantized, np.zeros((2, CHANNELS, STEPS)), np.zeros(2)), float)

    def test_forward_exposes_filter_activations(self):
        model = {
            "kernels": np.zeros((FILTERS, CHANNELS, KERNEL)),
            "bias": np.zeros(FILTERS),
            "output": np.zeros((FILTERS, CLASSES)),
            "output_bias": np.zeros(CLASSES),
        }
        filters, pooled, probabilities = cnn_forward(model, np.zeros((2, CHANNELS, STEPS)))
        self.assertEqual(filters.shape, (2, FILTERS, STEPS - KERNEL + 1))
        self.assertEqual(pooled.shape, (2, FILTERS))
        self.assertEqual(probabilities.shape, (2, CLASSES))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_roles_are_finite_and_named_by_shape(self):
        features = raw_role_features(np.ones((CHANNELS, STEPS)))
        self.assertEqual(features.shape, (CHANNELS * 6 + 3,))
        self.assertTrue(np.isfinite(features).all())

    def test_expanded_roles_preserve_windows_and_named_calculations(self):
        features = expanded_role_features(np.ones((CHANNELS, STEPS)))
        self.assertEqual(features.shape, (915,))
        self.assertEqual(len(expanded_role_names()), 915)
        self.assertTrue(np.isfinite(features).all())

    def test_temporal_roles_preserve_window_order_and_sensor_relations(self):
        features = temporal_role_features(np.ones((CHANNELS, STEPS)))
        self.assertEqual(features.shape, (147,))
        self.assertEqual(len(temporal_role_names()), 147)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
