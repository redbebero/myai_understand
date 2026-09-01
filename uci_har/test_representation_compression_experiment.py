import unittest

import numpy as np

from .representation_compression_experiment import (
    _class_separating_basis,
    _evaluate_method,
    _pairwise_distance_correlation,
    _project_and_reconstruct,
)
from .generalization_experiment import _init_model


class RepresentationCompressionTest(unittest.TestCase):
    def test_projection_reconstructs_original_shape(self):
        values = np.arange(12, dtype=float).reshape(4, 3)
        basis = np.eye(3)[:2]
        result = _project_and_reconstruct(values, basis, values.mean(axis=0))
        self.assertEqual(result.shape, values.shape)

    def test_class_basis_is_orthonormal(self):
        values = np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=float)
        labels = np.array([0, 0, 1, 1])
        basis = _class_separating_basis(values, labels, 2)
        np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-10)

    def test_identical_distance_geometry_has_perfect_correlation(self):
        values = np.array([[0, 0], [1, 0], [0, 2]], dtype=float)
        self.assertAlmostEqual(_pairwise_distance_correlation(values, values), 1.0)

    def test_identity_compression_preserves_predictions_and_model(self):
        model = _init_model(2, (2, 2), 2, 7)
        before = {name: value.copy() for name, value in model.items()}
        values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        labels = np.array([0, 1, 0])
        metrics = _evaluate_method(model, values, values, labels, np.eye(2), np.zeros(2))
        self.assertAlmostEqual(metrics["prediction_agreement"], 1.0)
        for name in model:
            np.testing.assert_array_equal(model[name], before[name])


if __name__ == "__main__":
    unittest.main()
