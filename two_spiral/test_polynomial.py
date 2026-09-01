import unittest
from pathlib import Path

from polynomial_experiment import features, fit, predict, solve_linear


class PolynomialExperimentTest(unittest.TestCase):
    def test_monomial_nodes_have_expected_count(self):
        self.assertEqual(len(features([2.0, 3.0], 2)), 6)

    def test_solver_and_polynomial_prediction(self):
        self.assertEqual(solve_linear([[2.0]], [6.0]), [3.0])
        rows = [{"inputs": [0.0, 0.0]}, {"inputs": [1.0, 0.0]}, {"inputs": [2.0, 0.0]}]
        model = fit(rows, [0.0, 1.0, 2.0], degree=1)
        self.assertAlmostEqual(predict(model, [1.0, 0.0]), 1 / (1 + __import__("math").exp(-1)), places=4)


if __name__ == "__main__":
    unittest.main()
