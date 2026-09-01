import unittest

import numpy as np

from .subspace_experiment import direction_geometry, shared_private_subspace


class SubspaceExperimentTest(unittest.TestCase):
    def test_direction_geometry_reports_angles_and_cosines(self):
        directions = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = direction_geometry(directions)
        self.assertAlmostEqual(result["cosines"]["0-1"], 0.0)
        self.assertIn("angles_degrees", result)

    def test_shared_private_subspace_reconstructs_each_direction(self):
        directions = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = shared_private_subspace(directions)
        self.assertEqual(result["shared_component"].shape, (2,))
        self.assertEqual(result["private_components"].shape, directions.shape)
        self.assertGreater(result["shared_explained_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
