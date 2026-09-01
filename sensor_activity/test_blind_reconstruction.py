import unittest

from blind_reconstruction import candidate_features, select_roles


class BlindReconstructionTest(unittest.TestCase):
    def test_generic_candidate_features_are_numeric(self):
        names, values = candidate_features([0.1] * 36)
        self.assertEqual(len(names), len(values))
        self.assertTrue(all(isinstance(value, float) for value in values))

    def test_role_selection_uses_activation_candidates(self):
        table = [{"node": 0, "top_candidate_roles": [{"name": "rms", "correlation": 0.9}]}]
        self.assertEqual(select_roles(table, 1), ["rms"])


if __name__ == "__main__":
    unittest.main()
