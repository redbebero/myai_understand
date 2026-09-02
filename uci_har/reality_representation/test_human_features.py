import unittest

import numpy as np

from human_features import extract_family_values, extract_human_features


class HumanFeatureTests(unittest.TestCase):
    def test_feature_matrix_and_family_columns(self):
        x = np.zeros((3, 9, 128), dtype=float)
        x[:, 0] = 2.0
        matrix, names, columns = extract_human_features(x)
        self.assertEqual(matrix.shape, (3, 144))
        self.assertEqual(matrix.shape[1], len(names))
        self.assertEqual(sorted(columns), sorted(extract_family_values(x)[0]))
        self.assertTrue(np.allclose(matrix[:, columns["movement_energy"][0]], 4.0))
        self.assertTrue(np.allclose(matrix[:, columns["sensor_level"][0]], 2.0))

        matrix, names, columns = extract_human_features(x)
        corr_columns = [i for i in columns["cross_channel_coordination"] if names[i].endswith(":correlation")]
        self.assertTrue(np.isfinite(matrix).all())
        self.assertGreater(len(corr_columns), 0)


if __name__ == "__main__":
    unittest.main()
