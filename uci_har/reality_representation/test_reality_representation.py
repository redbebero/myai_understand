import unittest

import numpy as np

from experiment import break_relations, dictionary, intervene_coupling, load_raw


class RealityRepresentationTests(unittest.TestCase):
    def test_raw_loader_has_all_channels_and_window(self):
        train_x, train_y, train_subject, test_x, test_y, test_subject = load_raw()
        self.assertEqual(train_x.shape[1:], (9, 128))
        self.assertEqual(test_x.shape[1:], (9, 128))
        self.assertEqual(len(train_x), len(train_y))
        self.assertEqual(len(test_x), len(test_y))
        self.assertEqual(len(train_x), len(train_subject))
        self.assertEqual(len(test_x), len(test_subject))

    def test_relation_breaking_preserves_each_channel_marginal(self):
        x = np.arange(3 * 9 * 128, dtype=float).reshape(3, 9, 128)
        broken = break_relations(x, seed=7)
        for sample in range(len(x)):
            for channel in range(9):
                np.testing.assert_array_equal(np.sort(x[sample, channel]), np.sort(broken[sample, channel]))

    def test_dictionary_contains_channel_and_coupling_features(self):
        x = np.ones((4, 9, 128))
        values, names = dictionary(x)
        self.assertEqual(values.shape[0], 4)
        self.assertIn("body_acc_x:energy", names)
        self.assertTrue(any("×" in name for name in names))

    def test_coupling_intervention_preserves_marginals(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 9, 128))
        changed = intervene_coupling(x, "body_acc_x×body_gyro_z:coupling", seed=2)
        np.testing.assert_array_equal(np.sort(x[:, 5]), np.sort(changed[:, 5]))
        self.assertFalse(np.array_equal(x[:, 5], changed[:, 5]))


if __name__ == "__main__":
    unittest.main()
