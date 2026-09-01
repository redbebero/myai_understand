import unittest

from experiment import INPUTS, LABELS, make_dataset, new_model, role_features


class SensorExperimentTest(unittest.TestCase):
    def test_dataset_shape_and_roles(self):
        rows = make_dataset("/tmp/sensor_activity_test.json", per_class=2, seed=3)
        self.assertEqual(len(rows), 2 * len(LABELS))
        self.assertEqual(len(rows[0]["inputs"]), INPUTS)
        self.assertEqual(len(role_features(rows[0]["inputs"])), 7)

    def test_model_architecture(self):
        self.assertEqual(new_model()["architecture"], [INPUTS, 4, 4, 4])


if __name__ == "__main__":
    unittest.main()
