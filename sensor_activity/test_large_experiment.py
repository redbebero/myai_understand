import unittest

from large_experiment import INPUTS, STEPS, features


class LargeExperimentTest(unittest.TestCase):
    def test_feature_shape(self):
        names, values = features([0.0] * INPUTS)
        self.assertEqual(len(names), len(values))
        self.assertEqual(STEPS, 24)


if __name__ == "__main__": unittest.main()
