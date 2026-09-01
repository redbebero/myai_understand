import unittest

from bitwise_experiment import binary_dot, quantize


class BitwiseExperimentTest(unittest.TestCase):
    def test_binary_dot_matches_signed_products(self):
        self.assertEqual(binary_dot([1, -1, 1], [1, 1, -1]), -1)

    def test_quantization_keeps_shape_and_limits_precision(self):
        values = [[-1.0, 0.1], [0.2, 1.0]]
        quantized = quantize(values, 2)
        self.assertEqual([len(row) for row in quantized], [2, 2])
        self.assertLessEqual(len(set(value for row in quantized for value in row)), 3)


if __name__ == "__main__":
    unittest.main()
