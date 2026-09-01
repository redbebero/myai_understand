import unittest

import numpy as np

from .quantization_vs_compression import _quantize_dequantize, model_storage_bytes


class QuantizationComparisonTest(unittest.TestCase):
    def test_quantization_preserves_shape_and_uses_requested_precision(self):
        values = np.array([-2.0, 0.0, 2.0])
        result = _quantize_dequantize(values, 8)
        self.assertEqual(result.shape, values.shape)
        self.assertLessEqual(np.max(np.abs(result - values)), 2.0 / 127)

    def test_storage_scales_with_bit_width(self):
        model = {"w0": np.zeros((2, 3)), "b0": np.zeros(3)}
        self.assertEqual(model_storage_bytes(model, 8), 9)
        self.assertEqual(model_storage_bytes(model, 4), 5)


if __name__ == "__main__":
    unittest.main()
