import unittest

from .bottleneck_experiment import activation_bytes, count_parameters, operation_counts


class BottleneckExperimentTest(unittest.TestCase):
    def test_parameter_count_changes_only_the_bottleneck_layers(self):
        self.assertEqual(count_parameters(561, (64, 32), 6), 38246)
        self.assertEqual(count_parameters(561, (64, 8), 6), 36542)

    def test_activation_bytes_exclude_input_and_use_fp32(self):
        self.assertEqual(activation_bytes((64, 32), 6), (64 + 32 + 6) * 4)
        self.assertEqual(activation_bytes((64, 8), 6), (64 + 8 + 6) * 4)

    def test_operation_counts_use_two_flops_per_mac(self):
        macs, flops = operation_counts(561, (64, 8), 6)
        self.assertEqual(macs, 36464)
        self.assertEqual(flops, 2 * macs)


if __name__ == "__main__":
    unittest.main()
