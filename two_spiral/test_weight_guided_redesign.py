import unittest

from weight_guided_redesign import readable_equations


class WeightGuidedRedesignTest(unittest.TestCase):
    def test_selected_node_equation_is_readable(self):
        model = {"weights_input_hidden1": [[1.0, -2.0]], "bias_hidden1": [0.5]}
        self.assertIn("tanh", readable_equations(model, [0])[0])
        self.assertIn("original hidden1[0]", readable_equations(model, [0])[0])


if __name__ == "__main__":
    unittest.main()
