import unittest

from recover import generate_probes, model_output, recover_formula, agreement


class SymbolicRecoveryTest(unittest.TestCase):
    def test_formula_matches_model_on_unseen_probes(self):
        probes = generate_probes(120, seed=11)
        train_probes = probes[:60]
        test_probes = probes[60:]
        formula = recover_formula(train_probes)
        self.assertLessEqual(formula["complexity"], 3)
        self.assertGreaterEqual(agreement(formula, test_probes), 0.70)

    def test_model_output_uses_saved_weights(self):
        output = model_output([0.1, -0.2])
        self.assertGreaterEqual(output, 0.0)
        self.assertLessEqual(output, 1.0)


if __name__ == "__main__":
    unittest.main()
