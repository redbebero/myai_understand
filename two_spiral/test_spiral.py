import copy
import json
import unittest
from pathlib import Path

from generate_spirals import generate
from train_spiral import evaluate, load_dataset, new_model, nonzero_weights, prune_neuron, train


ROOT = Path(__file__).parent


class SpiralExperimentTest(unittest.TestCase):
    def test_generate_train_prune_and_record(self):
        train_path = ROOT / "test_train.json"
        test_path = ROOT / "test_test.json"
        generate(train_path, points_per_class=60, noise=0.08, seed=7)
        generate(test_path, points_per_class=60, noise=0.08, seed=19)
        train_rows = load_dataset(train_path)
        test_rows = load_dataset(test_path)
        model = train(train_rows, epochs=1200, seed=3)
        self.assertGreaterEqual(evaluate(model, train_rows), 0.8)
        self.assertGreaterEqual(evaluate(model, test_rows), 0.7)

        before = nonzero_weights(model)
        candidate = copy.deepcopy(model)
        prune_neuron(candidate, 1, 0)
        self.assertEqual(nonzero_weights(candidate), before - 14)
        self.assertLessEqual(evaluate(candidate, test_rows), evaluate(model, test_rows))

        train_path.unlink()
        test_path.unlink()


if __name__ == "__main__":
    unittest.main()
