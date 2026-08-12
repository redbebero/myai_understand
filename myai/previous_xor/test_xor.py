import json
import tempfile
import unittest
from pathlib import Path

from generate_xor import generate_dataset
from train_xor import evaluate, load_dataset, nonzero_parameters, prune_connection, train


class XorExperimentTest(unittest.TestCase):
    def test_generate_train_and_prune(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "xor_dataset.json"
            model_path = Path(directory) / "xor_model.json"

            generate_dataset(dataset_path)
            rows = load_dataset(dataset_path)
            self.assertEqual(rows, [
                {"inputs": [0, 0], "target": 0},
                {"inputs": [0, 1], "target": 1},
                {"inputs": [1, 0], "target": 1},
                {"inputs": [1, 1], "target": 0},
            ])

            model = train(rows, epochs=10_000, seed=7)
            self.assertGreaterEqual(evaluate(model, rows), 1.0)

            before = nonzero_parameters(model)
            prune_connection(model, layer=0, row=0, column=0)
            self.assertEqual(nonzero_parameters(model), before - 1)
            self.assertLess(evaluate(model, rows), 1.0)

            model_path.write_text(json.dumps(model), encoding="utf-8")
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()
