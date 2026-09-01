import unittest
from pathlib import Path

from generate_dataset import generate
from train_dynamic import active_connections, evaluate, load, train


ROOT = Path(__file__).parent


class DynamicSparseTest(unittest.TestCase):
    def test_training_and_rewiring(self):
        path = ROOT / "test_data.json"
        generate(path, points_per_class=20, seed=7)
        rows = load(path)
        model, history = train(rows, epochs=150, rewire_every=50)
        self.assertEqual(active_connections(model), 44)
        self.assertEqual(len(history), 3)
        self.assertEqual(len(model["rewire_events"]), 3)
        self.assertGreaterEqual(evaluate(model, rows), 0.5)
        path.unlink()


if __name__ == "__main__":
    unittest.main()
