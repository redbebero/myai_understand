import random
import unittest
from pathlib import Path

from train_spiral import forward
from weight_readable_model import predict


class WeightReadableModelTest(unittest.TestCase):
    def test_explicit_equations_match_original_weights(self):
        import json

        model = json.loads((Path(__file__).parent / "spiral_model.json").read_text())
        rng = random.Random(7)
        for _ in range(100):
            point = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
            self.assertAlmostEqual(predict(point), forward(model, point)[2], places=14)


if __name__ == "__main__":
    unittest.main()
