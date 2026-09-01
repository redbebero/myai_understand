import unittest

from .training_geometry_experiment import CHECKPOINTS, _onset


class TrainingGeometryTest(unittest.TestCase):
    def test_checkpoint_schedule_and_onset(self):
        self.assertEqual(CHECKPOINTS, (0, 1, 2, 5, 10, 20, 80))
        points = [{"epoch": 0, "gain": 0.0}, {"epoch": 2, "gain": 0.2}, {"epoch": 5, "gain": 1.0}]
        self.assertEqual(_onset(points, "gain"), 2)


if __name__ == "__main__":
    unittest.main()
