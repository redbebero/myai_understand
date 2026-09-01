import unittest

from .gradient_geometry_experiment import _onset


class GradientGeometryTest(unittest.TestCase):
    def test_onset_uses_update_index(self):
        records = [{"update": 0, "probe": {"accuracy": 0.0, "geometry": {"separation_ratio": 0.0}}}, {"update": 1, "probe": {"accuracy": 0.5, "geometry": {"separation_ratio": 0.2}}}, {"update": 2, "probe": {"accuracy": 1.0, "geometry": {"separation_ratio": 1.0}}}]
        self.assertEqual(_onset(records, ("probe", "accuracy")), 1)


if __name__ == "__main__":
    unittest.main()
