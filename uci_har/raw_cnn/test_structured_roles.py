import unittest

import numpy as np

from .structured_role_model import (
    structured_role_features,
    structured_role_names,
    STRUCTURED_FEATURE_COUNT,
)


class StructuredRoleModelTest(unittest.TestCase):
    def test_features_have_named_ordered_roles(self):
        values = structured_role_features(np.ones((9, 128)))
        self.assertEqual(values.shape, (STRUCTURED_FEATURE_COUNT,))
        self.assertEqual(len(structured_role_names()), STRUCTURED_FEATURE_COUNT)
        self.assertTrue(np.isfinite(values).all())

    def test_time_order_changes_features(self):
        values = np.zeros((9, 128))
        values[0, 16:32] = 1.0
        reversed_values = values[:, ::-1]
        self.assertFalse(np.allclose(structured_role_features(values), structured_role_features(reversed_values)))


if __name__ == "__main__":
    unittest.main()
