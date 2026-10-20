"""Known full-orientation, inclination, and singular-heading contracts."""

import importlib.util
import math
import unittest

import numpy as np


class BroadOrientationMetricsTests(unittest.TestCase):
    def test_known_rotations_separate_heading_inclination_and_singular_samples(self):
        self.assertIsNotNone(importlib.util.find_spec("model.broad_evaluate"),
                             "BROAD held-out metrics module must exist")
        from model.broad_evaluate import orientation_metrics

        predictions = np.array([[1,0,0,0], [math.sqrt(.5),0,0,math.sqrt(.5)],
                                [math.cos(math.pi/6),math.sin(math.pi/6),0,0],
                                [0,1,0,0], [-1,0,0,0]], dtype=np.float64)
        targets = np.tile([1.,0,0,0], (5,1))
        result = orientation_metrics(predictions, targets, np.ones(5,dtype=bool), 1e-12)
        self.assertAlmostEqual(result["full_rmse_deg"], math.sqrt((90**2+60**2+180**2)/5), places=6)
        self.assertAlmostEqual(result["inclination_rmse_deg"], math.sqrt((60**2+180**2)/5), places=6)
        self.assertAlmostEqual(result["heading_rmse_deg"], 45, places=6)
        self.assertEqual(result["samples"], 5)
        self.assertEqual(result["heading_samples"], 4)
        self.assertEqual(result["heading_singular_samples"], 1)
        singular = orientation_metrics(predictions[3:4], targets[3:4], np.array([True]), 1e-12)
        self.assertIsNone(singular["heading_rmse_deg"])


if __name__ == "__main__":
    unittest.main()
