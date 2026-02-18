from __future__ import annotations

from pathlib import Path
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
    from trajectory_generation.spline import CatmullRomSpline
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


@unittest.skipUnless(_HAS_NUMPY, "NumPy-based spline tests require numpy")
class CatmullRomSplineTest(unittest.TestCase):
    def test_spline_starts_and_ends_at_control_points(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        )
        spline = CatmullRomSpline(points)
        sampled = spline.evaluate(50)

        self.assertTrue(np.allclose(sampled[0], points[0], atol=1e-6))
        self.assertTrue(np.allclose(sampled[-1], points[-1], atol=1e-6))

    def test_single_segment_returns_array(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        spline = CatmullRomSpline(points)
        sampled = spline.evaluate(10)
        self.assertEqual(sampled.shape[0], 10)
        self.assertEqual(sampled.shape[1], 3)

    def test_coincident_control_points_still_return_requested_sample_count(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        spline = CatmullRomSpline(points)
        sampled = spline.evaluate(128)
        self.assertEqual(sampled.shape, (128, 3))
        self.assertTrue(np.all(np.isfinite(sampled)))

    def test_zero_samples_returns_empty_array(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        spline = CatmullRomSpline(points)
        sampled = spline.evaluate(0)
        self.assertEqual(sampled.shape, (0, 3))

    def test_single_point_evaluate_returns_copy(self) -> None:
        points = np.array([[2.0, 3.0, 4.0]])
        spline = CatmullRomSpline(points)
        sampled = spline.evaluate(5)
        self.assertEqual(sampled.shape, (1, 3))
        sampled[0, 0] = -100.0
        self.assertNotEqual(float(spline.points[0, 0]), -100.0)


if __name__ == "__main__":
    unittest.main()
