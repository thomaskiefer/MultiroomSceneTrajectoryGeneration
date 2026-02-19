from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import trajectory_generation.validation as validation_module
from trajectory_generation.validation import validate_trajectory

try:
    from shapely.geometry import Polygon

    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


class ValidationTest(unittest.TestCase):
    def test_validate_trajectory_ok(self) -> None:
        frames = [
            {
                "id": 0,
                "position": [0.0, 0.0, 1.6],
                "look_at": [1.0, 0.0, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            },
            {
                "id": 1,
                "position": [0.2, 0.0, 1.6],
                "look_at": [1.2, 0.0, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            },
        ]
        warnings = validate_trajectory(frames)
        self.assertEqual(warnings, [])

    def test_validate_trajectory_detects_invalid_vectors(self) -> None:
        frames = [
            {
                "id": 0,
                "position": [0.0, 0.0, 1.6],
                "look_at": [1.0, 0.0, 1.6],
                "forward": [2.0, 0.0, 0.0],
                "up": [1.0, 0.0, 0.0],
                "fov": 60.0,
            },
            {
                "id": 1,
                "position": [float("nan"), 0.0, 1.6],
                "look_at": [1.0, 0.0, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            },
        ]
        warnings = validate_trajectory(frames)
        self.assertGreaterEqual(len(warnings), 2)
        combined = " ".join(warnings).lower()
        self.assertIn("non-unit forward", combined)
        self.assertIn("nan/inf", combined)
        self.assertIn("not perpendicular", combined)

    @unittest.skipUnless(_HAS_SHAPELY, "shapely required")
    def test_validate_trajectory_detects_positions_outside_floor_polygon(self) -> None:
        floor = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        frames = [
            {
                "id": 0,
                "position": [0.5, 0.5, 1.6],
                "look_at": [1.5, 0.5, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            },
            {
                "id": 1,
                "position": [3.0, 3.0, 1.6],
                "look_at": [4.0, 3.0, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            },
        ]
        warnings = validate_trajectory(frames, floor_polygon=floor)
        combined = " ".join(warnings).lower()
        self.assertIn("outside the floor footprint", combined)

    def test_validate_trajectory_reports_floor_validation_error(self) -> None:
        class _BrokenFloor:
            def covers(self, _pt):
                raise RuntimeError("bad geometry")

        frames = [
            {
                "id": 0,
                "position": [0.5, 0.5, 1.6],
                "look_at": [1.5, 0.5, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            },
            {
                "id": 1,
                "position": [0.6, 0.6, 1.6],
                "look_at": [1.6, 0.6, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            },
        ]
        warnings = validate_trajectory(frames, floor_polygon=_BrokenFloor())
        self.assertTrue(any("floor-boundary validation failed" in w.lower() for w in warnings))
        self.assertTrue(any("2 time(s)" in w for w in warnings))

    def test_non_finite_frames_skip_other_vector_checks(self) -> None:
        frames = [
            {
                "id": 0,
                "position": [0.0, 0.0, 1.6],
                "look_at": [float("nan"), 0.0, 1.6],
                "forward": [2.0, 0.0, 0.0],
                "up": [1.0, 0.0, 0.0],
                "fov": 60.0,
            }
        ]
        warnings = validate_trajectory(frames)
        combined = " ".join(warnings).lower()
        self.assertIn("nan/inf", combined)
        self.assertNotIn("non-unit forward", combined)
        self.assertNotIn("not perpendicular", combined)

    def test_floor_validation_warns_when_shapely_unavailable(self) -> None:
        frames = [
            {
                "id": 0,
                "position": [0.0, 0.0, 1.6],
                "look_at": [1.0, 0.0, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            }
        ]
        with patch.object(validation_module, "ShapelyPoint", None):
            warnings = validate_trajectory(frames, floor_polygon=object())
        self.assertTrue(any("shapely is not installed" in w.lower() for w in warnings))


if __name__ == "__main__":
    unittest.main()
