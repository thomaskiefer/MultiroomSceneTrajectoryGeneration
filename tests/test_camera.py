from __future__ import annotations

from pathlib import Path
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from trajectory_generation.camera import CameraPathBuilder
from trajectory_generation.config import WalkthroughBehaviorConfig
from trajectory_generation.control_points import ControlPointSequence


class CameraPathBuilderTest(unittest.TestCase):
    def test_apply_heading_constraints_preserves_pitch_component(self) -> None:
        behavior = WalkthroughBehaviorConfig()
        builder = CameraPathBuilder(behavior)
        forwards = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.5656854],  # ~34.4 deg pitch
                [0.6, 0.4, 0.6928203],  # ~43.8 deg pitch
            ],
            dtype=float,
        )
        forwards = forwards / np.linalg.norm(forwards, axis=1, keepdims=True)
        out = builder.apply_heading_constraints(forwards, fps=30)
        self.assertEqual(out.shape, forwards.shape)
        self.assertTrue(np.allclose(out[:, 2], forwards[:, 2], atol=1e-6))
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6))

    def test_interpolate_path_rejects_non_finite_segment_times(self) -> None:
        behavior = WalkthroughBehaviorConfig()
        builder = CameraPathBuilder(behavior)
        seq = ControlPointSequence(
            positions=np.array([[0.0, 0.0, 1.6], [1.0, 0.0, 1.6]], dtype=float),
            look_targets=np.array([[1.0, 0.0, 1.6], [2.0, 0.0, 1.6]], dtype=float),
            segment_speeds=np.array([float("nan")], dtype=float),
        )
        with self.assertRaises(ValueError):
            builder.interpolate_path(seq, fps=30)


if __name__ == "__main__":
    unittest.main()
