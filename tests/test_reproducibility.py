from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.adapters.structural_json import run_structural_json
from trajectory_generation.config import TrajectoryGenerationConfig


def _total_distance(frames: list[dict]) -> float:
    if len(frames) < 2:
        return 0.0
    positions = np.asarray([f["position"] for f in frames], dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


class ReproducibilityTest(unittest.TestCase):
    @staticmethod
    def _example_scene_path(repo_root: Path) -> Path:
        structural_dir = repo_root / "examples" / "structural"
        path = structural_dir / "demo_apartment.json"
        if not path.exists():
            raise FileNotFoundError(
                "Missing demo structural scene: examples/structural/demo_apartment.json"
            )
        return path

    def test_demo_apartment_generation_is_deterministic(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scene_path = self._example_scene_path(repo_root)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg1 = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_path,
                output_dir=tmp_path / "out_a",
                dataset_root=repo_root,
            )
            cfg1.walkthrough.behavior.travel_speed = 0.8
            cfg1.walkthrough.behavior.spin_segment_speed = 0.8
            cfg1.walkthrough.behavior.passthrough_speed = 0.8
            cfg1.walkthrough.behavior.disconnected_transition_mode = "bridge"

            art1 = run_structural_json(cfg1, project_root=repo_root)
            self.assertEqual(len(art1.floor_trajectories), 1)
            frames1 = json.loads(art1.floor_trajectories[0].output_file.read_text())

            cfg2 = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_path,
                output_dir=tmp_path / "out_b",
                dataset_root=repo_root,
            )
            cfg2.walkthrough.behavior.travel_speed = 0.8
            cfg2.walkthrough.behavior.spin_segment_speed = 0.8
            cfg2.walkthrough.behavior.passthrough_speed = 0.8
            cfg2.walkthrough.behavior.disconnected_transition_mode = "bridge"

            art2 = run_structural_json(cfg2, project_root=repo_root)
            self.assertEqual(len(art2.floor_trajectories), 1)
            frames2 = json.loads(art2.floor_trajectories[0].output_file.read_text())

            self.assertEqual(len(frames1), len(frames2))
            self.assertGreater(len(frames1), 100)
            self.assertLess(len(frames1), 5000)

            dist1 = _total_distance(frames1)
            dist2 = _total_distance(frames2)
            self.assertAlmostEqual(dist1, dist2, places=6)

            p_start_1 = np.asarray(frames1[0]["position"], dtype=float)
            p_start_2 = np.asarray(frames2[0]["position"], dtype=float)
            p_end_1 = np.asarray(frames1[-1]["position"], dtype=float)
            p_end_2 = np.asarray(frames2[-1]["position"], dtype=float)
            self.assertTrue(np.allclose(p_start_1, p_start_2, atol=1e-9))
            self.assertTrue(np.allclose(p_end_1, p_end_2, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
