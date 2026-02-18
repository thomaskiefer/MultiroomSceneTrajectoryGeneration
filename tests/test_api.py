from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch
import sys

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.api import (  # noqa: E402
    generate_frames_from_graph,
    generate_from_config,
    generate_from_structural_json,
)
from trajectory_generation.config import TrajectoryGenerationConfig  # noqa: E402
from trajectory_generation.room_graph import (  # noqa: E402
    RoomConnection,
    RoomGraph,
    RoomGraphRoomNode,
    RoomGraphWaypoint,
)


class _Poly:
    def __init__(self, area: float = 1.0):
        self.area = area

    def representative_point(self):
        class _Point:
            x = 0.0
            y = 0.0

        return _Point()


class ApiTest(unittest.TestCase):
    def test_generate_from_config_delegates_to_pipeline(self) -> None:
        cfg = TrajectoryGenerationConfig.houselayout3d_matterport(
            dataset_root=Path("/tmp/dataset"),
            scene="scene_1",
        )
        with patch("trajectory_generation.api.run", return_value="ok") as mock_run:
            out = generate_from_config(cfg, project_root=Path("/tmp/root"))
        self.assertEqual(out, "ok")
        mock_run.assert_called_once()

    def test_generate_from_structural_json_builds_config_and_runs(self) -> None:
        with patch("trajectory_generation.api.run", return_value="ok_struct") as mock_run:
            out = generate_from_structural_json(
                scene_input_json=Path("/tmp/scene.json"),
                output_dir=Path("outputs"),
            )
        self.assertEqual(out, "ok_struct")
        mock_run.assert_called_once()
        cfg = mock_run.call_args.kwargs["config"]
        self.assertEqual(cfg.workflow, "structural_json")
        self.assertEqual(cfg.dataset.scene_input_json, Path("/tmp/scene.json"))

    def test_generate_frames_from_graph_returns_frames(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly(4.0)),
            "b": (RoomGraphRoomNode("b", "living_room", np.array([2.0, 0.0, 0.0])), _Poly(4.0)),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": ["b"], "b": ["a"]},
            connections=[
                RoomConnection(
                    room1_id="a",
                    room2_id="b",
                    waypoint=RoomGraphWaypoint(position=np.array([1.0, 0.0]), normal=np.array([1.0, 0.0])),
                )
            ],
        )
        frames = generate_frames_from_graph(graph=graph, floor_z=0.0, fps=10)
        self.assertGreater(len(frames), 1)
        self.assertEqual(
            set(frames[0].keys()),
            {"id", "position", "look_at", "forward", "up", "fov"},
        )


if __name__ == "__main__":
    unittest.main()
