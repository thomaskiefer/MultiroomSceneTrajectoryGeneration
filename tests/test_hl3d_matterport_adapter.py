from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import TrajectoryGenerationConfig
import trajectory_generation.adapters.houselayout3d_matterport as hl3d
from trajectory_generation.adapters.houselayout3d_matterport import ModulePorts


class _FakeFloor:
    def __init__(self):
        self.level_index = 0
        self.mean_height = 0.0
        self.footprint = "fake_polygon"


class _FakeRoom:
    def __init__(self, room_id: str, z: float):
        self.room_id = room_id
        self.centroid = (0.0, 0.0, z)
        self.label_semantic = "living_room"


class _FakeOpening:
    def __init__(self, opening_type: str):
        self.opening_type = opening_type


class _FakeFloorplanConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeWalkthrough3DGS:
    def __init__(self, graph, floor_z: float, camera_height: float):
        self.graph = graph
        self.floor_z = floor_z
        self.camera_height = camera_height

    def generate_exploration_path(self, fps: int):
        return [
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
                "position": [1.0, 0.0, 1.6],
                "look_at": [2.0, 0.0, 1.6],
                "forward": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 60.0,
            },
        ]


class _FakeGeometryModule:
    @staticmethod
    def load_stair_height_ranges(dataset_root: Path, scene: str):
        return []

    @staticmethod
    def load_surfaces(scene_dir: Path, stair_ranges, config):
        return ["surface"], ["wall"]

    @staticmethod
    def cluster_by_height(surfaces, tolerance: float):
        return [["surface"]]

    @staticmethod
    def build_floorplans(clusters, config):
        return [_FakeFloor()]


class _FakeRoomsModule:
    @staticmethod
    def parse_house_file(house_file: Path):
        return {"rooms": [_FakeRoom("R_1", 0.2), _FakeRoom("R_2", 0.4)]}

    @staticmethod
    def assign_rooms_priority_overlay(rooms, floor, config):
        return [(rooms[0], "poly", 12.0, "claimed"), (rooms[1], "poly", 9.0, "claimed")]

    @staticmethod
    def match_rooms_with_intersections(rooms, floor, config):
        return [(rooms[0], "poly", 12.0)]

    @staticmethod
    def split_hallways_at_doors(rooms_data, doors, split_length: float):
        return rooms_data


class _FakeOpeningsModule:
    @staticmethod
    def load_openings(path: Path, opening_type: str):
        if opening_type == "door":
            return [_FakeOpening("door")]
        return []

    @staticmethod
    def match_openings_to_floors(all_openings, floorplans, config, rooms_by_floor):
        return {0: all_openings}

    @staticmethod
    def match_walls_to_floors(walls, floorplans):
        return {0: walls}

    @staticmethod
    def match_openings_to_walls(openings_by_floor_raw, walls_by_floor, tolerance: float):
        floor_openings = openings_by_floor_raw.get(0, [])
        return {0: [(o, None) for o in floor_openings]}


class _FakeConnectivityModule:
    @staticmethod
    def build_connectivity_graphs(**kwargs):
        return {0: object()}

    @staticmethod
    def compute_graph_statistics(graph):
        return {
            "floor_level": 0,
            "num_rooms": 2,
            "num_connections": 1,
            "actual_doors": 1,
            "synthetic_doors": 0,
            "avg_degree": 1.0,
            "max_degree": 1,
            "isolated_rooms": [],
            "num_isolated": 0,
        }


class _FakeCompatibleConnectivityModule:
    @staticmethod
    def build_connectivity_graphs(**kwargs):
        room = type("Room", (), {"label_semantic": "entryway", "centroid": (0.0, 0.0, 0.0)})()
        poly = type(
            "Poly",
            (),
            {
                "area": 1.0,
                "representative_point": staticmethod(
                    lambda: type("Pt", (), {"x": 0.0, "y": 0.0})()
                ),
            },
        )()
        graph = type(
            "Graph",
            (),
            {"rooms": {"r1": (room, poly)}, "adjacency": {"r1": []}, "connections": []},
        )()
        return {0: graph}

    @staticmethod
    def compute_graph_statistics(graph):
        return _FakeConnectivityModule.compute_graph_statistics(graph)


class HouselayoutMatterportAdapterTest(unittest.TestCase):
    def _prepare_project_tree(self, root: Path, with_doors: bool = True) -> tuple[Path, str]:
        scene = "scene_abc"
        (root / "tools" / "floorplan").mkdir(parents=True)

        dataset_root = root / "dataset"
        (dataset_root / "structures" / "layouts_split_by_entity" / scene).mkdir(parents=True)
        (dataset_root / "house_segmentations" / scene).mkdir(parents=True)
        (dataset_root / "house_segmentations" / scene / f"{scene}.house").write_text("R")

        if with_doors:
            (dataset_root / "doors").mkdir(parents=True)
            (dataset_root / "doors" / f"{scene}.json").write_text("{}")

        return dataset_root, scene

    def test_runs_end_to_end_with_mocked_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root, scene = self._prepare_project_tree(root, with_doors=True)

            config = TrajectoryGenerationConfig.houselayout3d_matterport(
                dataset_root=dataset_root,
                scene=scene,
                output_dir=root / "outputs",
            )

            with patch.object(
                hl3d,
                "_load_floorplan_modules",
                return_value=ModulePorts(
                    connectivity=_FakeConnectivityModule,
                    geometry=_FakeGeometryModule,
                    openings=_FakeOpeningsModule,
                    rooms=_FakeRoomsModule,
                    floorplan_config_factory=_FakeFloorplanConfig,
                    walkthrough_factory=_FakeWalkthrough3DGS,
                ),
            ):
                artifacts = hl3d.run_houselayout3d_matterport(config, project_root=root)

            self.assertEqual(len(artifacts.floor_trajectories), 1)
            self.assertEqual(artifacts.floor_trajectories[0].num_frames, 2)
            self.assertIn(0, artifacts.connectivity_statistics)
            self.assertEqual(artifacts.connectivity_statistics[0]["num_rooms"], 2)

            trajectory_path = artifacts.floor_trajectories[0].output_file
            self.assertTrue(trajectory_path.exists())
            frames = json.loads(trajectory_path.read_text())
            self.assertEqual(len(frames), 2)

            summary_path = (root / "outputs" / f"{scene}_trajectory_generation_summary.json")
            self.assertTrue(summary_path.exists())

    def test_warns_when_no_openings_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root, scene = self._prepare_project_tree(root, with_doors=False)

            config = TrajectoryGenerationConfig.houselayout3d_matterport(
                dataset_root=dataset_root,
                scene=scene,
                output_dir=root / "outputs",
            )

            with patch.object(
                hl3d,
                "_load_floorplan_modules",
                return_value=ModulePorts(
                    connectivity=_FakeConnectivityModule,
                    geometry=_FakeGeometryModule,
                    openings=_FakeOpeningsModule,
                    rooms=_FakeRoomsModule,
                    floorplan_config_factory=_FakeFloorplanConfig,
                    walkthrough_factory=_FakeWalkthrough3DGS,
                ),
            ):
                artifacts = hl3d.run_houselayout3d_matterport(config, project_root=root)

            self.assertTrue(
                any("No openings found" in warning for warning in artifacts.warnings)
            )

    def test_write_trajectory_frames_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stem = Path(tmp_dir) / "nested" / "dir" / "trajectory"
            frames = [{"id": 0, "position": [0, 0, 0]}]
            out = hl3d.write_trajectory_frames(frames, stem)
            self.assertEqual(out.suffix, ".json")
            self.assertTrue(out.exists())
            self.assertEqual(json.loads(out.read_text()), frames)
            self.assertEqual(out.parent, stem.parent)

    def test_local_walkthrough_rejects_incompatible_graph_when_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root, scene = self._prepare_project_tree(root, with_doors=True)

            config = TrajectoryGenerationConfig.houselayout3d_matterport(
                dataset_root=dataset_root,
                scene=scene,
                output_dir=root / "outputs",
            )
            config.walkthrough.use_local_walkthrough = True

            with patch.object(
                hl3d,
                "_load_floorplan_modules",
                return_value=ModulePorts(
                    connectivity=_FakeConnectivityModule,
                    geometry=_FakeGeometryModule,
                    openings=_FakeOpeningsModule,
                    rooms=_FakeRoomsModule,
                    floorplan_config_factory=_FakeFloorplanConfig,
                    walkthrough_factory=_FakeWalkthrough3DGS,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    hl3d.run_houselayout3d_matterport(config, project_root=root)

    def test_local_walkthrough_uses_local_when_graph_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root, scene = self._prepare_project_tree(root, with_doors=True)

            config = TrajectoryGenerationConfig.houselayout3d_matterport(
                dataset_root=dataset_root,
                scene=scene,
                output_dir=root / "outputs",
            )
            config.walkthrough.use_local_walkthrough = True

            with patch.object(
                hl3d,
                "_load_floorplan_modules",
                return_value=ModulePorts(
                    connectivity=_FakeCompatibleConnectivityModule,
                    geometry=_FakeGeometryModule,
                    openings=_FakeOpeningsModule,
                    rooms=_FakeRoomsModule,
                    floorplan_config_factory=_FakeFloorplanConfig,
                    walkthrough_factory=_FakeWalkthrough3DGS,
                ),
            ):
                artifacts = hl3d.run_houselayout3d_matterport(config, project_root=root)

            self.assertEqual(len(artifacts.floor_trajectories), 1)
            self.assertGreater(artifacts.floor_trajectories[0].num_frames, 1)

    def test_load_floorplan_modules_removes_injected_sys_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            floorplan_dir = root / "tools" / "floorplan"
            floorplan_dir.mkdir(parents=True)
            (floorplan_dir / "__init__.py").write_text("")
            (floorplan_dir / "connectivity.py").write_text("MARK = 'connectivity'\n")
            (floorplan_dir / "geometry.py").write_text("MARK = 'geometry'\n")
            (floorplan_dir / "openings.py").write_text("MARK = 'openings'\n")
            (floorplan_dir / "rooms.py").write_text("MARK = 'rooms'\n")
            (floorplan_dir / "models.py").write_text("class FloorplanConfig:\n    pass\n")
            (floorplan_dir / "walkthrough.py").write_text("class Walkthrough3DGS:\n    pass\n")

            tools_dir = str(root / "tools")
            existing_modules = {
                k: v
                for k, v in sys.modules.items()
                if k == "floorplan" or k.startswith("floorplan.")
            }
            for key in list(existing_modules):
                del sys.modules[key]

            try:
                self.assertNotIn(tools_dir, sys.path)
                ports = hl3d._load_floorplan_modules(root)
                self.assertTrue(hasattr(ports, "geometry"))
                self.assertNotIn(tools_dir, sys.path)
            finally:
                for key in list(sys.modules):
                    if key == "floorplan" or key.startswith("floorplan."):
                        del sys.modules[key]
                sys.modules.update(existing_modules)


if __name__ == "__main__":
    unittest.main()
