from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.adapters.houselayout3d_matterport import (
    _assign_rooms_to_closest_floor,
    _resolve_path,
    _resolve_project_root,
)


class _Room:
    def __init__(self, room_id: str, z: float):
        self.room_id = room_id
        self.centroid = (0.0, 0.0, z)


class _Floor:
    def __init__(self, level_index: int, mean_height: float):
        self.level_index = level_index
        self.mean_height = mean_height


class AdapterHelpersTest(unittest.TestCase):
    def test_assign_rooms_to_closest_floor(self) -> None:
        rooms = [
            _Room("R0", 0.1),
            _Room("R1", 2.9),
            _Room("R_far", 10.0),
        ]
        floors = [_Floor(0, 0.0), _Floor(1, 3.0)]

        assignment = _assign_rooms_to_closest_floor(rooms, floors)

        self.assertEqual(len(assignment[0]), 1)
        self.assertEqual(len(assignment[1]), 1)
        self.assertEqual(assignment[0][0].room_id, "R0")
        self.assertEqual(assignment[1][0].room_id, "R1")

    def test_assign_rooms_to_closest_floor_respects_distance_threshold(self) -> None:
        rooms = [
            _Room("R_far", 10.0),
        ]
        floors = [_Floor(0, 0.0)]

        dropped = _assign_rooms_to_closest_floor(rooms, floors, max_room_floor_distance=4.0)
        kept = _assign_rooms_to_closest_floor(rooms, floors, max_room_floor_distance=12.0)

        self.assertEqual(len(dropped[0]), 0)
        self.assertEqual(len(kept[0]), 1)
        self.assertEqual(kept[0][0].room_id, "R_far")

    def test_resolve_path_handles_relative_and_absolute(self) -> None:
        root = Path("/tmp/root")
        rel = Path("data")
        abs_path = Path("/var/tmp/data")

        self.assertEqual(_resolve_path(root, rel), (root / "data").resolve())
        self.assertEqual(_resolve_path(root, abs_path), abs_path)

    def test_resolve_project_root_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "tools" / "floorplan").mkdir(parents=True)
            resolved = _resolve_project_root(root)
            self.assertEqual(resolved, root.resolve())

    def test_resolve_project_root_explicit_invalid_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with self.assertRaises(FileNotFoundError):
                _resolve_project_root(root)


if __name__ == "__main__":
    unittest.main()
