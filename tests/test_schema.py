from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.schema import (
    STRUCTURAL_SCHEMA_VERSION,
    parse_structural_scene_file,
    validate_structural_scene_file,
)


class StructuralSchemaTest(unittest.TestCase):
    def _write_scene(self, root: Path, payload: dict) -> Path:
        scene_path = root / "scene.json"
        scene_path.write_text(json.dumps(payload, indent=2))
        return scene_path

    def test_parse_accepts_explicit_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        }
                    ],
                },
            )
            parsed = parse_structural_scene_file(scene_json)
            self.assertEqual(parsed.schema_version, STRUCTURAL_SCHEMA_VERSION)
            self.assertEqual(parsed.scene, "scene_x")
            self.assertEqual(len(parsed.warnings), 0)

    def test_parse_missing_schema_version_warns_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        }
                    ],
                },
            )
            parsed = parse_structural_scene_file(scene_json)
            self.assertEqual(parsed.schema_version, STRUCTURAL_SCHEMA_VERSION)
            self.assertGreaterEqual(len(parsed.warnings), 1)

    def test_invalid_schema_version_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": "scene.schema.v0",
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        }
                    ],
                },
            )
            with self.assertRaises(ValueError):
                parse_structural_scene_file(scene_json)

    def test_validate_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        }
                    ],
                    "connections": [],
                },
            )
            report = validate_structural_scene_file(scene_json)
            self.assertEqual(report["schema_version"], STRUCTURAL_SCHEMA_VERSION)
            self.assertEqual(report["scene"], "scene_x")
            self.assertEqual(report["num_floors"], 1)
            self.assertEqual(report["num_rooms"], 1)
            self.assertEqual(report["num_connections"], 0)

    def test_connection_references_unknown_or_self_are_ignored_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        },
                        {
                            "room_id": "r2",
                            "semantic": "living_room",
                            "bbox": {"min": [1, 0, 0], "max": [2, 1, 2]},
                        },
                    ],
                    "connections": [
                        {"room1_id": "r1", "room2_id": "r1"},
                        {"room1_id": "r1", "room2_id": "missing"},
                        {"room1_id": "r1", "room2_id": "r2"},
                    ],
                },
            )
            parsed = parse_structural_scene_file(scene_json)
            self.assertEqual(len(parsed.connections), 1)
            self.assertGreaterEqual(len(parsed.warnings), 2)
            warning_text = " ".join(parsed.warnings)
            self.assertIn("self-connection", warning_text)
            self.assertIn("unknown room ids", warning_text)

    def test_duplicate_connections_are_ignored_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        },
                        {
                            "room_id": "r2",
                            "semantic": "living_room",
                            "bbox": {"min": [1, 0, 0], "max": [2, 1, 2]},
                        },
                    ],
                    "connections": [
                        {"room1_id": "r1", "room2_id": "r2"},
                        {"room1_id": "r2", "room2_id": "r1"},
                    ],
                },
            )
            parsed = parse_structural_scene_file(scene_json)
            self.assertEqual(len(parsed.connections), 1)
            self.assertTrue(any("duplicate" in w.lower() for w in parsed.warnings))

    def test_floor_index_requires_integer_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0.5, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        }
                    ],
                },
            )
            with self.assertRaises(ValueError):
                parse_structural_scene_file(scene_json)

    def test_collinear_polygon_xy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                            "polygon_xy": [[0, 0], [1, 0], [2, 0]],
                        }
                    ],
                },
            )
            with self.assertRaises(ValueError):
                parse_structural_scene_file(scene_json)

    def test_non_finite_floor_z_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": float("nan"), "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        }
                    ],
                },
            )
            with self.assertRaises(ValueError):
                parse_structural_scene_file(scene_json)

    def test_non_finite_bbox_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, float("inf"), 2]},
                        }
                    ],
                },
            )
            with self.assertRaises(ValueError):
                parse_structural_scene_file(scene_json)

    def test_empty_room_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [2, 0], [2, 2]]}],
                    "rooms": [
                        {
                            "room_id": "   ",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                        }
                    ],
                },
            )
            with self.assertRaises(ValueError):
                parse_structural_scene_file(scene_json)

    def test_polygon_room_with_zero_xy_bbox_extent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "schema_version": STRUCTURAL_SCHEMA_VERSION,
                    "scene": "scene_x",
                    "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [3, 0], [3, 3]]}],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [0, 2, 2]},
                            "polygon_xy": [[0, 0], [2, 0], [2, 2], [0, 2]],
                        }
                    ],
                },
            )
            with self.assertRaises(ValueError):
                parse_structural_scene_file(scene_json)


if __name__ == "__main__":
    unittest.main()
