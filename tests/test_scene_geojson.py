from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import TrajectoryGenerationConfig
from trajectory_generation.scene_geojson import build_connectivity_geojson_from_structural_scene


class SceneGeoJsonTest(unittest.TestCase):
    def test_builds_geojson_with_trajectory_centers(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scene_path = repo_root / "examples" / "structural" / "demo_apartment.json"
        self.assertTrue(scene_path.exists(), f"Missing scene fixture: {scene_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "connectivity.geojson"
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_path,
                output_dir=Path(tmp_dir) / "out",
                dataset_root=repo_root,
            )
            written = build_connectivity_geojson_from_structural_scene(
                scene_input=scene_path,
                geojson_output=output_path,
                config=cfg,
            )
            self.assertEqual(written, output_path)
            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text())
            features = payload.get("features", [])
            self.assertGreater(len(features), 0)

            type_counts: dict[str, int] = {}
            for feature in features:
                ftype = feature.get("properties", {}).get("type", "")
                type_counts[ftype] = type_counts.get(ftype, 0) + 1

            self.assertGreater(type_counts.get("floor_footprint", 0), 0)
            self.assertGreater(type_counts.get("room", 0), 0)
            self.assertGreater(type_counts.get("trajectory_room_center", 0), 0)
            self.assertGreater(type_counts.get("room_connection", 0), 0)
            self.assertGreater(type_counts.get("door_waypoint", 0), 0)

            room_features = [
                f for f in features if f.get("properties", {}).get("type") == "room"
            ]
            self.assertGreater(len(room_features), 0)
            for room in room_features:
                props = room.get("properties", {})
                self.assertIn("trajectory_center_xy", props)
                self.assertIn("trajectory_center_3d", props)

    def test_builds_geojson_with_openings_and_stairs(self) -> None:
        payload = {
            "schema_version": "scene.schema.v1",
            "scene": "demo_with_openings_stairs",
            "floors": [
                {
                    "floor_index": 0,
                    "z": 0.0,
                    "footprint_xy": [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]],
                }
            ],
            "rooms": [
                {
                    "room_id": "R_A",
                    "floor_index": 0,
                    "semantic": "living_room",
                    "bbox": {"min": [0.5, 0.5, 0.0], "max": [2.5, 3.5, 2.6]},
                },
                {
                    "room_id": "R_B",
                    "floor_index": 0,
                    "semantic": "kitchen",
                    "bbox": {"min": [3.2, 0.5, 0.0], "max": [5.5, 3.0, 2.6]},
                },
            ],
            "connections": [
                {
                    "room1_id": "R_A",
                    "room2_id": "R_B",
                    "waypoint_xy": [2.9, 1.6],
                    "normal_xy": [1.0, 0.0],
                    "door_type": "actual",
                }
            ],
            "openings": [
                {
                    "opening_type": "door",
                    "opening_id": 10,
                    "floor_index": 0,
                    "waypoint_xy": [2.9, 1.6],
                    "normal_xy": [1.0, 0.0],
                    "bbox": {"min": [2.8, 1.5, 0.0], "max": [3.0, 1.7, 2.1]},
                },
                {
                    "opening_type": "window",
                    "opening_id": 11,
                    "floor_index": 0,
                    "segment_xy": [[0.9, 3.9], [1.9, 3.9]],
                    "bbox": {"min": [1.0, 3.8, 1.0], "max": [1.8, 4.0, 1.8]},
                },
            ],
            "stairs": [
                {
                    "stair_id": 1,
                    "z_min": 0.0,
                    "z_max": 2.8,
                    "from_floor_index": 0,
                    "to_floor_index": 1,
                    "bbox": {"min": [5.2, 2.2, 0.0], "max": [5.8, 3.2, 2.8]},
                },
                {
                    "stair_id": 2,
                    "z_min": 3.0,
                    "z_max": 5.8,
                    "from_floor_index": 1,
                    "to_floor_index": 2,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene.json"
            scene_path.write_text(json.dumps(payload, indent=2))
            output_path = Path(tmp_dir) / "connectivity.geojson"
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_path,
                output_dir=Path(tmp_dir) / "out",
                dataset_root=Path(tmp_dir),
            )

            build_connectivity_geojson_from_structural_scene(
                scene_input=scene_path,
                geojson_output=output_path,
                config=cfg,
            )
            written = json.loads(output_path.read_text())
            features = written.get("features", [])

            types = [f.get("properties", {}).get("type") for f in features]
            self.assertIn("door", types)
            self.assertIn("window", types)
            self.assertIn("stairs", types)
            conn_features = [f for f in features if f.get("properties", {}).get("type") == "room_connection"]
            self.assertEqual(len(conn_features), 1)
            self.assertEqual(conn_features[0]["properties"].get("door_type"), "actual")

            window_features = [f for f in features if f.get("properties", {}).get("type") == "window"]
            self.assertEqual(len(window_features), 1)
            self.assertEqual(
                window_features[0]["geometry"]["coordinates"],
                [[0.9, 3.9], [1.9, 3.9]],
            )

            metadata = written.get("metadata", {})
            self.assertIn("stairs", metadata)
            self.assertEqual(len(metadata["stairs"]), 2)

    def test_infers_actual_connection_from_nearby_door_opening(self) -> None:
        payload = {
            "schema_version": "scene.schema.v1",
            "scene": "door_infer_scene",
            "floors": [
                {"floor_index": 0, "z": 0.0, "footprint_xy": [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]}
            ],
            "rooms": [
                {"room_id": "R_1", "floor_index": 0, "semantic": "entryway", "bbox": {"min": [0.5, 0.5, 0.0], "max": [2.5, 3.5, 2.5]}},
                {"room_id": "R_2", "floor_index": 0, "semantic": "living_room", "bbox": {"min": [3.0, 0.5, 0.0], "max": [5.5, 3.5, 2.5]}},
            ],
            "connections": [
                {"room1_id": "R_1", "room2_id": "R_2", "waypoint_xy": [2.75, 2.0]}
            ],
            "openings": [
                {"opening_type": "door", "floor_index": 0, "waypoint_xy": [2.8, 2.02]}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_path = Path(tmp_dir) / "scene.json"
            scene_path.write_text(json.dumps(payload, indent=2))
            output_path = Path(tmp_dir) / "connectivity.geojson"
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_path,
                output_dir=Path(tmp_dir) / "out",
                dataset_root=Path(tmp_dir),
            )
            build_connectivity_geojson_from_structural_scene(
                scene_input=scene_path,
                geojson_output=output_path,
                config=cfg,
            )
            written = json.loads(output_path.read_text())
            conn_features = [f for f in written.get("features", []) if f.get("properties", {}).get("type") == "room_connection"]
            self.assertEqual(len(conn_features), 1)
            self.assertEqual(conn_features[0]["properties"].get("door_type"), "actual")


if __name__ == "__main__":
    unittest.main()
