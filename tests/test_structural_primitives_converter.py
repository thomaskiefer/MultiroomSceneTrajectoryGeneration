from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.preprocess import (  # noqa: E402
    convert_structural_primitives_file,
    convert_structural_primitives_payload,
)
from trajectory_generation.schema import parse_structural_scene_file  # noqa: E402


def _base_payload() -> dict:
    return {
        "scene": "demo_scene",
        "floors": [
            {
                "floor_index": 0,
                "z": 0.0,
                "footprint_xy": [[0.0, 0.0], [8.0, 0.0], [8.0, 4.0], [0.0, 4.0]],
            }
        ],
        "rooms": [
            {
                "room_id": "entry",
                "semantic": "entryway",
                "bbox": {"min": [0.0, 0.0, 0.0], "max": [3.0, 4.0, 2.8]},
            },
            {
                "room_id": "living",
                "semantic": "living_room",
                "bbox": {"min": [3.0, 0.0, 0.0], "max": [8.0, 4.0, 2.8]},
            },
        ],
    }


class StructuralPrimitivesConverterTest(unittest.TestCase):
    def test_prefers_explicit_connections(self) -> None:
        payload = _base_payload()
        payload["connections"] = [
            {
                "room1_id": "entry",
                "room2_id": "living",
                "waypoint_xy": [3.0, 2.0],
                "normal_xy": [1.0, 0.0],
            }
        ]
        payload["openings"] = [
            {"waypoint_xy": [3.0, 1.5], "normal_xy": [0.0, 1.0]},
        ]

        scene = convert_structural_primitives_payload(payload)
        self.assertEqual(scene["schema_version"], "scene.schema.v1")
        self.assertEqual(len(scene["connections"]), 1)
        self.assertEqual(scene["connections"][0]["waypoint_xy"], [3.0, 2.0])
        self.assertEqual(scene["connections"][0]["normal_xy"], [1.0, 0.0])

    def test_derives_connections_from_openings_when_explicit_missing(self) -> None:
        payload = _base_payload()
        payload["openings"] = [
            {
                "bbox": {"min": [2.9, 1.5, 0.0], "max": [3.1, 2.5, 2.1]},
                "normal_xy": [1.0, 0.0],
            }
        ]

        scene = convert_structural_primitives_payload(payload)
        self.assertEqual(len(scene["connections"]), 1)
        conn = scene["connections"][0]
        self.assertEqual({conn["room1_id"], conn["room2_id"]}, {"entry", "living"})
        self.assertIn("waypoint_xy", conn)
        self.assertIn("normal_xy", conn)

    def test_falls_back_to_bbox_proximity_without_openings(self) -> None:
        payload = _base_payload()
        scene = convert_structural_primitives_payload(payload, proximity_threshold=0.05)
        self.assertEqual(len(scene["connections"]), 1)
        self.assertEqual({scene["connections"][0]["room1_id"], scene["connections"][0]["room2_id"]}, {"entry", "living"})

    def test_assigns_floor_by_nearest_z_when_missing_floor_index(self) -> None:
        payload = {
            "scene": "multi_floor_demo",
            "floors": [
                {
                    "floor_index": 0,
                    "z": 0.0,
                    "footprint_xy": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                },
                {
                    "floor_index": 1,
                    "z": 3.0,
                    "footprint_xy": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                },
            ],
            "rooms": [
                {
                    "room_id": "r0",
                    "semantic": "other",
                    "bbox": {"min": [0.0, 0.0, 0.0], "max": [2.0, 2.0, 2.8]},
                },
                {
                    "room_id": "r1",
                    "semantic": "other",
                    "bbox": {"min": [0.0, 0.0, 3.1], "max": [2.0, 2.0, 5.8]},
                },
            ],
        }

        scene = convert_structural_primitives_payload(payload)
        room_floor = {room["room_id"]: room["floor_index"] for room in scene["rooms"]}
        self.assertEqual(room_floor["r0"], 0)
        self.assertEqual(room_floor["r1"], 1)

    def test_file_conversion_writes_schema_valid_scene(self) -> None:
        payload = _base_payload()
        payload["openings"] = [
            {
                "bbox": {"min": [2.9, 1.5, 0.0], "max": [3.1, 2.5, 2.1]},
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "in.json"
            out = tmpdir / "out.json"
            inp.write_text(json.dumps(payload, indent=2))

            written = convert_structural_primitives_file(inp, out)
            self.assertEqual(written, out)
            self.assertTrue(out.exists())

            parsed = parse_structural_scene_file(out)
            self.assertEqual(parsed.scene, "demo_scene")
            self.assertGreaterEqual(len(parsed.connections), 1)

    def test_rejects_non_finite_numeric_vectors(self) -> None:
        payload = _base_payload()
        payload["rooms"][0]["bbox"]["min"] = [0.0, float("nan"), 0.0]
        with self.assertRaises(ValueError):
            convert_structural_primitives_payload(payload)


if __name__ == "__main__":
    unittest.main()
