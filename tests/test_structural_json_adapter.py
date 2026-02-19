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
import trajectory_generation.adapters.structural_json as structural_json
from trajectory_generation.adapters.structural_json import run_structural_json


class StructuralJsonAdapterTest(unittest.TestCase):
    def _write_scene(self, root: Path, payload: dict) -> Path:
        scene_path = root / "scene.json"
        scene_path.write_text(json.dumps(payload, indent=2))
        return scene_path

    def test_valid_minimal_schema_with_explicit_connections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "bbox": {"min": [3, 0, 0], "max": [5, 2, 3]},
                        },
                    ],
                    "connections": [
                        {
                            "room1_id": "r1",
                            "room2_id": "r2",
                            "waypoint_xy": [2.5, 1.0],
                            "normal_xy": [1.0, 0.0],
                        }
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )

            artifacts = run_structural_json(cfg, project_root=root)

            self.assertEqual(artifacts.scene, "scene_struct")
            self.assertEqual(len(artifacts.floor_trajectories), 1)
            self.assertIn(0, artifacts.connectivity_statistics)
            summary = artifacts.to_dict()
            self.assertIn("component_transfers_by_floor", summary)
            self.assertFalse(Path(summary["floor_trajectories"][0]["output_file"]).is_absolute())

    def test_derived_connectivity_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "bbox": {"min": [2.1, 0, 0], "max": [4, 2, 3]},
                        },
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            cfg.connectivity.proximity_threshold = 0.3

            artifacts = run_structural_json(cfg, project_root=root)

            self.assertEqual(len(artifacts.floor_trajectories), 1)
            self.assertEqual(artifacts.connectivity_statistics[0]["num_connections"], 1)

    def test_explicit_connections_auto_classify_actual_from_door_openings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_doors",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "bbox": {"min": [3, 0, 0], "max": [5, 2, 3]},
                        },
                    ],
                    "connections": [
                        {
                            "room1_id": "r1",
                            "room2_id": "r2",
                            "waypoint_xy": [2.5, 1.0],
                        }
                    ],
                    "openings": [
                        {
                            "opening_type": "door",
                            "floor_index": 0,
                            "waypoint_xy": [2.55, 1.02],
                        }
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )

            artifacts = run_structural_json(cfg, project_root=root)
            self.assertEqual(artifacts.connectivity_statistics[0]["actual_doors"], 1)
            self.assertEqual(artifacts.connectivity_statistics[0]["synthetic_doors"], 0)

    def test_missing_required_keys_fails_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct",
                    "floors": [],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            with self.assertRaises(ValueError):
                run_structural_json(cfg, project_root=root)

    def test_floor_polygon_build_failure_is_added_to_artifact_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        }
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            with patch.object(structural_json, "ShapelyPolygon", side_effect=ValueError("bad floor polygon")):
                artifacts = run_structural_json(cfg, project_root=root)
            self.assertTrue(
                any("failed to build floor polygon" in warning.lower() for warning in artifacts.warnings)
            )

    def test_no_rooms_or_floors_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        }
                    ],
                    "rooms": [],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            with self.assertRaises(ValueError):
                run_structural_json(cfg, project_root=root)

    def test_all_components_restart_mode_emits_warning_without_transfer_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [100, 0], [100, 100], [0, 100]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "bbox": {"min": [80, 80, 0], "max": [82, 82, 3]},
                        },
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            cfg.connectivity.proximity_threshold = 0.1
            cfg.walkthrough.behavior.disconnected_transition_mode = "bridge"
            cfg.walkthrough.behavior.disconnected_component_policy = "all_components"
            cfg.walkthrough.behavior.__post_init__()

            artifacts = run_structural_json(cfg, project_root=root)
            summary = artifacts.to_dict()
            self.assertIn("component_transfers_by_floor", summary)
            self.assertEqual(summary["component_transfers_by_floor"], {})
            self.assertTrue(
                any("restarts per component (no cross-component links)" in warning for warning in artifacts.warnings)
            )

    def test_disconnected_default_skips_non_largest_component_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [100, 0], [100, 100], [0, 100]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "bbox": {"min": [4, 0, 0], "max": [6, 2, 3]},
                        },
                        {
                            "room_id": "r3",
                            "floor_index": 0,
                            "semantic": "bedroom",
                            "bbox": {"min": [80, 80, 0], "max": [82, 82, 3]},
                        },
                    ],
                    "connections": [
                        {"room1_id": "r1", "room2_id": "r2", "waypoint_xy": [3.0, 1.0]}
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )

            artifacts = run_structural_json(cfg, project_root=root)
            summary = artifacts.to_dict()
            self.assertEqual(summary["component_transfers_by_floor"], {})
            self.assertTrue(
                any("outside largest connected component" in warning for warning in artifacts.warnings)
            )

    def test_optional_polygon_room_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_poly",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "polygon_xy": [[0, 0], [2, 0], [2, 2], [0, 2]],
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "polygon_xy": [[3, 0], [5, 0], [5, 2], [3, 2]],
                            "bbox": {"min": [3, 0, 0], "max": [5, 2, 3]},
                        },
                    ],
                    "connections": [
                        {"room1_id": "r1", "room2_id": "r2", "waypoint_xy": [2.5, 1.0]}
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            artifacts = run_structural_json(cfg, project_root=root)
            self.assertEqual(len(artifacts.floor_trajectories), 1)

    def test_polygon_auto_connectivity_uses_polygon_adjacency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_poly_auto",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 4], [0, 4]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "polygon_xy": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "polygon_xy": [[2.15, 0.0], [4.0, 0.0], [4.0, 2.0], [2.15, 2.0]],
                            "bbox": {"min": [2.15, 0, 0], "max": [4.0, 2, 3]},
                        },
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            cfg.connectivity.proximity_threshold = 0.2

            artifacts = run_structural_json(cfg, project_root=root)
            self.assertEqual(artifacts.connectivity_statistics[0]["num_connections"], 1)

    def test_polygon_auto_connectivity_avoids_bbox_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_poly_false_positive",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 8], [0, 8]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "polygon_xy": [[0.0, 0.0], [1.2, 0.0], [1.2, 1.2], [0.0, 1.2]],
                            "bbox": {"min": [0.0, 0.0, 0.0], "max": [4.0, 4.0, 3.0]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "polygon_xy": [[6.6, 6.6], [7.8, 6.6], [7.8, 7.8], [6.6, 7.8]],
                            "bbox": {"min": [3.0, 3.0, 0.0], "max": [8.0, 8.0, 3.0]},
                        },
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            cfg.connectivity.proximity_threshold = 0.3
            cfg.walkthrough.behavior.disconnected_transition_mode = "bridge"
            cfg.walkthrough.behavior.disconnected_component_policy = "all_components"
            cfg.walkthrough.behavior.__post_init__()

            artifacts = run_structural_json(cfg, project_root=root)
            self.assertEqual(artifacts.connectivity_statistics[0]["num_connections"], 0)

    def test_invalid_polygon_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_poly",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "polygon_xy": [[0, 0], [1, 1], [2, 2]],  # collinear invalid polygon
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            with self.assertRaises(ValueError):
                run_structural_json(cfg, project_root=root)

    def test_zero_thickness_z_bbox_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_flat_z",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [6, 0], [6, 6], [0, 6]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 0]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "bbox": {"min": [3, 0, 0], "max": [5, 2, 0]},
                        },
                    ],
                    "connections": [{"room1_id": "r1", "room2_id": "r2"}],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            artifacts = run_structural_json(cfg, project_root=root)
            self.assertEqual(len(artifacts.floor_trajectories), 1)

    def test_zero_xy_extent_bbox_rejected_without_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_bad_xy",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [6, 0], [6, 6], [0, 6]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [0, 2, 3]},
                        },
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            with self.assertRaises(ValueError):
                run_structural_json(cfg, project_root=root)

    def test_polygon_requires_shapely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_poly_no_shapely",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [6, 0], [6, 6], [0, 6]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "polygon_xy": [[0, 0], [2, 0], [2, 2], [0, 2]],
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            with patch.object(structural_json, "ShapelyPolygon", None):
                with self.assertRaises(RuntimeError):
                    run_structural_json(cfg, project_root=root)

    def test_realistic_polygonal_layout_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_layout",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [12, 0], [12, 10], [0, 10]],
                        }
                    ],
                    "rooms": [
                        {
                            "room_id": "entry",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "polygon_xy": [[0, 0], [3, 0], [3, 2], [0, 2]],
                            "bbox": {"min": [0, 0, 0], "max": [3, 2, 3]},
                        },
                        {
                            "room_id": "hall",
                            "floor_index": 0,
                            "semantic": "hallway",
                            "polygon_xy": [[3, 0.5], [6, 0.5], [6, 1.5], [3, 1.5]],
                            "bbox": {"min": [3, 0.5, 0], "max": [6, 1.5, 3]},
                        },
                        {
                            "room_id": "living",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "polygon_xy": [[6, 0], [10, 0], [10, 5], [8, 5], [8, 3], [6, 3]],
                            "bbox": {"min": [6, 0, 0], "max": [10, 5, 3]},
                        },
                    ],
                    "connections": [
                        {"room1_id": "entry", "room2_id": "hall", "waypoint_xy": [3.0, 1.0]},
                        {"room1_id": "hall", "room2_id": "living", "waypoint_xy": [6.0, 1.0]},
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )
            cfg.walkthrough.fps = 20

            artifacts = run_structural_json(cfg, project_root=root)

            self.assertEqual(len(artifacts.floor_trajectories), 1)
            floor = artifacts.floor_trajectories[0]
            self.assertGreater(floor.num_frames, 20)
            self.assertEqual(artifacts.connectivity_statistics[0]["num_connections"], 2)

    def test_malformed_connections_are_safely_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene_json = self._write_scene(
                root,
                {
                    "scene": "scene_struct_malformed_connections",
                    "floors": [
                        {
                            "floor_index": 0,
                            "z": 0.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        },
                        {
                            "floor_index": 1,
                            "z": 3.0,
                            "footprint_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        },
                    ],
                    "rooms": [
                        {
                            "room_id": "r1",
                            "floor_index": 0,
                            "semantic": "entryway",
                            "bbox": {"min": [0, 0, 0], "max": [2, 2, 3]},
                        },
                        {
                            "room_id": "r2",
                            "floor_index": 0,
                            "semantic": "living_room",
                            "bbox": {"min": [3, 0, 0], "max": [5, 2, 3]},
                        },
                        {
                            "room_id": "r3",
                            "floor_index": 1,
                            "semantic": "bedroom",
                            "bbox": {"min": [0, 0, 3], "max": [2, 2, 6]},
                        },
                    ],
                    "connections": [
                        {"room1_id": "r1", "room2_id": "r2"},
                        {"room1_id": "r1", "room2_id": "r2"},
                        {"room1_id": "r1", "room2_id": "r1"},
                        {"room1_id": "r1", "room2_id": "r3"},
                        {"room1_id": "r1", "room2_id": "unknown_room"},
                    ],
                },
            )
            cfg = TrajectoryGenerationConfig.structural_json(
                scene_input_json=scene_json,
                output_dir=Path("outputs"),
                dataset_root=root,
            )

            artifacts = run_structural_json(cfg, project_root=root)

            self.assertEqual(artifacts.connectivity_statistics[0]["num_connections"], 1)
            self.assertTrue(
                any("Skipped cross-floor connection" in warning for warning in artifacts.warnings)
            )
            self.assertTrue(
                any("unknown room id" in warning for warning in artifacts.warnings)
            )


if __name__ == "__main__":
    unittest.main()
