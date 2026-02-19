from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import (
    DatasetConfig,
    TrajectoryGenerationConfig,
    WalkthroughBehaviorConfig,
    WalkthroughConfig,
)


class TrajectoryGenerationConfigTest(unittest.TestCase):
    def test_houselayout3d_matterport_preset(self) -> None:
        cfg = TrajectoryGenerationConfig.houselayout3d_matterport(
            dataset_root=Path("/tmp/dataset"),
            scene="scene_123",
            output_dir=Path("outputs/tg"),
            house_segmentation_dir=Path("/tmp/house"),
        )

        self.assertEqual(cfg.dataset.scene, "scene_123")
        self.assertEqual(cfg.dataset.dataset_root, Path("/tmp/dataset"))
        self.assertEqual(cfg.dataset.house_segmentation_dir, Path("/tmp/house"))
        self.assertTrue(cfg.floorplan.include_openings)
        self.assertEqual(cfg.floorplan.max_room_floor_distance, 4.0)
        self.assertEqual(cfg.walkthrough.output_dir, Path("outputs/tg"))
        self.assertFalse(cfg.walkthrough.use_local_walkthrough)
        self.assertEqual(cfg.walkthrough.behavior.spin_points, 12)
        self.assertEqual(cfg.walkthrough.behavior.spin_orbit_scale, 0.10)
        self.assertEqual(cfg.walkthrough.behavior.max_linear_speed, 1.2)
        self.assertEqual(cfg.walkthrough.behavior.max_angular_speed_deg, 360.0)
        self.assertEqual(cfg.walkthrough.behavior.look_at_mode, "tangent")
        self.assertEqual(cfg.walkthrough.behavior.disconnected_transition_mode, "bridge")
        self.assertEqual(cfg.walkthrough.behavior.neighbor_priority_mode, "human_like")
        self.assertEqual(cfg.walkthrough.behavior.revisit_transition_mode, "center_arc")
        self.assertEqual(cfg.walkthrough.behavior.loop_closure_mode, "auto")
        self.assertTrue(cfg.walkthrough.behavior.prefer_outer_revisit_arc)
        self.assertEqual(cfg.walkthrough.behavior.revisit_arc_angle_search_deg, 30.0)
        self.assertEqual(cfg.walkthrough.behavior.revisit_arc_search_steps, 7)
        self.assertEqual(cfg.walkthrough.behavior.revisit_arc_max_span_deg, 360.0)
        self.assertEqual(cfg.walkthrough.behavior.revisit_arc_reverse_pref_deg, 155.0)
        self.assertEqual(cfg.walkthrough.behavior.revisit_arc_reverse_long_arc_bonus, 0.35)
        self.assertEqual(cfg.walkthrough.behavior.revisit_arc_transition_risk_distance_weight, 0.35)
        self.assertEqual(cfg.walkthrough.behavior.revisit_arc_transition_risk_angle_weight, 0.50)
        self.assertEqual(
            cfg.walkthrough.behavior.disconnected_component_policy,
            "largest_component_only",
        )
        self.assertEqual(cfg.workflow, "houselayout3d_matterport")

    def test_dataset_default_house_segmentation_dir(self) -> None:
        dataset = DatasetConfig(dataset_root=Path("/root/data"), scene="s")
        self.assertEqual(
            dataset.resolved_house_segmentation_dir,
            Path("/root/data/house_segmentations"),
        )
        self.assertIsNone(dataset.scene_input_json)

    def test_json_round_trip(self) -> None:
        cfg = TrajectoryGenerationConfig.houselayout3d_matterport(
            dataset_root=Path("/tmp/dataset"),
            scene="scene_abc",
            output_dir=Path("outputs/trajectory_generation"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            cfg.to_json(config_path)

            loaded = TrajectoryGenerationConfig.from_json(config_path)

        self.assertEqual(loaded.dataset.dataset_root, Path("/tmp/dataset"))
        self.assertEqual(loaded.dataset.scene, "scene_abc")
        self.assertEqual(loaded.walkthrough.output_dir, Path("outputs/trajectory_generation"))
        self.assertEqual(loaded.walkthrough.behavior.travel_speed, 0.8)
        self.assertEqual(loaded.workflow, "houselayout3d_matterport")

        as_dict = loaded.to_dict()
        self.assertIsInstance(as_dict["dataset"]["dataset_root"], str)
        self.assertIsInstance(as_dict["walkthrough"]["output_dir"], str)

    def test_from_dict_walkthrough_uses_dataclass_mapping(self) -> None:
        payload = {
            "dataset": {
                "dataset_root": "/tmp/dataset",
                "scene": "scene_x",
                "house_segmentation_dir": "/tmp/house",
                "scene_input_json": "/tmp/scene.json",
            },
            "walkthrough": {
                "fps": 24,
                "camera_height": 1.8,
                "output_dir": "outputs/custom",
                "write_debug_summary": False,
                "use_local_walkthrough": True,
                "behavior": {
                    "spin_points": 12,
                    "travel_speed": 0.9,
                    "look_at_mode": "spline_target",
                    "disconnected_transition_mode": "jump",
                },
            },
            "floorplan": {
                "max_room_floor_distance": 5.5,
            },
            "workflow": "structural_json",
        }

        loaded = TrajectoryGenerationConfig.from_dict(payload)

        self.assertEqual(loaded.walkthrough.fps, 24)
        self.assertEqual(loaded.walkthrough.camera_height, 1.8)
        self.assertEqual(loaded.walkthrough.output_dir, Path("outputs/custom"))
        self.assertFalse(loaded.walkthrough.write_debug_summary)
        self.assertTrue(loaded.walkthrough.use_local_walkthrough)
        self.assertEqual(loaded.walkthrough.behavior.spin_points, 12)
        self.assertEqual(loaded.walkthrough.behavior.travel_speed, 0.9)
        self.assertEqual(loaded.walkthrough.behavior.look_at_mode, "spline_target")
        self.assertEqual(loaded.walkthrough.behavior.disconnected_transition_mode, "jump")
        self.assertEqual(loaded.floorplan.max_room_floor_distance, 5.5)
        self.assertEqual(loaded.workflow, "structural_json")
        self.assertEqual(loaded.dataset.scene_input_json, Path("/tmp/scene.json"))

    def test_structural_json_preset(self) -> None:
        cfg = TrajectoryGenerationConfig.structural_json(
            scene_input_json=Path("/tmp/scene.json"),
            output_dir=Path("outputs/tg"),
            dataset_root=Path("/tmp"),
        )
        self.assertEqual(cfg.workflow, "structural_json")
        self.assertEqual(cfg.dataset.scene_input_json, Path("/tmp/scene.json"))
        self.assertEqual(cfg.dataset.dataset_root, Path("/tmp"))

    def test_rejects_invalid_workflow(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryGenerationConfig.from_dict(
                {
                    "workflow": "unknown",
                    "dataset": {"dataset_root": "/tmp/data", "scene": "s"},
                }
            )

    def test_rejects_invalid_disconnected_transition_mode(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryGenerationConfig.from_dict(
                {
                    "dataset": {"dataset_root": "/tmp/data", "scene": "s"},
                    "walkthrough": {
                        "behavior": {"disconnected_transition_mode": "teleport"},
                    },
                }
            )

    def test_rejects_invalid_disconnected_component_policy(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryGenerationConfig.from_dict(
                {
                    "dataset": {"dataset_root": "/tmp/data", "scene": "s"},
                    "walkthrough": {
                        "behavior": {"disconnected_component_policy": "teleport_all"},
                    },
                }
            )

    def test_rejects_invalid_neighbor_priority_mode(self) -> None:
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(neighbor_priority_mode="random_walk")

    def test_rejects_invalid_revisit_transition_mode(self) -> None:
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_transition_mode="teleport_arc")

    def test_rejects_invalid_loop_closure_mode(self) -> None:
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(loop_closure_mode="force")

    def test_rejects_missing_dataset_in_from_dict(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required top-level field `dataset`"):
            TrajectoryGenerationConfig.from_dict(
                {
                    "workflow": "structural_json",
                }
            )

    def test_rejects_unknown_section_keys_with_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid `floorplan` config keys"):
            TrajectoryGenerationConfig.from_dict(
                {
                    "dataset": {"dataset_root": "/tmp/data", "scene": "s"},
                    "floorplan": {"unknown_field": 1},
                }
            )
        with self.assertRaisesRegex(ValueError, "Invalid `walkthrough.behavior` config keys"):
            TrajectoryGenerationConfig.from_dict(
                {
                    "dataset": {"dataset_root": "/tmp/data", "scene": "s"},
                    "walkthrough": {"behavior": {"unknown_behavior_key": 1}},
                }
            )

    def test_from_dict_ignores_legacy_output_format_field(self) -> None:
        payload = {
            "dataset": {"dataset_root": "/tmp/data", "scene": "s"},
            "walkthrough": {
                "output_dir": "outputs/x",
                "output_format": "npz",
            },
        }
        loaded = TrajectoryGenerationConfig.from_dict(payload)
        self.assertEqual(loaded.walkthrough.output_dir, Path("outputs/x"))

    def test_rejects_invalid_motion_constraints(self) -> None:
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(max_linear_speed=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(max_angular_speed_deg=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(angular_smoothing_window_s=-0.1)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(polylabel_min_gain=-0.01)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(spin_points=1)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(travel_speed=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(spin_segment_speed=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(passthrough_speed=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_angle_search_deg=-1.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_angle_search_deg=180.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_search_steps=0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_length_weight=-0.1)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_outer_bias=-0.1)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_max_span_deg=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_max_span_deg=361.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_reverse_pref_deg=180.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_reverse_long_arc_bonus=-0.1)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_transition_risk_distance_weight=-0.1)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_transition_risk_angle_weight=-0.1)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(revisit_arc_max_tangent_mismatch_deg=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(passthrough_min_turn_deg=-1.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(passthrough_min_turn_deg=180.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(min_speed_clamp=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(spin_look_radius=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(door_buffer=0.0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(dense_samples_per_meter=0)
        with self.assertRaises(ValueError):
            WalkthroughBehaviorConfig(fov=180.0)
        with self.assertRaises(ValueError):
            WalkthroughConfig(fps=0)


if __name__ == "__main__":
    unittest.main()
