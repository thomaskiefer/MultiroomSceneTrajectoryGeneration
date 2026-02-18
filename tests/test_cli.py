from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.cli import (
    _build_generate_parser,
    _build_preprocess_config,
    _configure_logging,
    _parse_cli_args,
    build_config,
)
from trajectory_generation.config import TrajectoryGenerationConfig


class CliConfigBuildTest(unittest.TestCase):
    def _base_args(self) -> Namespace:
        return Namespace(
            dataset_root=None,
            scene=None,
            output_dir=Path("outputs/trajectory_generation"),
            house_segmentation_dir=None,
            workflow=None,
            scene_input_json=None,
            config_json=None,
            project_root=None,
            use_local_walkthrough=False,
            use_legacy_walkthrough=False,
            validate_schema=False,
            log_level="INFO",
        )

    def _base_preprocess_args(self) -> Namespace:
        return Namespace(
            dataset_root=None,
            scene=None,
            house_segmentation_dir=None,
            config_json=None,
            project_root=None,
            geojson_output=None,
            structural_output=None,
            scene_id=None,
            geojson_only=False,
            emit_debug_artifacts=False,
            debug_output_dir=None,
            no_floor_geojson=False,
            no_combined_plot=False,
            no_floor_plots=False,
            use_raw_room_centers=False,
            no_diagnostics=False,
            log_level="INFO",
        )

    def test_structural_json_workflow_builds_config(self) -> None:
        args = self._base_args()
        args.workflow = "structural_json"
        args.scene_input_json = Path("/tmp/scene.json")
        args.dataset_root = Path("/tmp/dataset")

        cfg = build_config(args)

        self.assertEqual(cfg.workflow, "structural_json")
        self.assertEqual(cfg.dataset.scene_input_json, Path("/tmp/scene.json"))
        self.assertEqual(cfg.dataset.dataset_root, Path("/tmp/dataset"))

    def test_config_json_overrides_workflow_and_scene_input_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            base_cfg = TrajectoryGenerationConfig.houselayout3d_matterport(
                dataset_root=Path("/tmp/dataset"),
                scene="scene_123",
            )
            base_cfg.to_json(config_path)

            args = self._base_args()
            args.config_json = config_path
            args.workflow = "structural_json"
            args.scene_input_json = Path("/tmp/override_scene.json")

            cfg = build_config(args)

        self.assertEqual(cfg.workflow, "structural_json")
        self.assertEqual(cfg.dataset.scene_input_json, Path("/tmp/override_scene.json"))

    def test_matterport_build_path_unchanged(self) -> None:
        args = self._base_args()
        args.dataset_root = Path("/tmp/dataset")
        args.scene = "scene_123"

        cfg = build_config(args)

        self.assertEqual(cfg.workflow, "houselayout3d_matterport")
        self.assertEqual(cfg.dataset.dataset_root, Path("/tmp/dataset"))
        self.assertEqual(cfg.dataset.scene, "scene_123")

    def test_cli_can_force_local_walkthrough(self) -> None:
        args = self._base_args()
        args.dataset_root = Path("/tmp/dataset")
        args.scene = "scene_123"
        args.use_local_walkthrough = True

        cfg = build_config(args)

        self.assertTrue(cfg.walkthrough.use_local_walkthrough)

    def test_cli_can_force_legacy_walkthrough_even_with_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            base_cfg = TrajectoryGenerationConfig.houselayout3d_matterport(
                dataset_root=Path("/tmp/dataset"),
                scene="scene_123",
            )
            base_cfg.walkthrough.use_local_walkthrough = True
            base_cfg.to_json(config_path)

            args = self._base_args()
            args.config_json = config_path
            args.use_legacy_walkthrough = True

            cfg = build_config(args)

        self.assertFalse(cfg.walkthrough.use_local_walkthrough)

    def test_preprocess_builds_hl3d_config(self) -> None:
        args = self._base_preprocess_args()
        args.dataset_root = Path("/tmp/dataset")
        args.scene = "scene_123"
        cfg = _build_preprocess_config(args)
        self.assertEqual(cfg.workflow, "houselayout3d_matterport")
        self.assertEqual(cfg.dataset.scene, "scene_123")

    def test_preprocess_rejects_non_hl3d_workflow_from_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            TrajectoryGenerationConfig.structural_json(
                scene_input_json=Path("/tmp/scene.json")
            ).to_json(config_path)
            args = self._base_preprocess_args()
            args.config_json = config_path
            with self.assertRaises(ValueError):
                _build_preprocess_config(args)

    def test_parse_cli_args_supports_preprocess_subcommand(self) -> None:
        command, args = _parse_cli_args(
            [
                "preprocess",
                "--dataset-root",
                "/tmp/dataset",
                "--scene",
                "scene_123",
                "--geojson-only",
            ]
        )
        self.assertEqual(command, "preprocess")
        self.assertEqual(args.scene, "scene_123")
        self.assertTrue(args.geojson_only)

    def test_parse_cli_args_generate_without_subcommand_is_backward_compatible(self) -> None:
        command, args = _parse_cli_args(
            [
                "--workflow",
                "structural_json",
                "--scene-input-json",
                "/tmp/scene.json",
            ]
        )
        self.assertEqual(command, "generate")
        self.assertEqual(args.workflow, "structural_json")
        self.assertEqual(args.log_level, "INFO")

    def test_infers_structural_workflow_from_scene_input(self) -> None:
        args = self._base_args()
        args.scene_input_json = Path("/tmp/scene.json")
        cfg = build_config(args)
        self.assertEqual(cfg.workflow, "structural_json")

    def test_generate_help_lists_key_inputs(self) -> None:
        help_text = _build_generate_parser().format_help()
        self.assertIn("--dataset-root", help_text)
        self.assertIn("--output-dir", help_text)
        self.assertIn("--scene-input-json", help_text)

    def test_configure_logging_rejects_invalid_level(self) -> None:
        with self.assertRaises(ValueError):
            _configure_logging("NOPE")


if __name__ == "__main__":
    unittest.main()
