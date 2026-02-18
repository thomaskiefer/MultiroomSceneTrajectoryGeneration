from __future__ import annotations

from pathlib import Path
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import TrajectoryGenerationConfig
import trajectory_generation.pipeline as pipeline


class PipelineDispatchTest(unittest.TestCase):
    def test_dispatches_houselayout3d_matterport(self) -> None:
        cfg = TrajectoryGenerationConfig.houselayout3d_matterport(
            dataset_root=Path("/tmp/dataset"),
            scene="scene_123",
        )
        calls: list[tuple] = []

        def _runner(config, project_root):
            calls.append((config, project_root))
            return "ok_hl3d"

        original = pipeline.RUNNERS.get("houselayout3d_matterport")
        pipeline.register_runner("houselayout3d_matterport", _runner)
        try:
            out = pipeline.run(cfg, project_root=Path("/tmp/root"))
        finally:
            pipeline.RUNNERS["houselayout3d_matterport"] = original

        self.assertEqual(out, "ok_hl3d")
        self.assertEqual(len(calls), 1)

    def test_dispatches_structural_json(self) -> None:
        cfg = TrajectoryGenerationConfig.structural_json(
            scene_input_json=Path("/tmp/scene.json"),
        )
        calls: list[tuple] = []

        def _runner(config, project_root):
            calls.append((config, project_root))
            return "ok_struct"

        original = pipeline.RUNNERS.get("structural_json")
        pipeline.register_runner("structural_json", _runner)
        try:
            out = pipeline.run(cfg, project_root=Path("/tmp/root"))
        finally:
            pipeline.RUNNERS["structural_json"] = original

        self.assertEqual(out, "ok_struct")
        self.assertEqual(len(calls), 1)

    def test_rejects_unknown_workflow(self) -> None:
        cfg = TrajectoryGenerationConfig.houselayout3d_matterport(
            dataset_root=Path("/tmp/dataset"),
            scene="scene_123",
        )
        cfg.workflow = "unknown"
        with self.assertRaises(ValueError):
            pipeline.run(cfg)

    def test_register_runner_replaces_callable(self) -> None:
        cfg = TrajectoryGenerationConfig.structural_json(scene_input_json=Path("/tmp/scene.json"))
        original = pipeline.RUNNERS.get("structural_json")

        def _stub_runner(config, project_root):
            return "stub_ok"

        pipeline.register_runner("structural_json", _stub_runner)
        try:
            out = pipeline.run(cfg, project_root=Path("/tmp/root"))
        finally:
            pipeline.RUNNERS["structural_json"] = original

        self.assertEqual(out, "stub_ok")

    def test_backwards_compatible_run_trajectory_generation_alias(self) -> None:
        cfg = TrajectoryGenerationConfig.structural_json(scene_input_json=Path("/tmp/scene.json"))
        original = pipeline.RUNNERS.get("structural_json")

        def _stub_runner(config, project_root):
            return "alias_ok"

        pipeline.register_runner("structural_json", _stub_runner)
        try:
            out = pipeline.run_trajectory_generation(cfg, project_root=Path("/tmp/root"))
        finally:
            pipeline.RUNNERS["structural_json"] = original

        self.assertEqual(out, "alias_ok")


if __name__ == "__main__":
    unittest.main()
