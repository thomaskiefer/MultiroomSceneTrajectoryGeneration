from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest


class CliEndToEndTest(unittest.TestCase):
    @staticmethod
    def _example_scene_path(repo_root: Path) -> Path:
        structural_dir = repo_root / "examples" / "structural"
        path = structural_dir / "demo_apartment.json"
        if not path.exists():
            raise FileNotFoundError(
                "Missing demo structural scene: examples/structural/demo_apartment.json"
            )
        return path

    @staticmethod
    def _scene_name(scene_json: Path) -> str:
        payload = json.loads(scene_json.read_text())
        return str(payload["scene"])

    def test_cli_version_flag(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        src_path = str(repo_root / "src")
        env["PYTHONPATH"] = (
            src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
        )

        proc = subprocess.run(
            [sys.executable, "-m", "trajectory_generation", "--version"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("mrstg", proc.stdout)
        self.assertIn("0.1.0", proc.stdout)

    def test_structural_json_cli_generates_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scene_json = self._example_scene_path(repo_root)
        scene_name = self._scene_name(scene_json)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "cli_out"
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            src_path = str(repo_root / "src")
            env["PYTHONPATH"] = (
                src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "trajectory_generation",
                    "--workflow",
                    "structural_json",
                    "--scene-input-json",
                    str(scene_json),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Floors with trajectories: 1", proc.stdout)
            trajectory_json = output_dir / f"{scene_name}_floor_0_trajectory.json"
            summary_json = output_dir / f"{scene_name}_trajectory_generation_summary.json"
            self.assertTrue(trajectory_json.exists())
            self.assertTrue(summary_json.exists())

    def test_structural_json_validate_schema_mode(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scene_json = self._example_scene_path(repo_root)

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        src_path = str(repo_root / "src")
        env["PYTHONPATH"] = (
            src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
        )

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "trajectory_generation",
                "--validate-schema",
                "--scene-input-json",
                str(scene_json),
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("Schema valid:", proc.stdout)
        self.assertIn("schema_version:", proc.stdout)

    def test_cli_user_error_is_reported_without_traceback(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        src_path = str(repo_root / "src")
        env["PYTHONPATH"] = (
            src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
        )

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "trajectory_generation",
                "--workflow",
                "structural_json",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Error:", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_legacy_cli_module_invocation_generates_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scene_json = self._example_scene_path(repo_root)
        scene_name = self._scene_name(scene_json)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "cli_module_out"
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            src_path = str(repo_root / "src")
            env["PYTHONPATH"] = (
                src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "trajectory_generation.cli",
                    "--workflow",
                    "structural_json",
                    "--scene-input-json",
                    str(scene_json),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Floors with trajectories: 1", proc.stdout)
            trajectory_json = output_dir / f"{scene_name}_floor_0_trajectory.json"
            summary_json = output_dir / f"{scene_name}_trajectory_generation_summary.json"
            self.assertTrue(trajectory_json.exists())
            self.assertTrue(summary_json.exists())


if __name__ == "__main__":
    unittest.main()
