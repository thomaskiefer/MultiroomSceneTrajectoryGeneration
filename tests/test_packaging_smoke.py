from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import subprocess
import shutil
import sys
import tempfile
import unittest


class PackagingSmokeTest(unittest.TestCase):
    def test_editable_install_then_import_from_clean_cwd(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        pip_available = importlib.util.find_spec("pip") is not None
        uv_bin = shutil.which("uv")
        if not pip_available and uv_bin is None:
            self.skipTest("Neither pip nor uv is available in this Python environment")

        if pip_available:
            install_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                f"{repo_root}[geometry]",
                "--no-deps",
            ]
        else:
            install_cmd = [
                uv_bin,
                "pip",
                "install",
                "--python",
                sys.executable,
                "-e",
                f"{repo_root}[geometry]",
                "--no-deps",
            ]

        subprocess.run(
            install_cmd,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import trajectory_generation as tg; print(tg.__name__)",
                ],
                cwd=tmp_dir,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertIn("trajectory_generation", proc.stdout.strip())


if __name__ == "__main__":
    unittest.main()
