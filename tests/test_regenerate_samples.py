from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


class RegenerateSamplesScriptTest(unittest.TestCase):
    def test_rejects_output_root_that_contains_project_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "regenerate_samples.py"
        # Parent of repo_root is an ancestor and must be rejected.
        unsafe_output_root = repo_root.parent

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(repo_root),
                "--output-root",
                str(unsafe_output_root),
                "--no-clean",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Refusing unsafe output_root", proc.stderr)


if __name__ == "__main__":
    unittest.main()
