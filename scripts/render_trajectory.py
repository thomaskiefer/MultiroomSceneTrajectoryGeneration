#!/usr/bin/env python3
"""Source-tree wrapper for unified trajectory visualization CLI."""

from __future__ import annotations

from pathlib import Path
import sys


def _resolve_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_DIR = _resolve_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.visualization import main  # noqa: E402


if __name__ == "__main__":
    main()
