#!/usr/bin/env python3
"""Convert connectivity GeoJSON into canonical structural scene JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _resolve_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_DIR = _resolve_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.preprocess import convert_connectivity_geojson_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert connectivity GeoJSON to structural scene JSON")
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional explicit scene id (defaults to geojson stem prefix).",
    )
    args = parser.parse_args()

    out = convert_connectivity_geojson_file(
        geojson_path=args.geojson,
        output_path=args.output,
        scene_id=args.scene,
    )
    print(f"Wrote structural scene JSON: {out}")


if __name__ == "__main__":
    main()
