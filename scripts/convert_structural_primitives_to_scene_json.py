#!/usr/bin/env python3
"""Convert neutral structural-primitives input into scene.schema.v1 JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _resolve_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_DIR = _resolve_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.preprocess import convert_structural_primitives_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert structural-primitives JSON to canonical scene.schema.v1 JSON"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input structural-primitives JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output scene.schema.v1 JSON path.")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional explicit scene id override.",
    )
    parser.add_argument(
        "--proximity-threshold",
        type=float,
        default=0.25,
        help="Bbox-proximity threshold [m] used when deriving connections without openings.",
    )
    parser.add_argument(
        "--opening-room-distance-threshold",
        type=float,
        default=3.0,
        help="Max room-center distance [m] for assigning openings to room pairs.",
    )
    args = parser.parse_args()

    out = convert_structural_primitives_file(
        input_path=args.input,
        output_path=args.output,
        scene_id=args.scene,
        proximity_threshold=args.proximity_threshold,
        opening_room_distance_threshold=args.opening_room_distance_threshold,
    )
    print(f"Wrote structural scene JSON: {out}")


if __name__ == "__main__":
    main()
