#!/usr/bin/env python3
"""Lightweight benchmark for structural_json trajectory generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys
import time


def _resolve_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_DIR = _resolve_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import TrajectoryGenerationConfig  # noqa: E402
from trajectory_generation.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark structural_json generation runtime")
    parser.add_argument(
        "--scene-input-json",
        type=Path,
        default=Path("examples/structural/demo_apartment.json"),
        help="Input structural scene JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench"),
        help="Temporary output directory for benchmark runs",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Number of repeated runs")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional output JSON report path",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep generated artifacts in output dir",
    )
    args = parser.parse_args()

    project_root = (args.project_root or Path(__file__).resolve().parents[1]).resolve()
    scene_input = args.scene_input_json
    if not scene_input.is_absolute():
        scene_input = (project_root / scene_input).resolve()

    runtimes_s: list[float] = []
    frame_counts: list[int] = []

    for i in range(args.repeat):
        run_out = args.output_dir / f"run_{i:02d}"
        cfg = TrajectoryGenerationConfig.structural_json(
            scene_input_json=scene_input,
            output_dir=run_out,
            dataset_root=project_root,
        )

        t0 = time.perf_counter()
        artifacts = run(cfg, project_root=project_root)
        dt = time.perf_counter() - t0
        runtimes_s.append(dt)

        total_frames = sum(item.num_frames for item in artifacts.floor_trajectories)
        frame_counts.append(total_frames)

    summary = {
        "scene_input_json": str(scene_input),
        "repeat": args.repeat,
        "runtime_s": {
            "min": min(runtimes_s),
            "max": max(runtimes_s),
            "mean": statistics.mean(runtimes_s),
            "median": statistics.median(runtimes_s),
        },
        "frame_count": {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "mean": statistics.mean(frame_counts),
            "median": statistics.median(frame_counts),
        },
        "runs": [
            {"run_index": i, "runtime_s": runtimes_s[i], "total_frames": frame_counts[i]}
            for i in range(args.repeat)
        ],
    }

    print("Benchmark summary")
    print(json.dumps(summary, indent=2))

    if args.report_json is not None:
        report_path = args.report_json
        if not report_path.is_absolute():
            report_path = (project_root / report_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote report: {report_path}")

    if not args.keep_artifacts:
        shutil.rmtree(args.output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
