#!/usr/bin/env python3
"""Minimal end-to-end demo: generate trajectories and render visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _resolve_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_DIR = _resolve_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import TrajectoryGenerationConfig  # noqa: E402
from trajectory_generation.pipeline import run  # noqa: E402
from trajectory_generation.scene_geojson import build_connectivity_geojson_from_structural_scene  # noqa: E402
from trajectory_generation.visualization import render_trajectory_image, render_trajectory_video  # noqa: E402


def _resolve_path(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal standalone trajectory generation + visualization script."
    )
    parser.add_argument(
        "--scene-input-json",
        type=Path,
        default=Path("examples/structural/demo_apartment.json"),
        help="Input structural scene JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/minimal_demo"),
        help="Output directory for trajectories and visualizations.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root used to resolve relative paths.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Trajectory/video FPS.",
    )
    parser.add_argument(
        "--with-video",
        action="store_true",
        help="Also render MP4 videos in addition to PNG images.",
    )
    parser.add_argument(
        "--video-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for rendered videos.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    scene_input_json = _resolve_path(project_root, args.scene_input_json)
    output_dir = _resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrajectoryGenerationConfig.structural_json(
        scene_input_json=scene_input_json,
        output_dir=output_dir,
        dataset_root=project_root,
    )
    config.walkthrough.fps = args.fps

    artifacts = run(config=config, project_root=project_root)
    if not artifacts.floor_trajectories:
        raise RuntimeError("No floor trajectories were generated.")

    geojson_path = output_dir / f"{artifacts.scene}_connectivity.geojson"
    build_connectivity_geojson_from_structural_scene(
        scene_input=scene_input_json,
        geojson_output=geojson_path,
        config=config,
    )

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene: {artifacts.scene}")
    print(f"Trajectory output: {output_dir}")
    print(f"Connectivity GeoJSON: {geojson_path}")
    for floor in artifacts.floor_trajectories:
        trajectory_path = Path(floor.output_file)
        image_path = viz_dir / f"{trajectory_path.stem}_visualization.png"
        render_trajectory_image(
            geojson_path=geojson_path,
            trajectory_path=trajectory_path,
            output_path=image_path,
            scene_name=artifacts.scene,
            fps=config.walkthrough.fps,
        )
        print(f"Floor {floor.floor_index} image: {image_path}")

        if args.with_video:
            video_path = viz_dir / f"{trajectory_path.stem}.mp4"
            render_trajectory_video(
                geojson_path=geojson_path,
                trajectory_path=trajectory_path,
                output_path=video_path,
                scene_name=artifacts.scene,
                fps=config.walkthrough.fps,
                speed=args.video_speed,
            )
            print(f"Floor {floor.floor_index} video: {video_path}")


if __name__ == "__main__":
    main()
