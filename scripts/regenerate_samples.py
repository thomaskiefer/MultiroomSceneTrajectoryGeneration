#!/usr/bin/env python3
"""Regenerate canonical sample artifacts under samples/."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys


def _resolve_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_DIR = _resolve_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import TrajectoryGenerationConfig  # noqa: E402
from trajectory_generation.schema import parse_structural_scene_file  # noqa: E402
from trajectory_generation.pipeline import run  # noqa: E402
from trajectory_generation.scene_geojson import build_connectivity_geojson_from_structural_scene  # noqa: E402
from trajectory_generation.visualization import render_trajectory_image, render_trajectory_video  # noqa: E402


@dataclass(frozen=True)
class SampleCase:
    scene_input: Path


def _sample_cases(repo_root: Path) -> list[SampleCase]:
    structural_root = repo_root / "examples" / "structural"
    return [
        SampleCase(scene_input=structural_root / "demo_apartment.json"),
        SampleCase(scene_input=structural_root / "2t7WUuJeko7.json"),
        SampleCase(scene_input=structural_root / "structural_template.json"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate samples/ from examples/structural")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("samples"),
        help="Sample output root (relative to project root unless absolute)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove existing output root before regeneration",
    )
    parser.add_argument(
        "--with-video",
        action="store_true",
        help="Also render MP4 videos (slower)",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else (project_root / args.output_root).resolve()
    if output_root == project_root:
        raise ValueError(
            f"Refusing to use project root as output_root: {output_root}. "
            "Choose a subdirectory (for example `samples`)."
        )

    if not args.no_clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for case in _sample_cases(project_root):
        payload = parse_structural_scene_file(case.scene_input)
        scene_name = payload.scene
        scene_out_dir = output_root / scene_name
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        cfg = TrajectoryGenerationConfig.structural_json(
            scene_input_json=case.scene_input,
            output_dir=scene_out_dir,
            dataset_root=project_root,
        )
        artifacts = run(cfg, project_root=project_root)

        geojson_path = scene_out_dir / f"{scene_name}_connectivity.geojson"
        build_connectivity_geojson_from_structural_scene(case.scene_input, geojson_path, cfg)

        for floor_artifact in artifacts.floor_trajectories:
            trajectory_path = Path(floor_artifact.output_file)
            viz_dir = scene_out_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            image_path = viz_dir / f"{trajectory_path.stem}_visualization.png"
            render_trajectory_image(
                geojson_path=geojson_path,
                trajectory_path=trajectory_path,
                output_path=image_path,
                scene_name=scene_name,
                fps=cfg.walkthrough.fps,
            )
            if args.with_video:
                video_path = viz_dir / f"{trajectory_path.stem}.mp4"
                render_trajectory_video(
                    geojson_path=geojson_path,
                    trajectory_path=trajectory_path,
                    output_path=video_path,
                    scene_name=scene_name,
                    fps=cfg.walkthrough.fps,
                    speed=1.0,
                )

        print(f"[ok] {scene_name}: {scene_out_dir}")


if __name__ == "__main__":
    main()
