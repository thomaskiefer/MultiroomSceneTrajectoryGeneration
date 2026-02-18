#!/usr/bin/env python3
"""Regenerate canonical sample artifacts under samples/."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import numpy as np


def _resolve_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_DIR = _resolve_src_dir()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import TrajectoryGenerationConfig  # noqa: E402
from trajectory_generation.schema import parse_structural_scene_file  # noqa: E402
from trajectory_generation.adapters.structural_json import _build_graph_for_floor  # noqa: E402
from trajectory_generation.pipeline import run  # noqa: E402
from trajectory_generation.walkthrough_local import LocalWalkthroughGenerator  # noqa: E402
from trajectory_generation.visualization import render_trajectory_image, render_trajectory_video  # noqa: E402


@dataclass(frozen=True)
class SampleCase:
    scene_input: Path


def _close_ring(points: list[list[float]]) -> list[list[float]]:
    if points and points[0] != points[-1]:
        return points + [points[0]]
    return points


def _build_geojson_for_structural_scene(
    scene_input: Path,
    geojson_output: Path,
    cfg: TrajectoryGenerationConfig,
) -> Path:
    payload = parse_structural_scene_file(scene_input)
    floors = payload.floors
    rooms = payload.rooms
    conns = payload.connections

    floors_by_idx = {f.floor_index: f for f in floors}
    rooms_by_floor: dict[int, list] = {f.floor_index: [] for f in floors}
    for room in rooms:
        rooms_by_floor.setdefault(room.floor_index, []).append(room)

    room_floor = {room.room_id: room.floor_index for room in rooms}
    explicit_by_floor: dict[int, list] = {f.floor_index: [] for f in floors}
    for conn in conns:
        if (
            conn.room1_id in room_floor
            and conn.room2_id in room_floor
            and room_floor[conn.room1_id] == room_floor[conn.room2_id]
        ):
            explicit_by_floor[room_floor[conn.room1_id]].append(conn)

    features: list[dict] = []

    for floor_idx, floor in sorted(floors_by_idx.items()):
        floor_rooms = rooms_by_floor.get(floor_idx, [])
        graph = _build_graph_for_floor(
            floor_rooms=floor_rooms,
            explicit_connections=explicit_by_floor.get(floor_idx, []),
            proximity_threshold=cfg.connectivity.proximity_threshold,
        )
        walker = LocalWalkthroughGenerator(
            graph=graph,
            floor_z=floor.z,
            camera_height=cfg.walkthrough.camera_height,
            behavior=cfg.walkthrough.behavior,
        )
        center_map = {room_id: walker._get_room_center(room_id) for room_id in graph.rooms.keys()}

        floor_ring = _close_ring([[float(x), float(y)] for x, y in floor.footprint_xy])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [floor_ring]},
                "properties": {
                    "type": "floor_footprint",
                    "level_index": int(floor_idx),
                    "mean_height": float(floor.z),
                },
            }
        )

        room_specs = {room.room_id: room for room in floor_rooms}
        for room_id, (node, _) in graph.rooms.items():
            room = room_specs[room_id]
            if room.polygon_xy is not None:
                ring = _close_ring([[float(x), float(y)] for x, y in room.polygon_xy])
            else:
                x0, y0 = float(room.min_xy[0]), float(room.min_xy[1])
                x1, y1 = float(room.max_xy[0]), float(room.max_xy[1])
                ring = _close_ring([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

            center = center_map[room_id]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "type": "room",
                        "room_id": room_id,
                        "label_semantic": node.label_semantic,
                        "level_index": int(floor_idx),
                        "bbox_3d_min": [float(v) for v in room.min_xyz],
                        "bbox_3d_max": [float(v) for v in room.max_xyz],
                        "centroid_3d": [float(v) for v in room.centroid],
                        "trajectory_center_xy": [float(center[0]), float(center[1])],
                        "trajectory_center_3d": [float(center[0]), float(center[1]), float(center[2])],
                    },
                }
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(center[0]), float(center[1])]},
                    "properties": {
                        "type": "trajectory_room_center",
                        "room_id": room_id,
                        "level_index": int(floor_idx),
                    },
                }
            )

        for conn in graph.connections:
            c1 = center_map[conn.room1_id]
            c2 = center_map[conn.room2_id]
            wp = np.asarray(conn.waypoint.position, dtype=float).reshape(-1)
            wx, wy = float(wp[0]), float(wp[1])

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(c1[0]), float(c1[1])], [wx, wy], [float(c2[0]), float(c2[1])]],
                    },
                    "properties": {
                        "type": "room_connection",
                        "room1_id": conn.room1_id,
                        "room2_id": conn.room2_id,
                        "door_type": getattr(conn.waypoint, "door_type", "synthetic"),
                        "level_index": int(floor_idx),
                    },
                }
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [wx, wy]},
                    "properties": {
                        "type": "door_waypoint",
                        "room1_id": conn.room1_id,
                        "room2_id": conn.room2_id,
                        "door_type": getattr(conn.waypoint, "door_type", "synthetic"),
                        "level_index": int(floor_idx),
                    },
                }
            )

    geojson_output.parent.mkdir(parents=True, exist_ok=True)
    geojson_output.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
    return geojson_output


def _sample_cases(repo_root: Path) -> list[SampleCase]:
    structural_root = repo_root / "examples" / "structural"
    return [
        SampleCase(scene_input=structural_root / "demo_apartment.json"),
        SampleCase(scene_input=structural_root / "matterport_2t7WUuJeko7.json"),
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
        _build_geojson_for_structural_scene(case.scene_input, geojson_path, cfg)

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
