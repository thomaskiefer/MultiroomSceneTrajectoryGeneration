"""HouseLayout3D + Matterport3D adapter for end-to-end trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..artifacts import (
    FloorTrajectoryArtifact,
    TrajectoryGenerationArtifacts,
    write_trajectory_frames,
)
from ..config import TrajectoryGenerationConfig
from .._imports import import_with_tools_fallback
from ..validation import validate_trajectory


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModulePorts:
    """Named bundle of floorplan adapter dependencies."""

    connectivity: Any
    geometry: Any
    openings: Any
    rooms: Any
    floorplan_config_factory: Any
    walkthrough_factory: Any


def _is_local_walkthrough_compatible_graph(graph: Any) -> tuple[bool, str]:
    """Check whether a graph object matches LocalWalkthroughGenerator expectations."""
    if not hasattr(graph, "rooms") or not isinstance(graph.rooms, dict):
        return False, "missing graph.rooms dict"
    if not hasattr(graph, "adjacency") or not isinstance(graph.adjacency, dict):
        return False, "missing graph.adjacency dict"
    if not hasattr(graph, "connections"):
        return False, "missing graph.connections"

    if not graph.rooms:
        return True, ""

    sample = next(iter(graph.rooms.values()))
    if not (isinstance(sample, tuple) and len(sample) == 2):
        return False, "graph.rooms values must be (room, polygon) tuples"
    room_obj, poly_obj = sample
    if not hasattr(room_obj, "label_semantic") or not hasattr(room_obj, "centroid"):
        return False, "room object missing label_semantic/centroid"
    if not hasattr(poly_obj, "area") or not hasattr(poly_obj, "representative_point"):
        return False, "polygon object missing area/representative_point"
    return True, ""

def _resolve_project_root(project_root: Optional[Path]) -> Path:
    if project_root is not None:
        candidate = project_root.resolve()
        if (candidate / "tools" / "floorplan").exists():
            return candidate
        raise FileNotFoundError(
            "Explicit `project_root` does not contain tools/floorplan: "
            f"{candidate}"
        )

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tools" / "floorplan").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate project root containing tools/floorplan. "
        "Pass `project_root` explicitly."
    )


def _resolve_path(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_floorplan_modules(project_root: Path) -> ModulePorts:
    """
    Load floorplan modules, preferring standard import resolution.

    Only fallback to `sys.path` injection when running from local source tree.
    """
    def _import_ports() -> ModulePorts:
        from floorplan import connectivity, geometry, openings, rooms
        from floorplan.models import FloorplanConfig
        from floorplan.walkthrough import Walkthrough3DGS
        return ModulePorts(
            connectivity=connectivity,
            geometry=geometry,
            openings=openings,
            rooms=rooms,
            floorplan_config_factory=FloorplanConfig,
            walkthrough_factory=Walkthrough3DGS,
        )

    return import_with_tools_fallback(
        project_root=project_root,
        import_fn=_import_ports,
        module_prefix="floorplan",
        logger=logger,
    )


def _assign_rooms_to_closest_floor(
    all_rooms,
    floorplans,
    max_room_floor_distance: float = 4.0,
) -> dict[int, list]:
    rooms_for_floor_idx: dict[int, list] = {floor.level_index: [] for floor in floorplans}

    for room in all_rooms:
        room_z = room.centroid[2]
        best_floor = None
        min_dist = float("inf")

        for floor in floorplans:
            dist = abs(room_z - floor.mean_height)
            if dist < min_dist:
                min_dist = dist
                best_floor = floor

        if best_floor is not None and min_dist < max_room_floor_distance:
            rooms_for_floor_idx[best_floor.level_index].append(room)
        else:
            logger.warning(
                "Dropping room %s: nearest floor distance %.3fm exceeds threshold %.3fm.",
                getattr(room, "room_id", "<unknown>"),
                min_dist,
                max_room_floor_distance,
            )

    return rooms_for_floor_idx


def run_houselayout3d_matterport(
    config: TrajectoryGenerationConfig,
    project_root: Optional[Path] = None,
) -> TrajectoryGenerationArtifacts:
    """
    Run full pipeline from HouseLayout3D geometry + Matterport room annotations:
    floorplan extraction -> room/opening processing -> connectivity -> trajectories.
    """
    scene = config.dataset.scene.strip()
    if not scene:
        raise ValueError(
            "Dataset scene id is empty. Set `dataset.scene` to a valid HouseLayout3D/Matterport scene id."
        )

    resolved_root = _resolve_project_root(project_root)
    ports = _load_floorplan_modules(resolved_root)

    dataset_root = _resolve_path(resolved_root, config.dataset.dataset_root)
    house_segmentation_dir = _resolve_path(
        resolved_root, config.dataset.resolved_house_segmentation_dir
    )
    output_dir = _resolve_path(resolved_root, config.walkthrough.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = dataset_root / "structures" / "layouts_split_by_entity" / scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene meshes not found at {scene_dir}")

    floorplan_cfg = ports.floorplan_config_factory(
        height_tolerance=config.floorplan.height_tolerance,
        room_height_tolerance=config.floorplan.room_height_tolerance,
        room_height_tolerance_above=config.floorplan.room_height_tolerance_above,
        min_floor_area=config.floorplan.min_floor_area,
        merge_buffer=config.floorplan.merge_buffer,
        keep_only_largest=config.floorplan.keep_only_largest,
        opening_wall_tolerance=config.floorplan.opening_wall_tolerance,
    )

    artifacts = TrajectoryGenerationArtifacts(
        scene=scene,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    stair_ranges = ports.geometry.load_stair_height_ranges(dataset_root, scene)
    horizontal_surfaces, walls = ports.geometry.load_surfaces(scene_dir, stair_ranges, floorplan_cfg)
    clusters = ports.geometry.cluster_by_height(horizontal_surfaces, floorplan_cfg.height_tolerance)
    floorplans = ports.geometry.build_floorplans(clusters, floorplan_cfg)

    if not floorplans:
        raise RuntimeError("No floorplans were generated from scene geometry.")

    house_file = house_segmentation_dir / scene / f"{scene}.house"
    if not house_file.exists():
        raise FileNotFoundError(f"Matterport room annotations not found at {house_file}")

    house_data = ports.rooms.parse_house_file(house_file)
    all_rooms = house_data.get("rooms", [])
    if not all_rooms:
        raise RuntimeError(f"No rooms parsed from {house_file}")

    rooms_for_floor_idx = _assign_rooms_to_closest_floor(
        all_rooms,
        floorplans,
        max_room_floor_distance=config.floorplan.max_room_floor_distance,
    )
    rooms_by_floor = {}

    for floor in floorplans:
        floor_rooms = rooms_for_floor_idx.get(floor.level_index, [])
        if not floor_rooms:
            continue
        if config.floorplan.use_priority_assignment:
            room_data = ports.rooms.assign_rooms_priority_overlay(floor_rooms, floor, floorplan_cfg)
        else:
            room_data = ports.rooms.match_rooms_with_intersections(floor_rooms, floor, floorplan_cfg)
        if room_data:
            rooms_by_floor[floor.level_index] = room_data

    if not rooms_by_floor:
        raise RuntimeError("No rooms matched to generated floorplans.")

    openings_by_floor = {}
    doors_by_floor = {}
    if config.floorplan.include_openings:
        doors_file = dataset_root / "doors" / f"{scene}.json"
        windows_file = dataset_root / "windows" / f"{scene}.json"
        doors_list = ports.openings.load_openings(doors_file, "door") if doors_file.exists() else []
        windows_list = (
            ports.openings.load_openings(windows_file, "window") if windows_file.exists() else []
        )
        all_openings = doors_list + windows_list

        if all_openings:
            openings_by_floor_raw = ports.openings.match_openings_to_floors(
                all_openings, floorplans, floorplan_cfg, rooms_by_floor
            )
            walls_by_floor = ports.openings.match_walls_to_floors(walls, floorplans)
            openings_by_floor = ports.openings.match_openings_to_walls(
                openings_by_floor_raw,
                walls_by_floor,
                config.floorplan.opening_wall_tolerance,
            )
            for floor_idx, floor_openings in openings_by_floor.items():
                floor_doors = [o for o, _ in floor_openings if o.opening_type == "door"]
                if floor_doors:
                    doors_by_floor[floor_idx] = floor_doors
        else:
            artifacts.warnings.append("No openings found (doors/windows); using synthetic-only passages.")

    if config.connectivity.split_hallways_at_doors and doors_by_floor:
        for floor_idx in list(rooms_by_floor.keys()):
            if floor_idx in doors_by_floor:
                rooms_by_floor[floor_idx] = ports.rooms.split_hallways_at_doors(
                    rooms_by_floor[floor_idx],
                    doors_by_floor[floor_idx],
                    split_length=config.connectivity.hallway_split_length,
                )

    floor_polygons_by_floor = {
        floor.level_index: floor.footprint
        for floor in floorplans
    }
    connectivity_graphs = ports.connectivity.build_connectivity_graphs(
        rooms_by_floor=rooms_by_floor,
        floor_polygons_by_floor=floor_polygons_by_floor,
        doors_by_floor=doors_by_floor,
        proximity_threshold=config.connectivity.proximity_threshold,
        min_passage_area=config.connectivity.min_passage_area,
        door_tolerance=config.connectivity.door_match_tolerance,
    )

    if not connectivity_graphs:
        raise RuntimeError("Connectivity graph generation produced no graphs.")

    walkthrough_factory = ports.walkthrough_factory
    if config.walkthrough.use_local_walkthrough:
        from ..walkthrough_local import LocalWalkthroughGenerator

        def _local_factory(graph, floor_z, camera_height):
            compatible, reason = _is_local_walkthrough_compatible_graph(graph)
            if not compatible:
                raise RuntimeError(
                    "Local walkthrough was explicitly requested, but graph is incompatible "
                    f"({reason}). Use --use-legacy-walkthrough to force legacy behavior."
                )
            return LocalWalkthroughGenerator(
                graph=graph,
                floor_z=floor_z,
                camera_height=camera_height,
                behavior=config.walkthrough.behavior,
            )

        walkthrough_factory = _local_factory
        logger.info("Using local configurable walkthrough generator.")
    else:
        artifacts.warnings.append(
            "Using legacy walkthrough generator (Walkthrough3DGS). "
            "Some behavior settings are ignored in this mode (e.g. spin_orbit_scale). "
            "Use `--use-local-walkthrough` to apply refactored behavior config."
        )

    for floor_idx, graph in connectivity_graphs.items():
        stats = ports.connectivity.compute_graph_statistics(graph)
        artifacts.connectivity_statistics[floor_idx] = stats

        floor_plan = next((f for f in floorplans if f.level_index == floor_idx), None)
        if floor_plan is None:
            artifacts.warnings.append(f"Missing floorplan for floor index {floor_idx}; skipped trajectory.")
            continue

        walker = walkthrough_factory(
            graph=graph,
            floor_z=floor_plan.mean_height,
            camera_height=config.walkthrough.camera_height,
        )
        frames = walker.generate_exploration_path(fps=config.walkthrough.fps)
        if not frames:
            artifacts.warnings.append(f"No trajectory frames generated for floor {floor_idx}.")
            continue
        skipped_rooms = list(getattr(walker, "last_skipped_disconnected_rooms", []))
        if skipped_rooms:
            artifacts.warnings.append(
                f"[floor {floor_idx}] skipped {len(skipped_rooms)} room(s) outside largest connected component: "
                + ", ".join(skipped_rooms)
            )
        comp_count = int(getattr(walker, "last_disconnected_component_count", 1))
        if (
            config.walkthrough.behavior.disconnected_component_policy == "all_components"
            and comp_count > 1
        ):
            artifacts.warnings.append(
                f"[floor {floor_idx}] graph has {comp_count} disconnected components; "
                "trajectory restarts per component (no cross-component links)."
            )
        for warning in validate_trajectory(
            frames,
            floor_polygon=getattr(floor_plan, "footprint", None),
        ):
            artifacts.warnings.append(f"[floor {floor_idx}] {warning}")
        transfers = getattr(walker, "last_component_transfers", [])
        if transfers:
            artifacts.component_transfers_by_floor[floor_idx] = transfers

        trajectory_file = write_trajectory_frames(
            frames=frames,
            output_path_stem=output_dir / f"{scene}_floor_{floor_idx}_trajectory",
        )

        artifacts.floor_trajectories.append(
            FloorTrajectoryArtifact(
                floor_index=floor_idx,
                floor_z=floor_plan.mean_height,
                num_frames=len(frames),
                num_rooms=stats["num_rooms"],
                num_connections=stats["num_connections"],
                output_file=trajectory_file,
            )
        )

        if config.walkthrough.plot_output:
            try:
                from ..visualization import plot_trajectory
                plot_trajectory(
                    frames=frames,
                    floor_polygon=getattr(floor_plan, "footprint", None),
                    title=f"Floor {floor_idx} — {len(frames)} frames",
                    output_path=output_dir / f"{scene}_floor_{floor_idx}_trajectory_visualization.png",
                    viz_config=config.walkthrough.viz,
                )
            except ImportError:
                logger.warning("matplotlib not available; skipping trajectory plot.")

    if config.walkthrough.write_debug_summary:
        summary_path = output_dir / f"{scene}_trajectory_generation_summary.json"
        summary_path.write_text(json.dumps(artifacts.to_dict(), indent=2))

    return artifacts
