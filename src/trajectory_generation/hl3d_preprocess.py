"""Dataset-specific to canonical structural scene preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._imports import import_with_tools_fallback
from .config import TrajectoryGenerationConfig
from .debug_visualization import render_hl3d_debug_plots
from .geojson_converter import convert_connectivity_geojson_file


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hl3dPreprocessContext:
    """In-memory preprocessing context reused across artifact exporters."""

    scene: str
    floorplans: list[Any]
    rooms_by_floor: dict[int, list[Any]]
    openings_by_floor: dict[int, list[Any]] | None
    connectivity_graphs: dict[int, Any]
    stair_height_ranges: tuple[tuple[float, float], ...] = ()
    unmatched_room_ids: tuple[str, ...] = ()
    unmatched_opening_ids: tuple[str, ...] = ()
    matched_to_floor_not_wall_opening_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hl3dDebugArtifacts:
    """Debug artifacts exported from HL3D/Matterport preprocessing context."""

    connectivity_geojson: Path
    floor_geojson_files: tuple[Path, ...] = ()
    combined_plot_file: Optional[Path] = None
    floor_plot_files: tuple[Path, ...] = ()
    diagnostics_json_file: Optional[Path] = None
    room_matching_stats_csv: Optional[Path] = None

def _load_floorplan_export_module(project_root: Path) -> Any:
    def _import_export_module() -> Any:
        from floorplan import export as floorplan_module

        return floorplan_module

    return import_with_tools_fallback(
        project_root=project_root,
        import_fn=_import_export_module,
        module_prefix="floorplan",
        logger=logger,
    )


def _build_hl3d_matterport_context(
    config: TrajectoryGenerationConfig,
    project_root: Path | None = None,
) -> tuple[Hl3dPreprocessContext, Path]:
    """Build in-memory preprocessing context from HL3D/Matterport sources."""
    from .adapters import houselayout3d_matterport as hl3d

    resolved_root = hl3d._resolve_project_root(project_root)
    ports = hl3d._load_floorplan_modules(resolved_root)

    dataset_root = hl3d._resolve_path(resolved_root, config.dataset.dataset_root)
    house_segmentation_dir = hl3d._resolve_path(
        resolved_root, config.dataset.resolved_house_segmentation_dir
    )

    scene = config.dataset.scene
    if not scene:
        raise ValueError("`dataset.scene` is required for HouseLayout3D preprocessing.")

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

    rooms_for_floor_idx = hl3d._assign_rooms_to_closest_floor(
        all_rooms,
        floorplans,
        max_room_floor_distance=config.floorplan.max_room_floor_distance,
    )
    rooms_by_floor: dict[int, list[Any]] = {}
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
    matched_room_ids = {
        str(room_tuple[0].room_id)
        for room_data in rooms_by_floor.values()
        for room_tuple in room_data
        if isinstance(room_tuple, tuple) and room_tuple
    }
    unmatched_room_ids = tuple(
        sorted(str(room.room_id) for room in all_rooms if str(room.room_id) not in matched_room_ids)
    )

    openings_by_floor: dict[int, list[Any]] | None = {}
    doors_by_floor: dict[int, list[Any]] = {}
    openings_by_floor_raw: dict[int, list[Any]] = {}
    all_openings: list[Any] = []
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

    unmatched_opening_ids: tuple[str, ...] = ()
    matched_to_floor_not_wall_opening_ids: tuple[str, ...] = ()
    if all_openings:
        if hasattr(ports.openings, "find_unmatched_openings"):
            unmatched_openings = ports.openings.find_unmatched_openings(
                all_openings,
                openings_by_floor or {},
            )
            unmatched_opening_ids = tuple(
                sorted(str(getattr(opening, "opening_id", "")) for opening in unmatched_openings)
            )
        if openings_by_floor_raw and openings_by_floor:
            floor_not_wall_ids: list[str] = []
            for floor in floorplans:
                floor_idx = floor.level_index
                floor_openings_raw = openings_by_floor_raw.get(floor_idx, [])
                floor_matched = (openings_by_floor or {}).get(floor_idx, [])
                floor_matched_ids = {id(o) for o, _ in floor_matched}
                for opening in floor_openings_raw:
                    if id(opening) not in floor_matched_ids:
                        floor_not_wall_ids.append(str(getattr(opening, "opening_id", "")))
            matched_to_floor_not_wall_opening_ids = tuple(sorted(floor_not_wall_ids))

    if config.connectivity.split_hallways_at_doors and doors_by_floor:
        for floor_idx in list(rooms_by_floor.keys()):
            if floor_idx in doors_by_floor:
                rooms_by_floor[floor_idx] = ports.rooms.split_hallways_at_doors(
                    rooms_by_floor[floor_idx],
                    doors_by_floor[floor_idx],
                    split_length=config.connectivity.hallway_split_length,
                )

    floor_polygons_by_floor = {floor.level_index: floor.footprint for floor in floorplans}
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

    stair_height_ranges: list[tuple[float, float]] = []
    for idx, stair_range in enumerate(stair_ranges):
        if not isinstance(stair_range, (list, tuple)) or len(stair_range) != 2:
            logger.warning("Ignoring stair range %d with invalid format: %r", idx, stair_range)
            continue
        try:
            z0 = float(stair_range[0])
            z1 = float(stair_range[1])
        except (TypeError, ValueError):
            logger.warning("Ignoring stair range %d with non-numeric values: %r", idx, stair_range)
            continue
        if not (np.isfinite(z0) and np.isfinite(z1)):
            logger.warning("Ignoring stair range %d with non-finite values: %r", idx, stair_range)
            continue
        if z0 <= z1:
            stair_height_ranges.append((z0, z1))
        else:
            stair_height_ranges.append((z1, z0))

    return (
        Hl3dPreprocessContext(
            scene=scene,
            floorplans=floorplans,
            rooms_by_floor=rooms_by_floor,
            openings_by_floor=openings_by_floor or None,
            connectivity_graphs=connectivity_graphs,
            stair_height_ranges=tuple(stair_height_ranges),
            unmatched_room_ids=unmatched_room_ids,
            unmatched_opening_ids=unmatched_opening_ids,
            matched_to_floor_not_wall_opening_ids=matched_to_floor_not_wall_opening_ids,
        ),
        resolved_root,
    )


def _write_connectivity_geojson(
    context: Hl3dPreprocessContext,
    output_path: Path,
    export_mod: Any,
    center_map_by_floor: dict[int, dict[str, np.ndarray]] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_mod.export_geojson(
        floorplans=context.floorplans,
        output_path=output_path,
        rooms_by_floor=context.rooms_by_floor,
        openings_by_floor=context.openings_by_floor,
        connectivity_graphs=context.connectivity_graphs,
    )
    if center_map_by_floor:
        _inject_trajectory_centers_into_geojson(
            geojson_path=output_path,
            center_map_by_floor=center_map_by_floor,
        )
    return output_path


def _compute_center_map_by_floor(
    context: Hl3dPreprocessContext,
    behavior: Any,
) -> dict[int, dict[str, np.ndarray]]:
    center_map_by_floor: dict[int, dict[str, np.ndarray]] = {}
    for floor in context.floorplans:
        floor_idx = int(getattr(floor, "level_index", 0))
        floor_graph = context.connectivity_graphs.get(floor_idx)
        if floor_graph is None:
            continue
        center_map = _compute_trajectory_room_centers(
            graph=floor_graph,
            floor_z=float(getattr(floor, "mean_height", 0.0)),
            behavior=behavior,
        )
        if center_map:
            center_map_by_floor[floor_idx] = center_map
    return center_map_by_floor


def _inject_trajectory_centers_into_geojson(
    geojson_path: Path,
    center_map_by_floor: dict[int, dict[str, np.ndarray]],
) -> None:
    """Persist trajectory room centers in GeoJSON for downstream visualization."""
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not isinstance(features, list):
        return

    center_features: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties", {})
        if props.get("type") != "room":
            continue

        room_id = str(props.get("room_id", ""))
        try:
            floor_idx = int(props.get("level_index", 0))
        except Exception:
            continue
        floor_centers = center_map_by_floor.get(floor_idx, {})
        center = floor_centers.get(room_id)
        if center is None:
            continue

        center_xyz = np.asarray(center, dtype=float).reshape(-1)
        if center_xyz.size < 3:
            continue
        center_xy = [float(center_xyz[0]), float(center_xyz[1])]
        center_3d = [float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])]
        props["trajectory_center_xy"] = center_xy
        props["trajectory_center_3d"] = center_3d

        center_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": center_xy},
                "properties": {
                    "type": "trajectory_room_center",
                    "room_id": room_id,
                    "level_index": floor_idx,
                },
            }
        )

    if center_features:
        features.extend(center_features)
        payload["features"] = features
        geojson_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_floor_geojson_files(
    context: Hl3dPreprocessContext,
    output_dir: Path,
    export_mod: Any,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for floor in context.floorplans:
        floor_idx = floor.level_index
        floor_rooms = None
        if floor_idx in context.rooms_by_floor:
            floor_rooms = {floor_idx: context.rooms_by_floor[floor_idx]}
        floor_openings = None
        if context.openings_by_floor and floor_idx in context.openings_by_floor:
            floor_openings = {floor_idx: context.openings_by_floor[floor_idx]}
        floor_graphs = None
        if floor_idx in context.connectivity_graphs:
            floor_graphs = {floor_idx: context.connectivity_graphs[floor_idx]}

        payload = export_mod.floorplans_to_geojson(
            [floor],
            floor_rooms,
            floor_openings,
            floor_graphs,
        )
        floor_path = output_dir / f"{context.scene}_floor_{floor_idx}.geojson"
        floor_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        outputs.append(floor_path)
    return tuple(outputs)


def _write_plot_files(
    context: Hl3dPreprocessContext,
    output_dir: Path,
    *,
    write_combined_plot: bool,
    write_floor_plots: bool,
    show_room_bboxes: bool,
    color_room_intersections: bool,
    show_connectivity: bool,
    behavior: Any,
    use_trajectory_centers: bool,
) -> tuple[Optional[Path], tuple[Path, ...]]:
    center_map_by_floor: dict[int, dict[str, np.ndarray]] = {}
    if use_trajectory_centers:
        center_map_by_floor = _compute_center_map_by_floor(context=context, behavior=behavior)

    return render_hl3d_debug_plots(
        scene=context.scene,
        floorplans=context.floorplans,
        rooms_by_floor=context.rooms_by_floor,
        openings_by_floor=context.openings_by_floor,
        connectivity_graphs=context.connectivity_graphs,
        output_dir=output_dir,
        center_map_by_floor=center_map_by_floor or None,
        write_combined_plot=write_combined_plot,
        write_floor_plots=write_floor_plots,
        show_room_bboxes=show_room_bboxes,
        color_room_intersections=color_room_intersections,
        show_connectivity=show_connectivity,
    )


def _write_diagnostics_artifacts(
    context: Hl3dPreprocessContext,
    output_dir: Path,
    export_mod: Any,
) -> tuple[Path, Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / f"{context.scene}_preprocess_diagnostics.json"
    diagnostics_payload = {
        "scene": context.scene,
        "num_floors": len(context.floorplans),
        "num_rooms": sum(len(v) for v in context.rooms_by_floor.values()),
        "num_connections": sum(
            len(getattr(graph, "connections", [])) for graph in context.connectivity_graphs.values()
        ),
        "num_stairs": len(context.stair_height_ranges),
        "stair_height_ranges": [list(pair) for pair in context.stair_height_ranges],
        "unmatched_room_ids": list(context.unmatched_room_ids),
        "unmatched_opening_ids": list(context.unmatched_opening_ids),
        "matched_to_floor_not_wall_opening_ids": list(context.matched_to_floor_not_wall_opening_ids),
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")

    room_stats_csv: Optional[Path] = None
    if hasattr(export_mod, "export_room_matching_stats") and context.rooms_by_floor:
        room_stats_csv = output_dir / f"{context.scene}_room_stats.csv"
        export_mod.export_room_matching_stats(context.rooms_by_floor, room_stats_csv)
    return diagnostics_path, room_stats_csv


def export_hl3d_matterport_debug_artifacts(
    config: TrajectoryGenerationConfig,
    output_dir: Path,
    project_root: Path | None = None,
    *,
    write_floor_geojson: bool = True,
    write_combined_plot: bool = True,
    write_floor_plots: bool = True,
    show_room_bboxes: bool = False,
    color_room_intersections: bool = True,
    show_connectivity: bool = True,
    use_trajectory_centers_for_debug: bool = True,
    write_diagnostics: bool = True,
) -> Hl3dDebugArtifacts:
    """Export debug artifacts (geojson/plots) for HL3D/Matterport preprocessing."""
    context, resolved_root = _build_hl3d_matterport_context(config=config, project_root=project_root)
    export_mod = _load_floorplan_export_module(resolved_root)
    center_map_by_floor = _compute_center_map_by_floor(
        context=context,
        behavior=config.walkthrough.behavior,
    )
    connectivity_geojson = _write_connectivity_geojson(
        context=context,
        output_path=output_dir / f"{context.scene}_connectivity.geojson",
        export_mod=export_mod,
        center_map_by_floor=center_map_by_floor,
    )

    floor_geojson_files: tuple[Path, ...] = ()
    if write_floor_geojson:
        floor_geojson_files = _write_floor_geojson_files(
            context=context,
            output_dir=output_dir,
            export_mod=export_mod,
        )

    combined_plot_file: Optional[Path] = None
    floor_plot_files: tuple[Path, ...] = ()
    if write_combined_plot or write_floor_plots:
        combined_plot_file, floor_plot_files = _write_plot_files(
            context=context,
            output_dir=output_dir,
            write_combined_plot=write_combined_plot,
            write_floor_plots=write_floor_plots,
            show_room_bboxes=show_room_bboxes,
            color_room_intersections=color_room_intersections,
            show_connectivity=show_connectivity,
            behavior=config.walkthrough.behavior,
            use_trajectory_centers=use_trajectory_centers_for_debug,
        )

    diagnostics_json_file: Optional[Path] = None
    room_matching_stats_csv: Optional[Path] = None
    if write_diagnostics:
        diagnostics_json_file, room_matching_stats_csv = _write_diagnostics_artifacts(
            context=context,
            output_dir=output_dir,
            export_mod=export_mod,
        )

    return Hl3dDebugArtifacts(
        connectivity_geojson=connectivity_geojson,
        floor_geojson_files=floor_geojson_files,
        combined_plot_file=combined_plot_file,
        floor_plot_files=floor_plot_files,
        diagnostics_json_file=diagnostics_json_file,
        room_matching_stats_csv=room_matching_stats_csv,
    )


def build_hl3d_matterport_connectivity_geojson(
    config: TrajectoryGenerationConfig,
    geojson_output_path: Path,
    project_root: Path | None = None,
) -> Path:
    """Build floorplan/connectivity GeoJSON from HouseLayout3D + Matterport inputs."""
    context, resolved_root = _build_hl3d_matterport_context(config=config, project_root=project_root)
    export_mod = _load_floorplan_export_module(resolved_root)
    center_map_by_floor = _compute_center_map_by_floor(
        context=context,
        behavior=config.walkthrough.behavior,
    )
    return _write_connectivity_geojson(
        context=context,
        output_path=geojson_output_path,
        export_mod=export_mod,
        center_map_by_floor=center_map_by_floor,
    )


def preprocess_hl3d_matterport_to_structural_json(
    config: TrajectoryGenerationConfig,
    structural_output_path: Path,
    geojson_output_path: Path | None = None,
    scene_id: str | None = None,
    project_root: Path | None = None,
    *,
    emit_debug_artifacts: bool = False,
    debug_output_dir: Path | None = None,
    write_floor_geojson: bool = True,
    write_combined_plot: bool = True,
    write_floor_plots: bool = True,
    show_room_bboxes: bool = False,
    color_room_intersections: bool = True,
    show_connectivity: bool = True,
    use_trajectory_centers_for_debug: bool = True,
    write_diagnostics: bool = True,
) -> tuple[Path, Path]:
    """Run end-to-end preprocessing from HL3D/Matterport inputs to structural JSON."""
    context, resolved_root = _build_hl3d_matterport_context(config=config, project_root=project_root)
    export_mod = _load_floorplan_export_module(resolved_root)
    center_map_by_floor = _compute_center_map_by_floor(
        context=context,
        behavior=config.walkthrough.behavior,
    )

    if geojson_output_path is None:
        geojson_output_path = structural_output_path.with_name(f"{context.scene}_connectivity.geojson")
    _write_connectivity_geojson(
        context=context,
        output_path=geojson_output_path,
        export_mod=export_mod,
        center_map_by_floor=center_map_by_floor,
    )

    if emit_debug_artifacts:
        debug_dir = debug_output_dir or geojson_output_path.parent
        if write_floor_geojson:
            _write_floor_geojson_files(
                context=context,
                output_dir=debug_dir,
                export_mod=export_mod,
            )
        if write_combined_plot or write_floor_plots:
            _write_plot_files(
                context=context,
                output_dir=debug_dir,
                write_combined_plot=write_combined_plot,
                write_floor_plots=write_floor_plots,
                show_room_bboxes=show_room_bboxes,
                color_room_intersections=color_room_intersections,
                show_connectivity=show_connectivity,
                behavior=config.walkthrough.behavior,
                use_trajectory_centers=use_trajectory_centers_for_debug,
            )
        if write_diagnostics:
            _write_diagnostics_artifacts(
                context=context,
                output_dir=debug_dir,
                export_mod=export_mod,
            )

    resolved_scene_id = scene_id or context.scene or None
    convert_connectivity_geojson_file(
        geojson_path=geojson_output_path,
        output_path=structural_output_path,
        scene_id=resolved_scene_id,
    )
    _inject_hl3d_metadata_into_structural_scene(
        structural_scene_path=structural_output_path,
        context=context,
    )
    return geojson_output_path, structural_output_path


def _opening_waypoint_from_context(opening: Any) -> list[float] | None:
    centroid = getattr(opening, "centroid_2d", None)
    if centroid is None:
        return None
    arr = np.asarray(centroid, dtype=float).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return [float(arr[0]), float(arr[1])]


def _opening_normal_from_context(opening: Any) -> list[float] | None:
    normal = getattr(opening, "normal_3d", None)
    if normal is None:
        return None
    arr = np.asarray(normal, dtype=float).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    norm_xy = float(np.linalg.norm(arr[:2]))
    if norm_xy <= 1e-9:
        return None
    return [float(arr[0] / norm_xy), float(arr[1] / norm_xy)]


def _opening_bbox_from_context(opening: Any) -> dict[str, list[float]] | None:
    vertices = getattr(opening, "vertices_3d", None)
    if vertices is None:
        return None
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] < 1 or not np.all(np.isfinite(arr[:, :3])):
        return None
    xyz_min = arr[:, :3].min(axis=0)
    xyz_max = arr[:, :3].max(axis=0)
    return {
        "min": [float(xyz_min[0]), float(xyz_min[1]), float(xyz_min[2])],
        "max": [float(xyz_max[0]), float(xyz_max[1]), float(xyz_max[2])],
    }


def _serialize_openings_from_context(context: Hl3dPreprocessContext) -> list[dict[str, Any]]:
    if not context.openings_by_floor:
        return []

    serialized: list[dict[str, Any]] = []
    for floor_idx, floor_openings in context.openings_by_floor.items():
        for entry in floor_openings:
            if not isinstance(entry, (tuple, list)) or not entry:
                continue
            opening = entry[0]
            opening_type = str(getattr(opening, "opening_type", "")).strip()
            if opening_type not in {"door", "window"}:
                continue

            item: dict[str, Any] = {
                "opening_type": opening_type,
                "floor_index": int(floor_idx),
            }
            opening_id = getattr(opening, "opening_id", None)
            if opening_id is not None:
                item["opening_id"] = opening_id

            waypoint_xy = _opening_waypoint_from_context(opening)
            if waypoint_xy is not None:
                item["waypoint_xy"] = waypoint_xy

            normal_xy = _opening_normal_from_context(opening)
            if normal_xy is not None:
                item["normal_xy"] = normal_xy

            bbox = _opening_bbox_from_context(opening)
            if bbox is not None:
                item["bbox"] = bbox

            if "waypoint_xy" in item or "bbox" in item:
                serialized.append(item)
    return serialized


def _opening_key(opening: dict[str, Any]) -> tuple[Any, ...]:
    waypoint = opening.get("waypoint_xy")
    if isinstance(waypoint, list) and len(waypoint) >= 2:
        wp = (round(float(waypoint[0]), 6), round(float(waypoint[1]), 6))
    else:
        wp = None
    return (
        opening.get("opening_type"),
        opening.get("opening_id"),
        opening.get("floor_index"),
        wp,
    )


def _serialize_stairs_from_context(context: Hl3dPreprocessContext) -> list[dict[str, Any]]:
    if not context.stair_height_ranges:
        return []

    floor_levels = sorted(
        (int(floor.level_index), float(floor.mean_height))
        for floor in context.floorplans
    )

    def _nearest_floor(z_value: float) -> int | None:
        if not floor_levels:
            return None
        return min(floor_levels, key=lambda item: abs(item[1] - z_value))[0]

    stairs: list[dict[str, Any]] = []
    for idx, (z_min, z_max) in enumerate(context.stair_height_ranges):
        stair_payload: dict[str, Any] = {
            "stair_id": idx,
            "z_min": float(z_min),
            "z_max": float(z_max),
        }
        from_floor = _nearest_floor(float(z_min))
        to_floor = _nearest_floor(float(z_max))
        if from_floor is not None:
            stair_payload["from_floor_index"] = int(from_floor)
        if to_floor is not None:
            stair_payload["to_floor_index"] = int(to_floor)
        stairs.append(stair_payload)
    return stairs


def _inject_hl3d_metadata_into_structural_scene(
    structural_scene_path: Path,
    context: Hl3dPreprocessContext,
) -> None:
    payload = json.loads(structural_scene_path.read_text(encoding="utf-8"))

    context_openings = _serialize_openings_from_context(context)
    existing_openings = payload.get("openings", [])
    merged_openings: list[dict[str, Any]] = []
    seen_openings: set[tuple[Any, ...]] = set()
    for opening in existing_openings if isinstance(existing_openings, list) else []:
        if not isinstance(opening, dict):
            continue
        key = _opening_key(opening)
        if key in seen_openings:
            continue
        seen_openings.add(key)
        merged_openings.append(opening)
    for opening in context_openings:
        key = _opening_key(opening)
        if key in seen_openings:
            continue
        seen_openings.add(key)
        merged_openings.append(opening)
    if merged_openings:
        payload["openings"] = merged_openings

    stairs = _serialize_stairs_from_context(context)
    if stairs:
        payload["stairs"] = stairs

    structural_scene_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compute_trajectory_room_centers(
    graph: Any,
    floor_z: float,
    behavior: Any,
) -> dict[str, np.ndarray]:
    try:
        if not hasattr(graph, "rooms") or not isinstance(graph.rooms, dict):
            return {}
        from .walkthrough_local import LocalWalkthroughGenerator

        walker = LocalWalkthroughGenerator(
            graph=graph,
            floor_z=floor_z,
            camera_height=1.6,
            behavior=behavior,
        )
        return {room_id: walker._get_room_center(room_id) for room_id in graph.rooms.keys()}
    except Exception:
        logger.debug("Failed to compute trajectory room centers for debug plot.", exc_info=True)
        return {}
