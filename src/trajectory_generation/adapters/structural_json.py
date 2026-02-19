"""Structural JSON adapter for reusable trajectory generation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
try:
    from shapely.geometry import Polygon as ShapelyPolygon
except ImportError:  # pragma: no cover - dependency is optional for polygon mode
    ShapelyPolygon = None

from ..artifacts import (
    FloorTrajectoryArtifact,
    TrajectoryGenerationArtifacts,
    write_trajectory_frames,
)
from ..config import TrajectoryGenerationConfig
from ..geometry_exceptions import geometry_exceptions
from ..opening_match import (
    classify_connection_door_type,
    extract_door_centers_by_floor,
)
from ..room_graph import (
    RectPolygon,
    RoomConnection,
    RoomGraph,
    RoomPolygon,
    RoomGraphRoomNode,
    RoomGraphWaypoint,
)
from ..schema import (
    ConnectionSpec as _ConnectionSpec,
    RoomSpec as _RoomSpec,
    parse_structural_scene_file,
)
from ..validation import validate_trajectory
from ..walkthrough_local import LocalWalkthroughGenerator


logger = logging.getLogger(__name__)
_GEOM_EXCEPTIONS = geometry_exceptions()


def _resolve_path(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _room_pair_distance_xy(room_a: _RoomSpec, room_b: _RoomSpec) -> float:
    dx = max(room_a.min_xy[0] - room_b.max_xy[0], room_b.min_xy[0] - room_a.max_xy[0], 0.0)
    dy = max(room_a.min_xy[1] - room_b.max_xy[1], room_b.min_xy[1] - room_a.max_xy[1], 0.0)
    return math.hypot(dx, dy)


def _unit_xy(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return None
    return vec / norm


def _iter_line_coordinate_sets(geometry: Any) -> list[np.ndarray]:
    if geometry is None:
        return []
    if getattr(geometry, "is_empty", False):
        return []
    coords_attr = None
    try:
        coords_attr = geometry.coords
    except (AttributeError, NotImplementedError):
        coords_attr = None
    if coords_attr is not None:
        coords = np.asarray(coords_attr, dtype=float)
        if coords.ndim == 2 and coords.shape[0] >= 2 and coords.shape[1] >= 2:
            return [coords[:, :2]]
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return []
    outputs: list[np.ndarray] = []
    for geom in geoms:
        outputs.extend(_iter_line_coordinate_sets(geom))
    return outputs


def _derive_polygon_connection(
    room_a: _RoomSpec,
    room_b: _RoomSpec,
    poly_a: Any,
    poly_b: Any,
    proximity_threshold: float,
) -> Optional[_ConnectionSpec]:
    if proximity_threshold < 0:
        return None
    buffered_a = poly_a.buffer(proximity_threshold)
    if not buffered_a.intersects(poly_b):
        return None

    overlap = buffered_a.intersection(poly_b)
    if overlap.is_empty:
        return None

    waypoint_xy = None
    try:
        overlap_centroid = overlap.centroid
        waypoint_xy = (float(overlap_centroid.x), float(overlap_centroid.y))
    except _GEOM_EXCEPTIONS:
        logger.warning(
            "Failed to compute overlap centroid for rooms (%s, %s); using centroid midpoint fallback.",
            room_a.room_id,
            room_b.room_id,
            exc_info=True,
        )

    c1 = room_a.centroid
    c2 = room_b.centroid
    direction = _unit_xy(np.array([float(c2[0] - c1[0]), float(c2[1] - c1[1])], dtype=float))
    if direction is None:
        direction = np.array([1.0, 0.0], dtype=float)

    normal_xy: Optional[tuple[float, float]] = None
    shared_boundary = buffered_a.boundary.intersection(poly_b.boundary)
    if getattr(shared_boundary, "is_empty", False):
        shared_boundary = poly_a.boundary.intersection(poly_b.boundary)
    coordinate_sets = _iter_line_coordinate_sets(shared_boundary)
    if coordinate_sets:
        best = max(
            coordinate_sets,
            key=lambda pts: float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))),
        )
        tangent = np.array([float(best[-1, 0] - best[0, 0]), float(best[-1, 1] - best[0, 1])], dtype=float)
        tangent_u = _unit_xy(tangent)
        if tangent_u is not None:
            normal = np.array([-tangent_u[1], tangent_u[0]], dtype=float)
            if float(np.dot(normal, direction)) < 0:
                normal = -normal
            normal_xy = (float(normal[0]), float(normal[1]))

    if normal_xy is None:
        normal_xy = (float(direction[0]), float(direction[1]))

    return _ConnectionSpec(
        room1_id=room_a.room_id,
        room2_id=room_b.room_id,
        waypoint_xy=waypoint_xy,
        normal_xy=normal_xy,
        door_type="synthetic",
    )


def _build_graph_for_floor(
    floor_rooms: list[_RoomSpec],
    explicit_connections: list[_ConnectionSpec],
    proximity_threshold: float,
    door_centers: list[np.ndarray] | None = None,
    door_match_tolerance: float = 0.3,
) -> RoomGraph:
    rooms_by_id = {room.room_id: room for room in floor_rooms}
    room_nodes: dict[str, tuple[RoomGraphRoomNode, RoomPolygon]] = {}
    shapely_polygons: dict[str, Any] = {}
    adjacency: dict[str, list[str]] = {}
    for room in floor_rooms:
        centroid = room.centroid
        room_node = RoomGraphRoomNode(
            room_id=room.room_id,
            label_semantic=room.semantic,
            centroid=np.array([float(centroid[0]), float(centroid[1]), float(centroid[2])]),
        )
        room_poly: RoomPolygon
        if room.polygon_xy is not None:
            if ShapelyPolygon is None:
                raise RuntimeError(
                    "polygon_xy requires shapely. Install shapely or omit polygon_xy to use bbox-derived rectangles."
                )
            polygon = ShapelyPolygon(room.polygon_xy)
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError(
                    f"Invalid polygon for room {room.room_id}: polygon_xy must be a valid non-zero area polygon."
                )
            room_poly = polygon
            shapely_polygons[room.room_id] = polygon
        else:
            room_poly = RectPolygon(room.min_xy, room.max_xy)

        room_nodes[room.room_id] = (
            room_node,
            room_poly,
        )
        adjacency[room.room_id] = []

    connection_specs: list[_ConnectionSpec] = []
    if explicit_connections:
        connection_specs = explicit_connections
    else:
        for i, room_a in enumerate(floor_rooms):
            for room_b in floor_rooms[i + 1 :]:
                poly_a = shapely_polygons.get(room_a.room_id)
                poly_b = shapely_polygons.get(room_b.room_id)
                if poly_a is not None and poly_b is not None:
                    derived = _derive_polygon_connection(
                        room_a=room_a,
                        room_b=room_b,
                        poly_a=poly_a,
                        poly_b=poly_b,
                        proximity_threshold=proximity_threshold,
                    )
                    if derived is not None:
                        connection_specs.append(derived)
                    continue
                if _room_pair_distance_xy(room_a, room_b) <= proximity_threshold:
                    connection_specs.append(
                        _ConnectionSpec(
                            room1_id=room_a.room_id,
                            room2_id=room_b.room_id,
                            waypoint_xy=None,
                            normal_xy=None,
                            door_type="synthetic",
                        )
                    )

    seen_pairs: set[frozenset[str]] = set()
    connections: list[RoomConnection] = []
    for conn in connection_specs:
        if conn.room1_id not in rooms_by_id or conn.room2_id not in rooms_by_id:
            continue
        key = frozenset((conn.room1_id, conn.room2_id))
        if conn.room1_id == conn.room2_id or key in seen_pairs:
            continue
        seen_pairs.add(key)

        room1 = rooms_by_id[conn.room1_id]
        room2 = rooms_by_id[conn.room2_id]
        c1 = room1.centroid
        c2 = room2.centroid

        if conn.waypoint_xy is None:
            waypoint_xy = np.array([(c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0], dtype=float)
        else:
            waypoint_xy = np.array([conn.waypoint_xy[0], conn.waypoint_xy[1]], dtype=float)

        normal_arr = None
        if conn.normal_xy is not None:
            normal_arr = np.array([conn.normal_xy[0], conn.normal_xy[1]], dtype=float)

        room1_xy = np.array([float(c1[0]), float(c1[1])], dtype=float)
        room2_xy = np.array([float(c2[0]), float(c2[1])], dtype=float)
        door_type = classify_connection_door_type(
            explicit_door_type=conn.door_type,
            waypoint_xy=waypoint_xy,
            room1_xy=room1_xy,
            room2_xy=room2_xy,
            door_centers=door_centers,
            distance_threshold=door_match_tolerance,
        )

        connections.append(
            RoomConnection(
                room1_id=conn.room1_id,
                room2_id=conn.room2_id,
                waypoint=RoomGraphWaypoint(
                    position=waypoint_xy,
                    normal=normal_arr,
                    door_type=door_type,
                ),
            )
        )
        adjacency[conn.room1_id].append(conn.room2_id)
        adjacency[conn.room2_id].append(conn.room1_id)

    for room_id in adjacency:
        adjacency[room_id] = sorted(set(adjacency[room_id]))

    return RoomGraph(rooms=room_nodes, adjacency=adjacency, connections=connections)


def _compute_graph_statistics(
    graph: RoomGraph,
    floor_level: int,
    explicit_pairs: Optional[set[frozenset[str]]] = None,
) -> dict[str, Any]:
    room_ids = sorted(graph.rooms.keys())
    degrees = [len(graph.adjacency.get(rid, [])) for rid in room_ids]
    isolated = [rid for rid in room_ids if len(graph.adjacency.get(rid, [])) == 0]
    num_rooms = len(room_ids)
    num_connections = len(graph.connections)
    explicit_pairs = explicit_pairs or set()
    actual_doors = sum(
        1
        for conn in graph.connections
        if frozenset((conn.room1_id, conn.room2_id)) in explicit_pairs
    )
    synthetic_doors = max(0, num_connections - actual_doors)
    avg_degree = float(sum(degrees) / num_rooms) if num_rooms > 0 else 0.0
    return {
        "floor_level": floor_level,
        "num_rooms": num_rooms,
        "num_connections": num_connections,
        "actual_doors": actual_doors,
        "synthetic_doors": synthetic_doors,
        "avg_degree": avg_degree,
        "max_degree": max(degrees) if degrees else 0,
        "isolated_rooms": isolated,
        "num_isolated": len(isolated),
    }


def run_structural_json(
    config: TrajectoryGenerationConfig,
    project_root: Optional[Path] = None,
) -> TrajectoryGenerationArtifacts:
    """Run trajectory generation from canonical structural JSON scene input."""
    root = (project_root or Path.cwd()).resolve()
    if config.dataset.scene_input_json is None:
        raise ValueError(
            "`dataset.scene_input_json` is required when workflow='structural_json'."
        )

    scene_input_path = _resolve_path(root, config.dataset.scene_input_json)
    if not scene_input_path.exists():
        raise FileNotFoundError(f"Structural scene JSON not found at {scene_input_path}")

    dataset_root = _resolve_path(root, config.dataset.dataset_root)
    output_dir = _resolve_path(root, config.walkthrough.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_payload = parse_structural_scene_file(scene_input_path)
    raw_scene_payload = json.loads(scene_input_path.read_text(encoding="utf-8"))
    scene_name = scene_payload.scene
    floors = scene_payload.floors
    rooms = scene_payload.rooms
    connections = scene_payload.connections
    scene = config.dataset.scene or scene_name

    artifacts = TrajectoryGenerationArtifacts(
        scene=scene,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )
    artifacts.warnings.extend(scene_payload.warnings)

    floors_by_idx = {f.floor_index: f for f in floors}
    floor_levels = sorted((int(f.floor_index), float(f.z)) for f in floors)
    door_centers_by_floor = extract_door_centers_by_floor(raw_scene_payload, floor_levels)
    rooms_by_floor: dict[int, list[_RoomSpec]] = {f.floor_index: [] for f in floors}
    for room in rooms:
        rooms_by_floor.setdefault(room.floor_index, []).append(room)

    explicit_by_floor: dict[int, list[_ConnectionSpec]] = {f.floor_index: [] for f in floors}
    if connections:
        room_floor = {room.room_id: room.floor_index for room in rooms}
        for conn in connections:
            if conn.room1_id not in room_floor or conn.room2_id not in room_floor:
                artifacts.warnings.append(
                    f"Skipped connection ({conn.room1_id}, {conn.room2_id}): unknown room id."
                )
                continue
            floor1 = room_floor[conn.room1_id]
            floor2 = room_floor[conn.room2_id]
            if floor1 != floor2:
                artifacts.warnings.append(
                    f"Skipped cross-floor connection ({conn.room1_id}, {conn.room2_id})."
                )
                continue
            explicit_by_floor[floor1].append(conn)

    floor_polygons: dict[int, Any] = {}
    if ShapelyPolygon is not None:
        for fi, fs in floors_by_idx.items():
            try:
                poly = ShapelyPolygon(fs.footprint_xy)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_valid:
                    raise ValueError("invalid floor footprint polygon")
                floor_polygons[fi] = poly
            except _GEOM_EXCEPTIONS:
                warning = (
                    f"[floor {fi}] Failed to build floor polygon; "
                    "floor-boundary validation disabled for this floor."
                )
                artifacts.warnings.append(warning)
                logger.warning(
                    "Failed to build floor polygon for floor %s; floor-boundary validation disabled for this floor.",
                    fi,
                )
            except ValueError:
                warning = (
                    f"[floor {fi}] Failed to build floor polygon; "
                    "floor-boundary validation disabled for this floor."
                )
                artifacts.warnings.append(warning)
                logger.warning(
                    "Failed to build floor polygon for floor %s; floor-boundary validation disabled for this floor.",
                    fi,
                )

    for floor_idx, floor in floors_by_idx.items():
        floor_rooms = rooms_by_floor.get(floor_idx, [])
        if not floor_rooms:
            artifacts.warnings.append(f"No rooms found for floor {floor_idx}; skipped trajectory.")
            continue

        floor_connections = explicit_by_floor.get(floor_idx, [])
        explicit_pairs = {
            frozenset((conn.room1_id, conn.room2_id))
            for conn in floor_connections
        }
        graph = _build_graph_for_floor(
            floor_rooms=floor_rooms,
            explicit_connections=floor_connections,
            proximity_threshold=config.connectivity.proximity_threshold,
            door_centers=door_centers_by_floor.get(floor_idx),
            door_match_tolerance=config.connectivity.door_match_tolerance,
        )
        stats = _compute_graph_statistics(
            graph,
            floor_level=floor_idx,
            explicit_pairs=explicit_pairs,
        )
        artifacts.connectivity_statistics[floor_idx] = stats

        walker = LocalWalkthroughGenerator(
            graph=graph,
            floor_z=floor.z,
            camera_height=config.walkthrough.camera_height,
            behavior=config.walkthrough.behavior,
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

        for warning in validate_trajectory(frames, floor_polygon=floor_polygons.get(floor_idx)):
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
                floor_z=floor.z,
                num_frames=len(frames),
                num_rooms=stats["num_rooms"],
                num_connections=stats["num_connections"],
                output_file=trajectory_file,
            )
        )

        if config.walkthrough.plot_output:
            try:
                from ..visualization import plot_trajectory
                floor_poly = floor_polygons.get(floor_idx)
                plot_trajectory(
                    frames=frames,
                    floor_polygon=floor_poly,
                    title=f"Floor {floor_idx} — {len(frames)} frames",
                    output_path=output_dir / f"{scene}_floor_{floor_idx}_trajectory_visualization.png",
                    viz_config=config.walkthrough.viz,
                )
            except ImportError:
                logger.warning("matplotlib not available; skipping trajectory plot.")

    if config.walkthrough.write_debug_summary:
        summary_path = output_dir / f"{scene}_trajectory_generation_summary.json"
        summary_path.write_text(json.dumps(artifacts.to_dict(), indent=2), encoding="utf-8")

    return artifacts
