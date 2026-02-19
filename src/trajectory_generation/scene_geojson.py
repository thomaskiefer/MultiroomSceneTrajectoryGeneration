"""Build visualization GeoJSON from structural scene inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .adapters.structural_json import _build_graph_for_floor
from .config import TrajectoryGenerationConfig
from .opening_match import extract_door_centers_by_floor
from .schema import parse_structural_scene_file
from .walkthrough_local import LocalWalkthroughGenerator


def _close_ring(points: list[list[float]]) -> list[list[float]]:
    if points and points[0] != points[-1]:
        return points + [points[0]]
    return points


def _as_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (x, y)


def _as_bbox(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(value, dict):
        return None
    min_raw = value.get("min")
    max_raw = value.get("max")
    if not isinstance(min_raw, (list, tuple)) or not isinstance(max_raw, (list, tuple)):
        return None
    if len(min_raw) < 3 or len(max_raw) < 3:
        return None
    try:
        vmin = np.array([float(min_raw[0]), float(min_raw[1]), float(min_raw[2])], dtype=float)
        vmax = np.array([float(max_raw[0]), float(max_raw[1]), float(max_raw[2])], dtype=float)
    except (TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(vmin)) and np.all(np.isfinite(vmax))):
        return None
    return (np.minimum(vmin, vmax), np.maximum(vmin, vmax))


def _as_floor_index(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_floor_index_from_bbox(
    bbox: tuple[np.ndarray, np.ndarray] | None,
    floor_levels: list[tuple[int, float]],
) -> int | None:
    if bbox is None or not floor_levels:
        return None
    z_center = float((bbox[0][2] + bbox[1][2]) * 0.5)
    return min(floor_levels, key=lambda item: abs(item[1] - z_center))[0]


def _opening_line_coordinates(opening: dict[str, Any]) -> list[list[float]] | None:
    opening_type = str(opening.get("opening_type", opening.get("kind", ""))).strip().lower()
    if opening_type not in {"door", "window"}:
        return None

    segment_xy = opening.get("segment_xy")
    if isinstance(segment_xy, list) and len(segment_xy) >= 2:
        seg_p0 = _as_xy(segment_xy[0])
        seg_p1 = _as_xy(segment_xy[1])
        if seg_p0 is not None and seg_p1 is not None:
            return [[float(seg_p0[0]), float(seg_p0[1])], [float(seg_p1[0]), float(seg_p1[1])]]

    bbox = _as_bbox(opening.get("bbox"))
    waypoint = _as_xy(opening.get("waypoint_xy"))
    normal = _as_xy(opening.get("normal_xy"))

    if waypoint is not None:
        center_xy = np.array([float(waypoint[0]), float(waypoint[1])], dtype=float)
    elif bbox is not None:
        center_xy = (bbox[0][:2] + bbox[1][:2]) / 2.0
    else:
        return None

    width = 1.0 if opening_type == "door" else 1.2
    if bbox is not None:
        span_xy = bbox[1][:2] - bbox[0][:2]
        width = max(width, float(np.max(span_xy)))

    if normal is not None:
        normal_vec = np.array([normal[0], normal[1]], dtype=float)
        n_norm = float(np.linalg.norm(normal_vec))
        if n_norm > 1e-9:
            normal_vec /= n_norm
            tangent = np.array([-normal_vec[1], normal_vec[0]], dtype=float)
        else:
            tangent = np.array([1.0, 0.0], dtype=float)
    elif bbox is not None:
        span_xy = bbox[1][:2] - bbox[0][:2]
        tangent = np.array([1.0, 0.0], dtype=float) if span_xy[0] >= span_xy[1] else np.array([0.0, 1.0], dtype=float)
    else:
        tangent = np.array([1.0, 0.0], dtype=float)

    t_norm = float(np.linalg.norm(tangent))
    if t_norm <= 1e-9:
        tangent = np.array([1.0, 0.0], dtype=float)
    else:
        tangent /= t_norm

    half = 0.5 * width
    p0 = center_xy - half * tangent
    p1 = center_xy + half * tangent
    return [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]


def _opening_to_feature(opening: dict[str, Any], floor_levels: list[tuple[int, float]]) -> dict[str, Any] | None:
    line_coords = _opening_line_coordinates(opening)
    if line_coords is None:
        return None

    opening_type = str(opening.get("opening_type", opening.get("kind", ""))).strip().lower()
    bbox = _as_bbox(opening.get("bbox"))
    floor_index = _as_floor_index(opening.get("floor_index"))
    if floor_index is None:
        floor_index = _infer_floor_index_from_bbox(bbox, floor_levels)

    props: dict[str, Any] = {"type": opening_type}
    if floor_index is not None:
        props["level_index"] = floor_index
    if opening.get("opening_id") is not None:
        props["opening_id"] = opening.get("opening_id")
    if bbox is not None:
        props["z_min"] = float(bbox[0][2])
        props["z_max"] = float(bbox[1][2])
    normal_xy = _as_xy(opening.get("normal_xy"))
    if normal_xy is not None:
        props["normal_xy"] = [float(normal_xy[0]), float(normal_xy[1])]
    for key in ("width", "height", "wall_id"):
        if key in opening and opening.get(key) is not None:
            props[key] = opening.get(key)

    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": line_coords},
        "properties": props,
    }


def _stair_to_feature(stair: dict[str, Any], floor_levels: list[tuple[int, float]]) -> dict[str, Any] | None:
    floor_index = _as_floor_index(stair.get("floor_index"))
    if floor_index is None:
        floor_index = _as_floor_index(stair.get("from_floor_index"))

    bbox = _as_bbox(stair.get("bbox"))
    polygon_xy = stair.get("polygon_xy")
    waypoint_xy = _as_xy(stair.get("waypoint_xy"))

    geometry: dict[str, Any] | None = None
    if isinstance(polygon_xy, list) and len(polygon_xy) >= 3:
        ring: list[list[float]] = []
        for point in polygon_xy:
            xy = _as_xy(point)
            if xy is None:
                ring = []
                break
            ring.append([float(xy[0]), float(xy[1])])
        if len(ring) >= 3:
            geometry = {"type": "Polygon", "coordinates": [_close_ring(ring)]}
    elif bbox is not None:
        x0, y0 = float(bbox[0][0]), float(bbox[0][1])
        x1, y1 = float(bbox[1][0]), float(bbox[1][1])
        geometry = {
            "type": "Polygon",
            "coordinates": [_close_ring([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])],
        }
        if floor_index is None:
            floor_index = _infer_floor_index_from_bbox(bbox, floor_levels)
    elif waypoint_xy is not None:
        geometry = {"type": "Point", "coordinates": [float(waypoint_xy[0]), float(waypoint_xy[1])]}

    if geometry is None:
        return None

    props: dict[str, Any] = {"type": "stairs"}
    for key in ("stair_id", "from_floor_index", "to_floor_index", "z_min", "z_max"):
        if key in stair and stair.get(key) is not None:
            props[key] = stair.get(key)
    if floor_index is not None:
        props["level_index"] = floor_index

    return {"type": "Feature", "geometry": geometry, "properties": props}


def build_connectivity_geojson_from_structural_scene(
    scene_input: Path,
    geojson_output: Path,
    config: TrajectoryGenerationConfig,
) -> Path:
    """Build connectivity GeoJSON aligned with trajectory-center logic."""
    raw_payload = json.loads(scene_input.read_text(encoding="utf-8"))
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
    for conn_spec in conns:
        if (
            conn_spec.room1_id in room_floor
            and conn_spec.room2_id in room_floor
            and room_floor[conn_spec.room1_id] == room_floor[conn_spec.room2_id]
        ):
            explicit_by_floor[room_floor[conn_spec.room1_id]].append(conn_spec)

    floor_levels = sorted((int(floor.floor_index), float(floor.z)) for floor in floors)
    door_centers_by_floor = extract_door_centers_by_floor(raw_payload, floor_levels)
    features: list[dict[str, Any]] = []
    for floor_idx, floor in sorted(floors_by_idx.items()):
        floor_rooms = rooms_by_floor.get(floor_idx, [])
        graph = _build_graph_for_floor(
            floor_rooms=floor_rooms,
            explicit_connections=explicit_by_floor.get(floor_idx, []),
            proximity_threshold=config.connectivity.proximity_threshold,
            door_centers=door_centers_by_floor.get(floor_idx),
            door_match_tolerance=config.connectivity.door_match_tolerance,
        )
        walker = LocalWalkthroughGenerator(
            graph=graph,
            floor_z=floor.z,
            camera_height=config.walkthrough.camera_height,
            behavior=config.walkthrough.behavior,
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

        for room_conn in graph.connections:
            c1 = center_map[room_conn.room1_id]
            c2 = center_map[room_conn.room2_id]
            wx = float(room_conn.waypoint.position[0])
            wy = float(room_conn.waypoint.position[1])

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(c1[0]), float(c1[1])], [wx, wy], [float(c2[0]), float(c2[1])]],
                    },
                    "properties": {
                        "type": "room_connection",
                        "room1_id": room_conn.room1_id,
                        "room2_id": room_conn.room2_id,
                        "door_type": getattr(room_conn.waypoint, "door_type", "synthetic"),
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
                        "room1_id": room_conn.room1_id,
                        "room2_id": room_conn.room2_id,
                        "door_type": getattr(room_conn.waypoint, "door_type", "synthetic"),
                        "level_index": int(floor_idx),
                    },
                }
            )

    raw_openings = raw_payload.get("openings", [])
    if isinstance(raw_openings, list):
        for raw_opening in raw_openings:
            if isinstance(raw_opening, dict):
                feature = _opening_to_feature(raw_opening, floor_levels=floor_levels)
                if feature is not None:
                    features.append(feature)

    raw_stairs = raw_payload.get("stairs", [])
    stairs_metadata: list[dict[str, Any]] = []
    if isinstance(raw_stairs, list):
        for raw_stair in raw_stairs:
            if not isinstance(raw_stair, dict):
                continue
            stairs_metadata.append(dict(raw_stair))
            feature = _stair_to_feature(raw_stair, floor_levels=floor_levels)
            if feature is not None:
                features.append(feature)

    geojson_payload: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if stairs_metadata:
        geojson_payload["metadata"] = {"stairs": stairs_metadata}

    geojson_output.parent.mkdir(parents=True, exist_ok=True)
    geojson_output.write_text(json.dumps(geojson_payload, indent=2), encoding="utf-8")
    return geojson_output
