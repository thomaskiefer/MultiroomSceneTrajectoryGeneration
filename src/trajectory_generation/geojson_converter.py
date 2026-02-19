"""GeoJSON -> canonical structural scene conversion helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _feature_type(feature: dict[str, Any]) -> str:
    return str(feature.get("properties", {}).get("type", ""))


def _ring_without_duplicate_close(ring: list[list[float]]) -> list[list[float]]:
    if len(ring) >= 2 and ring[0] == ring[-1]:
        return ring[:-1]
    return ring


def _pair_key(room1_id: str, room2_id: str) -> tuple[str, str]:
    return (room1_id, room2_id) if room1_id <= room2_id else (room2_id, room1_id)


def _extract_normal_xy(props: dict[str, Any]) -> list[float] | None:
    normal = props.get("normal_xy")
    if normal is None:
        normal = props.get("normal")
    if not isinstance(normal, (list, tuple)) or len(normal) < 2:
        return None
    try:
        return [float(normal[0]), float(normal[1])]
    except (TypeError, ValueError):
        return None


def _extract_level_index(props: dict[str, Any]) -> int | None:
    raw = props.get("level_index", props.get("floor_index"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _collect_xy_points(geometry: dict[str, Any]) -> list[list[float]]:
    gtype = str(geometry.get("type", ""))
    coords = geometry.get("coordinates")
    points: list[list[float]] = []
    if gtype == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        points.append([float(coords[0]), float(coords[1])])
    elif gtype == "LineString" and isinstance(coords, list):
        for p in coords:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append([float(p[0]), float(p[1])])
    elif gtype == "Polygon" and isinstance(coords, list) and coords:
        ring = coords[0]
        if isinstance(ring, list):
            for p in ring:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    points.append([float(p[0]), float(p[1])])
    return points


def _opening_segment_xy(geometry: dict[str, Any]) -> list[list[float]] | None:
    gtype = str(geometry.get("type", ""))
    coords = geometry.get("coordinates")
    if gtype != "LineString" or not isinstance(coords, list) or len(coords) < 2:
        return None
    p0 = coords[0]
    p1 = coords[-1]
    if not isinstance(p0, (list, tuple)) or not isinstance(p1, (list, tuple)):
        return None
    if len(p0) < 2 or len(p1) < 2:
        return None
    try:
        return [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]
    except (TypeError, ValueError):
        return None


def _opening_waypoint_xy(geometry: dict[str, Any]) -> list[float] | None:
    points = _collect_xy_points(geometry)
    if not points:
        return None
    arr = np.array(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    center = arr.mean(axis=0)
    if not np.all(np.isfinite(center)):
        return None
    return [float(center[0]), float(center[1])]


def _opening_bbox(
    geometry: dict[str, Any],
    props: dict[str, Any],
) -> dict[str, list[float]] | None:
    z_min_raw = props.get("z_min")
    z_max_raw = props.get("z_max")
    if z_min_raw is None or z_max_raw is None:
        return None
    points = _collect_xy_points(geometry)
    if not points:
        return None
    try:
        z_min = float(z_min_raw)
        z_max = float(z_max_raw)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(z_min) and math.isfinite(z_max)):
        return None
    arr = np.array(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    xy_min = arr.min(axis=0)
    xy_max = arr.max(axis=0)
    z_lo, z_hi = (z_min, z_max) if z_min <= z_max else (z_max, z_min)
    return {
        "min": [float(xy_min[0]), float(xy_min[1]), float(z_lo)],
        "max": [float(xy_max[0]), float(xy_max[1]), float(z_hi)],
    }


def convert_connectivity_geojson_payload(
    payload: dict[str, Any],
    scene_id: str,
) -> dict[str, Any]:
    """Convert connectivity GeoJSON payload into canonical scene.schema.v1 JSON."""
    features = payload.get("features", [])

    floor_features = [f for f in features if _feature_type(f) == "floor_footprint"]
    room_features = [f for f in features if _feature_type(f) == "room"]
    conn_features = [f for f in features if _feature_type(f) == "room_connection"]
    waypoint_features = [f for f in features if _feature_type(f) == "door_waypoint"]
    opening_features = [f for f in features if _feature_type(f) in {"door", "window"}]

    if not floor_features:
        raise ValueError("No floor_footprint features found in GeoJSON.")
    if not room_features:
        raise ValueError("No room features found in GeoJSON.")

    floors: list[dict[str, Any]] = []
    for floor in sorted(floor_features, key=lambda f: int(f.get("properties", {}).get("level_index", 0))):
        props = floor.get("properties", {})
        level_index = int(props.get("level_index", 0))
        z = float(props.get("mean_height", 0.0))

        geom = floor.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            raise ValueError(f"floor_footprint level {level_index} missing coordinates")
        outer_ring = coords[0]
        footprint_xy = _ring_without_duplicate_close([[float(p[0]), float(p[1])] for p in outer_ring])

        floors.append(
            {
                "floor_index": level_index,
                "z": z,
                "footprint_xy": footprint_xy,
            }
        )

    rooms: list[dict[str, Any]] = []
    for room in sorted(room_features, key=lambda f: str(f.get("properties", {}).get("room_id", ""))):
        props = room.get("properties", {})
        room_id = str(props.get("room_id", "")).strip()
        if not room_id:
            raise ValueError("room feature missing room_id")

        label_semantic = str(props.get("label_semantic", "other")).strip() or "other"
        level_index = int(props.get("level_index", 0))

        bbox_min = props.get("bbox_3d_min")
        bbox_max = props.get("bbox_3d_max")
        if bbox_min is None or bbox_max is None:
            raise ValueError(f"room {room_id} missing bbox_3d_min/bbox_3d_max")

        bbox_min = [float(v) for v in bbox_min]
        bbox_max = [float(v) for v in bbox_max]

        geom = room.get("geometry", {})
        coords = geom.get("coordinates", [])
        polygon_xy = None
        if coords:
            polygon_xy = _ring_without_duplicate_close(
                [[float(p[0]), float(p[1])] for p in coords[0]]
            )

        room_payload = {
            "room_id": room_id,
            "floor_index": level_index,
            "semantic": label_semantic,
            "bbox": {
                "min": bbox_min,
                "max": bbox_max,
            },
        }
        if polygon_xy and len(polygon_xy) >= 3:
            room_payload["polygon_xy"] = polygon_xy
        rooms.append(room_payload)

    waypoint_by_pair: dict[tuple[str, str], list[list[float]]] = {}
    for wp in waypoint_features:
        props = wp.get("properties", {})
        room1_id = str(props.get("room1_id", "")).strip()
        room2_id = str(props.get("room2_id", "")).strip()
        if not room1_id or not room2_id:
            continue
        pair = _pair_key(room1_id, room2_id)
        coords = wp.get("geometry", {}).get("coordinates")
        if not coords or len(coords) < 2:
            continue
        waypoint = [float(coords[0]), float(coords[1])]
        waypoint_by_pair.setdefault(pair, []).append(waypoint)

    connections: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for conn in conn_features:
        props = conn.get("properties", {})
        room1_id = str(props.get("room1_id", "")).strip()
        room2_id = str(props.get("room2_id", "")).strip()
        if not room1_id or not room2_id or room1_id == room2_id:
            continue

        pair = _pair_key(room1_id, room2_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        conn_payload: dict[str, Any] = {
            "room1_id": room1_id,
            "room2_id": room2_id,
        }
        waypoints = waypoint_by_pair.get(pair)
        if waypoints:
            conn_payload["waypoint_xy"] = waypoints[0]
        normal_xy = _extract_normal_xy(props)
        if normal_xy is not None:
            conn_payload["normal_xy"] = normal_xy
        door_type = props.get("door_type")
        if isinstance(door_type, str):
            parsed_door_type = door_type.strip().lower()
            if parsed_door_type in {"actual", "synthetic"}:
                conn_payload["door_type"] = parsed_door_type
        connections.append(conn_payload)

    openings: list[dict[str, Any]] = []
    for opening in opening_features:
        props = opening.get("properties", {})
        geometry = opening.get("geometry", {})
        opening_type = _feature_type(opening)
        if opening_type not in {"door", "window"}:
            continue

        waypoint_xy = _opening_waypoint_xy(geometry)
        bbox = _opening_bbox(geometry, props)
        if waypoint_xy is None and bbox is None:
            continue

        item: dict[str, Any] = {"opening_type": opening_type}
        segment_xy = _opening_segment_xy(geometry)
        if segment_xy is not None:
            item["segment_xy"] = segment_xy
        opening_id = props.get("opening_id")
        if opening_id is not None:
            item["opening_id"] = opening_id
        opening_floor_index = _extract_level_index(props)
        if opening_floor_index is not None:
            item["floor_index"] = opening_floor_index
        if waypoint_xy is not None:
            item["waypoint_xy"] = waypoint_xy
        if bbox is not None:
            item["bbox"] = bbox
        normal_xy = _extract_normal_xy(props)
        if normal_xy is not None:
            item["normal_xy"] = normal_xy
        width = props.get("width")
        if width is not None:
            try:
                item["width"] = float(width)
            except (TypeError, ValueError):
                pass
        height = props.get("height")
        if height is not None:
            try:
                item["height"] = float(height)
            except (TypeError, ValueError):
                pass
        wall_id = props.get("wall_id")
        if wall_id is not None:
            item["wall_id"] = wall_id
        openings.append(item)

    output = {
        "schema_version": "scene.schema.v1",
        "scene": scene_id,
        "floors": floors,
        "rooms": rooms,
        "connections": connections,
    }
    if openings:
        output["openings"] = openings
    return output


def convert_connectivity_geojson_file(
    geojson_path: Path,
    output_path: Path,
    scene_id: str | None = None,
) -> Path:
    """Convert connectivity GeoJSON file and write canonical scene.schema.v1 JSON."""
    payload = json.loads(geojson_path.read_text())
    if not scene_id:
        stem = geojson_path.stem
        scene_id = stem.split("_connectivity")[0]
    scene_json = convert_connectivity_geojson_payload(payload, scene_id=scene_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scene_json, indent=2))
    return output_path
