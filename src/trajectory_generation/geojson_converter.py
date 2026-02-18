"""GeoJSON -> canonical structural scene conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
        connections.append(conn_payload)

    return {
        "schema_version": "scene.schema.v1",
        "scene": scene_id,
        "floors": floors,
        "rooms": rooms,
        "connections": connections,
    }


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
