"""Template converter for structural-primitives input to scene.schema.v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .schema import STRUCTURAL_SCHEMA_VERSION


def _expect_type(value: Any, expected: type, path: str) -> None:
    if not isinstance(value, expected):
        raise ValueError(f"Invalid `{path}`: expected {expected.__name__}, got {type(value).__name__}.")


def _parse_vec(value: Any, path: str, size: int) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"Invalid `{path}`: expected list[{size}] of numbers.")
    try:
        vec = np.array([float(v) for v in value], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{path}`: values must be numeric.") from exc
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"Invalid `{path}`: values must be finite (not NaN/Inf).")
    return vec


def _pair_key(room1_id: str, room2_id: str) -> tuple[str, str]:
    return (room1_id, room2_id) if room1_id <= room2_id else (room2_id, room1_id)


def _normalize_xy(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return None
    return vec / norm


def _room_pair_distance_xy(room_a: dict[str, Any], room_b: dict[str, Any]) -> float:
    min_a = room_a["bbox"]["min"]
    max_a = room_a["bbox"]["max"]
    min_b = room_b["bbox"]["min"]
    max_b = room_b["bbox"]["max"]
    dx = max(min_a[0] - max_b[0], min_b[0] - max_a[0], 0.0)
    dy = max(min_a[1] - max_b[1], min_b[1] - max_a[1], 0.0)
    return float(np.hypot(dx, dy))


def _bbox_center_xyz(bbox: dict[str, list[float]]) -> np.ndarray:
    return (np.array(bbox["min"], dtype=float) + np.array(bbox["max"], dtype=float)) / 2.0


def _nearest_floor_index(z_center: float, floors: list[dict[str, Any]]) -> int:
    return min(floors, key=lambda f: abs(float(f["z"]) - float(z_center)))["floor_index"]


def _parse_floors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "floors" not in payload:
        raise ValueError("Missing required field `floors`.")
    raw_floors = payload["floors"]
    _expect_type(raw_floors, list, "floors")
    if not raw_floors:
        raise ValueError("`floors` must contain at least one floor.")

    floors: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for i, raw in enumerate(raw_floors):
        _expect_type(raw, dict, f"floors[{i}]")
        for key in ("floor_index", "z", "footprint_xy"):
            if key not in raw:
                raise ValueError(f"Missing required field `floors[{i}].{key}`.")
        floor_index = int(raw["floor_index"])
        if floor_index in seen_indices:
            raise ValueError(f"Duplicate floor index: {floor_index}.")
        seen_indices.add(floor_index)

        footprint_raw = raw["footprint_xy"]
        _expect_type(footprint_raw, list, f"floors[{i}].footprint_xy")
        if len(footprint_raw) < 3:
            raise ValueError(f"Invalid `floors[{i}].footprint_xy`: expected at least 3 points.")

        footprint_xy: list[list[float]] = []
        for j, p in enumerate(footprint_raw):
            v = _parse_vec(p, f"floors[{i}].footprint_xy[{j}]", 2)
            footprint_xy.append([float(v[0]), float(v[1])])

        floors.append(
            {
                "floor_index": floor_index,
                "z": float(raw["z"]),
                "footprint_xy": footprint_xy,
            }
        )
    floors.sort(key=lambda f: int(f["floor_index"]))
    return floors


def _parse_rooms(payload: dict[str, Any], floors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if "rooms" not in payload:
        raise ValueError("Missing required field `rooms`.")
    raw_rooms = payload["rooms"]
    _expect_type(raw_rooms, list, "rooms")
    if not raw_rooms:
        raise ValueError("`rooms` must contain at least one room.")

    rooms: list[dict[str, Any]] = []
    seen_room_ids: set[str] = set()
    for i, raw in enumerate(raw_rooms):
        _expect_type(raw, dict, f"rooms[{i}]")
        if "room_id" not in raw:
            raise ValueError(f"Missing required field `rooms[{i}].room_id`.")
        if "bbox" not in raw:
            raise ValueError(f"Missing required field `rooms[{i}].bbox`.")
        room_id = str(raw["room_id"]).strip()
        if not room_id:
            raise ValueError(f"Invalid `rooms[{i}].room_id`: must be non-empty.")
        if room_id in seen_room_ids:
            raise ValueError(f"Duplicate room_id: {room_id}.")
        seen_room_ids.add(room_id)

        bbox = raw["bbox"]
        _expect_type(bbox, dict, f"rooms[{i}].bbox")
        if "min" not in bbox or "max" not in bbox:
            raise ValueError(f"Missing `min` or `max` in rooms[{i}].bbox.")
        min_xyz = _parse_vec(bbox["min"], f"rooms[{i}].bbox.min", 3)
        max_xyz = _parse_vec(bbox["max"], f"rooms[{i}].bbox.max", 3)
        if np.any(max_xyz < min_xyz):
            raise ValueError(
                f"Invalid `rooms[{i}].bbox`: bbox.max must be >= bbox.min on all axes."
            )

        if "floor_index" in raw and raw["floor_index"] is not None:
            floor_index = int(raw["floor_index"])
        else:
            floor_index = _nearest_floor_index(float((min_xyz[2] + max_xyz[2]) / 2.0), floors)

        semantic = str(raw.get("semantic") or raw.get("type") or raw.get("label") or "other")
        room_payload: dict[str, Any] = {
            "room_id": room_id,
            "floor_index": floor_index,
            "semantic": semantic,
            "bbox": {
                "min": [float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])],
                "max": [float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2])],
            },
        }

        polygon_raw = raw.get("polygon_xy")
        if polygon_raw is not None:
            _expect_type(polygon_raw, list, f"rooms[{i}].polygon_xy")
            if len(polygon_raw) < 3:
                raise ValueError(f"Invalid `rooms[{i}].polygon_xy`: expected at least 3 points.")
            polygon_xy: list[list[float]] = []
            for j, p in enumerate(polygon_raw):
                v = _parse_vec(p, f"rooms[{i}].polygon_xy[{j}]", 2)
                polygon_xy.append([float(v[0]), float(v[1])])
            room_payload["polygon_xy"] = polygon_xy

        rooms.append(room_payload)
    rooms.sort(key=lambda r: str(r["room_id"]))
    return rooms


def _parse_explicit_connections(payload: dict[str, Any], room_ids: set[str]) -> list[dict[str, Any]]:
    raw_connections = payload.get("connections", [])
    _expect_type(raw_connections, list, "connections")
    connections: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, raw in enumerate(raw_connections):
        _expect_type(raw, dict, f"connections[{i}]")
        if "room1_id" not in raw or "room2_id" not in raw:
            raise ValueError(
                f"Missing required fields in `connections[{i}]`: expected room1_id, room2_id."
            )
        room1_id = str(raw["room1_id"]).strip()
        room2_id = str(raw["room2_id"]).strip()
        if room1_id == room2_id:
            continue
        if room1_id not in room_ids or room2_id not in room_ids:
            continue
        pair = _pair_key(room1_id, room2_id)
        if pair in seen:
            continue
        seen.add(pair)
        conn_payload: dict[str, Any] = {"room1_id": room1_id, "room2_id": room2_id}
        if raw.get("waypoint_xy") is not None:
            wp = _parse_vec(raw["waypoint_xy"], f"connections[{i}].waypoint_xy", 2)
            conn_payload["waypoint_xy"] = [float(wp[0]), float(wp[1])]
        if raw.get("normal_xy") is not None:
            n = _parse_vec(raw["normal_xy"], f"connections[{i}].normal_xy", 2)
            nxy = _normalize_xy(n)
            if nxy is not None:
                conn_payload["normal_xy"] = [float(nxy[0]), float(nxy[1])]
        connections.append(conn_payload)
    return connections


def _derive_connections_from_openings(
    payload: dict[str, Any],
    rooms: list[dict[str, Any]],
    opening_room_distance_threshold: float,
) -> list[dict[str, Any]]:
    raw_openings = payload.get("openings", [])
    _expect_type(raw_openings, list, "openings")
    if not raw_openings:
        return []

    room_centers: dict[str, np.ndarray] = {
        room["room_id"]: _bbox_center_xyz(room["bbox"]) for room in rooms
    }
    rooms_by_floor: dict[int, list[str]] = {}
    for room in rooms:
        rooms_by_floor.setdefault(int(room["floor_index"]), []).append(room["room_id"])

    connections: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, raw in enumerate(raw_openings):
        _expect_type(raw, dict, f"openings[{i}]")

        floor_index: Optional[int] = None
        if raw.get("floor_index") is not None:
            floor_index = int(raw["floor_index"])

        if raw.get("waypoint_xy") is not None:
            wp = _parse_vec(raw["waypoint_xy"], f"openings[{i}].waypoint_xy", 2)
            opening_center = np.array([float(wp[0]), float(wp[1])], dtype=float)
            opening_center_z = None
        elif raw.get("bbox") is not None:
            bbox = raw["bbox"]
            _expect_type(bbox, dict, f"openings[{i}].bbox")
            min_xyz = _parse_vec(bbox.get("min"), f"openings[{i}].bbox.min", 3)
            max_xyz = _parse_vec(bbox.get("max"), f"openings[{i}].bbox.max", 3)
            center = (min_xyz + max_xyz) / 2.0
            opening_center = center[:2]
            opening_center_z = float(center[2])
        else:
            continue

        if floor_index is None:
            # Infer from nearest room center z when opening bbox has z; otherwise skip.
            if opening_center_z is None:
                continue
            floor_index = min(
                (int(room["floor_index"]) for room in rooms),
                key=lambda fi: min(
                    abs(float(room_centers[rid][2]) - opening_center_z)
                    for rid in rooms_by_floor.get(fi, [])
                ),
            )

        candidates = rooms_by_floor.get(floor_index, [])
        if len(candidates) < 2:
            continue

        ranked: list[tuple[float, str]] = []
        for room_id in candidates:
            c = room_centers[room_id][:2]
            dist = float(np.linalg.norm(opening_center - c))
            if dist <= opening_room_distance_threshold:
                ranked.append((dist, room_id))
        ranked.sort(key=lambda x: (x[0], x[1]))
        if len(ranked) < 2:
            continue

        room1_id = ranked[0][1]
        room2_id = ranked[1][1]
        if room1_id == room2_id:
            continue

        pair = _pair_key(room1_id, room2_id)
        if pair in seen:
            continue
        seen.add(pair)

        conn_payload: dict[str, Any] = {
            "room1_id": pair[0],
            "room2_id": pair[1],
            "waypoint_xy": [float(opening_center[0]), float(opening_center[1])],
        }

        if raw.get("normal_xy") is not None:
            raw_normal = _parse_vec(raw["normal_xy"], f"openings[{i}].normal_xy", 2)
            normal_xy = _normalize_xy(raw_normal)
        else:
            c1 = room_centers[pair[0]][:2]
            c2 = room_centers[pair[1]][:2]
            normal_xy = _normalize_xy(np.array([float(c2[0] - c1[0]), float(c2[1] - c1[1])], dtype=float))

        if normal_xy is not None:
            conn_payload["normal_xy"] = [float(normal_xy[0]), float(normal_xy[1])]
        connections.append(conn_payload)

    return connections


def _derive_connections_from_bbox_proximity(
    rooms: list[dict[str, Any]],
    proximity_threshold: float,
) -> list[dict[str, Any]]:
    rooms_by_floor: dict[int, list[dict[str, Any]]] = {}
    for room in rooms:
        rooms_by_floor.setdefault(int(room["floor_index"]), []).append(room)

    connections: list[dict[str, Any]] = []
    for floor_rooms in rooms_by_floor.values():
        floor_rooms = sorted(floor_rooms, key=lambda r: str(r["room_id"]))
        for i, room_a in enumerate(floor_rooms):
            for room_b in floor_rooms[i + 1 :]:
                dist = _room_pair_distance_xy(room_a, room_b)
                if dist > proximity_threshold:
                    continue
                center_a = _bbox_center_xyz(room_a["bbox"])
                center_b = _bbox_center_xyz(room_b["bbox"])
                normal_xy = _normalize_xy(np.array([center_b[0] - center_a[0], center_b[1] - center_a[1]]))
                midpoint = (center_a + center_b) / 2.0
                conn_payload: dict[str, Any] = {
                    "room1_id": str(room_a["room_id"]),
                    "room2_id": str(room_b["room_id"]),
                    "waypoint_xy": [float(midpoint[0]), float(midpoint[1])],
                }
                if normal_xy is not None:
                    conn_payload["normal_xy"] = [float(normal_xy[0]), float(normal_xy[1])]
                connections.append(conn_payload)
    return connections


def convert_structural_primitives_payload(
    payload: dict[str, Any],
    *,
    scene_id: str | None = None,
    proximity_threshold: float = 0.25,
    opening_room_distance_threshold: float = 3.0,
) -> dict[str, Any]:
    """Convert neutral structural-primitives payload into scene.schema.v1 JSON.

    Input template (minimal):
    - scene: str (optional if scene_id argument is provided)
    - floors[]: floor_index, z, footprint_xy
    - rooms[]: room_id, semantic (optional), bbox.min/max, floor_index (optional), polygon_xy (optional)
    - connections[] (optional): room1_id, room2_id, waypoint_xy (optional), normal_xy (optional)
    - openings[] (optional): floor_index (optional), waypoint_xy or bbox, normal_xy (optional)
    """
    _expect_type(payload, dict, "root")
    floors = _parse_floors(payload)
    rooms = _parse_rooms(payload, floors=floors)

    resolved_scene = scene_id or str(payload.get("scene", "")).strip()
    if not resolved_scene:
        raise ValueError("Missing `scene`. Provide `scene` in payload or `scene_id` argument.")

    room_ids = {str(room["room_id"]) for room in rooms}
    explicit_connections = _parse_explicit_connections(payload, room_ids=room_ids)
    if explicit_connections:
        connections = explicit_connections
    else:
        opening_connections = _derive_connections_from_openings(
            payload=payload,
            rooms=rooms,
            opening_room_distance_threshold=opening_room_distance_threshold,
        )
        if opening_connections:
            connections = opening_connections
        else:
            connections = _derive_connections_from_bbox_proximity(
                rooms=rooms,
                proximity_threshold=proximity_threshold,
            )

    return {
        "schema_version": STRUCTURAL_SCHEMA_VERSION,
        "scene": resolved_scene,
        "floors": floors,
        "rooms": rooms,
        "connections": connections,
    }


def convert_structural_primitives_file(
    input_path: Path,
    output_path: Path,
    *,
    scene_id: str | None = None,
    proximity_threshold: float = 0.25,
    opening_room_distance_threshold: float = 3.0,
) -> Path:
    """Convert structural-primitives JSON file and write scene.schema.v1 JSON."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    scene_json = convert_structural_primitives_payload(
        payload=payload,
        scene_id=scene_id,
        proximity_threshold=proximity_threshold,
        opening_room_distance_threshold=opening_room_distance_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scene_json, indent=2), encoding="utf-8")
    return output_path
