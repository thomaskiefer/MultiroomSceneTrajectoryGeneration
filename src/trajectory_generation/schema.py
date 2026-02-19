"""Canonical structural scene schema parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


STRUCTURAL_SCHEMA_VERSION = "scene.schema.v1"
SUPPORTED_STRUCTURAL_SCHEMA_VERSIONS = (STRUCTURAL_SCHEMA_VERSION,)


@dataclass(frozen=True)
class FloorSpec:
    floor_index: int
    z: float
    footprint_xy: list[tuple[float, float]]


@dataclass(eq=False)
class RoomSpec:
    room_id: str
    floor_index: int
    semantic: str
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    polygon_xy: Optional[list[tuple[float, float]]] = None

    @property
    def centroid(self) -> np.ndarray:
        return (self.min_xyz + self.max_xyz) / 2.0

    @property
    def min_xy(self) -> tuple[float, float]:
        return (float(self.min_xyz[0]), float(self.min_xyz[1]))

    @property
    def max_xy(self) -> tuple[float, float]:
        return (float(self.max_xyz[0]), float(self.max_xyz[1]))


@dataclass(frozen=True)
class ConnectionSpec:
    room1_id: str
    room2_id: str
    waypoint_xy: Optional[tuple[float, float]]
    normal_xy: Optional[tuple[float, float]]
    door_type: Optional[str]


@dataclass(frozen=True)
class StructuralScene:
    schema_version: str
    scene: str
    floors: list[FloorSpec]
    rooms: list[RoomSpec]
    connections: list[ConnectionSpec]
    warnings: list[str] = field(default_factory=list)


def _expect_type(value: Any, expected: type, path: str) -> None:
    if not isinstance(value, expected):
        raise ValueError(f"Invalid `{path}`: expected {expected.__name__}, got {type(value).__name__}.")


def _parse_vec(value: Any, path: str, size: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"Invalid `{path}`: expected list[{size}] of numbers.")
    try:
        arr = np.array([float(v) for v in value], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{path}`: values must be numeric.") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Invalid `{path}`: values must be finite (no NaN/Inf).")
    return arr


def _parse_int_index(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid `{path}`: expected integer, got bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    raise ValueError(f"Invalid `{path}`: expected integer value.")


def _polygon_area_xy(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area2 += (x1 * y2) - (x2 * y1)
    return abs(area2) * 0.5


def _parse_structural_payload(payload: dict[str, Any]) -> StructuralScene:
    warnings: list[str] = []

    schema_version = payload.get("schema_version")
    if schema_version is None:
        schema_version = STRUCTURAL_SCHEMA_VERSION
        warnings.append(
            "Missing `schema_version`; assuming scene.schema.v1. "
            "Please add `schema_version` explicitly."
        )
    _expect_type(schema_version, str, "schema_version")
    if schema_version not in SUPPORTED_STRUCTURAL_SCHEMA_VERSIONS:
        allowed = ", ".join(SUPPORTED_STRUCTURAL_SCHEMA_VERSIONS)
        raise ValueError(
            f"Unsupported `schema_version`: {schema_version!r}. Supported values: {allowed}."
        )

    if "scene" not in payload:
        raise ValueError("Missing required field `scene`.")
    if "floors" not in payload:
        raise ValueError("Missing required field `floors`.")
    if "rooms" not in payload:
        raise ValueError("Missing required field `rooms`.")

    scene = payload["scene"]
    _expect_type(scene, str, "scene")
    if not scene.strip():
        raise ValueError("Invalid `scene`: must be non-empty.")

    raw_floors = payload["floors"]
    _expect_type(raw_floors, list, "floors")
    if not raw_floors:
        raise ValueError("Structural scene has no floors.")

    floors: list[FloorSpec] = []
    floor_map: dict[int, FloorSpec] = {}
    for i, raw in enumerate(raw_floors):
        _expect_type(raw, dict, f"floors[{i}]")
        if "floor_index" not in raw or "z" not in raw or "footprint_xy" not in raw:
            raise ValueError(
                f"Missing required fields in `floors[{i}]`: expected floor_index, z, footprint_xy."
            )
        floor_index = _parse_int_index(raw["floor_index"], f"floors[{i}].floor_index")
        z = float(raw["z"])
        if not np.isfinite(z):
            raise ValueError(f"Invalid `floors[{i}].z`: value must be finite (no NaN/Inf).")
        footprint_raw = raw["footprint_xy"]
        _expect_type(footprint_raw, list, f"floors[{i}].footprint_xy")
        if len(footprint_raw) < 3:
            raise ValueError(f"Invalid `floors[{i}].footprint_xy`: expected at least 3 points.")
        footprint_xy: list[tuple[float, float]] = []
        for j, p in enumerate(footprint_raw):
            vec = _parse_vec(p, f"floors[{i}].footprint_xy[{j}]", 2)
            footprint_xy.append((float(vec[0]), float(vec[1])))
        floor_spec = FloorSpec(floor_index=floor_index, z=z, footprint_xy=footprint_xy)
        if floor_index in floor_map:
            raise ValueError(f"Duplicate floor_index in floors: {floor_index}.")
        floor_map[floor_index] = floor_spec
        floors.append(floor_spec)

    raw_rooms = payload["rooms"]
    _expect_type(raw_rooms, list, "rooms")
    if not raw_rooms:
        raise ValueError("Structural scene has no rooms.")

    def _nearest_floor_index(z_center: float) -> int:
        return min(floor_map.keys(), key=lambda idx: abs(floor_map[idx].z - z_center))

    rooms: list[RoomSpec] = []
    seen_room_ids: set[str] = set()
    for i, raw in enumerate(raw_rooms):
        _expect_type(raw, dict, f"rooms[{i}]")
        for required_key in ("room_id", "semantic", "bbox"):
            if required_key not in raw:
                raise ValueError(f"Missing required field `rooms[{i}].{required_key}`.")

        room_id = raw["room_id"]
        semantic = raw["semantic"]
        _expect_type(room_id, str, f"rooms[{i}].room_id")
        _expect_type(semantic, str, f"rooms[{i}].semantic")
        room_id = room_id.strip()
        semantic = semantic.strip()
        if not room_id:
            raise ValueError(f"Invalid `rooms[{i}].room_id`: must be non-empty.")
        if not semantic:
            raise ValueError(f"Invalid `rooms[{i}].semantic`: must be non-empty.")
        if room_id in seen_room_ids:
            raise ValueError(f"Duplicate room_id in rooms: {room_id}.")
        seen_room_ids.add(room_id)

        polygon_xy = None
        if "polygon_xy" in raw and raw["polygon_xy"] is not None:
            polygon_raw = raw["polygon_xy"]
            _expect_type(polygon_raw, list, f"rooms[{i}].polygon_xy")
            if len(polygon_raw) < 3:
                raise ValueError(f"Invalid `rooms[{i}].polygon_xy`: expected at least 3 points.")
            polygon_xy = []
            for j, p in enumerate(polygon_raw):
                vec = _parse_vec(p, f"rooms[{i}].polygon_xy[{j}]", 2)
                polygon_xy.append((float(vec[0]), float(vec[1])))
            if _polygon_area_xy(polygon_xy) <= 1e-9:
                raise ValueError(
                    f"Invalid `rooms[{i}].polygon_xy`: polygon must have non-zero area."
                )

        bbox = raw["bbox"]
        _expect_type(bbox, dict, f"rooms[{i}].bbox")
        if "min" not in bbox or "max" not in bbox:
            raise ValueError(f"Missing `min` or `max` in rooms[{i}].bbox.")
        min_xyz = _parse_vec(bbox["min"], f"rooms[{i}].bbox.min", 3)
        max_xyz = _parse_vec(bbox["max"], f"rooms[{i}].bbox.max", 3)
        if np.any(max_xyz < min_xyz):
            raise ValueError(f"Invalid bbox in rooms[{i}]: bbox.max must be >= bbox.min on all axes.")
        if max_xyz[0] <= min_xyz[0] or max_xyz[1] <= min_xyz[1]:
            raise ValueError(
                f"Invalid bbox in rooms[{i}]: x/y extents must be strictly positive."
            )

        if "floor_index" in raw and raw["floor_index"] is not None:
            floor_index = _parse_int_index(raw["floor_index"], f"rooms[{i}].floor_index")
            if floor_index not in floor_map:
                raise ValueError(f"rooms[{i}].floor_index={floor_index} not found in floors.")
        else:
            floor_index = _nearest_floor_index(float((min_xyz[2] + max_xyz[2]) / 2.0))

        rooms.append(
            RoomSpec(
                room_id=room_id,
                floor_index=floor_index,
                semantic=semantic,
                min_xyz=min_xyz,
                max_xyz=max_xyz,
                polygon_xy=polygon_xy,
            )
        )

    raw_connections = payload.get("connections", [])
    _expect_type(raw_connections, list, "connections")
    connections: list[ConnectionSpec] = []
    seen_connection_pairs: set[tuple[str, str]] = set()
    for i, raw in enumerate(raw_connections):
        _expect_type(raw, dict, f"connections[{i}]")
        for required_key in ("room1_id", "room2_id"):
            if required_key not in raw:
                raise ValueError(f"Missing required field `connections[{i}].{required_key}`.")
        room1_id = raw["room1_id"]
        room2_id = raw["room2_id"]
        _expect_type(room1_id, str, f"connections[{i}].room1_id")
        _expect_type(room2_id, str, f"connections[{i}].room2_id")
        room1_id = room1_id.strip()
        room2_id = room2_id.strip()
        if not room1_id or not room2_id:
            warnings.append(
                f"Ignoring connections[{i}] with empty room id(s): {room1_id!r}, {room2_id!r}."
            )
            continue
        if room1_id == room2_id:
            warnings.append(
                f"Ignoring connections[{i}] self-connection for room_id={room1_id!r}."
            )
            continue
        if room1_id not in seen_room_ids or room2_id not in seen_room_ids:
            warnings.append(
                f"Ignoring connections[{i}] referencing unknown room ids: "
                f"{room1_id!r}, {room2_id!r}."
            )
            continue
        pair = tuple(sorted((room1_id, room2_id)))
        if pair in seen_connection_pairs:
            warnings.append(
                f"Ignoring duplicate connections[{i}] for pair: {pair[0]!r}, {pair[1]!r}."
            )
            continue
        seen_connection_pairs.add(pair)
        waypoint_xy = None
        normal_xy = None
        door_type = None
        if "waypoint_xy" in raw and raw["waypoint_xy"] is not None:
            wp = _parse_vec(raw["waypoint_xy"], f"connections[{i}].waypoint_xy", 2)
            waypoint_xy = (float(wp[0]), float(wp[1]))
        if "normal_xy" in raw and raw["normal_xy"] is not None:
            n = _parse_vec(raw["normal_xy"], f"connections[{i}].normal_xy", 2)
            normal_xy = (float(n[0]), float(n[1]))
        if "door_type" in raw and raw["door_type"] is not None:
            _expect_type(raw["door_type"], str, f"connections[{i}].door_type")
            parsed_door_type = str(raw["door_type"]).strip().lower()
            if parsed_door_type not in {"actual", "synthetic"}:
                raise ValueError(
                    f"Invalid `connections[{i}].door_type`: expected 'actual' or 'synthetic'."
                )
            door_type = parsed_door_type
        connections.append(
            ConnectionSpec(
                room1_id=room1_id,
                room2_id=room2_id,
                waypoint_xy=waypoint_xy,
                normal_xy=normal_xy,
                door_type=door_type,
            )
        )

    return StructuralScene(
        schema_version=schema_version,
        scene=scene,
        floors=floors,
        rooms=rooms,
        connections=connections,
        warnings=warnings,
    )


def parse_structural_scene_file(scene_path: Path) -> StructuralScene:
    try:
        payload = json.loads(scene_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid structural scene JSON at {scene_path}: {exc}") from exc
    _expect_type(payload, dict, "root")
    return _parse_structural_payload(payload)


def validate_structural_scene_file(scene_path: Path) -> dict[str, Any]:
    scene = parse_structural_scene_file(scene_path)
    return {
        "schema_version": scene.schema_version,
        "scene": scene.scene,
        "num_floors": len(scene.floors),
        "num_rooms": len(scene.rooms),
        "num_connections": len(scene.connections),
        "warnings": list(scene.warnings),
    }
