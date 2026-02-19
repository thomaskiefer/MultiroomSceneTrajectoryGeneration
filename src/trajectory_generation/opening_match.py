"""Helpers for matching door openings to room connections in structural scenes."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_xy(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        point = np.array([float(value[0]), float(value[1])], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def _as_floor_index(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opening_type(opening: dict[str, Any]) -> str:
    return str(opening.get("opening_type", opening.get("kind", ""))).strip().lower()


def opening_center_xy(opening: dict[str, Any]) -> np.ndarray | None:
    waypoint_xy = _as_xy(opening.get("waypoint_xy"))
    if waypoint_xy is not None:
        return waypoint_xy

    segment_xy = opening.get("segment_xy")
    if isinstance(segment_xy, list) and len(segment_xy) >= 2:
        p0 = _as_xy(segment_xy[0])
        p1 = _as_xy(segment_xy[1])
        if p0 is not None and p1 is not None:
            return (p0 + p1) / 2.0

    bbox = opening.get("bbox")
    if isinstance(bbox, dict):
        vmin = _as_xy(bbox.get("min"))
        vmax = _as_xy(bbox.get("max"))
        if vmin is not None and vmax is not None:
            return (vmin + vmax) / 2.0

    return None


def _opening_floor_index(opening: dict[str, Any], floor_levels: list[tuple[int, float]]) -> int | None:
    floor_index = _as_floor_index(opening.get("floor_index"))
    if floor_index is not None:
        return floor_index

    bbox = opening.get("bbox")
    if isinstance(bbox, dict):
        z_min_raw = None
        z_max_raw = None
        min_raw = bbox.get("min")
        max_raw = bbox.get("max")
        if isinstance(min_raw, (list, tuple)) and len(min_raw) >= 3:
            z_min_raw = min_raw[2]
        if isinstance(max_raw, (list, tuple)) and len(max_raw) >= 3:
            z_max_raw = max_raw[2]
        try:
            if z_min_raw is not None and z_max_raw is not None and floor_levels:
                z_center = (float(z_min_raw) + float(z_max_raw)) / 2.0
                return min(floor_levels, key=lambda item: abs(item[1] - z_center))[0]
        except (TypeError, ValueError):
            return None
    return None


def extract_door_centers_by_floor(
    scene_payload: dict[str, Any],
    floor_levels: list[tuple[int, float]],
) -> dict[int, list[np.ndarray]]:
    door_centers_by_floor: dict[int, list[np.ndarray]] = {}
    raw_openings = scene_payload.get("openings", [])
    if not isinstance(raw_openings, list):
        return door_centers_by_floor

    for opening in raw_openings:
        if not isinstance(opening, dict):
            continue
        if _opening_type(opening) != "door":
            continue
        center = opening_center_xy(opening)
        if center is None:
            continue
        floor_index = _opening_floor_index(opening, floor_levels)
        if floor_index is None:
            continue
        door_centers_by_floor.setdefault(floor_index, []).append(center)

    return door_centers_by_floor


def _point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = min(max(t, 0.0), 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def classify_connection_door_type(
    explicit_door_type: str | None,
    waypoint_xy: np.ndarray,
    room1_xy: np.ndarray,
    room2_xy: np.ndarray,
    door_centers: list[np.ndarray] | None,
    distance_threshold: float,
) -> str:
    if explicit_door_type is not None:
        normalized = str(explicit_door_type).strip().lower()
        if normalized in {"actual", "synthetic"}:
            return normalized

    if not door_centers:
        return "synthetic"

    threshold = max(float(distance_threshold), 1e-6)
    midpoint = (room1_xy + room2_xy) / 2.0
    for center in door_centers:
        if float(np.linalg.norm(center - waypoint_xy)) <= threshold:
            return "actual"
        if float(np.linalg.norm(center - midpoint)) <= threshold:
            return "actual"
        if _point_to_segment_distance(center, room1_xy, room2_xy) <= threshold:
            return "actual"

    return "synthetic"
