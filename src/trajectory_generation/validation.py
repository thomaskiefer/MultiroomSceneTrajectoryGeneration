"""Trajectory post-generation validation helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .geometry_exceptions import geometry_exceptions

try:
    from shapely.geometry import Point as ShapelyPoint
except ImportError:  # pragma: no cover
    ShapelyPoint = None

logger = logging.getLogger(__name__)
_GEOM_EXCEPTIONS = geometry_exceptions(include_runtime=True)


def _as_vec3(frame: dict[str, Any], key: str) -> Optional[np.ndarray]:
    raw = frame.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (3,):
        return None
    return arr


def validate_trajectory(
    frames: list[dict[str, Any]],
    floor_polygon: Any = None,
) -> list[str]:
    """
    Validate generated frame payloads and return warning strings.

    The function is non-throwing and intended for debug/QA hardening.
    """
    warnings: list[str] = []
    if not frames:
        return warnings

    invalid_shape_count = 0
    non_finite_count = 0
    non_unit_forward_count = 0
    non_orthogonal_up_count = 0
    outside_floor_count = 0
    floor_check_error_count = 0

    check_floor = (
        floor_polygon is not None
        and ShapelyPoint is not None
        and hasattr(floor_polygon, "covers")
    )
    if floor_polygon is not None and ShapelyPoint is None:
        warnings.append("Floor-boundary validation skipped: shapely is not installed.")
    elif floor_polygon is not None and not hasattr(floor_polygon, "covers"):
        warnings.append("Floor-boundary validation skipped: floor polygon object lacks a `covers` method.")

    for frame in frames:
        position = _as_vec3(frame, "position")
        look_at = _as_vec3(frame, "look_at")
        forward = _as_vec3(frame, "forward")
        up = _as_vec3(frame, "up")
        if position is None or look_at is None or forward is None or up is None:
            invalid_shape_count += 1
            continue

        stacked = np.concatenate([position, look_at, forward, up])
        if not np.all(np.isfinite(stacked)):
            non_finite_count += 1
            continue

        forward_norm = float(np.linalg.norm(forward))
        if abs(forward_norm - 1.0) > 0.05:
            non_unit_forward_count += 1

        up_norm = float(np.linalg.norm(up))
        if up_norm > 1e-8:
            up_u = up / up_norm
            forward_u = forward / max(forward_norm, 1e-8)
            if abs(float(np.dot(up_u, forward_u))) > 0.1:
                non_orthogonal_up_count += 1

        if check_floor:
            pt = ShapelyPoint(float(position[0]), float(position[1]))
            try:
                if not bool(floor_polygon.covers(pt)):
                    outside_floor_count += 1
            except _GEOM_EXCEPTIONS:
                floor_check_error_count += 1
                logger.debug(
                    "Floor-boundary validation failed on frame id=%s; skipping this frame.",
                    frame.get("id", "?"),
                    exc_info=True,
                )

    if invalid_shape_count:
        warnings.append(
            f"{invalid_shape_count} frame(s) missing required 3D vectors "
            "(position/look_at/forward/up)."
        )
    if non_finite_count:
        warnings.append(f"{non_finite_count} frame(s) contain NaN/Inf values.")
    if non_unit_forward_count:
        warnings.append(
            f"{non_unit_forward_count} frame(s) have non-unit forward vectors (|norm-1| > 0.05)."
        )
    if non_orthogonal_up_count:
        warnings.append(
            f"{non_orthogonal_up_count} frame(s) have up vectors not perpendicular to forward."
        )
    if outside_floor_count:
        warnings.append(f"{outside_floor_count} frame(s) lie outside the floor footprint.")
    if floor_check_error_count:
        warnings.append(
            f"Floor-boundary validation failed {floor_check_error_count} time(s) due to floor geometry errors."
        )
    return warnings
