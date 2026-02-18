"""Debug floorplan visualization for preprocessing artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised in integration env
    raise RuntimeError("matplotlib is required for debug visualization") from exc


_FLOOR_FILL = "#f5f5f5"
_FLOOR_EDGE = "#333333"
_ROOM_EDGE = "#555555"
_ROOM_ALPHA = 0.35
_DOOR_COLOR = "#c0392b"
_WINDOW_COLOR = "#2980b9"
_ACTUAL_CONN_COLOR = "#1f6f8b"
_INFERRED_CONN_COLOR = "#1f6f8b"

_ROOM_COLORS = {
    "bathroom": "#a8d8ea",
    "bedroom": "#b8d4e3",
    "kitchen": "#f7c59f",
    "living_room": "#b5d8a0",
    "hallway": "#e0e0e0",
    "entryway": "#f0e68c",
    "other": "#e0e0e0",
    "unlabeled": "#eeeeee",
}


def _iter_polygons(geom: Any) -> list[Any]:
    if geom is None:
        return []
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        return [geom]
    if geom_type == "MultiPolygon":
        return list(getattr(geom, "geoms", []))
    return []


def _draw_floor(ax: Any, floor: Any) -> None:
    for poly in _iter_polygons(getattr(floor, "footprint", None)):
        x, y = poly.exterior.xy
        ax.fill(x, y, facecolor=_FLOOR_FILL, edgecolor=_FLOOR_EDGE, linewidth=1.5, zorder=1)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, facecolor="white", edgecolor=_FLOOR_EDGE, linewidth=1.0, zorder=2)

    outer_shell = getattr(floor, "outer_shell", None)
    for poly in _iter_polygons(outer_shell):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=_FLOOR_EDGE, linewidth=1.6, zorder=3)


def _draw_rooms(
    ax: Any,
    room_data: list[Any] | None,
    center_map: dict[str, np.ndarray] | None,
    color_room_intersections: bool,
    show_room_bboxes: bool,
) -> None:
    if not room_data:
        return
    for item in room_data:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        room = item[0]
        intersection_poly = item[1]
        label = str(getattr(room, "label_semantic", "other"))
        color = _ROOM_COLORS.get(label, _ROOM_COLORS["other"])
        if color_room_intersections:
            for poly in _iter_polygons(intersection_poly):
                x, y = poly.exterior.xy
                ax.fill(x, y, facecolor=color, edgecolor="none", alpha=_ROOM_ALPHA, zorder=4)
        for poly in _iter_polygons(intersection_poly):
            x, y = poly.exterior.xy
            ax.plot(x, y, color=_ROOM_EDGE, linewidth=0.9, zorder=5)

        room_id = str(getattr(room, "room_id", ""))
        label_xy: Optional[np.ndarray] = None
        if center_map and room_id in center_map:
            label_xy = np.asarray(center_map[room_id][:2], dtype=float)
        else:
            centroid = getattr(room, "centroid", None)
            if centroid is not None and len(centroid) >= 2:
                label_xy = np.array([float(centroid[0]), float(centroid[1])], dtype=float)
        if label_xy is not None:
            ax.text(
                float(label_xy[0]),
                float(label_xy[1]),
                label.replace("_", " ").title(),
                fontsize=8,
                color="#4b4b4b",
                ha="center",
                va="center",
                style="italic",
                zorder=10,
            )

        if show_room_bboxes and hasattr(room, "bbox_min") and hasattr(room, "bbox_max"):
            x0, y0 = float(room.bbox_min[0]), float(room.bbox_min[1])
            x1, y1 = float(room.bbox_max[0]), float(room.bbox_max[1])
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", linewidth=0.8, alpha=0.6, zorder=6)


def _draw_openings(ax: Any, opening_data: list[Any] | None) -> None:
    if not opening_data:
        return
    for item in opening_data:
        if isinstance(item, tuple):
            opening = item[0]
        else:
            opening = item
        center = np.asarray(getattr(opening, "centroid_2d", [0.0, 0.0]), dtype=float)
        width = float(getattr(opening, "width", 0.0))
        normal_3d = np.asarray(getattr(opening, "normal_3d", [1.0, 0.0, 0.0]), dtype=float).reshape(-1)
        normal_xy = normal_3d[:2] if normal_3d.size >= 2 else np.array([1.0, 0.0], dtype=float)
        norm = np.linalg.norm(normal_xy)
        if norm < 1e-6:
            normal_xy = np.array([1.0, 0.0], dtype=float)
        else:
            normal_xy = normal_xy / norm
        direction = np.array([-normal_xy[1], normal_xy[0]], dtype=float)
        half = 0.5 * max(width, 0.05)
        p0 = center - direction * half
        p1 = center + direction * half
        opening_type = str(getattr(opening, "opening_type", "window"))
        if opening_type == "door":
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=_DOOR_COLOR, linewidth=2.2, alpha=0.85, zorder=8)
        else:
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=_WINDOW_COLOR,
                linewidth=1.4,
                alpha=0.7,
                linestyle=(0, (3, 1.5)),
                zorder=8,
            )


def _draw_connectivity(
    ax: Any,
    graph: Any,
    center_map: dict[str, np.ndarray] | None,
    show_connectivity: bool,
) -> None:
    if not show_connectivity or graph is None:
        return
    if not hasattr(graph, "connections") or not hasattr(graph, "rooms"):
        return

    actual_label_used = False
    inferred_label_used = False
    actual_waypoints: list[tuple[float, float]] = []
    inferred_waypoints: list[tuple[float, float]] = []

    for conn in getattr(graph, "connections", []):
        room1 = graph.rooms.get(conn.room1_id)
        room2 = graph.rooms.get(conn.room2_id)
        if room1 is None or room2 is None:
            continue
        room1_obj = room1[0]
        room2_obj = room2[0]
        r1_id = str(getattr(room1_obj, "room_id", conn.room1_id))
        r2_id = str(getattr(room2_obj, "room_id", conn.room2_id))
        if center_map and r1_id in center_map:
            c1 = np.asarray(center_map[r1_id][:2], dtype=float)
        else:
            c1 = np.asarray(room1_obj.centroid[:2], dtype=float)
        if center_map and r2_id in center_map:
            c2 = np.asarray(center_map[r2_id][:2], dtype=float)
        else:
            c2 = np.asarray(room2_obj.centroid[:2], dtype=float)
        wx, wy = float(conn.waypoint.position[0]), float(conn.waypoint.position[1])
        door_type = str(getattr(conn.waypoint, "door_type", "synthetic"))
        if door_type == "actual":
            kwargs = {
                "color": _ACTUAL_CONN_COLOR,
                "linestyle": "-",
                "linewidth": 1.4,
                "alpha": 0.72,
            }
            if not actual_label_used:
                kwargs["label"] = "Observed door passage (actual opening)"
                actual_label_used = True
            actual_waypoints.append((wx, wy))
        else:
            kwargs = {
                "color": _INFERRED_CONN_COLOR,
                "linestyle": (0, (4, 2)),
                "linewidth": 1.4,
                "alpha": 0.6,
            }
            if not inferred_label_used:
                kwargs["label"] = "Inferred passage (no detected door mesh)"
                inferred_label_used = True
            inferred_waypoints.append((wx, wy))

        ax.plot([c1[0], wx, c2[0]], [c1[1], wy, c2[1]], zorder=6, **kwargs)

    if actual_waypoints:
        pts = np.asarray(actual_waypoints, dtype=float)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=_ACTUAL_CONN_COLOR,
            marker="o",
            s=58,
            edgecolors="white",
            linewidth=1.0,
            alpha=0.9,
            zorder=9,
        )
    if inferred_waypoints:
        pts = np.asarray(inferred_waypoints, dtype=float)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            facecolors="none",
            edgecolors=_INFERRED_CONN_COLOR,
            marker="o",
            s=58,
            linewidth=1.4,
            alpha=0.85,
            zorder=9,
        )


def _configure(ax: Any, title: str) -> None:
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def render_hl3d_debug_plots(
    *,
    scene: str,
    floorplans: list[Any],
    rooms_by_floor: dict[int, list[Any]],
    openings_by_floor: dict[int, list[Any]] | None,
    connectivity_graphs: dict[int, Any],
    output_dir: Path,
    center_map_by_floor: dict[int, dict[str, np.ndarray]] | None = None,
    write_combined_plot: bool = True,
    write_floor_plots: bool = True,
    show_room_bboxes: bool = False,
    color_room_intersections: bool = True,
    show_connectivity: bool = True,
) -> tuple[Optional[Path], tuple[Path, ...]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path: Optional[Path] = None
    floor_paths: list[Path] = []

    if write_combined_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        for floor in floorplans:
            floor_idx = int(getattr(floor, "level_index", 0))
            center_map = (center_map_by_floor or {}).get(floor_idx)
            _draw_floor(ax, floor)
            _draw_rooms(
                ax,
                rooms_by_floor.get(floor_idx),
                center_map=center_map,
                color_room_intersections=color_room_intersections,
                show_room_bboxes=show_room_bboxes,
            )
            opening_data = openings_by_floor.get(floor_idx) if openings_by_floor else None
            _draw_openings(ax, opening_data)
            _draw_connectivity(ax, connectivity_graphs.get(floor_idx), center_map, show_connectivity)
        _configure(ax, f"{scene} — Floorplan Debug")
        combined_path = output_dir / f"{scene}_floorplan.png"
        fig.tight_layout()
        fig.savefig(combined_path, dpi=220)
        plt.close(fig)

    if write_floor_plots:
        for floor in floorplans:
            floor_idx = int(getattr(floor, "level_index", 0))
            fig, ax = plt.subplots(figsize=(10, 10))
            center_map = (center_map_by_floor or {}).get(floor_idx)
            _draw_floor(ax, floor)
            _draw_rooms(
                ax,
                rooms_by_floor.get(floor_idx),
                center_map=center_map,
                color_room_intersections=color_room_intersections,
                show_room_bboxes=show_room_bboxes,
            )
            opening_data = openings_by_floor.get(floor_idx) if openings_by_floor else None
            _draw_openings(ax, opening_data)
            _draw_connectivity(ax, connectivity_graphs.get(floor_idx), center_map, show_connectivity)
            _configure(ax, f"{scene} — Floor {floor_idx} Debug")
            floor_path = output_dir / f"{scene}_floor_{floor_idx}.png"
            fig.tight_layout()
            fig.savefig(floor_path, dpi=220)
            plt.close(fig)
            floor_paths.append(floor_path)

    return combined_path, tuple(floor_paths)
