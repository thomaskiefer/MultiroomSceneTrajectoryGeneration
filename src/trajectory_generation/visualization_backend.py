"""Trajectory visualization: clean scientific images and animated MP4 video.

Reads GeoJSON floorplan data (rooms, doors, windows, connectivity) alongside
trajectory JSON frames to produce publication-quality visualizations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_ROOM_COLORS = {
    "bathroom": "#a8d8ea",
    "bedroom": "#b8d4e3",
    "kitchen": "#f7c59f",
    "living_room": "#b5d8a0",
    "hallway": "#e0e0e0",
    "dining_room": "#f2b5b5",
    "closet": "#d4b8e0",
    "entryway": "#f0e68c",
    "toilet": "#b2dfdb",
    "stairs": "#d7ccc8",
    "office": "#c5b3d9",
    "garage": "#cfd8dc",
    "utility_room": "#cfd8dc",
    "laundry_room": "#b3c6e0",
    "library": "#d7ccc8",
    "porch": "#e8d5b7",
    "balcony": "#ffd8b1",
    "outdoor": "#c8e6c9",
    "family_room": "#c8e6c9",
    "lounge": "#b2ebf2",
    "rec_room": "#e6ee9c",
    "other": "#e0e0e0",
    "unlabeled": "#eeeeee",
}

_TRAJ_CMAP = LinearSegmentedColormap.from_list(
    "traj", ["#1a237e", "#4a148c", "#b71c1c", "#e65100", "#f57f17"]
)

_FLOOR_FILL = "#f5f5f5"
_FLOOR_EDGE = "#333333"
_FLOOR_EDGE_LW = 1.8
_ROOM_EDGE = "#555555"
_ROOM_EDGE_LW = 1.0
_ROOM_FILL_ALPHA = 0.35
_ROOM_LABEL_COLOR = "#555555"
_ROOM_LABEL_SIZE = 8
_TRAJ_LW = 2.5
_MARKER_SIZE = 160
_DIR_ARROW_MAX = 28
_DIR_ARROW_MIN_SPACING_M = 0.8
_DIR_HEAD_SIZE = 120
_ACTUAL_CONN_COLOR = "#1f6f8b"
_INFERRED_CONN_COLOR = "#1f6f8b"
_CONN_LW = 1.4
_DOOR_COLOR = "#c0392b"
_DOOR_LW = 2.5
_WINDOW_COLOR = "#2980b9"
_WINDOW_LW = 1.5
_WINDOW_END_TICK_L = 0.22
_STAIRS_FILL = "#c8b7a6"
_STAIRS_EDGE = "#8d6e63"
_STAIRS_LW = 1.6


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_features(gj: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    _MAP = {
        "floor_footprint": "floor", "room": "rooms", "door": "doors",
        "window": "windows", "stairs": "stairs", "room_connection": "connections",
        "door_waypoint": "waypoints", "trajectory_room_center": "trajectory_centers",
    }
    out: dict[str, list[dict[str, Any]]] = {v: [] for v in _MAP.values()}
    for f in gj.get("features", []):
        key = _MAP.get(f["properties"].get("type", ""))
        if key:
            out[key].append(f)
    return out


def _data_bounds(feats: dict, frames: list, padding: float = 0.5):
    xs, ys = [], []
    for feat in feats["floor"]:
        for ring in feat["geometry"]["coordinates"]:
            for c in ring:
                xs.append(c[0])
                ys.append(c[1])
    for f in frames:
        xs.append(f["position"][0])
        ys.append(f["position"][1])
    if not xs:
        return -1, 1, -1, 1
    return min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding


def _room_color(label: str) -> str:
    return _ROOM_COLORS.get(label, _ROOM_COLORS["other"])


def _build_room_center_map(feats: dict) -> dict[str, np.ndarray]:
    """Build room center map with precedence:
    trajectory_room_center point > room.trajectory_center_xy > room.trajectory_center_3d
    > room.centroid_3d > polygon mean.
    """
    room_center: dict[str, np.ndarray] = {}

    # Lowest-precedence sources from room features.
    for f in feats["rooms"]:
        props = f.get("properties", {})
        room_id = str(props.get("room_id", "")).strip()
        if not room_id:
            continue
        center_xy = props.get("trajectory_center_xy")
        if isinstance(center_xy, (list, tuple)) and len(center_xy) >= 2:
            room_center[room_id] = np.array([float(center_xy[0]), float(center_xy[1])], dtype=float)
            continue
        center_3d = props.get("trajectory_center_3d")
        if isinstance(center_3d, (list, tuple)) and len(center_3d) >= 2:
            room_center[room_id] = np.array([float(center_3d[0]), float(center_3d[1])], dtype=float)
            continue
        centroid_3d = props.get("centroid_3d")
        if isinstance(centroid_3d, (list, tuple)) and len(centroid_3d) >= 2:
            room_center[room_id] = np.array([float(centroid_3d[0]), float(centroid_3d[1])], dtype=float)
            continue
        coords = f.get("geometry", {}).get("coordinates", [])
        if coords and coords[0]:
            c = coords[0]
            room_center[room_id] = np.array(
                [float(np.mean([p[0] for p in c])), float(np.mean([p[1] for p in c]))],
                dtype=float,
            )

    # Highest-precedence trajectory center points.
    for f in feats["trajectory_centers"]:
        props = f.get("properties", {})
        room_id = str(props.get("room_id", "")).strip()
        c = f.get("geometry", {}).get("coordinates", [])
        if room_id and isinstance(c, list) and len(c) >= 2:
            room_center[room_id] = np.array([float(c[0]), float(c[1])], dtype=float)

    return room_center


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _setup_axes(ax, x0, x1, y0, y1):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.tick_params(direction="in", top=True, right=True)


def _draw_floor(ax, feats):
    for f in feats["floor"]:
        for ring in f["geometry"]["coordinates"]:
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            ax.fill(xs, ys, color=_FLOOR_FILL, zorder=1)
            ax.plot(xs, ys, color=_FLOOR_EDGE, linewidth=_FLOOR_EDGE_LW,
                    solid_capstyle="round", zorder=2)


def _draw_rooms(ax, feats, room_center: dict[str, np.ndarray] | None = None):
    # Pass 1: fills
    for f in feats["rooms"]:
        label = f["properties"].get("label_semantic", "other")
        color = _room_color(label)
        for ring in f["geometry"]["coordinates"]:
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            ax.fill(xs, ys, color=color, alpha=_ROOM_FILL_ALPHA,
                    edgecolor="none", zorder=3)

    # Pass 2: outlines
    for f in feats["rooms"]:
        for ring in f["geometry"]["coordinates"]:
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            ax.plot(xs, ys, color=_ROOM_EDGE, linewidth=_ROOM_EDGE_LW, zorder=4)

    # Pass 3: labels
    for f in feats["rooms"]:
        label = f["properties"].get("label_semantic", "other")
        props = f.get("properties", {})
        room_id = str(props.get("room_id", "")).strip()
        center = room_center.get(room_id) if room_center else None
        if center is not None:
            cx = float(center[0])
            cy = float(center[1])
        else:
            coords = f["geometry"]["coordinates"][0]
            cx = float(np.mean([c[0] for c in coords]))
            cy = float(np.mean([c[1] for c in coords]))
        ax.text(cx, cy, label.replace("_", " ").title(),
                fontsize=_ROOM_LABEL_SIZE, ha="center", va="center",
                color=_ROOM_LABEL_COLOR, fontstyle="italic", zorder=25)


def _draw_doors(ax, feats):
    for f in feats["doors"]:
        coords = f["geometry"]["coordinates"]
        if len(coords) >= 2:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color=_DOOR_COLOR, linewidth=_DOOR_LW,
                    solid_capstyle="butt", alpha=0.8, zorder=8)


def _draw_windows(ax, feats):
    for f in feats["windows"]:
        coords = f["geometry"]["coordinates"]
        if len(coords) >= 2:
            p0 = np.array([float(coords[0][0]), float(coords[0][1])], dtype=float)
            p1 = np.array([float(coords[-1][0]), float(coords[-1][1])], dtype=float)
            seg = p1 - p0
            seg_norm = float(np.linalg.norm(seg))
            if seg_norm <= 1e-9:
                continue
            direction = seg / seg_norm
            perp = np.array([-direction[1], direction[0]], dtype=float)

            # Main window span (solid), matching HL3D's parallel style baseline.
            ax.plot(
                [float(p0[0]), float(p1[0])],
                [float(p0[1]), float(p1[1])],
                color=_WINDOW_COLOR,
                linewidth=_WINDOW_LW * 1.45,
                alpha=0.78,
                solid_capstyle="butt",
                zorder=8,
            )

            # Two short perpendicular marks at the span ends.
            half_tick = 0.5 * _WINDOW_END_TICK_L
            for anchor in (p0, p1):
                s = anchor - perp * half_tick
                e = anchor + perp * half_tick
                ax.plot(
                    [float(s[0]), float(e[0])],
                    [float(s[1]), float(e[1])],
                    color=_WINDOW_COLOR,
                    linewidth=_WINDOW_LW,
                    alpha=0.72,
                    zorder=8,
                )


def _draw_stairs(ax, feats):
    used_label = False
    for f in feats["stairs"]:
        geometry = f.get("geometry", {})
        geom_type = str(geometry.get("type", ""))
        coords = geometry.get("coordinates")
        if geom_type == "Polygon" and isinstance(coords, list) and coords:
            ring = coords[0]
            if not isinstance(ring, list) or len(ring) < 3:
                continue
            xs = [float(c[0]) for c in ring]
            ys = [float(c[1]) for c in ring]
            fill_kwargs = {"color": _STAIRS_FILL, "alpha": 0.30, "zorder": 7, "hatch": "///"}
            if not used_label:
                fill_kwargs["label"] = "Stairs"
            ax.fill(xs, ys, **fill_kwargs)
            ax.plot(xs, ys, color=_STAIRS_EDGE, linewidth=_STAIRS_LW, alpha=0.95, zorder=8)
            used_label = True
        elif geom_type == "LineString" and isinstance(coords, list) and len(coords) >= 2:
            xs = [float(c[0]) for c in coords]
            ys = [float(c[1]) for c in coords]
            line_kwargs = {"color": _STAIRS_EDGE, "linewidth": _STAIRS_LW, "alpha": 0.95, "zorder": 8}
            if not used_label:
                line_kwargs["label"] = "Stairs"
            ax.plot(xs, ys, **line_kwargs)
            used_label = True
        elif geom_type == "Point" and isinstance(coords, list) and len(coords) >= 2:
            marker_kwargs = {
                "s": 72,
                "marker": "s",
                "color": _STAIRS_FILL,
                "edgecolors": _STAIRS_EDGE,
                "linewidths": 1.0,
                "alpha": 0.95,
                "zorder": 9,
            }
            if not used_label:
                marker_kwargs["label"] = "Stairs"
            ax.scatter([float(coords[0])], [float(coords[1])], **marker_kwargs)
            used_label = True


def _draw_connections(ax, feats, room_center: dict[str, np.ndarray] | None = None):
    room_center = room_center or _build_room_center_map(feats)
    wp_by_pair: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    for f in feats["waypoints"]:
        props = f.get("properties", {})
        room1 = str(props.get("room1_id", "")).strip()
        room2 = str(props.get("room2_id", "")).strip()
        if not room1 or not room2:
            continue
        pair = (room1, room2) if room1 <= room2 else (room2, room1)
        c = f.get("geometry", {}).get("coordinates", [])
        if not isinstance(c, list) or len(c) < 2:
            continue
        opening_id = str(props.get("opening_id", ""))
        wp_by_pair.setdefault(pair, []).append((float(c[0]), float(c[1]), opening_id))

    actual_waypoints: list[tuple[float, float]] = []
    inferred_waypoints: list[tuple[float, float]] = []
    used_actual_label = False
    used_inferred_label = False

    for f in feats["connections"]:
        props = f.get("properties", {})
        room1 = str(props.get("room1_id", "")).strip()
        room2 = str(props.get("room2_id", "")).strip()
        if not room1 or not room2:
            continue
        c1 = room_center.get(room1)
        c2 = room_center.get(room2)
        if c1 is None or c2 is None:
            coords = f.get("geometry", {}).get("coordinates", [])
            if isinstance(coords, list) and len(coords) >= 2:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                ax.plot(xs, ys, color="#999999", linewidth=0.8, alpha=0.6, linestyle="--", zorder=5)
            continue

        pair = (room1, room2) if room1 <= room2 else (room2, room1)
        opening_id = str(props.get("opening_id", ""))
        wp_xy: tuple[float, float] | None = None
        waypoints = wp_by_pair.get(pair, [])
        if waypoints:
            if opening_id:
                for wx, wy, oid in waypoints:
                    if oid == opening_id:
                        wp_xy = (wx, wy)
                        break
            if wp_xy is None:
                wp_xy = (waypoints[0][0], waypoints[0][1])
        else:
            coords = f.get("geometry", {}).get("coordinates", [])
            if isinstance(coords, list) and len(coords) >= 2:
                mid = np.array(
                    [
                        float(np.mean([coords[0][0], coords[-1][0]])),
                        float(np.mean([coords[0][1], coords[-1][1]])),
                    ],
                    dtype=float,
                )
                wp_xy = (float(mid[0]), float(mid[1]))

        if wp_xy is None:
            continue

        door_type = str(props.get("door_type", "synthetic"))
        if door_type == "actual":
            style = {"linestyle": "-", "alpha": 0.72, "color": _ACTUAL_CONN_COLOR}
            if not used_actual_label:
                style["label"] = "Observed door passage"
                used_actual_label = True
            actual_waypoints.append(wp_xy)
        else:
            style = {"linestyle": (0, (4, 2)), "alpha": 0.60, "color": _INFERRED_CONN_COLOR}
            if not used_inferred_label:
                style["label"] = "Inferred passage"
                used_inferred_label = True
            inferred_waypoints.append(wp_xy)

        ax.plot(
            [float(c1[0]), float(wp_xy[0]), float(c2[0])],
            [float(c1[1]), float(wp_xy[1]), float(c2[1])],
            linewidth=_CONN_LW,
            zorder=6,
            **style,
        )

    if actual_waypoints:
        pts = np.asarray(actual_waypoints, dtype=float)
        ax.scatter(
            pts[:, 0], pts[:, 1],
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
            pts[:, 0], pts[:, 1],
            facecolors="none",
            edgecolors=_INFERRED_CONN_COLOR,
            marker="o",
            s=58,
            linewidth=1.4,
            alpha=0.85,
            zorder=9,
        )


def _draw_trajectory(ax, frames, fps: int = 30):
    """Draw time-colored trajectory using LineCollection."""
    positions = np.array([f["position"][:2] for f in frames])
    n = len(positions)
    if n == 0:
        raise ValueError("Trajectory contains no frames.")

    pts = positions.reshape(-1, 1, 2)
    time_s = np.arange(n) / float(fps)
    if n == 1:
        norm = Normalize(vmin=0.0, vmax=1.0)
        lc = LineCollection([], cmap=_TRAJ_CMAP, norm=norm, linewidths=_TRAJ_LW, capstyle="round", zorder=15)
        lc.set_array(np.array([0.0]))
        ax.add_collection(lc)
        return positions, lc, time_s

    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vmax = float(time_s[-1]) if time_s[-1] > 0.0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    lc = LineCollection(segs.tolist(), cmap=_TRAJ_CMAP, norm=norm,
                        linewidths=_TRAJ_LW, capstyle="round", zorder=15)
    lc.set_array(time_s[:-1])
    ax.add_collection(lc)

    return positions, lc, time_s


def _draw_direction_arrows(ax, positions: np.ndarray):
    """Overlay head-only directional markers aligned to trajectory tangent."""
    if len(positions) < 3:
        return

    seg_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_len = float(seg_lengths.sum())
    if total_len < 1e-6:
        return

    max_by_distance = max(1, int(total_len / _DIR_ARROW_MIN_SPACING_M))
    num_arrows = min(_DIR_ARROW_MAX, max_by_distance)
    if num_arrows < 1:
        return

    cum = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    targets = np.linspace(
        total_len / (num_arrows + 1),
        total_len * num_arrows / (num_arrows + 1),
        num_arrows,
    )
    idxs = np.searchsorted(cum, targets, side="left")
    color_norm = Normalize(vmin=0, vmax=max(len(positions) - 1, 1))
    used = set()
    for idx in idxs:
        idx = int(np.clip(idx, 1, len(positions) - 2))
        if idx in used:
            continue
        used.add(idx)

        k_prev = max(idx - 2, 0)
        k_next = min(idx + 3, len(positions) - 1)
        direction = positions[k_next] - positions[k_prev]
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            continue
        direction /= direction_norm

        anchor = positions[idx]
        color = _TRAJ_CMAP(color_norm(idx))
        angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))

        # Draw only the arrow head (triangle) so markers sit directly on the path.
        ax.scatter(
            [anchor[0]],
            [anchor[1]],
            marker=(3, 0, angle_deg - 90.0),
            s=_DIR_HEAD_SIZE,
            c=[color],
            edgecolors="none",
            alpha=0.98,
            zorder=18,
        )


def _draw_markers(ax, positions):
    ax.scatter(*positions[0], s=_MARKER_SIZE, color="#27ae60", marker="^",
              edgecolors="white", linewidths=0.8, zorder=22, label="Start")
    ax.scatter(*positions[-1], s=_MARKER_SIZE, color="#e74c3c", marker="s",
              edgecolors="white", linewidths=0.8, zorder=22, label="End")


def _draw_chrome(ax, fig, num_frames, num_rooms, scene, lc, time_s):
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
              edgecolor="0.85")

    stats = f"{num_frames:,} frames  |  {time_s[-1]:.0f} s  |  {num_rooms} rooms"
    ax.text(0.01, 0.01, stats, transform=ax.transAxes, fontsize=7,
            color="0.45", va="bottom", ha="left")

    if scene:
        ax.set_title(scene, fontsize=12, fontfamily="monospace", pad=10)

    cbar = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("time [s]", fontsize=9)
    cbar.ax.tick_params(labelsize=7, length=2)
    ticks = [0, time_s[-1] / 2, time_s[-1]]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.0f}" for t in ticks])


def _draw_base(ax, feats, x0, x1, y0, y1):
    _setup_axes(ax, x0, x1, y0, y1)
    room_center = _build_room_center_map(feats)
    _draw_floor(ax, feats)
    _draw_rooms(ax, feats, room_center=room_center)
    _draw_stairs(ax, feats)
    _draw_connections(ax, feats, room_center=room_center)
    _draw_doors(ax, feats)
    _draw_windows(ax, feats)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_trajectory_image(
    geojson_path: Path,
    trajectory_path: Path,
    output_path: Path,
    scene_name: str = "",
    dpi: int = 200,
    fps: int = 30,
) -> Path:
    """Generate a publication-quality trajectory image."""
    gj = _load(geojson_path)
    frames = _load(trajectory_path)
    feats = _parse_features(gj)
    x0, x1, y0, y1 = _data_bounds(feats, frames)

    aspect = (x1 - x0) / max(y1 - y0, 0.01)
    fig_h = 8
    fig_w = max(6, fig_h * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    _draw_base(ax, feats, x0, x1, y0, y1)
    positions, lc, time_s = _draw_trajectory(ax, frames, fps=fps)
    _draw_direction_arrows(ax, positions)
    _draw_markers(ax, positions)
    _draw_chrome(ax, fig, len(frames), len(feats["rooms"]), scene_name, lc, time_s)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info("Saved trajectory image to %s", output_path)
    return output_path


def render_trajectory_video(
    geojson_path: Path,
    trajectory_path: Path,
    output_path: Path,
    scene_name: str = "",
    fps: int = 30,
    speed: float = 1.0,
    dpi: int = 120,
) -> Path:
    """Generate an animated MP4 video of the camera traversal.

    Renders the static base map once with matplotlib, then uses OpenCV to
    composite the animated time-colored trail and camera indicator per frame.
    The trail style matches the static image (same colormap, line weight).
    """
    import cv2

    if speed <= 0:
        raise ValueError("`speed` must be > 0.")

    gj = _load(geojson_path)
    frames = _load(trajectory_path)
    if not frames:
        raise ValueError("Trajectory contains no frames.")
    feats = _parse_features(gj)
    x0, x1, y0, y1 = _data_bounds(feats, frames)
    positions = np.array([f["position"][:2] for f in frames])

    # --- Render static base map once ---
    aspect = (x1 - x0) / max(y1 - y0, 0.01)
    fig_h = 8
    fig_w = max(6, fig_h * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_base(ax, feats, x0, x1, y0, y1)

    # Ghost trail (faint full path so the viewer sees what's coming)
    pts = positions.reshape(-1, 1, 2)
    all_segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ghost_lc = LineCollection(
        all_segs.tolist(), colors=[(0.75, 0.75, 0.75, 0.15)] * len(all_segs),
        linewidths=0.5, zorder=10,
    )
    ax.add_collection(ghost_lc)

    if scene_name:
        ax.set_title(scene_name, fontsize=12, fontfamily="monospace", pad=10)

    fig.canvas.draw()
    base_img = np.asarray(fig.canvas.buffer_rgba()).copy()  # type: ignore[attr-defined]

    bbox = ax.get_window_extent(fig.canvas.get_renderer())  # type: ignore[attr-defined]
    px_left, px_bottom = bbox.x0, bbox.y0
    px_right, px_top = bbox.x1, bbox.y1
    fig_px_h = base_img.shape[0]
    plt.close(fig)

    # Ensure even dimensions for H264
    img_h, img_w = base_img.shape[:2]
    img_w = img_w if img_w % 2 == 0 else img_w - 1
    img_h = img_h if img_h % 2 == 0 else img_h - 1
    base_img = base_img[:img_h, :img_w]
    base_bgr = cv2.cvtColor(base_img, cv2.COLOR_RGBA2BGR)

    # Coordinate transform: data space → pixel space
    def data_to_px(xy):
        x_frac = (xy[0] - x0) / (x1 - x0)
        y_frac = (xy[1] - y0) / (y1 - y0)
        col = int(px_left + x_frac * (px_right - px_left))
        row = int(fig_px_h - (px_bottom + y_frac * (px_top - px_bottom)))
        return col, row

    px_positions = np.array([data_to_px(p) for p in positions])
    n_traj = len(positions)
    n_segs = n_traj - 1
    n_anim = int(np.ceil(n_traj / speed))

    # Pre-compute per-segment colors matching the trajectory colormap
    time_s = np.arange(n_traj) / float(fps)
    norm = Normalize(vmin=0, vmax=(time_s[-1] if time_s[-1] > 0 else 1.0))
    seg_colors_rgba = _TRAJ_CMAP(norm(time_s[:n_segs]))
    seg_colors_bgr = (seg_colors_rgba[:, [2, 1, 0]] * 255).astype(np.uint8)

    # Scale line thickness to match the image feel at video resolution
    meters_per_px = (x1 - x0) / (px_right - px_left)
    trail_thick = max(2, int(0.04 / meters_per_px))
    dot_radius = max(4, int(0.06 / meters_per_px))
    arrow_len_px = max(10, int(0.30 / meters_per_px))
    fov_length_px = max(15, int(0.6 / meters_per_px))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (img_w, img_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    logger.info("Encoding %d video frames at %dfps (%dx%d) ...",
                n_anim, fps, img_w, img_h)

    for anim_frame in range(n_anim):
        idx = min(int(anim_frame * speed), n_traj - 1)
        seg_n = min(idx, n_segs)
        frame = base_bgr.copy()

        # Draw trail: per-segment color for smooth gradient
        for i in range(seg_n):
            p1 = tuple(px_positions[i])
            p2 = tuple(px_positions[i + 1])
            color = seg_colors_bgr[i].tolist()
            cv2.line(frame, p1, p2, color, trail_thick, cv2.LINE_AA)

        # Camera dot: white outline + dark fill
        pos_px = tuple(px_positions[idx])
        cv2.circle(frame, pos_px, dot_radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pos_px, dot_radius, (40, 40, 40), -1, cv2.LINE_AA)

        # Direction arrow + FOV cone
        forward_raw = frames[idx].get("forward")
        if isinstance(forward_raw, list) and len(forward_raw) >= 2:
            fwd = np.array(forward_raw[:2], dtype=float)
        else:
            look_at = np.array(frames[idx].get("look_at", frames[idx]["position"])[:2], dtype=float)
            pos_xy = np.array(frames[idx]["position"][:2], dtype=float)
            fwd = look_at - pos_xy
            if np.linalg.norm(fwd) <= 1e-6:
                if idx > 0:
                    fwd = positions[idx] - positions[idx - 1]
                elif n_traj > 1:
                    fwd = positions[1] - positions[0]
                else:
                    fwd = np.array([1.0, 0.0], dtype=float)
        fn = np.linalg.norm(fwd)
        if fn > 1e-6:
            fwd /= fn
            angle = np.arctan2(fwd[1], fwd[0])
            tip = (
                pos_px[0] + int(arrow_len_px * np.cos(angle)),
                pos_px[1] - int(arrow_len_px * np.sin(angle)),
            )
            cv2.arrowedLine(
                frame,
                pos_px,
                tip,
                (50, 50, 50),
                2,
                tipLength=0.35,
                line_type=cv2.LINE_AA,
            )

            half_fov = np.radians(frames[idx].get("fov", 60) / 2)
            left_ray = (
                pos_px[0] + int(fov_length_px * np.cos(angle + half_fov)),
                pos_px[1] - int(fov_length_px * np.sin(angle + half_fov)),
            )
            right_ray = (
                pos_px[0] + int(fov_length_px * np.cos(angle - half_fov)),
                pos_px[1] - int(fov_length_px * np.sin(angle - half_fov)),
            )
            tri = np.array([pos_px, left_ray, right_ray], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [tri], (200, 200, 200), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Time counter
        elapsed = idx / float(fps)
        total = n_traj / float(fps)
        text = f"{elapsed:.1f}s / {total:.0f}s  [{speed:.0f}x]"
        cv2.putText(frame, text, (img_w - 230, img_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1,
                    cv2.LINE_AA)

        writer.write(frame)
        if anim_frame % 200 == 0 and anim_frame > 0:
            logger.info("  %d / %d video frames ...", anim_frame, n_anim)

    writer.release()

    # Re-encode with H264
    h264_path = output_path.with_suffix(".tmp.mp4")
    import subprocess
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(output_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(h264_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning("ffmpeg re-encode failed (code=%s); keeping original mp4.", proc.returncode)
        elif h264_path.exists():
            h264_path.replace(output_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found; keeping original mp4 without H264 re-encode.")
    except Exception as exc:
        logger.warning("ffmpeg re-encode failed with unexpected error: %s", exc)

    logger.info("Saved trajectory video to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Trajectory visualization")
    p.add_argument("--geojson", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--scene", default="")
    p.add_argument("--image", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback compression factor: 1.0 = real-time.",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--video-dpi", type=int, default=120)
    args = p.parse_args()

    if not args.image and not args.video:
        args.image = True
        args.video = True

    stem = args.trajectory.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        out = args.output_dir / f"{stem}_visualization.png"
        render_trajectory_image(
            args.geojson, args.trajectory, out, args.scene,
            args.dpi, fps=args.fps,
        )
        print(f"Image -> {out}")

    if args.video:
        out = args.output_dir / f"{stem}.mp4"
        render_trajectory_video(
            args.geojson, args.trajectory, out, args.scene,
            fps=args.fps, speed=args.speed, dpi=args.video_dpi,
        )
        print(f"Video -> {out}")


if __name__ == "__main__":
    main()
