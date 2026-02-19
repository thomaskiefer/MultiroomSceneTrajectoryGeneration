"""Unified trajectory visualization surface.

This module is the canonical visualization API:
- lightweight plotting (`plot_trajectory`, `plot_trajectory_from_artifacts`)
- rich renderers (`render_trajectory_image`, `render_trajectory_video`)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from .artifacts import TrajectoryGenerationArtifacts
    from .config import TrajectoryVisualizationConfig

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def plot_trajectory(
    frames: list[dict[str, Any]],
    floor_polygon: Any = None,
    room_polygons: Optional[dict[str, Any]] = None,
    show_look_at: bool = True,
    title: Optional[str] = None,
    viz_config: Optional["TrajectoryVisualizationConfig"] = None,
    output_path: Optional[Path] = None,
    ax: Any = None,
) -> Any:
    """Plot a camera trajectory on a 2D floor layout.

    Args:
        frames: List of frame dicts with ``position`` and ``look_at`` keys.
        floor_polygon: Optional Shapely Polygon for the floor outline.
        room_polygons: Optional ``{label: Polygon}`` dict for room outlines.
        show_look_at: Draw look-direction arrows when True.
        title: Plot title.
        viz_config: Visual style overrides.
        output_path: Save PNG to this path when set.
        ax: Existing matplotlib Axes to draw on. A new figure is created if None.

    Returns:
        The matplotlib Axes object used for drawing.
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for trajectory visualization. "
            "Install it with: pip install matplotlib"
        )

    from .config import TrajectoryVisualizationConfig
    cfg = viz_config or TrajectoryVisualizationConfig()

    if not frames:
        logger.warning("No frames to plot.")
        return ax

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        created_fig = True

    # Draw floor outline.
    if floor_polygon is not None:
        _draw_polygon(ax, floor_polygon, edgecolor="#333333", facecolor="#F5F5F5", linewidth=1.5, zorder=1)

    # Draw room outlines.
    if room_polygons:
        for label, poly in room_polygons.items():
            _draw_polygon(ax, poly, edgecolor="#888888", facecolor="none", linewidth=0.8, linestyle="--", zorder=2)
            try:
                cx, cy = poly.centroid.x, poly.centroid.y
                ax.text(cx, cy, label, fontsize=7, ha="center", va="center", color="#666666", zorder=3)
            except (AttributeError, TypeError, ValueError):
                logger.warning("Failed to place room label for %s.", label, exc_info=True)

    # Extract positions.
    positions = np.array([f["position"][:2] for f in frames])

    # Camera path.
    ax.plot(
        positions[:, 0], positions[:, 1],
        color=cfg.path_color,
        linewidth=cfg.path_linewidth,
        alpha=0.8,
        label="Camera Path",
        zorder=15,
    )

    # Start / end markers.
    ax.scatter(
        positions[0, 0], positions[0, 1],
        color=cfg.start_color, s=cfg.marker_size, marker="^",
        edgecolors="white", zorder=20, label="Start",
    )
    ax.scatter(
        positions[-1, 0], positions[-1, 1],
        color=cfg.end_color, s=cfg.marker_size, marker="s",
        edgecolors="white", zorder=20, label="End",
    )

    # Look-direction arrows.
    if show_look_at and len(frames) > 1:
        step = max(1, len(positions) // cfg.max_arrows)
        for i in range(0, len(frames), step):
            frame = frames[i]
            if "position" not in frame or "look_at" not in frame:
                continue
            pos = np.array(frame["position"][:2], dtype=float)
            target = np.array(frame["look_at"][:2], dtype=float)
            direction = target - pos
            norm = np.linalg.norm(direction)
            if norm > 0.01:
                direction /= norm
                ax.arrow(
                    pos[0], pos[1],
                    direction[0] * cfg.arrow_length,
                    direction[1] * cfg.arrow_length,
                    head_width=cfg.arrow_head_width,
                    head_length=cfg.arrow_head_length,
                    fc=cfg.arrow_color,
                    ec=cfg.arrow_color,
                    alpha=0.6,
                    zorder=16,
                )

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ax.get_figure().savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved trajectory plot to %s", output_path)

    if created_fig:
        plt.close(fig)

    return ax


def plot_trajectory_from_artifacts(
    artifacts: "TrajectoryGenerationArtifacts",
    floor_index: Optional[int] = None,
    **kwargs: Any,
) -> list[Path]:
    """Plot trajectories for floors in *artifacts*, loading frames from saved JSON.

    Args:
        artifacts: Output from a trajectory generation run.
        floor_index: Restrict to a single floor. Plots all floors when None.
        **kwargs: Forwarded to :func:`plot_trajectory`.

    Returns:
        List of output PNG paths that were written.
    """
    output_paths: list[Path] = []
    for fa in artifacts.floor_trajectories:
        if floor_index is not None and fa.floor_index != floor_index:
            continue

        frames_path = Path(fa.output_file)
        if not frames_path.exists():
            logger.warning("Trajectory file not found: %s", frames_path)
            continue

        frames = json.loads(frames_path.read_text(encoding="utf-8"))
        out = frames_path.with_name(frames_path.stem + "_visualization.png")

        plot_trajectory(
            frames=frames,
            title=f"Floor {fa.floor_index} — {fa.num_frames} frames, {fa.num_rooms} rooms",
            output_path=out,
            **kwargs,
        )
        output_paths.append(out)

    return output_paths


def render_trajectory_image(
    geojson_path: Path,
    trajectory_path: Path,
    output_path: Path,
    scene_name: str = "",
    dpi: int = 200,
    fps: int = 30,
) -> Path:
    """Render a publication-quality trajectory image from GeoJSON + trajectory."""
    from .visualization_backend import render_trajectory_image as _render_trajectory_image

    return _render_trajectory_image(
        geojson_path=geojson_path,
        trajectory_path=trajectory_path,
        output_path=output_path,
        scene_name=scene_name,
        dpi=dpi,
        fps=fps,
    )


def render_trajectory_video(
    geojson_path: Path,
    trajectory_path: Path,
    output_path: Path,
    scene_name: str = "",
    fps: int = 30,
    speed: float = 1.0,
    dpi: int = 120,
) -> Path:
    """Render an animated trajectory MP4 from GeoJSON + trajectory."""
    from .visualization_backend import render_trajectory_video as _render_trajectory_video

    return _render_trajectory_video(
        geojson_path=geojson_path,
        trajectory_path=trajectory_path,
        output_path=output_path,
        scene_name=scene_name,
        fps=fps,
        speed=speed,
        dpi=dpi,
    )


def main() -> None:
    """CLI wrapper for the canonical visualization entrypoint."""
    from .visualization_backend import main as _visualization_backend_main

    _visualization_backend_main()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _draw_polygon(
    ax: Any,
    polygon: Any,
    edgecolor: str = "#333333",
    facecolor: str = "none",
    linewidth: float = 1.0,
    linestyle: str = "-",
    zorder: int = 1,
) -> None:
    """Draw a Shapely Polygon or MultiPolygon on *ax*."""
    from shapely.geometry import Polygon as ShapelyPolygon

    if hasattr(polygon, "geom_type"):
        if polygon.geom_type == "Polygon":
            geoms = [polygon]
        elif polygon.geom_type == "MultiPolygon":
            geoms = list(polygon.geoms)
        else:
            return
    else:
        return

    for geom in geoms:
        if not isinstance(geom, ShapelyPolygon):
            continue
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, color=edgecolor, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
        if facecolor != "none":
            ax.fill(xs, ys, color=facecolor, alpha=0.3, zorder=zorder - 1)
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, color=edgecolor, linewidth=linewidth * 0.7, linestyle=linestyle, zorder=zorder)
