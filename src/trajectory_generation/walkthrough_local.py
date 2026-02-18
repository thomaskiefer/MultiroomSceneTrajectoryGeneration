"""Local walkthrough generation orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

try:
    from shapely.ops import polylabel
except ImportError:
    polylabel = None

from .camera import CameraPathBuilder
from .config import WalkthroughBehaviorConfig
from .control_points import ControlPointPlanner, ControlPointSequence
from .geometry_exceptions import geometry_exceptions
from .room_graph import RoomGraph
from .spline import CatmullRomSpline  # noqa: F401 - backward-compatible re-export
from .traversal import plan_room_sequence
from .validation import validate_trajectory


logger = logging.getLogger(__name__)
_GEOM_EXCEPTIONS = geometry_exceptions(include_runtime=True)

# Backward-compatible alias used by tests/internal callers.
_ControlPointSequence = ControlPointSequence


class LocalWalkthroughGenerator:
    """Generate walkthrough trajectories from a room graph."""

    def __init__(
        self,
        graph: RoomGraph,
        floor_z: float,
        camera_height: float = 1.6,
        behavior: Optional[WalkthroughBehaviorConfig] = None,
    ):
        self.graph = graph
        self.floor_z = floor_z
        self.eye_level = floor_z + camera_height
        self.behavior = behavior or WalkthroughBehaviorConfig()
        self.last_component_transfers: list[dict[str, Any]] = []
        self.last_validation_warnings: list[str] = []
        self.last_skipped_disconnected_rooms: list[str] = []
        self.last_disconnected_component_count: int = 1

        self._connection_lookup: dict[frozenset[str], Any] = {}
        for conn in getattr(self.graph, "connections", []):
            key = frozenset((conn.room1_id, conn.room2_id))
            if key not in self._connection_lookup:
                self._connection_lookup[key] = conn

        self._camera_builder = CameraPathBuilder(self.behavior)
        self._planner = ControlPointPlanner(
            behavior=self.behavior,
            eye_level=self.eye_level,
            get_room_center=self._get_room_center,
            get_connection=self._get_connection,
            get_door_normal=self._get_door_normal,
            component_transfers=self.last_component_transfers,
        )
        self.start_room_id = self._find_start_room()

    def _find_start_room(self, room_ids: Optional[set[str]] = None) -> Optional[str]:
        fallback_candidates: list[tuple[str, float]]
        for semantic in self.behavior.start_room_priority:
            candidates: list[tuple[str, float]] = []
            for rid, (room, poly) in self.graph.rooms.items():
                if room_ids is not None and rid not in room_ids:
                    continue
                if room.label_semantic == semantic:
                    candidates.append((rid, float(poly.area)))
            if candidates:
                return max(candidates, key=lambda item: (item[1], item[0]))[0]

        if room_ids is None:
            fallback_candidates = [
                (rid, float(poly.area))
                for rid, (_room, poly) in self.graph.rooms.items()
            ]
        else:
            fallback_candidates = [
                (rid, float(self.graph.rooms[rid][1].area))
                for rid in room_ids
                if rid in self.graph.rooms
            ]
        if not fallback_candidates:
            return None
        return max(fallback_candidates, key=lambda item: (item[1], item[0]))[0]

    def _connected_components(self, room_ids: set[str]) -> list[set[str]]:
        if hasattr(self.graph, "connected_components"):
            connected = [set(comp) & set(room_ids) for comp in self.graph.connected_components()]
            return [comp for comp in connected if comp]

        unseen = set(room_ids)
        components: list[set[str]] = []
        while unseen:
            root = next(iter(unseen))
            stack = [root]
            comp: set[str] = set()
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                for nxt in self.graph.adjacency.get(node, []):
                    if nxt in room_ids and nxt not in comp:
                        stack.append(nxt)
            components.append(comp)
            unseen -= comp
        return components

    def _largest_component_room_ids(self, all_room_ids: set[str], start_room_id: Optional[str]) -> set[str]:
        components = self._connected_components(all_room_ids)
        if not components:
            return set()
        max_size = max(len(comp) for comp in components)
        largest = [comp for comp in components if len(comp) == max_size]
        if start_room_id is not None:
            for comp in largest:
                if start_room_id in comp:
                    return set(comp)
        # Deterministic tie-break for equal-size components.
        return set(min(largest, key=lambda comp: sorted(comp)))

    def _generate_frames_for_room_subset(
        self,
        room_ids: set[str],
        start_room_id: str,
        transition_mode: str,
        fps: int,
    ) -> list[dict[str, Any]]:
        active_adjacency = {
            rid: [nbr for nbr in self.graph.adjacency.get(rid, []) if nbr in room_ids]
            for rid in room_ids
        }
        path_sequence = plan_room_sequence(
            start_room_id=start_room_id,
            all_room_ids=room_ids,
            adjacency=active_adjacency,
            room_center_xy=lambda rid: (
                float(self._get_room_center(rid)[0]),
                float(self._get_room_center(rid)[1]),
            ),
        )
        seq = self._generate_control_points(path_sequence, transition_mode)
        seq = self._deduplicate_control_points(seq)

        if len(seq.positions) == 0:
            return []
        if len(seq.positions) < 2:
            return self._build_static_frames(seq.positions[0], seq.look_targets[0], num_frames=2)

        smooth_pos, smooth_look, _total_time = self._interpolate_path(seq, fps)
        return self._build_camera_frames(smooth_pos, smooth_look, fps)

    def _reindex_frames(self, frames: list[dict[str, Any]], id_offset: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, frame in enumerate(frames):
            fr = dict(frame)
            fr["id"] = int(id_offset + i)
            out.append(fr)
        return out

    def _get_room_center(self, room_id: str) -> np.ndarray:
        """Get room center, preferring POI unless centroid is nearly as good."""
        room, poly = self.graph.rooms[room_id]
        centroid_xy = np.array([room.centroid[0], room.centroid[1]], dtype=float)

        if hasattr(poly, "centroid"):
            try:
                poly_centroid = poly.centroid
                centroid_xy = np.array([poly_centroid.x, poly_centroid.y], dtype=float)
            except _GEOM_EXCEPTIONS:
                logger.warning(
                    "Failed to compute polygon centroid for room %s; using room centroid.",
                    room_id,
                    exc_info=True,
                )

        if polylabel and hasattr(poly, "exterior"):
            try:
                pole = polylabel(poly, tolerance=self.behavior.polylabel_tolerance)
                try:
                    if hasattr(poly, "boundary") and hasattr(poly, "covers") and hasattr(poly, "centroid"):
                        centroid_pt = poly.centroid
                        if poly.covers(centroid_pt):
                            pole_clearance = float(poly.boundary.distance(pole))
                            centroid_clearance = float(poly.boundary.distance(centroid_pt))
                            if (pole_clearance - centroid_clearance) < self.behavior.polylabel_min_gain:
                                return np.array([centroid_pt.x, centroid_pt.y, self.eye_level])
                except _GEOM_EXCEPTIONS:
                    logger.debug("clearance comparison failed for room %s; using polylabel.", room_id)
                return np.array([pole.x, pole.y, self.eye_level])
            except _GEOM_EXCEPTIONS:
                logger.warning("polylabel failed for room %s; falling back.", room_id, exc_info=True)

        try:
            pt = poly.representative_point()
            return np.array([pt.x, pt.y, self.eye_level])
        except _GEOM_EXCEPTIONS:
            logger.warning(
                "representative_point failed for room %s; using room centroid.",
                room_id,
                exc_info=True,
            )
            return np.array([centroid_xy[0], centroid_xy[1], self.eye_level])

    def _get_door_normal(self, waypoint: Any) -> np.ndarray:
        """Calculate doorway normal for perpendicular crossing preference."""
        if hasattr(waypoint, "normal") and waypoint.normal is not None:
            normal_arr = np.asarray(waypoint.normal, dtype=float).reshape(-1)
            if normal_arr.size >= 2:
                normal_xy = normal_arr[:2]
                norm = np.linalg.norm(normal_xy)
                if norm > 1e-9:
                    return normal_xy / norm

        if not hasattr(waypoint, "shared_boundary") or waypoint.shared_boundary is None:
            return np.array([0.0, 0.0], dtype=float)

        coords = np.asarray(waypoint.shared_boundary.coords, dtype=float)
        if coords.ndim != 2 or len(coords) < 2 or coords.shape[1] < 2:
            return np.array([0.0, 0.0], dtype=float)
        coords = coords[:, :2]
        if np.max(np.ptp(coords, axis=0)) < 1e-9:
            return np.array([0.0, 0.0], dtype=float)

        if len(coords) == 2:
            segment = coords[1] - coords[0]
            normal = np.array([-segment[1], segment[0]], dtype=float)
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                return normal / norm
            return np.array([0.0, 0.0], dtype=float)

        coords_centered = coords - np.mean(coords, axis=0)
        cov = np.cov(coords_centered.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            if eigenvalues[-1] < 1e-12:
                return np.array([0.0, 0.0], dtype=float)
            normal = eigenvectors[:, 0]
            norm = np.linalg.norm(normal)
            if norm > 0:
                return normal / norm
        except np.linalg.LinAlgError:
            logger.warning("Door boundary PCA failed; falling back to heuristic normal.")

        return np.array([0.0, 0.0], dtype=float)

    def _make_control_point_planner(self) -> ControlPointPlanner:
        return self._planner

    # Compatibility wrappers retained for tests and internal callers.
    def _generate_control_points(self, path_sequence: list[str], transition_mode: str) -> _ControlPointSequence:
        return self._make_control_point_planner().build(path_sequence, transition_mode)

    def _deduplicate_control_points(self, seq: _ControlPointSequence) -> _ControlPointSequence:
        return self._make_control_point_planner().deduplicate(seq)

    def _wrap_angle_delta(self, delta: np.ndarray | float) -> np.ndarray | float:
        return self._camera_builder._wrap_angle_delta(delta)

    def _smooth_headings(self, headings: np.ndarray, fps: int) -> np.ndarray:
        return self._camera_builder._smooth_headings(headings, fps)

    def _apply_heading_constraints(self, forward_vectors: np.ndarray, fps: int) -> np.ndarray:
        return self._camera_builder.apply_heading_constraints(forward_vectors, fps)

    def _interpolate_path(
        self,
        seq: _ControlPointSequence,
        fps: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray], float]:
        return self._camera_builder.interpolate_path(seq, fps)

    def _build_camera_frames(
        self,
        smooth_pos: np.ndarray,
        smooth_look: Optional[np.ndarray],
        fps: int,
    ) -> list[dict[str, Any]]:
        return self._camera_builder.build_camera_frames(smooth_pos, smooth_look, fps)

    def _build_static_frames(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        num_frames: int,
    ) -> list[dict[str, Any]]:
        return self._camera_builder.build_static_frames(position, look_at, num_frames)

    def generate_exploration_path(self, fps: int = 30) -> list[dict[str, Any]]:
        """Generate full frame sequence for a walkthrough trajectory."""
        if not self.start_room_id:
            self.last_validation_warnings = []
            return []

        self.last_component_transfers.clear()
        self.last_validation_warnings = []
        self.last_skipped_disconnected_rooms = []
        self.last_disconnected_component_count = 1

        transition_mode = self.behavior.disconnected_transition_mode.lower()
        if transition_mode not in {"bridge", "jump"}:
            raise ValueError(
                f"Unsupported disconnected_transition_mode: {self.behavior.disconnected_transition_mode}. "
                "Expected one of: bridge, jump."
            )
        component_policy = self.behavior.disconnected_component_policy.lower()
        if component_policy not in {"largest_component_only", "all_components"}:
            raise ValueError(
                "Unsupported disconnected_component_policy: "
                f"{self.behavior.disconnected_component_policy}. "
                "Expected one of: largest_component_only, all_components."
            )

        active_room_ids = set(self.graph.rooms.keys())
        if component_policy == "largest_component_only":
            active_room_ids = self._largest_component_room_ids(active_room_ids, self.start_room_id)
            skipped_rooms = sorted(set(self.graph.rooms.keys()) - active_room_ids)
            self.last_skipped_disconnected_rooms = skipped_rooms
            if skipped_rooms:
                logger.warning(
                    "Skipping %d room(s) outside largest connected component: %s",
                    len(skipped_rooms),
                    ", ".join(skipped_rooms),
                )
        active_start_room_id = self._find_start_room(active_room_ids)
        if not active_start_room_id:
            self.last_validation_warnings = []
            return []
        components = self._connected_components(active_room_ids)
        self.last_disconnected_component_count = max(1, len(components))

        if component_policy == "all_components" and len(components) > 1:
            logger.warning(
                "Graph has %d disconnected components; restarting trajectory per component (no cross-component links).",
                len(components),
            )
            start_comp_idx = next(
                (i for i, comp in enumerate(components) if active_start_room_id in comp),
                0,
            )
            start_comp = components[start_comp_idx]
            other_comps = [components[i] for i in range(len(components)) if i != start_comp_idx]
            other_comps = sorted(other_comps, key=lambda comp: sorted(comp))
            ordered_components = [start_comp] + other_comps

            frames: list[dict[str, Any]] = []
            self.last_component_transfers.clear()
            for comp in ordered_components:
                comp_start = self._find_start_room(comp)
                if not comp_start:
                    continue
                comp_frames = self._generate_frames_for_room_subset(
                    room_ids=comp,
                    start_room_id=comp_start,
                    transition_mode=transition_mode,
                    fps=fps,
                )
                if not comp_frames:
                    continue
                frames.extend(self._reindex_frames(comp_frames, id_offset=len(frames)))
        else:
            frames = self._generate_frames_for_room_subset(
                room_ids=active_room_ids,
                start_room_id=active_start_room_id,
                transition_mode=transition_mode,
                fps=fps,
            )

        if not frames:
            return []

        self.last_validation_warnings = validate_trajectory(frames)
        for warning in self.last_validation_warnings:
            logger.warning("Trajectory validation warning: %s", warning)
        return frames

    def _get_connection(self, room_a: str, room_b: str) -> Any:
        return self._connection_lookup.get(frozenset((room_a, room_b)))
