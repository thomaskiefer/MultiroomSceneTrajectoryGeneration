"""Control-point generation and deduplication for walkthrough trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable

import numpy as np

from .config import WalkthroughBehaviorConfig


logger = logging.getLogger(__name__)


@dataclass
class ControlPointSequence:
    """Intermediate control-point sequence used before spline interpolation."""

    positions: np.ndarray       # (N, 3)
    look_targets: np.ndarray    # (N, 3)
    segment_speeds: np.ndarray  # (N-1,)


class ControlPointPlanner:
    """Build and sanitize control points for local walkthrough generation."""

    def __init__(
        self,
        behavior: WalkthroughBehaviorConfig,
        eye_level: float,
        get_room_center: Callable[[str], np.ndarray],
        get_connection: Callable[[str, str], Any],
        get_door_normal: Callable[[Any], np.ndarray],
        component_transfers: list[dict[str, Any]],
    ) -> None:
        self.behavior = behavior
        self.eye_level = eye_level
        self._get_room_center = get_room_center
        self._get_connection = get_connection
        self._get_door_normal = get_door_normal
        self._component_transfers = component_transfers

    def build(
        self,
        path_sequence: list[str],
        transition_mode: str,
    ) -> ControlPointSequence:
        """Build control points from room visit sequence."""
        control_points_pos: list[np.ndarray] = []
        control_points_look: list[np.ndarray] = []
        segment_speeds: list[float] = []

        seen_rooms: set[str] = set()
        for i, room_id in enumerate(path_sequence):
            center_pos = self._get_room_center(room_id)

            is_first_visit = room_id not in seen_rooms
            seen_rooms.add(room_id)

            if is_first_visit:
                self._add_spin_points(
                    center_pos,
                    control_points_pos,
                    control_points_look,
                    segment_speeds,
                )
            else:
                self._add_passthrough_arc(
                    i,
                    room_id,
                    center_pos,
                    path_sequence,
                    control_points_pos,
                    control_points_look,
                    segment_speeds,
                )

            if i < len(path_sequence) - 1:
                next_room_id = path_sequence[i + 1]
                if next_room_id != room_id:
                    conn = self._get_connection(room_id, next_room_id)
                    if conn:
                        self._add_door_crossing(
                            conn,
                            center_pos,
                            next_room_id,
                            control_points_pos,
                            control_points_look,
                            segment_speeds,
                        )
                    else:
                        self._add_component_transfer(
                            room_id,
                            next_room_id,
                            center_pos,
                            transition_mode,
                            control_points_pos,
                            control_points_look,
                            segment_speeds,
                        )

        if not control_points_pos:
            return ControlPointSequence(
                positions=np.empty((0, 3)),
                look_targets=np.empty((0, 3)),
                segment_speeds=np.empty(0),
            )

        cp_pos_arr = np.array(control_points_pos)
        cp_look_arr = np.array(control_points_look)

        num_expected_segments = len(cp_pos_arr) - 1
        if len(segment_speeds) < num_expected_segments:
            pad_count = num_expected_segments - len(segment_speeds)
            logger.warning(
                "segment_speeds shorter than expected (%s < %s); padding %s entries.",
                len(segment_speeds),
                num_expected_segments,
                pad_count,
            )
            segment_speeds.extend([self.behavior.travel_speed] * pad_count)
        if len(segment_speeds) > num_expected_segments:
            logger.warning(
                "segment_speeds longer than expected (%s > %s); clipping.",
                len(segment_speeds),
                num_expected_segments,
            )
            segment_speeds = segment_speeds[:num_expected_segments]

        return ControlPointSequence(
            positions=cp_pos_arr,
            look_targets=cp_look_arr,
            segment_speeds=np.array(segment_speeds),
        )

    def deduplicate(self, seq: ControlPointSequence) -> ControlPointSequence:
        """Remove consecutive duplicate control points and apply speed safety overrides."""
        if len(seq.positions) == 0:
            return seq

        cp_pos = seq.positions
        cp_look = seq.look_targets
        seg_speed_arr = seq.segment_speeds
        orig_seg_lengths = np.linalg.norm(np.diff(cp_pos, axis=0), axis=1) if len(cp_pos) >= 2 else np.empty(0)

        valid_indices = [0]
        for k in range(1, len(cp_pos)):
            if np.linalg.norm(cp_pos[k] - cp_pos[k - 1]) > 1e-3 or \
               np.linalg.norm(cp_look[k] - cp_look[k - 1]) > 1e-3:
                valid_indices.append(k)

        cp_pos = cp_pos[valid_indices]
        cp_look = cp_look[valid_indices]

        # Keep transfer index ranges aligned with post-dedup control-point indices.
        if self._component_transfers and valid_indices:
            max_new = len(valid_indices) - 1

            def _map_start_old_idx(old_idx: int) -> int:
                mapped = int(np.searchsorted(valid_indices, old_idx, side="left"))
                return max(0, min(mapped, max_new))

            def _map_end_old_idx(old_idx: int) -> int:
                mapped = int(np.searchsorted(valid_indices, old_idx, side="right") - 1)
                return max(0, min(mapped, max_new))

            for transfer in self._component_transfers:
                cp_range = transfer.get("control_point_indices")
                if not isinstance(cp_range, list) or len(cp_range) != 2:
                    continue
                start_old, end_old = int(cp_range[0]), int(cp_range[1])
                start_new = _map_start_old_idx(start_old)
                end_new = _map_end_old_idx(end_old)
                if end_new < start_new:
                    end_new = start_new
                transfer["control_point_indices"] = [start_new, end_new]

        remapped: list[float] = []
        for i in range(len(valid_indices) - 1):
            start = valid_indices[i]
            end = valid_indices[i + 1]
            span_speeds = seg_speed_arr[start:end]
            span_lengths = orig_seg_lengths[start:end]
            if len(span_speeds) == 0:
                remapped.append(self.behavior.travel_speed)
            elif len(span_lengths) == len(span_speeds) and float(np.sum(span_lengths)) > 1e-9:
                remapped.append(float(np.average(span_speeds, weights=span_lengths)))
            else:
                remapped.append(float(np.median(span_speeds)))
        seg_speed_arr = np.array(remapped, dtype=float)

        seg_lengths = np.linalg.norm(np.diff(cp_pos, axis=0), axis=1)
        long_mask = seg_lengths > self.behavior.long_segment_threshold
        slow_mask = seg_speed_arr < self.behavior.slow_speed_threshold
        override_mask = long_mask & slow_mask
        if np.any(override_mask):
            logger.debug(
                "Overriding speed for %s long segments detected with slow speed.",
                int(np.sum(override_mask)),
            )
            seg_speed_arr[override_mask] = self.behavior.travel_speed

        return ControlPointSequence(
            positions=cp_pos,
            look_targets=cp_look,
            segment_speeds=seg_speed_arr,
        )

    def _add_spin_points(
        self,
        center_pos: np.ndarray,
        cp_pos: list,
        cp_look: list,
        seg_speeds: list,
    ) -> None:
        num_spin_points = self.behavior.spin_points
        radius = self.behavior.spin_look_radius

        start_angle = 0.0
        if len(cp_pos) > 0:
            incoming_vec = center_pos[:2] - np.array(cp_pos[-1][:2], dtype=float)
            if np.linalg.norm(incoming_vec) > 1e-6:
                start_angle = float(np.arctan2(incoming_vec[1], incoming_vec[0]))

        dx0 = radius * np.cos(start_angle)
        dy0 = radius * np.sin(start_angle)
        spin0_offset = np.array([-dx0, -dy0, 0.0]) * self.behavior.spin_orbit_scale
        spin0_pos = center_pos + spin0_offset

        if len(cp_pos) > 0:
            last_pos = np.array(cp_pos[-1], dtype=float)
            dist_to_spin = float(np.linalg.norm(spin0_pos - last_pos))
            if dist_to_spin > 0.1:
                lead_in = (last_pos + spin0_pos) / 2.0
                cp_pos.append(lead_in)
                cp_look.append(center_pos.copy())
                seg_speeds.append(self.behavior.travel_speed)
                seg_speeds.append(self.behavior.travel_speed)
            else:
                seg_speeds.append(self.behavior.travel_speed)

        for k in range(num_spin_points + 1):
            angle_rad = start_angle + k * (2 * np.pi / num_spin_points)
            dx = radius * np.cos(angle_rad)
            dy = radius * np.sin(angle_rad)
            look_target = center_pos + np.array([dx, dy, 0])
            orbit_offset = np.array([-dx, -dy, 0]) * self.behavior.spin_orbit_scale
            cam_pos = center_pos + orbit_offset
            cp_pos.append(cam_pos)
            cp_look.append(look_target)
            if k < num_spin_points:
                seg_speeds.append(self.behavior.spin_segment_speed)

    def _add_passthrough_arc(
        self,
        seq_idx: int,
        room_id: str,
        center_pos: np.ndarray,
        path_sequence: list[str],
        cp_pos: list,
        cp_look: list,
        seg_speeds: list,
    ) -> None:
        radius = self.behavior.spin_look_radius
        orbit_scale = self.behavior.spin_orbit_scale

        entry_angle = 0.0
        if len(cp_pos) > 0:
            incoming = center_pos[:2] - np.array(cp_pos[-1][:2], dtype=float)
            if np.linalg.norm(incoming) > 1e-6:
                entry_angle = float(np.arctan2(incoming[1], incoming[0]))

        exit_angle = entry_angle + np.pi
        exit_vec = None
        if seq_idx < len(path_sequence) - 1:
            next_room_id = path_sequence[seq_idx + 1]
            if next_room_id != room_id:
                conn = self._get_connection(room_id, next_room_id)
                if conn:
                    door_xy = conn.waypoint.position
                    exit_vec = np.array([door_xy[0], door_xy[1]], dtype=float) - center_pos[:2]
                else:
                    next_center = self._get_room_center(next_room_id)
                    exit_vec = next_center[:2] - center_pos[:2]
                if exit_vec is not None and np.linalg.norm(exit_vec) > 1e-6:
                    exit_angle = float(np.arctan2(exit_vec[1], exit_vec[0]))

        delta = float((exit_angle - entry_angle + np.pi) % (2 * np.pi) - np.pi)
        if abs(delta) < np.deg2rad(30):
            # Deterministic tie-break for near-collinear entry/exit directions.
            delta = np.deg2rad(90)

        num_arc_points = max(3, int(abs(delta) / (2 * np.pi) * self.behavior.spin_points))
        angles = np.linspace(entry_angle, entry_angle + delta, num_arc_points + 1)

        if len(cp_pos) > 0:
            seg_speeds.append(self.behavior.travel_speed)

        for k, angle in enumerate(angles):
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            look_target = center_pos + np.array([dx, dy, 0.0])
            orbit_offset = np.array([-dx, -dy, 0.0]) * orbit_scale
            cam_pos = center_pos + orbit_offset
            cp_pos.append(cam_pos)
            cp_look.append(look_target)
            if k < len(angles) - 1:
                seg_speeds.append(self.behavior.passthrough_speed)

    def _add_door_crossing(
        self,
        conn: Any,
        center_pos: np.ndarray,
        next_room_id: str,
        cp_pos: list,
        cp_look: list,
        seg_speeds: list,
    ) -> None:
        door_xy = conn.waypoint.position
        door_pos = np.array([door_xy[0], door_xy[1], self.eye_level])
        next_center = self._get_room_center(next_room_id)

        door_normal = self._get_door_normal(conn.waypoint)

        if np.linalg.norm(door_normal) < 0.001:
            vec_in = (door_pos - center_pos)[:2]
            vec_out = (next_center - door_pos)[:2]
            n_in = np.linalg.norm(vec_in)
            n_out = np.linalg.norm(vec_out)
            if n_in > 0:
                vec_in /= n_in
            if n_out > 0:
                vec_out /= n_out
            flow_dir = (vec_in + vec_out)
            if np.linalg.norm(flow_dir) < 1e-6:
                flow_dir = vec_out if np.linalg.norm(vec_out) > 1e-6 else vec_in
            flow_dir = np.array([flow_dir[0], flow_dir[1], 0.0], dtype=float)
        else:
            travel_vec = next_center - center_pos
            if np.dot(door_normal, travel_vec[:2]) < 0:
                door_normal = -door_normal
            flow_dir = np.array([door_normal[0], door_normal[1], 0.0], dtype=float)

        norm_flow = np.linalg.norm(flow_dir)
        if norm_flow > 0.001:
            flow_dir /= norm_flow
        else:
            fallback = next_center - center_pos
            fallback[2] = 0.0
            fallback_norm = np.linalg.norm(fallback)
            if fallback_norm > 1e-6:
                flow_dir = fallback / fallback_norm
            else:
                flow_dir = np.array([1.0, 0.0, 0.0], dtype=float)

        p_approach = door_pos - flow_dir * self.behavior.door_buffer
        p_depart = door_pos + flow_dir * self.behavior.door_buffer

        vec_to_center = next_center - door_pos
        dist_to_center = np.linalg.norm(vec_to_center)
        if dist_to_center > 1e-3:
            dir_to_center = vec_to_center / dist_to_center
            look_dist = min(dist_to_center, self.behavior.lookahead_inside_next_room)
            look_target = door_pos + dir_to_center * look_dist
        else:
            look_dist = max(self.behavior.door_buffer, 0.25)
            look_target = door_pos + flow_dir * look_dist

        if len(cp_pos) > 0:
            seg_speeds.append(self.behavior.travel_speed)  # previous -> approach
        cp_pos.extend([p_approach, door_pos, p_depart])
        cp_look.extend([door_pos, look_target, look_target])
        seg_speeds.extend([self.behavior.travel_speed, self.behavior.travel_speed])

    def _add_component_transfer(
        self,
        room_id: str,
        next_room_id: str,
        center_pos: np.ndarray,
        transition_mode: str,
        cp_pos: list,
        cp_look: list,
        seg_speeds: list,
    ) -> None:
        if transition_mode != "bridge":
            return

        next_center = self._get_room_center(next_room_id)
        transfer_vec = next_center - center_pos
        transfer_norm = np.linalg.norm(transfer_vec)
        if transfer_norm > 1e-6:
            transfer_dir = transfer_vec / transfer_norm
        else:
            transfer_dir = np.array([1.0, 0.0, 0.0], dtype=float)

        mid_transfer = (center_pos + next_center) / 2.0
        new_points: list[np.ndarray] = []
        new_looks: list[np.ndarray] = []
        if len(cp_pos) == 0 or float(np.linalg.norm(np.asarray(cp_pos[-1], dtype=float) - center_pos)) > 1e-6:
            new_points.append(center_pos)
            new_looks.append(center_pos + transfer_dir)
        new_points.append(mid_transfer)
        new_looks.append(mid_transfer + transfer_dir)
        new_points.append(next_center)
        new_looks.append(next_center + transfer_dir)

        if not new_points:
            return

        start_cp_idx = len(cp_pos)
        if len(cp_pos) > 0:
            seg_speeds.append(self.behavior.travel_speed)  # previous -> first transfer point
        cp_pos.extend(new_points)
        cp_look.extend(new_looks)
        seg_speeds.extend([self.behavior.travel_speed] * (len(new_points) - 1))
        end_cp_idx = len(cp_pos) - 1
        self._component_transfers.append(
            {
                "from_room": room_id,
                "to_room": next_room_id,
                "control_point_indices": [start_cp_idx, end_cp_idx],
            }
        )
