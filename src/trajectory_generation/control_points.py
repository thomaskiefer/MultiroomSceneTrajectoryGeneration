"""Control-point generation and deduplication for walkthrough trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Optional

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
        suppress_terminal_revisit_motion: bool = False,
    ) -> ControlPointSequence:
        """Build control points from room visit sequence."""
        control_points_pos: list[np.ndarray] = []
        control_points_look: list[np.ndarray] = []
        segment_speeds: list[float] = []

        seen_rooms: set[str] = set()
        for i, room_id in enumerate(path_sequence):
            center_pos = self._get_room_center(room_id)
            consumed_next_transition = False

            is_first_visit = room_id not in seen_rooms
            seen_rooms.add(room_id)

            is_terminal_revisit = (not is_first_visit) and (i == len(path_sequence) - 1)
            if is_terminal_revisit and suppress_terminal_revisit_motion:
                # For loop-closure runs, avoid adding an extra end-of-sequence
                # revisit arc in the final room; closure logic will bring the
                # path to start pose from the returned doorway point.
                pass
            elif is_first_visit:
                preferred_departure_angle = self._get_departure_angle(
                    seq_idx=i,
                    room_id=room_id,
                    center_pos=center_pos,
                    path_sequence=path_sequence,
                )
                self._add_spin_points(
                    center_pos,
                    control_points_pos,
                    control_points_look,
                    segment_speeds,
                    preferred_departure_angle=preferred_departure_angle,
                )
            else:
                if self.behavior.revisit_transition_mode == "door_shortcut":
                    consumed_next_transition = self._add_passthrough_door_shortcut(
                        i,
                        room_id,
                        center_pos,
                        path_sequence,
                        control_points_pos,
                        control_points_look,
                        segment_speeds,
                    )
                if not consumed_next_transition:
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
                if next_room_id != room_id and not consumed_next_transition:
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
        preferred_departure_angle: Optional[float] = None,
    ) -> None:
        num_spin_points = int(self.behavior.spin_points)
        if num_spin_points < 2:
            logger.warning(
                "Invalid spin_points=%s; clamping to 2 to avoid degenerate spin orbit.",
                num_spin_points,
            )
            num_spin_points = 2
        radius = self.behavior.spin_look_radius

        # Angle parameter controls look-target phase; camera position is opposite.
        # Prefer aligning spin start/end camera position with departure direction
        # toward the next transition target. This keeps room-orbit exits consistent
        # with downstream door/room crossings.
        start_angle = 0.0
        if preferred_departure_angle is not None:
            start_angle = float(preferred_departure_angle - np.pi)
        elif len(cp_pos) > 0:
            incoming_vec = center_pos[:2] - np.array(cp_pos[-1][:2], dtype=float)
            if np.linalg.norm(incoming_vec) > 1e-6:
                start_angle = float(np.arctan2(incoming_vec[1], incoming_vec[0]))

        spin_sign = self._choose_spin_direction(
            start_angle=start_angle,
            previous_pos=(np.asarray(cp_pos[-1], dtype=float) if len(cp_pos) > 0 else None),
            spin_center=center_pos,
            preferred_departure_angle=preferred_departure_angle,
            orbit_radius=radius * self.behavior.spin_orbit_scale,
        )

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
            angle_rad = start_angle + spin_sign * k * (2 * np.pi / num_spin_points)
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
        orbit_radius = radius * orbit_scale

        entry_angle = 0.0
        if len(cp_pos) > 0:
            incoming = center_pos[:2] - np.array(cp_pos[-1][:2], dtype=float)
            if np.linalg.norm(incoming) > 1e-6:
                entry_angle = float(np.arctan2(incoming[1], incoming[0]))

        # Angles here parameterize LOOK direction. Camera position is opposite.
        # To exit on the side of the next transition target, we therefore shift
        # look-angle by +pi relative to the target direction.
        exit_angle = entry_angle
        exit_vec = None
        transition_target_xy: Optional[np.ndarray] = None
        if seq_idx < len(path_sequence) - 1:
            next_room_id = path_sequence[seq_idx + 1]
            if next_room_id != room_id:
                conn = self._get_connection(room_id, next_room_id)
                if conn:
                    door_xy = conn.waypoint.position
                    exit_vec = np.array([door_xy[0], door_xy[1]], dtype=float) - center_pos[:2]
                    transition_target_xy = np.array([door_xy[0], door_xy[1]], dtype=float)
                else:
                    next_center = self._get_room_center(next_room_id)
                    exit_vec = next_center[:2] - center_pos[:2]
                    transition_target_xy = np.array(next_center[:2], dtype=float)
                if exit_vec is not None and np.linalg.norm(exit_vec) > 1e-6:
                    target_angle = float(np.arctan2(exit_vec[1], exit_vec[0]))
                    exit_angle = target_angle + np.pi

        delta = float((exit_angle - entry_angle + np.pi) % (2 * np.pi) - np.pi)
        if self.behavior.revisit_transition_mode == "center_arc":
            entry_angle, delta = self._optimize_revisit_arc_segment(
                entry_angle_nominal=entry_angle,
                exit_angle_nominal=exit_angle,
                center_xy=np.array(center_pos[:2], dtype=float),
                orbit_radius=orbit_radius,
                previous_pos_xy=(np.asarray(cp_pos[-1][:2], dtype=float) if len(cp_pos) > 0 else None),
                transition_target_xy=transition_target_xy,
            )
        abs_delta = abs(delta)
        min_turn_rad = np.deg2rad(self.behavior.passthrough_min_turn_deg)
        if abs_delta < min_turn_rad:
            # Keep a compact easing arc for tiny heading changes instead of an abrupt kink.
            turn_sign = 1.0 if delta >= 0.0 else -1.0
            delta = turn_sign * min_turn_rad
            abs_delta = min_turn_rad

        # Reuse the same nominal orbit radius as first-visit spins so revisit
        # arcs stay on the same room-centered trajectory family.
        num_arc_points = max(3, int(np.ceil(abs_delta / (2 * np.pi) * self.behavior.spin_points)))
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

        # Keep door progression monotonic along flow direction to avoid
        # "passed the door, then went back" artifacts in shortcut/revisit paths.
        last_progress = float("-inf")
        if len(cp_pos) > 0:
            last_pos = np.asarray(cp_pos[-1], dtype=float)
            last_progress = float(np.dot((last_pos - door_pos)[:2], flow_dir[:2]))

        candidates = [
            (-self.behavior.door_buffer, p_approach, door_pos),
            (0.0, door_pos, look_target),
            (self.behavior.door_buffer, p_depart, look_target),
        ]
        added = False
        for progress, point, look in candidates:
            # Skip any point that would move backwards along the crossing axis.
            if progress <= last_progress + 1e-4:
                continue
            if len(cp_pos) > 0:
                seg_speeds.append(self.behavior.travel_speed)
            cp_pos.append(point)
            cp_look.append(look)
            last_progress = progress
            added = True

        # Safety fallback: ensure at least departure gets inserted.
        if not added:
            if len(cp_pos) > 0:
                seg_speeds.append(self.behavior.travel_speed)
            cp_pos.append(p_depart)
            cp_look.append(look_target)

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

    def _add_passthrough_door_shortcut(
        self,
        seq_idx: int,
        room_id: str,
        center_pos: np.ndarray,
        path_sequence: list[str],
        cp_pos: list,
        cp_look: list,
        seg_speeds: list,
    ) -> bool:
        """Route revisit transitions through observed room doors instead of center arcs."""
        if seq_idx <= 0 or seq_idx >= len(path_sequence) - 1:
            return False

        prev_room_id = path_sequence[seq_idx - 1]
        next_room_id = path_sequence[seq_idx + 1]
        if prev_room_id == room_id or next_room_id == room_id:
            return False

        conn_in = self._get_connection(prev_room_id, room_id)
        conn_out = self._get_connection(room_id, next_room_id)
        if conn_in is None or conn_out is None:
            return False

        in_xy = conn_in.waypoint.position
        out_xy = conn_out.waypoint.position
        in_pos = np.array([in_xy[0], in_xy[1], self.eye_level], dtype=float)
        out_pos = np.array([out_xy[0], out_xy[1], self.eye_level], dtype=float)

        transit_vec = out_pos - in_pos
        transit_vec[2] = 0.0
        transit_norm = float(np.linalg.norm(transit_vec))
        if transit_norm > 1e-6:
            transit_dir = transit_vec / transit_norm
        else:
            fallback = center_pos - in_pos
            fallback[2] = 0.0
            fallback_norm = float(np.linalg.norm(fallback))
            transit_dir = fallback / fallback_norm if fallback_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

        if len(cp_pos) > 0:
            last = np.asarray(cp_pos[-1], dtype=float)
            dist_last_in = float(np.linalg.norm(last - in_pos))
            # Progress along in->out axis. Positive means we are already past
            # the incoming door waypoint toward the outgoing door.
            progress_last = float(np.dot((last - in_pos)[:2], transit_dir[:2]))
            should_insert_in = dist_last_in > 0.08 and progress_last < -0.05
            if should_insert_in:
                cp_pos.append(in_pos)
                cp_look.append(in_pos + transit_dir)
                seg_speeds.append(self.behavior.travel_speed)
        else:
            cp_pos.append(in_pos)
            cp_look.append(in_pos + transit_dir)

        # Intentionally avoid inserting an explicit midpoint between doors.
        # Keeping this transition direct reduces artificial oscillation.

        # Reuse canonical door-crossing logic for the outgoing transition to next room.
        self._add_door_crossing(
            conn_out,
            center_pos,
            next_room_id,
            cp_pos,
            cp_look,
            seg_speeds,
        )
        return True

    def _get_departure_angle(
        self,
        seq_idx: int,
        room_id: str,
        center_pos: np.ndarray,
        path_sequence: list[str],
    ) -> Optional[float]:
        """Return angle from current room center toward the next transition target."""
        if seq_idx >= len(path_sequence) - 1:
            return None
        next_room_id = path_sequence[seq_idx + 1]
        if next_room_id == room_id:
            return None

        conn = self._get_connection(room_id, next_room_id)
        if conn is not None:
            door_xy = conn.waypoint.position
            vec = np.array([door_xy[0], door_xy[1]], dtype=float) - center_pos[:2]
        else:
            next_center = self._get_room_center(next_room_id)
            vec = next_center[:2] - center_pos[:2]

        if np.linalg.norm(vec) <= 1e-6:
            return None
        return float(np.arctan2(vec[1], vec[0]))

    def _choose_spin_direction(
        self,
        start_angle: float,
        previous_pos: Optional[np.ndarray],
        spin_center: np.ndarray,
        preferred_departure_angle: Optional[float],
        orbit_radius: float,
    ) -> float:
        """Choose CW/CCW direction that better matches incoming/outgoing motion."""
        if orbit_radius <= 1e-9:
            return 1.0

        spin0 = spin_center + np.array(
            [
                orbit_radius * np.cos(start_angle + np.pi),
                orbit_radius * np.sin(start_angle + np.pi),
                0.0,
            ],
            dtype=float,
        )

        incoming_dir = None
        if previous_pos is not None:
            vec = spin0[:2] - previous_pos[:2]
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                incoming_dir = vec / norm

        outgoing_dir = None
        if preferred_departure_angle is not None:
            outgoing_dir = np.array(
                [np.cos(preferred_departure_angle), np.sin(preferred_departure_angle)],
                dtype=float,
            )

        def _score(sign: float) -> float:
            tangent = np.array(
                [sign * np.sin(start_angle), -sign * np.cos(start_angle)],
                dtype=float,
            )
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1e-9:
                tangent /= tangent_norm
            score = 0.0
            if incoming_dir is not None:
                score += self._angle_cost(tangent, incoming_dir)
            if outgoing_dir is not None:
                score += self._angle_cost(tangent, outgoing_dir)
            return score

        ccw = _score(1.0)
        cw = _score(-1.0)
        return -1.0 if cw < ccw else 1.0

    @staticmethod
    def _normalize_dir(vec: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            return None
        return vec / norm

    @staticmethod
    def _angle_cost(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return float(np.arccos(dot))

    def _optimize_revisit_arc_segment(
        self,
        entry_angle_nominal: float,
        exit_angle_nominal: float,
        center_xy: np.ndarray,
        orbit_radius: float,
        previous_pos_xy: Optional[np.ndarray],
        transition_target_xy: Optional[np.ndarray],
    ) -> tuple[float, float]:
        """
        Optimize revisit arc segment by searching entry/exit angle offsets and
        selecting the best arc (short/long) by entry/exit tangent smoothness and
        transition-risk terms.
        """
        if orbit_radius <= 1e-9:
            delta = float((exit_angle_nominal - entry_angle_nominal + np.pi) % (2 * np.pi) - np.pi)
            return float(entry_angle_nominal), delta

        search_half = np.deg2rad(float(self.behavior.revisit_arc_angle_search_deg))
        steps = int(self.behavior.revisit_arc_search_steps)
        if search_half <= 1e-12 or steps <= 1:
            offsets = np.array([0.0], dtype=float)
        else:
            offsets = np.linspace(-search_half, search_half, steps)

        mismatch_max = np.deg2rad(float(self.behavior.revisit_arc_max_tangent_mismatch_deg))
        reverse_pref = np.deg2rad(float(self.behavior.revisit_arc_reverse_pref_deg))
        reverse_long_bonus = float(self.behavior.revisit_arc_reverse_long_arc_bonus)
        risk_distance_weight = float(self.behavior.revisit_arc_transition_risk_distance_weight)
        risk_angle_weight = float(self.behavior.revisit_arc_transition_risk_angle_weight)

        best_entry = float(entry_angle_nominal)
        best_delta = float((exit_angle_nominal - entry_angle_nominal + np.pi) % (2 * np.pi) - np.pi)
        best_score = float("inf")
        found_valid = False

        # Determine if this transition is a near-reversal at nominal geometry.
        reverse_regime = False
        reversal_strength = 0.0
        if previous_pos_xy is not None and transition_target_xy is not None:
            entry_cam_nom = center_xy + orbit_radius * np.array(
                [-np.cos(entry_angle_nominal), -np.sin(entry_angle_nominal)],
                dtype=float,
            )
            exit_cam_nom = center_xy + orbit_radius * np.array(
                [-np.cos(exit_angle_nominal), -np.sin(exit_angle_nominal)],
                dtype=float,
            )
            incoming_nom = self._normalize_dir(entry_cam_nom - previous_pos_xy)
            outgoing_nom = self._normalize_dir(transition_target_xy - exit_cam_nom)
            if incoming_nom is not None and outgoing_nom is not None:
                reversal_angle = self._angle_cost(incoming_nom, outgoing_nom)
                reverse_regime = reversal_angle >= reverse_pref
                if reverse_regime:
                    denom = max(np.pi - reverse_pref, 1e-6)
                    reversal_strength = float(np.clip((reversal_angle - reverse_pref) / denom, 0.0, 1.0))

        def _search_candidates() -> None:
            nonlocal best_entry, best_delta, best_score, found_valid
            for d_entry in offsets:
                entry_angle = float(entry_angle_nominal + d_entry)
                entry_cam = center_xy + orbit_radius * np.array(
                    [-np.cos(entry_angle), -np.sin(entry_angle)],
                    dtype=float,
                )
                incoming_dir = None
                if previous_pos_xy is not None:
                    incoming_dir = self._normalize_dir(entry_cam - previous_pos_xy)

                for d_exit in offsets:
                    exit_angle = float(exit_angle_nominal + d_exit)
                    delta_short = float((exit_angle - entry_angle + np.pi) % (2.0 * np.pi) - np.pi)
                    if abs(delta_short) < 1e-6:
                        # Keep a true short/zero candidate when entry and exit
                        # are aligned; otherwise the optimizer would force a
                        # full ±2pi loop at room revisits near sequence end.
                        short_candidates: list[float] = [0.0]
                        long_candidates = [2.0 * np.pi, -2.0 * np.pi]
                    else:
                        short_candidates = [delta_short]
                        long_candidates = [float(delta_short - np.sign(delta_short) * (2.0 * np.pi))]

                    candidate_deltas = [*long_candidates, *short_candidates]

                    exit_cam = center_xy + orbit_radius * np.array(
                        [-np.cos(exit_angle), -np.sin(exit_angle)],
                        dtype=float,
                    )
                    outgoing_dir = None
                    normalized_exit_distance = 0.0
                    if transition_target_xy is not None:
                        exit_vec = transition_target_xy - exit_cam
                        outgoing_dir = self._normalize_dir(exit_vec)
                        room_scale = max(
                            float(np.linalg.norm(transition_target_xy - center_xy)),
                            float(self.behavior.spin_look_radius),
                            1e-3,
                        )
                        normalized_exit_distance = float(np.linalg.norm(exit_vec) / room_scale)

                    for delta in candidate_deltas:
                        is_long_arc = abs(delta) > (np.pi + 1e-6)
                        sign = 1.0 if delta >= 0.0 else -1.0
                        start_tangent = np.array(
                            [sign * np.sin(entry_angle), -sign * np.cos(entry_angle)],
                            dtype=float,
                        )

                        end_angle = entry_angle + delta
                        end_tangent = np.array(
                            [sign * np.sin(end_angle), -sign * np.cos(end_angle)],
                            dtype=float,
                        )

                        mismatch_in = 0.0
                        if incoming_dir is not None:
                            mismatch_in = self._angle_cost(start_tangent, incoming_dir)

                        mismatch_out = 0.0
                        if outgoing_dir is not None:
                            mismatch_out = self._angle_cost(end_tangent, outgoing_dir)

                        valid = mismatch_in <= mismatch_max and mismatch_out <= mismatch_max
                        # Keep inbound mismatch as the base term and model exit
                        # mismatch with an explicit combined weight:
                        #   1.0 (base smoothness) + risk_angle_weight (exit risk)
                        # This is equivalent to the previous additive form while
                        # making the intent unambiguous.
                        score = mismatch_in + (1.0 + risk_angle_weight) * mismatch_out
                        score += risk_distance_weight * normalized_exit_distance
                        if reverse_regime and reverse_long_bonus > 0.0:
                            arc_bias = reverse_long_bonus * (1.0 + reversal_strength)
                            if is_long_arc:
                                score -= arc_bias
                            else:
                                score += arc_bias

                        if valid:
                            found_valid = True
                        elif found_valid:
                            continue
                        else:
                            score += 2.0

                        # Deterministic tie-break only.
                        tie_break = abs(float(d_entry)) + abs(float(d_exit)) + 1e-3 * abs(delta)
                        best_tie = (
                            abs(float(best_entry - entry_angle_nominal))
                            + abs(float((best_entry + best_delta) - exit_angle_nominal))
                            + 1e-3 * abs(best_delta)
                        )
                        if (score + 1e-12) < best_score or (
                            abs(score - best_score) <= 1e-12 and tie_break < best_tie
                        ):
                            best_score = score
                            best_entry = entry_angle
                            best_delta = float(delta)

        _search_candidates()

        return best_entry, best_delta
