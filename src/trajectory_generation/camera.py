"""Camera orientation, interpolation, and frame construction helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .config import WalkthroughBehaviorConfig
from .control_points import ControlPointSequence
from .spline import CatmullRomSpline


logger = logging.getLogger(__name__)


class CameraPathBuilder:
    """Interpolate control points and build camera frames."""

    def __init__(self, behavior: WalkthroughBehaviorConfig) -> None:
        self.behavior = behavior

    @staticmethod
    def _wrap_angle_delta(delta: np.ndarray | float) -> np.ndarray | float:
        return (delta + np.pi) % (2.0 * np.pi) - np.pi

    def _smooth_headings(self, headings: np.ndarray, fps: int) -> np.ndarray:
        if len(headings) < 3:
            return headings
        window = int(round(self.behavior.angular_smoothing_window_s * max(fps, 1)))
        if window < 2:
            return headings
        if window % 2 == 0:
            window += 1
        window = min(window, len(headings) if len(headings) % 2 == 1 else len(headings) - 1)
        if window < 3:
            return headings
        kernel = np.ones(window, dtype=float) / float(window)
        pad = window // 2
        padded = np.pad(headings, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        if len(smoothed) != len(headings):
            raise RuntimeError(
                f"Heading smoothing length mismatch: expected {len(headings)}, got {len(smoothed)}."
            )
        return smoothed

    def apply_heading_constraints(self, forward_vectors: np.ndarray, fps: int) -> np.ndarray:
        """Smooth and cap yaw-rate to avoid abrupt rotational motion."""
        if len(forward_vectors) == 0:
            return forward_vectors

        constrained = np.array(forward_vectors, dtype=float, copy=True)
        last_full = np.array([1.0, 0.0, 0.0], dtype=float)
        for i in range(len(constrained)):
            norm3 = float(np.linalg.norm(constrained[i]))
            if norm3 > 1e-6:
                constrained[i] = constrained[i] / norm3
                last_full = constrained[i]
            else:
                constrained[i] = last_full

        xy = constrained[:, :2]
        xy_norm = np.linalg.norm(xy, axis=1)

        last_dir = np.array([1.0, 0.0], dtype=float)
        for i in range(len(xy)):
            if xy_norm[i] > 1e-6:
                xy[i] = xy[i] / xy_norm[i]
                last_dir = xy[i]
            else:
                xy[i] = last_dir

        headings = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        headings = self._smooth_headings(headings, fps=fps)

        max_step = np.deg2rad(self.behavior.max_angular_speed_deg) / max(float(fps), 1.0)
        limited = np.zeros_like(headings)
        limited[0] = headings[0]
        for i in range(1, len(headings)):
            delta = float(self._wrap_angle_delta(headings[i] - limited[i - 1]))
            delta = float(np.clip(delta, -max_step, max_step))
            limited[i] = limited[i - 1] + delta

        z_vals = np.clip(constrained[:, 2], -1.0, 1.0)
        xy_scale = np.sqrt(np.maximum(0.0, 1.0 - z_vals * z_vals))
        constrained[:, 0] = np.cos(limited) * xy_scale
        constrained[:, 1] = np.sin(limited) * xy_scale
        constrained[:, 2] = z_vals

        norms = np.linalg.norm(constrained, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        constrained = constrained / norms
        return constrained

    def interpolate_path(
        self,
        seq: ControlPointSequence,
        fps: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray], float]:
        """Interpolate control points with Catmull-Rom splines and resample at variable speed."""
        cp_pos = seq.positions
        cp_look = seq.look_targets
        seg_speed_arr = seq.segment_speeds

        look_mode = self.behavior.look_at_mode.lower()
        if look_mode not in {"tangent", "spline_target"}:
            raise ValueError(
                f"Unsupported look_at_mode: {self.behavior.look_at_mode}. "
                "Expected one of: tangent, spline_target."
            )

        if len(cp_pos) == 0:
            empty = np.empty((0, 3), dtype=float)
            return empty, (empty.copy() if look_mode == "spline_target" else None), 0.0
        if len(cp_pos) == 1:
            pos = np.asarray(cp_pos, dtype=float).copy()
            look = np.asarray(cp_look, dtype=float).copy() if look_mode == "spline_target" else None
            return pos, look, 0.0

        num_segments = len(cp_pos) - 1
        if len(seg_speed_arr) < num_segments:
            pad_count = num_segments - len(seg_speed_arr)
            padding = np.full(pad_count, self.behavior.travel_speed)
            seg_speed_arr = np.concatenate([seg_speed_arr, padding])
            logger.warning(
                "Segment speed array shorter than segments; padded %s entries.",
                pad_count,
            )
        elif len(seg_speed_arr) > num_segments:
            logger.warning(
                "Segment speed array longer than segments; trimming %s entries.",
                len(seg_speed_arr) - num_segments,
            )
            seg_speed_arr = seg_speed_arr[:num_segments]

        spline_pos = CatmullRomSpline(cp_pos)
        spline_look = (
            CatmullRomSpline(cp_look) if look_mode == "spline_target" else None
        )

        est_dist = np.sum(np.linalg.norm(np.diff(cp_pos, axis=0), axis=1))
        n_dense = int(est_dist * self.behavior.dense_samples_per_meter) + self.behavior.dense_samples_base

        dense_pos = spline_pos.evaluate(n_dense)
        dense_look = spline_look.evaluate(n_dense) if spline_look is not None else None

        cp_lengths = np.linalg.norm(np.diff(cp_pos, axis=0), axis=1)
        cp_cum = np.insert(np.cumsum(cp_lengths), 0, 0.0)

        dense_step_lengths = np.linalg.norm(np.diff(dense_pos, axis=0), axis=1)
        dense_cum = np.insert(np.cumsum(dense_step_lengths), 0, 0.0)

        cp_total = float(cp_cum[-1])
        dense_total = float(dense_cum[-1])
        if cp_total <= 1e-9 or dense_total <= 1e-9:
            segment_indices = np.zeros(len(dense_pos), dtype=int)
        else:
            mapped_cp = (dense_cum / dense_total) * cp_total
            segment_indices = np.searchsorted(cp_cum, mapped_cp, side="right") - 1
        segment_indices = np.clip(segment_indices, 0, len(seg_speed_arr) - 1)
        dense_speed = seg_speed_arr[segment_indices]

        dists = np.linalg.norm(np.diff(dense_pos, axis=0), axis=1)
        avg_speeds = (dense_speed[:-1] + dense_speed[1:]) / 2.0
        avg_speeds = np.clip(avg_speeds, self.behavior.min_speed_clamp, self.behavior.max_linear_speed)
        segment_times = dists / avg_speeds
        if not np.all(np.isfinite(segment_times)):
            raise ValueError("Non-finite segment_times encountered during interpolation.")
        cum_time = np.insert(np.cumsum(segment_times), 0, 0.0)
        total_time = cum_time[-1]

        num_final_frames = max(2, int(total_time * fps))
        target_times = np.linspace(0, total_time, num_final_frames)

        smooth_pos = np.zeros((num_final_frames, 3))
        smooth_look = None
        if dense_look is not None:
            smooth_look = np.zeros((num_final_frames, 3))

        for k in range(3):
            smooth_pos[:, k] = np.interp(target_times, cum_time, dense_pos[:, k])
            if smooth_look is not None and dense_look is not None:
                smooth_look[:, k] = np.interp(target_times, cum_time, dense_look[:, k])

        avg_speed = est_dist / total_time if total_time > 1e-9 else 0.0
        logger.debug(
            "Generated %s frames. Duration: %.2fs. Avg Speed: %.2fm/s",
            num_final_frames,
            total_time,
            avg_speed,
        )

        return smooth_pos, smooth_look, total_time

    @staticmethod
    def _frame_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construct a stable right/up pair for a given normalized forward vector."""
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        fwd = np.asarray(forward, dtype=float)
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm <= 1e-9:
            fwd = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            fwd = fwd / fwd_norm

        up_ref = world_up
        if abs(float(np.dot(fwd, up_ref))) > 0.98:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=float)

        right = np.cross(fwd, up_ref)
        right_norm = np.linalg.norm(right)
        if right_norm <= 1e-9:
            up_ref = np.array([1.0, 0.0, 0.0], dtype=float)
            right = np.cross(fwd, up_ref)
            right_norm = np.linalg.norm(right)
        right = right / max(right_norm, 1e-9)
        true_up = np.cross(right, fwd)
        true_up = true_up / max(np.linalg.norm(true_up), 1e-9)
        return right, true_up

    def build_camera_frames(
        self,
        smooth_pos: np.ndarray,
        smooth_look: Optional[np.ndarray],
        fps: int,
    ) -> list[dict[str, Any]]:
        """Convert interpolated positions and look targets into camera frames."""
        tangents = np.gradient(smooth_pos, axis=0)
        norms = np.linalg.norm(tangents, axis=1)

        last_valid = np.array([1.0, 0.0, 0.0], dtype=float)
        for k in range(len(tangents)):
            if norms[k] > 1e-6:
                tangents[k] /= norms[k]
                last_valid = tangents[k].copy()
            else:
                tangents[k] = last_valid

        tangent_forwards = self.apply_heading_constraints(tangents, fps=fps)

        target_distances = None
        if smooth_look is not None:
            raw_forward = smooth_look - smooth_pos
            forward_norm = np.linalg.norm(raw_forward, axis=1)
            target_distances = np.maximum(forward_norm, 1e-3)
            look_forwards = np.zeros_like(raw_forward)
            for idx in range(len(raw_forward)):
                if forward_norm[idx] > 1e-6:
                    look_forwards[idx] = raw_forward[idx] / forward_norm[idx]
                else:
                    look_forwards[idx] = tangent_forwards[idx]
            constrained_forwards = self.apply_heading_constraints(look_forwards, fps=fps)
        else:
            constrained_forwards = tangent_forwards

        fov = self.behavior.fov
        frames: list[dict[str, Any]] = []
        for i in range(len(smooth_pos)):
            pos = smooth_pos[i]

            if smooth_look is not None:
                if target_distances is None:
                    raise RuntimeError("target_distances missing while smooth_look is set.")
                forward = constrained_forwards[i]
                target = pos + forward * float(target_distances[i])
            else:
                forward = constrained_forwards[i]
                target = pos + forward

            _right, true_up = self._frame_basis(forward)

            frames.append(
                {
                    "id": i,
                    "position": pos.tolist(),
                    "look_at": target.tolist(),
                    "forward": forward.tolist(),
                    "up": true_up.tolist(),
                    "fov": fov,
                }
            )

        return frames

    def build_static_frames(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        num_frames: int,
    ) -> list[dict[str, Any]]:
        target = np.array(look_at, dtype=float)
        pos = np.array(position, dtype=float)
        forward = target - pos
        norm = np.linalg.norm(forward)
        if norm <= 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=float)
            target = pos + forward
        else:
            forward = forward / norm

        _right, true_up = self._frame_basis(forward)

        fov = self.behavior.fov
        frames: list[dict[str, Any]] = []
        for i in range(max(1, num_frames)):
            frames.append(
                {
                    "id": i,
                    "position": pos.tolist(),
                    "look_at": target.tolist(),
                    "forward": forward.tolist(),
                    "up": true_up.tolist(),
                    "fov": fov,
                }
            )
        return frames
