"""Spline primitives reused by walkthrough generation."""

from __future__ import annotations

import numpy as np


class CatmullRomSpline:
    """
    Chordal Catmull-Rom spline.
    Guarantees passing through control points with no self-intersections.
    """

    def __init__(self, points: np.ndarray):
        """
        Args:
            points: Array of shape (N, 3)
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError(f"`points` must be a 2D array of shape (N, 3); got ndim={pts.ndim}.")
        if pts.shape[1] != 3:
            raise ValueError(f"`points` must have shape (N, 3); got shape={pts.shape}.")
        if not np.all(np.isfinite(pts)):
            raise ValueError("`points` contains non-finite values (NaN/Inf).")
        self.points = pts

        # Chordal parameterization: t increments by Euclidean distance.
        self.t = np.zeros(len(pts))
        for i in range(1, len(pts)):
            dist = float(np.linalg.norm(pts[i] - pts[i - 1]))
            if dist < 1e-6:
                dist = 1e-6
            self.t[i] = self.t[i - 1] + dist

    def evaluate(self, num_samples: int) -> np.ndarray:
        """Sample the spline at uniform intervals."""
        if len(self.points) < 2:
            return np.array(self.points, dtype=float, copy=True)
        if num_samples <= 0:
            return np.empty((0, self.points.shape[1]), dtype=float)

        t_new = np.linspace(self.t[0], self.t[-1], num_samples)
        # Map each sample to its segment index. This guarantees output length
        # equals num_samples even when adjacent knots are extremely close.
        seg_idx = np.searchsorted(self.t, t_new, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(self.points) - 2)

        result = np.zeros((num_samples, self.points.shape[1]), dtype=float)
        for i in np.unique(seg_idx):
            idx1 = int(i)
            idx2 = idx1 + 1
            mask = seg_idx == i
            t_vals = t_new[mask]

            p1 = self.points[idx1]
            p2 = self.points[idx2]
            t1 = self.t[idx1]
            t2 = self.t[idx2]

            # Use reflected phantom endpoints to improve boundary smoothness.
            if idx1 == 0:
                p0 = 2 * p1 - p2
                t0 = t1 - (t2 - t1)
            else:
                p0 = self.points[idx1 - 1]
                t0 = self.t[idx1 - 1]

            if idx1 == len(self.points) - 2:
                p3 = 2 * p2 - p1
                t3 = t2 + (t2 - t1)
            else:
                p3 = self.points[idx2 + 1]
                t3 = self.t[idx2 + 1]

            result[mask] = self._interpolate_segment_batch(
                p0, p1, p2, p3, t0, t1, t2, t3, t_vals
            )

        return result

    def _interpolate_segment_batch(self, p0, p1, p2, p3, t0, t1, t2, t3, t_values):
        t_values = np.asarray(t_values, dtype=float).reshape(-1, 1)
        p0 = np.asarray(p0, dtype=float).reshape(1, -1)
        p1 = np.asarray(p1, dtype=float).reshape(1, -1)
        p2 = np.asarray(p2, dtype=float).reshape(1, -1)
        p3 = np.asarray(p3, dtype=float).reshape(1, -1)

        if abs(t1 - t0) < 1e-9:
            a1 = np.repeat(p0, t_values.shape[0], axis=0)
        else:
            a1 = (t1 - t_values) / (t1 - t0) * p0 + (t_values - t0) / (t1 - t0) * p1

        if abs(t2 - t1) < 1e-9:
            a2 = np.repeat(p1, t_values.shape[0], axis=0)
        else:
            a2 = (t2 - t_values) / (t2 - t1) * p1 + (t_values - t1) / (t2 - t1) * p2

        if abs(t3 - t2) < 1e-9:
            a3 = np.repeat(p2, t_values.shape[0], axis=0)
        else:
            a3 = (t3 - t_values) / (t3 - t2) * p2 + (t_values - t2) / (t3 - t2) * p3

        if abs(t2 - t0) < 1e-9:
            b1 = a1
        else:
            b1 = (t2 - t_values) / (t2 - t0) * a1 + (t_values - t0) / (t2 - t0) * a2

        if abs(t3 - t1) < 1e-9:
            b2 = a2
        else:
            b2 = (t3 - t_values) / (t3 - t1) * a2 + (t_values - t1) / (t3 - t1) * a3

        if abs(t2 - t1) < 1e-9:
            c = b1
        else:
            c = (t2 - t_values) / (t2 - t1) * b1 + (t_values - t1) / (t2 - t1) * b2
        return c
