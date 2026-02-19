from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import matplotlib
    matplotlib.use("Agg")
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

from trajectory_generation.config import TrajectoryVisualizationConfig


_MOCK_FRAMES = [
    {"id": i, "position": [float(i), 0.0, 1.6], "look_at": [float(i) + 1, 0.0, 1.6]}
    for i in range(20)
]


@unittest.skipUnless(_HAS_MATPLOTLIB, "matplotlib required")
class PlotTrajectoryTest(unittest.TestCase):
    def test_smoke_renders_without_error(self) -> None:
        from trajectory_generation.visualization import plot_trajectory

        ax = plot_trajectory(frames=_MOCK_FRAMES, show_look_at=True)
        self.assertIsNotNone(ax)

    def test_saves_png_to_output_path(self) -> None:
        from trajectory_generation.visualization import plot_trajectory

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_traj.png"
            plot_trajectory(frames=_MOCK_FRAMES, output_path=out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_custom_viz_config(self) -> None:
        from trajectory_generation.visualization import plot_trajectory

        cfg = TrajectoryVisualizationConfig(
            path_color="#FF0000",
            max_arrows=5,
        )
        ax = plot_trajectory(frames=_MOCK_FRAMES, viz_config=cfg, show_look_at=True)
        self.assertIsNotNone(ax)

    def test_empty_frames_returns_none(self) -> None:
        from trajectory_generation.visualization import plot_trajectory

        ax = plot_trajectory(frames=[])
        self.assertIsNone(ax)

    @unittest.skipUnless(_HAS_SHAPELY, "shapely required for polygon tests")
    def test_with_floor_and_room_polygons(self) -> None:
        from trajectory_generation.visualization import plot_trajectory

        floor_poly = ShapelyPolygon([(-1, -1), (25, -1), (25, 5), (-1, 5)])
        room_polys = {
            "room_a": ShapelyPolygon([(0, 0), (10, 0), (10, 4), (0, 4)]),
            "room_b": ShapelyPolygon([(10, 0), (20, 0), (20, 4), (10, 4)]),
        }
        ax = plot_trajectory(
            frames=_MOCK_FRAMES,
            floor_polygon=floor_poly,
            room_polygons=room_polys,
            title="Test with polygons",
        )
        self.assertIsNotNone(ax)

    def test_draws_on_existing_axes(self) -> None:
        import matplotlib.pyplot as plt
        from trajectory_generation.visualization import plot_trajectory

        fig, ax = plt.subplots()
        returned_ax = plot_trajectory(frames=_MOCK_FRAMES, ax=ax)
        self.assertIs(returned_ax, ax)
        plt.close(fig)

    def test_show_look_at_skips_frames_missing_look_at(self) -> None:
        from trajectory_generation.visualization import plot_trajectory

        frames = [
            {"id": 0, "position": [0.0, 0.0, 1.6], "look_at": [1.0, 0.0, 1.6]},
            {"id": 1, "position": [1.0, 0.0, 1.6]},
        ]
        ax = plot_trajectory(frames=frames, show_look_at=True)
        self.assertIsNotNone(ax)


@unittest.skipUnless(_HAS_MATPLOTLIB, "matplotlib required")
class PlotTrajectoryFromArtifactsTest(unittest.TestCase):
    def test_plots_from_saved_artifacts(self) -> None:
        import json
        from trajectory_generation.artifacts import (
            FloorTrajectoryArtifact,
            TrajectoryGenerationArtifacts,
        )
        from trajectory_generation.visualization import plot_trajectory_from_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frames_path = tmpdir / "scene_floor_0_trajectory.json"
            frames_path.write_text(json.dumps(_MOCK_FRAMES))

            artifacts = TrajectoryGenerationArtifacts(
                scene="test",
                dataset_root=tmpdir,
                output_dir=tmpdir,
                floor_trajectories=[
                    FloorTrajectoryArtifact(
                        floor_index=0,
                        floor_z=0.0,
                        num_frames=len(_MOCK_FRAMES),
                        num_rooms=2,
                        num_connections=1,
                        output_file=frames_path,
                    )
                ],
            )

            paths = plot_trajectory_from_artifacts(artifacts)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())


@unittest.skipUnless(_HAS_MATPLOTLIB, "matplotlib required")
class RichVisualizationBackendTest(unittest.TestCase):
    def _write_minimal_inputs(self, root: Path, frames: list[dict]) -> tuple[Path, Path]:
        import json

        geojson_path = root / "layout.geojson"
        geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
        trajectory_path = root / "traj.json"
        trajectory_path.write_text(json.dumps(frames))
        return geojson_path, trajectory_path

    def test_render_trajectory_image_with_single_frame(self) -> None:
        from trajectory_generation.visualization import render_trajectory_image

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frames = [
                {
                    "id": 0,
                    "position": [0.0, 0.0, 1.6],
                    "look_at": [1.0, 0.0, 1.6],
                    "forward": [1.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                    "fov": 60.0,
                }
            ]
            geojson_path, trajectory_path = self._write_minimal_inputs(tmp, frames)
            out = tmp / "single_frame.png"
            render_trajectory_image(geojson_path, trajectory_path, out, scene_name="single")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_parse_features_handles_null_properties(self) -> None:
        from trajectory_generation.visualization_backend import _parse_features

        payload = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": None, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}},
            ],
        }
        parsed = _parse_features(payload)
        self.assertIn("rooms", parsed)
        self.assertEqual(sum(len(v) for v in parsed.values()), 0)

    @unittest.skipUnless(_HAS_CV2, "opencv required")
    def test_render_trajectory_video_without_forward_key(self) -> None:
        from trajectory_generation.visualization import render_trajectory_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frames = [
                {
                    "id": 0,
                    "position": [0.0, 0.0, 1.6],
                    "look_at": [1.0, 0.0, 1.6],
                    "up": [0.0, 0.0, 1.0],
                    "fov": 60.0,
                },
                {
                    "id": 1,
                    "position": [0.5, 0.0, 1.6],
                    "look_at": [1.5, 0.0, 1.6],
                    "up": [0.0, 0.0, 1.0],
                    "fov": 60.0,
                },
            ]
            geojson_path, trajectory_path = self._write_minimal_inputs(tmp, frames)
            out = tmp / "video.mp4"
            render_trajectory_video(geojson_path, trajectory_path, out, scene_name="video", speed=1.0, fps=5)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    @unittest.skipUnless(_HAS_CV2, "opencv required")
    def test_render_trajectory_video_rejects_non_positive_speed(self) -> None:
        from trajectory_generation.visualization import render_trajectory_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frames = [
                {
                    "id": 0,
                    "position": [0.0, 0.0, 1.6],
                    "look_at": [1.0, 0.0, 1.6],
                    "forward": [1.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                    "fov": 60.0,
                }
            ]
            geojson_path, trajectory_path = self._write_minimal_inputs(tmp, frames)
            with self.assertRaises(ValueError):
                render_trajectory_video(geojson_path, trajectory_path, tmp / "bad.mp4", speed=0.0)

    @unittest.skipUnless(_HAS_CV2, "opencv required")
    def test_render_trajectory_video_rejects_non_positive_fps(self) -> None:
        from trajectory_generation.visualization import render_trajectory_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frames = [
                {
                    "id": 0,
                    "position": [0.0, 0.0, 1.6],
                    "look_at": [1.0, 0.0, 1.6],
                    "forward": [1.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                    "fov": 60.0,
                }
            ]
            geojson_path, trajectory_path = self._write_minimal_inputs(tmp, frames)
            with self.assertRaises(ValueError):
                render_trajectory_video(geojson_path, trajectory_path, tmp / "bad_fps.mp4", fps=0)

    @unittest.skipUnless(_HAS_CV2, "opencv required")
    def test_render_trajectory_video_cleans_temp_file_on_ffmpeg_failure(self) -> None:
        from trajectory_generation.visualization import render_trajectory_video

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frames = [
                {
                    "id": 0,
                    "position": [0.0, 0.0, 1.6],
                    "look_at": [1.0, 0.0, 1.6],
                    "forward": [1.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                    "fov": 60.0,
                }
            ]
            geojson_path, trajectory_path = self._write_minimal_inputs(tmp, frames)
            out = tmp / "video_fail.mp4"
            temp_h264 = out.with_suffix(".tmp.mp4")

            class _Proc:
                returncode = 1

            def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
                Path(cmd[-1]).write_bytes(b"partial")
                return _Proc()

            with patch("subprocess.run", side_effect=_fake_run):
                render_trajectory_video(geojson_path, trajectory_path, out, scene_name="video", speed=1.0, fps=5)

            self.assertFalse(temp_h264.exists())


if __name__ == "__main__":
    unittest.main()
