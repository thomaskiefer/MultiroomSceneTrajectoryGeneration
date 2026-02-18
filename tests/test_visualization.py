from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
