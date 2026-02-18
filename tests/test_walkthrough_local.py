from __future__ import annotations

from pathlib import Path
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
    from trajectory_generation.walkthrough_local import (
        CatmullRomSpline,
        LocalWalkthroughGenerator,
        _ControlPointSequence,
    )
    from trajectory_generation.config import WalkthroughBehaviorConfig
    from trajectory_generation.room_graph import RectPolygon
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    from shapely.geometry import Polygon
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


class _Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _Poly:
    def __init__(self, area: float, x: float = 0.0, y: float = 0.0):
        self.area = area
        self._pt = _Point(x, y)

    def representative_point(self):
        return self._pt


class _BrokenPoly:
    def __init__(self, area: float):
        self.area = area

    def representative_point(self):
        raise RuntimeError("representative point unavailable")


class _Room:
    def __init__(self, label_semantic: str, centroid):
        self.label_semantic = label_semantic
        self.centroid = centroid


class _Graph:
    def __init__(self, rooms, adjacency=None, connections=None):
        self.rooms = rooms
        self.adjacency = adjacency or {}
        self.connections = connections or []


class _Boundary:
    def __init__(self, coords):
        self.coords = coords


class _Waypoint:
    def __init__(self, position, normal=None, shared_boundary=None):
        self.position = position
        self.normal = normal
        self.shared_boundary = shared_boundary


class _Connection:
    def __init__(self, room1_id: str, room2_id: str, waypoint):
        self.room1_id = room1_id
        self.room2_id = room2_id
        self.waypoint = waypoint


@unittest.skipUnless(_HAS_NUMPY, "Local walkthrough tests require numpy")
class LocalWalkthroughLogicTest(unittest.TestCase):
    def test_find_start_room_prioritizes_semantic_and_area(self) -> None:
        rooms = {
            "r1": (_Room("living_room", np.array([0.0, 0.0, 0.0])), _Poly(20.0)),
            "r2": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(5.0)),
            "r3": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(8.0)),
            "r4": (_Room("kitchen", np.array([0.0, 0.0, 0.0])), _Poly(30.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        self.assertEqual(walker.start_room_id, "r3")

    def test_find_start_room_falls_back_to_largest(self) -> None:
        rooms = {
            "a": (_Room("bathroom", np.array([0.0, 0.0, 0.0])), _Poly(6.0)),
            "b": (_Room("office", np.array([0.0, 0.0, 0.0])), _Poly(10.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        self.assertEqual(walker.start_room_id, "b")

    def test_spline_collinear_points_are_monotonic(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        spline = CatmullRomSpline(points)
        sampled = spline.evaluate(30)
        x = sampled[:, 0]
        self.assertGreaterEqual(x[0], -1e-6)
        self.assertLessEqual(x[-1], 2.0 + 1e-6)
        self.assertTrue(np.all(np.diff(x) >= -1e-6))
        self.assertTrue(np.allclose(sampled[:, 1], 0.0, atol=1e-5))

    def test_find_start_room_returns_none_when_graph_is_empty(self) -> None:
        walker = LocalWalkthroughGenerator(_Graph({}), floor_z=0.0)
        self.assertIsNone(walker.start_room_id)
        self.assertEqual(walker.generate_exploration_path(fps=30), [])

    def test_get_door_normal_accepts_3d_normal(self) -> None:
        rooms = {
            "a": (_Room("living_room", np.array([0.0, 0.0, 0.0])), _Poly(12.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        normal = walker._get_door_normal(_Waypoint(position=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0, 0.0])))
        self.assertEqual(normal.shape, (2,))
        self.assertTrue(np.allclose(normal, np.array([0.0, 1.0]), atol=1e-6))

    def test_get_door_normal_from_shared_boundary(self) -> None:
        rooms = {
            "a": (_Room("living_room", np.array([0.0, 0.0, 0.0])), _Poly(12.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        boundary = _Boundary([(1.0, -0.5), (1.0, 0.0), (1.0, 0.5)])
        normal = walker._get_door_normal(_Waypoint(position=np.array([1.0, 0.0]), shared_boundary=boundary))
        self.assertEqual(normal.shape, (2,))
        self.assertAlmostEqual(abs(float(normal[0])), 1.0, places=5)
        self.assertLess(abs(float(normal[1])), 1e-5)

    def test_get_door_normal_from_two_point_shared_boundary(self) -> None:
        rooms = {
            "a": (_Room("living_room", np.array([0.0, 0.0, 0.0])), _Poly(12.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        boundary = _Boundary([(1.0, -0.5), (1.0, 0.5)])
        normal = walker._get_door_normal(_Waypoint(position=np.array([1.0, 0.0]), shared_boundary=boundary))
        self.assertEqual(normal.shape, (2,))
        self.assertAlmostEqual(abs(float(normal[0])), 1.0, places=5)
        self.assertLess(abs(float(normal[1])), 1e-5)

    def test_get_door_normal_degenerate_shared_boundary_returns_zero(self) -> None:
        rooms = {
            "a": (_Room("living_room", np.array([0.0, 0.0, 0.0])), _Poly(12.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        boundary = _Boundary([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)])
        normal = walker._get_door_normal(_Waypoint(position=np.array([1.0, 1.0]), shared_boundary=boundary))
        self.assertTrue(np.allclose(normal, np.array([0.0, 0.0]), atol=1e-9))

    @unittest.skipUnless(_HAS_SHAPELY, "shapely required")
    def test_get_room_center_prefers_centroid_when_polylabel_gain_is_small(self) -> None:
        # Notched rectangle where polylabel shifts but clearance gain is essentially zero.
        poly = Polygon(
            [
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 1.8),
                (9.0, 1.8),
                (9.0, 2.2),
                (10.0, 2.2),
                (10.0, 4.0),
                (0.0, 4.0),
                (0.0, 0.0),
            ]
        )
        rooms = {
            "a": (_Room("living_room", np.array([poly.centroid.x, poly.centroid.y, 0.0])), poly),
        }
        behavior = WalkthroughBehaviorConfig(
            polylabel_tolerance=0.01,
            polylabel_min_gain=0.05,
        )
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0, behavior=behavior)
        center = walker._get_room_center("a")

        self.assertAlmostEqual(center[0], poly.centroid.x, places=4)
        self.assertAlmostEqual(center[1], poly.centroid.y, places=4)

    @unittest.skipUnless(_HAS_SHAPELY, "shapely required")
    def test_get_room_center_prefers_polylabel_when_centroid_is_outside(self) -> None:
        # L-shape where centroid falls into the missing corner.
        poly = Polygon(
            [
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 1.0),
                (1.0, 1.0),
                (1.0, 4.0),
                (0.0, 4.0),
                (0.0, 0.0),
            ]
        )
        rooms = {
            "a": (_Room("living_room", np.array([poly.centroid.x, poly.centroid.y, 0.0])), poly),
        }
        behavior = WalkthroughBehaviorConfig(polylabel_tolerance=0.01, polylabel_min_gain=0.0)
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0, behavior=behavior)
        center = walker._get_room_center("a")

        # Ensure we do not keep the (outside) centroid position.
        self.assertGreater(float(np.linalg.norm(center[:2] - np.array([poly.centroid.x, poly.centroid.y]))), 0.25)

    @unittest.skipUnless(_HAS_SHAPELY, "shapely required")
    def test_get_room_center_narrow_room_can_prefer_centroid_with_high_gain_threshold(self) -> None:
        poly = Polygon(
            [
                (0.0, 0.0),
                (8.0, 0.0),
                (8.0, 0.4),
                (4.2, 0.4),
                (4.2, 0.6),
                (8.0, 0.6),
                (8.0, 1.0),
                (0.0, 1.0),
                (0.0, 0.0),
            ]
        )
        rooms = {
            "a": (_Room("living_room", np.array([poly.centroid.x, poly.centroid.y, 0.0])), poly),
        }
        behavior = WalkthroughBehaviorConfig(polylabel_tolerance=0.01, polylabel_min_gain=0.2)
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0, behavior=behavior)
        center = walker._get_room_center("a")

        self.assertAlmostEqual(center[0], poly.centroid.x, places=4)
        self.assertAlmostEqual(center[1], poly.centroid.y, places=4)

    def test_get_room_center_falls_back_to_room_centroid_when_polygon_methods_fail(self) -> None:
        room_centroid = np.array([2.5, -1.0, 0.0])
        rooms = {
            "a": (_Room("living_room", room_centroid), _BrokenPoly(area=5.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        center = walker._get_room_center("a")
        self.assertAlmostEqual(float(center[0]), 2.5, places=6)
        self.assertAlmostEqual(float(center[1]), -1.0, places=6)

    def test_get_room_center_rectpolygon_fallback(self) -> None:
        room_centroid = np.array([50.0, 50.0, 0.0])
        rooms = {
            "a": (_Room("living_room", room_centroid), RectPolygon((1.0, 2.0), (3.0, 6.0))),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        center = walker._get_room_center("a")
        self.assertAlmostEqual(float(center[0]), 2.0, places=6)
        self.assertAlmostEqual(float(center[1]), 4.0, places=6)

    def test_generate_exploration_path_handles_static_path(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
        }
        behavior = WalkthroughBehaviorConfig(
            spin_points=8,
            spin_orbit_scale=0.0,
            look_at_mode="tangent",
        )
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0, behavior=behavior)
        frames = walker.generate_exploration_path(fps=30)
        self.assertGreaterEqual(len(frames), 2)
        for frame in frames:
            self.assertEqual(frame["position"], frames[0]["position"])
            self.assertEqual(frame["look_at"], frames[0]["look_at"])

    def test_generate_exploration_path_uses_spline_target_mode(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([2.0, 0.0, 0.0])), _Poly(14.0, x=2.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a"]}
        boundary = _Boundary([(1.0, -0.5), (1.0, 0.5), (1.0, -0.5)])
        conn = _Connection(
            "a",
            "b",
            _Waypoint(position=np.array([1.0, 0.0]), normal=np.array([1.0, 0.0]), shared_boundary=boundary),
        )
        behavior = WalkthroughBehaviorConfig(look_at_mode="spline_target")
        walker = LocalWalkthroughGenerator(_Graph(rooms, adjacency=adjacency, connections=[conn]), floor_z=0.0, behavior=behavior)
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 2)

        # In spline_target mode, look_at is not constrained to pos + forward unit vector.
        found_non_tangent_target = False
        for frame in frames:
            pos = np.array(frame["position"], dtype=float)
            look_at = np.array(frame["look_at"], dtype=float)
            forward = np.array(frame["forward"], dtype=float)
            if not np.allclose(look_at, pos + forward, atol=1e-5):
                found_non_tangent_target = True
                break
        self.assertTrue(found_non_tangent_target)

    def test_generate_exploration_path_rejects_invalid_look_mode(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([2.0, 0.0, 0.0])), _Poly(14.0, x=2.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a"]}
        conn = _Connection(
            "a",
            "b",
            _Waypoint(position=np.array([1.0, 0.0]), normal=np.array([1.0, 0.0])),
        )
        with self.assertRaises(ValueError):
            behavior = WalkthroughBehaviorConfig(look_at_mode="invalid_mode")
            LocalWalkthroughGenerator(
                _Graph(rooms, adjacency=adjacency, connections=[conn]),
                floor_z=0.0,
                behavior=behavior,
            )

    def test_disconnected_bridge_mode_restarts_components_when_all_components_enabled(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([50.0, 0.0, 0.0])), _Poly(14.0, x=50.0, y=0.0)),
        }
        adjacency = {"a": [], "b": []}
        behavior = WalkthroughBehaviorConfig(
            disconnected_transition_mode="bridge",
            disconnected_component_policy="all_components",
        )
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[]),
            floor_z=0.0,
            behavior=behavior,
        )
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 2)
        self.assertEqual(walker.last_component_transfers, [])
        self.assertEqual(walker.last_disconnected_component_count, 2)
        self.assertEqual(
            set(frames[0].keys()),
            {"id", "position", "look_at", "forward", "up", "fov"},
        )

    def test_disconnected_jump_mode_has_no_transfer_metadata_when_all_components_enabled(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([50.0, 0.0, 0.0])), _Poly(14.0, x=50.0, y=0.0)),
        }
        adjacency = {"a": [], "b": []}
        behavior = WalkthroughBehaviorConfig(
            disconnected_transition_mode="jump",
            disconnected_component_policy="all_components",
        )
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[]),
            floor_z=0.0,
            behavior=behavior,
        )
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 2)
        self.assertEqual(walker.last_component_transfers, [])

    def test_disconnected_components_default_to_largest_component_only(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([3.0, 0.0, 0.0])), _Poly(14.0, x=3.0, y=0.0)),
            "c": (_Room("bedroom", np.array([50.0, 0.0, 0.0])), _Poly(10.0, x=50.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a"], "c": []}
        conn = _Connection(
            "a",
            "b",
            _Waypoint(position=np.array([1.5, 0.0]), normal=np.array([1.0, 0.0])),
        )
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(),
        )
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 2)
        self.assertEqual(walker.last_component_transfers, [])
        self.assertEqual(walker.last_skipped_disconnected_rooms, ["c"])

    def test_angular_speed_is_capped(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
        }
        fps = 30
        behavior = WalkthroughBehaviorConfig(
            max_angular_speed_deg=20.0,
            angular_smoothing_window_s=0.0,
            spin_points=12,
            spin_orbit_scale=0.15,
        )
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0, behavior=behavior)
        frames = walker.generate_exploration_path(fps=fps)
        self.assertGreater(len(frames), 10)

        forwards = np.array([f["forward"] for f in frames], dtype=float)
        headings = np.unwrap(np.arctan2(forwards[:, 1], forwards[:, 0]))
        yaw_rate_deg = np.abs(np.diff(headings)) * fps * (180.0 / np.pi)
        self.assertLessEqual(float(np.max(yaw_rate_deg)), 20.5)

    def test_linear_speed_is_capped(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([6.0, 0.0, 0.0])), _Poly(14.0, x=6.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a"]}
        conn = _Connection(
            "a",
            "b",
            _Waypoint(position=np.array([3.0, 0.0]), normal=np.array([1.0, 0.0])),
        )
        fps = 30
        behavior = WalkthroughBehaviorConfig(
            max_linear_speed=0.35,
            travel_speed=1.0,
            spin_segment_speed=0.5,
        )
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn]),
            floor_z=0.0,
            behavior=behavior,
        )
        frames = walker.generate_exploration_path(fps=fps)
        self.assertGreater(len(frames), 10)

        positions = np.array([f["position"] for f in frames], dtype=float)
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps
        self.assertLessEqual(float(np.max(speeds)), 0.38)


    def test_spin_start_aligned_with_incoming_direction(self) -> None:
        """Spin[0] should be on the entry side of the orbit, not the exit side."""
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([6.0, 0.0, 0.0])), _Poly(14.0, x=6.0, y=0.0)),
            "c": (_Room("kitchen", np.array([6.0, 6.0, 0.0])), _Poly(12.0, x=6.0, y=6.0)),
        }
        adjacency = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([3.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([6.0, 3.0]), normal=np.array([0.0, 1.0])))
        behavior = WalkthroughBehaviorConfig(spin_points=12, spin_orbit_scale=0.15)
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_bc]),
            floor_z=0.0, behavior=behavior,
        )
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 20)

    def test_passthrough_arc_produces_multiple_frames(self) -> None:
        """Revisiting a room should produce an arc with multiple control points, not a single cut."""
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
            "c": (_Room("kitchen", np.array([0.0, 4.0, 0.0])), _Poly(12.0, x=0.0, y=4.0)),
        }
        # DFS: a -> b -> (backtrack to a) -> c
        adjacency = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_ac = _Connection("a", "c", _Waypoint(position=np.array([0.0, 2.0]), normal=np.array([0.0, 1.0])))
        behavior = WalkthroughBehaviorConfig(passthrough_speed=0.3)
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_ac]),
            floor_z=0.0, behavior=behavior,
        )
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 20)

    def test_passthrough_arc_single_room_no_crash(self) -> None:
        """A single room with no connections should still produce valid frames."""
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
        }
        behavior = WalkthroughBehaviorConfig(passthrough_speed=0.3)
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0, behavior=behavior)
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreaterEqual(len(frames), 2)

    def test_passthrough_speed_config_default(self) -> None:
        """The passthrough_speed config field should default to match travel_speed."""
        behavior = WalkthroughBehaviorConfig()
        self.assertAlmostEqual(behavior.passthrough_speed, behavior.travel_speed)

    def test_deduplicate_overrides_long_slow_segment_speed(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        seq = _ControlPointSequence(
            positions=np.array([[0.0, 0.0, 1.6], [1.0, 0.0, 1.6]], dtype=float),
            look_targets=np.array([[1.0, 0.0, 1.6], [2.0, 0.0, 1.6]], dtype=float),
            segment_speeds=np.array([0.1], dtype=float),
        )
        out = walker._deduplicate_control_points(seq)
        self.assertEqual(len(out.segment_speeds), 1)
        self.assertAlmostEqual(float(out.segment_speeds[0]), walker.behavior.travel_speed, places=6)

    def test_interpolate_path_handles_single_control_point(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(_Graph(rooms), floor_z=0.0)
        seq = _ControlPointSequence(
            positions=np.array([[0.0, 0.0, 1.6]], dtype=float),
            look_targets=np.array([[1.0, 0.0, 1.6]], dtype=float),
            segment_speeds=np.array([], dtype=float),
        )
        smooth_pos, smooth_look, total_time = walker._interpolate_path(seq, fps=30)
        self.assertEqual(smooth_pos.shape, (1, 3))
        self.assertEqual(total_time, 0.0)
        self.assertIsNone(smooth_look)

    def test_component_transfer_indices_remap_after_deduplicate(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([0.0, 0.0, 0.0])), _Poly(12.0, x=0.0, y=0.0)),
        }
        adjacency = {"a": [], "b": []}
        behavior = WalkthroughBehaviorConfig(
            disconnected_transition_mode="bridge",
            spin_orbit_scale=0.0,
            spin_points=8,
        )
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[]),
            floor_z=0.0,
            behavior=behavior,
        )
        seq = walker._generate_control_points(["a", "b"], "bridge")
        dedup_seq = walker._deduplicate_control_points(seq)
        self.assertGreaterEqual(len(walker.last_component_transfers), 1)
        transfer = walker.last_component_transfers[0]
        start_idx, end_idx = transfer["control_point_indices"]
        self.assertGreaterEqual(start_idx, 0)
        self.assertGreaterEqual(end_idx, start_idx)
        self.assertLess(end_idx, len(dedup_seq.positions))


if __name__ == "__main__":
    unittest.main()
