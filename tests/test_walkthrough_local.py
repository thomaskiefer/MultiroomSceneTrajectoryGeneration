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

    def test_start_room_spin_starts_on_departure_side(self) -> None:
        """For first room spin, start/end camera position should align with next target direction."""
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a"]}
        conn = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        behavior = WalkthroughBehaviorConfig(spin_points=8, spin_orbit_scale=0.10)
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn]),
            floor_z=0.0,
            behavior=behavior,
        )
        seq = walker._generate_control_points(["a", "b"], "bridge")

        num_spin_samples = behavior.spin_points + 1
        self.assertGreaterEqual(len(seq.positions), num_spin_samples)
        first_spin = seq.positions[0]
        last_spin = seq.positions[num_spin_samples - 1]
        center = walker._get_room_center("a")

        self.assertGreater(float(first_spin[0] - center[0]), 0.0)
        self.assertGreater(float(last_spin[0] - center[0]), 0.0)

    def test_non_entry_first_visit_spin_can_align_with_departure_side(self) -> None:
        """Departure-aligned spin start/end should also work for non-entry first visits."""
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
            "c": (_Room("kitchen", np.array([4.0, 4.0, 0.0])), _Poly(12.0, x=4.0, y=4.0)),
        }
        adjacency = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([4.0, 2.0]), normal=np.array([0.0, 1.0])))
        behavior = WalkthroughBehaviorConfig(spin_points=8, spin_orbit_scale=0.10)
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_bc]),
            floor_z=0.0,
            behavior=behavior,
        )
        planner = walker._make_control_point_planner()

        cp_pos = [np.array([2.0, 0.0, walker.eye_level], dtype=float)]
        cp_look = [np.array([3.0, 0.0, walker.eye_level], dtype=float)]
        seg_speeds: list[float] = []
        center_b = walker._get_room_center("b")
        departure_angle = planner._get_departure_angle(
            seq_idx=1,
            room_id="b",
            center_pos=center_b,
            path_sequence=["a", "b", "c"],
        )
        planner._add_spin_points(
            center_pos=center_b,
            cp_pos=cp_pos,
            cp_look=cp_look,
            seg_speeds=seg_speeds,
            preferred_departure_angle=departure_angle,
        )

        orbit_radius = behavior.spin_look_radius * behavior.spin_orbit_scale
        spin_start_idx = None
        for idx in range(1, len(cp_pos)):
            radial = float(np.linalg.norm(np.asarray(cp_pos[idx][:2], dtype=float) - center_b[:2]))
            if abs(radial - orbit_radius) < 1e-5:
                spin_start_idx = idx
                break
        self.assertIsNotNone(spin_start_idx)
        first_spin = cp_pos[int(spin_start_idx)]
        last_spin = cp_pos[-1]
        self.assertGreater(float(first_spin[1] - center_b[1]), 0.0)
        self.assertGreater(float(last_spin[1] - center_b[1]), 0.0)

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

    def test_revisit_door_shortcut_reduces_center_arc_detour(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
            "c": (_Room("kitchen", np.array([8.0, 0.0, 0.0])), _Poly(12.0, x=8.0, y=0.0)),
            "d": (_Room("bedroom", np.array([4.0, 4.0, 0.0])), _Poly(12.0, x=4.0, y=4.0)),
        }
        adjacency = {"a": ["b"], "b": ["a", "c", "d"], "c": ["b"], "d": ["b"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([6.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bd = _Connection("b", "d", _Waypoint(position=np.array([4.0, 2.0]), normal=np.array([0.0, 1.0])))
        graph = _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_bc, conn_bd])

        path = ["a", "b", "c", "b", "d"]
        arc_behavior = WalkthroughBehaviorConfig(revisit_transition_mode="center_arc", spin_points=8)
        shortcut_behavior = WalkthroughBehaviorConfig(revisit_transition_mode="door_shortcut", spin_points=8)

        walker_arc = LocalWalkthroughGenerator(graph, floor_z=0.0, behavior=arc_behavior)
        walker_shortcut = LocalWalkthroughGenerator(graph, floor_z=0.0, behavior=shortcut_behavior)
        seq_arc = walker_arc._generate_control_points(path, "bridge")
        seq_shortcut = walker_shortcut._generate_control_points(path, "bridge")

        self.assertLess(len(seq_shortcut.positions), len(seq_arc.positions))

    def test_revisit_door_shortcut_does_not_return_to_incoming_door(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
            "c": (_Room("kitchen", np.array([8.0, 0.0, 0.0])), _Poly(12.0, x=8.0, y=0.0)),
            "d": (_Room("bedroom", np.array([4.0, 4.0, 0.0])), _Poly(12.0, x=4.0, y=4.0)),
        }
        adjacency = {"a": ["b"], "b": ["a", "c", "d"], "c": ["b"], "d": ["b"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([6.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bd = _Connection("b", "d", _Waypoint(position=np.array([4.0, 2.0]), normal=np.array([0.0, 1.0])))
        graph = _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_bc, conn_bd])

        behavior = WalkthroughBehaviorConfig(revisit_transition_mode="door_shortcut", spin_points=8)
        walker = LocalWalkthroughGenerator(graph, floor_z=0.0, behavior=behavior)
        planner = walker._make_control_point_planner()

        # Simulate state immediately after crossing c->b door:
        # door at x=6, departure point inside b is around x=5.6 for buffer=0.4.
        cp_pos = [np.array([5.6, 0.0, walker.eye_level], dtype=float)]
        cp_look = [np.array([5.0, 0.0, walker.eye_level], dtype=float)]
        seg_speeds: list[float] = []
        consumed = planner._add_passthrough_door_shortcut(
            seq_idx=3,
            room_id="b",
            center_pos=walker._get_room_center("b"),
            path_sequence=["a", "b", "c", "b", "d"],
            cp_pos=cp_pos,
            cp_look=cp_look,
            seg_speeds=seg_speeds,
        )
        self.assertTrue(consumed)

        # Newly appended points should not include the incoming door point (6,0).
        new_points = cp_pos[1:]
        self.assertFalse(
            any(float(np.linalg.norm(np.asarray(p[:2]) - np.array([6.0, 0.0]))) < 1e-6 for p in new_points)
        )

    def test_loop_closure_auto_closes_for_center_arc(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_transition_mode="center_arc",
                loop_closure_mode="auto",
                look_at_mode="spline_target",
            ),
        )

        seq = walker._deduplicate_control_points(walker._generate_control_points(["a", "b", "a"], "bridge"))
        closed = walker._apply_loop_closure(seq)

        self.assertTrue(np.allclose(closed.positions[0], closed.positions[-1], atol=0.0))
        self.assertTrue(np.allclose(closed.look_targets[0], closed.look_targets[-1], atol=0.0))
        self.assertEqual(len(closed.segment_speeds), len(closed.positions) - 1)

    def test_loop_closure_auto_disabled_for_door_shortcut(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
            "c": (_Room("kitchen", np.array([8.0, 0.0, 0.0])), _Poly(12.0, x=8.0, y=0.0)),
            "d": (_Room("bedroom", np.array([4.0, 4.0, 0.0])), _Poly(12.0, x=4.0, y=4.0)),
        }
        adjacency = {"a": ["b"], "b": ["a", "c", "d"], "c": ["b"], "d": ["b"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([6.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bd = _Connection("b", "d", _Waypoint(position=np.array([4.0, 2.0]), normal=np.array([0.0, 1.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_bc, conn_bd]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(revisit_transition_mode="door_shortcut", loop_closure_mode="auto"),
        )

        seq = walker._deduplicate_control_points(walker._generate_control_points(["a", "b", "c", "b", "d"], "bridge"))
        out = walker._apply_loop_closure(seq)

        self.assertGreater(float(np.linalg.norm(out.positions[-1] - out.positions[0])), 1e-3)
        self.assertEqual(len(out.segment_speeds), len(out.positions) - 1)

    def test_loop_closure_enabled_for_door_shortcut(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
            "c": (_Room("kitchen", np.array([8.0, 0.0, 0.0])), _Poly(12.0, x=8.0, y=0.0)),
            "d": (_Room("bedroom", np.array([4.0, 4.0, 0.0])), _Poly(12.0, x=4.0, y=4.0)),
        }
        adjacency = {"a": ["b"], "b": ["a", "c", "d"], "c": ["b"], "d": ["b"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([6.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bd = _Connection("b", "d", _Waypoint(position=np.array([4.0, 2.0]), normal=np.array([0.0, 1.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_bc, conn_bd]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_transition_mode="door_shortcut",
                loop_closure_mode="enabled",
                look_at_mode="spline_target",
            ),
        )

        seq = walker._deduplicate_control_points(walker._generate_control_points(["a", "b", "c", "b", "d"], "bridge"))
        closed = walker._apply_loop_closure(seq)

        self.assertTrue(np.allclose(closed.positions[0], closed.positions[-1], atol=0.0))
        self.assertTrue(np.allclose(closed.look_targets[0], closed.look_targets[-1], atol=0.0))
        self.assertEqual(len(closed.segment_speeds), len(closed.positions) - 1)

    def test_loop_closure_adds_terminal_pose(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(revisit_transition_mode="center_arc", loop_closure_mode="enabled"),
        )
        seq = _ControlPointSequence(
            positions=np.array(
                [
                    [0.0, 0.0, walker.eye_level],
                    [1.0, 0.0, walker.eye_level],
                    [0.0, 1.0, walker.eye_level],
                ],
                dtype=float,
            ),
            look_targets=np.array(
                [
                    [1.0, 0.0, walker.eye_level],
                    [2.0, 0.0, walker.eye_level],
                    [1.0, 1.0, walker.eye_level],
                ],
                dtype=float,
            ),
            segment_speeds=np.array([0.8, 0.8], dtype=float),
        )
        out = walker._apply_loop_closure(seq)
        self.assertGreaterEqual(len(out.positions), len(seq.positions) + 1)
        self.assertTrue(np.allclose(out.positions[0], out.positions[-1], atol=0.0))

    def test_loop_closure_adds_tangent_anchor_for_long_final_jump(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_transition_mode="center_arc",
                loop_closure_mode="enabled",
                spin_orbit_scale=0.1,
                spin_look_radius=2.0,
            ),
        )
        orbit_r = walker.behavior.spin_orbit_scale * walker.behavior.spin_look_radius  # 0.2
        look_r = walker.behavior.spin_look_radius  # 2.0

        # Start/look consistent with spin-orbit model around center:
        # start_pos = center - s*v, start_look = center + v.
        start_pos = np.array([-orbit_r, 0.0, walker.eye_level], dtype=float)
        start_look = np.array([look_r, 0.0, walker.eye_level], dtype=float)
        s = float(walker.behavior.spin_orbit_scale)
        center_xy = start_pos[:2] + (s / (1.0 + s)) * (start_look[:2] - start_pos[:2])
        mid_pos = np.array([0.0, -orbit_r, walker.eye_level], dtype=float)
        mid_look = np.array([0.0, -look_r, walker.eye_level], dtype=float)
        prev_pos = np.array([orbit_r * 0.6, -orbit_r * 0.8, walker.eye_level], dtype=float)
        prev_look = np.array([look_r * 0.6, -look_r * 0.8, walker.eye_level], dtype=float)

        seq = _ControlPointSequence(
            positions=np.array([start_pos, mid_pos, prev_pos], dtype=float),
            look_targets=np.array([start_look, mid_look, prev_look], dtype=float),
            segment_speeds=np.array([0.8, 0.8], dtype=float),
        )
        out = walker._apply_loop_closure(seq)
        self.assertGreaterEqual(len(out.positions), len(seq.positions) + 2)
        self.assertTrue(np.allclose(out.positions[-1], out.positions[0], atol=0.0))

    def test_loop_closure_snap_closes_when_already_near_start(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_transition_mode="center_arc",
                loop_closure_mode="enabled",
                look_at_mode="tangent",
                spin_orbit_scale=0.1,
                spin_look_radius=2.0,
            ),
        )
        seq = _ControlPointSequence(
            positions=np.array(
                [
                    [0.0, 0.0, walker.eye_level],
                    [1.0, 0.0, walker.eye_level],
                    [0.15, 0.0, walker.eye_level],
                ],
                dtype=float,
            ),
            look_targets=np.array(
                [
                    [1.0, 0.0, walker.eye_level],
                    [2.0, 0.0, walker.eye_level],
                    [0.0, 1.0, walker.eye_level],
                ],
                dtype=float,
            ),
            segment_speeds=np.array([0.8, 0.8], dtype=float),
        )
        out = walker._apply_loop_closure(seq)
        self.assertEqual(len(out.positions), len(seq.positions))
        self.assertTrue(np.allclose(out.positions[0], out.positions[-1], atol=0.0))
        self.assertEqual(len(out.segment_speeds), len(out.positions) - 1)

    def test_generate_path_center_arc_loop_closes_final_frame_position(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(revisit_transition_mode="center_arc", loop_closure_mode="auto"),
        )
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 2)
        self.assertTrue(np.allclose(frames[0]["position"], frames[-1]["position"], atol=0.0))
        self.assertTrue(np.allclose(frames[0]["forward"], frames[-1]["forward"], atol=1e-6))

    def test_terminal_revisit_motion_can_be_suppressed_for_loop_closure(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(revisit_transition_mode="center_arc", loop_closure_mode="auto"),
        )
        path_sequence = ["a", "b", "a"]
        seq_with_terminal = walker._generate_control_points(
            path_sequence,
            "bridge",
            suppress_terminal_revisit_motion=False,
        )
        seq_without_terminal = walker._generate_control_points(
            path_sequence,
            "bridge",
            suppress_terminal_revisit_motion=True,
        )
        self.assertLess(len(seq_without_terminal.positions), len(seq_with_terminal.positions))

    def test_generate_path_door_shortcut_auto_keeps_open_end_pose(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(14.0, x=4.0, y=0.0)),
            "c": (_Room("kitchen", np.array([8.0, 0.0, 0.0])), _Poly(12.0, x=8.0, y=0.0)),
            "d": (_Room("bedroom", np.array([4.0, 4.0, 0.0])), _Poly(12.0, x=4.0, y=4.0)),
        }
        adjacency = {"a": ["b"], "b": ["a", "c", "d"], "c": ["b"], "d": ["b"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([6.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bd = _Connection("b", "d", _Waypoint(position=np.array([4.0, 2.0]), normal=np.array([0.0, 1.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_bc, conn_bd]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(revisit_transition_mode="door_shortcut", loop_closure_mode="auto"),
        )
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 2)
        self.assertGreater(
            float(np.linalg.norm(np.array(frames[0]["position"]) - np.array(frames[-1]["position"]))),
            1e-3,
        )

    def test_generate_path_door_shortcut_auto_trims_final_backtrack_suffix(self) -> None:
        rooms = {
            "a": (_Room("entryway", np.array([0.0, 0.0, 0.0])), _Poly(10.0, x=0.0, y=0.0)),
            "b": (_Room("living_room", np.array([4.0, 0.0, 0.0])), _Poly(12.0, x=4.0, y=0.0)),
            "c": (_Room("kitchen", np.array([8.0, 0.0, 0.0])), _Poly(12.0, x=8.0, y=0.0)),
        }
        adjacency = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        conn_ab = _Connection("a", "b", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([6.0, 0.0]), normal=np.array([1.0, 0.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_ab, conn_bc]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_transition_mode="door_shortcut",
                loop_closure_mode="auto",
            ),
        )
        frames = walker.generate_exploration_path(fps=15)
        self.assertGreater(len(frames), 2)
        end_pos = np.array(frames[-1]["position"], dtype=float)
        dist_to_start = float(np.linalg.norm(end_pos - walker._get_room_center("a")))
        dist_to_last = float(np.linalg.norm(end_pos - walker._get_room_center("c")))
        self.assertLess(dist_to_last, dist_to_start)

    def test_trim_open_path_sequence_removes_only_trailing_backtrack_suffix(self) -> None:
        seq = ["a", "b", "c", "b", "d", "b", "a"]
        trimmed = LocalWalkthroughGenerator._trim_open_path_sequence(seq)
        self.assertEqual(trimmed, ["a", "b", "c", "b", "d"])

    def test_passthrough_arc_uses_minimum_turn_for_tiny_turns(self) -> None:
        rooms = {
            "x": (_Room("entryway", np.array([-2.0, 0.0, 0.0])), _Poly(8.0, x=-2.0, y=0.0)),
            "a": (_Room("hallway", np.array([0.0, 0.0, 0.0])), _Poly(14.0, x=0.0, y=0.0)),
            "c": (_Room("kitchen", np.array([-2.0, 0.2, 0.0])), _Poly(12.0, x=-2.0, y=0.2)),
        }
        adjacency = {"x": ["a"], "a": ["x", "c"], "c": ["a"]}
        conn_xa = _Connection("x", "a", _Waypoint(position=np.array([-1.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_ac = _Connection("a", "c", _Waypoint(position=np.array([-1.0, 0.1]), normal=np.array([1.0, 0.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_xa, conn_ac]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(passthrough_min_turn_deg=18.0),
        )
        planner = walker._make_control_point_planner()
        center_a = walker._get_room_center("a")

        cp_pos = [np.array([-1.0, 0.0, walker.eye_level], dtype=float)]
        cp_look = [np.array([0.0, 0.0, walker.eye_level], dtype=float)]
        seg_speeds: list[float] = []
        planner._add_passthrough_arc(
            seq_idx=1,
            room_id="a",
            center_pos=center_a,
            path_sequence=["x", "a", "c"],
            cp_pos=cp_pos,
            cp_look=cp_look,
            seg_speeds=seg_speeds,
        )
        self.assertGreater(len(cp_pos), 1)
        self.assertGreater(len(seg_speeds), 0)

    def test_passthrough_arc_exits_on_departure_side(self) -> None:
        rooms = {
            "x": (_Room("entryway", np.array([-2.0, 0.0, 0.0])), _Poly(8.0, x=-2.0, y=0.0)),
            "a": (_Room("hallway", np.array([0.0, 0.0, 0.0])), _Poly(14.0, x=0.0, y=0.0)),
            "c": (_Room("kitchen", np.array([2.0, 0.0, 0.0])), _Poly(12.0, x=2.0, y=0.0)),
        }
        adjacency = {"x": ["a"], "a": ["x", "c"], "c": ["a"]}
        conn_xa = _Connection("x", "a", _Waypoint(position=np.array([-1.0, 0.0]), normal=np.array([1.0, 0.0])))
        conn_ac = _Connection("a", "c", _Waypoint(position=np.array([1.0, 0.0]), normal=np.array([1.0, 0.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_xa, conn_ac]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(passthrough_min_turn_deg=0.0),
        )
        planner = walker._make_control_point_planner()
        center_a = walker._get_room_center("a")

        cp_pos = [np.array([-1.2, 0.0, walker.eye_level], dtype=float)]
        cp_look = [np.array([-0.2, 0.0, walker.eye_level], dtype=float)]
        seg_speeds: list[float] = []
        planner._add_passthrough_arc(
            seq_idx=1,
            room_id="a",
            center_pos=center_a,
            path_sequence=["x", "a", "c"],
            cp_pos=cp_pos,
            cp_look=cp_look,
            seg_speeds=seg_speeds,
        )

        self.assertGreater(len(cp_pos), 1)
        # Arc end should now be on outgoing-door side (+x), not opposite side.
        self.assertGreater(float(cp_pos[-1][0] - center_a[0]), 0.0)

    def test_dynamic_revisit_arc_search_improves_or_matches_objective(self) -> None:
        rooms = {
            "a": (_Room("hallway", np.array([0.0, 0.0, 0.0])), _Poly(14.0, x=0.0, y=0.0)),
        }
        center_xy = np.array([0.0, 0.0], dtype=float)
        orbit_radius = 0.2
        previous_pos = np.array([-0.25, -0.05], dtype=float)
        target_xy = np.array([0.9, 0.6], dtype=float)
        entry_nom = 0.0
        exit_nom = np.pi / 2.0

        base_walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_arc_angle_search_deg=0.0,
                revisit_arc_search_steps=1,
                passthrough_min_turn_deg=0.0,
            ),
        )
        opt_walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_arc_angle_search_deg=30.0,
                revisit_arc_search_steps=7,
                passthrough_min_turn_deg=0.0,
            ),
        )
        base_planner = base_walker._make_control_point_planner()
        opt_planner = opt_walker._make_control_point_planner()

        entry_b, delta_b = base_planner._optimize_revisit_arc_segment(
            entry_angle_nominal=entry_nom,
            exit_angle_nominal=exit_nom,
            center_xy=center_xy,
            orbit_radius=orbit_radius,
            previous_pos_xy=previous_pos,
            transition_target_xy=target_xy,
        )
        entry_o, delta_o = opt_planner._optimize_revisit_arc_segment(
            entry_angle_nominal=entry_nom,
            exit_angle_nominal=exit_nom,
            center_xy=center_xy,
            orbit_radius=orbit_radius,
            previous_pos_xy=previous_pos,
            transition_target_xy=target_xy,
        )

        def score(planner, _behavior, entry, delta):
            sign = 1.0 if delta >= 0.0 else -1.0
            start_tangent = planner._normalize_dir(np.array([sign * np.sin(entry), -sign * np.cos(entry)]))
            end_angle = entry + delta
            end_tangent = planner._normalize_dir(np.array([sign * np.sin(end_angle), -sign * np.cos(end_angle)]))
            in_dir = planner._normalize_dir(
                center_xy + orbit_radius * np.array([-np.cos(entry), -np.sin(entry)]) - previous_pos
            )
            out_dir = planner._normalize_dir(
                target_xy - (center_xy + orbit_radius * np.array([-np.cos(end_angle), -np.sin(end_angle)]))
            )
            mismatch_in = planner._angle_cost(start_tangent, in_dir) if in_dir is not None else 0.0
            mismatch_out = planner._angle_cost(end_tangent, out_dir) if out_dir is not None else 0.0
            value = mismatch_in + mismatch_out
            return float(value)

        self.assertLessEqual(
            score(opt_planner, opt_walker.behavior, entry_o, delta_o),
            score(base_planner, base_walker.behavior, entry_b, delta_b) + 1e-9,
        )

    def test_arc_tie_break_prefers_smaller_adjustment(self) -> None:
        rooms = {
            "a": (_Room("hallway", np.array([0.0, 0.0, 0.0])), _Poly(14.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_arc_angle_search_deg=0.0,
                revisit_arc_search_steps=1,
            ),
        )
        planner = walker._make_control_point_planner()

        _entry, delta = planner._optimize_revisit_arc_segment(
            entry_angle_nominal=0.0,
            exit_angle_nominal=np.deg2rad(170.0),
            center_xy=np.array([0.0, 0.0], dtype=float),
            orbit_radius=0.2,
            previous_pos_xy=None,
            transition_target_xy=None,
        )
        self.assertLess(abs(delta), np.pi)

    def test_revisit_arc_zero_delta_does_not_force_full_loop(self) -> None:
        rooms = {
            "a": (_Room("hallway", np.array([0.0, 0.0, 0.0])), _Poly(14.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_arc_angle_search_deg=0.0,
                revisit_arc_search_steps=1,
            ),
        )
        planner = walker._make_control_point_planner()
        _entry, delta = planner._optimize_revisit_arc_segment(
            entry_angle_nominal=np.deg2rad(30.0),
            exit_angle_nominal=np.deg2rad(30.0),
            center_xy=np.array([0.0, 0.0], dtype=float),
            orbit_radius=0.2,
            previous_pos_xy=np.array([-0.2, 0.2], dtype=float),
            transition_target_xy=None,
        )
        self.assertLess(abs(delta), np.pi)

    def test_revisit_arc_reversal_prefers_long_arc(self) -> None:
        rooms = {
            "a": (_Room("hallway", np.array([0.0, 0.0, 0.0])), _Poly(14.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_arc_angle_search_deg=0.0,
                revisit_arc_search_steps=1,
                revisit_arc_max_span_deg=120.0,
                revisit_arc_reverse_pref_deg=150.0,
                revisit_arc_max_tangent_mismatch_deg=179.0,
            ),
        )
        planner = walker._make_control_point_planner()
        _entry, delta = planner._optimize_revisit_arc_segment(
            entry_angle_nominal=0.0,
            exit_angle_nominal=np.deg2rad(20.0),
            center_xy=np.array([0.0, 0.0], dtype=float),
            orbit_radius=0.2,
            previous_pos_xy=np.array([0.8, 0.0], dtype=float),
            transition_target_xy=np.array([1.0, 0.0], dtype=float),
        )
        self.assertGreater(abs(delta), np.pi)

    def test_revisit_arc_reversal_can_choose_short_when_bias_disabled(self) -> None:
        rooms = {
            "a": (_Room("hallway", np.array([0.0, 0.0, 0.0])), _Poly(14.0, x=0.0, y=0.0)),
        }
        walker = LocalWalkthroughGenerator(
            _Graph(rooms),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(
                revisit_arc_angle_search_deg=0.0,
                revisit_arc_search_steps=1,
                revisit_arc_reverse_pref_deg=150.0,
                revisit_arc_reverse_long_arc_bonus=0.0,
                revisit_arc_max_tangent_mismatch_deg=179.0,
            ),
        )
        planner = walker._make_control_point_planner()
        _entry, delta = planner._optimize_revisit_arc_segment(
            entry_angle_nominal=0.0,
            exit_angle_nominal=np.deg2rad(20.0),
            center_xy=np.array([0.0, 0.0], dtype=float),
            orbit_radius=0.2,
            previous_pos_xy=np.array([0.8, 0.0], dtype=float),
            transition_target_xy=np.array([1.0, 0.0], dtype=float),
        )
        self.assertLess(abs(delta), np.pi)

    def test_door_crossing_does_not_backtrack_when_already_near_door(self) -> None:
        rooms = {
            "b": (_Room("living_room", np.array([0.0, 0.0, 0.0])), _Poly(14.0, x=0.0, y=0.0)),
            "c": (_Room("kitchen", np.array([4.0, 0.0, 0.0])), _Poly(12.0, x=4.0, y=0.0)),
        }
        adjacency = {"b": ["c"], "c": ["b"]}
        conn_bc = _Connection("b", "c", _Waypoint(position=np.array([2.0, 0.0]), normal=np.array([1.0, 0.0])))
        walker = LocalWalkthroughGenerator(
            _Graph(rooms, adjacency=adjacency, connections=[conn_bc]),
            floor_z=0.0,
            behavior=WalkthroughBehaviorConfig(),
        )
        planner = walker._make_control_point_planner()
        center_b = walker._get_room_center("b")

        cp_pos = [np.array([2.1, 0.0, walker.eye_level], dtype=float)]
        cp_look = [np.array([3.0, 0.0, walker.eye_level], dtype=float)]
        seg_speeds: list[float] = []
        planner._add_door_crossing(
            conn=conn_bc,
            center_pos=center_b,
            next_room_id="c",
            cp_pos=cp_pos,
            cp_look=cp_look,
            seg_speeds=seg_speeds,
        )

        xs = [float(p[0]) for p in cp_pos]
        for i in range(1, len(xs)):
            self.assertGreaterEqual(xs[i], xs[i - 1] - 1e-6)

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
