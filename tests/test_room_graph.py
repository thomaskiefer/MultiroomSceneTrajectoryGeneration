from __future__ import annotations

from pathlib import Path
import unittest
import sys

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.room_graph import (  # noqa: E402
    RectPolygon,
    RoomConnection,
    RoomGraph,
    RoomGraphRoomNode,
    RoomGraphWaypoint,
)


class _Poly:
    def __init__(self, area: float = 1.0):
        self.area = area

    def representative_point(self):
        class _Point:
            x = 0.0
            y = 0.0

        return _Point()


class RoomGraphTest(unittest.TestCase):
    def test_validate_accepts_symmetric_graph(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly()),
            "b": (RoomGraphRoomNode("b", "living_room", np.array([1.0, 0.0, 0.0])), _Poly()),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": ["b"], "b": ["a"]},
            connections=[
                RoomConnection(
                    room1_id="a",
                    room2_id="b",
                    waypoint=RoomGraphWaypoint(position=np.array([0.5, 0.0])),
                )
            ],
        )
        graph.validate()
        self.assertEqual(graph.neighbors("a"), ["b"])

    def test_validate_rejects_asymmetric_adjacency(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly()),
            "b": (RoomGraphRoomNode("b", "living_room", np.array([1.0, 0.0, 0.0])), _Poly()),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": ["b"], "b": []},
            connections=[],
        )
        with self.assertRaises(ValueError):
            graph.validate()

    def test_validate_rejects_connection_missing_from_adjacency(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly()),
            "b": (RoomGraphRoomNode("b", "living_room", np.array([1.0, 0.0, 0.0])), _Poly()),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": [], "b": []},
            connections=[
                RoomConnection(
                    room1_id="a",
                    room2_id="b",
                    waypoint=RoomGraphWaypoint(position=np.array([0.5, 0.0])),
                )
            ],
        )
        with self.assertRaises(ValueError):
            graph.validate()

    def test_validate_rejects_missing_adjacency_entry(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly()),
            "b": (RoomGraphRoomNode("b", "living_room", np.array([1.0, 0.0, 0.0])), _Poly()),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": ["b"]},
            connections=[],
        )
        with self.assertRaises(ValueError):
            graph.validate()

    def test_validate_rejects_adjacency_edge_missing_connection_metadata(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly()),
            "b": (RoomGraphRoomNode("b", "living_room", np.array([1.0, 0.0, 0.0])), _Poly()),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": ["b"], "b": ["a"]},
            connections=[],
        )
        with self.assertRaises(ValueError):
            graph.validate()

    def test_connected_components(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly()),
            "b": (RoomGraphRoomNode("b", "living_room", np.array([1.0, 0.0, 0.0])), _Poly()),
            "c": (RoomGraphRoomNode("c", "kitchen", np.array([5.0, 0.0, 0.0])), _Poly()),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": ["b"], "b": ["a"], "c": []},
            connections=[],
        )
        comps = graph.connected_components()
        comp_sizes = sorted(len(c) for c in comps)
        self.assertEqual(comp_sizes, [1, 2])

    def test_to_dict_contains_required_sections(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly()),
            "b": (RoomGraphRoomNode("b", "living_room", np.array([1.0, 0.0, 0.0])), _Poly()),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": ["b"], "b": ["a"]},
            connections=[
                RoomConnection(
                    room1_id="a",
                    room2_id="b",
                    waypoint=RoomGraphWaypoint(position=np.array([0.5, 0.0])),
                )
            ],
        )
        payload = graph.to_dict()
        self.assertIn("rooms", payload)
        self.assertIn("adjacency", payload)
        self.assertIn("connections", payload)

    def test_neighbors_unknown_room_raises_key_error(self) -> None:
        rooms = {
            "a": (RoomGraphRoomNode("a", "entryway", np.array([0.0, 0.0, 0.0])), _Poly()),
        }
        graph = RoomGraph(
            rooms=rooms,
            adjacency={"a": []},
            connections=[],
        )
        with self.assertRaises(KeyError):
            graph.neighbors("missing")

    def test_rectpolygon_rejects_non_finite_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RectPolygon((0.0, 0.0), (float("nan"), 1.0))
        with self.assertRaises(ValueError):
            RectPolygon((0.0, 0.0), (1.0, float("inf")))


if __name__ == "__main__":
    unittest.main()
