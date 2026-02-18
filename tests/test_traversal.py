from __future__ import annotations

from pathlib import Path
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.traversal import plan_room_sequence


class TraversalTest(unittest.TestCase):
    def test_connected_graph_dfs_with_backtracking(self) -> None:
        adjacency = {
            "A": ["B"],
            "B": ["A", "C"],
            "C": ["B"],
        }
        centers = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0)}

        seq = plan_room_sequence(
            start_room_id="A",
            all_room_ids=adjacency.keys(),
            adjacency=adjacency,
            room_center_xy=lambda rid: centers[rid],
        )

        self.assertEqual(seq, ["A", "B", "C", "B", "A"])

    def test_disconnected_graph_component_jump_to_nearest(self) -> None:
        adjacency = {
            "A": ["B"],
            "B": ["A"],
            "C": ["D"],
            "D": ["C"],
        }
        centers = {
            "A": (0.0, 0.0),
            "B": (1.0, 0.0),
            "C": (10.0, 0.0),
            "D": (11.0, 0.0),
        }

        seq = plan_room_sequence(
            start_room_id="A",
            all_room_ids=adjacency.keys(),
            adjacency=adjacency,
            room_center_xy=lambda rid: centers[rid],
        )

        # Component 1 traversal with backtrack
        self.assertEqual(seq[:3], ["A", "B", "A"])
        # Then component 2 traversal starts at nearest unvisited ("C"), then "D", then backtrack.
        self.assertEqual(seq[3:], ["C", "D", "C"])

    def test_invalid_start_returns_empty(self) -> None:
        seq = plan_room_sequence(
            start_room_id="Z",
            all_room_ids=["A", "B"],
            adjacency={"A": ["B"], "B": ["A"]},
            room_center_xy=lambda rid: (0.0, 0.0),
        )
        self.assertEqual(seq, [])

    def test_invalid_start_logs_warning(self) -> None:
        with self.assertLogs("trajectory_generation.traversal", level="WARNING") as cap:
            seq = plan_room_sequence(
                start_room_id="Z",
                all_room_ids=["A", "B"],
                adjacency={"A": ["B"], "B": ["A"]},
                room_center_xy=lambda rid: (0.0, 0.0),
            )
        self.assertEqual(seq, [])
        self.assertIn("invalid start_room_id", " ".join(cap.output).lower())

    def test_large_linear_graph_is_supported_without_recursion(self) -> None:
        n = 5000
        ids = [f"R{i}" for i in range(n)]
        adjacency = {}
        centers = {}
        for i, room_id in enumerate(ids):
            neighbors = []
            if i > 0:
                neighbors.append(ids[i - 1])
            if i < n - 1:
                neighbors.append(ids[i + 1])
            adjacency[room_id] = neighbors
            centers[room_id] = (float(i), 0.0)

        seq = plan_room_sequence(
            start_room_id=ids[0],
            all_room_ids=ids,
            adjacency=adjacency,
            room_center_xy=lambda rid: centers[rid],
        )

        self.assertEqual(seq[0], ids[0])
        self.assertEqual(seq[n - 1], ids[-1])
        self.assertEqual(len(seq), (2 * n) - 1)

    def test_disconnected_equidistant_tie_break_is_deterministic(self) -> None:
        adjacency = {
            "A": [],
            "B": [],
            "C": [],
        }
        centers = {
            "A": (0.0, 0.0),
            "B": (1.0, 1.0),
            "C": (1.0, -1.0),
        }
        seq = plan_room_sequence(
            start_room_id="A",
            all_room_ids=adjacency.keys(),
            adjacency=adjacency,
            room_center_xy=lambda rid: centers[rid],
        )
        self.assertEqual(seq, ["A", "B", "C"])

    def test_component_jump_uses_last_leaf_not_backtracked_root(self) -> None:
        adjacency = {
            "A": ["B"],
            "B": ["A", "C"],
            "C": ["B"],
            "D": [],
            "E": [],
        }
        centers = {
            "A": (0.0, 0.0),
            "B": (10.0, 0.0),
            "C": (20.0, 0.0),
            "D": (21.0, 0.0),  # nearest to last leaf C
            "E": (-1.0, 0.0),  # nearest to backtracked root A
        }
        seq = plan_room_sequence(
            start_room_id="A",
            all_room_ids=adjacency.keys(),
            adjacency=adjacency,
            room_center_xy=lambda rid: centers[rid],
        )
        self.assertEqual(seq[:5], ["A", "B", "C", "B", "A"])
        self.assertEqual(seq[5], "D")

    def test_phantom_neighbors_are_ignored(self) -> None:
        seq = plan_room_sequence(
            start_room_id="A",
            all_room_ids=["A", "B"],
            adjacency={"A": ["B", "X"], "B": ["A"]},
            room_center_xy=lambda rid: {"A": (0.0, 0.0), "B": (1.0, 0.0)}[rid],
        )
        self.assertEqual(seq, ["A", "B", "A"])


if __name__ == "__main__":
    unittest.main()
