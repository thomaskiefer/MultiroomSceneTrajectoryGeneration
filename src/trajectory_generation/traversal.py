"""Traversal planning utilities for room-graph trajectory generation."""

from __future__ import annotations

import logging
from math import hypot
from typing import Callable, Iterable


logger = logging.getLogger(__name__)


def plan_room_sequence(
    start_room_id: str,
    all_room_ids: Iterable[str],
    adjacency: dict[str, list[str]],
    room_center_xy: Callable[[str], tuple[float, float] | list[float]],
) -> list[str]:
    """
    Build a DFS-based room visitation sequence that covers all components.

    The returned sequence includes backtracking visits to keep paths connected
    within each component.
    """
    room_ids = set(all_room_ids)
    if not start_room_id or start_room_id not in room_ids:
        logger.warning(
            "Invalid start_room_id=%r for available rooms=%s; returning empty traversal.",
            start_room_id,
            sorted(room_ids),
        )
        return []

    path_sequence: list[str] = []
    visited: set[str] = set()

    def dfs_iterative(start_id: str) -> str:
        visited.add(start_id)
        path_sequence.append(start_id)
        last_forward_room = start_id
        stack: list[tuple[str, int, list[str]]] = [
            (
                start_id,
                0,
                sorted(n for n in adjacency.get(start_id, []) if n in room_ids),
            )
        ]

        while stack:
            curr_id, next_idx, neighbors = stack[-1]

            while next_idx < len(neighbors) and neighbors[next_idx] in visited:
                next_idx += 1

            if next_idx < len(neighbors):
                neighbor_id = neighbors[next_idx]
                stack[-1] = (curr_id, next_idx + 1, neighbors)
                visited.add(neighbor_id)
                path_sequence.append(neighbor_id)
                last_forward_room = neighbor_id
                stack.append(
                    (
                        neighbor_id,
                        0,
                        sorted(n for n in adjacency.get(neighbor_id, []) if n in room_ids),
                    )
                )
                continue

            stack.pop()
            if stack:
                path_sequence.append(stack[-1][0])
        return last_forward_room

    curr_start_node = start_room_id

    while len(visited) < len(room_ids):
        last_component_room = dfs_iterative(curr_start_node)

        unvisited = room_ids - visited
        if not unvisited:
            break

        last_pos = room_center_xy(last_component_room)
        curr_start_node = min(
            unvisited,
            key=lambda rid: (_distance(room_center_xy(rid), last_pos), rid),
        )

    return path_sequence


def _distance(a: tuple[float, float] | list[float], b: tuple[float, float] | list[float]) -> float:
    return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
