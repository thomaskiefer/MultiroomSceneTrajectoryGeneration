"""Traversal planning utilities for room-graph trajectory generation."""

from __future__ import annotations

import logging
from math import hypot
from typing import Callable, Iterable, Optional

import numpy as np


logger = logging.getLogger(__name__)


def plan_room_sequence(
    start_room_id: str,
    all_room_ids: Iterable[str],
    adjacency: dict[str, list[str]],
    room_center_xy: Callable[[str], tuple[float, float] | list[float]],
    neighbor_priority_mode: str = "lexicographic",
    room_semantic: Optional[Callable[[str], str]] = None,
    semantic_priority: Optional[list[str]] = None,
    connection_kind: Optional[Callable[[str, str], str]] = None,
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

    if neighbor_priority_mode not in {"lexicographic", "human_like"}:
        raise ValueError(
            f"Unsupported neighbor_priority_mode: {neighbor_priority_mode}. "
            "Expected one of: lexicographic, human_like."
        )

    path_sequence: list[str] = []
    visited: set[str] = set()
    center_cache: dict[str, tuple[float, float]] = {}

    priority_list = [s.lower() for s in (semantic_priority or [])]
    semantic_rank = {name: idx for idx, name in enumerate(priority_list)}

    def _center_xy(room_id: str) -> tuple[float, float]:
        cached = center_cache.get(room_id)
        if cached is not None:
            return cached
        value = room_center_xy(room_id)
        xy = (float(value[0]), float(value[1]))
        center_cache[room_id] = xy
        return xy

    def _connection_rank(curr_id: str, neighbor_id: str) -> int:
        if connection_kind is None:
            return 1
        kind = str(connection_kind(curr_id, neighbor_id)).lower()
        if kind in {"observed", "door", "actual"}:
            return 0
        if kind in {"inferred", "synthetic"}:
            return 1
        return 2

    def _semantic_rank(room_id: str) -> int:
        if room_semantic is None:
            return len(semantic_rank)
        semantic = str(room_semantic(room_id)).strip().lower()
        return semantic_rank.get(semantic, len(semantic_rank))

    def _turn_angle(curr_id: str, prev_id: Optional[str], neighbor_id: str) -> float:
        if prev_id is None:
            return 0.0
        prev_xy = np.asarray(_center_xy(prev_id), dtype=float)
        curr_xy = np.asarray(_center_xy(curr_id), dtype=float)
        next_xy = np.asarray(_center_xy(neighbor_id), dtype=float)
        incoming = curr_xy - prev_xy
        outgoing = next_xy - curr_xy
        norm_in = float(np.linalg.norm(incoming))
        norm_out = float(np.linalg.norm(outgoing))
        if norm_in < 1e-9 or norm_out < 1e-9:
            return float(np.pi)
        incoming /= norm_in
        outgoing /= norm_out
        dot = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
        return float(np.arccos(dot))

    def _order_neighbors(curr_id: str, prev_id: Optional[str]) -> list[str]:
        neighbors = [n for n in adjacency.get(curr_id, []) if n in room_ids]
        if neighbor_priority_mode == "lexicographic":
            return sorted(neighbors)
        curr_xy = _center_xy(curr_id)
        return sorted(
            neighbors,
            key=lambda neighbor_id: (
                _connection_rank(curr_id, neighbor_id),
                _turn_angle(curr_id, prev_id, neighbor_id),
                _distance(_center_xy(neighbor_id), curr_xy),
                _semantic_rank(neighbor_id),
                neighbor_id,
            ),
        )

    def dfs_iterative(start_id: str) -> str:
        visited.add(start_id)
        path_sequence.append(start_id)
        last_forward_room = start_id
        stack: list[tuple[str, Optional[str], int, list[str]]] = [
            (
                start_id,
                None,
                0,
                _order_neighbors(start_id, None),
            )
        ]

        while stack:
            curr_id, prev_id, next_idx, neighbors = stack[-1]

            while next_idx < len(neighbors) and neighbors[next_idx] in visited:
                next_idx += 1

            if next_idx < len(neighbors):
                neighbor_id = neighbors[next_idx]
                stack[-1] = (curr_id, prev_id, next_idx + 1, neighbors)
                visited.add(neighbor_id)
                path_sequence.append(neighbor_id)
                last_forward_room = neighbor_id
                stack.append(
                    (
                        neighbor_id,
                        curr_id,
                        0,
                        _order_neighbors(neighbor_id, curr_id),
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

        last_pos = _center_xy(last_component_room)
        curr_start_node = min(
            unvisited,
            key=lambda rid: (_distance(_center_xy(rid), last_pos), rid),
        )

    return path_sequence


def _distance(a: tuple[float, float] | list[float], b: tuple[float, float] | list[float]) -> float:
    return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
