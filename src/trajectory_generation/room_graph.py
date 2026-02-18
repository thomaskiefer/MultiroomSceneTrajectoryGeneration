"""Explicit room-graph contract shared by adapters and trajectory logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np


@dataclass(eq=False)
class RoomGraphRoomNode:
    """Semantic room node with a 3D centroid."""

    room_id: str
    label_semantic: str
    centroid: np.ndarray


@dataclass(eq=False)
class RoomGraphWaypoint:
    """Connection waypoint data used for doorway/transition handling."""

    position: np.ndarray  # [x, y]
    normal: Optional[np.ndarray] = None
    shared_boundary: Any = None


@dataclass(eq=False)
class RoomConnection:
    """Undirected connection between two rooms."""

    room1_id: str
    room2_id: str
    waypoint: RoomGraphWaypoint


class XYPoint(Protocol):
    """Minimal 2D point protocol used by room polygon representatives."""

    x: float
    y: float


class RoomPolygon(Protocol):
    """Minimal polygon protocol required by trajectory generation."""

    area: float

    def representative_point(self) -> XYPoint:
        ...


@dataclass
class Point2D:
    x: float
    y: float


class RectPolygon:
    """Axis-aligned rectangle polygon facade."""

    def __init__(self, min_xy: tuple[float, float], max_xy: tuple[float, float]):
        self.min_x = float(min_xy[0])
        self.min_y = float(min_xy[1])
        self.max_x = float(max_xy[0])
        self.max_y = float(max_xy[1])
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise ValueError(
                "Invalid RectPolygon bounds: max_xy must be >= min_xy for both x and y."
            )
        self.area = max(0.0, (self.max_x - self.min_x) * (self.max_y - self.min_y))

    def representative_point(self) -> XYPoint:
        return Point2D(
            x=(self.min_x + self.max_x) / 2.0,
            y=(self.min_y + self.max_y) / 2.0,
        )


@dataclass
class RoomGraph:
    """Shared room graph contract consumed by LocalWalkthroughGenerator."""

    rooms: dict[str, tuple[RoomGraphRoomNode, RoomPolygon]]
    adjacency: dict[str, list[str]]
    connections: list[RoomConnection]

    def neighbors(self, room_id: str) -> list[str]:
        if room_id not in self.rooms:
            raise KeyError(f"Unknown room_id: {room_id}")
        return list(self.adjacency.get(room_id, []))

    def connected_components(self) -> list[set[str]]:
        unseen = set(self.rooms.keys())
        components: list[set[str]] = []
        while unseen:
            root = next(iter(unseen))
            stack = [root]
            comp: set[str] = set()
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                for nxt in self.adjacency.get(node, []):
                    if nxt not in comp:
                        stack.append(nxt)
            components.append(comp)
            unseen -= comp
        return components

    def validate(self) -> None:
        room_ids = set(self.rooms.keys())
        for room_id in room_ids:
            if room_id not in self.adjacency:
                raise ValueError(f"Missing adjacency entry for room id: {room_id}")
        for room_id, neighbors in self.adjacency.items():
            if room_id not in room_ids:
                raise ValueError(f"Adjacency references unknown room id: {room_id}")
            for neighbor in neighbors:
                if neighbor not in room_ids:
                    raise ValueError(
                        f"Adjacency for room {room_id} references unknown neighbor: {neighbor}"
                    )

        for conn in self.connections:
            if conn.room1_id not in room_ids or conn.room2_id not in room_ids:
                raise ValueError(
                    f"Connection references unknown rooms: {conn.room1_id}, {conn.room2_id}"
                )
            if conn.room1_id == conn.room2_id:
                raise ValueError(f"Self-connection not allowed for room: {conn.room1_id}")

        for room_id, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                if room_id not in self.adjacency.get(neighbor, []):
                    raise ValueError(
                        f"Asymmetric adjacency detected between {room_id} and {neighbor}"
                    )

        for conn in self.connections:
            if conn.room2_id not in self.adjacency.get(conn.room1_id, []):
                raise ValueError(
                    f"Connection ({conn.room1_id}, {conn.room2_id}) missing from adjacency."
                )
            if conn.room1_id not in self.adjacency.get(conn.room2_id, []):
                raise ValueError(
                    f"Connection ({conn.room1_id}, {conn.room2_id}) missing from adjacency."
                )

        connection_pairs = {
            frozenset((conn.room1_id, conn.room2_id))
            for conn in self.connections
        }
        for room_id, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                pair = frozenset((room_id, neighbor))
                if pair not in connection_pairs:
                    raise ValueError(
                        f"Adjacency edge ({room_id}, {neighbor}) missing connection metadata."
                    )

    def to_dict(self) -> dict[str, Any]:
        rooms_payload: dict[str, Any] = {}
        for room_id, (room_node, _) in self.rooms.items():
            rooms_payload[room_id] = {
                "room_id": room_node.room_id,
                "label_semantic": room_node.label_semantic,
                "centroid": np.asarray(room_node.centroid, dtype=float).tolist(),
            }

        return {
            "rooms": rooms_payload,
            "adjacency": {k: list(v) for k, v in self.adjacency.items()},
            "connections": [
                {
                    "room1_id": c.room1_id,
                    "room2_id": c.room2_id,
                    "waypoint": {
                        "position": np.asarray(c.waypoint.position, dtype=float).tolist(),
                        "normal": (
                            None
                            if c.waypoint.normal is None
                            else np.asarray(c.waypoint.normal, dtype=float).tolist()
                        ),
                    },
                }
                for c in self.connections
            ],
        }
