"""Shared trajectory output artifacts and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


def _path_for_summary(path: Path, *, base: Path | None = None) -> str:
    candidate = Path(path)
    if base is not None:
        try:
            return str(candidate.relative_to(base))
        except ValueError:
            pass
    try:
        return str(candidate.relative_to(Path.cwd()))
    except ValueError:
        return str(candidate)


@dataclass
class FloorTrajectoryArtifact:
    """Trajectory artifact metadata for one floor."""

    floor_index: int
    floor_z: float
    num_frames: int
    num_rooms: int
    num_connections: int
    output_file: Path


@dataclass
class TrajectoryGenerationArtifacts:
    """Aggregate output metadata for a trajectory generation run."""

    scene: str
    dataset_root: Path
    output_dir: Path
    floor_trajectories: list[FloorTrajectoryArtifact] = field(default_factory=list)
    connectivity_statistics: dict[int, dict[str, Any]] = field(default_factory=dict)
    component_transfers_by_floor: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_root"] = _path_for_summary(self.dataset_root)
        payload["output_dir"] = _path_for_summary(self.output_dir)
        payload["floor_trajectories"] = [
            {
                **asdict(item),
                "output_file": _path_for_summary(item.output_file, base=self.output_dir),
            }
            for item in self.floor_trajectories
        ]
        payload["connectivity_statistics"] = {
            str(k): v for k, v in self.connectivity_statistics.items()
        }
        payload["component_transfers_by_floor"] = {
            str(k): v for k, v in self.component_transfers_by_floor.items()
        }
        return payload


def write_trajectory_frames(
    frames: list[dict[str, Any]],
    output_path_stem: Path,
) -> Path:
    """Serialize trajectory frames into JSON."""
    output_path = output_path_stem.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(frames, indent=2))
    return output_path
