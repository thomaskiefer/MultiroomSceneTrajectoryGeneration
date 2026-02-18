"""Public programmatic API for trajectory generation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .artifacts import TrajectoryGenerationArtifacts
from .config import TrajectoryGenerationConfig, WalkthroughBehaviorConfig
from .pipeline import run
from .room_graph import RoomGraph
from .walkthrough_local import LocalWalkthroughGenerator


def generate_from_config(
    config: TrajectoryGenerationConfig,
    project_root: Optional[Path] = None,
) -> TrajectoryGenerationArtifacts:
    """Run trajectory generation from an explicit config object."""
    return run(config=config, project_root=project_root)


def generate_from_structural_json(
    scene_input_json: Path,
    output_dir: Path = Path("outputs/trajectory_generation"),
    dataset_root: Optional[Path] = None,
    scene: str = "",
    project_root: Optional[Path] = None,
) -> TrajectoryGenerationArtifacts:
    """Run the structural_json workflow with minimal setup."""
    cfg = TrajectoryGenerationConfig.structural_json(
        scene_input_json=scene_input_json,
        output_dir=output_dir,
        dataset_root=dataset_root,
        scene=scene,
    )
    return run(config=cfg, project_root=project_root)


def generate_frames_from_graph(
    graph: RoomGraph,
    floor_z: float,
    fps: int = 30,
    camera_height: float = 1.6,
    behavior: Optional[WalkthroughBehaviorConfig] = None,
) -> list[dict]:
    """Generate trajectory frames from a prepared room graph."""
    walker = LocalWalkthroughGenerator(
        graph=graph,
        floor_z=floor_z,
        camera_height=camera_height,
        behavior=behavior,
    )
    return walker.generate_exploration_path(fps=fps)

