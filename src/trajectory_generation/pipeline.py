"""Public pipeline entrypoints for trajectory generation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .adapters.houselayout3d_matterport import run_houselayout3d_matterport
from .adapters.structural_json import run_structural_json
from .artifacts import TrajectoryGenerationArtifacts
from .config import TrajectoryGenerationConfig


Runner = Callable[[TrajectoryGenerationConfig, Optional[Path]], TrajectoryGenerationArtifacts]

RUNNERS: dict[str, Runner] = {
    "houselayout3d_matterport": run_houselayout3d_matterport,
    "structural_json": run_structural_json,
}


def register_runner(workflow: str, runner: Runner) -> None:
    """Register or override a workflow runner at runtime."""
    RUNNERS[workflow] = runner


def run(
    config: TrajectoryGenerationConfig,
    project_root: Optional[Path] = None,
) -> TrajectoryGenerationArtifacts:
    """Run trajectory generation for the configured workflow."""
    runner = RUNNERS.get(config.workflow)
    if runner is None:
        allowed = ", ".join(sorted(RUNNERS))
        raise ValueError(f"Unknown workflow: {config.workflow!r}. Allowed values: {allowed}.")
    return runner(config, project_root)


def run_trajectory_generation(
    config: TrajectoryGenerationConfig,
    project_root: Optional[Path] = None,
) -> TrajectoryGenerationArtifacts:
    """Backward-compatible alias for older imports."""
    return run(config=config, project_root=project_root)
