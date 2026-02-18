"""Standalone multi-room scene trajectory generation package."""

from .api import (
    generate_frames_from_graph,
    generate_from_config,
    generate_from_structural_json,
)
from .artifacts import FloorTrajectoryArtifact, TrajectoryGenerationArtifacts
from .config import (
    ConnectivityConfig,
    DatasetConfig,
    FloorplanPipelineConfig,
    TrajectoryGenerationConfig,
    TrajectoryVisualizationConfig,
    WalkthroughBehaviorConfig,
    WalkthroughConfig,
)
from .visualization import (
    plot_trajectory,
    plot_trajectory_from_artifacts,
    render_trajectory_image,
    render_trajectory_video,
)
from .pipeline import register_runner, run, run_trajectory_generation
from .preprocess import (
    Hl3dDebugArtifacts,
    build_hl3d_matterport_connectivity_geojson,
    convert_connectivity_geojson_file,
    convert_structural_primitives_file,
    convert_structural_primitives_payload,
    export_hl3d_matterport_debug_artifacts,
    preprocess_hl3d_matterport_to_structural_json,
)
from .room_graph import RoomConnection, RoomGraph, RoomGraphRoomNode, RoomGraphWaypoint
from .traversal import plan_room_sequence

__all__ = [
    "ConnectivityConfig",
    "DatasetConfig",
    "FloorplanPipelineConfig",
    "FloorTrajectoryArtifact",
    "generate_frames_from_graph",
    "generate_from_config",
    "generate_from_structural_json",
    "plan_room_sequence",
    "RoomConnection",
    "RoomGraph",
    "RoomGraphRoomNode",
    "RoomGraphWaypoint",
    "TrajectoryGenerationArtifacts",
    "TrajectoryGenerationConfig",
    "WalkthroughBehaviorConfig",
    "TrajectoryVisualizationConfig",
    "WalkthroughConfig",
    "plot_trajectory",
    "plot_trajectory_from_artifacts",
    "render_trajectory_image",
    "render_trajectory_video",
    "build_hl3d_matterport_connectivity_geojson",
    "convert_connectivity_geojson_file",
    "convert_structural_primitives_file",
    "convert_structural_primitives_payload",
    "export_hl3d_matterport_debug_artifacts",
    "Hl3dDebugArtifacts",
    "preprocess_hl3d_matterport_to_structural_json",
    "register_runner",
    "run",
    "run_trajectory_generation",
]
