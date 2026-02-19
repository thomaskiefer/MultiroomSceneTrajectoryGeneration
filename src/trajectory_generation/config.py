"""Configuration models for trajectory generation workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional


SUPPORTED_WORKFLOWS = ("houselayout3d_matterport", "structural_json")
SUPPORTED_LOOK_AT_MODES = ("tangent", "spline_target")
SUPPORTED_DISCONNECTED_TRANSITION_MODES = ("bridge", "jump")
SUPPORTED_DISCONNECTED_COMPONENT_POLICIES = ("largest_component_only", "all_components")
SUPPORTED_NEIGHBOR_PRIORITY_MODES = ("lexicographic", "human_like")
SUPPORTED_REVISIT_TRANSITION_MODES = ("center_arc", "door_shortcut")
SUPPORTED_LOOP_CLOSURE_MODES = ("auto", "enabled", "disabled")


def _path_or_none(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


def _serialize_paths(payload: Any) -> Any:
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, list):
        return [_serialize_paths(v) for v in payload]
    if isinstance(payload, dict):
        return {k: _serialize_paths(v) for k, v in payload.items()}
    return payload


def _validate_choice(field_name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        allowed_str = ", ".join(allowed)
        raise ValueError(f"Invalid `{field_name}`: {value!r}. Allowed values: {allowed_str}.")


@dataclass
class DatasetConfig:
    """Input dataset paths and scene identity."""

    dataset_root: Path = Path(".")
    scene: str = ""
    house_segmentation_dir: Optional[Path] = None
    scene_input_json: Optional[Path] = None

    @property
    def resolved_house_segmentation_dir(self) -> Path:
        if self.house_segmentation_dir is not None:
            return self.house_segmentation_dir
        return self.dataset_root / "house_segmentations"


@dataclass
class FloorplanPipelineConfig:
    """Extraction and room/opening assignment options."""

    height_tolerance: float = 0.3
    room_height_tolerance: float = 0.5
    room_height_tolerance_above: float = 3.0
    min_floor_area: float = 10.0
    merge_buffer: float = 0.01
    keep_only_largest: bool = False
    opening_wall_tolerance: float = 0.4
    use_priority_assignment: bool = True
    include_openings: bool = True
    max_room_floor_distance: float = 4.0


@dataclass
class ConnectivityConfig:
    """Room connectivity graph construction options."""

    proximity_threshold: float = 0.3
    min_passage_area: float = 0.05
    door_match_tolerance: float = 0.3
    split_hallways_at_doors: bool = True
    hallway_split_length: float = 5.0


@dataclass
class WalkthroughBehaviorConfig:
    """Behavior knobs for trajectory generation logic."""

    spin_points: int = 12  # orbit samples for a full 360-degree room spin.
    spin_look_radius: float = 2.0  # [m] look-target radius around room center.
    spin_orbit_scale: float = 0.10  # [-] camera orbit radius scale relative to look radius.
    spin_segment_speed: float = 0.8  # [m/s] target speed along spin segments.
    travel_speed: float = 0.8  # [m/s] target speed for room-to-room travel.
    door_buffer: float = 0.4  # [m] approach/depart offset from doorway waypoint.
    lookahead_inside_next_room: float = 1.5  # [m] look-ahead depth after crossing door.
    max_linear_speed: float = 1.2  # [m/s] hard speed cap applied after spline resampling.
    max_angular_speed_deg: float = 360.0  # [deg/s] yaw-rate cap for heading changes.
    angular_smoothing_window_s: float = 0.3  # [s] moving-average window for heading smoothing.
    dense_samples_per_meter: int = 100  # [samples/m] dense spline sampling density.
    dense_samples_base: int = 1000  # [samples] minimum dense samples regardless of distance.
    long_segment_threshold: float = 0.2  # [m] long segment threshold for speed safety override.
    slow_speed_threshold: float = 0.2  # [m/s] threshold considered too slow for long segments.
    look_at_mode: str = "tangent"  # tangent | spline_target.
    disconnected_transition_mode: str = "bridge"  # bridge | jump.
    disconnected_component_policy: str = "largest_component_only"  # largest_component_only | all_components.
    neighbor_priority_mode: str = "human_like"  # lexicographic | human_like.
    revisit_transition_mode: str = "center_arc"  # center_arc | door_shortcut.
    loop_closure_mode: str = "auto"  # auto | enabled | disabled.
    prefer_outer_revisit_arc: bool = True  # prefer long arc on revisits for smoother room-perimeter coverage.
    revisit_arc_angle_search_deg: float = 30.0  # [deg] +/- search window around nominal entry/exit angles.
    revisit_arc_search_steps: int = 7  # number of offset samples per angle in search window.
    revisit_arc_length_weight: float = 0.0  # deprecated/no-op (kept for config compatibility).
    revisit_arc_outer_bias: float = 0.0  # deprecated/no-op (kept for config compatibility).
    revisit_arc_max_span_deg: float = 360.0  # deprecated/no-op (kept for config compatibility).
    revisit_arc_max_tangent_mismatch_deg: float = 85.0  # [deg] per-end tangent mismatch gate for arc candidates.
    revisit_arc_reverse_pref_deg: float = 155.0  # [deg] reversal threshold where long arcs are preferred.
    revisit_arc_reverse_long_arc_bonus: float = 0.35  # [rad] score bias favoring long arcs in reversal regime.
    revisit_arc_transition_risk_distance_weight: float = 0.35  # [-] weight on normalized exit-to-target distance.
    revisit_arc_transition_risk_angle_weight: float = 0.50  # [-] weight on end-tangent vs exit-target direction.
    passthrough_speed: float = 0.8  # [m/s] speed for partial arcs on revisited rooms.
    passthrough_min_turn_deg: float = 18.0  # [deg] minimum revisit-arc turn to avoid sharp direction kinks.
    start_room_priority: list[str] = field(
        default_factory=lambda: ["entryway", "living_room", "kitchen"]
    )
    polylabel_tolerance: float = 0.1  # [m] polylabel solver tolerance.
    polylabel_min_gain: float = 0.05  # [m] clearance gain required to prefer polylabel over centroid.
    min_speed_clamp: float = 0.05  # [m/s] lower bound used during time resampling.
    fov: float = 60.0  # [deg] camera horizontal field-of-view.

    def __post_init__(self) -> None:
        _validate_choice("look_at_mode", self.look_at_mode, SUPPORTED_LOOK_AT_MODES)
        _validate_choice(
            "disconnected_transition_mode",
            self.disconnected_transition_mode,
            SUPPORTED_DISCONNECTED_TRANSITION_MODES,
        )
        _validate_choice(
            "disconnected_component_policy",
            self.disconnected_component_policy,
            SUPPORTED_DISCONNECTED_COMPONENT_POLICIES,
        )
        _validate_choice(
            "neighbor_priority_mode",
            self.neighbor_priority_mode,
            SUPPORTED_NEIGHBOR_PRIORITY_MODES,
        )
        _validate_choice(
            "revisit_transition_mode",
            self.revisit_transition_mode,
            SUPPORTED_REVISIT_TRANSITION_MODES,
        )
        _validate_choice(
            "loop_closure_mode",
            self.loop_closure_mode,
            SUPPORTED_LOOP_CLOSURE_MODES,
        )
        if self.spin_points < 2:
            raise ValueError("`spin_points` must be >= 2.")
        if self.spin_look_radius <= 0:
            raise ValueError("`spin_look_radius` must be > 0.")
        if self.spin_orbit_scale < 0:
            raise ValueError("`spin_orbit_scale` must be >= 0.")
        if self.door_buffer <= 0:
            raise ValueError("`door_buffer` must be > 0.")
        if self.lookahead_inside_next_room <= 0:
            raise ValueError("`lookahead_inside_next_room` must be > 0.")
        if self.travel_speed <= 0:
            raise ValueError("`travel_speed` must be > 0.")
        if self.spin_segment_speed <= 0:
            raise ValueError("`spin_segment_speed` must be > 0.")
        if self.passthrough_speed <= 0:
            raise ValueError("`passthrough_speed` must be > 0.")
        if self.revisit_arc_angle_search_deg < 0 or self.revisit_arc_angle_search_deg >= 180:
            raise ValueError("`revisit_arc_angle_search_deg` must be in [0, 180).")
        if self.revisit_arc_search_steps < 1:
            raise ValueError("`revisit_arc_search_steps` must be >= 1.")
        if self.revisit_arc_length_weight < 0:
            raise ValueError("`revisit_arc_length_weight` must be >= 0.")
        if self.revisit_arc_outer_bias < 0:
            raise ValueError("`revisit_arc_outer_bias` must be >= 0.")
        if self.revisit_arc_max_span_deg <= 0 or self.revisit_arc_max_span_deg > 360:
            raise ValueError("`revisit_arc_max_span_deg` must be in (0, 360].")
        if self.revisit_arc_max_tangent_mismatch_deg <= 0 or self.revisit_arc_max_tangent_mismatch_deg >= 180:
            raise ValueError("`revisit_arc_max_tangent_mismatch_deg` must be in (0, 180).")
        if self.revisit_arc_reverse_pref_deg < 0 or self.revisit_arc_reverse_pref_deg >= 180:
            raise ValueError("`revisit_arc_reverse_pref_deg` must be in [0, 180).")
        if self.revisit_arc_reverse_long_arc_bonus < 0:
            raise ValueError("`revisit_arc_reverse_long_arc_bonus` must be >= 0.")
        if self.revisit_arc_transition_risk_distance_weight < 0:
            raise ValueError("`revisit_arc_transition_risk_distance_weight` must be >= 0.")
        if self.revisit_arc_transition_risk_angle_weight < 0:
            raise ValueError("`revisit_arc_transition_risk_angle_weight` must be >= 0.")
        if self.passthrough_min_turn_deg < 0 or self.passthrough_min_turn_deg >= 180:
            raise ValueError("`passthrough_min_turn_deg` must be in [0, 180).")
        if self.dense_samples_per_meter <= 0:
            raise ValueError("`dense_samples_per_meter` must be > 0.")
        if self.dense_samples_base < 2:
            raise ValueError("`dense_samples_base` must be >= 2.")
        if self.min_speed_clamp <= 0:
            raise ValueError("`min_speed_clamp` must be > 0.")
        if self.max_linear_speed <= 0:
            raise ValueError("`max_linear_speed` must be > 0.")
        if self.max_angular_speed_deg <= 0:
            raise ValueError("`max_angular_speed_deg` must be > 0.")
        if self.angular_smoothing_window_s < 0:
            raise ValueError("`angular_smoothing_window_s` must be >= 0.")
        if self.polylabel_tolerance <= 0:
            raise ValueError("`polylabel_tolerance` must be > 0.")
        if self.polylabel_min_gain < 0:
            raise ValueError("`polylabel_min_gain` must be >= 0.")
        if self.fov <= 0 or self.fov >= 180:
            raise ValueError("`fov` must be in (0, 180) degrees.")


@dataclass
class TrajectoryVisualizationConfig:
    """Visual style for trajectory plots."""

    path_color: str = "#D500F9"
    path_linewidth: float = 2.0
    start_color: str = "#00E676"
    end_color: str = "#F44336"
    marker_size: float = 120.0
    arrow_color: str = "#AA00FF"
    arrow_length: float = 0.5
    arrow_head_width: float = 0.15
    arrow_head_length: float = 0.2
    max_arrows: int = 50


@dataclass
class WalkthroughConfig:
    """Walkthrough/trajectory synthesis options."""

    fps: int = 30
    camera_height: float = 1.6
    output_dir: Path = Path("outputs/trajectory_generation")
    write_debug_summary: bool = True
    use_local_walkthrough: bool = False
    behavior: WalkthroughBehaviorConfig = field(default_factory=WalkthroughBehaviorConfig)
    plot_output: bool = False
    viz: TrajectoryVisualizationConfig = field(default_factory=TrajectoryVisualizationConfig)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("`fps` must be > 0.")


@dataclass
class TrajectoryGenerationConfig:
    """Top-level single config for end-to-end trajectory generation."""

    dataset: DatasetConfig
    floorplan: FloorplanPipelineConfig = field(default_factory=FloorplanPipelineConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    walkthrough: WalkthroughConfig = field(default_factory=WalkthroughConfig)
    workflow: str = "houselayout3d_matterport"

    def __post_init__(self) -> None:
        _validate_choice("workflow", self.workflow, SUPPORTED_WORKFLOWS)

    @classmethod
    def houselayout3d_matterport(
        cls,
        dataset_root: Path,
        scene: str,
        output_dir: Optional[Path] = None,
        house_segmentation_dir: Optional[Path] = None,
    ) -> "TrajectoryGenerationConfig":
        """
        Preset for the current production use case:
        HouseLayout3D geometry + Matterport3D room annotations (.house).
        """
        walkthrough = WalkthroughConfig(
            output_dir=output_dir or Path("outputs/trajectory_generation"),
        )
        dataset = DatasetConfig(
            dataset_root=dataset_root,
            scene=scene,
            house_segmentation_dir=house_segmentation_dir,
        )
        return cls(
            dataset=dataset,
            walkthrough=walkthrough,
            workflow="houselayout3d_matterport",
        )

    @classmethod
    def structural_json(
        cls,
        scene_input_json: Path,
        output_dir: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
        scene: str = "",
    ) -> "TrajectoryGenerationConfig":
        walkthrough = WalkthroughConfig(
            output_dir=output_dir or Path("outputs/trajectory_generation"),
        )
        dataset = DatasetConfig(
            dataset_root=dataset_root or Path("."),
            scene=scene,
            scene_input_json=scene_input_json,
        )
        return cls(
            dataset=dataset,
            walkthrough=walkthrough,
            workflow="structural_json",
        )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return _serialize_paths(raw)

    def to_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrajectoryGenerationConfig":
        if not isinstance(payload, dict):
            raise ValueError("Invalid config payload: expected a JSON object at top-level.")
        if "dataset" not in payload:
            raise ValueError("Missing required top-level field `dataset`.")
        dataset_raw = payload["dataset"]
        if not isinstance(dataset_raw, dict):
            raise ValueError("Invalid `dataset`: expected an object.")
        floorplan_raw = payload.get("floorplan", {})
        connectivity_raw = payload.get("connectivity", {})
        walkthrough_value = payload.get("walkthrough", {})
        if not isinstance(walkthrough_value, dict):
            raise ValueError("Invalid `walkthrough`: expected an object.")
        walkthrough_raw = dict(walkthrough_value)
        workflow = payload.get("workflow", "houselayout3d_matterport")

        allowed_dataset_keys = {
            "dataset_root",
            "scene",
            "house_segmentation_dir",
            "scene_input_json",
        }
        unknown_dataset_keys = set(dataset_raw.keys()) - allowed_dataset_keys
        if unknown_dataset_keys:
            unknown_str = ", ".join(sorted(unknown_dataset_keys))
            raise ValueError(f"Unknown keys in `dataset`: {unknown_str}.")

        dataset = DatasetConfig(
            dataset_root=Path(dataset_raw.get("dataset_root", ".")),
            scene=dataset_raw.get("scene", ""),
            house_segmentation_dir=_path_or_none(dataset_raw.get("house_segmentation_dir")),
            scene_input_json=_path_or_none(dataset_raw.get("scene_input_json")),
        )
        if not isinstance(floorplan_raw, dict):
            raise ValueError("Invalid `floorplan`: expected an object.")
        if not isinstance(connectivity_raw, dict):
            raise ValueError("Invalid `connectivity`: expected an object.")
        try:
            floorplan = FloorplanPipelineConfig(**floorplan_raw)
        except TypeError as exc:
            raise ValueError(f"Invalid `floorplan` config keys: {exc}") from exc
        try:
            connectivity = ConnectivityConfig(**connectivity_raw)
        except TypeError as exc:
            raise ValueError(f"Invalid `connectivity` config keys: {exc}") from exc
        behavior_raw = dict(walkthrough_raw.pop("behavior", {}))
        viz_raw = dict(walkthrough_raw.pop("viz", {}))
        # Backward compatibility with earlier config payloads.
        walkthrough_raw.pop("output_format", None)
        if behavior_raw:
            try:
                walkthrough_raw["behavior"] = WalkthroughBehaviorConfig(**behavior_raw)
            except TypeError as exc:
                raise ValueError(f"Invalid `walkthrough.behavior` config keys: {exc}") from exc
        if viz_raw:
            try:
                walkthrough_raw["viz"] = TrajectoryVisualizationConfig(**viz_raw)
            except TypeError as exc:
                raise ValueError(f"Invalid `walkthrough.viz` config keys: {exc}") from exc
        if "output_dir" in walkthrough_raw:
            walkthrough_raw["output_dir"] = Path(walkthrough_raw["output_dir"])
        try:
            walkthrough = WalkthroughConfig(**walkthrough_raw)
        except TypeError as exc:
            raise ValueError(f"Invalid `walkthrough` config keys: {exc}") from exc
        return cls(
            dataset=dataset,
            floorplan=floorplan,
            connectivity=connectivity,
            walkthrough=walkthrough,
            workflow=workflow,
        )

    @classmethod
    def from_json(cls, input_path: Path) -> "TrajectoryGenerationConfig":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)
