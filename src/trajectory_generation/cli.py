"""Command-line entrypoints for trajectory_generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from importlib import metadata

from .config import SUPPORTED_WORKFLOWS, TrajectoryGenerationConfig
from .pipeline import run
from .preprocess import (
    build_hl3d_matterport_connectivity_geojson,
    export_hl3d_matterport_debug_artifacts,
    preprocess_hl3d_matterport_to_structural_json,
)


def _package_version() -> str:
    try:
        return metadata.version("multiroom-scene-trajectory-generation")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _configure_logging(log_level: str) -> None:
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(
            f"Invalid --log-level: {log_level!r}. "
            "Allowed values: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    logging.basicConfig(level=level, format="%(message)s", force=True)


def _build_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrstg",
        description=(
            "Generate walkthrough trajectories using either HouseLayout3D+Matterport "
            "or canonical structural JSON workflow. "
            "For dataset preprocessing use: `mrstg preprocess ...`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mrstg {_package_version()}",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=False,
        help="Dataset root for HouseLayout3D+Matterport workflow.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=False,
        help="Scene identifier for HouseLayout3D+Matterport workflow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/trajectory_generation"),
        help="Output directory for trajectories and summary artifacts.",
    )
    parser.add_argument(
        "--house-segmentation-dir",
        type=Path,
        default=None,
        help="Optional override for house segmentation directory.",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Workflow to run: houselayout3d_matterport | structural_json.",
    )
    parser.add_argument(
        "--scene-input-json",
        type=Path,
        default=None,
        help="Canonical structural scene JSON input (used by structural_json workflow).",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON file serialized from TrajectoryGenerationConfig.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional explicit repo root containing tools/floorplan.",
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate structural scene schema and exit (no trajectory generation).",
    )
    walkthrough_group = parser.add_mutually_exclusive_group()
    walkthrough_group.add_argument(
        "--use-local-walkthrough",
        action="store_true",
        help=(
            "Use refactored LocalWalkthroughGenerator (behavior config applies, including "
            "spin_orbit_scale)."
        ),
    )
    walkthrough_group.add_argument(
        "--use-legacy-walkthrough",
        action="store_true",
        help=(
            "Use legacy floorplan Walkthrough3DGS (some behavior knobs are ignored; "
            "orbit radius is hardcoded in legacy code)."
        ),
    )
    return parser


def _build_preprocess_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrstg preprocess",
        description=(
            "Preprocess HouseLayout3D+Matterport data into connectivity GeoJSON "
            "and canonical structural scene JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mrstg {_package_version()}",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=False,
        help="Dataset root containing structures and house segmentations.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=False,
        help="Scene identifier to preprocess.",
    )
    parser.add_argument(
        "--house-segmentation-dir",
        type=Path,
        default=None,
        help="Optional override for house segmentation directory.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON file serialized from TrajectoryGenerationConfig.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional explicit repo root containing tools/floorplan.",
    )
    parser.add_argument(
        "--geojson-output",
        type=Path,
        default=None,
        help="Output path for connectivity GeoJSON artifact.",
    )
    parser.add_argument(
        "--structural-output",
        type=Path,
        default=None,
        help="Output path for canonical structural scene JSON artifact.",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="Optional override for scene id written into structural JSON.",
    )
    parser.add_argument(
        "--geojson-only",
        action="store_true",
        help="Only export connectivity GeoJSON and skip structural JSON conversion.",
    )
    parser.add_argument(
        "--emit-debug-artifacts",
        action="store_true",
        help=(
            "Also emit debug artifacts: per-floor geojson, floor plots, and "
            "diagnostics JSON."
        ),
    )
    parser.add_argument(
        "--debug-output-dir",
        type=Path,
        default=None,
        help="Directory for optional debug artifacts.",
    )
    parser.add_argument(
        "--no-floor-geojson",
        action="store_true",
        help="Disable per-floor GeoJSON debug exports.",
    )
    parser.add_argument(
        "--no-combined-plot",
        action="store_true",
        help="Disable combined floor debug plot export.",
    )
    parser.add_argument(
        "--no-floor-plots",
        action="store_true",
        help="Disable per-floor debug plot export.",
    )
    parser.add_argument(
        "--use-raw-room-centers",
        action="store_true",
        help="Use raw room centers in debug visuals instead of trajectory centers.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable diagnostics artifact export.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Backward-compatible generate parser used by tests and legacy calls."""
    return _build_generate_parser().parse_args()


def _parse_cli_args(argv: list[str] | None = None) -> tuple[str, argparse.Namespace]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    command = "generate"
    if tokens and tokens[0] in {"generate", "preprocess"}:
        command = tokens[0]
        tokens = tokens[1:]

    if command == "preprocess":
        return command, _build_preprocess_parser().parse_args(tokens)
    return command, _build_generate_parser().parse_args(tokens)


def build_config(args: argparse.Namespace) -> TrajectoryGenerationConfig:
    def _apply_walkthrough_override(config: TrajectoryGenerationConfig) -> None:
        if getattr(args, "use_local_walkthrough", False):
            config.walkthrough.use_local_walkthrough = True
        elif getattr(args, "use_legacy_walkthrough", False):
            config.walkthrough.use_local_walkthrough = False

    if args.config_json is not None:
        config = TrajectoryGenerationConfig.from_json(args.config_json)
        if args.workflow is not None:
            config.workflow = args.workflow
        if args.scene_input_json is not None:
            config.dataset.scene_input_json = args.scene_input_json
        _apply_walkthrough_override(config)
        config.__post_init__()
        config.walkthrough.behavior.__post_init__()
        return config

    workflow = args.workflow
    if workflow is None:
        if args.scene_input_json is not None:
            workflow = "structural_json"
        elif args.dataset_root is not None or args.scene is not None:
            workflow = "houselayout3d_matterport"
        else:
            allowed = ", ".join(SUPPORTED_WORKFLOWS)
            raise ValueError(
                "Could not infer workflow from arguments. "
                f"Provide --workflow ({allowed}) or pass --scene-input-json / --dataset-root + --scene."
            )
    if workflow not in SUPPORTED_WORKFLOWS:
        allowed = ", ".join(SUPPORTED_WORKFLOWS)
        raise ValueError(f"Invalid --workflow: {workflow!r}. Allowed values: {allowed}.")

    if workflow == "structural_json":
        if args.scene_input_json is None:
            raise ValueError("Provide --scene-input-json when --workflow structural_json is used.")
        config = TrajectoryGenerationConfig.structural_json(
            scene_input_json=args.scene_input_json,
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
            scene=args.scene or "",
        )
        _apply_walkthrough_override(config)
        return config

    if args.dataset_root is None or args.scene is None:
        raise ValueError("Provide either --config-json or both --dataset-root and --scene.")

    config = TrajectoryGenerationConfig.houselayout3d_matterport(
        dataset_root=args.dataset_root,
        scene=args.scene,
        output_dir=args.output_dir,
        house_segmentation_dir=args.house_segmentation_dir,
    )
    _apply_walkthrough_override(config)
    return config


def _build_preprocess_config(args: argparse.Namespace) -> TrajectoryGenerationConfig:
    if args.config_json is not None:
        cfg = TrajectoryGenerationConfig.from_json(args.config_json)
        if cfg.workflow != "houselayout3d_matterport":
            raise ValueError(
                "Preprocess supports workflow='houselayout3d_matterport' only. "
                f"Found {cfg.workflow!r}."
            )
        return cfg
    if args.dataset_root is None or args.scene is None:
        raise ValueError("Provide either --config-json or both --dataset-root and --scene.")
    return TrajectoryGenerationConfig.houselayout3d_matterport(
        dataset_root=args.dataset_root,
        scene=args.scene,
        house_segmentation_dir=args.house_segmentation_dir,
    )


def _run_generate(args: argparse.Namespace) -> None:
    if args.validate_schema:
        from .schema import validate_structural_scene_file

        scene_input_json = args.scene_input_json
        if scene_input_json is None and args.config_json is not None:
            cfg = TrajectoryGenerationConfig.from_json(args.config_json)
            scene_input_json = cfg.dataset.scene_input_json
        if scene_input_json is None:
            raise ValueError("Provide --scene-input-json (or --config-json with dataset.scene_input_json) to validate.")

        root = (args.project_root or Path.cwd()).resolve()
        scene_path = scene_input_json if scene_input_json.is_absolute() else (root / scene_input_json).resolve()
        report = validate_structural_scene_file(scene_path)

        print(f"Schema valid: {scene_path}")
        print(f"  schema_version: {report['schema_version']}")
        print(f"  scene: {report['scene']}")
        print(
            f"  floors={report['num_floors']} rooms={report['num_rooms']} "
            f"connections={report['num_connections']}"
        )
        if report["warnings"]:
            print("Warnings:")
            for warning in report["warnings"]:
                print(f"  - {warning}")
        return

    config = build_config(args)
    artifacts = run(config=config, project_root=args.project_root)

    print(f"Scene: {artifacts.scene}")
    print(f"Output: {artifacts.output_dir}")
    print(f"Floors with trajectories: {len(artifacts.floor_trajectories)}")
    for item in artifacts.floor_trajectories:
        print(
            f"  Floor {item.floor_index}: {item.num_frames} frames, "
            f"{item.num_rooms} rooms, {item.num_connections} connections -> {item.output_file}"
        )
    if artifacts.warnings:
        print("Warnings:")
        for warning in artifacts.warnings:
            print(f"  - {warning}")


def _run_preprocess(args: argparse.Namespace) -> None:
    config = _build_preprocess_config(args)
    default_out_dir = Path("outputs/preprocess")
    geojson_output = args.geojson_output or (default_out_dir / f"{config.dataset.scene}_connectivity.geojson")
    structural_output = args.structural_output or (
        default_out_dir / f"{config.dataset.scene}_structural_scene.json"
    )
    scene_id = args.scene_id or config.dataset.scene or None
    write_diagnostics = not args.no_diagnostics

    if args.geojson_only:
        if args.emit_debug_artifacts:
            debug = export_hl3d_matterport_debug_artifacts(
                config=config,
                output_dir=args.debug_output_dir or geojson_output.parent,
                project_root=args.project_root,
                write_floor_geojson=not args.no_floor_geojson,
                write_combined_plot=not args.no_combined_plot,
                write_floor_plots=not args.no_floor_plots,
                show_room_bboxes=False,
                color_room_intersections=True,
                show_connectivity=True,
                use_trajectory_centers_for_debug=not args.use_raw_room_centers,
                write_diagnostics=write_diagnostics,
            )
            print(f"Wrote connectivity GeoJSON: {debug.connectivity_geojson}")
            if debug.floor_geojson_files:
                print(f"Wrote {len(debug.floor_geojson_files)} floor GeoJSON files")
            if debug.combined_plot_file:
                print(f"Wrote combined plot: {debug.combined_plot_file}")
            if debug.floor_plot_files:
                print(f"Wrote {len(debug.floor_plot_files)} floor plots")
            if debug.diagnostics_json_file:
                print(f"Wrote diagnostics: {debug.diagnostics_json_file}")
            if debug.room_matching_stats_csv:
                print(f"Wrote room stats: {debug.room_matching_stats_csv}")
            return

        geojson_path = build_hl3d_matterport_connectivity_geojson(
            config=config,
            geojson_output_path=geojson_output,
            project_root=args.project_root,
        )
        print(f"Wrote connectivity GeoJSON: {geojson_path}")
        return

    geojson_path, structural_path = preprocess_hl3d_matterport_to_structural_json(
        config=config,
        structural_output_path=structural_output,
        geojson_output_path=geojson_output,
        scene_id=scene_id,
        project_root=args.project_root,
        emit_debug_artifacts=args.emit_debug_artifacts,
        debug_output_dir=args.debug_output_dir,
        write_floor_geojson=not args.no_floor_geojson,
        write_combined_plot=not args.no_combined_plot,
        write_floor_plots=not args.no_floor_plots,
        show_room_bboxes=False,
        color_room_intersections=True,
        show_connectivity=True,
        use_trajectory_centers_for_debug=not args.use_raw_room_centers,
        write_diagnostics=write_diagnostics,
    )
    print(f"Wrote connectivity GeoJSON: {geojson_path}")
    print(f"Wrote structural scene JSON: {structural_path}")


def main() -> None:
    command, args = _parse_cli_args()
    try:
        _configure_logging(getattr(args, "log_level", "INFO"))
        if command == "preprocess":
            _run_preprocess(args)
            return
        _run_generate(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001
        logger = logging.getLogger(__name__)
        if logging.getLogger().handlers:
            logger.error("Error: %s", exc)
            if logger.isEnabledFor(logging.DEBUG):
                logger.exception("Detailed traceback")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
