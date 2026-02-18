from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_generation.config import TrajectoryGenerationConfig
from trajectory_generation.preprocess import (
    build_hl3d_matterport_connectivity_geojson,
    convert_connectivity_geojson_file,
    convert_connectivity_geojson_payload,
    export_hl3d_matterport_debug_artifacts,
    preprocess_hl3d_matterport_to_structural_json,
)


class _FakeFloor:
    def __init__(self):
        self.level_index = 0
        self.mean_height = 0.0
        self.footprint = "fake_polygon"
        self.outer_shell = None
        self.area = 16.0


class _FakeRoom:
    def __init__(self, room_id: str, z: float):
        self.room_id = room_id
        self.centroid = (0.0, 0.0, z)


class _FakeFloorplanConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeGeometryModule:
    @staticmethod
    def load_stair_height_ranges(dataset_root: Path, scene: str):
        return []

    @staticmethod
    def load_surfaces(scene_dir: Path, stair_ranges, config):
        return ["surface"], ["wall"]

    @staticmethod
    def cluster_by_height(surfaces, tolerance: float):
        return [["surface"]]

    @staticmethod
    def build_floorplans(clusters, config):
        return [_FakeFloor()]


class _FakeRoomsModule:
    @staticmethod
    def parse_house_file(house_file: Path):
        return {"rooms": [_FakeRoom("R_1", 0.2)]}

    @staticmethod
    def assign_rooms_priority_overlay(rooms, floor, config):
        return [(rooms[0], "poly", 12.0, "claimed")]

    @staticmethod
    def match_rooms_with_intersections(rooms, floor, config):
        return [(rooms[0], "poly", 12.0)]

    @staticmethod
    def split_hallways_at_doors(rooms_data, doors, split_length: float):
        return rooms_data


class _FakeOpeningsModule:
    @staticmethod
    def load_openings(path: Path, opening_type: str):
        return []

    @staticmethod
    def match_openings_to_floors(all_openings, floorplans, config, rooms_by_floor):
        return {}

    @staticmethod
    def match_walls_to_floors(walls, floorplans):
        return {}

    @staticmethod
    def match_openings_to_walls(openings_by_floor_raw, walls_by_floor, tolerance: float):
        return {}


class _FakeConnectivityModule:
    @staticmethod
    def build_connectivity_graphs(**kwargs):
        return {0: object()}


class _FakeExportModule:
    @staticmethod
    def export_geojson(
        floorplans,
        output_path: Path,
        rooms_by_floor=None,
        openings_by_floor=None,
        connectivity_graphs=None,
    ) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]]],
                    },
                    "properties": {
                        "type": "floor_footprint",
                        "level_index": 0,
                        "mean_height": 0.0,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.5, 0.5], [2.0, 0.5], [2.0, 2.0], [0.5, 2.0], [0.5, 0.5]]],
                    },
                    "properties": {
                        "type": "room",
                        "room_id": "R_1",
                        "label_semantic": "living_room",
                        "level_index": 0,
                        "bbox_3d_min": [0.5, 0.5, -0.1],
                        "bbox_3d_max": [2.0, 2.0, 2.5],
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]],
                    },
                    "properties": {
                        "type": "room_connection",
                        "room1_id": "R_1",
                        "room2_id": "R_2",
                        "level_index": 0,
                    },
                },
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def floorplans_to_geojson(
        floorplans,
        rooms_by_floor=None,
        openings_by_floor=None,
        connectivity_graphs=None,
    ):
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                    },
                    "properties": {"type": "floor_footprint", "level_index": 0, "mean_height": 0.0},
                }
            ],
        }


def _fake_render_hl3d_debug_plots(
    *,
    scene,
    floorplans,
    rooms_by_floor,
    openings_by_floor,
    connectivity_graphs,
    output_dir: Path,
    center_map_by_floor=None,
    write_combined_plot=True,
    write_floor_plots=True,
    show_room_bboxes=False,
    color_room_intersections=True,
    show_connectivity=True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = output_dir / f"{scene}_floorplan.png"
    floor = output_dir / f"{scene}_floor_0.png"
    if write_combined_plot:
        combined.write_text("combined_plot")
    if write_floor_plots:
        floor.write_text("single_plot")
    return (combined if write_combined_plot else None), ((floor,) if write_floor_plots else ())


class PreprocessTest(unittest.TestCase):
    def test_convert_connectivity_geojson_payload(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
                    },
                    "properties": {"type": "floor_footprint", "level_index": 0, "mean_height": 0.0},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.2, 0.2], [1.0, 0.2], [1.0, 1.0], [0.2, 1.0], [0.2, 0.2]]],
                    },
                    "properties": {
                        "type": "room",
                        "room_id": "R_1",
                        "label_semantic": "entryway",
                        "level_index": 0,
                        "bbox_3d_min": [0.2, 0.2, 0.0],
                        "bbox_3d_max": [1.0, 1.0, 2.5],
                    },
                },
            ],
        }
        scene = convert_connectivity_geojson_payload(payload, scene_id="demo_scene")
        self.assertEqual(scene["schema_version"], "scene.schema.v1")
        self.assertEqual(scene["scene"], "demo_scene")
        self.assertEqual(len(scene["floors"]), 1)
        self.assertEqual(len(scene["rooms"]), 1)

    def test_convert_connectivity_geojson_file_infers_scene_id(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
                    },
                    "properties": {"type": "floor_footprint", "level_index": 0, "mean_height": 0.0},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.2, 0.2], [1.0, 0.2], [1.0, 1.0], [0.2, 1.0], [0.2, 0.2]]],
                    },
                    "properties": {
                        "type": "room",
                        "room_id": "R_1",
                        "label_semantic": "living_room",
                        "level_index": 0,
                        "bbox_3d_min": [0.2, 0.2, 0.0],
                        "bbox_3d_max": [1.0, 1.0, 2.5],
                    },
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            geojson_path = root / "scene_abc_connectivity.geojson"
            out_path = root / "scene_abc_structural_scene.json"
            geojson_path.write_text(json.dumps(payload, indent=2))

            convert_connectivity_geojson_file(geojson_path=geojson_path, output_path=out_path)
            converted = json.loads(out_path.read_text())
            self.assertEqual(converted["scene"], "scene_abc")

    def test_convert_connectivity_geojson_preserves_connection_normal_when_present(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
                    },
                    "properties": {"type": "floor_footprint", "level_index": 0, "mean_height": 0.0},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]],
                    },
                    "properties": {
                        "type": "room",
                        "room_id": "R_1",
                        "label_semantic": "entryway",
                        "level_index": 0,
                        "bbox_3d_min": [0.2, 0.2, 0.0],
                        "bbox_3d_max": [0.8, 0.8, 2.5],
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1.2, 0.2], [1.8, 0.2], [1.8, 0.8], [1.2, 0.8], [1.2, 0.2]]],
                    },
                    "properties": {
                        "type": "room",
                        "room_id": "R_2",
                        "label_semantic": "living_room",
                        "level_index": 0,
                        "bbox_3d_min": [1.2, 0.2, 0.0],
                        "bbox_3d_max": [1.8, 0.8, 2.5],
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[0.8, 0.5], [1.2, 0.5]],
                    },
                    "properties": {
                        "type": "room_connection",
                        "room1_id": "R_1",
                        "room2_id": "R_2",
                        "normal_xy": [1.0, 0.0],
                    },
                },
            ],
        }
        scene = convert_connectivity_geojson_payload(payload, scene_id="demo_scene")
        self.assertEqual(len(scene["connections"]), 1)
        self.assertEqual(scene["connections"][0]["normal_xy"], [1.0, 0.0])

    def test_convert_connectivity_geojson_filters_duplicate_and_self_connections(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
                    },
                    "properties": {"type": "floor_footprint", "level_index": 0, "mean_height": 0.0},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]],
                    },
                    "properties": {
                        "type": "room",
                        "room_id": "R_1",
                        "label_semantic": "entryway",
                        "level_index": 0,
                        "bbox_3d_min": [0.2, 0.2, 0.0],
                        "bbox_3d_max": [0.8, 0.8, 2.5],
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1.2, 0.2], [1.8, 0.2], [1.8, 0.8], [1.2, 0.8], [1.2, 0.2]]],
                    },
                    "properties": {
                        "type": "room",
                        "room_id": "R_2",
                        "label_semantic": "living_room",
                        "level_index": 0,
                        "bbox_3d_min": [1.2, 0.2, 0.0],
                        "bbox_3d_max": [1.8, 0.8, 2.5],
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[0.8, 0.5], [1.2, 0.5]]},
                    "properties": {"type": "room_connection", "room1_id": "R_1", "room2_id": "R_2"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[0.8, 0.5], [1.2, 0.5]]},
                    "properties": {"type": "room_connection", "room1_id": "R_2", "room2_id": "R_1"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[0.3, 0.3], [0.6, 0.6]]},
                    "properties": {"type": "room_connection", "room1_id": "R_1", "room2_id": "R_1"},
                },
            ],
        }
        scene = convert_connectivity_geojson_payload(payload, scene_id="demo_scene")
        self.assertEqual(len(scene["connections"]), 1)
        self.assertEqual(scene["connections"][0]["room1_id"], "R_1")
        self.assertEqual(scene["connections"][0]["room2_id"], "R_2")

    def test_build_hl3d_geojson_and_convert_to_structural(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene = "scene_abc"
            (root / "tools" / "floorplan").mkdir(parents=True)
            dataset_root = root / "dataset"
            (dataset_root / "structures" / "layouts_split_by_entity" / scene).mkdir(parents=True)
            (dataset_root / "house_segmentations" / scene).mkdir(parents=True)
            (dataset_root / "house_segmentations" / scene / f"{scene}.house").write_text("R")

            config = TrajectoryGenerationConfig.houselayout3d_matterport(
                dataset_root=dataset_root,
                scene=scene,
            )

            geojson_out = root / "out" / f"{scene}_connectivity.geojson"
            structural_out = root / "out" / f"{scene}_structural_scene.json"

            fake_ports = type(
                "Ports",
                (),
                {
                    "connectivity": _FakeConnectivityModule,
                    "geometry": _FakeGeometryModule,
                    "openings": _FakeOpeningsModule,
                    "rooms": _FakeRoomsModule,
                    "floorplan_config_factory": _FakeFloorplanConfig,
                    "walkthrough_factory": object,
                },
            )()

            with patch(
                "trajectory_generation.adapters.houselayout3d_matterport._load_floorplan_modules",
                return_value=fake_ports,
            ), patch(
                "trajectory_generation.hl3d_preprocess._load_floorplan_export_module",
                return_value=_FakeExportModule,
            ), patch(
                "trajectory_generation.hl3d_preprocess._compute_center_map_by_floor",
                return_value={0: {"R_1": [1.25, 1.25, 1.6]}},
            ):
                written_geojson = build_hl3d_matterport_connectivity_geojson(
                    config=config,
                    geojson_output_path=geojson_out,
                    project_root=root,
                )
                self.assertEqual(written_geojson, geojson_out)
                self.assertTrue(geojson_out.exists())

                out_geojson, out_structural = preprocess_hl3d_matterport_to_structural_json(
                    config=config,
                    structural_output_path=structural_out,
                    geojson_output_path=geojson_out,
                    project_root=root,
                )
                self.assertEqual(out_geojson, geojson_out)
                self.assertEqual(out_structural, structural_out)
                self.assertTrue(structural_out.exists())

            converted = json.loads(structural_out.read_text())
            self.assertEqual(converted["schema_version"], "scene.schema.v1")
            self.assertEqual(converted["scene"], scene)
            self.assertEqual(len(converted["floors"]), 1)
            self.assertEqual(len(converted["rooms"]), 1)

            geojson_payload = json.loads(geojson_out.read_text())
            room_features = [
                f
                for f in geojson_payload.get("features", [])
                if f.get("properties", {}).get("type") == "room"
            ]
            self.assertEqual(len(room_features), 1)
            self.assertEqual(room_features[0]["properties"]["trajectory_center_xy"], [1.25, 1.25])
            self.assertEqual(room_features[0]["properties"]["trajectory_center_3d"], [1.25, 1.25, 1.6])

            center_features = [
                f
                for f in geojson_payload.get("features", [])
                if f.get("properties", {}).get("type") == "trajectory_room_center"
            ]
            self.assertEqual(len(center_features), 1)
            self.assertEqual(center_features[0]["properties"]["room_id"], "R_1")

    def test_export_hl3d_debug_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scene = "scene_abc"
            (root / "tools" / "floorplan").mkdir(parents=True)
            dataset_root = root / "dataset"
            (dataset_root / "structures" / "layouts_split_by_entity" / scene).mkdir(parents=True)
            (dataset_root / "house_segmentations" / scene).mkdir(parents=True)
            (dataset_root / "house_segmentations" / scene / f"{scene}.house").write_text("R")

            config = TrajectoryGenerationConfig.houselayout3d_matterport(
                dataset_root=dataset_root,
                scene=scene,
            )

            fake_ports = type(
                "Ports",
                (),
                {
                    "connectivity": _FakeConnectivityModule,
                    "geometry": _FakeGeometryModule,
                    "openings": _FakeOpeningsModule,
                    "rooms": _FakeRoomsModule,
                    "floorplan_config_factory": _FakeFloorplanConfig,
                    "walkthrough_factory": object,
                },
            )()

            with patch(
                "trajectory_generation.adapters.houselayout3d_matterport._load_floorplan_modules",
                return_value=fake_ports,
            ), patch(
                "trajectory_generation.hl3d_preprocess._load_floorplan_export_module",
                return_value=_FakeExportModule,
            ), patch(
                "trajectory_generation.hl3d_preprocess.render_hl3d_debug_plots",
                side_effect=_fake_render_hl3d_debug_plots,
            ):
                artifacts = export_hl3d_matterport_debug_artifacts(
                    config=config,
                    output_dir=root / "debug_out",
                    project_root=root,
                )

            self.assertTrue(artifacts.connectivity_geojson.exists())
            self.assertEqual(len(artifacts.floor_geojson_files), 1)
            self.assertTrue(artifacts.floor_geojson_files[0].exists())
            self.assertIsNotNone(artifacts.combined_plot_file)
            assert artifacts.combined_plot_file is not None
            self.assertTrue(artifacts.combined_plot_file.exists())
            self.assertEqual(len(artifacts.floor_plot_files), 1)
            self.assertTrue(artifacts.floor_plot_files[0].exists())
            self.assertIsNotNone(artifacts.diagnostics_json_file)
            assert artifacts.diagnostics_json_file is not None
            self.assertTrue(artifacts.diagnostics_json_file.exists())


if __name__ == "__main__":
    unittest.main()
